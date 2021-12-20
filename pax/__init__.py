from typing import Optional, Callable, List, Union, Type as Type_, Any, Dict
from typing_extensions import TypeGuard
import itertools
import functools
import re

import numpy as np
import jax.numpy as jnp

from mypy.types import Type, Instance, LiteralType, AnyType, TupleType, TypeVarType, CallableType, TypeVarDef, TypeVarId, NoneType, EllipsisType
from mypy.mro import calculate_mro
from mypy.nodes import TypeInfo, SymbolTable, ClassDef, Block, Argument, Var, ARG_POS, FuncDef, PassStmt, SymbolTableNode, MDEF, INVARIANT, SliceExpr, Node, MypyFile
from mypy.semanal import set_callable_name
from mypy.typeops import get_proper_type, fill_typevars
from mypy.util import get_unique_redefinition_name
from mypy.type_visitor import TypeQuery
from mypy.subtypes import is_subtype
from mypy.plugin import Plugin, FunctionContext, AnalyzeTypeContext, TypeAnalyzerPluginInterface, MethodContext, SemanticAnalyzerPluginInterface, CheckerPluginInterface
from mypy.lookup import lookup_fully_qualified


class IncompatibleBroadcastDimensionsError(Exception):
    def __init__(self, first_dimension, second_dimension):
        self.first_dimension = first_dimension
        self.second_dimension = second_dimension


class IncompatibleBroadcastDimensionsForArraysError(Exception):
    def __init__(self, first_array, second_array, first_dimension, second_dimension):
        self.first_array = first_array
        self.second_array = second_array
        self.first_dimension = first_dimension
        self.second_dimension = second_dimension


def get_dtype(fullname: str, first: Instance, second: Instance, modules) -> Optional[Instance]:
    # this is an awful solution. Should be implemented purely in terms
    # of mypy types, in order to avoid importing numpy and jax
    module_name, *_ = fullname.split('.')
    if module_name == 'numpy':
        module = np
    else:
        module = jnp
        module_name = 'jax.numpy'
    first_dtype = getattr(module, first.type.name)(0)
    second_dtype = getattr(module, second.type.name)(0)
    try:
        result = first_dtype + second_dtype
        dtype_name = f'{module_name}.{result.dtype.name}'
        dtype_info = lookup_fully_qualified(dtype_name, modules).node
        return Instance(dtype_info, [])
    except:
        return None


def _type_var_def(
    name: str,
    module: str,
    upper_bound: Type,
    values=(),
    meta_level: int = 0,
    variance: int = INVARIANT
) -> TypeVarDef:
    id_ = TypeVarId.new(meta_level)
    fullname = f'{module}.{name}'
    return TypeVarDef(name, fullname, id_, list(values), upper_bound, variance)


def analyze_array_hook(ctx: AnalyzeTypeContext, fullname: str) -> Type:
    analyzer: TypeAnalyzerPluginInterface = ctx.api
    if not ctx.type.args:
        args = [AnyType(0), AnyType(0)]
    else:
        args = [analyzer.analyze_type(arg) for arg in ctx.type.args]
        args = [get_proper_type(arg) for arg in args]
    *dims, type_ = args
    if not all(isinstance(dim, LiteralType) or isinstance(dim, AnyType) for dim in dims):
        analyzer.fail(f'All dimensions of {fullname} must be Literal types', ctx.type)
    elif any(type(dim.value) is not int for dim in dims if isinstance(dim, LiteralType)):
        analyzer.fail(
            f'All arguments of literals in dimensions of {fullname} must be int',
            ctx.type)
    ndarray = array_instance(fullname, args, ctx.type.line, ctx.type.column, ctx.api.api.modules, ctx.api.named_type('builtins.object'))
    return ndarray


def array_instance(fullname: str, args: List[Type], line: int, column: int, modules: Dict[str, MypyFile], object_type: Instance) -> Instance:
    array_info = lookup_fully_qualified(fullname, modules).node
    array = Instance(array_info, args, line=line, column=column)
    add_type_vars_to_array(array, object_type)
    return array


def ndarray_hook(ctx: AnalyzeTypeContext) -> Type:
    return analyze_array_hook(ctx, 'numpy.ndarray')


def device_array_hook(ctx: AnalyzeTypeContext) -> Type:
    return analyze_array_hook(ctx, 'jax.numpy.DeviceArray')


def add_type_vars_to_array(ndarray: Instance, upper_bound: Type) -> None:
    type_var_names = [f'S{i + 1}' for i in range(len(ndarray.args) - 1)]
    type_var_defs = [_type_var_def(name, 'numpy', upper_bound)
                     for name in type_var_names]
    defn = ClassDef(ndarray.type.defn.name,
                    ndarray.type.defn.defs,
                    [*type_var_defs, ndarray.type.defn.type_vars[-1]],
                    ndarray.type.defn.base_type_exprs,
                    ndarray.type.defn.metaclass)
    defn.fullname = ndarray.type.defn.fullname
    type = TypeInfo(ndarray.type.names, defn, ndarray.type.module_name)
    calculate_mro(type)
    ndarray.type = type


def is_array_type(t: Type) -> TypeGuard[Instance]:
    return isinstance(t, Instance) and t.type.fullname in ('numpy.ndarray', 'jax.numpy.DeviceArray')


def is_list_of_literals(ts: List[Type]) -> TypeGuard[List[LiteralType]]:
    return all(isinstance(t, LiteralType) for t in ts)


def check_broadcast_operation(first_dims: List[LiteralType], second_dims: List[LiteralType], api: CheckerPluginInterface) -> List[Type]:
    ret_dims = []
    for self_dim, other_dim in itertools.zip_longest(reversed(first_dims),
                                                     reversed(second_dims),
                                                     fillvalue=LiteralType(1, api.named_type('builtins.int'))):
        is_1 = self_dim.value == 1 or other_dim.value == 1
        are_equal = self_dim == other_dim
        if not is_1 and not are_equal:
            raise IncompatibleBroadcastDimensionsError(self_dim, other_dim)
        if are_equal:
            ret_dims.append(self_dim)
        else:
            # one dimension was 1
            ret_dims.append(max(self_dim, other_dim, key=lambda dim: dim.value))
    ret_dim_literals_ordered = reversed(ret_dims)
    return list(ret_dim_literals_ordered)


def check_ndarray_operation(fullname: str, first: Instance, second: Instance, operation: str, ctx: MethodContext) -> Type:
    *self_dims, self_type = first.args
    *other_dims, other_type = second.args
    ret_type = get_dtype(fullname, self_type, other_type, ctx.api.modules)
    if ret_type is None:
        ctx.api.fail(f'Incompatible scalar types ("{self_type}" and "{other_type}")', ctx.context)
        return ctx.default_return_type
    default_return_type = ctx.default_return_type.copy_modified(
        args=[AnyType(0), ret_type])
    if any(isinstance(dim, AnyType) for dim in self_dims):
        return first
    if any(isinstance(dim, AnyType) for dim in other_dims):
        return second
    if not is_list_of_literals(self_dims) or not is_list_of_literals(other_dims):
        return default_return_type
    try:
        ret_dims = check_broadcast_operation(self_dims, other_dims, ctx.api)
    except IncompatibleBroadcastDimensionsError as e:
        msg = (f'Unsupported operand type for {operation} '
               f'("{first}" and "{second}" because '
               f'dimensions {e.first_dimension} and {e.second_dimension} are incompatible)')
        ctx.api.fail(msg, ctx.context)
        ret_dims = [AnyType(0)]

    array = array_instance(fullname, [*ret_dims, ret_type], -1, -1, ctx.api.modules, ctx.api.named_type('builtins.object'))
    return array


def is_supported_builtin_operand_type(t: Type) -> TypeGuard[Instance]:
    return isinstance(t, Instance) and t.type.fullname in ('builtins.int', 'builtins.float')


def check_operation_with_builtin_type(first: Instance, second: Instance, operation: str, ctx: MethodContext) -> Type:
    # TODO map result dtype based on operand
    return first


def add_hook(fullname: str, ctx: MethodContext) -> Type:
    self = ctx.type
    other = ctx.arg_types[0][0]
    if is_array_type(other):
        return check_ndarray_operation(fullname, self, other, '+', ctx)
    elif is_supported_builtin_operand_type(other):
        return check_operation_with_builtin_type(self, other, '+', ctx)
    else:
        ctx.api.msg.unsupported_operand_types('+', self, other, ctx.context)
        return ctx.default_return_type


def is_literal_instance(t: Type) -> bool:
    return isinstance(t, Instance) and t.last_known_value is not None


def is_list_of_literal_instances(ts: List[Type]) -> bool:
    return all(is_literal_instance(t) for t in ts)


def zeros_hook(ctx: FunctionContext) -> Type:
    shape = ctx.arg_types[0][0]
    if not isinstance(shape, TupleType):
        return ctx.default_return_type
    if not all(is_literal_instance(item) or isinstance(item, LiteralType) for item in shape.items):
        return ctx.default_return_type
    args = [item.last_known_value if not isinstance(item, LiteralType) else item for item in shape.items]  # type: ignore
    *_, type_ = ctx.default_return_type.args
    return ctx.default_return_type.copy_modified(args=[*args, type_])


def matmul_hook(fullname: str, ctx: MethodContext) -> Type:
    #import ipdb; ipdb.set_trace()
    *self_dims, self_type = ctx.type.args
    *other_dims, other_type = ctx.arg_types[0][0].args
    ret_type = get_dtype(fullname, self_type, other_type, ctx.api.modules)
    int_type = ctx.api.named_type('builtins.int')
    if not is_list_of_literals(self_dims) or not is_list_of_literals(other_dims):
        return AnyType(0)
    prepended = False
    appended = False
    if len(self_dims) == 1:
        self_dims = [LiteralType(1, int_type), *self_dims]
        prepended = True
    if len(other_dims) == 1:
        other_dims = [*other_dims, LiteralType(1, int_type)]
        appended = True
    if self_dims[-1] != other_dims[-2]:
        ctx.api.fail(f'Unsupported operand type for @ ("{ctx.type}" and "{ctx.arg_types[0][0]}" because dimensions {self_dims[-1]} and {other_dims[-2]} are incompatible)', ctx.context)
        return ctx.default_return_type.copy_modified(args=[AnyType(0), ret_type])
    try:
        ret_dims = check_broadcast_operation(self_dims[:-1], other_dims[:-2], ctx.api)
    except IncompatibleBroadcastDimensionsError as e:
        msg = (f'Unsupported operand type for @ '
               f'("{ctx.type}" and "{ctx.arg_types[0][0]}" because '
               f'dimensions {e.first_dimension} and {e.second_dimension} are incompatible)')
        ctx.api.fail(msg, ctx.context)
        ret_dims = [AnyType(0)]
    else:
        ret_dims = [*ret_dims, other_dims[-1]]
    if prepended:
        ret_dims = ret_dims[1:]
    if appended:
        ret_dims = ret_dims[:-1]
    if not ret_dims:
        return ret_type
    array = array_instance(fullname, [*ret_dims, ret_type], -1, -1, ctx.api.modules, ctx.api.named_type('builtins.object'))
    return array


def array_hook(ctx: FunctionContext) -> Type:
    # todo infer return dtype with recursive sequences
    return ctx.default_return_type


def get_total_size(dims: List[LiteralType]) -> int:
    return functools.reduce(lambda result, dim: result * dim.value, dims, 1)


def flatten_hook(ctx: MethodContext) -> Type:
    *dims, type_ = ctx.type.args
    if not is_list_of_literals(dims):
        return ctx.default_return_type
    new_shape = get_total_size(dims)
    new_shape_literal = LiteralType(new_shape, ctx.api.named_type('builtins.int'))
    return ctx.default_return_type.copy_modified(args=[new_shape_literal, type_])


def reshape_hook(ctx: MethodContext) -> Type:
    *dims, type_ = ctx.type.args
    shape = ctx.arg_types[0][0]
    if not is_list_of_literals(dims):
        return ctx.default_return_type
    total_size = get_total_size(dims)
    if isinstance(shape, TupleType):
        if not all(is_literal_instance(item) or isinstance(item, LiteralType) for item in shape.items):
            return ctx.default_return_type
        args = [item.last_known_value if is_literal_instance(item) else item for item in shape.items]
        requested_size = get_total_size(args)
    elif is_literal_instance(shape):
        arg = shape.last_known_value
        args = [arg]
        requested_size = arg.value
    else:
        return ctx.default_return_type
    if requested_size != total_size:
        ctx.api.fail(f'Can\'t reshape "{ctx.type}" into shape "{args}"', ctx.context)
        return ctx.default_return_type
    return ctx.default_return_type.copy_modified(args=[*args, type_])


def is_scalar_type(t: Type, *names: str) -> bool:
    return isinstance(t, Instance) and t.type.fullname in names


def check_broadcast_operations(arrays: List[Instance], api) -> List[LiteralType]:
    first, *rest = arrays
    *first_dims, _ = first.args
    result_dims = first_dims
    for array in rest:
        *second_dims, _ = array.args
        result_dims = check_broadcast_operation(first_dims, second_dims, api)
    return result_dims


def ndarray(dims: List[LiteralType], scalar: Instance, api) -> Instance:
    args = [*dims, scalar]
    ndarray = api.named_generic_type('numpy.ndarray', args)
    object = api.named_type('builtins.object')
    add_type_vars_to_array(ndarray, object)
    return ndarray


def advanced_indexing(indices: List[Type], dims: List[Type], index_exprs: List[Node], type_: Type, ctx: MethodContext) -> Type:
    array_indices = []
    advanced_indices = []
    result_dims = []
    for dim_index, (dim, expr, index) in enumerate(zip(dims, index_exprs, indices)):
        if is_array_type(index):
            advanced_indices.append(dim_index)
            *index_dims, index_type = index.args
            integer_type = ctx.api.named_type('numpy.integer')
            bool_type = ctx.api.named_type('numpy.bool_')
            if not (is_subtype(index_type, integer_type) or is_subtype(index_type, bool_type)):
                ctx.api.fail('Arrays used as indices must be of integer (or boolean) type', ctx.context)
                return ctx.default_return_type
            if not is_list_of_literals(index_dims):
                return ctx.default_return_type
            array_indices.append(index)
        elif isinstance(index, TupleType):
            advanced_indices.append(dim_index)
            if not all(is_literal_instance(i) or isinstance(i, LiteralType) for i in index.items):
                return ctx.default_return_type
            index_indices = [
                i.last_known_value if is_literal_instance(i) else i for i in
                index.items]
            for i in index_indices:
                if i.value >= dim.value:
                    ctx.api.fail(f'Index {i.value} is out of bounds for axis with size {dim.value}', ctx.context)
                    return ctx.default_return_type
            array = ndarray(
                [LiteralType(len(index_indices), ctx.api.named_type('builtins.int'))],
                type_,
                ctx.api
            )
            array_indices.append(array)
        elif is_literal_instance(index) or isinstance(index, LiteralType):
            advanced_indices.append(dim_index)
            index_value = index.value if isinstance(index, LiteralType) else index.last_known_value.value
            if index_value >= dim.value:
                ctx.api.fail(f'Index {index_value} is out of bounds for axis with size {dim.value}', ctx.context)
                return ctx.default_return_type
            array = ndarray(
                [LiteralType(1, ctx.api.named_type('builtins.int'))],
                type_,
                ctx.api
            )
            array_indices.append(array)
        elif is_slice(index):
            return_dim = get_slice_return_dim(expr, dim, ctx)
            if return_dim is None:
                return ctx.default_return_type
            result_dims.append(return_dim)
        elif isinstance(index, NoneType):
            result_dims.append(LiteralType(1, ctx.api.named_type('builtins.int')))
        else:
            return ctx.default_return_type
    try:
        broadcasted_dims = check_broadcast_operations(array_indices, ctx.api)
        advanced_indices_are_next_to_eachother = advanced_indices == list(range(min(advanced_indices), max(advanced_indices) + 1))
        if advanced_indices_are_next_to_eachother:
            first_advanced_index = min(advanced_indices)
            broadcasted_indices = range(first_advanced_index, first_advanced_index + len(broadcasted_dims) + 1)
            for i, broadcasted_dim in zip(broadcasted_indices, broadcasted_dims):
                result_dims.insert(i, broadcasted_dim)
        else:
            result_dims = broadcasted_dims + result_dims
        return ctx.default_return_type.copy_modified(args=[*result_dims, type_])
    except IncompatibleBroadcastDimensionsError:
        # todo better message
        ctx.api.fail('broadcast error', ctx.context)
        return ctx.default_return_type


def is_ellipsis(t: Type) -> bool:
    return isinstance(t, Instance) and t.type.fullname == 'builtins.ellipsis'


def is_slice(t: Type) -> bool:
    return isinstance(t, Instance) and t.type.fullname == 'builtins.slice'


def get_slice_return_dim(index_expr: Node, dim: LiteralType, ctx: MethodContext) -> Optional[LiteralType]:
    if not isinstance(index_expr, SliceExpr):
        return None
    if not (index_expr.begin_index or index_expr.end_index):
        return dim
    else:
        if index_expr.end_index:
            end_index_type = index_expr.end_index.accept(ctx.api.expr_checker)
            if not end_index_type.last_known_value:
                return None
            end_index = end_index_type.last_known_value.value
        else:
            end_index = dim.value
        if index_expr.begin_index:
            begin_index_type = index_expr.begin_index.accept(ctx.api.expr_checker)
            if not begin_index_type.last_known_value:
                return None
            begin_index = begin_index_type.last_known_value.value
        else:
            begin_index = 0
        return LiteralType(
            min(len(range(begin_index, end_index)), dim.value),
            ctx.api.named_type('builtins.int')
        )


def get_item_hook(ctx: MethodContext) -> Type:
    arg = ctx.arg_types[0][0]
    self = ctx.type
    *dims, type_ = self.args
    if not is_list_of_literals(dims):
        return ctx.default_return_type

    if isinstance(arg, (Instance, LiteralType)):
        indices = [arg]
        index_exprs = [ctx.args[0][0]]
    elif isinstance(arg, TupleType):
        indices = arg.items
        index_exprs = ctx.args[0][0].items
    else:
        return ctx.default_return_type

    real_indices = [i for i in indices if not isinstance(i, NoneType) and not is_ellipsis(i)]

    if len(real_indices) > len(dims):
        ctx.api.fail(f'Too many indices: array is {len(dims)}-dimensional, but {len(real_indices)} were indexed', ctx.context)
        return ctx.default_return_type

    ellipses = [i for i in indices if is_ellipsis(i)]

    if len(ellipses) > 1:
        ctx.api.fail('An index can only have a single ellipsis (\'...\')', ctx.context)
        return ctx.default_return_type
    if len(ellipses) == 1:
        diff = len(dims) - len(real_indices)
        ellipsis_index = next(i for i, index in enumerate(indices) if is_ellipsis(index))

        indices_before, indices_after = indices[:ellipsis_index], indices[ellipsis_index + 1:]
        indices = indices_before + [ctx.api.named_type('builtins.slice')] * diff + indices_after

        index_exprs_before, index_exprs_after = index_exprs[:ellipsis_index], index_exprs[ellipsis_index + 1:]
        index_exprs = index_exprs_before + [SliceExpr(None, None, None)] * diff + index_exprs_after

    if len(real_indices) < len(dims) and len(ellipses) == 0:
        indices.extend(ctx.api.named_type('builtins.slice') for _ in range(len(dims) - len(real_indices)))
        index_exprs.extend(SliceExpr(None, None, None) for _ in range(len(dims) - len(real_indices)))

    # insert placeholder dimensions for indices that are None
    # in order that we can iterate over dimensions and indices
    # together below
    for index, i in enumerate(indices):
        if isinstance(i, NoneType):
            dims.insert(index, None)

    if any(isinstance(i, TupleType) or is_array_type(i) for i in indices):
        return advanced_indexing(indices, dims, index_exprs, type_, ctx)

    return basic_indexing(ctx, dims, index_exprs, indices, type_)


def basic_indexing(ctx, dims, index_exprs, indices, type_):
    return_dims = []
    if not len(indices) == len(index_exprs) == len(dims): import ipdb; ipdb.set_trace()
    for dim, index_expr, index in zip(dims, index_exprs, indices):
        if isinstance(index, NoneType):
            return_dims.append(LiteralType(1, ctx.api.named_type('builtins.int')))
        elif isinstance(index, LiteralType):
            index_value = index.value
            if index_value >= dim.value:
                ctx.api.fail(
                    f'Index {index_value} is out of bounds for axis with size {dim.value}',
                    ctx.context)
                return ctx.default_return_type
        elif is_slice(index):
            return_dim = get_slice_return_dim(index_expr, dim, ctx)
            if return_dim is None:
                return ctx.default_return_type
            return_dims.append(return_dim)
        elif is_literal_instance(index):
            index_value = index.last_known_value.value
            if index_value >= dim.value:
                ctx.api.fail(
                    f'Index {index_value} is out of bounds for axis with size {dim.value}',
                    ctx.context)
                return ctx.default_return_type
        else:
            return ctx.default_return_type
    if not return_dims:
        return type_
    ndarray = ctx.default_return_type.copy_modified(args=[*return_dims, type_])
    upper_bound = ctx.api.named_type('builtins.object')
    add_type_vars_to_array(ndarray, upper_bound)
    return ndarray


class Pax(Plugin):
    def get_type_analyze_hook(self, fullname: str
                              ) -> Optional[Callable[[AnalyzeTypeContext], Type]]:
        if fullname == 'numpy.ndarray':
            return ndarray_hook
        if fullname == 'jax.numpy.DeviceArray':
            return device_array_hook
        return None

    def get_method_hook(self, fullname: str
                        ) -> Optional[Callable[[MethodContext], Type]]:
        if fullname == 'numpy.ndarray.__add__':
            return functools.partial(add_hook, 'numpy.ndarray')
        if fullname == 'jax.numpy.DeviceArray.__add__':
            return functools.partial(add_hook, 'jax.numpy.DeviceArray')
        if fullname == 'numpy.ndarray.__matmul__':
            return functools.partial(matmul_hook, 'numpy.ndarray')
        if fullname == 'jax.numpy.DeviceArray.__matmul__':
            return functools.partial(matmul_hook, 'jax.numpy.DeviceArray')
        if fullname in ('numpy.ndarray.__getitem__', 'jax.numpy.DeviceArray.__getitem__'):
            return get_item_hook
        if fullname in ('numpy.ndarray.flatten', 'jax.numpy.DeviceArray.flatten'):
            return flatten_hook
        if fullname in ('numpy.ndarray.reshape', 'jax.numpy.DeviceArray.reshape'):
            return reshape_hook
        return None

    def get_function_hook(self, fullname: str
                          ) -> Optional[Callable[[FunctionContext], Type]]:
        if fullname in ('numpy.zeros', 'jax.numpy.zeros'):
            return zeros_hook
        if fullname == 'numpy.array':
            return array_hook
        return None


def plugin(_: Any) -> Type_[Pax]:
    return Pax
