import builtins
from typing import Any, Tuple, TypeVar, Type, overload, Union, Generic, final, Iterable, Sequence, Protocol, Literal


class generic:
    pass


class bool_(generic):
    pass


class object_(generic):
    pass


class number(generic):
    pass


class integer(number):
    pass


class inexact(number):
    ...


class flexible(generic):
    pass


# ---------------------------------- signed integers -----------------------------------

class signedinteger(integer):
    pass


class int64(signedinteger):
    ...


class int32(signedinteger):
    ...


class int16(signedinteger):
    ...


class int8(signedinteger):
    ...


# --------------------------------- unsigned integers ----------------------------------

class unsignedinteger(integer):
    ...


class uint64(unsignedinteger):
    ...


class uint32(unsignedinteger):
    ...


class uint16(unsignedinteger):
    ...


class uint8(unsignedinteger):
    ...


# ------------------------------------- floating ---------------------------------------


class floating(inexact):
    ...


class float96(floating):
    ...


class float128(floating):
    ...


class float64(floating):
    ...


class float32(floating):
    ...


class float16(floating):
    ...


class float8(floating):
    ...


# --------------------------------- complex floating -----------------------------------

class complexfloating(inexact):
    ...


class complex64(complexfloating):
    ...


class complex128(complexfloating):
    ...


class complex192(complexfloating):
    ...


class complex256(complexfloating):
    ...


# ------------------------------------- flexible ---------------------------------------

class void(flexible):
    pass


class character(flexible):
    pass


class str_(flexible):
    pass


DType = Union[float64,
              float32,
              float16,
              float8,

              int64,
              int32,
              int16,
              int8,

              uint8,
              uint16,
              uint32,
              uint64,

              bool_,

              complex64,
              complex128,
              complex192,
              complex256,

              void,

              str_]

S = TypeVar('S')
D = TypeVar('D', bound=DType, covariant=True)
D2 = TypeVar('D2')
Self = TypeVar('Self')


class SupportsIndex(Protocol):
    def __index__(self) -> Union[int, slice]:
        pass


newaxis = None


ArrayIndex = Union[int, bool, slice, SupportsIndex, Sequence, None, builtins.ellipsis]
ArrayLike = Union[Sequence, int, float, bool, str, complex, bytes, DType, SupportsArray]
A = TypeVar('A', bound=ArrayLike)


class ndarray(Generic[S, D], Sequence):

    def __add__(self, other: A) -> ndarray[Any, Any]:
        pass

    def __matmul__(self, other: A) -> ndarray[Any, Any]:
        pass

    def __pos__(self: Self) -> Self:
        pass


    @overload
    def __getitem__(self, item: int) -> ndarray:
        pass

    @overload
    def __getitem__(self, item: slice) -> ndarray:
        pass

    @overload
    def __getitem__(self, item: SupportsIndex) -> ndarray:
        pass

    @overload
    def __getitem__(self, item: Sequence) -> ndarray:
        pass

    @overload
    def __getitem__(self, item: builtins.ellipsis) -> ndarray:
        pass

    @overload
    def __getitem__(self, item: None) -> ndarray:
        pass

    @overload
    def __getitem__(self, item: ndarray) -> ndarray:
        pass

    def flatten(self, order: Literal['C', 'F', 'A', 'K'] = 'C') -> ndarray[Any, D]:
        pass

    def reshape(self, shape: Union[int, Tuple[int, ...]], order: Literal['C', 'F', 'A'] = 'C') -> ndarray[Any, D]:
        pass

    def __len__(self) -> int:
        pass


@overload
def zeros(shape: Tuple[int, ...]) -> ndarray[Any, float64]:
    pass


@overload
def zeros(shape: Tuple[int, ...], dtype: Type[D]) -> ndarray[Any, D]:
    pass


class SupportsArray(Protocol):
    def __array__(self) -> ndarray:
        ...


@overload
def array(object: A, dtype: Type[D]) -> ndarray[Any, D]:
    pass


@overload
def array(object: A) -> ndarray[Any, float64]:
    pass
