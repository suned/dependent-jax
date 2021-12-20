from typing import Tuple, Any, Type, TypeVar, overload, Union, Sequence, Generic

import numpy as np


class float32:
    pass


class int32:
    pass


DType = Union[float32, int32, np.DType]

newaxis = None


ArrayLike = Union[Sequence, int, float, bool, str, complex, bytes, DType, np.SupportsArray]

D = TypeVar('D', bound=DType, covariant=True)
S = TypeVar('S')

Self = TypeVar('Self')

A = TypeVar('A', bound=ArrayLike)


class DeviceArray(Generic[S, D]):
    def __pos__(self: Self) -> Self:
        pass

    def __add__(self, other: A) -> DeviceArray:
        pass

    def __matmul__(self, other: A) -> DeviceArray:
        pass

    def flatten(self, order: Literal['C', 'F', 'A', 'K'] = 'C') -> DeviceArray[Any, D]:
        pass

    def reshape(self, shape: Union[int, Tuple[int, ...]], order: Literal['C', 'F', 'A'] = 'C') -> DeviceArray[Any, D]:
        pass

    @overload
    def __getitem__(self, item: int) -> DeviceArray:
        pass

    @overload
    def __getitem__(self, item: slice) -> DeviceArray:
        pass

    @overload
    def __getitem__(self, item: np.SupportsIndex) -> DeviceArray:
        pass

    @overload
    def __getitem__(self, item: Sequence) -> DeviceArray:
        pass

    @overload
    def __getitem__(self, item: builtins.ellipsis) -> DeviceArray:
        pass

    @overload
    def __getitem__(self, item: None) -> DeviceArray:
        pass

    @overload
    def __getitem__(self, item: np.ndarray) -> DeviceArray:
        pass

    @overload
    def __getitem__(self, item: DeviceArray) -> DeviceArray:
        pass


@overload
def zeros(shape: Tuple[int, ...], dtype: Type[D]) -> DeviceArray[Any, D]:
    pass

@overload
def zeros(shape: Tuple[int, ...]) -> DeviceArray[Any, float32]:
    pass


@overload
def array(object: A, dtype: Type[D]) -> DeviceArray[Any, D]:
    pass

@overload
def array(object: A) -> DeviceArray[Any, float32]:
    pass
