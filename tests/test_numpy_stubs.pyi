from typing import Literal, Final, Any
import numpy as np

zero_final: Final = 0
one_final: Final = 1
two_final: Final = 2
three_final: Final = 3
four_final: Final = 4

zero: Literal[0] = 0
one: Literal[1] = 1
two: Literal[2] = 2
three: Literal[3] = 3
four: Literal[4] = 4


# Test ndarray analyze_type hook
a: np.ndarray[Literal[2], Literal[2], np.float64]
b: np.ndarray[Literal[2], Literal[1], np.float64]
c: np.ndarray[Any, Any]
d: np.ndarray[Any, str]  # error: Type argument "builtins.str" of "ndarray" must be a subtype of "Union[numpy.float64, numpy.float32, numpy.float16, numpy.float8, numpy.int64, numpy.int32, numpy.int16, numpy.int8, numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64, numpy.bool_, numpy.complex64, numpy.complex128, numpy.complex192, numpy.complex256, numpy.void, numpy.str_]"

reveal_type(a)  # note: Revealed type is "numpy.ndarray[Literal[2], Literal[2], numpy.float64]"
reveal_type(b)  # note: Revealed type is "numpy.ndarray[Literal[2], Literal[1], numpy.float64]"
reveal_type(c)  # note: Revealed type is "numpy.ndarray[Any, Any]"
reveal_type(d)  # note: Revealed type is "numpy.ndarray[Any, builtins.str]"


# Test ndarray.__matmul__ hook
reveal_type(a @ b)  # note: Revealed type is "numpy.ndarray[Literal[2], Literal[1], numpy.float64]"
a @ np.zeros((3, 1))  # error: Unsupported operand type for @ ("numpy.ndarray[Literal[2], Literal[2], numpy.float64]" and "numpy.ndarray[Literal[3], Literal[1], numpy.float64]" because dimensions Literal[2] and Literal[3] are incompatible)


# Test ndarray.__add__ hook
reveal_type(a + b)  # note: Revealed type is "numpy.ndarray[Literal[2], Literal[2], numpy.float64]"
a + np.zeros((3, 3))  # error: Unsupported operand type for + ("numpy.ndarray[Literal[2], Literal[2], numpy.float64]" and "numpy.ndarray[Literal[3], Literal[3], numpy.float64]" because dimensions Literal[2] and Literal[3] are incompatible)


# Test type annotation of ndarray.__pos__
reveal_type(+a)  # note: Revealed type is "numpy.ndarray[Literal[2], Literal[2], numpy.float64]"

# Test zeros hook
reveal_type(np.zeros((1, 2)))  # note: Revealed type is "numpy.ndarray[Literal[1], Literal[2], numpy.float64]"
reveal_type(np.zeros((one_final, two_final)))  # note: Revealed type is "numpy.ndarray[Literal[1], Literal[2], numpy.float64]"
reveal_type(np.zeros((one, two)))  # note: Revealed type is "numpy.ndarray[Literal[1], Literal[2], numpy.float64]"


# Test flatten hook
reveal_type(a.flatten())  # note: Revealed type is "numpy.ndarray[Literal[4], numpy.float64]"


# Test reshape hook
reveal_type(a.reshape((4, 1)))  # note: Revealed type is "numpy.ndarray[Literal[4], Literal[1], numpy.float64]"
reveal_type(a.reshape((four_final, one_final)))  # note: Revealed type is "numpy.ndarray[Literal[4], Literal[1], numpy.float64]"
reveal_type(a.reshape((four, one)))  # note: Revealed type is "numpy.ndarray[Literal[4], Literal[1], numpy.float64]"


# test that return values of np.array can be annotated to arbitrary shape
_: np.ndarray[Literal[2], Literal[2], np.float64] = np.array([])


# Test ndarray.__getitem__ with basic indexing
reveal_type(a[:])  # note: Revealed type is "numpy.ndarray[Literal[2], Literal[2], numpy.float64]"
reveal_type(a[0, :])  # note: Revealed type is "numpy.ndarray[Literal[2], numpy.float64]"
reveal_type(a[zero_final, :])  # note: Revealed type is "numpy.ndarray[Literal[2], numpy.float64]"
reveal_type(a[zero, :])  # note: Revealed type is "numpy.ndarray[Literal[2], numpy.float64]"
reveal_type(a[0, ...]) # note: Revealed type is "numpy.ndarray[Literal[2], numpy.float64]"
reveal_type(a[zero_final, ...]) # note: Revealed type is "numpy.ndarray[Literal[2], numpy.float64]"
reveal_type(a[zero, ...]) # note: Revealed type is "numpy.ndarray[Literal[2], numpy.float64]"
reveal_type(a[zero, np.newaxis]) # note: Revealed type is "numpy.ndarray[Literal[1], Literal[2], numpy.float64]"
reveal_type(a[0:1])  # note: Revealed type is "numpy.ndarray[Literal[1], Literal[2], numpy.float64]"
reveal_type(a[0:5])  # note: Revealed type is "numpy.ndarray[Literal[2], Literal[2], numpy.float64]"
a[:, :, :]  # error: Too many indices: array is 2-dimensional, but 3 were indexed
a[3]  # error: Index 3 is out of bounds for axis with size 2
a[three_final] # error: Index 3 is out of bounds for axis with size 2
a[three] # error: Index 3 is out of bounds for axis with size 2


# Test ndarray.__getitem__ with advanced indexing
i = np.zeros((3, 3), dtype=np.int64)
i2 = np.zeros((3, 4), dtype=np.int64)
reveal_type(a[np.newaxis, i]) # note: Revealed type is "numpy.ndarray[Literal[1], Literal[3], Literal[3], Literal[2], numpy.float64]"
a[np.zeros((3, 3))]  # error: Arrays used as indices must be of integer (or boolean) type
reveal_type(a[i])  # note: Revealed type is "numpy.ndarray[Literal[3], Literal[3], Literal[2], numpy.float64]"
reveal_type(a[i, i])  # note: Revealed type is "numpy.ndarray[Literal[3], Literal[3], numpy.float64]"
a[i, i2]  # error: broadcast error
reveal_type(a[(0, 1, 0),])  # note: Revealed type is "numpy.ndarray[Literal[3], Literal[2], numpy.float64]"
reveal_type(a[(zero_final, one_final, zero_final),])  # note: Revealed type is "numpy.ndarray[Literal[3], Literal[2], numpy.float64]"
reveal_type(a[(zero, one, zero),])  # note: Revealed type is "numpy.ndarray[Literal[3], Literal[2], numpy.float64]"
reveal_type(a[(0, 1, 0), (0, 1, 0)])  # note: Revealed type is "numpy.ndarray[Literal[3], numpy.float64]"
reveal_type(a[(0, 1, 0), 1])  # note: Revealed type is "numpy.ndarray[Literal[3], numpy.float64]"
reveal_type(a[(0, 1, 0), one_final])  # note: Revealed type is "numpy.ndarray[Literal[3], numpy.float64]"
reveal_type(a[(0, 1, 0), one])  # note: Revealed type is "numpy.ndarray[Literal[3], numpy.float64]"
reveal_type(a[(0, 1, 0), 0:1])  # note: Revealed type is "numpy.ndarray[Literal[3], Literal[1], numpy.float64]"
reveal_type(a[(0, 1, 0), 0:5])  # note: Revealed type is "numpy.ndarray[Literal[3], Literal[2], numpy.float64]"
reveal_type(a[(0, 1, 0), np.newaxis, 1]) # note: Revealed type is "numpy.ndarray[Literal[3], Literal[1], numpy.float64]"
a[(0, 2),]  # error: Index 2 is out of bounds for axis with size 2
a[(0, 1), 2]  # error: Index 2 is out of bounds for axis with size 2


# Test dtype conversion
a = np.zeros((2, 2), dtype=np.float32)
b = np.zeros((2, 2), dtype=np.int64)
reveal_type(a + b)  # note: Revealed type is "numpy.ndarray[Literal[2], Literal[2], numpy.float64]"
a + np.array([], dtype=np.str_)  # error: Incompatible scalar types ("numpy.float32*" and "numpy.str_")


