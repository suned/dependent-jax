from typing import Literal, Final, Any
import jax.numpy as jnp

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


# Test DeviceArray analyze_type hook
a: jnp.DeviceArray[Literal[2], Literal[2], jnp.float32]
b: jnp.DeviceArray[Literal[2], Literal[1], jnp.float32]
c: jnp.DeviceArray[Any, Any]
d: jnp.DeviceArray[Any, str]  # error: Type argument "builtins.str" of "DeviceArray" must be a subtype of "Union[jax.numpy.float32, jax.numpy.int32, Union[numpy.float64, numpy.float32, numpy.float16, numpy.float8, numpy.int64, numpy.int32, numpy.int16, numpy.int8, numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64, numpy.bool_, numpy.complex64, numpy.complex128, numpy.complex192, numpy.complex256, numpy.void, numpy.str_]]"

reveal_type(a)  # note: Revealed type is "jax.numpy.DeviceArray[Literal[2], Literal[2], jax.numpy.float32]"
reveal_type(b)  # note: Revealed type is "jax.numpy.DeviceArray[Literal[2], Literal[1], jax.numpy.float32]"
reveal_type(c)  # note: Revealed type is "jax.numpy.DeviceArray[Any, Any]"
reveal_type(d)  # note: Revealed type is "jax.numpy.DeviceArray[Any, builtins.str]"


# Test ndarray.__matmul__ hook
reveal_type(a @ b)  # note: Revealed type is "jax.numpy.DeviceArray[Literal[2], Literal[1], jax.numpy.float32]"
a @ jnp.zeros((3, 1))  # error: Unsupported operand type for @ ("jax.numpy.DeviceArray[Literal[2], Literal[2], jax.numpy.float32]" and "jax.numpy.DeviceArray[Literal[3], Literal[1], jax.numpy.float32]" because dimensions Literal[2] and Literal[3] are incompatible)


# Test ndarray.__add__ hook
reveal_type(a + b)  # note: Revealed type is "jax.numpy.DeviceArray[Literal[2], Literal[2], jax.numpy.float32]"
a + jnp.zeros((3, 3))  # error: Unsupported operand type for + ("jax.numpy.DeviceArray[Literal[2], Literal[2], jax.numpy.float32]" and "jax.numpy.DeviceArray[Literal[3], Literal[3], jax.numpy.float32]" because dimensions Literal[2] and Literal[3] are incompatible)

# Test type annotation of ndarray.__pos__
reveal_type(+a)  # note: Revealed type is "jax.numpy.DeviceArray[Literal[2], Literal[2], jax.numpy.float32]"

# Test zeros hook
reveal_type(jnp.zeros((1, 2)))  # note: Revealed type is "jax.numpy.DeviceArray[Literal[1], Literal[2], jax.numpy.float32]"
reveal_type(jnp.zeros((one_final, two_final)))  # note: Revealed type is "jax.numpy.DeviceArray[Literal[1], Literal[2], jax.numpy.float32]"
reveal_type(jnp.zeros((one, two)))  # note: Revealed type is "jax.numpy.DeviceArray[Literal[1], Literal[2], jax.numpy.float32]"

# Test flatten hook
reveal_type(a.flatten())  # note: Revealed type is "jax.numpy.DeviceArray[Literal[4], jax.numpy.float32]"

# Test reshape hook
reveal_type(a.reshape((4, 1)))  # note: Revealed type is "jax.numpy.DeviceArray[Literal[4], Literal[1], jax.numpy.float32]"
reveal_type(a.reshape((four_final, one_final)))  # note: Revealed type is "jax.numpy.DeviceArray[Literal[4], Literal[1], jax.numpy.float32]"
reveal_type(a.reshape((four, one)))  # note: Revealed type is "jax.numpy.DeviceArray[Literal[4], Literal[1], jax.numpy.float32]"

# test that return values of np.array can be annotated to arbitrary shape
_: jnp.DeviceArray[Literal[2], Literal[2], jnp.float32] = jnp.array([])

# Test ndarray.__getitem__ with basic indexing
reveal_type(a[:])  # note: Revealed type is "jax.numpy.DeviceArray[Literal[2], Literal[2], jax.numpy.float32]"
reveal_type(a[0, :])  # note: Revealed type is "jax.numpy.DeviceArray[Literal[2], jax.numpy.float32]"
reveal_type(a[zero_final, :])  # note: Revealed type is "jax.numpy.DeviceArray[Literal[2], jax.numpy.float32]"
reveal_type(a[zero, :])  # note: Revealed type is "jax.numpy.DeviceArray[Literal[2], jax.numpy.float32]"
reveal_type(a[0, ...]) # note: Revealed type is "jax.numpy.DeviceArray[Literal[2], jax.numpy.float32]"
reveal_type(a[zero_final, ...]) # note: Revealed type is "jax.numpy.DeviceArray[Literal[2], jax.numpy.float32]"
reveal_type(a[zero, ...]) # note: Revealed type is "jax.numpy.DeviceArray[Literal[2], jax.numpy.float32]"
reveal_type(a[zero, jnp.newaxis]) # note: Revealed type is "jax.numpy.DeviceArray[Literal[1], Literal[2], jax.numpy.float32]"
reveal_type(a[0:1])  # note: Revealed type is "jax.numpy.DeviceArray[Literal[1], Literal[2], jax.numpy.float32]"
reveal_type(a[0:5])  # note: Revealed type is "jax.numpy.DeviceArray[Literal[2], Literal[2], jax.numpy.float32]"
a[:, :, :]  # error: Too many indices: array is 2-dimensional, but 3 were indexed
a[3]  # error: Index 3 is out of bounds for axis with size 2
a[three_final] # error: Index 3 is out of bounds for axis with size 2
a[three] # error: Index 3 is out of bounds for axis with size 2
