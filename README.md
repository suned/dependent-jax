# dependent-jax

Proof-of-concept implementation of dependent types for statically verifiable n-dimensional array operations with `jax` and `numpy` 
by way of a [stubs only package](https://www.python.org/dev/peps/pep-0561/#stub-only-packages) 
and [mypy plugin](https://mypy.readthedocs.io/en/stable/extending_mypy.html#extending-mypy-using-plugins).

Note that this is very much a work in progress, and at present only a handful of operations are supported as a basic
proof-of-concept.

## What Is This?
In most type systems there is a bright line between _types_ and _values_. Values are
the stuff you assign to variables, e.g:
- `42`
- `"the string""`

Types on the other hand are _sets of values_ that you talk about with your type-checker
through type annotations and inference. Examples of types are:
- `int` (to which the value `42` belongs)
- `str` (to which the value `"the string"` belongs)

[Dependent types](https://en.wikipedia.org/wiki/Dependent_type) blurs the line between values and types by allowing you to talk about values with your type checker. In Python
this is done using the "[Literal](https://docs.python.org/3/library/typing.html#typing.Literal)" type:
```python
from typing import Literal


FourtyTwo = Literal[42]  # Type alias for a type (i.e a set) that only contains the value 42
x: FourtyTwo = 42
y: FourtyTwo = 0  # Type error because 0 does not belong to the type 42
```

`dependent-jax` is a proof-of-concept of how to use `Literal` to annotate `jax.numpy.DeviceArray` and `numpy.ndarray` types
with shape information, thereby providing _static verification of tensor operations_. In other words, `dependent-jax` helps `mypy`
to catch many errors related to tensor shape mismatch that would otherwise turn up as
runtime errors.


`dependent-jax` currently demonstrates feasibility of the following types of annotations/inferences:

- Annotating array types with shape information
- Inferring shapes of arrays returned from functions that accept a `shape` parameter
- Checking array shape compatibility and inferring shapes of arrays returned from binary broadcasting operations
- Inferring shapes of arrays returned from unary operations
- Checking array shape compatibility and inferring shapes of arrays returned from matrix multiplication
- Inferring shapes of arrays returned from un-parameterized shape manipulation (e.g `array.flatten()`)
- Inferring shapes of arrays returned from parameterized shape manipulation (e.g `array.reshape((...))`)
- Checking argument compatibility and inferring shapes of arrays returned from index operations

It should be possible to extend each of the approaches described above to many similar functions/methods
in the `jax`/`numpy` api with little effort.
## Install
From github, e.g using `pip`:
```commandline
pip install git+https://github.com/suned/dependent-jax
```
Add the following to your [mypy config file](https://mypy.readthedocs.io/en/stable/config_file.html) to enable the mypy plugin (this package doesn't make any sense without it):
```
[mypy]
plugins = dependent_jax
```
## Usage
When instantiating arrays from io or from Python values (e.g `list` instances), there
is no way to infer the array shape, and it should be supplied via annotation. `jax.numpy.DeviceArray` and `numpy.ndarray` accepts at
minimum two type paramaters. All type parameters to `jax.numpy.DeviceArray` and `numpy.ndarray` except the last must be `Literal` integer types. The last type parameter is always the scalar type of the array:
```python
from typing import Literal

import jax.numpy as jnp
import numpy as np


a: jnp.DeviceArray[Literal[3], Literal[2], jnp.float32] = jnp.array([[1, 2], [3, 4], [5, 6]])
b: np.ndarray[Literal[3], Literal[2], np.float64] = np.array([[1, 2], [3, 4], [5, 6]])

reveal_type(a)  # note: Revealed type is "jax.numpy.DeviceArray[Literal[3], Literal[2], jax.numpy.float32]"
reveal_type(b)  # note: Revealed type is "numpy.ndarray[Literal[3], Literal[2], numpy.float64]"
```

`typing.Any` in the place of the shape variable(s) always indicates an array of unknown shape:

```python
import jax.numpy as jnp


reveal_type(jnp.array([]))  # note: Revealed type is "jax.numpy.DeviceArray[Any, jax.numpy.float32]"
```

When instantiating arrays with functions that take a shape parameter,
the resulting shape can be inferred provided that the shape arguments are
literal types:
```python
import jax.numpy as jnp


a = jnp.zeros((2, 2))
reveal_type(a)  # Revealed type is: jax.numpy.DeviceArray[Literal[2], Literal[2], jax.numpy.float32)
```

With `mypy`, values can be interpreted as literal types when:
- The value is supplied directly as an argument (e.g `jnp.zeros((2, 2))`)
- A variable is annotated with `Literal` (e.g `two: Literal[2] = 2`)
- A variable is annotated with `Final` (e.g `two: Final = 2`)

This means that the return type of `jnp.zeros` can be inferred in the following examples:

```python
from typing import Literal, Final

import jax.numpy as jnp


a: Literal[2] = 2
b: Final = 2

jnp.zeros((2, 2))
jnp.zeros((a, a))
jnp.zeros((b, b))
```

but not in:

```python
import jax.numpy as jnp


a = 2
jnp.zeros((a, a))
```

The shape of arrays resulting from operations on arrays with known shape can be inferred, and errors
resulting from incompatible dimensions will be reported by `mypy`:

```python
import jax.numpy as jnp


a: jnp.DeviceArray[Literal[3], Literal[2], jnp.float32]
b: jnp.DeviceArray[Literal[2], Literal[1], jnp.float32]

reveal_type(a @ b)  # Revealed type is: jax.numpy.DeviceArray[Literal[3], Literal[1], np.float32]
```
The shape of arrays resulting from index operations can currently only be inferred when
the types of arguments are either:

- Literal integers
- In-line slice expressions
- `Tuple` types with literal integer element types in the case of [advanced indexing](https://numpy.org/doc/stable/reference/arrays.indexing.html#advanced-indexing)

For example in:
```python
from typing import Final

import jax.numpy as jnp


zero: Final = 0
a = jnp.zeros((3, 2))

reveal_type(a[0])    # Revealed type is: jax.numpy.DeviceArray[Literal[2], jax.numpy.float32] 
reveal_type(a[zero:2])  # Revealed type is: jax.numpy.DeviceArray[Literal[2], Literal[2], jax.numpy.float32]
```

But not in:
```python
s = slice(0, 1)
# Inference of index operations with slices only works with in-line slice expressions
reveal_type(a[s])    # Revealed type is: jax.numpy.DeviceArray[Any, jax.numpy.float32]
```
