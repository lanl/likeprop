'''This module defines generic Type Variables and Aliases for type hints'''

from typing import Union, Callable, Optional, Any
import numpy as np
from numpy.typing import ArrayLike

# Aliases for types

# a generic float type for numbers
FloatT = Union[float, np.float64]
# a generic slice type for extracting sub sequences
SliceT = Union[int, list[int], slice]
# a generic Array type of floats
ArrayFloatT = np.ndarray[Any, np.dtype[np.float64]]
# for now an array like object is what numpy thinks of an array like object
# this is list of lists, scalar, numpy ndarrays and objects that implement an array interface
ArrayLikeT = ArrayLike
# Fixed List
FixedListT = Union[tuple, list, np.ndarray]
# a generic bool like object.
BoolT = Union[bool, np.bool_]

# Aliases for functions
# f(time, vector, params) -> real number
TimeStateFunctional = Callable[[FloatT, ArrayFloatT, Optional[ArrayFloatT]],
                               FloatT]

# f(time, vector, params) -> vector of same dimension
TimeStateVector = Callable[[FloatT, ArrayFloatT, Optional[ArrayFloatT]],
                           ArrayFloatT]
