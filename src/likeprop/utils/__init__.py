'''initializes the utils subpackage for defining helper functions, custom typing aliases, 
collection objects and miscelleneous items.
'''
from .collections import (SetInEuclideanSpace, Rect, Ball)
from .helper_functions import (coordinate_names_from_array, compose)
from .typing import (FloatT, SliceT, ArrayFloatT, BoolT, ArrayLikeT,
                     TimeStateFunctional, TimeStateVector)

__all__ = [
    'SetInEuclideanSpace',
    'Rect',
    'Ball',
    'coordinate_names_from_array',
    'compose',
    'FloatT',
    'SliceT',
    'ArrayFloatT',
    'BoolT',
    'ArrayLikeT',
    'TimeStateFunctional',
    'TimeStateVector',
]
