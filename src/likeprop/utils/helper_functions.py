'''This module provides a lot of helper functions'''
import functools
import itertools
from typing import Any, Iterator
from numpy.typing import ArrayLike
import numpy as np


def coordinate_names_from_array(array: ArrayLike, prefix: str = "x",
                                math_style=False) -> list[str]:
    """returns names for each coordinate of an array

    Args:
        array (ArrayLike): the array holding coordinate indices that will get names
        prefix (str, optional): start of the name. Defaults to "x".
        math_style (bool, optional): use math indexing for name starting from 1,..., dim_state

    Raises:
        ValueError: Empty Array

    Returns:
        list[str]: returns names prefix[i] for each coordinate i of array
    """

    array = np.array(array)
    math_idx = 0
    if math_style:
        math_idx = 1
    if array.size > 0:
        return [prefix + "[" + str(val + math_idx) + "]" for _, val in enumerate(array)]
    raise ValueError("Empty array")

# Function objects


def identity(input_obj: Any) -> Any:
    """an identity function

    Args:
        input_obj (Any): any input object

    Returns:
        Any: input_obj
    """
    return input_obj


def compose(*functions):
    """composes a list of functions
    compose(f,g,h) returns f(g(h(x))). Unpack it as funcs = [f,g,h] and use 
    compose(*funcs)

    Returns:
        Callable: returns a function that is composition of functions
    """
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)


def batched(iterable, size=32) -> Iterator[tuple]:
    """batch data from iterables to length size
     e.g. list(batched("ABCDE", size=3))
    [('A', 'B', 'C'), ('D', 'E')]

    Args:
        iterable (Iterable): iterable
        size (int, optional): batch size. Defaults to 32.
    """
    iterator = iter(iterable)
    while True:
        batch = tuple(itertools.islice(iterator, size))
        if not batch:
            return
        yield batch
