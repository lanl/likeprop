'''This module provides basic collection objects '''

from typing import (Protocol, runtime_checkable, Sequence,
                    Iterator, Iterable, Optional, Any, Union)
from itertools import chain
from dataclasses import dataclass
import abc
import itertools
from collections.abc import Callable
from scipy.stats import qmc  # type: ignore
import pandas as pd
import numpy as np
import likeprop.utils.typing as likeprop_typing
from likeprop.utils.typing import FloatT
from likeprop.utils.helper_functions import batched

# create a type for the "points"
PointTypeT = Union[Sequence, np.ndarray]


@runtime_checkable
class SetInEuclideanSpace(Protocol):
    """Abstract Class for defining a set in a Euclidean space

    Args:
        Protocol (_type_): protocol class for describing interface
    """

    @property
    def dim(self) -> int:  # type: ignore
        """dimension of the parent space 

        Returns: int: dim
        """
    @property
    def lebesgue_measure(self) -> float:  # type:ignore
        """returns the lebesgue measure of the set

        Returns:
            float: volume (measure) of the set
        """

    def __contains__(self, point: PointTypeT) -> likeprop_typing.BoolT:
        ...

# A few important implementation of the abstract classes


class Rect:
    """Creats a rectangle set as the cartesian product of closed 1d intervals
    in Euclidean Space
    Examples:
        >>> rect1d = Rect.from_endpoints((-1,), (2,))
        >>> rect2d = Rect.from_endpoints((-1,1), (2,3))
    """
    @classmethod
    def from_endpoints(cls, left_ends: likeprop_typing.FixedListT,
                       right_ends: likeprop_typing.FixedListT):
        """creates Rectangle from list/tuple/array of end points

        Args:
            left_ends (likeprop_typing.FixedListT): left end point a_i
            right_ends (likeprop_typing.FixedListT): right end point b_i

        Raises:
            ValueError: length of end points must be the same

        Returns:
            _type_: Rect object 
        """
        if len(left_ends) != len(right_ends):
            raise ValueError("Incorrect left and right dimensions")

        intervals = [pd.Interval(left=left, right=right, closed="both")
                     for (left, right) in zip(left_ends, right_ends)]

        return cls(dim=len(left_ends), rect=intervals)

    def __init__(self, dim, rect: list[pd.Interval]):
        """Initializes a rectangular set in state space

        Args:
            dim (int): dimension of the parent state space
            rect (list[pd.Interval]): list of pandas 1d interval objects

        Raises:
            ValueError: raises value error when the rect object length is different from dim_state
        """
        self._dim = dim
        if len(rect) != dim:
            raise ValueError(
                f"dimension of {rect} doesn't match dimension of state {dim}")
        self._rect = rect
        lengths = [interval.length for interval in self._rect]
        self.lengths = np.diag(lengths)
        self.inverse_length = np.diag([1.0/length for length in lengths])
        self.left_ends = np.array([interval.left for interval in self._rect])
        self.right_ends = np.array([interval.right for interval in self._rect])
        self._lebesgue_measure = float(np.prod(lengths))
        self.boundaries, self.boundary_fixed_val = self._boundary()

    @property
    def lebesgue_measure(self) -> float:
        """returns the volume of the rectangle

        Returns:
            float: volume of the N-rectangle
        """
        return self._lebesgue_measure

    @property
    def dim(self) -> int:
        """sets dim as read-only

        Returns:
            int: dimension of state space
        """
        return self._dim

    @property
    def rect(self) -> list[pd.Interval]:
        """sets rect as read-only

        Returns:
            list[pd.Interval]: list of pandas 1d interval objects
        """
        return self._rect

    def __contains__(self, point: PointTypeT) -> likeprop_typing.BoolT:
        if len(point) != self.dim:
            raise ValueError(
                "the state and the set are in different dimensions")
        return np.all([point[i] in self.rect[i] for i in range(self.dim)])

    def _boundary(self) -> tuple[dict[int, 'Rect'], dict[int, tuple]]:
        r"""generates the boundaries of N Rectangle
        A given N-rectangle has 2N boundaries in N-1 dimension. 
        For a fixed coordinate :math:`1\leq i\leq N` we have a two N-1 rectangle 
        that have the fixed value of min_i, max_i.

        Returns:
            tuple: tuple of dictionaries carrying the N boundary rectangles and 
                their fixed coordinates
        """

        coordinates = range(self.dim)
        boundaries: dict[int, 'Rect'] = {}
        boundary_fixed_val: dict[int, tuple] = {}
        for i in coordinates:
            left_end_boundary = (self.left_ends[j]
                                 for j in coordinates if j != i)
            right_end_boundary = (self.right_ends[j]
                                  for j in coordinates if j != i)
            # create a rect from these left and right ends
            rect_boundary = Rect.from_endpoints(left_ends=tuple(left_end_boundary),
                                                right_ends=tuple(right_end_boundary))
            # get the min and max of the fixed coordinates
            min_val = self.left_ends[i]
            max_val = self.right_ends[i]
            # store boundaries in a dictionary
            boundaries[i] = rect_boundary
            boundary_fixed_val[i] = (min_val, max_val)

        return (boundaries, boundary_fixed_val)

    def map_to_unitcube(self, point: PointTypeT):
        r"""give an point :math:`\mathbf{y}` in the rectange map it to a point 
        :math:`\mathbf{x}` in unit cube :math:`[0,1]^{\text{dim}}`

        .. math::
            \mathbf{x} = \begin{bmatrix}
                \frac{1}{l_1}& & \\
                &\ddots & \\
                & &\frac{1}{l_{\text{dim}}}
            \end{bmatrix}
            \begin{bmatrix}
            y_1 \\
            \vdots \\
            y_{\text{dim}}
            \end{bmatrix}

        Args:
            point (Sequence): point in the rectange (array like)

        Raises:
            ValueError: if the point is not in the rectangle

        Returns:
            _type_: a vector x_i with 0\leq x_i\leq 1
        """
        if point not in self:
            raise ValueError("This point is not in the rectangle")
        return np.matmul(self.inverse_length, np.array(point)) - self.left_ends

    def map_from_unitcube(self, point: PointTypeT):
        r"""give an point :math:`\mathbf{x}` in the in unit cube :math:`[0,1]^{\text{dim}}`
         map it to a point :math:`\mathbf{x}` in the 
         rectange :math:`\prod\limits_{i}^{\text{dim}}[a_i,b_i]`

        .. math::
            \mathbf{y} = \begin{bmatrix}
                \frac{1}{l_1}& & \\
                &\ddots & \\
                & &\frac{1}{l_{\text{dim}}}
            \end{bmatrix}
            \begin{bmatrix}
            x_1 \\
            \vdots \\
            x_{\text{dim}}
            \end{bmatrix}

        Args:
            point (Sequence): point in the rectange (array like)

        Returns:
            _type_: a vector x_i with 0\leq x_i\leq 1
        """

        return np.matmul(self.lengths, np.array(point)) + self.left_ends


class Ball:
    """Creates an open ball in the Euclidean space
    """

    def __init__(self, center: PointTypeT, radius: float,
                 dist: Callable[[PointTypeT, PointTypeT], float]):
        """initializes the set with radius, center and a distance function

        Args:
            center (PointTypeT): center of the open ball
            radius (float): radius of the open ball
            dist (Callable[[PointTypeT, PointTypeT], float]): a valid distance function
        """
        self._center = center
        self._radius = radius
        self._dist = dist

    @property
    def dim(self) -> int:
        """sets the dim of the space as read-only

        Returns:
            int: dim of the state
        """
        return len(self._center)

    @property
    def radius(self) -> float:
        """sets the radius of the ball as read-only

        Returns:
            float: radius of the open ball
        """
        return self._radius

    @property
    def center(self) -> PointTypeT:
        """sets the center of the ball as read-only

        Returns:
            float: center of the open ball
        """
        return self._center

    @property
    def dist(self) -> Callable[[PointTypeT, PointTypeT], float]:
        """sets the distance function as read-only

        Returns:
            dist (Callable): distance function
        """
        return self._dist

    def __contains__(self, point: PointTypeT) -> likeprop_typing.BoolT:
        if len(point) != self.dim:
            raise ValueError(
                "the state and the set are in different dimensions")
        return self.dist(self.center, point) < self.radius


def indicator_set(set_obj: SetInEuclideanSpace, point: PointTypeT):
    """indicator function of a set

    Args:
        set_A (SetInEuclideanSpace): set object
        point (PointTypeT): point to check

    Returns:
        float: 1 if x is in the set, 0 otherwise
    """
    if point in set_obj:
        return 1.0
    return 0.0


def in_set(list_state: list[PointTypeT],
           set_obj: SetInEuclideanSpace,
           ) -> list:
    """returns all such states in a given list of states that are inside
    the set object

    Args:
        list_state (list[PointTypeT]): list of states (or array)
        set_obj (SetInEuclideanSpace, optional): set in euclidean space. 
        Defaults to Rect.from_endpoints(left_ends=(-1, -1), right_ends=(1, 1)).

    Returns:
        list: all points in the list that are in the set
    """

    return [x for x in list_state if x in set_obj]


def count_in_set(list_state: list[PointTypeT],
                 set_obj: SetInEuclideanSpace):
    """returns the total number of all such states in a given 
    list of states that are inside the set object

    Args:
        list_state (list[PointTypeT]): list of states (or array)
        set_obj (SetInEuclideanSpace, optional): set in euclidean space. 
        Defaults to Rect.from_endpoints(left_ends=(-1, -1), right_ends=(1, 1)).

    Returns:
        int: number of points in the list that are in the set
    """

    return len([x for x in list_state if x in set_obj])


class SampleSet(Protocol):
    """An abstract iterable class to get points from a set in the euclidean space

    Args:
        Protocol (_type_): Abstract class
    """

    set_obj: SetInEuclideanSpace

    @abc.abstractmethod
    def __iter__(self) -> Iterator[Any]:
        """generates a set of N d-dimensional points
        such that each point is in the set. 

        Returns:
            Iterable: an iterable of points (N,d)
        """

# Some basic SampleSet classes.


class GridRect:
    """grids a d-dimensional closed rectangle

    """

    def __init__(self, set_obj: Rect,
                 num_points_per_dim: int = 20,
                 batch_size: int = 32):

        self.set_obj = set_obj
        self.num_points_in_each_dim = num_points_per_dim
        self.batch_size = batch_size
        self.dim = self.set_obj.dim
        self.total_points = int(
            np.power(self.num_points_in_each_dim, self.dim))
        if self.batch_size > self.total_points:
            self.batch_size = self.total_points

    def __iter__(self) -> Iterator[tuple]:
        """grids a rectangle to produce (N, d) points that are 
        in the set

        Returns:
            Iterator: gridded points of size (N,d)
        """
        grids = (np.linspace(interval.left, interval.right, self.num_points_in_each_dim,
                             endpoint=True)for interval in self.set_obj.rect)

        return batched(itertools.product(*grids), size=self.batch_size)

# still memory inefficient for very large values of samples.


@dataclass
class SobolSampleOptions:
    """meta data for the most basic sobol 
    samples 
    """
    batch_size: int = 32
    total_samples: int = 1024
    one_batch: bool = False
    fixed_coord: Optional[int] = None
    fixed_coord_value: Optional[FloatT] = None


# use the default fields to initialize sample options for Sobol
default_sobol_options = SobolSampleOptions()


class SobolRect:
    """creates sobol samples for the rectangle. 
    batch_size and total_size are powers of 2.
    If you need the whole batch at once, pass in one_batch = True
    """

    def __init__(self, set_obj: Rect,
                 options: SobolSampleOptions = default_sobol_options):
        r"""iterator for sobol samples from a d-dim Rect Object

        Args:
            set_obj (Rect): a :math:`d` dimensional Rectangle
            options.batch_size (int, optional): batch size of samples. Defaults to 32.
            options.total_samples (int, optional): total number of samples. Defaults to 1024.
            options.one_batch (bool, optional): whether we only want one batch. Defaults to False.
            options.fixed_coord (int, optional): whether a particular 
                coordinate :math:`i < i < d+1` is fixed. Defaults to None.
            options.fixed_coord_value (float, optional): value at the fixed coordinate. 
                Defaults to None.


        Raises:
            ValueError: When total samples exceed 2^30
            ValueError: When batch size is greater than total samples
        """

        # make sure that the total samples are powers of 2.
        power_of_two = int(np.log2(options.total_samples))

        if power_of_two > 30:
            raise ValueError("Decrease total samples. \
                             Total samples allowed is 2**30")

        allowed_samples = np.power(2, power_of_two)
        self.total_samples = options.total_samples
        if allowed_samples != options.total_samples:

            print("Warning: total samples are not a power of 2 \
                  changing total samples to the nearest power of 2 ")
            self.total_samples = allowed_samples

        # check if we just need one batch of the entire data
        batch_size = options.batch_size
        if options.one_batch:
            batch_size = self.total_samples

        if self.total_samples < batch_size:
            raise ValueError("Batch size must be less than total samples")

        # store instance variables
        self.set_obj = set_obj
        self.dim = self.set_obj.dim

        self.batch_size = batch_size
        self.total_size_exponent = power_of_two

        self.fixed_coord = options.fixed_coord
        self.fixed_coord_value = options.fixed_coord_value

        # create a sobol sampler
        self.sampler = qmc.Sobol(d=self.dim)

    def __iter__(self) -> Iterator[np.ndarray]:
        """draws Sobol points to produce (Numpoints, dimension) points that are 
        in the set

        Returns:
            Iterator: sobol points of size (N,d)
        """
        # scale the qmc to make points in a unit cube fall inside the rectangle
        batch = qmc.scale(self.sampler.random_base2(self.total_size_exponent),
                          self.set_obj.left_ends, self.set_obj.right_ends)

        # return batched iterator
        for i in range(0, self.total_samples, self.batch_size):
            abatch = batch[i:i+self.batch_size, :]
            # add a fixed coordinate value if it is passed for e.g. in boundary
            if self.fixed_coord is not None:
                if self.fixed_coord_value is not None:
                    yield np.insert(abatch, obj=self.fixed_coord,
                                    values=self.fixed_coord_value, axis=1)
            else:
                yield abatch


def get_sobol_rect_boundary(rect: Rect,
                            options: SobolSampleOptions = default_sobol_options
                            ) -> Iterable:
    """Iterable that contains sobol samples from the 2*d boundaries of an d-dim rectangle

    Args:
        rect (Rect): d-dim rectangle
        options (SobolSampleOptions, optional): options for sobol samples. 
            Defaults to default_sobol_options.

    Returns:
        Iterable: returns (Batch size, d-dim) points from each 2*d boundaries
    """

    boundaries = rect.boundaries
    iterable = []

    for coordinate, boundary_rect in boundaries.items():
        if boundary_rect.dim == 0:
            # returns (2, 1) points in 1 d space
            iterable.append([np.array([rect.boundary_fixed_val[coordinate][0]])])
            iterable.append([np.array([rect.boundary_fixed_val[coordinate][1]])])
            return iterable

        # for each fixed coordinate there are two values on a d-1 dim rectangle
        options.fixed_coord = coordinate
        options.fixed_coord_value = rect.boundary_fixed_val[coordinate][0]
        iterable.append(SobolRect(boundary_rect, options=options))
        options.fixed_coord_value = rect.boundary_fixed_val[coordinate][1]
        iterable.append(SobolRect(boundary_rect, options=options))

    boundary_iterable = chain.from_iterable(iterable)
    return boundary_iterable
    # for point in boundary_iterator:
    #     print(point.shape)
    #     print(point)
