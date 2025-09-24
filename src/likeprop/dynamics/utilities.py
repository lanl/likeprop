'''This Module is for classes and functions useful for the dynamics package'''
import abc
from typing import Callable, NamedTuple, Optional, TypeVar, Generic
import likeprop.utils.helper_functions as luh
from likeprop.utils.typing import FloatT, ArrayFloatT

# An abstract trajectory class as collection of time series objects

T = TypeVar("T")
class Trajectory(Generic[T]):
    """creates an abstract trajectory class to store time series
    of some structured data of type T

    Args:
        collections (Sequence): returns a sequential object
    """

    def __init__(self, transformation: Optional[Callable[[T], T]] = None):
        """initializes a trajectory with initial object and a 
        transformation from object to object

        Args:
            transformation (Callable[[Any], Any], optional): a function to tranform 
            the objects. Defaults to None.
        """
        self._list = []
        if transformation is not None:
            self.transformation = transformation
        else:
            self.transformation = luh.identity

    def __getitem__(self, key) :
        return self._list[key]

    def __len__(self) -> int:
        return len(self._list)

    @abc.abstractmethod
    def store(self, obj) -> None:
        """store an object to the trajectory at some later time
        should store as a dict or a structured nd array or a tuple

        Args:
            obj (T): time varying object
        """

    def retrieve(self) -> list:
        """retrieve the entire trajectory
        """
        return self._list


class TimeData(NamedTuple):
    """a namedtuple object for holding time meta data in dynamical evolution 

    Args:
        NamedTuple:
    """
    initial_time: FloatT
    final_time: FloatT
    ordered_time_stamps: ArrayFloatT
