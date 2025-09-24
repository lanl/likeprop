'''This module defines an abstract state and augmented state class that is 
evolved by some dynamical system.
'''
from typing import (NamedTuple, Callable, TypeVar,
                    Optional, Any, cast)
from dataclasses import dataclass, field, asdict
from collections import namedtuple
from enum import Enum
import collections.abc
import numpy as np
import pandas as pd
import likeprop.utils.collections as luc
import likeprop.utils.helper_functions as luh
import likeprop.dynamics.utilities as ldu
import likeprop.utils.typing as likeprop_typing

# create the attributes associated with a state
StateFields = namedtuple(
    "state_fields", "dim_state time xvals coords likelihood")

# define the two types of state representation


class StateRepresentation(Enum):
    """two different representations of the state

    Args:
        Enum (_type_): categories of state representation
    """
    ARRAY = "array_representation"
    DICT = "dictionary_representation"

# create the State class allowing users to provide and get any of the two representations.
# provides an __array__ interface


@dataclass
class State:
    r"""Creates A Simple ArrayLike State
    .. math::
        \mathbf{x}(t) = (x_1(t), \cdots, x_{d}(t)).

    A state can have two different representations. 
    An array based representation of the state :math:`\mathbf{x}(t)` is
    given by the array
    .. math::
            xvals = \begin{bmatrix}
                x_1(t) \\
                \vdots \\
                x_d(t) \\
                \text{Optional likelihood}
            \end{bmatrix}

    Thus an array based representation is of size :math:`d` or :math:`d+1`. On the other
    hand a dictionary representation of the state :math:`\mathbf{x}(t)` is more descriptive and 
    given by a dictionary dict where dict['t'] is the current time :math:`t` of the state,
    dict['ith coordinate'] = x_i(t) and dict['likelihood'] is the likelihood associated
    with the state. This representation is useful for plotting. 

    Attributes:
        dim_state: dimension :math:`d` of state
        time: current time
        xvals: internal representation of the state :math:`\mathbf{x}`
        coords (list[str]): list of coordinate names for e.g :math:`x[1], x[2], \cdots`
        likelihood (likeprop.utils.typing.FloatT): Optional likelihood :math:`\ell(\mathbf{x},t)`
                                                    associated with the state, defaults to -1.
    Class Methods:
        dictionary_representation_to_fields: get attributes as named tuple from a coordinate 
                                            representation
        array_to_fields: get attributes as named tuple from an array representation
        from_dictionary_representation: get a State object fromdictionary representation

        from_array: get a State object from an array
        fields_to_dictionary_representation: get dictionary representation
         from (dictionary based) attributes
        fields_to_array_representation: get array representation from (dictionary based) 
         attributes
        array_to_dictionary_representation: change of representation from coordinate to array based

    Methods:
        to_array(): returns array representation of state
        to_dict(): returns dictionary representation
         of state
        __array__(): array interface to the state.
    """

    dim_state: int
    time: likeprop_typing.FloatT
    xvals: likeprop_typing.ArrayFloatT
    coords: list[str] = field(default_factory=list, repr=False)
    likelihood: likeprop_typing.FloatT = field(default=-1.0, repr=False)

    @classmethod
    def array_to_fields(cls, data_array: likeprop_typing.ArrayLike,
                        *,
                        time: likeprop_typing.FloatT,
                        likelihood: likeprop_typing.BoolT = False,
                        coords: Optional[list[str]] = None) -> StateFields:
        """get fields of State from an array representation

        Args:
            cls (Type[State]): State
            data_array (likeprop_typing.ArrayLike): array holding the data
            time (likeprop_typing.FloatT): current time of state
            coords (Optional[list[str]], optional): name of the coordinates. Defaults to None.
            likelihood (Optional[likeprop_typing.FloatT], optional): likelihood. Defaults to None.

        Raises:
            ValueError: Empty Array

        Returns:
            NamedTuple: named tuple with fields
        """

        arr = np.array(data_array, dtype=np.float64)
        if arr.size == 0:
            raise ValueError("Empty data_array")

        # default values
        likelihood_value = -1.0
        dim_state = arr.size
        xvals = arr

        # change default if likelihood is associated with the state
        if likelihood:
            likelihood_value = arr[-1]
            dim_state = arr.size - 1
            if dim_state == 0:
                raise ValueError("No State Only Likelihood")
            xvals = arr[:-1]

        if coords is None:
            coords = luh.coordinate_names_from_array(np.arange(dim_state))

        return StateFields(dim_state=dim_state,
                           time=time,
                           xvals=xvals,
                           coords=coords,
                           likelihood=likelihood_value)

    @classmethod
    def dictionary_representation_to_fields(cls, data_dict: dict[str, likeprop_typing.FloatT],
                                            *,
                                            prefix_xval="x") -> StateFields:
        """get fields of State from dictionary representation


        Args:
            cls (Type[State]): State class method
            data_dict (dict[str, likeprop_typing.FloatT]): dictionary for the coordinate 
            representation of state containing data of the class
            prefix_xval (str, optional): name of the coordinates. Defaults to "x".

        Returns:
            instance of State: State Instance
        """

        xvals = np.array([data_dict[key] for key in data_dict if key.startswith(prefix_xval)], # type: ignore
                         dtype=np.float64)
        coords = [key for key in data_dict if key.startswith(prefix_xval)]
        dim_state = xvals.size
        time = data_dict['t']

        # default value of likelihood
        likelihood_value = -1.0

        if 'likelihood' in data_dict:
            likelihood_value = data_dict['likelihood']

        return StateFields(dim_state=dim_state,
                           time=time,
                           xvals=xvals,
                           coords=coords,
                           likelihood=likelihood_value)

    @classmethod
    def from_dictionary_representation(cls, data_dict: dict[str, likeprop_typing.FloatT],
                                       *,
                                       prefix_xval="x"):
        """returns an object of State from a dictionary representation

        Args:
            cls (Type[State]): State class method
            data_dict (dict[str, likeprop_typing.FloatT]): dictionary 
            of dictionary representation
             containing data of the class
            prefix_xval (str, optional): name of the coordinates. Defaults to "x".

        Returns:
            instance of State: State Instance
        """
        fields = cls.dictionary_representation_to_fields(
            data_dict=data_dict, prefix_xval=prefix_xval)

        return cls(**fields._asdict())

    @classmethod
    def from_array(cls, data_array: likeprop_typing.ArrayLike,
                   *,
                   time: likeprop_typing.FloatT,
                   likelihood: likeprop_typing.BoolT = False,
                   coords: Optional[list[str]] = None):
        """create an instance from array

        Args:
            cls (Type[State]): State
            data_array (likeprop_typing.ArrayLike): array holding the data
            time (likeprop_typing.FloatT): current time of state
            coords (Optional[list[str]], optional): name of the coordinates. Defaults to None.
            likelihood (Optional[likeprop_typing.FloatT], optional): likelihood. Defaults to None.

        Raises:
            ValueError: Empty Array

        Returns:
            StateT: instance of State
        """

        fields = cls.array_to_fields(
            data_array=data_array, time=time, likelihood=likelihood, coords=coords)
        return cls(**fields._asdict()) # type: ignore

    @classmethod
    def fields_to_dictionary_representation(cls, fields: dict) -> dict:
        """returns a dictionary representation of the state given field values

        Args:
            fields (dict): all fields of the state

        Returns:
            dict: dictionary based representation of the state
        """
        time_dict = {'t': fields['time']}
        states_dict = dict(zip(fields['coords'], fields['xvals']))
        dict_representation = time_dict | states_dict
        if fields['likelihood'] >= 0.0:
            likelihood_dict = {'likelihood': fields['likelihood']}
            return dict_representation | likelihood_dict
        return dict_representation

    @classmethod
    def fields_to_array_representation(cls, fields: dict,
                                       dtype=np.float64) -> likeprop_typing.ArrayFloatT:
        """returns dictionary representation of the state given field values

        Args:
            fields (dict): all fields of the state

        Returns:
            dict: a description dictionary based representation of the state
        """
        if fields['likelihood'] >= 0.0:
            return np.concatenate((fields['xvals'], np.array([fields['likelihood']])))
        return np.array(fields['xvals'], dtype=dtype)

    @classmethod
    def array_to_dictionary_representation(cls, data_array: likeprop_typing.ArrayLike,
                                           *,
                                           time: likeprop_typing.FloatT,
                                           likelihood: likeprop_typing.BoolT = False,
                                           coords: Optional[list[str]] = None
                                           ) -> dict:
        """given an array get a dictionary respresentation of the fields

        Args:
            data_array (likeprop_typing.ArrayLike): array representation of the state
            time (likeprop_typing.FloatT): current time of the state
            likelihood (likeprop_typing.BoolT): is likelihood associated with the state
            coords (Optional[list[str]], optional): coordinate names of the state. Defaults to None.

        Returns:
            dict: dictionary representation of the state
        """

        fields = cls.array_to_fields(data_array=data_array,
                                     time=time,
                                     likelihood=likelihood,
                                     coords=coords)

        return cls.fields_to_dictionary_representation(fields=fields._asdict())

    def __post_init__(self):
        """initialize the coordinate names
        """
        if self.xvals.size != self.dim_state:
            raise ValueError(
                "Dimension of state and size of xvals don't match")
        if not self.coords:
            self.coords = luh.coordinate_names_from_array(
                np.arange(self.dim_state))

    def __array__(self, dtype=np.float64):
        """Array Interface to the State

        Args:
            dtype (type, optional): type of array elements. Defaults to np.float64.

        Returns:
            np.ndarray[dim_state (+1)]: numpy array copy of the xvals appended to likelihood
                                        if present.
        """
        fields = asdict(self)
        return self.__class__.fields_to_array_representation(fields=fields, dtype=dtype)

    def to_dict(self) -> dict[str, likeprop_typing.FloatT]:
        """returns the state as a dictionary of the dictionary representation
         of the state

        Returns:
            dict[str, float]: dict[key] = value
            where key = coord_i (or 't' or 'likelihood') and value is s.xvals[i] 
            (or s.time or s.likelihood if we have set it)
        """
        fields = asdict(self)
        return self.__class__.fields_to_dictionary_representation(fields=fields)

    def to_array(self) -> likeprop_typing.ArrayFloatT:
        """returns the array representation of the state
        by calling the array interface __array__

        Returns:
            likeprop_typing.ArrayFloatT: Array of Float (numpy)
        """
        return np.asarray(self)

# augment a dictionary representation
#  of state with parameters


def dictionary_representation_with_params(coord_representation: dict[str, likeprop_typing.FloatT],
                                          *,
                                          param_prefix: str = "theta",
                                          params: likeprop_typing.ArrayFloatT
                                          ) -> dict[str, likeprop_typing.FloatT]:
    """appends model parameters to the dictionary representation
     of a state

    Args:
        coord_representation (dict[str, likeprop_typing.FloatT]): dictionary representation
         of state
        params (likeprop_typing.ArrayFloatT): model parameters
        param_prefix (str, optional): name of the parameter coordinates. Defaults to "theta".

    Returns:
        dict[str, likeprop_typing.FloatT]: dictionary representation
         along with model parameters
    """
    # get names of params
    param_names = luh.coordinate_names_from_array(
        np.arange(params.size), prefix=param_prefix)
    params_dict = dict(zip(param_names, params))
    return coord_representation | params_dict  # type: ignore


def array_representation_with_params(data_array: likeprop_typing.ArrayLike,
                                     *,
                                     time: likeprop_typing.FloatT,
                                     likelihood: likeprop_typing.BoolT,
                                     coords: Optional[list[str]] = None,
                                     params: likeprop_typing.ArrayFloatT
                                     ) -> likeprop_typing.ArrayFloatT:
    """augment parameters with array representation of state

    Args:
        data_array (likeprop_typing.ArrayLike): array representation of state
        time (likeprop_typing.FloatT): current time
        likelihood (likeprop_typing.BoolT): is likelihood associated with state
        params (likeprop_typing.ArrayFloatT): model parameters
        coords (Optional[list[str]], optional): coordinate names. Defaults to None.

    Returns:
        Array of Floats: augmented array representation with (x,theta,likelihood)
    """

    fields = State.array_to_fields(data_array=data_array,
                                   time=time,
                                   likelihood=likelihood,
                                   coords=coords)

    if fields.likelihood >= 0.0:
        return np.concatenate((fields.xvals, params, np.array([fields.likelihood])))

    return np.concatenate((fields.xvals, params))


# create an augmented state as a namedtuple holding state and params

class AugmentedState(NamedTuple):
    r"""Augmented State Class with state and parameters
    .. math::
        \mathbf{s(t)} = (\mathbf{x}(t), \boldsymbol{\theta}) 
    An array based representation of the augmented state :math:`\mathbf{s}(t)` is
    given by the array
    .. math::
            xvals = \begin{bmatrix}
                x_1(t) \\
                \vdots \\
                x_d(t) \\
                \theta_1 \\
                \vdots \\
                \theta_p\\
                \text{Optional likelihood} \\
            \end{bmatrix}

    Args:
        NamedTuple (typing.NamedTuple): namedtuple base

    Attributes:
        state (State): state object at time :math:`t`
        params (likeprop_typing.ArrayFloat): Array of model parameters :math:`\boldsymbol{\theta}`

    Methods:
        to_array(): array representation of the augmented state
        to_dict(): dictionary representation
         of the augmented state

    Returns:
        AugmentedState: an augmented state class
    """
    state: State
    params: likeprop_typing.ArrayFloatT

    def to_array(self) -> likeprop_typing.ArrayFloatT:
        """array based representation of the augmented state

        Returns:
            np.ndarray: array of state.x and params
        """
        if self.state.likelihood >= 0.0:
            return np.concatenate((self.state.xvals, self.params,
                                   np.array([self.state.likelihood])))
        return np.concatenate((self.state.xvals, self.params))

    def to_dict(self, param_prefix="theta") -> dict[str, likeprop_typing.FloatT]:
        """returns dictionary representation
         of the augmented state

        Returns:
            dict[str, float]: dict[key] = value
            where key = coord_i (or theta_i or t) and value is s.x[i] or (params[i],s.t)
        """
        state_dict = self.state.to_dict()
        return dictionary_representation_with_params(coord_representation=state_dict,
                                                     params=self.params,
                                                     param_prefix=param_prefix)


# Type Variables and Type Aliases
# State type variable is State object or anything that inherits from it
StateTypeT = TypeVar('StateTypeT', bound=State)
# State to State functions
StateTransformationT = Callable[[StateTypeT], StateTypeT]


class StateTransformations:
    r"""Composes transformation of states
    given a sequence of functions 
    .. math::
        f_1, f_2, \cdots, f_n
    where each :math:`f_i`: is defined on :math:`\mathcal{X}` of suitable dimension i.e.
    each :math:`f_i` should take a state object and return another state object 

    Returns: 
        g (StateTransformationT): the following composition  
    .. math::
        g\mathbf{x}(t) = f_1(\quad f_2 \quad (\cdots f_n(\mathbf{x}(t)))) 
    """

    def __init__(self, *args: Callable[[Any], Any]) -> None:
        """takes a bunch of functions that map state to state
        """

        self._transformations = cast(StateTransformationT, luh.compose(*args))

    @property
    def transformation(self) -> StateTransformationT:
        """returns the composed function

        Returns:
            StateTransformationT: transformation function
        """
        return self._transformations


# create a projection function
def coordinate_projection(state_obj: StateTypeT,
                          coordinates: likeprop_typing.SliceT
                          ) -> StateTypeT:
    """projects a state to a state of reduced dimension given by coordinates

    Args:
        state_obj (StateTypeT): a State type object
        coordinates (Union[slice, list[int], int]): list or int or slice of 
                                                    coordinates to project along

    Raises:
        TypeError: If incorrect slice
        Exception: If coordinates are out of bound

    Returns:
        state (StateTypeT): projected state of reduced dimension
    """

    if isinstance(coordinates, int):
        coordinates = [coordinates]

    try:
        x_array = np.array(state_obj.xvals)[coordinates]
        if x_array.size > 0:
            projected_state = state_obj.__class__(dim_state=len(x_array), time=state_obj.time,
                                                  xvals=x_array,
                                                  likelihood=state_obj.likelihood)
            projected_state.coords = luh.coordinate_names_from_array(np.r_[
                                                                     coordinates])
            return projected_state
        # else:
        raise TypeError("Incorrect coordinate object")

    except Exception as error:
        raise error

# create Trajectory classes to store state


class StateTrajectory(ldu.Trajectory[StateTypeT]):
    """default state trajectory

    Args:
        (Trajectory): Trajectory type

    Attributes:
        representation_type (StateRepresentation): type of representation to store
        params (ArrayFloatT): model parameters. optional. Defaults to None.


    Returns:
        _type_: State Trajectory
    """

    def __init__(self, representation_type: StateRepresentation = StateRepresentation.ARRAY,
                 params: Optional[likeprop_typing.ArrayFloatT] = None,
                 transformation: Optional[Callable[[
                     StateTypeT], StateTypeT]] = None,
                 is_likelihood_evolved: bool = False,
                 ):
        super().__init__(transformation)

        self.representation_type = representation_type
        self.params = params
        self._time_list = []
        self.is_likelihood_evolved = is_likelihood_evolved

    @property
    def time_stamps(self):
        """returns the time stamps of the trajectory

        Returns:
            list: time values for which the state was stored
        """
        return self._time_list

    def store(self, obj: StateTypeT) -> None:
        """stores the state in the trajectory after applying transformation

        Args:
            obj (State): state object
            params (likeprop_typing.ArrayFloaT): optional array of parameters. Defaults to None
        """
        transformed = self.transformation(obj)
        current_time = obj.time
        self._time_list.append(current_time)
        if self.params is not None:
            transformed = AugmentedState(
                state=transformed, params=self.params)

        match self.representation_type.name:
            case 'DICT':
                self._list.append(transformed.to_dict())
            case 'ARRAY':
                self._list.append(transformed.to_array())
            case _:
                raise TypeError("Incorrect State Representation")

    def to_dataframe(self) -> pd.DataFrame:
        """returns a state or augmented state trajectory as a pandas dataframe

        Returns:
        pd.DataFrame: time series as pandas dataframe
        """
        data_frame = pd.DataFrame.from_records(self.retrieve())
        if self.representation_type.name == "ARRAY":
            data_frame.insert(0, 't', self._time_list)
        return data_frame

# create a state trajectory object by reading from array


class StateTrajectoryFromArray(StateTrajectory):
    """A Class for storing a trajectory of state From Array Based Representation 

    Args:
        ldu (Trajectory[StateTypeT]): base trajectory with State object

    Methods:
        store(state, params=None): store a state to the trajectory. 
    """

    # TODO: use transformation object currently neglected
    def store(self, obj: likeprop_typing.ArrayLike,
              *,
              time: likeprop_typing.FloatT = 0.0) -> None:
        """stores the state in the trajectory after applying transformation


        Args:
            obj (State): state object
            params (likeprop_typing.ArrayFloaT): optional array of parameters. Defaults to None
        """
        is_likelihood = self.is_likelihood_evolved
        self._time_list.append(time)
        match self.representation_type.name:
            case 'ARRAY':
                data = obj
                if self.params:
                    data = array_representation_with_params(data_array=obj,
                                                            time=time,
                                                            likelihood=is_likelihood,
                                                            coords=None,
                                                            params=self.params)
            case 'DICT':
                data = State.array_to_dictionary_representation(data_array=obj,
                                                                time=time,
                                                                likelihood=is_likelihood,
                                                                coords=None)
                if self.params:
                    data = dictionary_representation_with_params(coord_representation=data,
                                                                 params=self.params)
            case _:
                raise TypeError("Invalid object passed to store")
        self._list.append(data)

# Create Attribute Class For StateTrajectory Type


@dataclass
class StateTrajectoryFields:

    """Attributes Describing A State Trajectory

    Args:
        NamedTuple (_type_): field attributes for
        state trajectory type object
    """
    representation_type: StateRepresentation = StateRepresentation.ARRAY
    params: Optional[likeprop_typing.ArrayFloatT] = None
    transformation: Optional[StateTransformationT] = None
    is_likelihood_evolved: bool = False


# use the default attributes to initialize a StateTrajectory class
default_state_trajectory_fields = StateTrajectoryFields()

# **** A State Class Based On Numpy Interface
# create numpy handled functions for implementing numpy
# based functions on state vectors
# implemented functions are np.min, np.max, np.mean
# np.argmin, np.argmax and np.shape
HANDLED_FUNCTIONS = {}
StateVecTypeT = TypeVar('StateVecTypeT')


@dataclass
class StateVec(State, collections.abc.Sequence):
    r"""A numpy based implementation of the State class

    Args:
        State, collections (State, abc.Sequence): inherits from the Sequence and State

    Methods:
        __len__(): len(obj) returns the length of xvals
        __getitem__(key): obj[key] returns the coordinates given by obj.xvals[key] 
        __array_ufunc__(ufunc): unfunc(obj) applies a numpy ufunc to the object and 
                                returns a new obj
        __array_function(func): func(obj) applies numpy func like np.min, np.max to obj 
                                and returns a new obj
        shape(): returns the size of xvals
        in_set(set_obj): checks if obj is in the set_obj

    Raises:
        ValueError: When dim_state and length of xvals don't match
        NotImplementedError: only implements numpy functions described in HANDLED_FUNCTIONS

    Returns:
        StateVec (State, Sequence): a numpy wrapped state class
    """

    def __len__(self):
        return self.dim_state

    def __getitem__(self, key):
        return self.xvals[key]

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        '''
        Apply a numpy Universal Function (ufunc) to the state, 
        for e.g np.sin(state)
        '''
        # remove the instance from inputs
        inputs = [x for x in inputs if not isinstance(x, self.__class__)]

        if method == '__call__':
            inp = []

            inp.append(self.xvals)
            if len(inputs):
                inp.append(inputs[:])

            temp = ufunc.__call__(*inp, **kwargs)
            return self.__class__(dim_state=self.dim_state, time=self.time,
                                  xvals=temp, coords=self.coords, likelihood=self.likelihood)
        # else other methods are not needed for state vectors
        raise NotImplementedError(
            f"The method {method} in {ufunc} is not implemented for this class")

    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        # Note: this allows subclasses that don't override
        # __array_function__ to handle State objects
        if not all(issubclass(state_type, StateVec) for state_type in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def shape(self):
        """returns the shape as a tupel

        Returns:
            tuple: shape of the state vector
        """
        return np.shape(self.xvals)

    def in_set(self, set_obj: luc.SetInEuclideanSpace) -> likeprop_typing.BoolT:
        """checks if the state is in the set set_obj

        Args:
            set_obj (luc.SetInStateSpace): a set object

        Returns:
            bool: true if the state is in the set
        """
        return self in set_obj


# define decorator functions to implement np array functions
def implements(numpy_function):
    """Register an __array_function__ implementation for StateVec objects."""
    def decorator(func):
        HANDLED_FUNCTIONS[numpy_function] = func
        return func
    return decorator

# np.mean


@implements(np.mean)
def np_mean_for_statevec(state: StateVec, *args, **kwargs):
    "Implementation of np.sum for StateVec objects"
    mean_value = np.mean(state.xvals, *args, **kwargs)
    # construct a Physical instance with the result, using the same unit
    return mean_value

# np.max


@implements(np.max)
def np_max_for_statevec(state: StateVec, *args, **kwargs):
    "Implementation of np.max for StateVec objects"
    max_value = np.max(state.xvals, *args, **kwargs)
    # construct a Physical instance with the result, using the same unit
    return max_value

# np.min


@implements(np.min)
def np_min_for_statevec(state: StateVec, *args, **kwargs):
    "Implementation of np.min for StateVec objects"
    min_value = np.min(state.xvals, *args, **kwargs)
    # construct a Physical instance with the result, using the same unit
    return min_value

# np.shape


@implements(np.shape)
def np_shape_for_statevec(state: StateVec):
    "Implementation of np.shape for StateVec objects"
    shape = state.shape()
    # construct a Physical instance with the result, using the same unit
    return shape

# np.argmin


@implements(np.argmin)
def np_argmin_for_statevec(state: StateVec, *args, **kwargs):
    "Implementation of np.argmin for StateVec objects"
    argmin = np.argmin(state.xvals, *args, **kwargs)
    # construct a Physical instance with the result, using the same unit
    return argmin

# np.argmax


@implements(np.argmax)
def np_argmax_for_statevec(state: StateVec, *args, **kwargs):
    "Implementation of np.argmax for StateVec objects"
    argmax = np.argmax(state.xvals, *args, **kwargs)
    # construct a Physical instance with the result, using the same unit
    return argmax
