'''This module describes various pipelines to run likeprop for ensembles'''
from typing import Any, Union
from functools import partial
import numpy as np
from likeprop.utils.typing import ArrayFloatT
from likeprop.dynamics.ode_model import DynamicalSystem
from likeprop.dynamics.utilities import TimeData
from likeprop.dynamics.state import StateTrajectory
from likeprop.density.distributions import ProbabilityDensityFunction
from likeprop.utils.helper_functions import compose

# pylint: disable=invalid-name


def array_of_trajectories(ensemble_traj: list[StateTrajectory]):
    """return ensemble of trajectories as a array of size 
    (num_ensemble, time_stamps, dim_state)

    Args:
        ensemble_traj (list[StateTrajectory]): ensemble of trajectories

    Raises:
        ValueError: Empty list of trajectories

    Returns:
        np.ndarray: array of shape (num_ensemble, time_stamps, dim_state)
    """

    if not ensemble_traj:
        raise ValueError("Empty list of trajectories")

    # TODO: modify it when each trajectory is a dict type

    # if ensemble_traj[0].representation_type.name == 'DICT':
    #     return np.array([(traj) for traj in ensemble_traj])

    return np.array([np.array(traj) for traj in ensemble_traj])


def augment_likelihood_to_state(state: ArrayFloatT,
                                pdf: ProbabilityDensityFunction):
    """appends likelihood value to a state vector

    Args:
        state (ArrayFloatT): array representation of state without likelihood
        pdf (ProbabilityDensityFunction): Density Function

    Returns:
        _type_: _description_
    """
    likelihood = pdf.relative_likelihood(state)
    likelihood = np.expand_dims(likelihood, axis=-1)
    return np.concatenate((state, likelihood))


def augment_likelihood_to_augmented_state(augmented_state: ArrayFloatT,
                                          dim_state: int,
                                          pdf: ProbabilityDensityFunction):
    """appends likelihood value to a state vector augmented with model parameters

    Args:
        state (ArrayFloatT): array representation of state and 
        parameters without likelihood
        pdf (ProbabilityDensityFunction): Density Function

    Returns:
        _type_: _description_
    """
    likelihood = pdf.relative_likelihood(augmented_state)
    likelihood = np.expand_dims(likelihood, axis=-1)
    return np.concatenate((augmented_state[:dim_state], likelihood))


def get_final_state_from_trajectory(traj: StateTrajectory):
    """returns the final state stored in the trajectory object

    Args:
        traj (StateTrajectory): a trajectory of states

    Returns:
        _type_: array or dict representation of the final state
    """
    return traj[-1]


class SequentialPipeline:
    """A pipeline class for running functions sequentially on inputs
    pipeline(x) returns output where x passes through the functions defined in pipeline
    """

    def __init__(self):
        self._func_list = []
        self._name_list = []

    def add(self, func, description: str):
        """adds a function to the pipeline

        Args:
            func (Callable): function to applied to the output of the previous 
            function in the pipeline
            description (str): a short description of what the function does
        """
        self._func_list.insert(0, func)
        self._name_list.insert(0, description)
        return self

    def get_info(self, index: Union[int, slice] = slice(0, None, None)):
        """returns the info of selected functions in the pipeline

        Args:
            index (Union[int, slice]): indices in the pipeline whose info is needed

        Returns:
            list[str]: info of the queried functions in the pipeline 
        """
        return self._name_list[index]

    def __call__(self, x: Any) -> Any:
        """use pipeline as a function

        Args:
            x (Any): input

        Raises:
            ValueError: if pipeline is empty

        Returns:
            Any: output
        """
        if not self._func_list:
            raise ValueError("The pipeline is empty")
        pipeline = compose(*self._func_list)
        return pipeline(x)

# same basic pipelines


def get_ensemble_of_trajectories_from_state(time_data: TimeData, dynamical_system: DynamicalSystem,
                                            params=None) -> SequentialPipeline:
    """pipeline that takes (batch, initial_state) and returns (batch, time_steps, final_state)

    Args:
        time_data (TimeData): Time data for dynamical system
        state_dynamical_system (DynamicalSystem): dynamical system to evole an ODE
        params (np.ndarray, optional): optional model parameters. Defaults to None.

    Returns:
        SequentialPipeline: a pipeline object
    """
    evolve_state = partial(dynamical_system.evolve,
                           time_data=time_data, params=params)
    evolve_state = partial(map, evolve_state)
    pipeline = SequentialPipeline()
    pipeline.add(
        evolve_state, 'evolves state from an ensemble of initial states')
    pipeline.add(list, 'convert generator objects to list')
    pipeline.add(array_of_trajectories,
                 'convert list of trajectories to array of shape (ensemble, time, dim_state)')
    return pipeline
