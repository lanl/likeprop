'''This module describes methods to propagate and estimate relative likelihoods'''

import copy
import abc
from typing import Optional, Type, Protocol, Iterable, Callable, List,runtime_checkable, Any
from functools import partial
from dataclasses import dataclass
import numpy as np

from likeprop.utils.collections import (SetInEuclideanSpace, Rect, SobolSampleOptions, SobolRect,
                                        count_in_set, in_set, get_sobol_rect_boundary)
from likeprop.dynamics.ode_model import OdeModel, DynamicalSystem, DynamicalSystemSolveIvp
from likeprop.dynamics.state import (State, StateVec, StateTrajectory, StateTrajectoryFields,
                                     default_state_trajectory_fields)
from likeprop.dynamics.utilities import TimeData
from likeprop.density.distributions import ProbabilityDensityFunction
import likeprop.utils.typing as likeprop_typing
from likeprop.dynamics.pipelines import (
    SequentialPipeline, get_ensemble_of_trajectories_from_state)

# Abstract Likelihood Class


@runtime_checkable
class LikelihoodManager(Protocol):
    """abstract class for propagating and computing likelihood of states


    Args:
        Protocol (_type_): Interface
    """
    initial_density: ProbabilityDensityFunction

    @abc.abstractmethod
    def evolve(self, initial_state: State, *,
               time_data: TimeData,
               params: Optional[likeprop_typing.ArrayFloatT] = None
               ) -> StateTrajectory:
        """progates likelihood associated with state

        Args:
            time_data (TimeData): time data for evolving the ODE 
            initial_state (State): initial state
            params (Optional[likeprop_typing.ArrayFloatT], optional): model parameters. 
            Defaults to None.

        Returns:
            Any: a trajectory or dictionary of likelihood and states
        """

    @abc.abstractmethod
    def compute_likelihood(self, current_state: State, *,
                           current_time: likeprop_typing.FloatT,
                           initial_time: likeprop_typing.FloatT,
                           params: Optional[likeprop_typing.ArrayFloatT] = None
                           ) -> Any:
        """get the likelihood of the current state

        Args:
            current_time (likeprop_typing.FloatT): current time of the state
            current_state (State): state at the current time
            initial_time (likeprop_typing.FloatT): initial time of the dynamical system
            params (Optional[likeprop_typing.ArrayFloatT], optional): model parameters. 
            Defaults to None.

        Returns:
            float type: likelihood of the current state
        """


class LagrangeCharpitPerronFrobenius:
    """Returns the Characteristics Dynamical System For solving the Liouville equation

    Attributes:
        ode_model (OdeModel): differential equation governing the state evolution

    Methods:
        characteristic_dynamics (TimeStateVector): augemented dynamics using the 
        method of characteristics
        __call__: returns DynamicalSystem object

    """

    # pylint: disable=invalid-name
    def __init__(self, ode_model: OdeModel):
        if ode_model.divF is None:
            raise ValueError("Divergence of F is needed For This Class")

        self.F = ode_model.F
        self.divF = ode_model.divF
        self.dim = ode_model.dim

    def characteristic_dynamics(self, time: likeprop_typing.FloatT,
                                state_with_likelihood: likeprop_typing.ArrayFloatT,
                                parameters: Optional[likeprop_typing.ArrayFloatT] = None
                                ) -> likeprop_typing.ArrayFloatT:
        """creates the ode system for jointly evolving likelihood with state using 
        method of characteristics

        Args:
            time (likeprop_typing.FloatT): time
            state_with_likelihood (likeprop_typing.ArrayFloatT): 
                array representation of state with likelihood
            parameters (Optional[likeprop_typing.ArrayFloatT], optional): parameter array. 
                Defaults to None.

        Raises:
            ValueError: array should at least have one state coordinate and an associated likelihood

        Returns:
            likeprop_typing.ArrayFloatT: time rate of change of state with likelihood
        """

        if state_with_likelihood.size < 2:
            raise ValueError(
                "Array should at least have a state and associated likelihood")
        state = state_with_likelihood[:-1]
        likelihood = state_with_likelihood[-1]

        state_dynamics = self.F(time, state, parameters)
        likelihood_dynamics = -1.0*likelihood * \
            self.divF(time, state, parameters)

        return np.concatenate((state_dynamics, np.array([likelihood_dynamics])))

    def __call__(self,
                 dynamical_system_class: Type[DynamicalSystem] = DynamicalSystemSolveIvp,
                 trajectory_info: StateTrajectoryFields = default_state_trajectory_fields,
                 **kwargs,
                 ) -> DynamicalSystem:

        # set likelihood to be part of the trajectory
        trajectory_info.is_likelihood_evolved = True

        # create the OdeModel
        charactersitics_ode = OdeModel(dim=self.dim + 1,
                                       F=self.characteristic_dynamics)

        # create the dynamical system
        return dynamical_system_class(ode_model=charactersitics_ode,
                                      trajectory_info=trajectory_info,
                                      **kwargs)


@dataclass
class LikelihoodViaCharacteristics:
    """Evolves likelihood of a state

    A concrete representation of the LikelihoodManager

    Attributes:
        initial_density (ProbabilityDensityFunction): initial pdf on state (and params)
        likelihood_dynamical_system (Type[DynamicalSystem]) : augmented dynamical system for 
        evolving state and likelihood
        state_dynamical_system (Type[DynamicalSystem]) : dynamical system defined by the model

    Methods:
        evolve(time_data, state, params) : propagates likelihood of an initial state
        compute_likelihood(current_time, initial_time, current_state, params) : computes the 
        likelihood of the current state
    """

    initial_density: ProbabilityDensityFunction
    likelihood_dynamical_system: DynamicalSystem
    state_dynamical_system: DynamicalSystem

    def evolve(self, initial_state: State, *,
               time_data: TimeData,
               params: Optional[likeprop_typing.ArrayFloatT] = None
               ) -> StateTrajectory:
        """propagates likelihood associated with the state from an initial state

        Args:
            time_data (TimeData): an ordered time stamp for evolving ODE 
            initial_state (State): initial state    
            params (Optional[likeprop_typing.ArrayFloatT], optional): parameter array. 
            Defaults to None.

        Returns:
            Trajectory: trajectory of state with associated likelihood
        """

        # get the initial likelihood of the state
        initial_likelihood = self.initial_density.relative_likelihood(
            initial_state)
        # associate likelihood with the state
        init_state = copy.deepcopy(initial_state)
        init_state.likelihood = initial_likelihood
        # evolve state and likelihood jointly using method of characteristics
        return self.likelihood_dynamical_system.evolve(time_data=time_data,
                                                       initial_state=init_state,
                                                       params=params)

    def compute_likelihood(self, current_state: State, *,
                           current_time: likeprop_typing.FloatT,
                           initial_time: likeprop_typing.FloatT,
                           params: Optional[likeprop_typing.ArrayFloatT] = None
                           ) -> likeprop_typing.FloatT:
        """get the likelihood of the current state

        Args:
            current_time (likeprop_typing.FloatT): current time of the state
            current_state (State): state at the current time
            initial_time (likeprop_typing.FloatT): initial time of the dynamical system
            params (Optional[likeprop_typing.ArrayFloatT], optional): model parameters. 
            Defaults to None.

        Raises:
            ValueError: current time has to be greater than initial time

        Returns:
            likeprop_typing.FloatT: likelihood of the current state
        """

        if current_time <= initial_time:
            raise ValueError("current time can't be less than initial time")

        if current_state.likelihood >= 0.0:
            raise TypeError("Likelihood of the state is already set")

        # first evolve the current state backward
        time_data_backward = TimeData(initial_time=current_time,
                                      final_time=initial_time,
                                      ordered_time_stamps=np.array([current_time, initial_time]))

        traj = self.state_dynamical_system.evolve(time_data=time_data_backward,
                                                  initial_state=current_state,
                                                  params=params)

        # get the initial condition
        match traj.representation_type.name:
            case 'DICT':
                initial_state = State.from_dictionary_representation(traj[-1])
            case 'ARRAY':
                initial_state = State.from_array(
                    traj[-1], time=initial_time, likelihood=False)
            case _:
                raise ValueError(
                    "Incorrect state representation set in Trajectory object")

        # evolve the initial state with likelihood
        time_data_forward = TimeData(initial_time=initial_time,
                                     final_time=current_time,
                                     ordered_time_stamps=np.array([initial_time, current_time]))

        traj = self.evolve(time_data=time_data_forward,
                           initial_state=initial_state,
                           params=params)

        if traj.representation_type.name == 'DICT':
            return float(traj[-1]['likelihood'])

        return traj[-1][-1]


class MonteCarlo:
    """Basic Monte Carlo For Computing probabilities and expectation
    """

    def __init__(self, dynamical_system: DynamicalSystem, *, num_averages=2):
        self.num_averages = num_averages
        self.dynamical_system = dynamical_system

    def __monte_carlo_probability(self, array_of_traj, set_count_function):
        """computes the probability of the event using ensembles of trajectories
        using Monte Carlo. 

        Args:
            array_of_traj (np.ndarray): (ensemble, time_stamps, dim_state)
            set_count_function (Callable): Function the returns the number of ensemble that are in 
            a given set object.

        Returns:
            np.ndarray(time_stamps,): probability of event for eact time in time_stamps
        """
        mc_sum = []
        num_samples = array_of_traj.shape[0]
        for i in range(array_of_traj.shape[1]):
            state_list = array_of_traj[:, i, :]
            mc_sum.append(set_count_function(state_list)/num_samples)
        return np.array(mc_sum)

    def __create_partial_functions(self, set_obj, time_data, params):
        # create a pipeline
        evolve_state = partial(self.dynamical_system.evolve,
                               time_data=time_data, params=params)
        evolve_state = partial(map, evolve_state)
        num_of_points_in_set = partial(count_in_set, set_obj=set_obj)
        points_in_set = partial(in_set, set_obj=set_obj)
        return (evolve_state, points_in_set, num_of_points_in_set)

    def estimate_probability_event(self, initial_samples: np.ndarray,
                                   event: SetInEuclideanSpace,
                                   *,
                                   time_data: TimeData,
                                   params=None):
        """estimate the probability that the state is in a given set

        Args:
            initial_samples (np.ndarray): samples from initial distribution
            event (SetInEuclideanSpace): a compact set in the state space
            time_data (TimeData): time data information
            params (_type_, optional): model parameters. Defaults to None.

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """

        if initial_samples.size == 0:
            raise ValueError("Empty samples")

        _, _, num_points_in_event = self.__create_partial_functions(set_obj=event,
                                                                    time_data=time_data,
                                                                    params=params)
        mc_probabilities = partial(self.__monte_carlo_probability,
                                   set_count_function=num_points_in_event)

        pipeline = SequentialPipeline()
        evolve_state_pipeline = get_ensemble_of_trajectories_from_state(time_data=time_data,
                                                                        dynamical_system=self.dynamical_system,
                                                                        params=params)

        pipeline.add(
            evolve_state_pipeline, 'evolves state from an ensemble of initial states')
        pipeline.add(mc_probabilities,
                     'compute probabilities using MC for each time')

        return pipeline(initial_samples)

    def estimate_expectation(self, initial_samples: np.ndarray,
                             real_valued_state_function,
                             event: SetInEuclideanSpace,
                             *,
                             time_data: TimeData,
                             params=None):
        r"""computes the expectation of function :math:`f(\mathbf{x})` on
        a given set.

        Args:
            initial_samples (np.ndarray): _description_
            real_valued_state_function (_type_): _description_
            event (SetInEuclideanSpace): _description_
            time_data (TimeData): _description_
            params (_type_, optional): _description_. Defaults to None.
        """


class PerronFrobeniusEstimates:
    """Uses likeprop's PF operator to compute probabilities and expectations.
    """

    def __init__(self, initial_time: float, likelihood_manager: LikelihoodManager):
        self.like_manager = likelihood_manager
        self.initial_time = initial_time

    # pylint: disable=invalid-name
    def __initialize(self, x):
        """at initial time, the likelihood of states is given by
        initial density specified in the likelihood manager

        Args:
            x (Array of State): array representing state or state obj

        Returns:
            float: the initial likelihood of x
        """
        return self.like_manager.initial_density.relative_likelihood(x)

    def __compute_prob(self, list_of_likelihoods, set_obj: SetInEuclideanSpace):
        """given a list of likelihoods associated with the state in a set
        returns the probabablity measure of the set

        Args:
            list_of_likelihoods (list): list of likelihoods of the state in set
            set_obj (SetInEuclideanSpace): a set in euclidean space

        Returns:
            float: Probability measure of the set
        """
        return np.mean(list_of_likelihoods)*set_obj.lebesgue_measure

    def get_pipeline(self, time: float, params) -> SequentialPipeline:
        """returns a pipeline to compute probabilities and expectations

        Args:
            time (float): current time
            set_obj (SetInEuclideanSpace): compact set over which integration needs
            to be done
            params (np.ndarray): model parameters

        Returns:
            SequentialPipeline: pipeline to do the computation
            [array -> state]->batch->list[likelihoods]
        """
        # create partial functions to freeze current time
        get_state_from_array = partial(StateVec.from_array, time=time)
        if time == self.initial_time:
            get_likelihood_from_state = self.__initialize
        else:
            get_likelihood_from_state = partial(self.like_manager.compute_likelihood,
                                                initial_time=self.initial_time,
                                                current_time=time, params=params)

        # create a sequential pipeline
        get_likelihood_from_array = SequentialPipeline()
        get_likelihood_from_array.add(
            get_state_from_array, "convert array to state object")
        get_likelihood_from_array.add(
            get_likelihood_from_state, "compute likelihood at current time from state")

        # creat a pipeline to process batch
        batch_likelihood = partial(map, get_likelihood_from_array)
        batch_pipeline = SequentialPipeline()
        batch_pipeline.add(
            batch_likelihood, "apply likelihood pipeline to batch")
        batch_pipeline.add(list, "make batch a list")

        return batch_pipeline

    def estimate_probability_event(self, initial_samples: np.ndarray,
                                   set_obj: SetInEuclideanSpace,
                                   *,
                                   time_data: TimeData,
                                   params=None):
        """computes the probability measure of a set in state space over
        time

        Args:
            initial_samples (np.ndarray): discretization of the state space
            set_obj (SetInEuclideanSpace): a compact set in state space
            time_data (TimeData): time data describing the dynamical system
            params (np.ndarray, optional): model parameters. Defaults to None.

        Returns:
            np.ndarray (time_stamps,): Probability measure of the set over time
        """
        prob_in_time = []
        for time in time_data.ordered_time_stamps:
            pipeline = self.get_pipeline(time=time, params=params)
            pipeline.add(partial(self.__compute_prob, set_obj=set_obj),
                         "integrate the likelihoods in the set")
            probability = pipeline(initial_samples)
            prob_in_time.append(probability)

        return np.array(prob_in_time)


def support_of_rect(boundary_rect_iter: Iterable, time_data: TimeData,
                    state_dynamical_system: DynamicalSystem,
                    params=None) -> Rect:
    r"""returns the maximum support of a rectangle :math:`A` at time :math:`t=t^{\ast}` over the 
    time interval :math:`[t_0,t^{\ast}]`

    Args:
        boundary_rect_iter (Iterable): iterable of sobol points over all the boundaries 
            of a rectangle 
        time_data (TimeData): backward time data from :math:`t^{\ast}` to :math:`t_0`
        state_dynamical_system (DynamicalSystem): dynamical system governing the state evolution
        params (np.ndarray, optional): model parameters. Defaults to None.

    Returns:
        Rect: :math:`d`dimensional rectangle
    """

    # get pipelines to evolve state backward defined on boundaries of a d-dim Rect

    # takes (batch_samples) -> (batch, time_steps, state)
    pipeline = get_ensemble_of_trajectories_from_state(time_data=time_data,
                                                       dynamical_system=state_dynamical_system,
                                                       params=params)

    # run through each boundary and apply pipeline
    min_val = []
    max_val = []

    # iterate over each boundary of the rectangle (batch, boundary_state)
    # evolve boundary points backward to initial time using pipeline to
    #   get (batch, time_steps, state)
    # store the minimum values over batch and time steps for each state coordinate
    for batch in boundary_rect_iter:

        boundary_traj = pipeline(batch)
        # store the minimum and maximum value of each coordinate over time
        min_boundary = [np.min(boundary_traj[:, :, i])
                        for i in range(boundary_traj.shape[-1])]
        max_boundary = [np.max(boundary_traj[:, :, i])
                        for i in range(boundary_traj.shape[-1])]
        # append for each boundary
        min_val.append(min_boundary)
        max_val.append(max_boundary)

    # convert list to array
    min_val_arr = np.array(min_val)
    max_val_arr = np.array(max_val)

    # create a rect from min and max values of each state coordinate over time
    left_ends = [np.min(min_val_arr[:, i])
                 for i in range(min_val_arr.shape[-1])]
    right_ends = [np.max(max_val_arr[:, i])
                  for i in range(max_val_arr.shape[-1])]

    return Rect.from_endpoints(left_ends=left_ends, right_ends=right_ends)


@dataclass
class AdjointSensitivityInputs:
    """Meta Data For Computing Adjoint Sensitivity
    """

    density_estimator: dict[str, Callable]
    state_dynamical_system: DynamicalSystem
    partial_dynamics_parameters: dict[str, Callable]


class AdjointSolution:
    """class for solving the adjoint sensitivity problem
    """

    def __init__(self, time_data: TimeData,
                 adjoint_initial_condition: Callable,
                 meta_data: AdjointSensitivityInputs,
                 initial_s_i: Optional[Callable] = None) -> None:

        self.current_time = time_data.initial_time
        self.initial_time = time_data.final_time
        self.time_data = time_data
        self.adjoint_initial_condition = adjoint_initial_condition
        self.initial_s_i = initial_s_i
        self.state_dynamical_system = meta_data.state_dynamical_system
        self.partial_f_pi = meta_data.partial_dynamics_parameters
        self.density_estimator = meta_data.density_estimator

    def get_space_time_sobol_points(self, set_obj: Rect,
                                    sobol_options_boundary: SobolSampleOptions,
                                    sobol_options_support: SobolSampleOptions,
                                    params=None) -> tuple[SobolRect, SobolRect, Rect, Rect]:
        r"""returns sobol samples for the space x time rectangle and space rectangle

        Args:
            set_obj (Rect): the rectangle (A) over which probability is computed
            sobol_options_boundary (SobolSampleOptions): sobol samples over boundary of A
            sobol_options_support (SobolSampleOptions): sobol samples over :math:`X\cap` support of adoint
            params (np.ndarray, optional): model parameters. Defaults to None.

        Returns:
            tuple: sobol iterable, rect over support, rect over space time
        """
        # first get rect B on support of the adjoint solution for set_obj,
        boundary_iterator = get_sobol_rect_boundary(
            rect=set_obj, options=sobol_options_boundary)
        support_rect = support_of_rect(boundary_rect_iter=boundary_iterator,
                                       time_data=self.time_data,
                                       state_dynamical_system=self.state_dynamical_system,
                                       params=params)

        # Create sobol points for B x [initial_time,current_time]
        left_ends = list(support_rect.left_ends)
        left_ends.append(self.initial_time)
        right_ends = list(support_rect.right_ends)
        right_ends.append(self.current_time)
        space_time_rect = Rect.from_endpoints(
            left_ends=left_ends, right_ends=right_ends)

        sobol_points_space_time = SobolRect(
            set_obj=space_time_rect, options=sobol_options_support)
        sobol_points_support = SobolRect(
            set_obj=support_rect, options=sobol_options_support)

        # return both support points and space time points
        return (sobol_points_support, sobol_points_space_time, support_rect, space_time_rect)

    def adjoint_v(self, space_time_point, *,
                  params=None) -> likeprop_typing.FloatT:
        """solves the adjoint solution given an adjoint initial condition

        Args:
            space_time_point (likeprop_typing.ArrayFloatT): array of d + 1 dimensions (x,t)

        Returns:
            likeprop_typing.FloatT: adjoint v(x,t)
        """

        # get space point (x) and time (t) from space_time_point
        time = float(space_time_point[-1:])
        state = space_time_point[:-1]
        if time == self.current_time:
            return self.adjoint_initial_condition(state)
        # evolve (x,t) to current time (x(t_ast), t_ast))
        time_data = TimeData(initial_time=time, final_time=self.current_time,
                             ordered_time_stamps=np.array([time, self.current_time]))
        traj = self.state_dynamical_system.evolve(initial_state=state,
                                                  time_data=time_data,
                                                  params=params)
        # v(x,t) = adjoint_initial_condition(x(t_ast), t_ast)
        return self.adjoint_initial_condition(traj[-1])

    def divergence_rho_f_pi(self, space_time_point, *,
                            params=None) -> likeprop_typing.FloatT:
        r"""returns :math:`\text{div}(\varrho F_{p_i})` at a particular point in time

        Args:
            space_time_point (array): array of  (x,t) point in d+1 dimensions
            params (array, Optional): model parameters. Defaults to None

        Returns:
            likeprop_typing.FloatT: :math:`\text{div}(\varrho F_{p_i})(x,t) 
        """
        # partial f w.r.t p_i at (x,t)
        f_pi = self.partial_f_pi['f_pi'](space_time_point, params)
        # divergence of partial f w.r.t p_i at (x,t)
        div_f_pi = self.partial_f_pi['div_f_pi'](space_time_point, params)
        # gradient of rho at (x,t)
        grad_rho = self.density_estimator['grad_rho'](space_time_point)
        # rho at (x,t)
        rho = self.density_estimator['rho'](space_time_point)
        # div(rho f_{p_i}) = dot(grad(rho),  f_{p_i}) + rho div(f_{p_i})
        return np.dot(grad_rho, f_pi) + rho*div_f_pi

    def adoint_v_rho_f_pi(self, space_time_point, *,
                          params=None) -> likeprop_typing.FloatT:
        r"""returns :math:v`\text{div}(\varrho F_{p_i})` at a particular point in time

        Args:
            space_time_point (array): (x,t) point
            params (array, Optional): model parameters. Defaults to None

        Returns:
            likeprop_typing.FloatT: :math:v(x,t)`\text{div}(\varrho F_{p_i})(x,t) 
        """
        adjoint_v = self.adjoint_v(space_time_point, params=params)
        rho_f_pi = self.divergence_rho_f_pi(space_time_point, params=params)
        return adjoint_v*rho_f_pi

    def adjoint_v_s_i(self, space_point, *, params=None) -> likeprop_typing.FloatT:
        """returns adjoint v(x, t=0)*partial initial density partial p_i(x,t=0) for a given x

        Args:
            space_point (_type_): d dimensional spatial point
            params (np.ndarrat, optional): model parameters. Defaults to None.

        Returns:
            likeprop_typing.FloatT: v(x,t=0)s_i(x,t=0)
        """
        # evolve (x,initial_time) to current time (x(t_ast), t_ast))
        time_data = TimeData(initial_time=self.initial_time,
                             final_time=self.current_time,
                             ordered_time_stamps=np.array([self.initial_time,
                                                           self.current_time]))
        traj = self.state_dynamical_system.evolve(initial_state=space_point,
                                                  time_data=time_data,
                                                  params=params)
        # v(x,t) = adjoint_initial_condition(x(t_ast), t_ast)
        v_0 = self.adjoint_initial_condition(traj[-1])
        if self.initial_s_i is not None:
            s_i_0 = self.initial_s_i(space_point)
            return v_0*s_i_0
        return 0.0

    def _space_time_pipeline(self, params=None):
        # create pipeline
        # -> sobol points (batch, d+1) -> adjoint_v*rho_f_pi (batch,) -> sum (scalar)
        space_time_function = partial(self.adoint_v_rho_f_pi, params=params)
        space_time_function = partial(map, space_time_function)
        space_time_pipeline = SequentialPipeline()
        space_time_pipeline.add(space_time_function,
                                description="adjoint times rhoF_p_i")
        space_time_pipeline.add(list, "convert to list")
        space_time_pipeline.add(sum, "sum the function at space time points")

        # if the intial sensitivity is 0 i.e. initial pdf doesn't depend on p_i
        if self.initial_s_i is None:
            return (None, space_time_pipeline)

        # else
        # -> sobol points (batch, d) -> adjoint_v(t=0)*s_pi(t=0) (batch,) -> sum (scalar)
        space_function = partial(self.adjoint_v_s_i, params=params)
        space_function = partial(map, space_function)
        space_pipeline = SequentialPipeline()
        space_pipeline.add(space_function, "adjoint time s_i at t_0")
        space_pipeline.add(list, "convert to list")
        space_pipeline.add(sum, "sum the function at all spatial points")

        return (space_pipeline, space_time_pipeline)
    
    def _get_batch_integral(self, sobol_points, rect, pipeline):
        # space_time_integral
        total_points = 0
        total_sum = 0
        for batch in sobol_points:
            total_sum += pipeline(batch)
            total_points += batch.shape[0]

        integral = (total_sum/total_points)*rect.lebesgue_measure
        return integral


    def get_sensitivity_to_probability(self, set_obj: Rect,
                                       sobol_options_boundary: SobolSampleOptions,
                                       sobol_options_support: SobolSampleOptions,
                                       params=None):
        r"""compute adjoint sensitivity of probability Q w.r.t parameter p_i

        Args:
            set_obj (Rect): Rectangle (A) over which probablity is desired
            sobol_options_boundary (SobolSampleOptions): sobol options for boundary of A
            sobol_options_support (SobolSampleOptions): sobol options for space time points
            params (np.ndarray, optional): model parameters. Defaults to None.

        Returns:
            likeprop.typing.FloatT: :math:`\frac{dQ}{dp_i}`
        """

        # get the sobol points for the space time domain and the space domain
        sobol_x, sobol_xt, rect_x, rect_xt = self.get_space_time_sobol_points(set_obj=set_obj,
                                                                              sobol_options_boundary=sobol_options_boundary,
                                                                              sobol_options_support=sobol_options_support,
                                                                              params=params)
        # get pipelines to process sobol points
        pipeline_x, pipeline_xt = self._space_time_pipeline(params=params)
        space_time_integral = -1.0*self._get_batch_integral(sobol_xt, rect_xt, pipeline_xt)

        
        if pipeline_x is None:
            return space_time_integral

        space_integral = self._get_batch_integral(sobol_x, rect_x, pipeline_x)
        return space_time_integral + space_integral



class MonteCarloAdjoint:
    def __init__(self, time_data: TimeData,
                 adjoint_initial_conditions: list[Callable[[Any], likeprop_typing.FloatT]],
                 meta_data: AdjointSensitivityInputs,
                 dynamics:dict[str, Callable],
                 initial_pdf,
                 initial_s_i: Optional[Callable] = None) -> None:

        self.time_data = time_data
        self.adjoint_initial_condition = adjoint_initial_conditions
        self.initial_s_i = initial_s_i
        self.state_dynamical_system = meta_data.state_dynamical_system
        self.partial_f_pi = meta_data.partial_dynamics_parameters
        self.density_estimator = meta_data.density_estimator
        self.dynamics  = dynamics
        self.initial_pdf = initial_pdf

    def _initial_error(self, space_point):
        return self.initial_pdf(space_point) - self.density_estimator['initial_rho'](space_point)

    
    def adjoint_v(self, space_time_point, *,
                  params=None) -> list[likeprop_typing.FloatT]:
        """solves the adjoint solution given an adjoint initial condition

        Args:
            space_time_point (likeprop_typing.ArrayFloatT): array of d + 1 dimensions (x,t)

        Returns:
            likeprop_typing.FloatT: adjoint v(x,t)
        """

        # get space point (x) and time (t) from space_time_point
        time = float(space_time_point[-1:])
        state = space_time_point[:-1]

        current_time = self.time_data.initial_time
        #initial_time = self.time_data.final_time

        if time == current_time:
            return [qoi(state) for qoi in self.adjoint_initial_condition]
        # evolve (x,t) to current time (x(t_ast), t_ast))
        time_data = TimeData(initial_time=time, final_time=current_time,
                             ordered_time_stamps=np.array([time, current_time]))
        traj = self.state_dynamical_system.evolve(initial_state=state,
                                                  time_data=time_data,
                                                  params=params)
        # v(x,t) = adjoint_initial_condition(x(t_ast), t_ast)
        return [qoi(traj[-1]) for qoi in self.adjoint_initial_condition]
    
    def divergence_rho_f_pi(self, space_time_point) -> likeprop_typing.FloatT:

        r"""returns :math:`\text{div}(\varrho F_{p_i})` at a particular point in time

        Args:
            space_time_point (array): array of  (x,t) point in d+1 dimensions
            params (array, Optional): model parameters. Defaults to None

        Returns:
            likeprop_typing.FloatT: :math:`\text{div}(\varrho F_{p_i})(x,t) 
        """
        # partial f w.r.t p_i at (x,t)
        f_pi = self.partial_f_pi['f_pi'](space_time_point)
        # divergence of partial f w.r.t p_i at (x,t)
        div_f_pi = self.partial_f_pi['div_f_pi'](space_time_point)
        # gradient of rho at (x,t)
        grad_rho = self.density_estimator['grad_rho'](space_time_point)
        # rho at (x,t)
        rho = self.density_estimator['rho'](space_time_point)
        # div(rho f_{p_i}) = dot(grad(rho),  f_{p_i}) + rho div(f_{p_i})
        return np.dot(grad_rho, f_pi) + rho*div_f_pi
    
    def divergence_rho_f(self, space_time_point) -> likeprop_typing.FloatT:
        r"""returns :math:`\text{div}(\varrho F)` at a particular point in time

        Args:
            space_time_point (array): array of  (x,t) point in d+1 dimensions
            params (array, Optional): model parameters. Defaults to None

        Returns:
            likeprop_typing.FloatT: :math:`\text{div}(\varrho F_{p_i})(x,t) 
        """
        # partial f w.r.t p_i at (x,t)
        f = self.dynamics['f'](space_time_point)
        # divergence of partial f w.r.t p_i at (x,t)
        div_f = self.dynamics['div_f'](space_time_point)
        # gradient of rho at (x,t)
        grad_rho = self.density_estimator['grad_rho'](space_time_point)
        # rho at (x,t)
        rho = self.density_estimator['rho'](space_time_point)
        # div(rho f_{p_i}) = dot(grad(rho),  f_{p_i}) + rho div(f_{p_i})
        return np.dot(grad_rho, f) + rho*div_f
    
    def adjoint_v_rho_f_pi(self, space_time_point, *,
                          params=None) -> list[likeprop_typing.FloatT]:
        r"""returns :math:v`\text{div}(\varrho F_{p_i})` at a particular point in time

        Args:
            space_time_point (array): (x,t) point
            params (array, Optional): model parameters. Defaults to None

        Returns:
            likeprop_typing.FloatT: :math:v(x,t)`\text{div}(\varrho F_{p_i})(x,t)` 
        """
        adjoint_v = self.adjoint_v(space_time_point, params=params)
        rho_f_pi = self.divergence_rho_f_pi(space_time_point)
        return [v*rho_f_pi for v in adjoint_v]
    
    def adjoint_v_residual(self, space_time_point, *,
                          params=None) -> list[likeprop_typing.FloatT]:
        r"""returns :math:v`\text{div}(\varrho F_{p_i})` at a particular point in time

        Args:
            space_time_point (array): (x,t) point
            params (array, Optional): model parameters. Defaults to None

        Returns:
            likeprop_typing.FloatT: :math:v(x,t)`L[\widehat{\varrho}]` 
        """
        adjoint_v = self.adjoint_v(space_time_point, params=params)
        drho_dt = self.density_estimator['time_derivative'](space_time_point)
        div_rho_f = self.divergence_rho_f(space_time_point)
        residual = drho_dt + div_rho_f
        return [v*(residual) for v in adjoint_v]


    def adjoint_v_s_i(self, space_point, *, params=None) -> list[likeprop_typing.FloatT]:
        """returns adjoint v(x, t=0)*partial initial density partial p_i(x,t=0) for a given x

        Args:
            space_point (_type_): d dimensional spatial point
            params (np.ndarrat, optional): model parameters. Defaults to None.

        Returns:
            likeprop_typing.FloatT: v(x,t=0)s_i(x,t=0)
        """
        # evolve (x,initial_time) to current time (x(t_ast), t_ast))
        current_time = self.time_data.initial_time
        initial_time = self.time_data.final_time
        time_data = TimeData(initial_time=initial_time,
                             final_time=current_time,
                             ordered_time_stamps=np.array([initial_time,
                                                           current_time]))
        traj = self.state_dynamical_system.evolve(initial_state=space_point,
                                                  time_data=time_data,
                                                  params=params)
        # v(x,t) = adjoint_initial_condition(x(t_ast), t_ast)
        list_v_0 = [qoi(traj[-1]) for qoi in self.adjoint_initial_condition]
        if self.initial_s_i is not None:
            s_i_0 = self.initial_s_i(space_point)
            return [v_0*s_i_0 for v_0 in list_v_0]
        return [0.0]*len(list_v_0)
    
    def _space_time_pipeline(self, integrand, sample_function):
        
        # create the pipeline
        batch_pipeline = SequentialPipeline()
        # make all functions take one argument

        map_function = partial(map, integrand)
        mean_function = partial(np.mean, axis=0)

        # create pipeline
        # samples_at_time_t -> integrand applied to each sample -> list -> array (N x QoIs) -> Mean (QoIs,)
        batch_pipeline.add(sample_function, "space time samples")
        batch_pipeline.add(map_function, "apply func to space time points")
        batch_pipeline.add(list, "make batch a list")
        batch_pipeline.add(np.array, "make the list a (Number of samples) X (Number of qois)")
        batch_pipeline.add(mean_function,"get mean for each qoi")
        return batch_pipeline


    def _space_pipeline(self, integrand):
        # create the pipeline
        batch_pipeline = SequentialPipeline()
        # make all functions take one argument

        map_function = partial(map, integrand)
        mean_function = partial(np.mean, axis=0)
        # create pipeline
        # integrand applied to each sample -> list -> array (N x QoIs) -> Mean (QoIs,)
        batch_pipeline.add(map_function, "apply func to space points")
        batch_pipeline.add(list, "make batch a list")
        batch_pipeline.add(np.array, "make the list a (Number of samples) X (Number of qois)")
        batch_pipeline.add(mean_function,"get mean for each qoi")
        return batch_pipeline
    
    def _integrate_in_time(self, time_varying_function, num_of_time_pts, eps):

        end_time = self.time_data.initial_time
        start_time = self.time_data.final_time
        time_discretization = np.linspace(start=start_time+eps, stop=end_time, num=num_of_time_pts, endpoint=False)
        func_at_time = np.array([time_varying_function(t) for t in time_discretization])
        delta_t = end_time - start_time
        integral = -1.0*delta_t*np.mean(func_at_time, axis=0)
        return integral
    
    def sensitivity_G_space_time_function(self, space_time_point):
        # first get rho(space_time_point)
        rho_val = self.density_estimator['rho'](space_time_point)
        # then get the v*div(rho F_p_i)
        list_v_div_term = self.adjoint_v_rho_f_pi(space_time_point=space_time_point)
        # w*div_term = v_div_term/rho_val
        return [v_div_term/rho_val for v_div_term in list_v_div_term]
    
    def error_H_space_time_function(self, space_time_point):
        # first get rho(space_time_point)
        rho_val = self.density_estimator['rho'](space_time_point)
        # then get the v*div(rho F_p_i)
        list_v_div_term = self.adjoint_v_residual(space_time_point=space_time_point)
        # w*div_term = v_div_term/rho_val
        return [v_div_term/rho_val for v_div_term in list_v_div_term]


    def sensitivity_G_space_function(self, space_point):
        # first get rho(space_point)
        rho_val = self.density_estimator['initial_rho'](space_point)
        # then get the v*s_i
        list_v_s_term = self.adjoint_v_s_i(space_point)
        # w*s_i_term = vs_i/rho_val
        return [v_s_term/rho_val for v_s_term in list_v_s_term]

    def error_H_space_function(self, space_point, params=None):
        # first get rho(space_point)
        rho_val = self.density_estimator['initial_rho'](space_point)
        # get the error term
        initial_err = self._initial_error(space_point)
        # evolve (x,t_0) to current time (x(t_ast), t_ast))
        current_time = self.time_data.initial_time
        initial_time = self.time_data.final_time

        time_data = TimeData(initial_time=initial_time,
                             final_time=current_time,
                             ordered_time_stamps=np.array([initial_time,
                                                           current_time]))
        traj = self.state_dynamical_system.evolve(initial_state=space_point,
                                                  time_data=time_data,
                                                  params=params)
        # v(x,t_0) = adjoint_initial_condition(x(t_ast), t_ast)
        list_v0 = [qoi(traj[-1]) for qoi in self.adjoint_initial_condition]

        # compute error*v_0/rho
        return [initial_err*v_0/rho_val for v_0 in list_v0]

    def compute_sensitivity(self, sample_points=10000, ensembles=10, time_pts=10,eps=1e-3):

        sample_at_t = partial(self.density_estimator['sample'], num=sample_points)
        g_func_time = self._space_time_pipeline(integrand=self.sensitivity_G_space_time_function, sample_function=sample_at_t)
   
        mc_estimates = []
        for _ in range(ensembles):
            esitmate = self._integrate_in_time(g_func_time,time_pts,eps)
            
            if self.initial_s_i is not None:
                sample_at_t0 = sample_at_t(0.0)[:,0:-1]
                g_func = self._space_pipeline(integrand=self.sensitivity_G_space_function)
                esitmate += g_func(sample_at_t0)

            mc_estimates.append(esitmate)
        
        return np.array(mc_estimates)


    def compute_error(self, sample_points=10000, ensembles=10,time_pts=10,eps=1e-3):
        sample_at_t = partial(self.density_estimator['sample'], num=sample_points)
        h_func_time = self._space_time_pipeline(integrand=self.error_H_space_time_function, 
                                                sample_function=sample_at_t)
   
        mc_estimates = []
        for _ in range(ensembles):
            esitmate = self._integrate_in_time(h_func_time,time_pts,eps)
            sample_at_t0 = sample_at_t(0.0)[:,0:-1]
            h_func = self._space_pipeline(integrand=self.error_H_space_function)
            esitmate += h_func(sample_at_t0)

            mc_estimates.append(esitmate)
        
        return np.array(mc_estimates)
    


