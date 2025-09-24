'''This module describes classes and functions for creating an ode model for the dynamics'''

from typing import NamedTuple, Optional, Protocol, Union
from dataclasses import dataclass, asdict
import abc
import numpy as np
from scipy.integrate import solve_ivp  # type:ignore
from likeprop.dynamics.state import (State, StateTrajectory,
                                     StateTrajectoryFromArray,
                                     StateTrajectoryFields,
                                     default_state_trajectory_fields)
from likeprop.utils.typing import (TimeStateFunctional, TimeStateVector,
                                   ArrayFloatT, FloatT)
from likeprop.dynamics.utilities import TimeData

StateOrArrayType = Union[State, ArrayFloatT]


class OdeModel(NamedTuple):
    r"""An Ordinary Differential Equation (ODE)

    Given a state space in :math:`\mathbb{R}^d`, an ODE is defined by the evolution of a state
    :math:`\mathbf{x}(t) = (x_1(t), \cdots, x_d(t))` according to the followin equation
    .. math:
        \frac{d\mathbf{x}}{dt} = F(t, \mathbf{x},\text{ parameters}).

    where :math:`F = (f_1,\cdots, f_d)` and each :math:`f_i` is a real valued function defined on 
    the state space.

    An OdeModel is described by defining this function F. However, we also need each real valued 
    function :math:`\frac{\partial f_i}{\partial x_i}` in order to describe the evolution of 
    likelihoods associated with a state vector :math:`\mathbf{x}(t)`. This is 
    given by trace_jacobian_F (divF) which is defined as follows:

    .. math:
        \text{div}F(t, \mathbf{x}, \text{ parameters} = 
        \sum\limits_{i=1}^d\frac{\partial f_i}{\partial x_i} 

    Args:
        NamedTuple (namedtuple): a named tuple based class

    Attributes:
        dim(int): The dimension of the state space.
        F: (vector valued function) is the dynamics
        divF: (real valued function) is the trace of the jacobian of F a.k.a divergence of F

    """

    dim: int
    # dot(x) = F(t, x, parameters)
    F: TimeStateVector
    # \sum\limits_{i = 1}^{\text{dim_state}}dF_i/dx_i(t, x, parameters)
    divF: Optional[TimeStateFunctional] = None


@dataclass
class DynamicalSystem(Protocol):
    """An Abstract Dynamical System That Evolves States

    Args:
        Protocol (_type_): Abstract

    Attributes:
        ode_model (OdeModel): dim, dynamics F, optional[div F] of the ode
        trajectory_info (StateTrajectoryFields): all the meta data needed to initialize a state 
        trajectory 

    Methods:
        evolve(initial state, time_data=time_data,
                params=params) -> Trajectory
        evolves the state and returns a state trajectory object
    """

    ode_model: OdeModel
    trajectory_info: StateTrajectoryFields

    @abc.abstractmethod
    def evolve(self, initial_state: StateOrArrayType,
               *, time_data: TimeData,
               params: Optional[ArrayFloatT] = None) -> StateTrajectory:
        """evolves a initial state according to time stamp

        Args:
            initial_state (State): initial state
            time_data (TimeData): an ordered time data object
            params (ArrayFloatT): optional model parameters: defaults to None

        Returns:
            trajectory (StateTrajectoryType): stores the trajectory of the state
        """

@dataclass
class SolverOptionsIvp:
    """meta data for the most basic solve_ivp
    parameters
    """
    method: str = 'RK45'
    rtol: FloatT = 1e-12
    atol: FloatT = 1e-15
    args: tuple[Optional[ArrayFloatT]] = (None,)


# use the default fields to initialize a DynamicalSystemSolveIvp
default_solver_options = SolverOptionsIvp()


@dataclass
class DynamicalSystemSolveIvp(DynamicalSystem):
    """Concrete Representation of DynamicalSystem with a 
    wrapper around scipy's solve_ivp
    """

    def __init__(self,
                 ode_model: OdeModel,
                 trajectory_info: StateTrajectoryFields = default_state_trajectory_fields,
                 *,
                 solver_options_ivp: SolverOptionsIvp = default_solver_options,
                 **kwargs):

        self.ode_model = ode_model
        self.trajectory_info = trajectory_info

        # additional ivp related stuff
        self.solver_options = solver_options_ivp
        self.additional_ivp_options = kwargs

    def evolve(self, initial_state: StateOrArrayType, *,
               time_data: TimeData,
               params: Optional[ArrayFloatT] = None) -> StateTrajectoryFromArray:

        # modify trajectory options to include params
        self.trajectory_info.params = params
        # create the trajectory object with the trajectory options
        trajectory = StateTrajectoryFromArray(**asdict(self.trajectory_info))

        # if likelihood is part of the state then the np.array automatically
        # takes care of it.
        init_state = np.array(initial_state)
        dim_state = init_state.size

        # check to see if dimension of initial state and the dimension on which the
        # dynamical system acts are the same.
        if dim_state != self.ode_model.dim:
            raise ValueError(
                "The dimensions of the initial state and the dynamical system don't match")

        # set the parameters in the solver options
        self.solver_options.args = (params,)

        # merge solver options with any additional arguments needed
        solver_options = asdict(
            self.solver_options) | self.additional_ivp_options

        # generate the output dictionary from solve ivp
        output = solve_ivp(fun=self.ode_model.F,
                           y0=init_state,
                           t_span=[time_data.initial_time,
                                   time_data.final_time],
                           t_eval=time_data.ordered_time_stamps,
                           **solver_options)

        # check if we got the ODE to converge
        if not output['success']:
            raise ValueError("The dynamical system didn't converge")

        # store time and state in trajectory
        for _, (time, state_y) in enumerate(zip(time_data.ordered_time_stamps, output.y.T)):
            trajectory.store(state_y, time=time)

        return trajectory
