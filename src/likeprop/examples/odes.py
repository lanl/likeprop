'''This Module Defines Some Illustrative ODEs'''
import numpy as np
from likeprop.utils.typing import FloatT, ArrayFloatT

# **** A 2d ODE that spirals to the origin


# pylint: disable=unused-argument
# pylint: disable=invalid-name
class Spiral2d:
    """a 2d linear ODE
    """

    def __init__(self, coeff=np.array([-1.0])):
        self.dim = 2
        self.coeff = coeff
        self._matrix = np.zeros((self.dim, self.dim), dtype=np.float64)

    def _linear_transformation(self, time, parameters=None):
        coeff = self.coeff[0]
        if parameters is not None:
            coeff = parameters[0]

        a = np.exp(coeff*time)*np.cos(time)
        b = np.exp(coeff*time)*np.sin(time)

        self._matrix[0, 0] = a
        self._matrix[0, 1] = -b
        self._matrix[1, 0] = b
        self._matrix[1, 1] = a

    # dot(s) = f(t, s, parameters)
    def dynamics(self, time, state, parameters=None) -> ArrayFloatT:
        r"""a simple 2d dynamical system given by
        ..math::
         \text{ds} = \begin{bmatrix}
                       coeff[0] & -1  \\
                       1 & coeff[0] \\
                     \end{bmatrix}s


        Args:
            time (_type_): _description_
            state (_type_): _description_

        Returns:
            np.ndarray : time derivative of the state
        """
        coeff = self.coeff
        if parameters:
            coeff = parameters
        xdot_1 = coeff[0]*state[0] - state[1]
        xdot_2 = state[0] + coeff[0]*state[1]
        return np.array([xdot_1, xdot_2], dtype=np.float64)

    def sum_partials(self, time, state, parameters=None) -> FloatT:
        """returns sum of dF/ds[i]

        Args:
            time (float): time
            state (lds.StateVec): state vector
            parameters (Optional[np.ndarray[Any, np.dtype[np.float64]]]): optional parameters

        Returns:
            float: the sum of the partials dF/ds[i]
        """
        coeff = self.coeff[0]
        if parameters:
            coeff = parameters[0]
        return 2.0*coeff

    def closed_form_solution(self, time,
                             initial_condition: ArrayFloatT,
                             parameters=None):
        """Returns the closed form solution of the ODE at time t

        Args:
            time (FloatT): time
            initial_condition (ArrayFloatT): intial state
            parameters (ArrayFloatT, optional): model parameters. Defaults to None.

        Returns:
            ArrayFloatT: solution at time t
        """
        self._linear_transformation(time=time, parameters=parameters)
        return np.matmul(self._matrix, initial_condition)


class DoubleAttractor:
    '''
    A non-linear ODE with two attractors
    '''
    def __init__(self, coeff=np.array([2.0])):
        self.dim = 2
        self.coeff = coeff

    # dot(s) = f(t, s, parameters)

    def dynamics(self, time, state, parameters=None) -> ArrayFloatT:
        r"""a simple 2d dynamical system given by
        ..math::
            \dot{x_1} = \theta x_1 - x_1x_2 \\
            \dot{x_2} = \theta x_1^2 - x_2


        Args:
            time (_type_): _description_
            state (_type_): _description_

        Returns:
            np.ndarray : time derivative of the state
        """
        coeff = self.coeff
        if parameters:
            coeff = parameters
        xdot_1 = coeff[0]*state[0] - state[0]*state[1]
        xdot_2 = coeff[0]*(np.power(state[0], 2)) - state[1]

        return np.array([xdot_1, xdot_2], dtype=np.float64)

    def sum_partials(self, time, state, parameters=None) -> FloatT:
        """returns sum of dF/dx[i]

        Args:
            time (float): time
            state (lds.StateVec): state vector
            parameters (Optional[np.ndarray[Any, np.dtype[np.float64]]]): optional parameters

        Returns:
            float: the sum of the partials dF/dx[i]
        """
        coeff = self.coeff[0]
        if parameters:
            coeff = parameters[0]
        dF_dx1 = coeff - state[1]
        dF_dx2 = -1.0
        return dF_dx1 + dF_dx2
