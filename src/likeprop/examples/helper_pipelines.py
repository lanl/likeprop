'''This module defines helper classes and functions to make ensemble runs'''

from typing import Optional
from functools import partial
import numpy as np
from likeprop.utils.typing import FloatT, ArrayFloatT
from likeprop.dynamics.ode_model import DynamicalSystem
from likeprop.dynamics.utilities import TimeData
from likeprop.density.distributions import ProbabilityDensityFunction
from likeprop.dynamics.pipelines import (SequentialPipeline,
                                         augment_likelihood_to_state,
                                         get_final_state_from_trajectory)

# pylint: disable=invalid-name


class EvolveStateFixedParam:
    """A helper pipeline for evolving state from initial condition to final
    """

    def __init__(self, initial_time: FloatT,
                 final_time: FloatT,
                 dynamical_system: DynamicalSystem,
                 initial_density: Optional[ProbabilityDensityFunction] = None,
                 params: Optional[ArrayFloatT] = None):

        self.dynamical_system = dynamical_system
        self.params = params
        # create a TimeData object
        self.time_data = TimeData(initial_time=initial_time,
                                  final_time=final_time,
                                  ordered_time_stamps=np.array([initial_time, final_time]))

        self.evolve = partial(self.dynamical_system.evolve, time_data=self.time_data,
                              params=self.params)
        self.initial_density = initial_density

    def inital_to_final(self) -> SequentialPipeline:
        """initial state : array -> trajectory -> final state: array
        """
        pipeline = SequentialPipeline()
        if self.initial_density is not None:
            likelihood_with_state = partial(augment_likelihood_to_state,
                                            pdf=self.initial_density)
            pipeline.add(likelihood_with_state, "augment likelihood to state")
        pipeline.add(self.evolve, "evolve state")
        pipeline.add(get_final_state_from_trajectory, "retrieve final state")
        return pipeline
