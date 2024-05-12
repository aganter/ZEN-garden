"""""
Class defining Modeling to Generate Alternatives functionalities:
- Capability to generate random directions samples from  a normal distribution
- Capability to generate weight for the MGA objective functions based on the aggregated variables and the random
    directions
- Capability to generate a set of alternatives objective function based on the weights generated
"""

import random
import numpy as np
import xarray as xr

from zen_garden.utils import lp_sum
from zen_garden.model.optimization_setup import OptimizationSetup


class ModelingToGenerateAlternatives:
    """
    Class defining Modeling to Generate Alternatives functionalities
    """

    label = "Modeling_To_Generate_Alternatives"
    location_type = None

    def __init__(
        self,
        n_dimensions: int,
        n_objectives: int,
        optized_setup: OptimizationSetup,
    ):
        """
        Init generic element for the MGA

        :param mga_iteration: The iteration of the MGA
        :param optimization_setup: The OptimizationSetup the element is part of
        :param n_dimensions: Number of dimensions N_d of the aggregated decision variables to consider for the MGA
        :param n_objectives: Number of objectives functions N_k to consider for the MGA method
        """

        self.n_dimensions = n_dimensions
        self.n_objectives = n_objectives
        self.optimization_setup = optized_setup

        # DocString to access the MGA class
        self.__doc__ = "--- Modeling to Generate Alternatives accessed to generate near-optimal solutions ---"
        print(self.__doc__)

    def generate_random_directions(self, seed: int = 0):
        """
        Generate random directions samples from a normal distribution with mean 0 and standard deviation 1 for the MGA
        objective functions generation.

        :param seed: Seed for the random number generator

        :return: Random directions vector direction_search_vector
        """

        random.seed(seed)
        directions = [random.gauss(0, 1) for _ in range(self.n_dimensions)]
        direction_search_vector = np.array(directions).reshape(1, self.n_dimensions)

        return direction_search_vector

    def aggregate_variables(self, variables: xr.DataArray):
        """
        Aggregate the variables to consider for the MGA objective functions generation.

        :param variables: Variables to aggregate

        :return: Aggregated variables aggregated_variables
        """

        aggregated_variables = lp_sum(variables)

        return aggregated_variables

    def generate_characteristic_scales(self, aggregated_variables: xr.DataArray):
        """
        Generate characteristic scales for the aggregated variables to normalize the aggregate variables by dividing
        them by the values of the aggregated variables in the optimal solution, when available. When these are zero, the
        characteristic scales are estimated to roughly match the expected magnitude of the variables in the near-optimal
        space.

        :param aggregated_variables: Aggregated variables to consider for the MGA objective functions generation

        :return: Characteristic scales vector characteristic_scales
        """
        # read optimal solution  variables values

        optimal_solution = lp_sum(aggregated_variables)
        characteristic_scales = np.where(
            optimal_solution == 0,
            np.mean(aggregated_variables, axis=0),
            optimal_solution,
        )

        return characteristic_scales

    def generate_weights(
        self, aggregated_variables: xr.DataArray, direction_search_vector: np.array
    ):
        """
        Generate weights for the MGA objective functions based on the aggregated variables and the random directions.
        In order to improve performances in case where the aggregated variables are vastly different in scales, the
        random directions search vector must be standardized using the characteristic scales of the aggregate variables.
        This scale approximately normalizes the aggregate variables by dividing them by the values of the aggregated
        variables in the optimal solution, when available. When these are zero, the characteristic scales are estimated
        to roughly match the expected magnitude of the variables in the near-optimal space.

        :param aggregated_variables: Aggregated variables to consider for the MGA objective functions generation
        :param direction_search_vector: Random directions vector d

        :return: Weights vector w
        """
        standardized_direction_search_vector = (
            direction_search_vector
            / self.generate_characteristic_scales(aggregated_variables)
        )
        return standardized_direction_search_vector

    def generate_alternatives_objective_functions(
        self, aggregated_variables: xr.DataArray, weights: np.array
    ):
        """
        Generate a set of alternatives objective function based on the weights generated.

        :param aggregated_variables: Aggregated variables to consider for the MGA objective functions generation
        :param weights: Weights vector w

        :return: Alternatives objective functions vector f_k
        """

        return np.dot(aggregated_variables, weights.T)
