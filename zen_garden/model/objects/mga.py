"""""
Class defining Modeling to Generate Alternatives functionalities:
- Capability to generate random directions samples from  a normal distribution
- Capability to generate weight for the MGA objective functions based on the aggregated variables and the random
    directions
- Capability to generate a set of alternatives objective function based on the weights generated
"""

import logging
import random
import numpy as np
import xarray as xr
from scipy.stats import truncnorm

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
        characteristic_scales_config: dict,
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
        self.characteristic_scales_config = characteristic_scales_config
        self.optimization_setup = optized_setup
        self.direction_search_vector = {}
        self.characteristic_scales = None

        logging.info("--- Modeling to Generate Alternatives accessed to generate near-optimal solutions ---")

    def generate_random_directions(self, seed: int = 0) -> dict:
        """
        Generate random directions samples from a normal distribution with mean 0 and standard deviation 1 for the MGA
        objective functions generation.

        :param seed: Seed for the random number generator (type: int, default: 0)

        :return: Random directions dictionary direction_search_vector for each of the capacity technologies variables
            (type: dict)
        """

        random.seed(seed)
        for technology in self.optimization_setup.model.solution.capacity.set_technologies.values:
            random_value = truncnorm.rvs(-1, 1)
            self.direction_search_vector[technology] = random_value

        return self.direction_search_vector

    def generate_characteristic_scales(self) -> xr.DataArray:
        """
        Generate characteristic scales L for the new decision variables to be normalized. L is obtained by dividing the
        variables by their values in the optimal solution, when available. When these are zero, the characteristic
        scales are estimated to roughly match the expected magnitude of the variables in the near-optimal space.

        :return: Characteristic scales DataArray characteristic_scales (type: xr.DataArray)
        """
        capacity_variables = self.optimization_setup.model.solution.capacity
        self.characteristic_scales = xr.full_like(capacity_variables, fill_value=np.nan)
        logging.info(
            "Generating characteristic scales: in case where the variable is zero, the characteristic scale is"
            "estimated to roughly match the expected its magnitude in the near-optimal space.",
        )

        for index in np.ndindex(capacity_variables.shape):
            coords = {
                dim: capacity_variables.coords[dim].values[index[dim_idx]]
                for dim_idx, dim in enumerate(capacity_variables.dims)
            }
            capacity_value = capacity_variables.sel(coords)

            if np.isnan(capacity_value):
                characteristic_value = np.nan
            elif capacity_value > 1e-3:
                characteristic_value = capacity_value
            else:
                estimated_value = self.characteristic_scales_config[coords["set_technologies"]]["default_value"]
                characteristic_value = estimated_value

            self.characteristic_scales.values[index] = characteristic_value

        self.characteristic_scales = self.characteristic_scales.rename("characteristic_scales")

        return self.characteristic_scales

    def generate_weights(self) -> xr.DataArray:
        """
        Generate weights for the MGA objective functions based on the random direction and the characteristic scales.

        :return: Weights DataArray weights (type: xr.DataArray)
        """
        self.characteristic_scales = self.generate_characteristic_scales()
        self.direction_search_vector = self.generate_random_directions()

        weights = xr.full_like(self.characteristic_scales, fill_value=np.nan)

        for index in np.ndindex(self.characteristic_scales.shape):
            coords = {
                dim: self.characteristic_scales.coords[dim].values[index[dim_idx]]
                for dim_idx, dim in enumerate(self.characteristic_scales.dims)
            }
            characteristic_scale = self.characteristic_scales.sel(coords)
            direction_search = self.direction_search_vector[coords["set_technologies"]]

            weights.values[index] = direction_search / characteristic_scale

        weights = weights.rename("weights")

        return weights

    def generate_alternatives_objective_functions(self, aggregated_variables: xr.DataArray, weights: np.array):
        """
        Generate a set of alternatives objective function based on the weights generated.

        :param aggregated_variables: Aggregated variables to consider for the MGA objective functions generation
        :param weights: Weights vector w

        :return: Alternatives objective functions vector f_k
        """

        return np.dot(aggregated_variables, weights.T)
