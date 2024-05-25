"""""
Class defining Modeling to Generate Alternatives functionalities:
- Capability to generate random directions samples from  a normal distribution
- Capability to generate weight for the MGA objective functions based on the aggregated variables and the random
    directions
- Capability to generate a set of alternatives objective function based on the weights generated
"""

import os
import json
import importlib.util
import logging
import random
import numpy as np
import xarray as xr
from scipy.stats import truncnorm

from zen_garden.model.optimization_setup import OptimizationSetup
from zen_garden.model.objects.component import Constraint
from zen_garden.utils import InputDataChecks


class ModelingToGenerateAlternatives:
    """
    Class defining Modeling to Generate Alternatives functionalities
    """

    label = "Modeling_To_Generate_Alternatives"
    location_type = None

    def __init__(
        self,
        config_mga: dict,
        n_dimensions: int,
        optized_setup: OptimizationSetup,
        scenario_name: str,
        scenario_dict: dict,
    ):
        """
        Init generic element for the MGA

        :param config_mga: Configuration dictionary for the MGA method
        :param n_dimensions: Number of dimensions N_d of the aggregated decision variables to consider for the MGA
        :param optized_setup: OptimizationSetup object of the optimized original problem
        :param scenario_name: Name of the scenario
        :param scenario_dict: Dictionary of the scenario
        """

        self.config_mga = config_mga
        self.n_dimensions = n_dimensions
        self.optimizated_setup = optized_setup
        self.scenario_name = scenario_name
        self.scenario_dict = scenario_dict

        input_data_checks = InputDataChecks(config=self.config_mga, optimization_setup=None)
        input_data_checks.check_dataset()
        self.mga_solution: OptimizationSetup = OptimizationSetup(
            config=self.config_mga,
            model_name=self.optimizated_setup.model_name,
            scenario_name=self.scenario_name,
            scenario_dict=self.scenario_dict,
            input_data_checks=input_data_checks,
        )
        # It must be initialize with the scenario config to be
        # the iterations of the MGA process and then add the cost constraint at every iterations
        # This is not equal to the optimized set up but it is equal to the initialization of the optimized
        # set up but with scenarios to be the MGA iterations
        self.direction_search_vector = {}
        self.characteristic_scales = None

        self.cost_constraint = Constraint(self.optimizated_setup.sets)
        self.cost_slack_variables = self.config_mga["cost_slack_variables"]

    def generate_random_directions(self, seed: int = 0) -> dict:
        """
        Generate random directions samples from a normal distribution with mean 0 and standard deviation 1 for the MGA
        objective functions generation.

        :param seed: Seed for the random number generator (type: int, default: 0)

        :return: Random directions dictionary direction_search_vector for each of the capacity technologies variables
            (type: dict)
        """

        random.seed(seed)  # Just for debugging purposes
        # It will be elimitated in the final version to have different random values for each run
        for technology in self.optimizated_setup.model.solution.capacity.set_technologies.values:
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
        assert os.path.exists(
            self.config_mga["characteristic_scales_path"]
        ), f"Characteristic scales config JSON file not found at path {self.config_mga['characteristic_scales_path']}!"
        with open(self.config_mga["characteristic_scales_path"], "r", encoding="utf-8") as file:
            characteristic_scales_config = json.load(file)
        # TODO: check the structure of the file

        capacity_variables = self.optimizated_setup.model.solution.capacity
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
                estimated_value = characteristic_scales_config[coords["set_technologies"]]["default_value"]
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

    def add_cost_constraint(self):
        """
        Add a cost constraint to the optimization problem
        """
        self.mga_solution = self.optimizated_setup
        constraints = self.mga_solution.constraints
        original_model = self.optimizated_setup.model
        original_sets = self.optimizated_setup.sets

        constraints.add_constraint_rule(
            original_model,
            name="constraint_optimal_cost_total_deviation",
            index_sets=original_sets["set_time_steps_yearly"],
            rule=self.constraint_cost_total_deviation,
            doc="Limit on total cost of the energy system",
        )

    def constraint_cost_total_deviation(self, year):
        """
        Limit on the total cost objective of the energy system based on the optimized total cost of the energy system
        and a chosen deviation indicated by the cost_slack_variables.

        :return: Constraint for the total cost of the energy system (type: Constraint)
        """

        lhs = sum(
            self.mga_solution.model.variables["net_present_cost"][year]
            for year in self.mga_solution.sets["set_time_steps_yearly"]
        )
        rhs = (1 + self.cost_slack_variables) * self.optimizated_setup.model.objective_value
        cost_constraint = lhs <= rhs

        return self.cost_constraint.return_contraints(cost_constraint)

    def run(self):
        """
        Solve the optimization problem
        """

        for iteration in range(self.config_mga["n_objectives"]):
            logging.info("--- MGA Iteration %s ---", iteration + 1)
            self.generate_characteristic_scales()
            self.add_cost_constraint()
