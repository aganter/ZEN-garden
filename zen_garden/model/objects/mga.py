"""
  :Title:        ZEN-GARDEN
  :Created:      April-2024
  :Authors:      Maddalena Cenedese (mcenedese@student.ethz.ch)
  :Organization: Labratory of Reliability and Risk Engineering, ETH Zurich

    Class defining Modeling to Generate Alternatives functionalities:
    - Initialize the MGA object
    - Perform sanity checks on the MGA iteration scenario
    - Perform sanity checks on the characteristic scales file
    - Store the input data for the MGA scenario
    - Generate random directions for each decision variable
    - Generate characteristic scales for the new decision variables
    - Generate weights for the MGA objective function based on random direction and characteristic scales
    - Add a cost constraint to the optimization problem
    - Solve the optimization problem and write the results
"""

import os
import logging
import time
import numpy as np
import xarray as xr
import psutil
import math
from scipy.stats import truncnorm

from zen_garden.preprocess.extract_input_data import DataInputMGA
from zen_garden.model.optimization_setup import OptimizationSetup
from zen_garden.utils import InputDataChecks
from zen_garden.utils import StringUtils
from zen_garden.postprocess.postprocess import Postprocess


class ModelingToGenerateAlternatives:
    """
    Class defining Modeling to Generate Alternatives
    """

    label = "Modeling_To_Generate_Alternatives"

    def __init__(
        self,
        config_mga: dict,
        optimized_setup: OptimizationSetup,
        scenario_name: str,
        scenario_dict: dict,
    ):
        """
        Initialize the Modeling to Generate Alternatives object.

        :param config_mga: Configuration dictionary for the MGA method
        :param optimized_setup: OptimizationSetup object of the optimized original problem
        :param scenario_name: Name of the scenario
        :param scenario_dict: Dictionary of the scenario
        """
        self.name = "ModelingToGenerateAlternatives"
        self.config = config_mga
        self.config_mga = config_mga.mga
        self.optimized_setup = optimized_setup
        self.scenario_name = scenario_name
        self.scenario_dict = scenario_dict

        input_data_checks = InputDataChecks(config=self.config_mga, optimization_setup=None)
        input_data_checks.check_dataset()
        # Initialize the OptimizationSetup object for the MGA iteration model
        self.mga_solution: OptimizationSetup = OptimizationSetup(
            config=self.config_mga,
            model_name=self.optimized_setup.model_name,
            scenario_name=self.scenario_name,
            scenario_dict=self.scenario_dict,
            input_data_checks=input_data_checks,
        )
        self.mga_objective_obj = None
        self.mga_objective_loc = None

        self.input_path = self.config_mga["folder_path"]

        # Initialize the DataInputMGA object
        self.mga_data_input = DataInputMGA(
            element=self,
            system=self.config_mga.system,
            analysis=self.config_mga.analysis,
            solver=self.config_mga.solver,
            energy_system=self.mga_solution.energy_system,
            unit_handling=None,
            scenario_name=self.scenario_dict["base_scenario"],
        )
        self.sanity_checks_mga_iteration_scenario()
        self.decision_variables: list[list] = []
        self.store_input_data()

        self.direction_search_vector = {}
        self.sanity_checks_characteristic_scales_file()
        self.characteristic_scales = None

        self.cost_constraint = None
        self.cost_slack_variables = self.config_mga["cost_slack_variables"]

    def sanity_checks_mga_iteration_scenario(self):
        """
        Perform sanity checks on the MGA iteration scenario.
        """
        logging.info("--- Conduct sanity checks for the MGA iteration scenario")
        dict_to_check = self.mga_data_input.decision_variables_dict
        solution = self.optimized_setup.model.solution
        # Check the strcutre of the dictionary that must be a nested dictionary and contain the keys
        # "objective_variables" with a string as element and "objective_set" that is dictionary
        assert isinstance(dict_to_check, dict), f"{self.scenario_name} must be a dictionary"
        assert (
            "objective_variables" in dict_to_check.keys()
        ), f"{self.scenario_name} must contain key 'objective_variables'"
        assert isinstance(dict_to_check["objective_variables"], str), "Element of'objective_variables' must be a string"
        assert "objective_set" in dict_to_check.keys(), f"{self.scenario_name} must contain key 'objective_set'"

        # Check the existence of the "objective_variables" in the solution variables
        objective_variables = dict_to_check["objective_variables"]
        assert objective_variables in solution.variables, f"{objective_variables} does not exist in solution variables"

        # Check that keys of the "objective_set" are a subset of the coords of the objective variables
        xr_coords = getattr(solution, objective_variables).coords
        self.mga_solution.mga_objective_coords = list(xr_coords.keys())
        dict_keys = dict_to_check["objective_set"].keys()
        common_keys = [key for key in xr_coords.keys() if key in dict_keys]
        assert len(common_keys) == 2, "The keys of 'objective_set' do not match the coords of the objective variables"
        # Check that elements of "objective_set" are a subset of the values of the coords of the objective variables
        for key in common_keys:
            assert set(dict_to_check["objective_set"][key]).issubset(
                xr_coords[key].values
            ), f"Values of the key {key} are not a subset of the coordinates of the objective variable"

        # Check that key of "objective_set" belong to the allowed keys defined in the config and store them
        self.mga_objective_obj = next(key for key in dict_keys if key in self.config_mga.allowed_mga_objective_objects)
        self.mga_objective_loc = next(
            key for key in dict_keys if key in self.config_mga.allowed_mga_objective_locations
        )
        logging.info("Sanity checks for the MGA iteration scenario passed :) ---")

    def sanity_checks_characteristic_scales_file(self):
        """
        Perform sanity checks on the characteristic scales file
        """
        dict_to_check = self.mga_data_input.characteristics_scales_dict
        # Check that the value of the coords of the objective variables are a subset of keys of the dictionary
        subset_objective_vars = self.mga_data_input.decision_variables_dict["objective_set"][self.mga_objective_obj]
        set_solution_vars = getattr(self.optimized_setup.model.solution, self.mga_objective_obj).values.tolist()
        common_variables = list(set(subset_objective_vars).intersection(set_solution_vars))
        for key in common_variables:
            assert key in dict_to_check.keys(), f"{key} is not in the characteristic scales dictionary"
        # Check each element has keys "default_value" to be a number and "unit" to be a string or a number
        for key in dict_to_check.keys():
            assert "default_value" in dict_to_check[key].keys(), f"{key} must contain key 'default_value'"
            assert isinstance(dict_to_check[key]["default_value"], (int)), "default_value must be a number"
            assert "unit" in dict_to_check[key].keys(), f"{key} must contain key 'unit'"
            assert isinstance(dict_to_check[key]["unit"], (str, int)), "unit must be a string or a number"

    def store_input_data(self):
        """
        Read and store the input data for the MGA scenario updating the config and decision_variables
        """
        self.mga_data_input.scenario_dict = self.scenario_dict

        # ModelingToGenerateAlternatives must be a key of the mga_iteration scenarios dictionary
        if self.name in self.mga_data_input.scenario_dict.keys():
            self.mga_solution.config["objective_variables"] = self.mga_data_input.decision_variables_dict[
                "objective_variables"
            ]
            # Update the list of decision variables by combining the objectives and the locations
            for obj in self.mga_data_input.decision_variables_dict["objective_set"][self.mga_objective_obj]:
                for loc in self.mga_data_input.decision_variables_dict["objective_set"][self.mga_objective_loc]:
                    self.decision_variables.append([obj, loc])

    def generate_random_directions(self) -> dict:
        """
        Generate random directions from a normal distribution with mean 0 and standard deviation 1 for each of the
        decision variables. The samples are taken and truncated to the interval [-1, 1].

        :return: Random direction_search_vector for each of the decision variables (type: dict)
        """
        self.direction_search_vector = {tuple(component): truncnorm.rvs(-1, 1) for component in self.decision_variables}

        return self.direction_search_vector

    def generate_characteristic_scales(self) -> xr.DataArray:
        """
        Generate characteristic scales L for the new decision variables to be normalized. L is obtained by dividing the
        variables by their values in the optimal solution, when available.

        :return: Characteristic scales DataArray characteristic_scales (type: xr.DataArray)
        """
        logging.info(
            "Generating characteristic scales: in case where the variable is zero, the characteristic scale is"
            "estimated to roughly match its expected magnitude in the near-optimal space.",
        )
        # Extract the array named as the type of variables that we want to set as objective variables from
        # the original optimized solution (e.g. capacity)
        complete_xr_variables = getattr(
            self.optimized_setup.model.solution, self.mga_solution.config["objective_variables"]
        )
        # Extract the exact name of the variables that we want to set as objective variables from the input data
        subset_objective_vars = self.mga_data_input.decision_variables_dict["objective_set"][self.mga_objective_obj]
        # Generate an array that intersects the original array with the subset of variables that we want to set as
        # objective variables and sum over the time dimension because we are interested in the total amount of the
        # variable in the system
        xr_variables = complete_xr_variables.where(
            complete_xr_variables[self.mga_objective_obj].isin(subset_objective_vars), drop=True
        ).sum(dim=next(dim for dim in complete_xr_variables.dims if dim.startswith("set_time_steps")), skipna=False)
        # Initialize the characteristic scales array with the same shape as the variables array
        self.characteristic_scales = xr.full_like(xr_variables, fill_value=np.nan)

        for index in np.ndindex(xr_variables.shape):
            coords = {
                dim: xr_variables.coords[dim].values[index[dim_idx]] for dim_idx, dim in enumerate(xr_variables.dims)
            }
            value = xr_variables.sel(coords)

            if np.isnan(value):
                characteristic_value = np.nan
            elif value > 1e-3:
                characteristic_value = value
            else:
                characteristic_value, _ = self.mga_data_input.extract_attribute_value(
                    attribute_name=coords[f"{self.mga_objective_obj}"],
                    attribute_dict=self.mga_data_input.characteristics_scales_dict,
                )

            self.characteristic_scales.values[index] = characteristic_value

        return self.characteristic_scales.rename("characteristic_scales")

    def generate_weights(self):
        """
        Generate weights for MGA objective function based on random direction and characteristic scales.
        """
        self.characteristic_scales = self.generate_characteristic_scales()
        self.direction_search_vector = self.generate_random_directions()

        weights = xr.full_like(self.characteristic_scales, fill_value=np.nan)

        for index in np.ndindex(self.characteristic_scales.shape):
            # Extract the coordinates of interest of the characteristic scales: the name of objective and the location
            # and select the value of the characteristic scales at the found coordinates
            coords = {
                dim: self.characteristic_scales.coords[dim].values[index[dim_idx]]
                for dim_idx, dim in enumerate(self.characteristic_scales.dims)
            }
            coords_subset_tuple = tuple(
                {key: coords[key] for key in [f"{self.mga_objective_obj}", f"{self.mga_objective_loc}"]}.values()
            )
            characteristic_scale = self.characteristic_scales.sel(coords)
            #  Only if the same coordinates are in the direction_search_vector, and the characteristic scale is not NaN
            # the weights are calculated
            if coords_subset_tuple in self.direction_search_vector and not math.isnan(characteristic_scale):
                direction_search = self.direction_search_vector[coords_subset_tuple]
                weights.values[index] = direction_search / characteristic_scale

        self.mga_solution.mga_weights = weights.rename("weights")

    def add_cost_constraint(self):
        """
        Add a cost deviation constraint to the optimization problem
        """
        logging.info("Construct pe.Constraint for the Total Cost Deviation allowed")
        pid = os.getpid()
        t_start = time.perf_counter()
        constraints = self.mga_solution.constraints
        self.constraint_cost_total_deviation(model_constraints=constraints)
        t_end = time.perf_counter()
        logging.info("Time to construct pe.Sets: %.4f seconds", t_end - t_start)
        logging.info("Memory usage: %s MB", psutil.Process(pid).memory_info().rss / 1024**2)

    def constraint_cost_total_deviation(self, model_constraints):
        """
        Limit on the total cost objective of the energy system based on the optimized total cost of the energy system
        and a chosen deviation indicated by the cost_slack_variables.
        """
        lhs = self.mga_solution.model.variables["net_present_cost"].sum(dim="set_time_steps_yearly")
        rhs = (1 + self.cost_slack_variables) * self.optimized_setup.model.objective.value
        self.cost_constraint = lhs <= rhs

        model_constraints.add_constraint("constraint_optimal_cost_total_deviation", self.cost_constraint)

    def run(self):
        """
        Solve the optimization problem and postprocess the results
        """

        steps_horizon = self.mga_solution.get_optimization_horizon()
        self.generate_weights()

        for step in steps_horizon:
            StringUtils.print_optimization_progress(self.scenario_name, steps_horizon, step, self.config_mga.system)
            self.mga_solution.overwrite_time_indices(step)
            self.mga_solution.construct_optimization_problem()
            self.add_cost_constraint()
            self.mga_solution.solve()

            if not self.mga_solution.optimality:
                self.mga_solution.write_IIS()
                break

            self.mga_solution.add_results_of_optimization_step(step)
            scenario_name, subfolder, param_map = self.mga_solution.generate_output_paths(
                config_system=self.config_mga.system, step=step, steps_horizon=steps_horizon
            )
            Postprocess(
                model=self.mga_solution,
                scenarios=self.config_mga.scenarios,
                model_name=self.mga_solution.model_name,
                subfolder=subfolder,
                scenario_name=scenario_name,
                param_map=param_map,
            )
