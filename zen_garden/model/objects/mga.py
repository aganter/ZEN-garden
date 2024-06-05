"""""
Class defining Modeling to Generate Alternatives functionalities:
- Sanity checks for input data
- Load and store input data for the MGA scenario
- Generate weigths for the MGA objective function based on random direction and characteristic scales
- Add a cost constraint to the optimization problem
- Capability to solve the optimization problem for N iterations defined in the config and save the results
"""

from pathlib import Path
import os
import logging
import time
import numpy as np
import xarray as xr
import psutil
from scipy.stats import truncnorm

from zen_garden.preprocess.extract_input_data import DataInputMGA
from zen_garden.model.optimization_setup import OptimizationSetup
from zen_garden.model.objects.component import Constraint
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
        self.mga_data_input = DataInputMGA(
            element=self,
            system=self.config_mga.system,
            analysis=self.config_mga.analysis,
            solver=self.config_mga.solver,
            energy_system=self.mga_solution.energy_system,
            unit_handling=None,
            scenario_name=self.scenario_name,
        )
        self.sanity_checks_mga_iteration_scenario()
        self.decision_variables: list[list] = []
        self.store_input_data()

        self.direction_search_vector = {}
        self.sanity_checks_characteristic_scales_file()
        self.characteristic_scales = None

        self.cost_constraint = Constraint(self.optimized_setup.sets)
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
        subset_objective = self.mga_data_input.decision_variables_dict["objective_set"][self.mga_objective_obj]
        set_solution_variables = getattr(self.optimized_setup.model.solution, self.mga_objective_obj).values.tolist()
        common_variables = list(set(subset_objective).intersection(set_solution_variables))
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

        if self.name in self.mga_data_input.scenario_dict.keys():
            self.mga_solution.config["objective_variables"] = self.mga_data_input.decision_variables_dict[
                "objective_variables"
            ]

            for obj in self.mga_data_input.decision_variables_dict["objective_set"][self.mga_objective_obj]:
                for loc in self.mga_data_input.decision_variables_dict["objective_set"][self.mga_objective_loc]:
                    self.decision_variables.append([obj, loc])

    def generate_random_directions(self) -> dict:
        """
        Generate random directions from a normal distribution with mean 0 and standard deviation 1 for each of the
        decision variables.

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
        complete_xr_variables = getattr(
            self.optimized_setup.model.solution, self.mga_solution.config["objective_variables"]
        )
        subset_objective = self.mga_data_input.decision_variables_dict["objective_set"][self.mga_objective_obj]
        condition = complete_xr_variables[self.mga_objective_obj].isin(subset_objective)
        xr_variables = complete_xr_variables.where(condition, drop=True)
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

    def generate_weights(self) -> xr.DataArray:
        """
        Generate weights for MGA objective function based on random direction and characteristic scales.

        :return: Weights DataArray (type: xr.DataArray)
        """
        self.characteristic_scales = self.generate_characteristic_scales()
        self.direction_search_vector = self.generate_random_directions()

        weights = xr.full_like(self.characteristic_scales, fill_value=np.nan)

        for index in np.ndindex(self.characteristic_scales.shape):
            coords = {
                dim: self.characteristic_scales.coords[dim].values[index[dim_idx]]
                for dim_idx, dim in enumerate(self.characteristic_scales.dims)
            }
            coords_subset_tuple = tuple(
                {key: coords[key] for key in [f"{self.mga_objective_obj}", f"{self.mga_objective_loc}"]}.values()
            )
            characteristic_scale = self.characteristic_scales.sel(coords)

            if coords_subset_tuple in self.direction_search_vector:
                direction_search = self.direction_search_vector[coords_subset_tuple]
                weights.values[index] = direction_search / characteristic_scale

        self.mga_solution.mga_weights = weights.rename("weights")

    def add_cost_constraint(self):
        """
        Add a cost constraint to the optimization problem
        """
        logging.info("Construct pe.Constraint for the Total Cost Deviation allowed")
        pid = os.getpid()
        t_start = time.perf_counter()
        constraints = self.mga_solution.constraints
        constraints.add_constraint_rule(
            self.mga_solution.model,
            name="constraint_optimal_cost_total_deviation",
            index_sets=self.mga_solution.sets["set_time_steps_yearly"],
            rule=self.constraint_cost_total_deviation,
            doc="Limit on total cost of the energy system",
        )
        t_end = time.perf_counter()
        logging.info("Time to construct pe.Sets: %.4f seconds", t_end - t_start)
        logging.info("Memory usage: %s MB", psutil.Process(pid).memory_info().rss / 1024**2)

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
        rhs = (1 + self.cost_slack_variables) * self.optimized_setup.model.objective_value
        cost_constraint = lhs <= rhs

        return self.cost_constraint.return_contraints(cost_constraint)

    def run(self):
        """
        Solve the optimization problem
        """
        for iteration in range(self.config_mga["n_objectives"]):
            logging.info("")
            logging.info("--- MGA Iteration %s ---", iteration + 1)
            logging.info("")
            steps_horizon = self.mga_solution.get_optimization_horizon()
            self.generate_weights()

            for step in steps_horizon:
                StringUtils.print_optimization_progress(self.scenario_name, steps_horizon, step)
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
                subfolder = (subfolder, Path(f"iteration_{iteration + 1}"))
                Postprocess(
                    model=self.mga_solution,
                    scenarios=self.config_mga.scenarios,
                    model_name=self.mga_solution.model_name,
                    subfolder=subfolder,
                    scenario_name=scenario_name,
                    param_map=param_map,
                )
