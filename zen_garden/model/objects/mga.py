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

import logging
import math
import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import truncnorm

from zen_garden.utils import InputDataChecks, StringUtils, OptimizationError
from zen_garden.preprocess.extract_input_data import DataInputMGA
from zen_garden.model.optimization_setup import OptimizationSetup
from zen_garden.model.objects.benders_decomposition.benders import BendersDecomposition
from zen_garden.postprocess.postprocess import Postprocess
from zen_garden.utils import setup_logger, InputDataChecks, StringUtils, ScenarioUtils
from pathlib import Path

class ModelingToGenerateAlternatives:
    """
    Class defining Modeling to Generate Alternatives
    """

    label = "Modeling_To_Generate_Alternatives"

    def __init__(
        self,
        config: dict,
        optimized_setup: OptimizationSetup,
        scenario_name: str,
        scenario_dict: dict,
        iteration: int
    ):
        """
        Initialize the Modeling to Generate Alternatives object.

        :param config: Configuration dictionary
        :param optimized_setup: OptimizationSetup object of the optimized original problem
        :param scenario_name: Name of the scenario
        :param scenario_dict: Dictionary of the scenario
        """
        self.name = "ModelingToGenerateAlternatives"
        self.config = config
        self.config_mga = config.mga
        self.optimized_setup = optimized_setup
        self.scenario_name = scenario_name
        self.scenario_dict = scenario_dict
        self.iteration = iteration

        self.input_data_checks = InputDataChecks(config=self.config_mga, optimization_setup=None)
        self.input_data_checks.check_dataset()
        # Initialize the OptimizationSetup object for the MGA iteration model
        self.set_mga_solution()
        # set input path
        self.input_path = Path(self.config.analysis.dataset / self.config_mga.input_path) # changed Alissa

        # set DataInputMGA object
        self.set_data_input_mga()
        self.sanity_checks_mga_iteration_scenario()
        self.decision_variables: list[list] = []
        self.store_input_data()

        # weight generation
        self.direction_search_vector = {}
        #self.sanity_checks_characteristic_scales_file()
        #self.characteristic_scales = None
        self.generate_weights()

    def set_mga_solution(self):
        """set mga solution object"""
        self.mga_solution: OptimizationSetup = OptimizationSetup(
            config=self.config_mga,
            solver=self.config_mga.solver,
            model_name=self.optimized_setup.model_name,
            scenario_name=self.scenario_name,
            scenario_dict=self.scenario_dict,
            input_data_checks=self.input_data_checks,
        )
        self.mga_solution.cost_optimal_mga = self.optimized_setup.model.objective.value
        self.mga_objective_obj = None
        self.mga_objective_loc = None

    def set_data_input_mga(self):
        """set data input object for mga"""
        self.mga_data_input = DataInputMGA(
            element=self,
            system=self.config_mga.system,
            analysis=self.config_mga.analysis,
            solver=self.config_mga.solver,
            energy_system=self.mga_solution.energy_system,
            unit_handling=None,
            scenario_name=self.scenario_dict["base_scenario"],
        )

    def min_max_scaling(self):
        """
        get min-max values for min-max scaling
        """
        if os.path.exists(self.input_path / f"min_max_values.csv"):
            self.min_max_values = pd.read_csv(self.input_path / f"min_max_values.csv", index_col=[0, 1])
            return
        run_benders = self.config_mga.benders.benders_decomposition
        run_monolithic = self.config_mga.run_monolithic_optimization
        self.config_mga.benders.benders_decomposition = False
        self.config_mga.run_monolithic_optimization = True
        current_scenario_dict = self.scenario_dict
        current_subfolder = current_scenario_dict["sub_folder"]
        objective_weights = {"min": 1, "max": -1}
        objective_var = getattr(self.optimized_setup.model.solution, self.mga_solution.config["objective_variables"])
        subset_obj = self.mga_data_input.decision_variables_dict["objective_set"][self.mga_objective_obj]
        subset_loc =self.mga_data_input.decision_variables_dict["objective_set"][ self.mga_objective_loc]
        idx = pd.MultiIndex.from_product([subset_obj, subset_loc], names=[self.mga_objective_obj, self.mga_objective_loc])
        self.min_max_values = pd.DataFrame(np.nan, index=idx, columns=list(objective_weights.keys()))
        for tech, loc in idx:
            for sense, weight in objective_weights.items():
                sub_folder = "_".join([sense,tech,loc])
                self.scenario_dict["sub_folder"] = sub_folder
                self.scenario_dict["param_map"][sub_folder] = {"ModelingToGenerateAlternatives": {'objective_elements_definition': {'default_op': 1}}}
                weights = xr.full_like(objective_var, fill_value=0)
                weights.loc[tech,:,loc,:] = weight
                self.mga_solution.mga_weights = weights.rename("weights")
                self.fit()
                var = getattr(self.mga_solution.model.solution, self.mga_solution.config["objective_variables"])
                self.min_max_values.loc[(tech,loc),sense] = var.loc[tech,:,loc,:].sum().sum().values
        self.min_max_values.to_csv(self.input_path / f"min_max_values.csv", index=True)
        # reset scenario subfolder
        self.scenario_dict["sub_folder"] = current_subfolder
        # return to original settings
        self.config_mga.benders.benders_decomposition = run_benders
        self.config_mga.run_monolithic_optimization = run_monolithic

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
            self.mga_solution.config["objective_variables"] = self.mga_data_input.decision_variables_dict["objective_variables"]
            # Update the list of decision variables by combining the objectives and the locations
            for obj in self.mga_data_input.decision_variables_dict["objective_set"][self.mga_objective_obj]:
                for loc in self.mga_data_input.decision_variables_dict["objective_set"][self.mga_objective_loc]:
                    self.decision_variables.append([obj, loc])

            if self.mga_solution.config["objective_variables"] in ("capacity_addition", "capacity_addition_supernodes"):
                list_of_location = list(set([loc for _, loc in self.decision_variables]))
                if self.mga_solution.config["objective_variables"] == "capacity_addition":
                    set_edges = self.mga_data_input.energy_system.set_edges
                    all_possible_edges = [
                        f"{loc1}-{loc2}" for loc1 in list_of_location for loc2 in list_of_location if loc1 != loc2
                    ]
                else:
                    set_edges = list(set(self.mga_data_input.energy_system.superedges.keys()))
                    all_possible_edges = [f"{loc1}-{loc2}" for loc1 in list_of_location for loc2 in list_of_location]

                transport_technologies = self.mga_data_input.energy_system.set_transport_technologies
                expanded_decision_variables = []

                for obj, loc in self.decision_variables:
                    if obj in transport_technologies:
                        for edge in all_possible_edges:
                            if edge in set_edges:
                                if [obj, edge] not in expanded_decision_variables:
                                    expanded_decision_variables.append([obj, edge])
                    else:
                        if [obj, loc] not in expanded_decision_variables:
                            expanded_decision_variables.append([obj, loc])

            self.decision_variables = expanded_decision_variables

    def generate_random_directions(self) -> dict:
        """
        Generate random directions from a normal distribution with mean 0 and standard deviation 1 for each of the
        decision variables. The samples are taken and truncated to the interval [-1, 1].

        :return: Random direction_search_vector for each of the decision variables (type: dict)
        """
        scenarios = self.config_mga.scenarios.keys()
        idx_dv = pd.MultiIndex.from_tuples(self.decision_variables,
                                           names=[self.mga_objective_obj, self.mga_objective_loc])
        if os.path.exists(self.input_path / f"random_directions_matrix.csv"):
            direction_search_matrix = pd.read_csv(self.input_path / f"random_directions_matrix.csv", index_col=[0, 1])
            direction_search_matrix.rename({col: int(col) for col in direction_search_matrix.columns}, axis=1, inplace=True)
            idx = direction_search_matrix.index
            if len(direction_search_matrix.columns) >= len(scenarios) and set(idx_dv).issubset(set(idx)):
                self.direction_search_vector = direction_search_matrix.loc[idx_dv,self.iteration] #TODO check if is series
                return

        logging.info("Generating new random directions matrix")
        direction_search_matrix = pd.DataFrame((np.random.rand(len(idx_dv), len(scenarios)) - 0.5)*2, index=idx_dv, columns=np.arange(0,len(scenarios)))
        direction_search_matrix.to_csv(self.input_path / f"random_directions_matrix.csv", index=True)

        self.direction_search_vector = direction_search_matrix.loc[idx_dv,self.iteration]
        # self.direction_search_vector = {
        #     tuple(component): truncnorm.rvs(-1, 1, loc=0, scale=1) for component in self.decision_variables
        # }
        # return self.direction_search_vector

    def generate_characteristic_scales(self) -> xr.DataArray:
        """
        Generate characteristic scales L for the new decision variables to be normalized. L is obtained by dividing the
        variables by their values in the optimal solution, when available.

        :return: Characteristic scales DataArray characteristic_scales (type: xr.DataArray)
        """
        logging.info(
            "Generating characteristic scales: in case where the variable is zero, the characteristic scale is"
            " estimated to roughly match its expected magnitude in the near-optimal space.",
        )
        # Extract the array named as the type of variables that we want to set as objective variables from
        # the original optimized solution (e.g. capacity)
        complete_xr_variables = getattr(self.optimized_setup.model.solution, self.mga_solution.config["objective_variables"])
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
            elif value < 1e-3:
                characteristic_value = 1  # value
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
        if not hasattr(self, "min_max_values"):
            self.min_max_scaling()
        #self.characteristic_scales = self.generate_characteristic_scales()
        self.generate_random_directions()
        delta = self.min_max_values["max"] - self.min_max_values["min"]
        delta[delta.round(1)==0] = 1 # avoid division by zero if technology remains unused
        weights = self.direction_search_vector / delta
        weights.index.names = [self.mga_objective_obj, self.mga_objective_loc]
        weights = weights.to_xarray()
        objective_var = getattr(self.optimized_setup.model.solution, self.mga_solution.config["objective_variables"])
        weights = weights.broadcast_like(objective_var).fillna(0)
        self.mga_solution.mga_weights = weights.rename("weights")

    def fit(self):
        """
        Solve the optimization problem and postprocess the results
        """
        steps_horizon = self.mga_solution.get_optimization_horizon()
        # Just for completeness, iterate through horizon steps. It is not tested yet MGA class
        # with multiple steps
        for step in steps_horizon:
            StringUtils.print_optimization_progress(self.scenario_name, steps_horizon, step, self.config_mga.system)
            # Overwrite time indices
            self.mga_solution.overwrite_time_indices(step)
            # Create optimization problem
            self.mga_solution.construct_optimization_problem()
            if self.config_mga.solver["use_scaling"]:
                self.mga_solution.scaling.run_scaling()
            else:
                self.mga_solution.scaling.analyze_numerics()
            # Create scenario name, subfolder and param_map for postprocessing
            scenario_name, subfolder, param_map = StringUtils.generate_folder_path(
                config=self.config_mga,
                scenario=self.scenario_name,
                scenario_dict=self.scenario_dict,
                steps_horizon=steps_horizon,
                step=step,
            )
            if self.config_mga["run_monolithic_optimization"]:
                # Solve the optimization problem
                self.mga_solution.solve()

                # Break if infeasible
                if not self.mga_solution.optimality:
                    self.mga_solution.write_IIS()
                    raise OptimizationError(self.mga_solution.model.termination_condition)

                if self.config_mga.solver["use_scaling"]:
                    self.mga_solution.scaling.re_scale()

                # Save new capacity additions and cumulative carbon emissions for next time step
                self.mga_solution.add_results_of_optimization_step(step)
                # Evaluate results
                # Write results
                Postprocess(
                    model=self.mga_solution,
                    scenarios=self.config_mga.scenarios,
                    model_name=self.mga_solution.model_name,
                    subfolder=subfolder,
                    scenario_name=scenario_name,
                    param_map=param_map,
                )

            # Benders Decomposition
            if self.config_mga.benders.benders_decomposition:
                logging.info("--- Benders Decomposition accessed ---")
                logging.info("")
                if self.config_mga.solver["use_scaling"]:
                    self.mga_solution.scaling.scale_again_solution()

                benders_decomposition = BendersDecomposition(
                    config=self.config_mga,
                    analysis=self.config_mga.analysis,
                    monolithic_model=self.mga_solution,
                    scenario_name=str(subfolder),
                    use_monolithic_solution=self.config_mga.benders.use_monolithic_solution,
                )
                benders_decomposition.fit()
