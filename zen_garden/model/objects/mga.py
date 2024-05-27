"""""
Class defining Modeling to Generate Alternatives functionalities:
- Capability to generate random directions samples from  a normal distribution
- Capability to generate weight for the MGA objective functions based on the aggregated variables and the random
    directions
- Capability to generate a set of alternatives objective function based on the weights generated
"""

from pathlib import Path
import os
import json
import logging
import numpy as np
import xarray as xr
from scipy.stats import truncnorm

from zen_garden.preprocess.extract_input_data import DataInput
from zen_garden.model.optimization_setup import OptimizationSetup
from zen_garden.model.objects.component import Constraint
from zen_garden.utils import InputDataChecks
from zen_garden.utils import StringUtils
from zen_garden.postprocess.postprocess import Postprocess


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
        self.direction_search_vector = {}
        self.characteristic_scales = None

        self.cost_constraint = Constraint(self.optimizated_setup.sets)
        self.cost_slack_variables = self.config_mga["cost_slack_variables"]

        self.input_path = self.config_mga["folder_path"]
        self.objective_type = None
        self.agregated_variables = None
        self.data_input = DataInput(
            element=self,
            system=self.config_mga.system,
            analysis=self.config_mga.analysis,
            solver=self.config_mga.solver,
            energy_system=self.mga_solution.energy_system,
            unit_handling=None,
        )
        self.store_input_data()

    def store_input_data(self):
        """
        Read and store the input data for the MGA scenarios. The attributes file is composed of nested dictionaries,
        each of them having as a key the scenario name and containing the type of decision variables and the aggreagted
        variables with the corresponding nodes. Lastly, the objective type is stored in the attribute dictionary of the
        OptimizationSetup object to have access to it when building the optimization objective functions.
        """
        self.data_input.scenario_dict = self.scenario_dict
        element = "ModelingToGenerateAlternatives"
        if element in self.scenario_dict.keys():
            objective_key = self.scenario_dict[element]["objective_type"]["aggregated_variables"]
            assert objective_key in self.data_input.attribute_dict.keys(), f"No attributes found for {objective_key}"
            objective_dict = self.data_input.attribute_dict[objective_key]
            self.objective_type = objective_dict["objective_type"]
            self.aggregated_variables = [
                [key, node]
                for key in objective_dict["aggregated_variables"]
                for node in objective_dict["aggregated_variables"][key]["set_location"]
            ]
        self.mga_solution.mga_objective_type = self.objective_type

    def generate_random_directions(self) -> dict:
        """
        Generate random directions samples from a normal distribution with mean 0 and standard deviation 1 for the MGA
        objective functions generation.

        :return: Random directions dictionary direction_search_vector for each of the decision variables
            (type: dict)
        """
        self.direction_search_vector = {
            tuple(component): truncnorm.rvs(-1, 1) for component in self.aggregated_variables
        }

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

        if self.objective_type == "technologies":
            decision_variables = self.optimizated_setup.model.solution.capacity
            self.characteristic_scales = xr.full_like(decision_variables, fill_value=np.nan)
        elif self.objective_type == "carriers":
            decision_variables = self.optimizated_setup.model.solution.flow_import
            self.characteristic_scales = xr.full_like(decision_variables, fill_value=np.nan)
        else:
            raise ValueError(f"Objective type {self.objective_type} not recognized")
        logging.info(
            "Generating characteristic scales: in case where the variable is zero, the characteristic scale is"
            "estimated to roughly match the expected its magnitude in the near-optimal space.",
        )

        for index in np.ndindex(decision_variables.shape):
            coords = {
                dim: decision_variables.coords[dim].values[index[dim_idx]]
                for dim_idx, dim in enumerate(decision_variables.dims)
            }
            value = decision_variables.sel(coords)

            if np.isnan(value):
                characteristic_value = np.nan
            elif value > 1e-3:
                characteristic_value = value
            else:
                estimated_value = characteristic_scales_config[coords[f"set_{self.objective_type}"]]["default_value"]
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
            coords_subset_tuple = tuple(
                {key: coords[key] for key in ["set_technologies", "set_location"] if key in coords}.values()
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
        constraints = self.mga_solution.constraints
        constraints.add_constraint_rule(
            self.mga_solution.model,
            name="constraint_optimal_cost_total_deviation",
            index_sets=self.mga_solution.sets["set_time_steps_yearly"],
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
