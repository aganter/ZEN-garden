"""""
Class defining Modeling to Generate Alternatives functionalities:
- Capability to generate random directions samples from  a normal distribution
- Capability to generate weight for the MGA objective functions based on the aggregated variables and the random
    directions
- Capability to generate a set of alternatives objective function based on the weights generated
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
    Class defining Modeling to Generate Alternatives functionalities
    """

    label = "Modeling_To_Generate_Alternatives"

    def __init__(
        self,
        config_mga: dict,
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
        self.name = "Modeling_To_Generate_Alternatives"
        self.config = config_mga
        self.config_mga = config_mga.mga
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

        self.decision_variables = []

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
        self.store_input_data()

    def store_input_data(self):
        """
        Read and store the input data for the MGA scenario in the correct way. The data input has two attributes, one
        of them is the decision_variable dict. This function contains a sanity check for the structure of the dictionary
        and stores as follow:
        - The "objective_type" attrbute contains the type of the objective function to be generated
        - The "aggregated_variables" attribute is a list of list [[key, node], [key, node], ...] with the aggregated
            variables and the corresponding nodes
        Other attributed defined in the function are functional for the MGA iteration
        """
        self.mga_data_input.scenario_dict = self.scenario_dict
        element = "ModelingToGenerateAlternatives"
        if element in self.mga_data_input.scenario_dict.keys():
            self.mga_solution.config["objective_type"] = self.mga_data_input.decision_variables_dict["objective_type"]
            assert self.mga_solution.config["objective_type"] in [
                "technologies",
                "carriers",
            ], (
                f"Objective {self.mga_solution.config['objective_type']} not recognized."
                "Only 'technologies' and 'carriers' are allowed."
            )
            for tech in self.mga_data_input.decision_variables_dict["decision_variables"]:
                for country in self.mga_data_input.decision_variables_dict["countries"]:
                    self.decision_variables.append([tech, country])

    def generate_random_directions(self) -> dict:
        """
        Generate random directions samples from a normal distribution with mean 0 and standard deviation 1 for the MGA
        objective functions generation.

        :return: Random directions dictionary direction_search_vector for each of the decision variables
            (type: dict)
        """
        self.direction_search_vector = {tuple(component): truncnorm.rvs(-1, 1) for component in self.decision_variables}

        return self.direction_search_vector

    def generate_characteristic_scales(self) -> xr.DataArray:
        """
        Generate characteristic scales L for the new decision variables to be normalized. L is obtained by dividing the
        variables by their values in the optimal solution, when available. When these are zero, the characteristic
        scales are estimated to roughly match the expected magnitude of the variables in the near-optimal space.

        :return: Characteristic scales DataArray characteristic_scales (type: xr.DataArray)
        """
        if self.mga_solution.config["objective_type"] == "technologies":
            xr_variables = self.optimizated_setup.model.solution.capacity
            self.characteristic_scales = xr.full_like(xr_variables, fill_value=np.nan)
        elif self.mga_solution.config["objective_type"] == "carriers":
            xr_variables = self.optimizated_setup.model.solution.flow_import
            self.characteristic_scales = xr.full_like(xr_variables, fill_value=np.nan)
        else:
            raise ValueError(f"Objective type {self.mga_solution.config['objective_type']} not recognized")
        logging.info(
            "Generating characteristic scales: in case where the variable is zero, the characteristic scale is"
            "estimated to roughly match the expected its magnitude in the near-optimal space.",
        )

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
                estimated_value, _ = self.mga_data_input.extract_attribute_value(
                    attribute_name=coords[f"set_{self.mga_solution.config['objective_type']}"],
                    attribute_dict=self.mga_data_input.characteristics_scales_dict,
                )
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

        key_map = {"technologies": "location", "carriers": "nodes"}
        location_key = key_map[self.mga_solution.config["objective_type"]]

        weights = xr.full_like(self.characteristic_scales, fill_value=np.nan)

        for index in np.ndindex(self.characteristic_scales.shape):
            coords = {
                dim: self.characteristic_scales.coords[dim].values[index[dim_idx]]
                for dim_idx, dim in enumerate(self.characteristic_scales.dims)
            }
            coords_subset_tuple = tuple(
                {
                    key: coords[key]
                    for key in [f"set_{self.mga_solution.config['objective_type']}", f"set_{location_key}"]
                    if key in coords
                }.values()
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
