"""
  :Title:        ZEN-GARDEN
  :Created:      July-2024
  :Authors:      Maddalena Cenedese (mcenedese@student.ethz.ch)
  :Organization: Labratory of Reliability and Risk Engineering, ETH Zurich

    Class to define the master problem of the Benders Decomposition method.
    This class is a child class of the OptimizationSetup class and inherits all the methods and attributes of the
    parent class.
    The master problem is the design problem and includes only the design variables and constraints.
    In order to ensure only the presence of the design variables and constraints, this class removes the operational
    variables and constraints from the master problem.
"""

import logging
import os
import numpy as np
import pandas as pd
import xarray as xr

from zen_garden.model.optimization_setup import OptimizationSetup


class MasterProblem(OptimizationSetup):
    """
    Class defining the Master Problem of the Benders Decomposition method.
    """

    label = "MasterProblem"

    def __init__(
        self,
        config: dict,
        config_benders: dict,
        solver: dict,
        analysis: dict,
        monolithic_model: OptimizationSetup,
        model_name: str,
        scenario_name: str,
        scenario_dict: dict,
        input_data_checks,
        operational_variables,
        operational_constraints,
        benders_output_folder,
    ):
        """
        Initialize the MasterProblem object.

        :param config: dictionary containing the configuration of the optimization problem
        :param config_benders: dictionary containing the configuration of the Benders Decomposition method
        :param analysis: dictionary containing the analysis configuration
        :param monolithic_model: OptimizationSetup object of the monolithic problem
        :param model_name: name of the model
        :param scenario_name: name of the scenario
        :param scenario_dict: dictionary containing the scenario data
        :param input_data_checks: dictionary containing the input data checks
        :param operational_variables: list of operational variables
        :param operational_constraints: list of operational constraints
        :param benders_output_folder: path to the Benders Decomposition output
        """

        super().__init__(
            config=config,
            solver=solver,
            model_name=model_name,
            scenario_name=scenario_name,
            scenario_dict=scenario_dict,
            input_data_checks=input_data_checks,
        )

        self.name = "MasterProblem"

        self.config = config
        self.config_benders = config_benders
        self.solver = solver
        self.solver.warmstart_fn = None
        self.analysis = analysis

        self.monolithic_model = monolithic_model
        self.operational_variables = operational_variables
        self.operational_constraints = operational_constraints

        # Attributes from the monolithic problem needed to ensure robustness in case of solving Benders for MGA
        self.mga_weights = self.monolithic_model.mga_weights
        self.mga_objective_coords = self.monolithic_model.mga_objective_coords
        self.cost_optimal_mga = self.monolithic_model.cost_optimal_mga

        self.only_feasibility_checks = False

        self.create_master_problem()
        if self.config["run_monolithic_optimization"] and self.config_benders["use_valid_inequalities"]:
            if "constraint_capacity_factor_conversion" in self.model.constraints:
                self.valid_inequalities_conversion_technology()
            if "constraint_capacity_factor_transport" in self.model.constraints:
                self.valid_inequalities_transport_technology()
            if "constraint_capacity_factor_storage" in self.model.constraints:
                self.valid_inequalities_storage_technology()

        self.remove_operational()

        self.folder_output = os.path.abspath(benders_output_folder + "/" + "master_problem")
        self.optimized_time_steps = [0]

        self.add_dummy_constraint_for_binaries()
        self.add_upper_bounds_to_capacity()

    def theta_objective_master(self):
        """
        When the objective function do not include the design variables, we add the outer approximation of the
        subproblem objective function to the master problem.
        """
        self.variables.add_variable(
            self.model,
            name="outer_approximation",
            index_sets=([0], "theta_range"),
            doc="theta variable for the outer approximation of the subproblem objective function",
            unit_category={"time": -1},
            bounds=(0, np.inf),
        )
        self.model.add_objective(
            self.model.variables.outer_approximation.sum(),
            overwrite=True,
        )

    def create_master_problem(self):
        """
        Create the master problem, which is the design problem.
        It includes only the design constraints and the objective function is taken from the config as follow:
        - If the objective function is "mga", we check whether we optimize for design or operational variables:
            - If "capacity" --> design: the objective function of the master problem is the same as the one of the
            monolithic problem
            - If "flow_import" --> operational: the objective function of the master problem is the outer approximation
            of the subproblem objective function
        - If the objective function is "total_cost", or is "total_carbon_emissions", the objective function of the
        master problem is the outer approximation of the subproblem objective function
        """
        self.construct_optimization_problem()
        # We scale the model before removing the operational variables and constraints to avoid differences in the
        # scaling factors between the master and subproblems
        if self.config.solver["use_scaling"]:
            self.scaling.run_scaling()
        else:
            self.scaling.analyze_numerics()

        # Define the objective function
        if self.analysis["objective"] == "mga":
            if "capacity" in str(self.monolithic_model.model.objective):
                self.model.add_objective(self.monolithic_model.model.objective.expression, overwrite=True)
                self.only_feasibility_checks = True
            elif "flow_import" in str(self.monolithic_model.model.objective):
                self.theta_objective_master()
            else:
                raise AssertionError("Objective function not recognized for MGA.")
        elif self.analysis["objective"] == "total_cost" or self.analysis["objective"] == "total_carbon_emissions":
            self.theta_objective_master()
        else:
            logging.error(
                "Objective function %s not supported for Benders Decomposition at the moment.",
                self.config.analysis["objective"],
            )

    def remove_operational(self):
        """
        Remove the operational variables from the master problem.
        """
        logging.info("--- Removing operational constraints from the master problem ---")
        for operational_constraint in self.operational_constraints:
            self.model.constraints.remove(operational_constraint)
        logging.info("--- Removing operational variables from the master problem ---")
        for operational_variable in self.operational_variables:
            self.model.variables.remove(operational_variable)

    def add_dummy_constraint_for_binaries(self):
        """
        Add a constraint not binding for the binaries to ensure the model recognize them in the optimization.
        """
        if hasattr(self.model.variables, "technology_installation"):
            binaries = self.model.variables.technology_installation
            number_of_technologies = (
                self.model.variables.binaries.nvars + 1
            )  # Add one to avoid the constraint to be binding
            constraint = binaries.sum() <= number_of_technologies
            self.constraints.add_constraint("constraint_for_binaries", constraint)

    def add_upper_bounds_to_capacity(self):
        """
        Add upper bounds to the capacity variables to avoid unbounded solutions.
        """
        if self.config["run_monolithic_optimization"]:
            logging.info("Upper bound capacity multiplier: %s", self.config_benders["upper_bound_capacity_multiplier"])
            if hasattr(self.model.variables, "capacity"):
                upper_bound = self.monolithic_model.model.solution.capacity.where(
                    self.monolithic_model.model.solution.capacity != 0,
                    other=self.monolithic_model.model.solution.capacity.max()/2,
                )

                self.model.variables.capacity.upper = (
                    upper_bound * self.config_benders["upper_bound_capacity_multiplier"]
                )
        else:
            if hasattr(self.model.variables, "capacity"):
                self.model.variables.capacity.upper = self.config_benders["upper_bound_capacity_maximum"]

    def valid_inequalities_conversion_technology(self):
        """
        Add valid initial inequalities for conversion technologies if the monolithic model solution is available. This
        will speed up the convergence of the Benders Decomposition method.
        """
        coefficients = self.model.constraints.constraint_capacity_factor_conversion.coeffs
        dims_to_keep = ["set_conversion_technologies", "set_capacity_types", "set_nodes", "set_time_steps_operation"]
        coefficients = coefficients.drop_vars([coord for coord in coefficients.coords if coord not in dims_to_keep])
        coefficients_lhs = coefficients.rename(
            {"set_conversion_technologies": "set_technologies", "set_nodes": "set_location"}
        )
        techs = self.sets["set_conversion_technologies"]
        if len(techs) == 0:
            return
        nodes = self.sets["set_nodes"]
        times = coefficients.sel(_term=0).coords["set_time_steps_operation"]
        time_step_year = xr.DataArray(
            [self.energy_system.time_steps.convert_time_step_operation2year(t) for t in times.data],
            coords=[times],
        )
        term_capacity = (
            coefficients_lhs.sel(_term=0).loc[techs, nodes, :]
            * self.model.variables["capacity"].loc[techs, "power", nodes, time_step_year]
        ).rename({"set_technologies": "set_conversion_technologies", "set_location": "set_nodes"})
        lhs = term_capacity

        time_steps_operation = self.monolithic_model.model.solution["flow_conversion_output"][
            "set_time_steps_operation"
        ].values
        reference_flows_data = np.empty((len(techs), len(nodes), len(time_steps_operation)))
        for i, t in enumerate(techs):
            rc = self.sets["set_reference_carriers"][t][0]
            if rc in self.sets["set_input_carriers"][t]:
                reference_flows_data[i, :, :] = (
                    self.monolithic_model.model.solution["flow_conversion_input"].loc[t, rc, nodes, :].values
                )
            else:
                reference_flows_data[i, :, :] = (
                    self.monolithic_model.model.solution["flow_conversion_output"].loc[t, rc, nodes, :].values
                )

        reference_flows_da = xr.DataArray(
            data=reference_flows_data,
            dims=["set_conversion_technologies", "set_nodes", "set_time_steps_operation"],
            coords={
                "set_conversion_technologies": techs,
                "set_nodes": nodes,
                "set_time_steps_operation": time_steps_operation,
            },
            name="reference_flows",
        )

        rhs = reference_flows_da * (-coefficients.sel(_term=1))
        constraint = lhs >= rhs

        self.constraints.add_constraint("valid_inequalities_conversion_technology", constraint)

    def valid_inequalities_transport_technology(self):
        """
        Add valid initial inequalities for transport technologies if the monolithic model solution is available. This
        will speed up the convergence of the Benders Decomposition method.
        """
        coefficients = self.model.constraints.constraint_capacity_factor_transport.coeffs
        dims_to_keep = ["set_transport_technologies", "set_capacity_types", "set_edges", "set_time_steps_operation"]
        coefficients = coefficients.drop_vars([coord for coord in coefficients.coords if coord not in dims_to_keep])
        coefficients_lhs = coefficients.rename(
            {"set_transport_technologies": "set_technologies", "set_edges": "set_location"}
        )
        techs = self.sets["set_transport_technologies"]
        if len(techs) == 0:
            return
        edges = self.sets["set_edges"]
        times = self.model.variables["flow_transport"].coords["set_time_steps_operation"]
        time_step_year = xr.DataArray(
            [self.energy_system.time_steps.convert_time_step_operation2year(t) for t in times.data],
            coords=[times],
        )
        term_capacity = (
            coefficients_lhs.sel(_term=0).loc[techs, edges, :]
            * self.model.variables["capacity"].loc[techs, "power", edges, time_step_year]
        ).rename({"set_technologies": "set_transport_technologies", "set_location": "set_edges"})

        lhs = term_capacity
        rhs = self.monolithic_model.model.solution["flow_transport"].loc[techs, edges, :] * (-coefficients.sel(_term=1))
        constraints = lhs >= rhs

        self.constraints.add_constraint("valid_inequalities_transport_technology", constraints)

    def get_storage2year_time_step_array(self):
        """returns array with storage2year time steps"""
        times = {
            st: y
            for y in self.sets["set_time_steps_yearly"]
            for st in self.energy_system.time_steps.get_time_steps_year2storage(y)
        }
        times = pd.Series(times, name="set_time_steps_yearly")
        times.index.name = "set_time_steps_storage"
        return times

    def map_and_expand(self, array, mapping):
        """maps and expands array"""
        assert isinstance(mapping, pd.Series) or isinstance(
            mapping.index, pd.Index
        ), "Mapping must be a pd.Series or with a single-level pd.Index"
        # get mapping values
        array = array.sel({mapping.name: mapping.values})
        # rename
        array = array.rename({mapping.name: mapping.index.name})
        # assign coordinates
        array = array.assign_coords({mapping.index.name: mapping.index})
        return array

    def valid_inequalities_storage_technology(self):
        """
        Add valid initial inequalities for storage technologies if the monolithic model solution is available. This
        will speed up the convergence of the Benders Decomposition method.
        """
        coefficients = self.model.constraints.constraint_capacity_factor_storage.coeffs
        dims_to_keep = ["set_storage_technologies", "set_capacity_types", "set_nodes", "set_time_steps_operation"]
        coefficients = coefficients.drop_vars([coord for coord in coefficients.coords if coord not in dims_to_keep])
        coefficients_lhs = coefficients.rename(
            {"set_storage_technologies": "set_technologies", "set_nodes": "set_location"}
        )
        techs = self.sets["set_storage_technologies"]
        if len(techs) == 0:
            return
        nodes = self.sets["set_nodes"]
        times = self.model.variables.coords["set_time_steps_operation"]
        time_step_year = xr.DataArray(
            [self.energy_system.time_steps.convert_time_step_operation2year(t) for t in times.data], coords=[times]
        )
        term_capacity = (
            coefficients_lhs.sel(_term=0).loc[techs, nodes, :]
            * self.model.variables["capacity"].loc[techs, "power", nodes, time_step_year]
        ).rename({"set_technologies": "set_storage_technologies", "set_location": "set_nodes"})

        lhs_storage = term_capacity
        flow_expression_storage = (
            self.monolithic_model.model.solution["flow_storage_charge"]
            + self.monolithic_model.model.solution["flow_storage_discharge"]
        )
        rhs_storage = flow_expression_storage * (-coefficients.sel(_term=1))
        constraints_storage = lhs_storage >= rhs_storage

        self.constraints.add_constraint("valid_inequalities_storage_technology", constraints_storage)

        times = self.get_storage2year_time_step_array()
        capacity = self.map_and_expand(self.model.variables["capacity"], times)
        capacity = capacity.rename({"set_technologies": "set_storage_technologies", "set_location": "set_nodes"})
        capacity = capacity.sel({"set_nodes": nodes, "set_storage_technologies": techs})
        storage_level = self.monolithic_model.model.solution["storage_level"]
        mask_capacity_type = self.model.variables["capacity"].coords["set_capacity_types"] == "energy"
        lhs_max_level = capacity.where(mask_capacity_type, 0.0)
        rhs_max_level = storage_level.where(mask_capacity_type, 0.0)
        constraints_max_level = lhs_max_level >= rhs_max_level

        self.constraints.add_constraint("valid_inequalities_storage_level", constraints_max_level)
