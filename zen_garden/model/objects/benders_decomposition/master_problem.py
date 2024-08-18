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

        self.folder_output = os.path.abspath(benders_output_folder + "/" + "master_problem")
        self.optimized_time_steps = [0]

        self.add_dummy_constraint_for_binaries()

        self.feasibility_master_iteration = 0
        if self.config_benders["cap_capacity_bounds"]:
            self.add_upper_bounds_to_capacity()
        if self.config_benders["use_valid_inequalities"] and self.config["run_monolithic_optimization"]:
            logging.info("Adding valid inequalities to the master problem.")
            self.valid_inequalities_decision_variables()

        self.remove_operational()

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
                monolithic_capacity = self.monolithic_model.model.solution.capacity.where(
                    self.monolithic_model.model.solution.capacity > 0,
                    self.monolithic_model.model.solution.capacity.mean(),
                )
                upper_bound = self.model.variables.capacity.upper.where(
                    self.model.variables.capacity.upper != np.inf, monolithic_capacity
                )
                self.model.variables.capacity.upper = (
                    upper_bound * self.config_benders["upper_bound_capacity_multiplier"]
                )
        else:
            if hasattr(self.model.variables, "capacity"):
                self.model.variables.capacity.upper = self.config_benders["upper_bound_capacity_maximum"]

    def valid_inequalities_decision_variables(self):
        """
        Add valid inequalities to the master problem for the decision variables.
        """
        if hasattr(self.model.variables, "capacity_supernodes"):
            objective = "capacity_supernodes"
        elif hasattr(self.model.variables, "capacity"):
            objective = "capacity"
        else:
            logging.info("No valid inequalities for the decision variables.")
            return

        # Positive coefficients
        mask = self.monolithic_model.mga_weights > 0
        weights = self.monolithic_model.mga_weights.where(mask, drop=True)

        variables_positive = self.model.variables[objective].broadcast_like(weights)
        variables_positive = variables_positive.where(~np.isnan(weights), drop=True)
        positive_expression = variables_positive.sum()
        rhs_positive = self.monolithic_model.model.solution[objective].broadcast_like(weights)
        rhs_positive = rhs_positive.where(~np.isnan(weights), drop=True)
        rhs_positive = rhs_positive.sum().values.item()

        constraint_positive = positive_expression >= rhs_positive

        # Negative coefficients
        weights = self.monolithic_model.mga_weights.where(~mask, drop=True)
        variables_negative = self.model.variables[objective].broadcast_like(weights)
        variables_negative = variables_negative.where(~np.isnan(weights), drop=True)
        negative_expression = variables_negative.sum()
        rhs_negative = self.monolithic_model.model.solution[objective].broadcast_like(weights)
        rhs_negative = rhs_negative.where(~np.isnan(weights), drop=True)
        rhs_negative = rhs_negative.sum().values.item()

        constraint_negative = negative_expression <= rhs_negative

        self.constraints.add_constraint("valid_inequalities_decision_variables_positive", constraint_positive)
        self.constraints.add_constraint("valid_inequalities_decision_variables_negative", constraint_negative)

    def augment_upper_bound_capacity(self):
        """
        Augment the capacity bounds of the master problem in case of infeasibility.
        """
        gurobi_master_model = getattr(self.model, "solver_model")
        gurobi_master_model.computeIIS()
        upper_bounds_iis = [
            (int(variables.VarName[1:]))
            for variables, iis in zip(gurobi_master_model.getVars(), gurobi_master_model.IISUB)
            if iis != 0
        ]
        if not upper_bounds_iis:
            self.model.print_infeasibilities()
            if [
                "valid_inequalities_decision_variables_positive",
                "valid_inequalities_decision_variables_negative",
            ] in self.model.constraints:
                for constraint in [
                    "valid_inequalities_decision_variables_positive",
                    "valid_inequalities_decision_variables_negative",
                ]:
                    self.model.constraints.remove(constraint)
            else:
                logging.info("No other augmentation possible.")
                return False
        else:
            logging.info("--- Augment Upper Bounds that are Infeasible ---")
            self.model.print_infeasibilities()
            upper_bounds_iis_df = pd.DataFrame(upper_bounds_iis, columns=["labels"])
            invalid_upper_bounds = self.model.variables.flat.merge(upper_bounds_iis_df, on="labels")
            for _, row in invalid_upper_bounds.iterrows():
                label_position = self.model.variables.get_label_position(int(row["labels"]))
                existing_bound = self.model.variables[f"{label_position[0]}"].sel(label_position[1]).upper
                self.model.variables.capacity.upper.loc[
                    label_position[1]["set_technologies"],
                    label_position[1]["set_capacity_types"],
                    label_position[1]["set_location"],
                    label_position[1]["set_time_steps_yearly"],
                ] = (
                    existing_bound * 1.05
                )  # Increase the upper bound by 5% to avoid the same infeasibility

            self.solve()
            self.feasibility_master_iteration += 1

            if self.feasibility_master_iteration > self.config.benders["max_number_feasibility_iterations"]:
                self.model.print_infeasibilities()
                return False
            return True
