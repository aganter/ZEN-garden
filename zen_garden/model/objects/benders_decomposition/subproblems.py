"""
  :Title:        ZEN-GARDEN
  :Created:      July-2024
  :Authors:      Maddalena Cenedese (mcenedese@student.ethz.ch)
  :Organization: Labratory of Reliability and Risk Engineering, ETH Zurich

    Class to define the subproblems of the Benders Decomposition method.
    This class is a child class of the OptimizationSetup class and inherits all the methods and attributes of the
    parent class.
    A subproblem is defined as the operational problem and includes only the operational variables and constraints.
    In order to ensure only the presence of the operational constraints, this class removes the design constraints and
    the not coupling variables from the subproblem.
"""

import logging
import os
import time

from zen_garden.model.optimization_setup import OptimizationSetup


class Subproblem(OptimizationSetup):
    """
    Class defining the Subproblem of the Benders Decomposition method.
    """

    label = "Subproblem"

    def __init__(
        self,
        config: dict,
        config_benders: dict,
        solver: dict,
        analysis: dict,
        monolithic_model: OptimizationSetup,
        master_model: OptimizationSetup,
        model_name: str,
        scenario_name: str,
        scenario_dict: dict,
        input_data_checks,
        not_coupling_variables,
        design_constraints,
        benders_output_folder,
    ):
        """
        Initialize the Subproblem object.

        :param config: dictionary containing the configuration of the optimization problem
        :param config_benders: dictionary containing the configuration of the Benders Decomposition method
        :param analysis: dictionary containing the analysis configuration
        :param monolithic_model: OptimizationSetup object of the monolithic problem
        :param model_name: name of the model
        :param scenario_name: name of the scenario
        :param scenario_dict: dictionary containing the scenario data
        :param input_data_checks: dictionary containing the input data checks
        :param not_coupling_variables: list of not coupling variables
        :param design_constraints: list of design constraints
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

        self.name = "Subproblem"

        self.config = config
        self.config_benders = config_benders
        self.solver = solver
        self.analysis = analysis

        self.monolithic_model = monolithic_model
        self.master_model = master_model
        self.design_constraints = design_constraints
        self.not_coupling_variables = not_coupling_variables

        # Attributes from the monolithic problem needed to ensure robustness in case of solving Benders for MGA
        self.mga_weights = self.monolithic_model.mga_weights
        self.mga_objective_coords = self.monolithic_model.mga_objective_coords
        self.cost_optimal_mga = self.monolithic_model.cost_optimal_mga
        self.building_subproblem = True

        self.create_subproblem()

        self.folder_output = os.path.abspath(benders_output_folder + "/" + "subproblems")
        self.optimized_time_steps = [0]

        self.rhs_cuts = None
        self.lhs_cuts = None
        self.define_lhs_rhs_for_cuts()

    def create_subproblem(self):
        """
        Create the subproblem, which is the operational problem.
        It includes only the operational constraints and the objective function is taken from the config as follow:
        - If the objective function is "mga", we check whether we optimize for design or operational variables:
            - If "capacity" --> design: the objective function of the subproblem is a mock constant objective function
            - If "flow_import" --> operational: the objective function of the subproblem is the same as the one of the
            monolithic problem
            - If the objective function is "total_cost", or is "total_carbon_emissions", the objective function of the
            subproblem the same as the one of the monolithic problem.
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
                if self.config.benders["subproblem_objective"]:
                    self.model.add_objective(self.monolithic_model.model.objective.expression, overwrite=True)
                else:
                    self.variables.add_variable(
                        self.model,
                        name="mock_objective_subproblem",
                        index_sets=self.sets["set_time_steps_yearly"],
                        doc="mock variables for the objective of the subproblems",
                        unit_category={"time": -1},
                        bounds=(0, 0),
                    )
                    self.model.add_objective(
                        sum(
                            [
                                self.model.variables["mock_objective_subproblem"][year]
                                for year in self.energy_system.set_time_steps_yearly
                            ]
                        ).to_linexpr(),
                        overwrite=True,
                    )
            elif "flow_import" in str(self.monolithic_model.model.objective):
                self.model.add_objective(self.monolithic_model.model.objective.expression, overwrite=True)
            else:
                raise AssertionError("Objective function not recognized for MGA.")
        elif self.analysis["objective"] == "total_cost" or self.analysis["objective"] == "total_carbon_emissions":
            self.model.add_objective(self.monolithic_model.model.objective.expression, overwrite=True)
        else:
            logging.error(
                "Objective function %s not supported for Benders Decomposition at the moment.",
                self.config.analysis["objective"],
            )

        # Remove the design constraints and not coupling variables from the subproblem
        logging.info("--- Removing design constraints from the subproblem ---")
        for design_constraint in self.design_constraints:
            self.model.constraints.remove(design_constraint)
        logging.info("--- Removing not coupling design variables from the subproblem ---")
        for not_coupling_variable in self.not_coupling_variables:
            self.model.variables.remove(not_coupling_variable)

    def get_linopy_variable(self, row):
        """
        Helper function to get the corresponding Linopy variable of a constraint.
        """
        label_position = self.master_model.model.variables.get_label_position(row["labels"])
        return self.master_model.model.variables[f"{label_position[0]}"].sel(label_position[1])

    def define_lhs_rhs_for_cuts(self):
        """
        Define the left-hand side and right-hand side of the constraints for the Benders cuts.
        These are shared between the feasibility and optimality cuts and remain always the same. What changes are the
        multipliers of the cuts.
        """
        logging.info("--- Defining the right-hand side of the constraints for the Benders cuts ---")
        start_time = time.time()
        constraints = self.model.constraints.flat
        constraints.rename(columns={"labels": "labels_con", "vars": "labels"}, inplace=True)

        # Define the right-hand side of the constraints for the Benders cuts
        self.rhs_cuts = constraints[["labels_con", "rhs"]].drop_duplicates(subset="labels_con", keep="first")
        end_time = time.time()
        logging.info("--- Done in %.2f seconds ---", (end_time - start_time))

        logging.info("--- Defining the left-hand side of the constraints for the Benders cuts ---")
        start_time = time.time()
        # Define the left-hand side of the constraints for the Benders cuts, including only the variables that are
        # shared between the master and subproblems, so the design variables
        shared_labels = self.master_model.model.variables.flat[["labels"]]
        self.lhs_cuts = constraints.merge(shared_labels, on="labels")

        # Adding also the corresponding master variable of the Linopy model
        self.lhs_cuts["master_variable"] = self.lhs_cuts.apply(self.get_linopy_variable, axis=1)
        end_time = time.time()
        logging.info("--- Done in %.2f seconds ---", (end_time - start_time))

    def change_objective_to_constant(self):
        """
        Change the objective function of the subproblem to a constant value.
        This is needed when the objective function of the subproblem is a mock variable.
        """
        if self.analysis["objective"] == "mga":
            if "capacity" in str(self.monolithic_model.model.objective):
                if self.config.benders["subproblem_objective"]:
                    self.variables.add_variable(
                        self.model,
                        name="mock_objective_subproblem",
                        index_sets=self.sets["set_time_steps_yearly"],
                        doc="mock variables for the objective of the subproblems",
                        unit_category={"time": -1},
                        bounds=(0, 0),
                    )
                    self.model.add_objective(
                        sum(
                            [
                                self.model.variables["mock_objective_subproblem"][year]
                                for year in self.energy_system.set_time_steps_yearly
                            ]
                        ).to_linexpr(),
                        overwrite=True,
                    )
