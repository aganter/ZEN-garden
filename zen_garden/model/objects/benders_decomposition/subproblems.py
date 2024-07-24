"""
  :Title:        ZEN-GARDEN
  :Created:      July-2024
  :Authors:      Maddalena Cenedese (mcenedese@student.ethz.ch)
  :Organization: Labratory of Reliability and Risk Engineering, ETH Zurich

    Class to define the subproblems of the Benders Decomposition method.
    This class is a child class of the OptimizationSetup class and inherits all the methods and attributes of the
    parent class.
    A subproblem is defined as the operational problem under specific parametric uncertainties and includes only the
    operational variables and constraints.
"""

import logging
import os

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
        analysis: dict,
        monolithic_problem: OptimizationSetup,
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
        :param monolithic_problem: OptimizationSetup object of the monolithic problem
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
            model_name=model_name,
            scenario_name=scenario_name,
            scenario_dict=scenario_dict,
            input_data_checks=input_data_checks,
        )

        self.name = "Subproblem"

        self.config = config
        self.config_benders = config_benders
        self.analysis = analysis

        self.monolithic_problem = monolithic_problem
        self.design_constraints = design_constraints
        self.not_coupling_variables = not_coupling_variables

        self.mga_weights = self.monolithic_problem.mga_weights
        self.mga_objective_coords = self.monolithic_problem.mga_objective_coords

        self.create_subproblem()

        self.folder_output = os.path.abspath(benders_output_folder + "/" + "subproblems" + "/" + scenario_name)
        self.optimized_time_steps = [0]

    def create_subproblem(self):
        """
        Create the subproblem, which is the operational problem.
        It includes only the operational constraints and the objective function is taken from the config as follow:
        - If the objective function is "mga", we check whether we optimize for design or operational variables:
            - If design, the objective function of the subproblem is a dummy constant objective function
            - If operational, the objective function of the subproblem is the same as the one of the monolithic problem
            - If the objective function is "total_cost", or is "total_carbon_emissions", the objective function of the
            subproblem the same as the one of the monolithic problem.
        """
        self.construct_optimization_problem()
        if (
            str("modeling_to_generate_alternatives") in self.config
            and self.config[str("modeling_to_generate_alternatives")]
        ):
            self.model.add_constraints(
                lhs=self.model.variables.net_present_cost.sum(dim="set_time_steps_yearly"),
                sign="<=",
                rhs=self.monolithic_problem.model.constraints.constraint_optimal_cost_total_deviation.rhs,
                name="constraint_optimal_cost_total_deviation",
            )

        # Define the objective function
        if self.analysis["objective"] == "mga":
            # If the objective function optimizes for design variables, we use a dummy objective in the subproblem
            if "capacity" in str(self.monolithic_problem.model.objective):
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
            elif "flow_import" in str(self.monolithic_problem.model.objective):
                self.model.add_objective(self.monolithic_problem.model.objective.expression, overwrite=True)
            else:
                raise AssertionError("Objective function not recognized for MGA.")
        elif self.analysis["objective"] == "total_cost" or self.analysis["objective"] == "total_carbon_emissions":
            self.model.add_objective(self.monolithic_problem.model.objective.expression, overwrite=True)
        else:
            logging.error(
                "Objective function %s not supported for Benders Decomposition at the moment.",
                self.config.analysis["objective"],
            )

        # Remove the design constraints from the subproblem
        logging.info("Removing design constraints from the subproblem.")
        self.model.remove_constraints(self.design_constraints)
        logging.info("Removing not useful design variables from the subproblem.")
        for not_coupling_variable in self.not_coupling_variables:
            self.model.remove_variables(not_coupling_variable)
