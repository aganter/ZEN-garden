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

from zen_garden.model.optimization_setup import OptimizationSetup


class MasterProblem(OptimizationSetup):
    """
    Class defining the Master Problem of the Benders Decomposition method.
    Initialize the MasterProblem object.
    """

    label = "MasterProblem"

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
        operational_variables,
        operational_constraints,
        benders_output_folder,
    ):
        """
        Initialize the MasterProblem object.

        :param config: dictionary containing the configuration of the optimization problem
        :param config_benders: dictionary containing the configuration of the Benders Decomposition method
        :param analysis: dictionary containing the analysis configuration
        :param monolithic_problem: OptimizationSetup object of the monolithic problem
        :param model_name: name of the model
        :param scenario_name: name of the scenario
        :param scenario_dict: dictionary containing the scenario data
        :param input_data_checks: dictionary containing the input data checks
        :param operational_variables: list of operational variables
        :param operational_constraints: list of operational constraints
        """

        super().__init__(
            config=config,
            model_name=model_name,
            scenario_name=scenario_name,
            scenario_dict=scenario_dict,
            input_data_checks=input_data_checks,
        )

        self.name = "MasterProblem"
        self.config = config
        self.config_benders = config_benders
        self.analysis = analysis
        self.monolithic_problem = monolithic_problem
        self.operational_variables = operational_variables
        self.operational_constraints = operational_constraints
        self.mga_weights = self.monolithic_problem.mga_weights
        self.mga_objective_coords = self.monolithic_problem.mga_objective_coords

        self.master_model_gurobi = None

        self.create_master_problem()

        self.folder_output = os.path.abspath(benders_output_folder + "/" + "master_problem")
        self.optimized_time_steps = [0]

    def add_theta_variable(self, model, name):
        """
        Add a the outer approximation of the subproblem objective function to the master problem.
        """
        theta = model.add_variables(lower=0, name=name)
        return theta

    def create_master_problem(self):
        """
        Create the master problem, which is the design problem.
        It includes only the design constraints and the objective function is taken from the config as follow:
        - If the objective function is "mga", we check whether we optimize for design or operational variables:
            - If design, the objective function of the master problem is the same as the one of the monolithic problem
            - If operational, the objective function of the master problem is a dummy constant objective function
         TODO: Add the possibility to use Benders also when optimize for "total_cost" and "total_carbon_emissions", in
        the future also for "risk"
        - If the objective function is "total_cost", this is splitted into design and operational costs and the
        objective function of the master problem includes only the design costs (capex).
        - If the obejctive function is "total_carbon_emissions", the objective function of the master problem is a
        dummy constant objective function
        """
        self.construct_optimization_problem()
        mga = "modeling_to_generate_alternatives"
        if mga in self.config and self.config[mga]:
            self.model.add_constraints(
                lhs=self.model.variables.net_present_cost.sum(dim="set_time_steps_yearly"),
                sign="<=",
                rhs=self.monolithic_problem.model.constraints.constraint_optimal_cost_total_deviation.rhs,
                name="constraint_optimal_cost_total_deviation",
            )

        # Define the objective function
        if self.analysis["objective"] == "mga":
            if "capacity" in str(self.monolithic_problem.model.objective):
                self.model.add_objective(self.monolithic_problem.model.objective.expression, overwrite=True)
            # If the objective function optimizes for operational variables, we use a dummy objective in the master
            elif "flow_import" in str(self.monolithic_problem.model.objective):
                theta = self.add_theta_variable(self.model, name="theta_approximation_subproblem_objective")
                self.model.add_objective(1 * theta, overwrite=True)
                self.model.add_constraints(lhs=theta, sign="<=", rhs=1e6, name="theta_limit")
            else:
                raise AssertionError("Objective function not recognized for MGA.")
        else:
            logging.error(
                "Objective function %s not supported for Benders Decomposition at the moment.",
                self.config.analysis["objective"],
            )

        # Remove the operational variables and constraints from the master problem
        logging.info("Removing operational constraints from the master problem.")
        self.model.remove_constraints(self.operational_constraints)
        logging.info("Removing operational variables from the master problem.")
        for operational_variable in self.operational_variables:
            self.model.remove_variables(operational_variable)
