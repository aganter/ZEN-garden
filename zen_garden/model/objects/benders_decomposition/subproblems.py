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

from zen_garden.model.optimization_setup import OptimizationSetup


class Subproblem(OptimizationSetup):
    """
    Class defining the Subproblem of the Benders Decomposition method.
    """

    label = "MasterProblem"

    def __init__(
        self,
        config: dict,
        analysis: dict,
        monolithic_problem: OptimizationSetup,
        model_name: str,
        scenario_name: str,
        scenario_dict: dict,
        input_data_checks,
        design_variables,
        operational_variables,
        design_constraints,
        operational_constraints,
    ):
        """
        Initialize the Subproblem object.

        :param config: dictionary containing the configuration of the optimization problem
        :param analysis: dictionary containing the analysis configuration
        :param monolithic_problem: OptimizationSetup object of the monolithic problem
        :param model_name: name of the model
        :param scenario_name: name of the scenario
        :param scenario_dict: dictionary containing the scenario data
        :param input_data_checks: dictionary containing the input data checks
        :param design_variables: list of design variables
        :param operational_variables: list of operational variables
        :param design_constraints: list of design constraints
        :param operational_constraints: list of operational constraints
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
        self.analysis = analysis
        self.monolithic_problem = monolithic_problem
        self.design_variables = design_variables
        self.design_constraints = design_constraints
        self.operational_variables = operational_variables
        self.operational_constraints = operational_constraints
        self.mga_weights = self.monolithic_problem.mga_weights
        self.mga_objective_coords = self.monolithic_problem.mga_objective_coords

        self.create_subproblem()

    def add_dummy__constant_variable(self, model, name="dummy_variable"):
        """
        Add a dummy variable to the subproblem problem.
        """
        dummy_variable = model.add_variables(lower=1, upper=1, name=name, integer=True)
        return dummy_variable

    def create_subproblem(self):
        """
        Create the subproblem, which is the operational problem.
        It includes only the operational constraints and the objective function is taken from the config as follow:
        - If the objective function is "mga", we check whether we optimize for design or operational variables:
            - If design, the objective function of the subproblem is a dummy constant objective function
            - If operational, the objective function of the subproblem is the same as the one of the monolithic problem
        TODO: Add the possibility to use Benders also when optimize for "total_cost" and "total_carbon_emissions", in
        the future also for "risk"
        - If the objective function is "total_cost", this is splitted into design and operational costs and the
        objective function of the subproblem includes only the operational costs (opex and emissions costs).
        - If the obejctive function is "total_carbon_emissions", the objective function of the subproblem the same as
        the one of the monolithic problem.
        """
        self.construct_optimization_problem()
        mga = "modeling_to_generate_alternatives"
        if mga in self.config and self.config[mga]:
            self.model.constraints.add(
                self.monolithic_problem.model.constraints["constraint_optimal_cost_total_deviation"]
            )

        # Define the objective function
        if self.analysis["objective"] == "mga":
            # If the objective function optimizes for design variables, we use a dummy objective in the subproblem
            if "capacity" in str(self.monolithic_problem.model.objective):
                dummy_variable = self.add_dummy__constant_variable(self.model, name="dummy_subproblem_variable")
                self.model.add_objective(1 * dummy_variable, overwrite=True)
            elif "flow_import" in str(self.monolithic_problem.model.objective):
                self.model.add_objective(self.monolithic_problem.model.objective.expression, overwrite=True)
            else:
                raise AssertionError("Objective function not recognized for MGA.")
        else:
            logging.error(
                "Objective function %s not supported for Benders Decomposition at the moment.",
                self.config.analysis["objective"],
            )

        self.model
