"""
  :Title:        ZEN-GARDEN
  :Created:      July-2024
  :Authors:      Maddalena Cenedese (mcenedese@student.ethz.ch)
  :Organization: Labratory of Reliability and Risk Engineering, ETH Zurich

    Class to decompose the optimization problem into a MASTER problem, which is the design problem and a set of
    SUBPROBLEMS, which are the operational problems with different set of uncertaint parameters.
    The class is used to define the different benders decomposition method.
"""

import logging

from zen_garden.model.optimization_setup import OptimizationSetup


class MasterProblem(OptimizationSetup):
    """
    Class defining the Benders Decomposition method.
    Initialize the BendersDecomposition object.

    :param config_benders: dictionary containing the configuration of the Benders Decomposition method
    :param monolithic_problem: OptimizationSetup object of the monolithic problem
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
        super().__init__(
            config=config,
            model_name=model_name,
            scenario_name=scenario_name,
            scenario_dict=scenario_dict,
            input_data_checks=input_data_checks,
        )

        self.name = "MasterProblem"
        self.config = config
        self.analysis = analysis
        self.monolithic_problem = monolithic_problem
        self.design_variables = design_variables
        self.design_constraints = design_constraints
        self.operational_variables = operational_variables
        self.operational_constraints = operational_constraints
        self.mga_weights = self.monolithic_problem.mga_weights
        self.mga_objective_coords = self.monolithic_problem.mga_objective_coords

        self.create_master_problem()

    def add_dummy__constant_variable(self, model, name="dummy_variable"):
        """
        Add a dummy variable to the master problem.
        """
        dummy_variable = model.add_variables(lower=1, upper=1, name=name, integer=True)
        return dummy_variable

    def create_master_problem(self):
        """
        Create the master problem, which is the design problem.
        It includes only the design constraints and the objective function is taken from the config as follow:
        - If the objective function is "mga", we check whether we optimize for design or operational variables:
            - If design, we use the same objective function as the monolithic problem
            - If operational, in the master problem we use a dummy constant objective function
        TODO: Add the possibility to use Benders also when optimize for "total_cost" and "total_carbon_emissions", in
        the future also for "risk"
        - If the objective function is "total_cost", we split the objective function into design and operational costs
        and in the master problem we only include the design costs.
        - If the obejctive function is "total_carbon_emissions", in the master problem we use a dummy constant
        objective function
        """
        self.construct_optimization_problem()
        mga = "modeling_to_generate_alternatives"
        if mga in self.config and self.config[mga]:
            self.model.constraints.add(
                self.monolithic_problem.model.constraints["constraint_optimal_cost_total_deviation"]
            )

        # Define the objective function
        if self.analysis["objective"] == "mga":
            if "capacity" in str(self.monolithic_problem.model.objective):
                self.model.add_objective(self.monolithic_problem.model.objective.expression, overwrite=True)
            elif "flow_import" in str(self.monolithic_problem.model.objective):
                dummy_variable = self.add_dummy__constant_variable(self.model, name="dummy_master_variable")
                self.model.add_objective(1 * dummy_variable, overwrite=True)
            else:
                raise AssertionError("Objective function not recognized for MGA.")
        else:
            logging.error(
                "Objective function %s not supported for Benders Decomposition at the moment.",
                self.config.analysis["objective"],
            )

        # Romove the operational variables and constraints from the master problem
        self.model.remove_constraints(self.operational_constraints)
        for operational_variable in self.operational_variables:
            self.model.remove_variables(operational_variable)

        self.model.constraints
