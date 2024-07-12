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

from zen_garden.preprocess.extract_input_data import DataInput
from zen_garden.model.optimization_setup import OptimizationSetup
from zen_garden.model.objects.benders_decomposition.master_problem import MasterProblem
from zen_garden.model.objects.benders_decomposition.subproblems import Subproblem


class BendersDecomposition:
    """
    Class defining the Benders Decomposition method.
    Initialize the BendersDecomposition object.

    :param config_benders: dictionary containing the configuration of the Benders Decomposition method
    :param monolithic_problem: OptimizationSetup object of the monolithic problem
    """

    label = "BendersDecomposition"

    def __init__(
        self,
        config: dict,
        analysis: dict,
        config_benders: dict,
        monolithic_problem: OptimizationSetup,
    ):

        self.name = "BendersDecomposition"
        self.config = config
        self.analysis = analysis
        self.config_benders = config_benders
        self.monolithic_problem = monolithic_problem

        self.input_path = getattr(config_benders, "input_path")
        self.energy_system = monolithic_problem.energy_system
        self.data_input = DataInput(
            element=self,
            system=self.config.system,
            analysis=self.config.analysis,
            solver=self.config.solver,
            energy_system=self.energy_system,
            unit_handling=None,
        )

        self.monolithic_constraints = self.monolithic_problem.model.constraints
        self.monolithic_variables = self.monolithic_problem.model.variables
        self.design_constraints, self.operational_constraints = self.separate_design_operational_constraints()
        self.design_variables, self.operational_variables = self.separate_design_operational_variables()

        logging.info("Creating the master problem.")
        self.master_model = MasterProblem(
            config=self.monolithic_problem.config,
            analysis=self.analysis,
            monolithic_problem=self.monolithic_problem,
            model_name=self.monolithic_problem.model_name,
            scenario_name=self.monolithic_problem.scenario_name,
            scenario_dict=self.monolithic_problem.scenario_dict,
            input_data_checks=self.monolithic_problem.input_data_checks,
            design_variables=self.design_variables,
            operational_variables=self.operational_variables,
            design_constraints=self.design_constraints,
            operational_constraints=self.operational_constraints,
        )
        logging.info("Creating the subproblems.")
        self.subproblem_models = Subproblem(
            config=self.monolithic_problem.config,
            analysis=self.analysis,
            monolithic_problem=self.monolithic_problem,
            model_name=self.monolithic_problem.model_name,
            scenario_name=self.monolithic_problem.scenario_name,
            scenario_dict=self.monolithic_problem.scenario_dict,
            input_data_checks=self.monolithic_problem.input_data_checks,
            design_variables=self.design_variables,
            operational_variables=self.operational_variables,
            design_constraints=self.design_constraints,
            operational_constraints=self.operational_constraints,
        )

        self.solve_master_problem()

    def separate_design_operational_constraints(self) -> list:
        """
        Separate the design and operational constraints based on the user input preferences defined in the config file.
        It also needs to check and maintain only the constraints that are in the monolithic problem.

        :return: design_constraints, operational_constraints (type: lists of strings)
        """
        benders_constraints = self.data_input.read_input_csv("constraints")
        design_constraints = []
        operational_constraints = []

        # The benders_constraints is a dataframe with columns: constraint_name and constraint_type
        for _, constraint in benders_constraints.iterrows():
            if constraint["constraint_name"] in self.monolithic_constraints:
                if constraint["constraint_type"] == "design":
                    design_constraints.append(constraint["constraint_name"])
                elif constraint["constraint_type"] == "operational":
                    operational_constraints.append(constraint["constraint_name"])
                else:
                    raise AssertionError(f"Constraint {constraint['constraint_name']} has an invalid type.")
            else:
                logging.warning("Constraint %s is not in the monolithic problem.", constraint["constraint_name"])

        # At the end we need to ensure we have added all the constraints of the monolithic problem
        if len(design_constraints) + len(operational_constraints) != len(self.monolithic_constraints):
            missing_constraints = set(self.monolithic_constraints) - set(design_constraints + operational_constraints)
            raise AssertionError(
                f"The following constraints are missing in the benders decomposition: {missing_constraints}"
            )

        return design_constraints, operational_constraints

    def separate_design_operational_variables(self) -> list:
        """
        Separate the design and operational variables based on the user input preferences defined in the config file.

        :return: design_variables, operational_variables (type: lists of strings)
        """
        benders_variables = self.data_input.read_input_csv("variables")
        design_variables = []
        operational_variables = []

        # The benders_variables is a dataframe with columns: variable_name and variable_type
        for _, variable in benders_variables.iterrows():
            if variable["variable_name"] in self.monolithic_variables:
                if variable["variable_type"] == "design":
                    design_variables.append(variable["variable_name"])
                elif variable["variable_type"] == "operational":
                    operational_variables.append(variable["variable_name"])
                else:
                    raise AssertionError(f"Constraint {variable['variable_name']} has an invalid type.")
            else:
                logging.warning("Constraint %s is not in the monolithic problem.", variable["variable_name"])

        # At the end we need to ensure we have added all the variables of the monolithic problem
        if len(design_variables) + len(operational_variables) != len(self.monolithic_variables):
            missing_variables = set(self.monolithic_variables) - set(design_variables + operational_variables)
            raise AssertionError(
                f"The following variables are missing in the benders decomposition: {missing_variables}"
            )

        return design_variables, operational_variables

    def solve_master_problem(self):
        """
        Solve the master problem.
        """
        self.master_model.solve()

    def fix_subproblem_design_variables(self):
        """
        Fix the design variables of the subproblems to the optimal solution of the master problem.
        This function takes the solution of the master problem and fixes the values of the design variables in the
        subproblems by adding the corresponding constraints.
        """
        # Find the solution of the master problem, this include all the design variables we need  into Data variables
        solution_master = self.master_model.model.solution[self.design_variables]
        # We need to retrieve the design variables of the subproblems
        design_variables_subproblems = self.subproblem_models.model.variables[self.design_variables]
        for design_variable in self.design_variables:
            lhs = design_variables_subproblems[design_variable]
            rhs = solution_master[design_variable]
            constraint = lhs == rhs
            self.subproblem_models.constraints.add_constraint(
                f"constraint_design_variable_{design_variable}", constraint
            )

    def optimality_cuts(self):
        """
        Generate the optimality cuts.
        When the subproblem is feasible but we do not have the optimal solution, we need to add the optimality cuts
        to the master problem.
        This function generates the optimality cuts based on the solution of the subproblems as follow:


        """
        pass

    def feasibility_cuts(self):
        """
        Generate the feasibility cuts.
        When the subproblem is infeasible, we need to add the feasibility cuts to the master problem.
        This function generates the feasibility cuts based on the solution of the subproblems as follow:


        """
        pass

    # def benders_decomposition(self):
    #     """
    #     Run the Benders Decomposition method.
    #     """
    #     ABSOLUTE_OPTIMALITY_GAP = 1e-6
    #     for k in range(self.config_benders["maximum_iterations"]):
    #         self.master_model.solve()
    #         lower_bound = self.master_model
    #         self.fix_subproblem_design_variables()
    #         self.subproblem_models.solve()
    #         if self.subproblem_models.model.status == "infeasible":
    #             self.feasibility_cuts()
    #         elif self.subproblem_models.model.status == "optimal":
    #             self.optimality_cuts()
    #             upper_bound = self.subproblem_models + self.master_model
    #             gap = (upper_bound - lower_bound) / upper_bound
    #             logging.info(f"{k+1:9} {lower_bound:12.4e} {upper_bound:12.4e} {gap:12.4e}")

    #         if gap < ABSOLUTE_OPTIMALITY_GAP:
    #             logging.info("Terminating with the optimal solution")
    #             break

    #     self.master_model.solve()
    #     objective_optimal = self.master_model

    #     return objective_optimal
