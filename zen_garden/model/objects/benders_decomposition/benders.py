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

    def add_dummy__constant_variable(self, model, name="dummy_variable"):
        """
        Add a dummy variable to the master problem.
        """
        dummy_variable = model.add_variables(lower=1, upper=1, name=name, integer=True)
        return dummy_variable

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
