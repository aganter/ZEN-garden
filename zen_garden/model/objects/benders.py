"""
  :Title:        ZEN-GARDEN
  :Created:      July-2024
  :Authors:      Maddalena Cenedese (mcenedese@student.ethz.ch)
  :Organization: Labratory of Reliability and Risk Engineering, ETH Zurich

    Class to decompose the optimization problem into a MASTER problem, which is the design problem and a set of
    SUBPROBLEMS, which are the operational problems with different set of uncertaint parameters. 
    The class is used to define the different benders decomposition method.
"""

import os
import copy
import logging
import time
import numpy as np
import xarray as xr
import psutil
import math
import linopy as lp

from zen_garden.preprocess.extract_input_data import DataInput
from zen_garden.model.optimization_setup import OptimizationSetup
from zen_garden.utils import InputDataChecks
from zen_garden.utils import StringUtils
from zen_garden.postprocess.postprocess import Postprocess


class BendersDecomposition:
    """
    Class defining the Benders Decomposition method.
    """

    label = "BendersDecomposition"

    def __init__(
        self,
        config: dict,
        analysis: dict,
        config_benders: dict,
        monolithic_problem: OptimizationSetup,
    ):
        """
        Initialize the BendersDecomposition object.

        :param config_benders: dictionary containing the configuration of the Benders Decomposition method
        :param monolithic_problem: OptimizationSetup object of the monolithic problem
        """
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
        self.design_constraints, self.operational_constraints = self.separate_design_operational_constraints()

        self.master_model = None
        self.master_sets = None
        self.master_parameters = None

        self.subproblem_models = None

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
        # Create a copy of the monolithic problem for robustness
        if self.config.solver["solver_dir"] is not None and not os.path.exists(self.config.solver["solver_dir"]):
            os.makedirs(self.config.solver["solver_dir"])
        self.master_model = lp.Model(solver_dir=self.config.solver["solver_dir"])

        # Define the objective function
        if self.analysis["objective"] == "mga":
            if "capacity" in str(self.monolithic_problem.model.objective):
                self.master_model.add_objective(self.monolithic_problem.model.objective.expression, overwrite=True)
            elif "flow_import" in str(self.monolithic_problem.model.objective):
                dummy_variables = self.master_model.add_variables(
                    lower=1, upper=1, name="dummy_master_variable", integer=True
                )
                self.master_model.add_objective(lp.LinearExpression(dummy_variables, self.master_model), overwrite=True)
            else:
                raise AssertionError("Objective function not recognized for MGA.")
        else:
            logging.error(
                "Objective function %s not supported for Benders Decomposition at the moment.",
                self.config.analysis["objective"],
            )

        # Add the monolithic variables to the master problem
        for variable in self.monolithic_problem.model.variables:
            self.master_model.variables.add(self.monolithic_problem.model.variables[variable])

        # Add the design constraints to the master problem
        for constraint in self.monolithic_problem.model.constraints:
            if constraint in self.design_constraints:
                self.master_model.constraints.add(self.monolithic_problem.model.constraints[constraint])

        return self.master_model
