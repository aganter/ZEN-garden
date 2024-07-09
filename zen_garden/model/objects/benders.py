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
import logging
import time
import numpy as np
import xarray as xr
import psutil
import math

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
        self.benders_constraints = self.data_input.read_input_csv("constraints")

    def separate_design_operational_constraints(self) -> list:
        """
        Separate the design and operational constraints based on the user input preferences defined in the config file.
        It also needs to check and maintain only the constraints that are in the monolithic problem.

        :return: design_constraints, operational_constraints (type: lists of strings)
        """
        design_constraints = []
        operational_constraints = []

        # The benders_constraints is a dataframe with columns: constraint_name and constraint_type
        for _, constraint in self.benders_constraints.iterrows():
            if constraint["constraint_name"] in self.monolithic_constraints:
                if constraint["constraint_type"] == "design":
                    design_constraints.append(constraint["constraint_name"])
                elif constraint["constraint_type"] == "operational":
                    operational_constraints.append(constraint["constraint_name"])
                else:
                    logging.error("Constraint %s has an invalid type.", constraint["constraint_name"])
            else:
                logging.error("Constraint %s is not in the monolithic problem.", constraint["constraint_name"])

        # At the end we need to ensure we have added all the constraints of the monolithic problem
        if len(design_constraints) + len(operational_constraints) != len(self.monolithic_constraints):
            missing_constraints = set(self.monolithic_constraints) - set(design_constraints + operational_constraints)
            logging.error("The following constraints are missing in the benders decomposition: %s", missing_constraints)

        return design_constraints, operational_constraints

    def create_master_problem(self):
        """
        Create the master problem, which is the design problem.
        It includes only the design constraints and has the same objective function as the monolithic problem.
        The design constraints are defined by the user in the config file.
        """
