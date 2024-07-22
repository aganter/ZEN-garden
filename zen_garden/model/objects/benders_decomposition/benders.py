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
import os
import time
import pandas as pd
from pathlib import Path
import psutil
from gurobipy import GRB

from zen_garden.preprocess.extract_input_data import DataInput
from zen_garden.model.optimization_setup import OptimizationSetup
from zen_garden.model.objects.benders_decomposition.master_problem import MasterProblem
from zen_garden.model.objects.benders_decomposition.subproblems import Subproblem
from zen_garden.utils import StringUtils, ScenarioUtils

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
        monolithic_problem: OptimizationSetup,
        scenario_name: str = None,
    ):
        """
        Initialize the BendersDecomposition object.

        :param config: dictionary containing the configuration of the optimization problem
        :param analysis: dictionary containing the analysis configuration
        :param monolithic_problem: OptimizationSetup object of the monolithic problem
        :param scenario_name: name of the scenario
        """

        self.name = "BendersDecomposition"
        self.config = config
        self.analysis = analysis
        self.monolithic_problem = monolithic_problem

        self.input_path = getattr(self.config.benders, "input_path")
        self.energy_system = monolithic_problem.energy_system
        self.data_input = DataInput(
            element=self,
            system=self.config.system,
            analysis=self.config.analysis,
            solver=self.config.solver,
            energy_system=self.energy_system,
            unit_handling=None,
        )
        self.config.benders.system.update(self.config.system)
        self.config.benders.system.update(self.config.benders.immutable_system_elements)
        self.config.benders.analysis["dataset"] = os.path.abspath(self.config.benders.analysis["dataset"])
        self.benders_output_folder = StringUtils.get_output_folder(
            analysis=self.config.benders.analysis,
            system=self.config.benders.system,
            folder_output=self.config.benders.analysis.folder_output,
        )
        self.benders_output_folder = os.path.abspath(self.benders_output_folder) + "/" + scenario_name

        scenarios, elements = ScenarioUtils.get_scenarios(
            config=self.config.benders, scenario_script_name="benders_scenarios.py", job_index=None
        )
        ScenarioUtils.clean_scenario_folder(self.config.benders, self.benders_output_folder)

        self.monolithic_constraints = self.monolithic_problem.model.constraints
        self.monolithic_variables = self.monolithic_problem.model.variables
        self.design_constraints, self.operational_constraints = self.separate_design_operational_constraints()
        self.design_variables, self.operational_variables, self.not_coupling_variables = (
            self.separate_design_operational_variables()
        )

        self.monolithic_model_gurobi = None
        self.map_variables_monolithic_gurobi = {}
        self.map_constraints_monolithic_gurobi = {}

        self.save_monolithic_problem_in_gurobi_format_map_vars_constrs()

        # DataFrames to store information about the building and solving of the subproblems
        columns = ["subproblem", "build_time_sec", "build_memory_MB"]
        self.building_subproblem = pd.DataFrame(columns=columns)
        columns = ["subproblem", "iteration", "solve_time_sec", "solve_memory_MB"]
        self.solving_subproblem = pd.DataFrame(columns=columns)
        # DataFrames to store information about the building and solving of the master problem
        columns = ["iteration", "optimality_gap"]
        self.optimality_gap_df = pd.DataFrame(columns=columns)

        logging.info("")
        logging.info("--- Creating the master problem ---")
        self.master_model = MasterProblem(
            config=self.monolithic_problem.config,
            config_benders=self.config.benders,
            analysis=self.analysis,
            monolithic_problem=self.monolithic_problem,
            model_name=self.monolithic_problem.model_name,
            scenario_name=self.monolithic_problem.scenario_name,
            scenario_dict=self.monolithic_problem.scenario_dict,
            input_data_checks=self.monolithic_problem.input_data_checks,
            operational_variables=self.operational_variables,
            operational_constraints=self.operational_constraints,
            benders_output_folder=self.benders_output_folder,
        )

        logging.info("")
        logging.info("--- Creating the subproblems ---")
        self.subproblem_models = []
        for scenario, scenario_dict in zip(scenarios, elements):
            start_time = time.time()
            pid = os.getpid()
            subproblem = Subproblem(
                config=self.monolithic_problem.config,
                config_benders=self.config.benders,
                analysis=self.analysis,
                monolithic_problem=self.monolithic_problem,
                model_name=self.monolithic_problem.model_name,
                scenario_name=scenario,
                scenario_dict=scenario_dict,
                input_data_checks=self.monolithic_problem.input_data_checks,
                not_coupling_variables=self.not_coupling_variables,
                design_constraints=self.design_constraints,
                benders_output_folder=self.benders_output_folder,
            )
            self.subproblem_models.append(subproblem)
            build_time = time.time() - start_time
            build_memory = psutil.Process(pid).memory_info().rss / 1024**2
            if scenario == "":
                scenario = "default"
            new_row = pd.DataFrame(
                {"subproblem": [scenario], "build_time_sec": [build_time], "build_memory_MB": [build_memory]}
            )
            self.building_subproblem = pd.concat([self.building_subproblem, new_row], ignore_index=True)

    def save_monolithic_problem_in_gurobi_format_map_vars_constrs(self):
        """
        Save the monolithic problem in the gurobi format.
        """
        self.monolithic_model_gurobi = self.monolithic_problem.model.to_gurobipy()

        # Map variables name in the monolithic problem to gurobi variables name
        # The map will be a dictionary with key the name of the variable in the gurobi modle and three values:
        # 1. variable_name: the variable name in the monolithic problem
        # 2. variable_coords: the coordinates of the variable in the monolithic problem
        # 3. variable_gurobi: the variable in the gurobi model
        for i in range(self.monolithic_model_gurobi.NumVars):
            variable_gurobi = self.monolithic_model_gurobi.getVars()[i]
            key = variable_gurobi.VarName
            variable_name = self.monolithic_problem.model.variables.get_label_position(int(key[1:]))[0]
            variable_coords = self.monolithic_problem.model.variables.get_label_position(int(key[1:]))[1]
            self.map_variables_monolithic_gurobi[key] = {
                "variable_name": variable_name,
                "variable_coords": variable_coords,
                "variable_gurobi": variable_gurobi,
            }

        # Map constraints name in the monolithic problem to gurobi constraints name
        # The map will be a dictionary with key the name of the constraint in the gurobi model and three values:
        # 1. constraint_name: the constraint name in the monolithic problem
        # 2. constraint_coords: the coordinates of the constraint in the monolithic problem
        # 3. constraint_gurobi: the constraint in the gurobi model
        for i in range(self.monolithic_model_gurobi.NumConstrs):
            constraint_gurobi = self.monolithic_model_gurobi.getConstrs()[i]
            key = constraint_gurobi.ConstrName
            constraint_name = self.monolithic_problem.model.constraints.get_label_position(int(key[1:]))[0]
            constraint_coords = self.monolithic_problem.model.constraints.get_label_position(int(key[1:]))[1]
            self.map_constraints_monolithic_gurobi[key] = {
                "constraint_name": constraint_name,
                "constraint_coords": constraint_coords,
                "constraint_gurobi": constraint_gurobi,
            }

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
        not_subproblem_variables = self.data_input.read_input_csv("not_coupling_variables")
        design_variables = []
        operational_variables = []
        not_coupling_variables = []

        # The benders_variables is a dataframe with columns: variable_name and variable_type
        for _, variable in benders_variables.iterrows():
            if variable["variable_name"] in self.monolithic_variables:
                if variable["variable_type"] == "design":
                    design_variables.append(variable["variable_name"])
                elif variable["variable_type"] == "operational":
                    operational_variables.append(variable["variable_name"])
                else:
                    raise AssertionError(f"Constraint {variable['variable_name']} has an invalid type.")

        # At the end we need to ensure we have added all the variables of the monolithic problem
        if len(design_variables) + len(operational_variables) != len(self.monolithic_variables):
            missing_variables = set(self.monolithic_variables) - set(design_variables + operational_variables)
            raise AssertionError(
                f"The following variables are missing in the benders decomposition: {missing_variables}"
            )

        for _, variable in not_subproblem_variables.iterrows():
            if variable["variable_name"] in self.monolithic_variables:
                if variable["variable_type"] == "design":
                    not_coupling_variables.append(variable["variable_name"])
                else:
                    raise AssertionError(f"The variable {variable['variable_name']} must be a design variable.")

        return design_variables, operational_variables, not_coupling_variables

    def solve_master_problem(self, iteration):
        """
        Solve the master problem.
        """
        self.master_model.model.solve()
        optimality_gap = self.monolithic_problem.model.objective_value - self.master_model.model.objective_value
        new_row = pd.DataFrame({"iteration": [iteration], "optimality_gap": [optimality_gap]})
        self.optimality_gap_df = pd.concat([self.optimality_gap_df, new_row], ignore_index=True)

    def solve_subproblems(self, iteration):
        """
        Solve the subproblems given in the list self.subproblem_models.
        """
        for subproblem in self.subproblem_models:
            start_time = time.time()
            pid = os.getpid()

            subproblem.model.solve()

            solve_time = time.time() - start_time
            solve_memory = psutil.Process(pid).memory_info().rss / 1024**2
            subproblem_name = subproblem.scenario_name
            if subproblem_name == "":
                subproblem_name = "default"

            new_row = pd.DataFrame(
                {
                    "subproblem": [subproblem_name],
                    "iteration": [iteration],
                    "solve_time_sec": [solve_time],
                    "solve_memory_MB": [solve_memory],
                }
            )
            self.solving_subproblem = pd.concat([self.solving_subproblem, new_row], ignore_index=True)

    def fix_design_variables_in_subproblem_model(self):
        """
        Fix the design variables of the subproblems to the optimal solution of the master problem.
        This function takes the solution of the master problem and fixes the values of the design variables in the
        subproblems by adding the corresponding upper and lower bounds to the variables.
        """

        for variable_name in self.master_model.model.variables:
            for subproblem in self.subproblem_models:
                if variable_name in subproblem.model.variables:
                    variable_solution = self.master_model.model.solution[variable_name]
                    subproblem.model.variables[variable_name].lower = variable_solution
                    subproblem.model.variables[variable_name].upper = variable_solution

    def subproblem_to_gurobi(self, subproblem_solved):
        """
        Convert the subproblem model to gurobi, necessary to generate the feasibility cut.
        """
        subproblem_model_fixed_design_variable_gurobi = subproblem_solved.to_gurobipy()
        subproblem_model_fixed_design_variable_gurobi.setParam(GRB.Param.OutputFlag, 0)
        subproblem_model_fixed_design_variable_gurobi.setParam(GRB.Param.FeasibilityTol, 1e-9)
        subproblem_model_fixed_design_variable_gurobi.setParam(GRB.Param.InfUnbdInfo, 1)
        subproblem_model_fixed_design_variable_gurobi.setParam(GRB.Param.NumericFocus, 3)

        # Optimize gurobi model, compute IIS and save infeasible constraints
        subproblem_model_fixed_design_variable_gurobi.optimize()
        subproblem_model_fixed_design_variable_gurobi.computeIIS()

        return subproblem_model_fixed_design_variable_gurobi

    def generate_feasibility_cut(self, subproblem_model_fixed_design_variable_gurobi):
        """
        Generate the feasibility cut.
        """
        farkas_multipliers = [
            (constraint_name, multiplier)
            for constraint_name, multiplier, infeas in zip(
                [constr_name for constr_name in subproblem_model_fixed_design_variable_gurobi.getConstrs()],
                subproblem_model_fixed_design_variable_gurobi.getAttr(GRB.Attr.FarkasDual),
                subproblem_model_fixed_design_variable_gurobi.getAttr(GRB.Attr.IISConstr),
            )
            if infeas
        ]

        # Create the feasibility cut
        feasibility_cut_lhs = 0
        feasibility_cut_rhs = 0
        for gurobi_constr, farkas in farkas_multipliers:
            rhs = gurobi_constr.RHS
            feasibility_cut_rhs += farkas * rhs

            lhs = subproblem_model_fixed_design_variable_gurobi.getRow(gurobi_constr)
            for i in range(lhs.size()):
                var = lhs.getVar(i)
                subproblem_var = self.map_variables_monolithic_gurobi[var.VarName]
                subproblem_var_name = subproblem_var["variable_name"]
                subproblem_var_coords = subproblem_var["variable_coords"]
                coeff = lhs.getCoeff(i)
                if subproblem_var_name in self.master_model.model.variables:
                    master_variable = self.master_model.model.variables[subproblem_var_name].sel(subproblem_var_coords)
                    feasibility_cut_lhs += farkas * (master_variable * coeff)

        return feasibility_cut_lhs, feasibility_cut_rhs

    def define_list_of_feasibility_cuts(self):
        """
        Define the list of feasibility cuts when dealing with multiple subproblems.

        :return: list of feasibility cuts as tuples with the following structure:
            - subproblem_name: name of the subproblem (if the scenario name is not defined, it is set to "default")
            - feasibility_cut_lhs: left-hand side of the feasibility cut
            - feasibility_cut_rhs: right-hand side of the feasibility cut
        """
        logging.info("--- Generating feasibility cut ---")
        feasibility_cuts = []
        infeasible_subproblems = []
        for subproblem in self.subproblem_models:
            if subproblem.model.termination_condition == "infeasible":
                infeasible_subproblems.append(subproblem)
        for subproblem in infeasible_subproblems:
            subproblem_model_fixed_design_variable_gurobi = self.subproblem_to_gurobi(subproblem.model)

            feasibility_cut_lhs, feasibility_cut_rhs = self.generate_feasibility_cut(
                subproblem_model_fixed_design_variable_gurobi
            )
            name = subproblem.scenario_name if not subproblem.scenario_name == "" else "default"
            feasibility_cuts.append([name, feasibility_cut_lhs, feasibility_cut_rhs])

        return feasibility_cuts

    def add_feasibility_cuts_to_master(self, feasibility_cuts, iteration):
        """
        Add the feasibility cuts to the master model.

        :param feasibility_cuts: list of feasibility cuts as tuples with the following structure:
            - subproblem_name: name of the subproblem
            - feasibility_cut_lhs: left-hand side of the feasibility cut
            - feasibility_cut_rhs: right-hand side of the feasibility cut
        :param iteration: current iteration of the Benders Decomposition method
        """
        logging.info("--- Adding feasibility cut to master model")
        for subproblem_name, feasibility_cut_lhs, feasibility_cut_rhs in feasibility_cuts:
            self.master_model.model.add_constraints(
                lhs=feasibility_cut_lhs,
                sign="<=",
                rhs=feasibility_cut_rhs,
                name=f"feasibility_cuts_{subproblem_name}_iteration_{iteration}",
            )

    def save_master_and_subproblems(self):
        """
        Save the master model and the subproblem models in respective output folders.
        """
        logging.info("--- All the subproblem are optimal. Terminating iterations ---")
        Postprocess(model=self.master_model, scenarios="", model_name=self.master_model.model_name, subfolder=Path(""))

        for subproblem in self.subproblem_models:
            scenario_name_subproblem, subfolder_subproblem, param_map_subproblem = subproblem.generate_output_paths(
                config_system=self.config.benders.system, step=1, steps_horizon=[1]
            )
            Postprocess(
                model=subproblem,
                scenarios=self.config.benders.scenarios,
                model_name=subproblem.model_name,
                subfolder=subfolder_subproblem,
                scenario_name=scenario_name_subproblem,
                param_map=param_map_subproblem,
            )

        self.building_subproblem.to_csv(os.path.join(self.benders_output_folder, "building_subproblem.csv"))
        self.solving_subproblem.to_csv(os.path.join(self.benders_output_folder, "solving_subproblem.csv"))
        self.optimality_gap_df.to_csv(os.path.join(self.benders_output_folder, "optimality_gap.csv"))

    def fit(self):
        """
        Fit the Benders Decomposition model.
        """
        logger = logging.getLogger("gurobipy")
        logger.propagate = False

        iteration = 1
        while True:
            logging.info("")
            logging.info("")
            logging.info("--- Iteration %s ---", iteration)
            logging.info("--- Solving master problem, fixing design variables in subproblems and solve them ---")
            self.solve_master_problem(iteration)
            self.fix_design_variables_in_subproblem_model()
            self.solve_subproblems(iteration)

            if all(subproblem.model.termination_condition == "optimal" for subproblem in self.subproblem_models):
                self.save_master_and_subproblems()
                break

            logging.info("--- Subproblems are infeasible ---")
            feasibility_cuts = self.define_list_of_feasibility_cuts()
            self.add_feasibility_cuts_to_master(feasibility_cuts, iteration)

            iteration += 1
