"""
  :Title:        ZEN-GARDEN
  :Created:      July-2024
  :Authors:      Maddalena Cenedese (mcenedese@student.ethz.ch)
  :Organization: Labratory of Reliability and Risk Engineering, ETH Zurich

    Class to decompose the optimization problem into a MASTER problem, which is the design problem and a set of
    SUBPROBLEMS, which are the operational problems with different set of parameters
    The class is able to solve the master problem and the subproblems iteratively by generating feasibility and
    optimality cuts until the termination criteria is met:
        - The termination criteria is based on the optimality gap between the master problem and the subproblems if
        the objective function of the master problem is the outer approximation of the one of the subproblems
        - The termination criteria is just to satify the feasibility of the subproblems if the objective function of the
        subproblems is a mock objective function
"""

import logging
import os
from pathlib import Path
import warnings
import time
import shutil
import pandas as pd
import numpy as np
import linopy as lp
import xarray as xr
from tabulate import tabulate
from gurobipy import GRB, Env

from zen_garden.preprocess.extract_input_data import DataInput
from zen_garden.model.optimization_setup import OptimizationSetup
from zen_garden.model.objects.benders_decomposition.master_problem import MasterProblem
from zen_garden.model.objects.benders_decomposition.subproblems import Subproblem
from zen_garden.utils import StringUtils, ScenarioUtils

from zen_garden.postprocess.postprocess import Postprocess

warnings.filterwarnings("ignore")


class BendersDecomposition:
    """
    Class defining the Benders Decomposition method.
    """

    label = "BendersDecomposition"

    def __init__(
        self,
        config: dict,
        analysis: dict,
        monolithic_model: OptimizationSetup,
        scenario_name: str = None,
        use_monolithic_solution: bool = False,
    ):
        """
        Initialize the Benders Decomposition method.

        :param config: dictionary containing the configuration of the optimization problem
        :param analysis: dictionary containing the analysis configuration
        :param monolithic_model: OptimizationSetup object of the monolithic problem
        :param scenario_name: name of the scenario
        :param use_monolithic_solution: boolean to use the solution of the monolithic problem as initial solution
        """

        self.name = "BendersDecomposition"
        self.config = config
        self.analysis = analysis
        self.monolithic_model = monolithic_model
        self.use_monolithic_solution = use_monolithic_solution

        # Define the input path where the design and operational variables and constraints are stored
        self.input_path = getattr(self.config.benders, "input_path")
        self.energy_system = monolithic_model.energy_system
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
        self.config.benders.solver_master.solver_dir = (
            os.path.abspath(self.benders_output_folder) + "/" + "master_solver"
        )
        # Define the different scenarios for the Benders Decomposition method
        scenarios, elements = ScenarioUtils.get_scenarios(
            config=self.config.benders, scenario_script_name="benders_scenarios", job_index=None
        )
        # Save the user defined FeasilibityTol of the subproblem solver
        self.feasibility_tol_subproblem = self.config.benders.solver_subproblem.solver_options["FeasibilityTol"]

        self.monolithic_constraints = self.monolithic_model.model.constraints
        self.monolithic_variables = self.monolithic_model.model.variables
        self.design_constraints, self.operational_constraints = self.separate_design_operational_constraints()
        self.design_variables, self.operational_variables, self.not_coupling_variables = (
            self.separate_design_operational_variables()
        )

        logger = logging.getLogger("gurobipy")
        logging.getLogger("gurobipy").setLevel(logging.ERROR)
        logger.propagate = False

        # DataFrame to store information about the building and solving of the master problem
        columns = ["iteration", "optimality_gap"]
        self.optimality_gap_df_infeasibility = pd.DataFrame(columns=columns)
        self.optimality_gap_df_optimal = pd.DataFrame(columns=columns)
        # Counters for feasibility and optimality cuts
        self.feasibility_cuts_counter = 0
        self.optimality_cuts_counter = 0
        columns = ["number_of_monolithic_constraints", "number_of_feasibility_cuts", "number_of_optimality_cuts"]
        self.cuts_counter_df = pd.DataFrame(columns=columns)
        # Dataframe with name of the constraints added
        columns = ["constraint_name", "iteration"]
        self.constraints_added = pd.DataFrame(columns=columns)

        # Define lower and upper bounds of the objective
        self.lower_bound = None
        self.upper_bound = None

        logging.info("")
        logging.info("--- Creating the master problem ---")
        self.master_model = MasterProblem(
            config=self.monolithic_model.config,
            config_benders=self.config.benders,
            solver=self.config.benders.solver_master,
            analysis=self.analysis,
            monolithic_model=self.monolithic_model,
            model_name=self.monolithic_model.model_name,
            scenario_name=self.monolithic_model.scenario_name,
            scenario_dict=self.monolithic_model.scenario_dict,
            input_data_checks=self.monolithic_model.input_data_checks,
            operational_variables=self.operational_variables,
            operational_constraints=self.operational_constraints,
            benders_output_folder=self.benders_output_folder,
        )
        self.subproblem_models = []
        for scenario, scenario_dict in zip(scenarios, elements):
            logging.info("")
            logging.info("--- Creating the subproblem %s ---", scenario)
            subproblem = Subproblem(
                config=self.monolithic_model.config,
                config_benders=self.config.benders,
                solver=self.config.benders.solver_subproblem,
                analysis=self.analysis,
                monolithic_model=self.monolithic_model,
                master_model=self.master_model,
                model_name=self.monolithic_model.model_name,
                scenario_name=scenario,
                scenario_dict=scenario_dict,
                input_data_checks=self.monolithic_model.input_data_checks,
                not_coupling_variables=self.not_coupling_variables,
                design_constraints=self.design_constraints,
                benders_output_folder=self.benders_output_folder,
            )
            self.subproblem_models.append(subproblem)

    def separate_design_operational_constraints(self) -> list:
        """
        Separate the design and operational constraints based on the user input preferences defined in the config file.
        It also perfomers a sanity check to ensure that all the constraints of the monolithic problem are included in
        the benders decomposition.

        :return: design_constraints, operational_constraints (type: lists of strings)
        """
        benders_constraints = self.data_input.read_input_csv("constraints")
        design_constraints = []
        operational_constraints = []

        # The benders_constraints is a dataframe with columns: constraint_name and constraint_type
        for _, constraint in benders_constraints.iterrows():
            name = constraint["constraint_name"]
            if constraint["constraint_type"] == "design":
                if name == "exclusive_retrofit_techs_":
                    for base_tech in self.config.system["set_exclusive_retrofitting_technologies"].items():
                        design_constraints.append(f"{name}{base_tech[0]}")
                else:
                    design_constraints.append(name)
            elif constraint["constraint_type"] == "operational":
                if name == "exclusive_retrofit_techs_":
                    for base_tech in self.config.system["set_exclusive_retrofitting_technologies"].items():
                        operational_constraints.append(f"{name}{base_tech[0]}")
                else:
                    operational_constraints.append(name)
            else:
                raise AssertionError(f"Constraint {constraint['constraint_name']} has an invalid type.")

        # Remove from the two list the constraints that are not in the monolithic problem
        design_constraints = [
            constraint for constraint in design_constraints if constraint in self.monolithic_constraints
        ]
        operational_constraints = [
            constraint for constraint in operational_constraints if constraint in self.monolithic_constraints
        ]

        # Sanity check on the constraints
        if len(design_constraints) + len(operational_constraints) != len(self.monolithic_constraints):
            missing_constraints = set(self.monolithic_constraints) - set(design_constraints + operational_constraints)
            raise AssertionError(
                f"The following constraints are missing in the benders decomposition: {missing_constraints}"
            )

        return design_constraints, operational_constraints

    def separate_design_operational_variables(self) -> list:
        """
        Separate the design and operational variables based on the user input preferences defined in the config file.
        The function stores also variables defined as not coupling variables. These variables are design variables that
        are not included in the subproblems becasue they are not coupled with the operational variables.
        It also perfomers a sanity check to ensure that all the variables of the monolithic problem are included in the
        benders decomposition.

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
                    raise AssertionError(f"Variable {variable['variable_name']} has an invalid type.")

        # Sanity check on the variables
        if len(design_variables) + len(operational_variables) != len(self.monolithic_variables):
            missing_variables = set(self.monolithic_variables) - set(design_variables + operational_variables)
            raise AssertionError(
                f"The following variables are missing in the benders decomposition: {missing_variables}"
            )
        # Define the not coupling variables
        for _, variable in not_subproblem_variables.iterrows():
            if variable["variable_name"] in self.monolithic_variables:
                if variable["variable_type"] == "design":
                    not_coupling_variables.append(variable["variable_name"])
                else:
                    raise AssertionError(f"The variable {variable['variable_name']} must be a design variable.")

        return design_variables, operational_variables, not_coupling_variables

    def solve_master_model(self, iteration):
        """
        Solve the master model by leveraging on the OptimizationSetup method solve(). If there is the presence of
        oscillatory behavior, the function remove the forcing cuts after solving the master model.

        :param iteration: current iteration of the Benders Decomposition method (type: int)
        """
        # Make the directory for the basis file
        self.master_model.solve()

        if self.config["run_monolithic_optimization"] and self.master_model.only_feasibility_checks:
            optimality_gap = self.monolithic_model.model.objective.value - self.master_model.model.objective.value
            new_row = pd.DataFrame({"iteration": [iteration], "optimality_gap": [optimality_gap]})
            self.optimality_gap_df_infeasibility = pd.concat(
                [self.optimality_gap_df_infeasibility, new_row], ignore_index=True
            )

    def solve_master_model_with_monolithic_solution(self, iteration):
        """
        Fix the master variables bounds to the optimal solution of the monolithic problem and solve the master model.
        The remove the fixed bounds of the master variables and restore the original bounds.

        :param iteration: current iteration of the Benders Decomposition method (type: int)
        """
        # Save the original bounds of the master variables
        original_bounds = {}
        for variable_name in self.master_model.model.variables:
            original_bounds[variable_name] = (
                self.master_model.model.variables[variable_name].lower,
                self.master_model.model.variables[variable_name].upper,
            )
        # Fix the master variables to the optimal solution of the monolithic problem
        for variable_name in self.master_model.model.variables:
            variable_solution = self.monolithic_model.model.solution[variable_name]
            self.master_model.model.variables[variable_name].lower = variable_solution
            self.master_model.model.variables[variable_name].upper = variable_solution
        # Solve the master model
        self.master_model.solve()
        # Restore the original bounds of the master variables
        for variable_name in self.master_model.model.variables:
            self.master_model.model.variables[variable_name].lower = original_bounds[variable_name][0]
            self.master_model.model.variables[variable_name].upper = original_bounds[variable_name][1]

        if self.config["run_monolithic_optimization"] and self.master_model.only_feasibility_checks:
            optimality_gap = self.monolithic_model.model.objective.value - self.master_model.model.objective.value
            new_row = pd.DataFrame({"iteration": [iteration], "optimality_gap": [optimality_gap]})
            self.optimality_gap_df_infeasibility = pd.concat(
                [self.optimality_gap_df_infeasibility, new_row], ignore_index=True
            )

    def solve_subproblems_models(self, tolernace_change=False):
        """
        Solve the subproblems models given in the list self.subproblem_models by leveraging on the OptimizationSetup
        method solve().
        """
        if self.config.benders["cap_capacity_bounds"] and tolernace_change:
            self.config.benders.solver_subproblem.solver_options["FeasibilityTol"] = 1e-6
        else:
            self.config.benders.solver_subproblem.solver_options["FeasibilityTol"] = self.feasibility_tol_subproblem
        for subproblem in self.subproblem_models:
            env = Env(empty=True)
            env.setParam("OutputFlag", 0)
            env.setParam("LogToConsole", 0)
            env.start()
            self.config.benders.solver_subproblem.solver_options["env"] = env
            subproblem.solve()
            logging.info("Subproblem %s is %s", subproblem.scenario_name, subproblem.model.termination_condition)

    def fix_design_variables_in_subproblem_model(self):
        """
        Fix the design variables of the subproblems to the optimal solution of the master problem.
        This function takes the solution of the master problem and fixes the values of the design variables in the
        subproblems by adding the corresponding upper and lower bounds to the variables. In order to keep stability in
        the method, the solution lower than 1e-5 is set to 0, while the NaN values are kept as they are.
        """
        master_solution = self.master_model.model.solution

        for variable_name in self.master_model.model.variables:
            for subproblem in self.subproblem_models:
                if variable_name in subproblem.model.variables:
                    variable_solution = master_solution[variable_name]
                    variable_solution = variable_solution.where(
                        (variable_solution > 1e-5) | np.isnan(variable_solution), 0
                    )
                    subproblem.model.variables[variable_name].lower = variable_solution
                    subproblem.model.variables[variable_name].upper = variable_solution

    def add_capacity_bounds_to_master(self):
        """
        Add the capacity bounds to the master model. The capacity lower bound constraint is defined as
        the capacity of the master problem greater or equal to the capacity of the subproblem.
        """
        variable = None
        lhs_lower = None
        lhs_upper = None
        if hasattr(self.master_model.model.variables, "capacity"):
            variable = "capacity"
            lhs_lower = self.master_model.model.variables[variable]
            lhs_upper = self.master_model.model.variables[variable]
        if variable is not None:

            solution_arrays = [subproblem.model.solution[variable] for subproblem in self.subproblem_models]
            if len(solution_arrays) == 1:
                rhs_min = solution_arrays[0]
                rhs_max = solution_arrays[0]
            else:
                combined_solution_min = solution_arrays[0]
                combined_solution_max = solution_arrays[0]
                for solution in solution_arrays[1:]:
                    combined_solution_min = xr.apply_ufunc(np.minimum, combined_solution_min, solution)
                    combined_solution_max = xr.apply_ufunc(np.maximum, combined_solution_max, solution)

            coefficients = self.master_model.model.objective.coeffs
            max_coefficient = np.max(coefficients)
            min_coefficient = np.min(coefficients)
            mean_coefficient = np.mean(coefficients)
            std_coefficient = np.std(coefficients)

            scaling_factor_upper = 1 + 0.5 * (max_coefficient - mean_coefficient) / std_coefficient
            scaling_factor_lower = 1 - 0.5 * (mean_coefficient - min_coefficient) / std_coefficient

            rhs_min = combined_solution_min * scaling_factor_lower
            rhs_max = combined_solution_max * scaling_factor_upper

            rhs_min = xr.where(rhs_min < 0, 0, rhs_min)
            rhs_max = xr.where(rhs_max <= 0, 0, rhs_max)

            constraint_min = lhs_lower >= rhs_min
            constraint_max = lhs_upper <= rhs_max
            self.master_model.constraints.add_constraint("constraint_capacity_lower_bound", constraint_min)
            self.master_model.constraints.add_constraint("constraint_capacity_upper_bound", constraint_max)

    def solved_model_to_gurobi(self, linopy_model_solved):
        """
        Function to get the solver model (gurobi model solved) of the subproblem. Needed to extract the Farkas
        multipliers for feasibility cuts and the Dual multipliers for optimality cuts.

        :param linopy_model_solved: subproblem model to be converted to gurobi (type: OptimizationSetup)
        """

        gurobi_model = getattr(linopy_model_solved.model, "solver_model")
        return gurobi_model

    def detect_oscillatory_behavior(self, current_feasibility_cuts, feasibility_iteration):
        """
        Detect oscillatory behavior in the Benders Decomposition method. The oscillatory behavior is detected by
        comparing the feasibility cuts of the current iteration with the feasibility cuts of the previous iteration.
        If the feasibility cuts are the same, the oscillatory behavior is detected.
        """
        oscillatory_behavior = False
        if feasibility_iteration == 1:
            return oscillatory_behavior
        last_feasibility_iteration = str(feasibility_iteration - 1)
        for subproblem_name, feasibility_cut_lhs, feasibility_cut_rhs in current_feasibility_cuts:
            last_feasibility_cut_name = f"feasibility_cuts_{subproblem_name}_iteration_{last_feasibility_iteration}"
            if last_feasibility_cut_name in self.master_model.model.constraints:
                last_feasibility_cut_lhs = self.master_model.model.constraints[last_feasibility_cut_name].lhs
                last_feasibility_cut_rhs = self.master_model.model.constraints[last_feasibility_cut_name].rhs
                if (
                    last_feasibility_cut_lhs.equals(feasibility_cut_lhs)
                    and last_feasibility_cut_rhs.item() == feasibility_cut_rhs
                ):
                    logging.info("--- Oscillatory Behavior Detected ---")
                    oscillatory_behavior = True

        return oscillatory_behavior

    def generate_feasibility_cut(self, gurobi_model, subproblem):
        """
        Generate the feasibility cut. The feasibility cut is generated by the Farkas multipliers of the subproblem
        model. In order to keep stability and speed up the process, only multipliers higher than 1e-4 are considered.

        :param gurobi_model: subproblem model in gurobi format (type: Model)
        :param subproblem: subproblem model in OptimizationSetup format (type: OptimizationSetup)

        :return: feasibility_cut_lhs: left-hand side of the feasibility cut (type: LinExpr)
        :return: feasibility_cut_rhs: right-hand side of the feasibility cut (type: float)
        """
        start_time_cuts = time.time()
        # Initialize feasibility cut components
        feasibility_cut_lhs = 0
        feasibility_cut_rhs = 0
        # Get Farkas multipliers for constraints
        farkas_multipliers = [
            (
                int(constr.ConstrName[1:]),
                multiplier if abs(multiplier) > 0 else 0,
            )
            for constr, multiplier in zip(gurobi_model.getConstrs(), gurobi_model.getAttr(GRB.Attr.FarkasDual))
        ]
        farkas_multipliers_df = pd.DataFrame(farkas_multipliers, columns=["labels_con", "multiplier"])

        merged_rhs_df = pd.merge(farkas_multipliers_df, subproblem.rhs_cuts, on="labels_con")
        merged_rhs_df["product"] = merged_rhs_df["multiplier"] * merged_rhs_df["rhs"]
        feasibility_cut_rhs = merged_rhs_df["product"].sum()

        merged_lhs_df = pd.merge(farkas_multipliers_df, subproblem.lhs_cuts, on="labels_con")
        merged_lhs_df["product"] = merged_lhs_df["multiplier"] * merged_lhs_df["coeffs"]
        filtered_df = merged_lhs_df[merged_lhs_df["product"] != 0]
        if not filtered_df[filtered_df.duplicated(subset=["labels"], keep=False)].empty:
            product_df = filtered_df[["labels", "product"]]
            product_df = product_df.groupby("labels").sum()
            product_df = product_df.reset_index()
            unique_filtered_df = filtered_df.drop_duplicates(subset=["labels"])
            filtered_df = product_df.merge(unique_filtered_df[["labels", "master_variable"]], on="labels")
        filtered_df = filtered_df.loc[filtered_df["product"] != 0]
        linear_expression_list = list(filtered_df["product"] * filtered_df["master_variable"])
        feasibility_cut_lhs = lp.merge(linear_expression_list)
        end_time_cuts = time.time()
        total_time_cuts = end_time_cuts - start_time_cuts
        logging.info("Time to generate the feasibility cut: %s", total_time_cuts)
        return feasibility_cut_lhs, feasibility_cut_rhs

    def define_list_of_feasibility_cuts(self) -> list:
        """
        Define the list of feasibility cuts when dealing with multiple subproblems.

        :param iteration: current iteration of the Benders Decomposition method (type: int)

        :return: list of feasibility cuts as tuples with the following structure:
            - subproblem_name: name of the subproblem (if the scenario name is not defined, it is set to "default")
            - feasibility_cut_lhs: left-hand side of the feasibility cut
            - feasibility_cut_rhs: right-hand side of the feasibility cut
        """
        logging.info("--- Generating feasibility cuts ---")
        infeasible_subproblems = [
            subproblem
            for subproblem in self.subproblem_models
            if subproblem.model.termination_condition in ["infeasible", "infeasible_or_unbounded"]
        ]
        logging.info("Number of infeasible subproblems: %s", len(infeasible_subproblems))
        feasibility_cuts = [
            (
                subproblem.scenario_name if subproblem.scenario_name else "default",
                *self.generate_feasibility_cut(self.solved_model_to_gurobi(subproblem), subproblem),
            )
            for subproblem in infeasible_subproblems
        ]
        return feasibility_cuts

    def filter_redundant_cuts(self, feasibility_cuts):
        """
        Filter the redundant feasibility cuts. The function checks if the feasibility cuts are redundant by comparing
        the feasibility cuts of the current iteration with the feasibility cuts of the previous iteration. If the
        feasibility cuts are the same, the feasibility cut is considered redundant.

        :param feasibility_cuts: list of feasibility cuts (type: list)

        :return: filtered_feasibility_cuts: list of filtered feasibility cuts (type: list)
        """
        filtered_cuts = []
        for _, cut in enumerate(feasibility_cuts):
            is_unique = True
            for filtered_cut in filtered_cuts:
                if cut[1].equals(filtered_cut[1]) and cut[2] == filtered_cut[2]:
                    is_unique = False
                    break
            if is_unique:
                filtered_cuts.append(cut)

        return filtered_cuts

    def add_feasibility_cuts_to_master(self, feasibility_cuts, feasibility_iteration, iteration):
        """
        Add the feasibility cuts to the master model.

        :param feasibility_cuts: list of feasibility cuts (type: list)
        :param feasibility_iteration: current feasibility_iteration of the Benders Decomposition method (type: int)
        :param iteration: current iteration of the Benders Decomposition method (type: int)
        """
        for subproblem_name, feasibility_cut_lhs, feasibility_cut_rhs in feasibility_cuts:
            self.feasibility_cuts_counter += 1
            self.master_model.model.add_constraints(
                lhs=feasibility_cut_lhs,
                sign="<=",
                rhs=feasibility_cut_rhs,
                name=f"feasibility_cuts_{subproblem_name}_iteration_{feasibility_iteration}",
            )
            # Save the cut in the constraints_added dataframe
            new_row = pd.DataFrame(
                {
                    "constraint_name": [f"feasibility_cuts_{subproblem_name}_iteration_{feasibility_iteration}"],
                    "iteration": [iteration],
                }
            )
            self.constraints_added = pd.concat([self.constraints_added, new_row], ignore_index=True)

    def generate_optimality_cut(self, gurobi_model, subproblem) -> tuple:
        """
        Generate the optimality cut. The optimality cut is generated by the dual multipliers of the subproblem model. In
        order to keep stability and speed up the process, only multipliers higher than 1e-4 are considered.

        :param gurobi_model: subproblem model in gurobi format (type: Model)
        :param subproblem: subproblem model in OptimizationSetup format (type: OptimizationSetup)

        :return: optimality_cut_lhs: left-hand side of the optimality cut (type: LinExpr)
        :return: optimality_cut_rhs: right-hand side of the optimality cut (type: float)
        """
        start_time_cuts = time.time()
        # Initialize optimality cut components
        optimality_cut_lhs = 0
        optimality_cut_rhs = 0
        # Get dual multipliers for constraints
        duals_multipliers = [
            (
                int(constr.ConstrName[1:]),
                multiplier if abs(multiplier) > 0 else 0,
            )
            for constr, multiplier in zip(gurobi_model.getConstrs(), gurobi_model.getAttr(GRB.Attr.Pi))
        ]
        duals_multipliers_df = pd.DataFrame(duals_multipliers, columns=["labels_con", "multiplier"])
        merged_rhs_df = pd.merge(duals_multipliers_df, subproblem.rhs_cuts, on="labels_con")
        merged_rhs_df["product"] = merged_rhs_df["multiplier"] * merged_rhs_df["rhs"]
        optimality_cut_rhs = merged_rhs_df["product"].sum()

        merged_lhs_df = pd.merge(duals_multipliers_df, subproblem.lhs_cuts, on="labels_con")
        merged_lhs_df["product"] = merged_lhs_df["multiplier"] * merged_lhs_df["coeffs"]
        filtered_df = merged_lhs_df[merged_lhs_df["product"] != 0]
        linear_expression_list = list(filtered_df["product"] * filtered_df["master_variable"])
        optimality_cut_lhs = lp.merge(linear_expression_list)
        end_time_cuts = time.time()
        total_time_cuts = end_time_cuts - start_time_cuts
        logging.info("Time to generate the optimality cut: %s", total_time_cuts)
        return optimality_cut_lhs, optimality_cut_rhs

    def define_list_of_optimality_cuts(self) -> list:
        """
        Define the list of optimality cuts when dealing with multiple subproblems.

        :param iteration: current iteration of the Benders Decomposition method (type: int)

        :return: list of optimality cuts as tuples with the following structure:
            - subproblem_name: name of the subproblem (if the scenario name is not defined, it is set to "default")
            - optimality_cut_lhs: left-hand side of the optimality cut
            - optimality_cut_rhs: right-hand side of the optimality cut
        """
        logging.info("--- Generatings optimality cuts ---")
        optimality_cuts = [
            (
                subproblem.scenario_name if subproblem.scenario_name else "default",
                *self.generate_optimality_cut(self.solved_model_to_gurobi(subproblem), subproblem),
            )
            for subproblem in self.subproblem_models
        ]
        return optimality_cuts

    def add_optimality_cuts_to_master(self, optimality_cuts, optimality_iteration, iteration):
        """
        Add the optimality cuts to the master model.

        :param optimality_cuts: list of optimality cuts (type: list)
        :param optimality_iteration: current iteration of the Benders Decomposition method (type: int)
        :param iteration: current iteration of the Benders Decomposition method (type: int)
        """
        for subproblem_name, optimality_cut_lhs, optimality_cut_rhs in optimality_cuts:
            self.optimality_cuts_counter += 1
            lhs = optimality_cut_lhs + self.master_model.model.variables["outer_approximation"]
            self.master_model.model.add_constraints(
                lhs=lhs,
                sign=">=",
                rhs=optimality_cut_rhs,
                name=f"optimality_cuts_{subproblem_name}_iteration_{optimality_iteration}",
            )
            # Save the cut in the constraints_added dataframe
            new_row = pd.DataFrame(
                {
                    "constraint_name": [f"optimality_cuts_{subproblem_name}_iteration_{optimality_iteration}"],
                    "iteration": [iteration],
                }
            )
            self.constraints_added = pd.concat([self.constraints_added, new_row], ignore_index=True)

    def check_termination_criteria(self, iteration) -> list:
        """
        Check the termination condition of the Benders Decomposition method.
        The outer approximation variable is the variable that is added to the master problem when the objective function
        does not include the design variables.
        The function compute the lower and upper bounds of the objective function
        and check if the termination criteria is satisfied.

        :param iteration: current iteration of the Benders Decomposition method (type: int)

        :return: termination_criteria: list of tuples with the following structure:
            - subproblem_name: name of the subproblem
            - termination_criteria: boolean value indicating if the termination criteria is satisfied
        """
        termination_criteria = []
        table = []
        headers = ["Subproblem", "Lower Bound", "Upper Bound"]
        self.lower_bound = self.master_model.model.objective.value

        # Collect upper bounds for subproblems
        self.upper_bound = [
            (subproblem.scenario_name if subproblem.scenario_name else "default", subproblem.model.objective.value)
            for subproblem in self.subproblem_models
        ]

        # Process each subproblem's upper bound
        for name, upper_bound in self.upper_bound:
            table.append([name, self.lower_bound, upper_bound])
            # Check termination criteria
            optimality_gap = abs(upper_bound - self.lower_bound) / abs(upper_bound)
            is_terminated = optimality_gap <= self.config.benders["absolute_optimality_gap"]
            termination_criteria.append((name, is_terminated))
            # Update optimality gap DataFrame
            new_row = pd.DataFrame({"iteration": [iteration], "optimality_gap": [upper_bound - self.lower_bound]})
            self.optimality_gap_df_optimal = pd.concat([self.optimality_gap_df_optimal, new_row], ignore_index=True)

        # Log the table if lower bound is not zero
        if self.lower_bound != 0:
            logging.info("\n%s", tabulate(table, headers, tablefmt="grid", floatfmt=".6f"))
        return termination_criteria

    def remove_mock_variables_and_constraints(self):
        """
        Remove the mock variables from the subproblems or the theta variable from the master problem if they exist.
        Moreover, remove the constraint_for_binaries from the master problem.
        This is done in order to retrieve the same problem configuration as before than the scaling process.
        Consequently, the user is able to save the solution of the problem in the output folder.
        """

        for subproblem in self.subproblem_models:
            if "mock_objective_subproblem" in subproblem.model.variables:
                subproblem.model.variables.remove("mock_objective_subproblem")

        constraints_master_to_remove = ["constraint_for_binaries"]
        for constraint in constraints_master_to_remove:
            if constraint in self.master_model.model.constraints:
                self.master_model.model.constraints.remove(constraint)

        variables_master_to_remove = ["outer_approximation"]
        for variable in variables_master_to_remove:
            if variable in self.master_model.model.variables:
                self.master_model.model.variables.remove(variable)

    def save_csv_files(self):
        """
        Save the csv files in the output folder.
        """
        if self.master_model.only_feasibility_checks:
            self.optimality_gap_df_infeasibility.to_csv(
                os.path.join(self.benders_output_folder, "optimality_gap_infeasibility.csv")
            )
        else:
            self.optimality_gap_df_optimal.to_csv(
                os.path.join(self.benders_output_folder, "optimality_gap_optimal.csv")
            )

        self.cuts_counter_df = pd.DataFrame(
            {
                "number_of_monolithic_constraints": [self.monolithic_model.model.constraints.ncons],
                "number_of_feasibility_cuts": [self.feasibility_cuts_counter],
                "number_of_optimality_cuts": [self.optimality_cuts_counter],
            }
        )
        self.cuts_counter_df.to_csv(os.path.join(self.benders_output_folder, "cuts_counter.csv"))
        self.constraints_added.to_csv(os.path.join(self.benders_output_folder, "cuts_added.csv"))

    def save_master_and_subproblems(self):
        """
        Save the master model and the subproblem models in respective output folders.

        :param iteration: current iteration of the Benders Decomposition method (type: int)
        """
        if self.master_model.model.termination_condition == "optimal" and all(
            subproblem.model.termination_condition == "optimal" for subproblem in self.subproblem_models
        ):
            logging.info("--- Optimal solution found. Terminating iterations ---")
            # Re-scale the variables if scaling is used
            self.remove_mock_variables_and_constraints()
            if self.config.solver["use_scaling"]:
                self.master_model.scaling.re_scale()
                for subproblem in self.subproblem_models:
                    subproblem.scaling.re_scale()

            # Write the results
            Postprocess(
                model=self.master_model, scenarios={"": {}}, model_name=self.master_model.model_name, subfolder=Path("")
            )
            # Delete the solver_dir of the master model
            if os.path.exists(self.master_model.solver.solver_dir):
                shutil.rmtree(self.master_model.solver.solver_dir)

            for subproblem in self.subproblem_models:
                scenario_name_subproblem, subfolder_subproblem, param_map_subproblem = StringUtils.generate_folder_path(
                    config=self.config.benders,
                    scenario=subproblem.scenario_name,
                    scenario_dict=subproblem.scenario_dict,
                    steps_horizon=[1],
                    step=1,
                )
                Postprocess(
                    model=subproblem,
                    scenarios=self.config.benders.scenarios,
                    model_name=subproblem.model_name,
                    subfolder=subfolder_subproblem,
                    scenario_name=scenario_name_subproblem,
                    param_map=param_map_subproblem,
                )

        self.save_csv_files()

    def fit(self):
        """
        Fit the Benders Decomposition model.
        """
        iteration = 1
        feasibility_iteration = 1
        optimality_iteration = 1
        max_number_of_iterations = self.config.benders["max_number_of_iterations"]
        continue_iterations = True

        # While loop to solve the master problem and the subproblems iteratively
        while continue_iterations and iteration <= max_number_of_iterations:
            logging.info("")
            logging.info("")
            logging.info("--- Iteration %s ---", iteration)
            if iteration == 1:
                if self.config.benders["cap_capacity_bounds"]:
                    self.solve_subproblems_models(tolernace_change=True)
                    if any(
                        subproblem.model.termination_condition != "optimal" for subproblem in self.subproblem_models
                    ):
                        continue_iterations = False
                        break
                    self.add_capacity_bounds_to_master()
                    for subproblem in self.subproblem_models:
                        subproblem.change_objective_to_constant()
                        subproblem.remove_design_constraints()
                        subproblem.define_lhs_rhs_for_cuts()
                else:
                    self.master_model.check_variables_bounds()
                    for subproblem in self.subproblem_models:
                        subproblem.remove_design_constraints()
                        subproblem.define_lhs_rhs_for_cuts()

            logging.info("--- Solving master problem, fixing design variables in subproblems and solve them ---")
            if self.use_monolithic_solution and iteration == 1:
                self.solve_master_model_with_monolithic_solution(iteration)
            else:
                self.solve_master_model(iteration)

            if self.master_model.model.termination_condition != "optimal":
                logging.info("--- Master problem is infeasible ---")
                self.master_model.model.print_infeasibilities()
                self.save_csv_files()
                continue_iterations = False
                break

            # Fix the design variables in the subproblems to the optimal solution of the master problem and solve
            # the subproblems
            self.fix_design_variables_in_subproblem_model()
            self.solve_subproblems_models()

            # Check terminatio condition of the subproblems
            if any(subproblem.model.termination_condition != "optimal" for subproblem in self.subproblem_models):
                logging.info("--- Subproblems are infeasible ---")
                feasibility_cuts = self.define_list_of_feasibility_cuts()
                oscillatory_behavior = self.detect_oscillatory_behavior(
                    feasibility_cuts,
                    feasibility_iteration,
                )
                filtered_feasibility_cuts = self.filter_redundant_cuts(feasibility_cuts)
                if not oscillatory_behavior:
                    self.add_feasibility_cuts_to_master(filtered_feasibility_cuts, feasibility_iteration, iteration)
                    feasibility_iteration += 1
                else:
                    continue_iterations = not oscillatory_behavior

            if all(subproblem.model.termination_condition == "optimal" for subproblem in self.subproblem_models):
                logging.info("--- All the subproblems are optimal ---")
                if self.master_model.only_feasibility_checks:
                    continue_iterations = False
                else:
                    optimality_cuts = self.define_list_of_optimality_cuts()
                    self.add_optimality_cuts_to_master(optimality_cuts, optimality_iteration, iteration)
                    optimality_iteration += 1
                    termination_criteria = self.check_termination_criteria(iteration)
                    if all(value for _, value in termination_criteria):
                        continue_iterations = False

            if self.master_model.only_feasibility_checks:
                table = []
                headers = ["Objective", "Monolithic", "Master"]
                table.append(
                    ["Master", self.monolithic_model.model.objective.value, self.master_model.model.objective.value]
                )
                logging.info("\n%s", tabulate(table, headers, tablefmt="grid", floatfmt=".6f"))

            if continue_iterations is False:
                self.save_master_and_subproblems()

            if iteration == max_number_of_iterations:
                logging.info("--- Maximum number of iterations reached ---")
                logging.info("--- Saving possible results ---")
                self.save_csv_files()

            iteration += 1
