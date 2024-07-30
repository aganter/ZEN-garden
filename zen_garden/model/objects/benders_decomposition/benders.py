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
import time
from pathlib import Path
import warnings
import pandas as pd
import psutil
from tabulate import tabulate
from gurobipy import GRB

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
        # Define the different scenarios for the Benders Decomposition method
        scenarios, elements = ScenarioUtils.get_scenarios(
            config=self.config.benders, scenario_script_name="benders_scenarios", job_index=None
        )

        self.monolithic_constraints = self.monolithic_model.model.constraints
        self.monolithic_variables = self.monolithic_model.model.variables
        self.design_constraints, self.operational_constraints = self.separate_design_operational_constraints()
        self.design_variables, self.operational_variables, self.not_coupling_variables = (
            self.separate_design_operational_variables()
        )

        self.monolithic_gurobi = None
        self.map_variables_monolithic_gurobi = {}

        logger = logging.getLogger("gurobipy")
        logging.getLogger("gurobipy").setLevel(logging.ERROR)
        logger.propagate = False
        # Save the monolithic problem in the gurobi format and map the variables
        self.save_monolithic_model_in_gurobi_format_map_variables()

        # DataFrames to store information about the building and solving of the subproblems
        columns = ["subproblem", "build_time_sec", "build_memory_MB"]
        self.building_subproblem = pd.DataFrame(columns=columns)
        columns = ["subproblem", "iteration", "solve_time_sec", "solve_memory_MB"]
        self.solving_subproblem = pd.DataFrame(columns=columns)
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
        # Dataframe with the time for the construction of the constraints
        columns = ["iteration", "constraint_type", "construction_time_sec"]
        self.constraints_construction_time = pd.DataFrame(columns=columns)
        # Dataframe with the time for adding the constraints
        columns = ["iteration", "constraint_type", "addition_time_sec"]
        self.constraints_addition_time = pd.DataFrame(columns=columns)

        self.subproblems_gurobi = []
        self.rhs_subproblmes = []
        self.lhs_subproblems_design = []

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

        logging.info("")
        logging.info("--- Creating the subproblems ---")
        self.subproblem_models = []
        for scenario, scenario_dict in zip(scenarios, elements):
            start_time = time.time()
            pid = os.getpid()
            subproblem = Subproblem(
                config=self.monolithic_model.config,
                config_benders=self.config.benders,
                solver=self.config.benders.solver_subproblem,
                analysis=self.analysis,
                monolithic_model=self.monolithic_model,
                model_name=self.monolithic_model.model_name,
                scenario_name=scenario,
                scenario_dict=scenario_dict,
                input_data_checks=self.monolithic_model.input_data_checks,
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

    def save_monolithic_model_in_gurobi_format_map_variables(self):
        """
        Save the monolithic problem in the gurobi format. This is necessary to map the variables of the monolithic
        problem to the gurobi variables. The mapping is a dictionary with key the name of the variable in the gurobi
        model and following three values:
            1. variable_name: the variable name in the monolithic problem
            2. variable_coords: the coordinates of the variable in the monolithic problem
            3. variable_gurobi: the variable in the gurobi model

        """
        self.monolithic_gurobi = getattr(self.monolithic_model.model, "solver_model")
        variables = self.monolithic_gurobi.getVars()
        label_positions = [
            self.monolithic_model.model.variables.get_label_position(int(var.VarName[1:])) for var in variables
        ]
        self.map_variables_monolithic_gurobi = {
            var.VarName: {"variable_name": label_pos[0], "variable_coords": label_pos[1], "variable_gurobi": var}
            for var, label_pos in zip(variables, label_positions)
        }

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
            if constraint["constraint_name"] in self.monolithic_constraints:
                if constraint["constraint_type"] == "design":
                    design_constraints.append(constraint["constraint_name"])
                elif constraint["constraint_type"] == "operational":
                    operational_constraints.append(constraint["constraint_name"])
                else:
                    raise AssertionError(f"Constraint {constraint['constraint_name']} has an invalid type.")

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
        Solve the master model by leveraging on the OptimizationSetup method solve().

        :param iteration: current iteration of the Benders Decomposition method (type: int)
        """
        # Get the solution file of the last master problem
        warm_start = self.master_model.model.get_solution_file()
        # Update the solver options
        self.config.benders.solver_master["warmstart_fn"] = warm_start

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

    def solve_subproblems_models(self, iteration):
        """
        Solve the subproblems models given in the list self.subproblem_models by leveraging on the OptimizationSetup
        method solve().

        :param iteration: current iteration of the Benders Decomposition method (type: int)
        """
        for subproblem in self.subproblem_models:
            start_time = time.time()
            pid = os.getpid()

            subproblem.solve()

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

        :param iteration: current iteration of the Benders Decomposition method (type: int)
        """
        master_solution = self.master_model.model.solution

        for variable_name in self.master_model.model.variables:
            for subproblem in self.subproblem_models:
                if variable_name in subproblem.model.variables:
                    variable_solution = master_solution[variable_name]
                    subproblem.model.variables[variable_name].lower = variable_solution
                    subproblem.model.variables[variable_name].upper = variable_solution

    def subproblem_to_gurobi(self, subproblem_solved):
        """
        Function to get the solver model (gurobi model solved) of the subproblem. Needed to extract the Farkas
        multipliers for feasibility cuts and the Dual multipliers for optimality cuts.

        :param subproblem_solved: subproblem model to be converted to gurobi (type: OptimizationSetup)
        """

        gurobi_model = getattr(subproblem_solved.model, "solver_model")
        return gurobi_model

    def generate_feasibility_cut(self, gurobi_model, iteration) -> tuple:
        """
        Generate the feasibility cut. The feasibility cut is generated by the Farkas multipliers of the subproblem
        model.

        :param gurobi_model: subproblem model in gurobi format (type: Model)

        :return: feasibility_cut_lhs: left-hand side of the feasibility cut (type: LinExpr)
        :return: feasibility_cut_rhs: right-hand side of the feasibility cut (type: float)
        """
        # Get Farkas multipliers for constraints
        starting_time = time.time()
        farkas_multipliers = [
            (constraint_name, multiplier)
            for constraint_name, multiplier in zip(
                [constr for constr in gurobi_model.getConstrs()],
                gurobi_model.getAttr(GRB.Attr.FarkasDual),
            )
            if multiplier != 0
        ]

        # Initialize feasibility cut components
        feasibility_cut_lhs = 0
        feasibility_cut_rhs = 0

        # Iterate over constraints and their Farkas multipliers
        for gurobi_constr, farkas in farkas_multipliers:
            rhs = gurobi_constr.RHS
            feasibility_cut_rhs += farkas * rhs

            lhs = gurobi_model.getRow(gurobi_constr)
            for i in range(lhs.size()):
                var = lhs.getVar(i)
                subproblem_var = self.map_variables_monolithic_gurobi[var.VarName]
                subproblem_var_name = subproblem_var["variable_name"]
                subproblem_var_coords = subproblem_var["variable_coords"]
                coeff = lhs.getCoeff(i)
                if subproblem_var_name in self.master_model.model.variables:
                    master_variable = self.master_model.model.variables[subproblem_var_name].sel(subproblem_var_coords)
                    feasibility_cut_lhs += farkas * (master_variable * coeff)

        end_time = time.time()
        construction_time = end_time - starting_time
        new_row = pd.DataFrame(
            {"iteration": [iteration], "constraint_type": ["feasibility"], "construction_time_sec": [construction_time]}
        )
        self.constraints_construction_time = pd.concat([self.constraints_construction_time, new_row], ignore_index=True)
        return feasibility_cut_lhs, feasibility_cut_rhs

    def define_list_of_feasibility_cuts(self, iteration) -> list:
        """
        Define the list of feasibility cuts when dealing with multiple subproblems.

        :return: list of feasibility cuts as tuples with the following structure:
            - subproblem_name: name of the subproblem (if the scenario name is not defined, it is set to "default")
            - feasibility_cut_lhs: left-hand side of the feasibility cut
            - feasibility_cut_rhs: right-hand side of the feasibility cut
        """
        logging.info("--- Generating feasibility cut ---")
        infeasible_subproblems = [
            subproblem
            for subproblem in self.subproblem_models
            if subproblem.model.termination_condition in ["infeasible", "infeasible_or_unbounded"]
        ]
        logging.info("Number of infeasible subproblems: %s", len(infeasible_subproblems))
        feasibility_cuts = [
            (
                subproblem.scenario_name if subproblem.scenario_name else "default",
                *self.generate_feasibility_cut(self.subproblem_to_gurobi(subproblem), iteration),
            )
            for subproblem in infeasible_subproblems
        ]
        return feasibility_cuts

    def add_feasibility_cuts_to_master(self, feasibility_cuts, feasibility_iteration, iteration):
        """
        Add the feasibility cuts to the master model.

        :param feasibility_cuts: list of feasibility cuts (type: list)
        :param feasibility_iteration: current feasibility_iteration of the Benders Decomposition method (type: int)
        """
        logging.info("--- Adding feasibility cut to master model")
        starting_time = time.time()
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
        end_time = time.time()
        addition_time = end_time - starting_time
        new_row = pd.DataFrame(
            {"iteration": [iteration], "constraint_type": ["feasibility"], "addition_time_sec": [addition_time]}
        )
        self.constraints_addition_time = pd.concat([self.constraints_addition_time, new_row], ignore_index=True)

    def check_feasibility_cut_effectiveness(self, feasibility_iteration, current_feasibility_cuts, iteration):
        """
        Check the effectiveness of the feasibility cuts.
        The tolerance of the solution of the master model can lead to oscillatory behavior.
        The function checks if the last generated feasibility cuts are identical to the previous ones. If they are, the
        function check the values of the master variables appearing in the feasibility cuts. If the latters do not
        respect the bounds by a small amount allowed by the solver Gurobi, the function add an "forcing_cut" to the
        master model. The forcing cut is a cut that forces the master variables to respect the bounds (so
        adding/subtracting the to tolerance of the solver).

        :param feasibility_iteration: current iteration of the Benders Decomposition method (type: int)
        :param current_feasibility_cuts: list of feasibility cuts (type: list)
        """
        oscillatory_behavior = False
        if feasibility_iteration == 1:
            return oscillatory_behavior

        last_feasibility_iteration = str(feasibility_iteration - 1)
        redundant_feasibility_cuts = []
        master_variables_in_redundant_feasibility_cuts = []
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
                    redundant_feasibility_cuts.append([subproblem_name, feasibility_cut_lhs, feasibility_cut_rhs])
                    master_variables_in_redundant_feasibility_cuts += last_feasibility_cut_lhs.vars.data.tolist()

        master_variables_in_redundant_feasibility_cuts = list(set(master_variables_in_redundant_feasibility_cuts))
        for var in master_variables_in_redundant_feasibility_cuts:
            counter = 1
            name = self.master_model.model.variables.get_label_position(var)[0]
            coords = self.master_model.model.variables.get_label_position(var)[1]
            solution = self.master_model.model.solution[name].sel(coords)
            if (
                abs(solution) <= self.master_model.model.variables[name].sel(coords).lower + 1e-3
                or solution <= self.master_model.model.variables[name].sel(coords).upper
            ):
                self.master_model.model.add_constraints(
                    lhs=self.master_model.model.variables[name].sel(coords),
                    sign="=>",
                    rhs=self.master_model.model.variables[name].sel(coords).lower + 1e-3,
                    name=f"forcing_cut_{name}_{counter}_iteration_{feasibility_iteration}",
                )
                oscillatory_behavior = True
                # Save the cut in the constraints_added dataframe
                new_row = pd.DataFrame(
                    {
                        "constraint_name": [f"forcing_cut_{name}_{counter}_iteration_{feasibility_iteration}"],
                        "iteration": [iteration],
                    }
                )
                self.constraints_added = pd.concat([self.constraints_added, new_row], ignore_index=True)
                counter += 1
        return oscillatory_behavior

    def generate_optimality_cut(self, gurobi_model, iteration) -> tuple:
        """
        Generate the optimality cut. The optimality cut is generated by the dual multipliers of the subproblem model.

        :param gurobi_model: subproblem model in gurobi format (type: Model)

        :return: optimality_cut_lhs: left-hand side of the optimality cut (type: LinExpr)
        :return: optimality_cut_rhs: right-hand side of the optimality cut (type: float)
        """
        starting_time = time.time()
        # Get dual multipliers for constraints
        duals_multiplier = [
            (constraint_name, multiplier)
            for constraint_name, multiplier in zip(
                [constr for constr in gurobi_model.getConstrs()],
                gurobi_model.getAttr(GRB.Attr.Pi),
            )
            if multiplier != 0
        ]

        # Initialize optimality cut components
        optimality_cut_lhs = 0
        optimality_cut_rhs = 0

        # Iterate over constraints and their dual multipliers
        for gurobi_constr, dual in duals_multiplier:
            rhs = gurobi_constr.RHS
            optimality_cut_rhs += dual * rhs

            lhs = gurobi_model.getRow(gurobi_constr)
            for i in range(lhs.size()):
                var = lhs.getVar(i)
                subproblem_var = self.map_variables_monolithic_gurobi[var.VarName]
                subproblem_var_name = subproblem_var["variable_name"]
                subproblem_var_coords = subproblem_var["variable_coords"]
                coeff = lhs.getCoeff(i)
                if subproblem_var_name in self.master_model.model.variables:
                    master_variable = self.master_model.model.variables[subproblem_var_name].sel(subproblem_var_coords)
                    optimality_cut_lhs += dual * (master_variable * coeff)

        end_time = time.time()
        construction_time = end_time - starting_time
        new_row = pd.DataFrame(
            {"iteration": [iteration], "constraint_type": ["optimality"], "construction_time_sec": [construction_time]}
        )
        self.constraints_construction_time = pd.concat([self.constraints_construction_time, new_row], ignore_index=True)
        return optimality_cut_lhs, optimality_cut_rhs

    def define_list_of_optimality_cuts(self, iteration) -> list:
        """
        Define the list of optimality cuts when dealing with multiple subproblems.

        :return: list of optimality cuts as tuples with the following structure:
            - subproblem_name: name of the subproblem (if the scenario name is not defined, it is set to "default")
            - optimality_cut_lhs: left-hand side of the optimality cut
            - optimality_cut_rhs: right-hand side of the optimality cut
        """
        logging.info("--- Generating optimality cut ---")
        optimality_cuts = [
            (
                subproblem.scenario_name if subproblem.scenario_name else "default",
                *self.generate_optimality_cut(self.subproblem_to_gurobi(subproblem), iteration),
            )
            for subproblem in self.subproblem_models
        ]
        return optimality_cuts

    def add_optimality_cuts_to_master(self, optimality_cuts, optimality_iteration, iteration):
        """
        Add the optimality cuts to the master model.

        :param optimality_cuts: list of optimality cuts as tuples with the following structure:
            - subproblem_name: name of the subproblem
            - optimality_cut_lhs: left-hand side of the optimality cut
            - optimality_cut_rhs: right-hand side of the optimality cut
        :param optimality_iteration: current iteration of the Benders Decomposition method (type: int)
        """
        starting_time = time.time()
        logging.info("--- Adding optimality cut to master model")
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

        end_time = time.time()
        addition_time = end_time - starting_time
        new_row = pd.DataFrame(
            {"iteration": [iteration], "constraint_type": ["optimality"], "addition_time_sec": [addition_time]}
        )
        self.constraints_addition_time = pd.concat([self.constraints_addition_time, new_row], ignore_index=True)

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
            logging.info("\n%s", tabulate(table, headers, tablefmt="grid", floatfmt=".4f"))
        return termination_criteria

    def remove_mock_variables(self):
        """
        Remove the mock variables from the subproblems or the theta variable from the master problem if they exist.
        This is done in order to retrieve the same problem configuration as before than the scaling process.
        Consequently, the user is able to save the solution of the problem in the output folder.
        """
        if "outer_approximation" in self.master_model.model.variables:
            self.master_model.model.variables.remove("outer_approximation")
        for subproblem in self.subproblem_models:
            if "mock_objective_subproblem" in subproblem.model.variables:
                subproblem.model.variables.remove("mock_objective_subproblem")

    def save_master_and_subproblems(self):
        """
        Save the master model and the subproblem models in respective output folders.

        :param iteration: current iteration of the Benders Decomposition method (type: int)
        """
        logging.info("--- Optimal solution found. Terminating iterations ---")
        # Re-scale the variables if scaling is used
        self.remove_mock_variables()
        if self.config.solver["use_scaling"]:
            self.master_model.scaling.re_scale()
            for subproblem in self.subproblem_models:
                subproblem.scaling.re_scale()

        # Write the results
        Postprocess(model=self.master_model, scenarios="", model_name=self.master_model.model_name, subfolder=Path(""))

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

        self.building_subproblem.to_csv(os.path.join(self.benders_output_folder, "building_subproblem.csv"))
        self.solving_subproblem.to_csv(os.path.join(self.benders_output_folder, "solving_subproblem.csv"))
        self.optimality_gap_df_infeasibility.to_csv(
            os.path.join(self.benders_output_folder, "optimality_gap_infeasibility.csv")
        )
        self.optimality_gap_df_optimal.to_csv(os.path.join(self.benders_output_folder, "optimality_gap_optimal.csv"))
        self.cuts_counter_df = pd.DataFrame(
            {
                "number_of_monolithic_constraints": [self.monolithic_model.model.constraints.ncons],
                "number_of_feasibility_cuts": [self.feasibility_cuts_counter],
                "number_of_optimality_cuts": [self.optimality_cuts_counter],
            }
        )
        self.cuts_counter_df.to_csv(os.path.join(self.benders_output_folder, "cuts_counter.csv"))
        self.constraints_added.to_csv(os.path.join(self.benders_output_folder, "cuts_added.csv"))

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

            logging.info("--- Solving master problem, fixing design variables in subproblems and solve them ---")
            if self.use_monolithic_solution and iteration == 1:
                self.solve_master_model_with_monolithic_solution(iteration)
            else:
                self.solve_master_model(iteration)
            self.fix_design_variables_in_subproblem_model()

            self.solve_subproblems_models(iteration)

            if any(subproblem.model.termination_condition != "optimal" for subproblem in self.subproblem_models):
                logging.info("--- Subproblems are infeasible ---")
                feasibility_cuts = self.define_list_of_feasibility_cuts(iteration)
                oscillatory_behavior = self.check_feasibility_cut_effectiveness(
                    feasibility_iteration, feasibility_cuts, iteration
                )
                if not oscillatory_behavior:
                    self.add_feasibility_cuts_to_master(feasibility_cuts, feasibility_iteration, iteration)
                feasibility_iteration += 1

            if all(subproblem.model.termination_condition == "optimal" for subproblem in self.subproblem_models):
                logging.info("--- All the subproblems are optimal ---")
                if self.master_model.only_feasibility_checks:
                    continue_iterations = False
                else:
                    optimality_cuts = self.define_list_of_optimality_cuts(iteration)
                    self.add_optimality_cuts_to_master(optimality_cuts, optimality_iteration, iteration)
                    optimality_iteration += 1
                    termination_criteria = self.check_termination_criteria(iteration)
                    if all(value for _, value in termination_criteria):
                        continue_iterations = False

            if continue_iterations is False:
                self.save_master_and_subproblems()

            if iteration == max_number_of_iterations:
                logging.info("--- Maximum number of iterations reached ---")
                logging.info("--- Saving possible results ---")
                self.building_subproblem.to_csv(os.path.join(self.benders_output_folder, "building_subproblem.csv"))
                self.solving_subproblem.to_csv(os.path.join(self.benders_output_folder, "solving_subproblem.csv"))
                self.optimality_gap_df_infeasibility.to_csv(
                    os.path.join(self.benders_output_folder, "optimality_gap_infeasibility.csv")
                )
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

            iteration += 1
