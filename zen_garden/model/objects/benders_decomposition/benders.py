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
from gurobipy import GRB

from zen_garden.preprocess.extract_input_data import DataInput
from zen_garden.model.optimization_setup import OptimizationSetup
from zen_garden.model.objects.benders_decomposition.master_problem import MasterProblem
from zen_garden.model.objects.benders_decomposition.subproblems import Subproblem


class BendersDecomposition:
    """
    Class defining the Benders Decomposition method.
    Solve the MILPs by decomposing the problem into a master problem and a set of subproblems and utilizing the
    cutting-plane method to solve the problem.

    min f(x,y)
    s.t. A*x + B*y <= b (m constraints)
         D*y >= d       (n constraints)
         x >= 0         (Nx-dimensional vector of operational variables)
         y >= 0         (Ny-dimensional vector of design variables)
    """

    label = "BendersDecomposition"

    def __init__(
        self,
        config: dict,
        analysis: dict,
        config_benders: dict,
        monolithic_problem: OptimizationSetup,
        save_first_problems: bool = False,
    ):

        self.name = "BendersDecomposition"
        self.config = config
        self.analysis = analysis
        self.config_benders = config_benders
        self.monolithic_problem = monolithic_problem
        self.save_first_problems = save_first_problems

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
        """
        Initialize the BendersDecomposition object.

        :param config: dictionary containing the configuration of the optimization problem
        :param analysis: dictionary containing the analysis configuration
        :param config_benders: dictionary containing the configuration of the Benders Decomposition method
        :param monolithic_problem: OptimizationSetup object of the monolithic problem
        """

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

        logging.info("")
        logging.info("Creating the master problem.")
        if self.save_first_problems:
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

            logging.info("")
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
                not_coupling_variables=self.not_coupling_variables,
                design_constraints=self.design_constraints,
                operational_constraints=self.operational_constraints,
            )

    def save_monolithic_problem_in_gurobi_format_map_vars_constrs(self):
        """
        Save the monolithic problem in the gurobi format.
        """
        self.monolithic_model_gurobi = self.monolithic_problem.model.to_gurobipy()
        self.monolithic_model_gurobi.write("gurobi_monolithic_model.lp")

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
            else:
                logging.warning("Constraint %s is not in the monolithic problem.", variable["variable_name"])

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

    def solve_master_problem(self):
        """
        Solve the master problem.
        """
        self.master_model.model.solve()

    def fix_design_variables_in_subproblem_model(self):
        """
        Fix the design variables of the subproblems to the optimal solution of the master problem.
        This function takes the solution of the master problem and fixes the values of the design variables in the
        subproblems by adding the corresponding upper and lower bounds to the variables.
        """
        for variable_name in self.master_model.model.variables:
            if variable_name in self.subproblem_models.model.variables:
                variable_solution = self.master_model.model.solution[variable_name]
                self.subproblem_models.model.variables[variable_name].lower = variable_solution
                self.subproblem_models.model.variables[variable_name].upper = variable_solution

    def subproblem_to_gurobi(self, subproblem_solved, iteration):
        """
        Convert the subproblem model to gurobi, set the parameter InfUnbdInfo to 1 and do the mapping of variables and
        constraints.
        """
        subproblem_model_fixed_design_variable_gurobi = subproblem_solved.to_gurobipy()
        subproblem_model_fixed_design_variable_gurobi.write(
            f"gurobi_subproblem_model_fixed_design_variable_{iteration}.lp"
        )
        subproblem_model_fixed_design_variable_gurobi.setParam(GRB.Param.InfUnbdInfo, 1)

        # Optimize gurobi model, compute IIS and save infeasible constraints
        subproblem_model_fixed_design_variable_gurobi.optimize()
        subproblem_model_fixed_design_variable_gurobi.computeIIS()
        infeasibilities = [
            constr
            for constr, infeas in zip(
                subproblem_model_fixed_design_variable_gurobi.getConstrs(),
                subproblem_model_fixed_design_variable_gurobi.getAttr(GRB.Attr.IISConstr),
            )
            if infeas
        ]

        return subproblem_model_fixed_design_variable_gurobi, infeasibilities

    def generate_feasibility_cut(self, subproblem_model_fixed_design_variable_gurobi):
        """
        Generate the feasibility cut.
        """
        farkas_multipliers = [
            (constraint_name, multiplier)
            for constraint_name, multiplier, infeas in zip(
                [constr_name.ConstrName for constr_name in subproblem_model_fixed_design_variable_gurobi.getConstrs()],
                subproblem_model_fixed_design_variable_gurobi.getAttr(GRB.Attr.FarkasDual),
                subproblem_model_fixed_design_variable_gurobi.getAttr(GRB.Attr.IISConstr),
            )
            if infeas
        ]

        # Create the feasibility cut
        feasibility_cut = 0
        rhs = []
        for gurobi_constr_name, farkas in farkas_multipliers:
            lp_constraint_subproblem_name = self.map_constraints_monolithic_gurobi[gurobi_constr_name][
                "constraint_name"
            ]
            lp_constraint_coords = self.map_constraints_monolithic_gurobi[gurobi_constr_name]["constraint_coords"]
            rhs_constant = (
                self.subproblem_models.model.constraints[lp_constraint_subproblem_name]
                .sel(lp_constraint_coords)
                .rhs.item()
                * farkas
            )
            rhs.append(rhs_constant)
            lhs_subproblem_constraint = (
                self.subproblem_models.model.constraints[lp_constraint_subproblem_name].sel(lp_constraint_coords).lhs
            )
            for i, var in enumerate(lhs_subproblem_constraint.vars):
                if var.values.item() != -1:
                    var_value = var.values.item()
                    mapped_variable = self.map_variables_monolithic_gurobi[f"x{var_value}"]

                    if mapped_variable["variable_name"] in self.master_model.model.variables:
                        master_variable_name = mapped_variable["variable_name"]
                        master_variable_coords = mapped_variable["variable_coords"]
                        master_variable = self.master_model.model.variables[master_variable_name].sel(
                            master_variable_coords
                        )
                        feasibility_cut += -farkas * (master_variable * lhs_subproblem_constraint.coeffs[i].item())

        for i, rhs_value in enumerate(rhs):
            if rhs_value != 0:
                feasibility_cut += rhs_value

        return feasibility_cut

    def fit(self):
        """
        Fit the Benders Decomposition model.
        """
        iteration = 1
        while True:
            logging.info("--- Iteration %s: Solving master problem ---", iteration)
            self.solve_master_problem()

            logging.info("--- Iteration %s: Fixing design variables in subproblem model ---", iteration)
            self.fix_design_variables_in_subproblem_model()

            logging.info("--- Iteration %s: Solving subproblem model ---", iteration)
            self.subproblem_models.model.solve()
            subproblem_solved = self.subproblem_models.model

            # Check if subproblem is optimal
            if self.subproblem_models.model.termination_condition == "optimal":
                logging.info("--- Subproblem is optimal. Terminating iterations ---")
                break

            logging.info("--- Iteration %s: Subproblem is infeasible ---", iteration)
            subproblem_model_fixed_design_variable_gurobi, infeasibilities = self.subproblem_to_gurobi(
                subproblem_solved, iteration
            )
            # Log the name of the infeasible constraints using the monolithic problem names
            infeasible_constraint_names_gurobi = [constr.ConstrName for constr in infeasibilities]
            infeasible_constraint_names_monolithic = [
                self.map_constraints_monolithic_gurobi[constr_name]["constraint_name"]
                for constr_name in infeasible_constraint_names_gurobi
            ]
            logging.info("--- Infeasible constraints: %s", infeasible_constraint_names_monolithic)
            logging.info("--- Generating feasibility cut")
            feasibility_cut = self.generate_feasibility_cut(subproblem_model_fixed_design_variable_gurobi)

            logging.info("--- Adding feasibility cut to master model")
            self.master_model.model.add_constraints(
                lhs=feasibility_cut, sign=">=", rhs=0, name=f"feasibility_cut_iteration_{iteration}"
            )

            iteration += 1
