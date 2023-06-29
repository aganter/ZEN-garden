"""===========================================================================================================================================================================
Title:          ZEN-GARDEN
Created:        January-2022
Authors:        Jacob Mannhardt (jmannhardt@ethz.ch)
Organization:   Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:    Class defining a standard EnergySystem. Contains methods to add parameters, variables and constraints to the
                optimization problem. Parent class of the Carrier and Technology classes .The class takes the abstract
                optimization model as an input.
==========================================================================================================================================================================="""
import copy
import logging

import numpy as np
import pyomo.environ as pe

from zen_garden.preprocess.functions.extract_input_data import DataInput
from zen_garden.preprocess.functions.unit_handling import UnitHandling
from .time_steps import TimeStepsDicts


class EnergySystem:

    def __init__(self, optimization_setup):
        """ initialization of the energy_system
        :param optimization_setup: The OptimizationSetup of the EnergySystem class"""

        # the name
        self.name = "energy_system"
        # set attributes
        self.optimization_setup = optimization_setup
        # quick access
        self.system = self.optimization_setup.system
        # empty dict of technologies of carrier
        self.dict_technology_of_carrier = {}
        # The timesteps
        self.time_steps = TimeStepsDicts()

        # empty list of indexing sets
        self.indexing_sets = []

        # set indexing sets
        for key in self.system:
            if "set" in key:
                self.indexing_sets.append(key)

        # set input path
        _folder_label = self.optimization_setup.analysis["folder_name_system_specification"]
        self.input_path = self.optimization_setup.paths[_folder_label]["folder"]

        # create UnitHandling object
        self.unit_handling = UnitHandling(self.input_path,
                                          self.optimization_setup.solver["rounding_decimal_points"],
                                          self.optimization_setup.solver["define_ton_as_metric_ton"])

        # create DataInput object
        self.data_input = DataInput(element=self, system=self.system,
                                    analysis=self.optimization_setup.analysis, solver=self.optimization_setup.solver,
                                    energy_system=self, unit_handling=self.unit_handling)

        # store input data
        self.store_input_data()

        # create the rules
        self.rules = EnergySystemRules(optimization_setup)

    def store_input_data(self):
        """ retrieves and stores input data for element as attributes. Each Child class overwrites method to store different attributes """

        # in class <EnergySystem>, all sets are constructed
        self.set_nodes = self.data_input.extract_locations()
        self.set_country_nodes = self.data_input.extract_locations(extract_nodes="country")
        self.set_nodes_on_edges = self.calculate_edges_from_nodes()
        self.set_edges = list(self.set_nodes_on_edges.keys())
        self.set_carriers = []
        self.set_technologies = self.system["set_technologies"]
        # base time steps
        self.set_base_time_steps = list(range(0, self.system["unaggregated_time_steps_per_year"] * self.system["optimized_years"]))
        self.set_base_time_steps_yearly = list(range(0, self.system["unaggregated_time_steps_per_year"]))

        # yearly time steps
        self.set_time_steps_yearly = list(range(self.system["optimized_years"]))
        self.set_time_steps_yearly_entire_horizon = copy.deepcopy(self.set_time_steps_yearly)
        time_steps_yearly_duration = self.time_steps.calculate_time_step_duration(self.set_time_steps_yearly, self.set_base_time_steps)
        self.sequence_time_steps_yearly = np.concatenate([[time_step] * time_steps_yearly_duration[time_step] for time_step in time_steps_yearly_duration])
        self.time_steps.set_sequence_time_steps(None, self.sequence_time_steps_yearly, time_step_type="yearly")
        # list containing simulated years (needed for convert_real_to_generic_time_indices() in extract_input_data.py)
        self.set_time_steps_years = list(range(self.system["reference_year"],self.system["reference_year"] + self.system["optimized_years"]*self.system["interval_between_years"],self.system["interval_between_years"]))
        # parameters whose time-dependant data should not be interpolated (for years without data) in the extract_input_data.py convertRealToGenericTimeIndices() function
        self.parameters_interpolation_off = self.data_input.read_input_data("parameters_interpolation_off")
        # technology-specific
        self.set_conversion_technologies = self.system["set_conversion_technologies"]
        self.set_transport_technologies = self.system["set_transport_technologies"]
        self.set_storage_technologies = self.system["set_storage_technologies"]
        # carbon emissions limit
        self.carbon_emissions_limit = self.data_input.extract_input_data("carbon_emissions_limit", index_sets=["set_time_steps_yearly"], time_steps=self.set_time_steps_yearly)
        _fraction_year = self.system["unaggregated_time_steps_per_year"] / self.system["total_hours_per_year"]
        self.carbon_emissions_limit = self.carbon_emissions_limit * _fraction_year  # reduce to fraction of year
        self.carbon_emissions_budget = self.data_input.extract_input_data("carbon_emissions_budget", index_sets=[])
        self.carbon_emissions_cumulative_existing = self.data_input.extract_input_data("carbon_emissions_cumulative_existing", index_sets=[])
        # price carbon emissions
        self.price_carbon_emissions = self.data_input.extract_input_data("price_carbon_emissions", index_sets=["set_time_steps_yearly"], time_steps=self.set_time_steps_yearly)
        self.price_carbon_emissions_overshoot = self.data_input.extract_input_data("price_carbon_emissions_overshoot", index_sets=[])
        # market share unbounded
        self.market_share_unbounded = self.data_input.extract_input_data("market_share_unbounded", index_sets=[])
        # knowledge_spillover_rate
        self.knowledge_spillover_rate = self.data_input.extract_input_data("knowledge_spillover_rate", index_sets=[])
        # LCA impact categories
        self.set_lca_impact_categories = self.system['set_lca_impact_categories']

    def calculate_edges_from_nodes(self):
        """ calculates set_nodes_on_edges from set_nodes
        :return set_nodes_on_edges: dict with edges and corresponding nodes """

        set_nodes_on_edges = {}
        # read edge file
        set_edges_input = self.data_input.extract_locations(extract_nodes=False)
        if set_edges_input is not None:
            for edge in set_edges_input.index:
                set_nodes_on_edges[edge] = (set_edges_input.loc[edge, "node_from"], set_edges_input.loc[edge, "node_to"])
        else:
            logging.warning(f"DeprecationWarning: Implicit creation of edges will be deprecated. Provide 'set_edges.csv' in folder '{self.system['''folder_name_system_specification''']}' instead!")
            for node_from in self.set_nodes:
                for node_to in self.set_nodes:
                    if node_from != node_to:
                        set_nodes_on_edges[node_from + "-" + node_to] = (node_from, node_to)
        return set_nodes_on_edges

    def set_technology_of_carrier(self, technology, list_technology_of_carrier):
        """ appends technology to carrier in dict_technology_of_carrier
        :param technology: name of technology in model
        :param list_technology_of_carrier: list of carriers correspondent to technology"""
        for carrier in list_technology_of_carrier:
            if carrier not in self.dict_technology_of_carrier:
                self.dict_technology_of_carrier[carrier] = [technology]
                self.set_carriers.append(carrier)
            elif technology not in self.dict_technology_of_carrier[carrier]:
                self.dict_technology_of_carrier[carrier].append(technology)

    def calculate_connected_edges(self, node, direction: str):
        """ calculates connected edges going in (direction = 'in') or going out (direction = 'out')
        :param node: current node, connected by edges
        :param direction: direction of edges, either in or out. In: node = endnode, out: node = startnode
        :return _set_connected_edges: list of connected edges """
        if direction == "in":
            # second entry is node into which the flow goes
            _set_connected_edges = [edge for edge in self.set_nodes_on_edges if self.set_nodes_on_edges[edge][1] == node]
        elif direction == "out":
            # first entry is node out of which the flow starts
            _set_connected_edges = [edge for edge in self.set_nodes_on_edges if self.set_nodes_on_edges[edge][0] == node]
        else:
            raise KeyError(f"invalid direction '{direction}'")
        return _set_connected_edges

    def calculate_reversed_edge(self, edge):
        """ calculates the reversed edge corresponding to an edge
        :param edge: input edge
        :return _reversed_edge: edge which corresponds to the reversed direction of edge"""
        _node_out, _node_in = self.set_nodes_on_edges[edge]
        for _reversed_edge in self.set_nodes_on_edges:
            if _node_out == self.set_nodes_on_edges[_reversed_edge][1] and _node_in == self.set_nodes_on_edges[_reversed_edge][0]:
                return _reversed_edge
        raise KeyError(f"Edge {edge} has no reversed edge. However, at least one transport technology is bidirectional")

    ### --- classmethods to construct sets, parameters, variables, and constraints, that correspond to EnergySystem --- ###

    def construct_sets(self):
        """ constructs the pe.Sets of the class <EnergySystem> """
        # construct pe.Sets of the class <EnergySystem>
        pyomo_model = self.optimization_setup.model
        # nodes
        pyomo_model.set_nodes = pe.Set(initialize=self.set_nodes, doc='Set of nodes')
        # nodes
        pyomo_model.set_country_nodes = pe.Set(initialize=self.set_country_nodes, doc='Set of nodes')
        # edges
        pyomo_model.set_edges = pe.Set(initialize=self.set_edges, doc='Set of edges')
        # nodes on edges
        pyomo_model.set_nodes_on_edges = pe.Set(pyomo_model.set_edges, initialize=self.set_nodes_on_edges, doc='Set of nodes that constitute an edge. Edge connects first node with second node.')
        # carriers
        pyomo_model.set_carriers = pe.Set(initialize=self.set_carriers, doc='Set of carriers')
        # technologies
        pyomo_model.set_technologies = pe.Set(initialize=self.set_technologies, doc='Set of technologies')
        # all elements
        pyomo_model.set_elements = pe.Set(initialize=pyomo_model.set_technologies | pyomo_model.set_carriers, doc='Set of elements')
        # set set_elements to indexing_sets
        self.indexing_sets.append("set_elements")
        # time-steps
        pyomo_model.set_base_time_steps = pe.Set(initialize=self.set_base_time_steps, doc='Set of base time-steps')
        # yearly time steps
        pyomo_model.set_time_steps_yearly = pe.Set(initialize=self.set_time_steps_yearly, doc='Set of yearly time-steps')
        # yearly time steps of entire optimization horizon
        pyomo_model.set_time_steps_yearly_entire_horizon = pe.Set(initialize=self.set_time_steps_yearly_entire_horizon, doc='Set of yearly time-steps of entire optimization horizon')
        # impact categories for LCA, only if flag to include LCA categories is True
        if self.system['load_lca_factors']:
            pyomo_model.set_lca_impact_categories = pe.Set(initialize=self.set_lca_impact_categories, doc='Set of the LCIA impact categories to be investigated')

    def construct_params(self):
        """ constructs the pe.Params of the class <EnergySystem> """
        # carbon emissions limit
        cls = self.__class__
        parameters = self.optimization_setup.parameters
        pyomo_model = self.optimization_setup.model
        parameters.add_parameter(name="carbon_emissions_limit", data=self.optimization_setup.initialize_component(cls, "carbon_emissions_limit", set_time_steps=pyomo_model.set_time_steps_yearly),
            doc='Parameter which specifies the total limit on carbon emissions')
        # carbon emissions budget
        parameters.add_parameter(name="carbon_emissions_budget", data=self.optimization_setup.initialize_component(cls, "carbon_emissions_budget"),
            doc='Parameter which specifies the total budget of carbon emissions until the end of the entire time horizon')
        # carbon emissions budget
        parameters.add_parameter(name="carbon_emissions_cumulative_existing", data=self.optimization_setup.initialize_component(cls, "carbon_emissions_cumulative_existing"), doc='Parameter which specifies the total previous carbon emissions')
        # carbon price
        parameters.add_parameter(name="price_carbon_emissions", data=self.optimization_setup.initialize_component(cls, "price_carbon_emissions", set_time_steps=pyomo_model.set_time_steps_yearly),
            doc='Parameter which specifies the yearly carbon price')
        # carbon price of overshoot
        parameters.add_parameter(name="price_carbon_emissions_overshoot", data=self.optimization_setup.initialize_component(cls, "price_carbon_emissions_overshoot"), doc='Parameter which specifies the carbon price for budget overshoot')
        # carbon price of overshoot
        parameters.add_parameter(name="market_share_unbounded", data=self.optimization_setup.initialize_component(cls, "market_share_unbounded"),
                                                    doc='Parameter which specifies the unbounded market share')
        # carbon price of overshoot
        parameters.add_parameter(name="knowledge_spillover_rate", data=self.optimization_setup.initialize_component(cls, "knowledge_spillover_rate"),
                                                    doc='Parameter which specifies the knowledge spillover rate')

    def construct_vars(self):
        """ constructs the pe.Vars of the class <EnergySystem> """
        variables = self.optimization_setup.variables
        pyomo_model = self.optimization_setup.model
        # carbon emissions
        variables.add_variable(pyomo_model, name="carbon_emissions_total", index_sets=pyomo_model.set_time_steps_yearly, domain=pe.Reals, doc="total carbon emissions of energy system")
        # cumulative carbon emissions
        variables.add_variable(pyomo_model, name="carbon_emissions_cumulative", index_sets=pyomo_model.set_time_steps_yearly, domain=pe.Reals, doc="cumulative carbon emissions of energy system over time for each year")
        # carbon emission overshoot
        variables.add_variable(pyomo_model, name="carbon_emissions_overshoot", index_sets=pyomo_model.set_time_steps_yearly, domain=pe.NonNegativeReals, doc="overshoot carbon emissions of energy system at the end of the time horizon")
        # cost of carbon emissions
        variables.add_variable(pyomo_model, name="cost_carbon_emissions_total", index_sets=pyomo_model.set_time_steps_yearly, domain=pe.Reals, doc="total cost of carbon emissions of energy system")
        # costs
        variables.add_variable(pyomo_model, name="cost_total", index_sets=pyomo_model.set_time_steps_yearly, domain=pe.Reals, doc="total cost of energy system")
        # net_present_cost
        variables.add_variable(pyomo_model, name="net_present_cost", index_sets=pyomo_model.set_time_steps_yearly, domain=pe.Reals, doc="net_present_cost of energy system")

    def construct_constraints(self):
        """ constructs the pe.Constraints of the class <EnergySystem> """
        constraints = self.optimization_setup.constraints
        pyomo_model = self.optimization_setup.model
        # carbon emissions
        constraints.add_constraint(pyomo_model, name="constraint_carbon_emissions_total", index_sets=pyomo_model.set_time_steps_yearly, rule=self.rules.constraint_carbon_emissions_total_rule,
            doc="total carbon emissions of energy system")
        # carbon emissions
        constraints.add_constraint(pyomo_model, name="constraint_carbon_emissions_cumulative", index_sets=pyomo_model.set_time_steps_yearly, rule=self.rules.constraint_carbon_emissions_cumulative_rule,
            doc="cumulative carbon emissions of energy system over time")
        # cost of carbon emissions
        constraints.add_constraint(pyomo_model, name="constraint_carbon_cost_total", index_sets=pyomo_model.set_time_steps_yearly, rule=self.rules.constraint_carbon_cost_total_rule, doc="total carbon cost of energy system")
        # carbon emissions
        constraints.add_constraint(pyomo_model, name="constraint_carbon_emissions_limit", index_sets=pyomo_model.set_time_steps_yearly, rule=self.rules.constraint_carbon_emissions_limit_rule,
            doc="limit of total carbon emissions of energy system")
        # carbon emission budget
        constraints.add_constraint(pyomo_model, name="constraint_carbon_emissions_budget", index_sets=pyomo_model.set_time_steps_yearly, rule=self.rules.constraint_carbon_emissions_budget_rule,
            doc="Budget of total carbon emissions of energy system")
        # limit carbon emission overshoot
        constraints.add_constraint(pyomo_model, name="constraint_carbon_emissions_overshoot_limit", index_sets=pyomo_model.set_time_steps_yearly, rule=self.rules.constraint_carbon_emissions_overshoot_limit_rule,
            doc="Limit of overshot carbon emissions of energy system")
        # costs
        constraints.add_constraint(pyomo_model, name="constraint_cost_total", index_sets=pyomo_model.set_time_steps_yearly, rule=self.rules.constraint_cost_total_rule, doc="total cost of energy system")
        # net_present_cost
        constraints.add_constraint(pyomo_model, name="constraint_net_present_cost", index_sets=pyomo_model.set_time_steps_yearly, rule=self.rules.constraint_net_present_cost_rule, doc="net_present_cost of energy system")

    def construct_objective(self):
        """ constructs the pe.Objective of the class <EnergySystem> """
        logging.info("Construct pe.Objective")

        # get selected objective rule
        if self.optimization_setup.analysis["objective"] == "total_cost":
            objective_rule = self.rules.objective_total_cost_rule
        elif self.optimization_setup.analysis["objective"] == "total_carbon_emissions":
            objective_rule = self.rules.objective_total_carbon_emissions_rule
        elif self.optimization_setup.analysis["objective"] == "risk":
            logging.info("Objective of minimizing risk not yet implemented")
            objective_rule = self.rules.objective_risk_rule
        else:
            raise KeyError(f"Objective type {self.optimization_setup.analysis['objective']} not known")

        # get selected objective sense
        if self.optimization_setup.analysis["sense"] == "minimize":
            objective_sense = pe.minimize
        elif self.optimization_setup.analysis["sense"] == "maximize":
            objective_sense = pe.maximize
        else:
            raise KeyError(f"Objective sense {self.optimization_setup.analysis['sense']} not known")

        # construct objective
        self.optimization_setup.model.objective = pe.Objective(rule=objective_rule, sense=objective_sense)


class EnergySystemRules:
    """
    This class takes care of the rules for the EnergySystem
    """

    def __init__(self, optimization_setup):
        """
        Inits the constraints for a given energy syste,
        optimization_setup
        """

        self.optimization_setup = optimization_setup

    def constraint_carbon_emissions_total_rule(self, model, year):
        """ add up all carbon emissions from technologies and carriers """
        return (model.carbon_emissions_total[year] ==
                # technologies
                model.carbon_emissions_technology_total[year] +
                # carriers
                model.carbon_emissions_carrier_total[year])

    def constraint_carbon_emissions_cumulative_rule(self, model, year):
        """ cumulative carbon emissions over time """
        # get parameter object
        params = self.optimization_setup.parameters
        interval_between_years = self.optimization_setup.system["interval_between_years"]
        if year == model.set_time_steps_yearly.at(1):
            return (model.carbon_emissions_cumulative[year] == (model.carbon_emissions_total[year] - model.carbon_emissions_overshoot[year]) + params.carbon_emissions_cumulative_existing)
        else:
            return (model.carbon_emissions_cumulative[year] == model.carbon_emissions_cumulative[year - 1] + (model.carbon_emissions_total[year - 1] - model.carbon_emissions_overshoot[year - 1]) * (interval_between_years - 1) +
                    (model.carbon_emissions_total[year]-model.carbon_emissions_overshoot[year]))

    def constraint_carbon_cost_total_rule(self, model, year):
        """ carbon cost associated with the carbon emissions of the system in each year """
        # get parameter object
        params = self.optimization_setup.parameters
        # return (model.cost_carbon_emissions_total[year] == params.price_carbon_emissions[year] * model.carbon_emissions_total[year]  # add overshoot price
        #         + model.carbon_emissions_overshoot[year] * params.price_carbon_emissions_overshoot )
        # TODO this constraint is adjusted for country limit analysis!
        if params.price_carbon_emissions_overshoot != np.inf:
           return (model.cost_carbon_emissions_total[year] == params.price_carbon_emissions[year] * model.carbon_emissions_total[year] # add overshoot price
                   + model.carbon_emissions_overshoot[year] * params.price_carbon_emissions_overshoot)
        else:
           return (model.cost_carbon_emissions_total[year] == params.price_carbon_emissions[year] * model.carbon_emissions_total[year])

    def constraint_carbon_emissions_limit_rule(self, model, year):
        """ time dependent carbon emissions limit from technologies and carriers"""
        # get parameter object
        params = self.optimization_setup.parameters
        if params.carbon_emissions_limit[year] != np.inf:
            return (params.carbon_emissions_limit[year] >= model.carbon_emissions_total[year] - model.carbon_emissions_overshoot[year])
        else:
            return pe.Constraint.Skip

    def constraint_carbon_emissions_budget_rule(self, model, year):
        """ carbon emissions budget of entire time horizon from technologies and carriers.
        The prediction extends until the end of the horizon, i.e.,
        last optimization time step plus the current carbon emissions until the end of the horizon """
        # get parameter object
        params = self.optimization_setup.parameters
        interval_between_years = self.optimization_setup.system["interval_between_years"]
        if params.carbon_emissions_budget != np.inf:
            max_budget = max(params.carbon_emissions_budget, params.carbon_emissions_cumulative_existing)
            if year == model.set_time_steps_yearly_entire_horizon.at(-1):
                return (max_budget >= model.carbon_emissions_cumulative[year])
            else:
                return (max_budget >= model.carbon_emissions_cumulative[year] + (model.carbon_emissions_total[year] - model.carbon_emissions_overshoot[year]) * (interval_between_years - 1))
        else:
            return pe.Constraint.Skip

    def constraint_carbon_emissions_overshoot_limit_rule(self, model, year):
        """ ensure that overshoot is lower or equal to total carbon emissions -> overshoot cannot be banked """
        params = self.optimization_setup.parameters
        if params.price_carbon_emissions_overshoot != np.inf:
            return (model.carbon_emissions_total[year] >= model.carbon_emissions_overshoot[year])
        else:
            return(model.carbon_emissions_overshoot[year]==0)

    def constraint_cost_total_rule(self, model, year):
        """ add up all costs from technologies and carriers"""
        return (model.cost_total[year] ==
                # capex
                model.cost_capex_total[year] + # capex
                model.cost_opex_total[year] + # opex
                model.cost_carrier_total[year] + # carrier costs
                model.cost_carbon_emissions_total[year] # carbon costs
                )

    def constraint_net_present_cost_rule(self, model, year):
        """ discounts the annual capital flows to calculate the net_present_cost """
        system = self.optimization_setup.system
        discount_rate = self.optimization_setup.analysis["discount_rate"]
        if year == model.set_time_steps_yearly_entire_horizon.at(-1):
            interval_between_years = 1
        else:
            interval_between_years = system["interval_between_years"]

        return (model.net_present_cost[year] == model.cost_total[year] * sum(# economic discount
            ((1 / (1 + discount_rate)) ** (interval_between_years * (year - model.set_time_steps_yearly.at(1)) + _intermediate_time_step)) for _intermediate_time_step in range(0, interval_between_years)))

    # objective rules
    def objective_total_cost_rule(self, model):
        """objective function to minimize the total net present cost"""
        return (sum(model.net_present_cost[year] for year in model.set_time_steps_yearly))

    def objective_net_present_cost_rule(self, model):
        """ objective function to minimize NPV """
        return (sum(model.net_present_cost[year] for year in model.set_time_steps_yearly))

    def objective_total_carbon_emissions_rule(self, model):
        """objective function to minimize total emissions"""
        return (sum(model.carbon_emissions_total[year] for year in model.set_time_steps_yearly))

    def objective_risk_rule(self, model):
        """objective function to minimize total risk"""
        # TODO implement objective functions for risk
        return pe.Constraint.Skip
