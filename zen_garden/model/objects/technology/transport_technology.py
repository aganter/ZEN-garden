"""
:Title:          ZEN-GARDEN
:Created:        October-2021
:Authors:        Alissa Ganter (aganter@ethz.ch),
                Jacob Mannhardt (jmannhardt@ethz.ch)
:Organization:   Laboratory of Reliability and Risk Engineering, ETH Zurich

Class defining the parameters, variables and constraints that hold for all transport technologies.
The class takes the abstract optimization model as an input, and returns the parameters, variables and
constraints that hold for the transport technologies.
"""
import logging

import numpy as np
import xarray as xr

from .technology import Technology
from ..component import ZenIndex, IndexSet
from ..element import Element, GenericRule


class TransportTechnology(Technology):
    # set label
    label = "set_transport_technologies"
    location_type = "set_edges"

    def __init__(self, tech: str, optimization_setup):
        """init transport technology object

        :param tech: name of added technology
        :param optimization_setup: The OptimizationSetup the element is part of """
        super().__init__(tech, optimization_setup)
        # dict of reversed edges
        self.dict_reversed_edges = {}
        # store carriers of transport technology
        self.store_carriers()
        # # store input data
        # self.store_input_data()

    def store_carriers(self):
        """ retrieves and stores information on reference, input and output carriers """
        # get reference carrier from class <Technology>
        super().store_carriers()

    def store_input_data(self):
        """ retrieves and stores input data for element as attributes. Each Child class overwrites method to store different attributes """
        # get attributes from class <Technology>
        super().store_input_data()
        # set attributes for parameters of child class <TransportTechnology>
        self.distance = self.data_input.extract_input_data("distance", index_sets=["set_edges"])
        self.transport_loss_factor = self.data_input.extract_attribute("transport_loss_factor")["value"]
        # get capex of transport technology
        self.get_capex_transport()
        # annualize capex
        self.convert_to_fraction_of_capex()
        # calculate capex of existing capacity
        self.capex_capacity_existing = self.calculate_capex_of_capacities_existing()

    def get_capex_transport(self):
        """get capex of transport technology"""
        set_time_steps_yearly = self.energy_system.set_time_steps_yearly
        # check if there are separate capex for capacity and distance
        if self.optimization_setup.system['double_capex_transport']:
            # both capex terms must be specified
            self.capex_specific = self.data_input.extract_input_data("capex_specific", index_sets=["set_edges", "set_time_steps_yearly"], time_steps=set_time_steps_yearly)
            self.capex_per_distance_transport = self.data_input.extract_input_data("capex_per_distance_transport", index_sets=["set_edges", "set_time_steps_yearly"], time_steps=set_time_steps_yearly)
        else:  # Here only capex_specific is used, and capex_per_distance_transport is set to Zero.
            if self.data_input.exists_attribute("capex_per_distance_transport"):
                self.capex_per_distance_transport = self.data_input.extract_input_data("capex_per_distance_transport", index_sets=["set_edges", "set_time_steps_yearly"], time_steps=set_time_steps_yearly)
                self.capex_specific = self.capex_per_distance_transport * self.distance
                self.opex_specific_fixed = self.opex_specific_fixed * self.distance
            elif self.data_input.exists_attribute("capex_specific"):
                self.capex_specific = self.data_input.extract_input_data("capex_specific", index_sets=["set_edges", "set_time_steps_yearly"], time_steps=set_time_steps_yearly)
            else:
                raise AttributeError(f"The transport technology {self.name} has neither capex_per_distance_transport nor capex_specific attribute.")
            self.capex_per_distance_transport = self.capex_specific * 0.0

    def convert_to_fraction_of_capex(self):
        """ this method converts the total capex to fraction of capex, depending on how many hours per year are calculated """
        fraction_year = self.calculate_fraction_of_year()
        self.opex_specific_fixed = self.opex_specific_fixed * fraction_year
        self.capex_specific = self.capex_specific * fraction_year
        self.capex_per_distance_transport = self.capex_per_distance_transport * fraction_year

    def calculate_capex_of_single_capacity(self, capacity, index):
        """ this method calculates the annualized capex of a single existing capacity.

        :param capacity: #TODO describe parameter/return
        :param index: #TODO describe parameter/return
        :return: #TODO describe parameter/return
        """
        # TODO check existing capex of transport techs -> Hannes
        if np.isnan(self.capex_specific[index[0]].iloc[0]):
            return 0
        else:
            return self.capex_specific[index[0]].iloc[0] * capacity

    ### --- getter/setter classmethods
    def set_reversed_edge(self, edge, _reversed_edge):
        """ maps the reversed edge to an edge

        :param edge: #TODO describe parameter/return
        :param _reversed_edge: #TODO describe parameter/return
        """
        self.dict_reversed_edges[edge] = _reversed_edge

    def get_reversed_edge(self, edge):
        """ get the reversed edge corresponding to an edge

        :param edge: #TODO describe parameter/return
        :return: #TODO describe parameter/return
        """
        return self.dict_reversed_edges[edge]

    ### --- classmethods to construct sets, parameters, variables, and constraints, that correspond to TransportTechnology --- ###
    @classmethod
    def construct_sets(cls, optimization_setup):
        """ constructs the pe.Sets of the class <TransportTechnology>

        :param optimization_setup: The OptimizationSetup the element is part of """
        pass

    @classmethod
    def construct_params(cls, optimization_setup):
        """ constructs the pe.Params of the class <TransportTechnology>

        :param optimization_setup: The OptimizationSetup the element is part of """

        # distance between nodes
        optimization_setup.parameters.add_parameter(name="distance", data=optimization_setup.initialize_component(cls, "distance", index_names=["set_transport_technologies", "set_edges"]),
            doc='distance between two nodes for transport technologies')
        # capital cost per unit
        optimization_setup.parameters.add_parameter(name="capex_specific_transport",
            data=optimization_setup.initialize_component(cls, "capex_specific", index_names=["set_transport_technologies", "set_edges", "set_time_steps_yearly"]),
            doc='capex per unit for transport technologies')
        # capital cost per distance
        optimization_setup.parameters.add_parameter(name="capex_per_distance_transport",
            data=optimization_setup.initialize_component(cls, 'capex_per_distance_transport', index_names=['set_transport_technologies', "set_edges", "set_time_steps_yearly"]),
            doc='capex per distance for transport technologies')
        # carrier losses
        optimization_setup.parameters.add_parameter(name="transport_loss_factor", data=optimization_setup.initialize_component(cls, "transport_loss_factor", index_names=["set_transport_technologies"]),
            doc='carrier losses due to transport with transport technologies')

    @classmethod
    def construct_vars(cls, optimization_setup):
        """ constructs the pe.Vars of the class <TransportTechnology>

        :param optimization_setup: The OptimizationSetup the element is part of """
        model = optimization_setup.model
        variables = optimization_setup.variables
        sets = optimization_setup.sets

        def flow_transport_bounds(index_values, index_list):
            """ return bounds of carrier_flow for bigM expression
            :param index_values: list of tuples with the index values
            :param index_list: The names of the indices
            :return bounds: bounds of carrier_flow"""

            # get the arrays
            tech_arr, edge_arr, time_arr = sets.tuple_to_arr(index_values, index_list)
            # convert operationTimeStep to time_step_year: operationTimeStep -> base_time_step -> time_step_year
            time_step_year = xr.DataArray([optimization_setup.energy_system.time_steps.convert_time_step_operation2year(tech, time) for tech, time in zip(tech_arr.data, time_arr.data)])

            lower = model.variables["capacity"].lower.loc[tech_arr, "power", edge_arr, time_step_year].data
            upper = model.variables["capacity"].upper.loc[tech_arr, "power", edge_arr, time_step_year].data
            return np.stack([lower, upper], axis=-1)

        # flow of carrier on edge
        index_values, index_names = cls.create_custom_set(["set_transport_technologies", "set_edges", "set_time_steps_operation"], optimization_setup)
        bounds = flow_transport_bounds(index_values, index_names)
        variables.add_variable(model, name="flow_transport", index_sets=(index_values, index_names),
            bounds=bounds, doc='carrier flow through transport technology on edge i and time t')
        # loss of carrier on edge
        variables.add_variable(model, name="flow_transport_loss", index_sets=(index_values, index_names), bounds=(0,np.inf),
            doc='carrier flow through transport technology on edge i and time t')

    @classmethod
    def construct_constraints(cls, optimization_setup):
        """ constructs the pe.Constraints of the class <TransportTechnology>

        :param optimization_setup: The OptimizationSetup the element is part of """
        model = optimization_setup.model
        constraints = optimization_setup.constraints
        rules = TransportTechnologyRules(optimization_setup)
        # Carrier Flow Losses
        constraints.add_constraint_block(model, name="constraint_transport_technology_losses_flow",
                                         constraint=rules.constraint_transport_technology_losses_flow_block(),
                                         doc='Carrier loss due to transport with through transport technology')
        # anyaxie
        if not optimization_setup.system['use_endogenous_learning']:
            # capex of transport technologies
            constraints.add_constraint_block(model, name="constraint_transport_technology_capex",
                                             constraint=rules.constraint_transport_technology_capex_block(),
                                             doc='Capital expenditures for installing transport technology')

    # defines disjuncts if technology on/off
    @classmethod
    def disjunct_on_technology_rule(cls, optimization_setup, tech, capacity_type, edge, time, binary_var):
        """definition of disjunct constraints if technology is on

        :param optimization_setup: #TODO describe parameter/return
        :param tech: #TODO describe parameter/return
        :param capacity_type: #TODO describe parameter/return:
        :param edge: #TODO describe parameter/return
        :param time: #TODO describe parameter/return
        :param binary_var: #TODO describe parameter/return
        """
        model = optimization_setup.model
        # get parameter object
        params = optimization_setup.parameters
        constraints = optimization_setup.constraints
        # get invest time step
        time_step_year = optimization_setup.energy_system.time_steps.convert_time_step_operation2year(tech, time)

        # disjunct constraints min load
        constraints.add_constraint_block(model, name=f"disjunct_transport_technology_min_load_{tech}_{capacity_type}_{edge}_{time}",
                                         constraint=(model.variables["flow_transport"][tech, edge, time].to_linexpr()
                                                     - params.min_load.loc[tech, capacity_type, edge, time].item() * model.variables["capacity"][tech, capacity_type, edge, time_step_year]
                                                     >= 0),
                                         disjunction_var=binary_var)

    @classmethod
    def disjunct_off_technology_rule(cls, optimization_setup, tech, capacity_type, edge, time, binary_var):
        """definition of disjunct constraints if technology is off

        :param optimization_setup: #TODO describe parameter/return
        :param tech: #TODO describe parameter/return
        :param capacity_type: #TODO describe parameter/return #TODO describe parameter/return
        :param edge: #TODO describe parameter/return
        :param time: #TODO describe parameter/return
        :param binary_var: #TODO describe parameter/return
        """
        model = optimization_setup.model
        constraints = optimization_setup.constraints

        # since it is an equality con we add lower and upper bounds
        constraints.add_constraint_block(model, name=f"disjunct_transport_technology_off_{tech}_{capacity_type}_{edge}_{time}_lower",
                                         constraint=(model.variables["flow_transport"][tech, edge, time].to_linexpr()
                                                     == 0),
                                         disjunction_var=binary_var)


class TransportTechnologyRules(GenericRule):
    """
    Rules for the TransportTechnology class
    """

    def __init__(self, optimization_setup):
        """
        Inits the rules for a given EnergySystem

        :param optimization_setup: The OptimizationSetup the element is part of
        """

        super().__init__(optimization_setup)

    # Rule-based constraints
    # ----------------------

    # Block-based constraints
    # -----------------------

    def constraint_transport_technology_losses_flow_block(self):
        """compute the flow losses for a carrier through a transport technology

        .. math::
            \mathrm{if\ transport\ distance\ set\ to\ inf}\ F^\mathrm{l}_{j,e,t} = 0
        .. math::
            \mathrm{else}\ F^\mathrm{l}_{j,e,t} = h_{j,e} \\rho_{j} F_{j,e,t}

        :return: #TODO describe parameter/return
        """

        ### index sets
        # not necessary

        ### masks
        # This mask checks the distance between nodes
        mask = np.isinf(self.parameters.distance).astype(float)
        # This mask ensure we only get constraints where we want them
        cons_mask = self.variables["flow_transport_loss"].mask

        ### index loop
        # not necessary

        ### auxiliary calculations
        term_distance_inf = mask * self.variables["flow_transport_loss"]
        term_distance_not_inf = (1 - mask) * (self.variables["flow_transport_loss"] - self.parameters.transport_loss_factor * self.parameters.distance * self.variables["flow_transport"])

        ### formulate constraint
        lhs = term_distance_inf + term_distance_not_inf
        rhs = 0
        constraints = lhs == rhs

        ### return
        return self.constraints.return_contraints(constraints, mask=cons_mask)

    def constraint_transport_technology_capex_block(self):
        """ definition of the capital expenditures for the transport technology

        .. math::
            \mathrm{if\ transport\ distance\ set\ to\ inf}\ \Delta S_{h,p,y}^\mathrm{power} = 0
        .. math::
            \mathrm{else}\ CAPEX_{y,n,i}^\mathrm{cost, power} = \\Delta S_{h,p,y}^\mathrm{power} \\alpha_{j,n,y} + B_{i,p,y} h_{j,e} \\alpha^\mathrm{d}_{j,y}

        :return: #TODO describe parameter/return
        """

        ### index sets
        index_values, index_list = Element.create_custom_set(["set_transport_technologies", "set_edges", "set_time_steps_yearly"], self.optimization_setup)
        # check if we even need to continue
        if len(index_values) == 0:
            return []
        # get the coords
        coords = [self.variables.coords["set_transport_technologies"], self.variables.coords["set_edges"], self.variables.coords["set_time_steps_yearly"]]

        ### masks
        # This mask checks the distance between nodes for the condition
        mask = np.isinf(self.parameters.distance).astype(float)

        # This mask ensure we only get constraints where we want them
        index_arrs = IndexSet.tuple_to_arr(index_values, index_list)
        global_mask = xr.DataArray(False, coords=coords)
        global_mask.loc[index_arrs] = True

        ### index loop
        # not necessary

        ### auxiliary calculations
        term_distance_inf = mask * self.variables["capacity_addition"].loc[coords[0], "power", coords[1], coords[2]]
        term_distance_not_inf = (1 - mask) * (self.variables["cost_capex"].loc[coords[0], "power", coords[1], coords[2]]
                                              - self.variables["capacity_addition"].loc[coords[0], "power", coords[1], coords[2]] * self.parameters.capex_specific_transport.loc[coords[0], coords[1]])
        # we have an additional check here to avoid binary variables when their coefficient is 0
        if np.any(self.parameters.distance.loc[coords[0], coords[1]] * self.parameters.capex_per_distance_transport.loc[coords[0], coords[1]] != 0):
            term_distance_not_inf -= (1 - mask) * self.variables["technology_installation"].loc[coords[0], "power", coords[1], coords[2]] * (self.parameters.distance.loc[coords[0], coords[1]] * self.parameters.capex_per_distance_transport.loc[coords[0], coords[1]])

        ### formulate constraint
        lhs = term_distance_inf + term_distance_not_inf
        rhs = 0
        constraints = lhs == rhs

        ### return
        return self.constraints.return_contraints(constraints, mask=global_mask)