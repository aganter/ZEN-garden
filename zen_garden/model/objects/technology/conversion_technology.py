"""===========================================================================================================================================================================
Title:          ZEN-GARDEN
Created:        October-2021
Authors:        Alissa Ganter (aganter@ethz.ch)
                Jacob Mannhardt (jmannhardt@ethz.ch)
Organization:   Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:    Class defining the parameters, variables and constraints of the conversion technologies.
                The class takes the abstract optimization model as an input, and adds parameters, variables and
                constraints of the conversion technologies.
==========================================================================================================================================================================="""
import logging

import numpy as np
import pandas as pd
import pyomo.environ as pe

from .technology import Technology


class ConversionTechnology(Technology):
    # set label
    label = "set_conversion_technologies"
    location_type = "set_nodes"

    def __init__(self, tech, optimization_setup):
        """init conversion technology object
        :param tech: name of added technology
        :param optimization_setup: The OptimizationSetup the element is part of """

        logging.info(f'Initialize conversion technology {tech}')
        super().__init__(tech, optimization_setup)
        # store input data
        self.store_input_data()

    def store_input_data(self):
        """ retrieves and stores input data for element as attributes. Each Child class overwrites method to store different attributes """
        # get attributes from class <Technology>
        super().store_input_data()
        # define input and output carrier
        self.input_carrier = self.data_input.extract_conversion_carriers()["input_carrier"]
        self.output_carrier = self.data_input.extract_conversion_carriers()["output_carrier"]
        self.energy_system.set_technology_of_carrier(self.name, self.input_carrier + self.output_carrier)
        # check if reference carrier in input and output carriers and set technology to correspondent carrier
        assert self.reference_carrier[0] in (self.input_carrier + self.output_carrier), \
            f"reference carrier {self.reference_carrier} of technology {self.name} not in input and output carriers {self.input_carrier + self.output_carrier}"
        # get conversion efficiency and capex
        self.get_conversion_factor()
        self.convert_to_fraction_of_capex()

    def get_conversion_factor(self):
        """retrieves and stores conversion_factor for <ConversionTechnology>.
        Each Child class overwrites method to store different conversion_factor """
        # TODO read pwa Dict and set Params
        _pwa_conversion_factor, self.conversion_factor_is_pwa = self.data_input.extract_pwa_data("conversion_factor")
        if self.conversion_factor_is_pwa:
            self.pwa_conversion_factor = _pwa_conversion_factor
        else:
            self.conversion_factor_linear = _pwa_conversion_factor

    def convert_to_fraction_of_capex(self):
        """ this method retrieves the total capex and converts it to annualized capex """
        _pwa_capex, self.capex_is_pwa = self.data_input.extract_pwa_data("capex")
        # annualize cost_capex
        fraction_year = self.calculate_fraction_of_year()
        self.opex_specific_fixed = self.opex_specific_fixed * fraction_year
        if not self.capex_is_pwa:
            self.capex_specific = _pwa_capex["capex"] * fraction_year
        else:
            self.pwa_capex = _pwa_capex
            self.pwa_capex["capex"] = [value * fraction_year for value in self.pwa_capex["capex"]]
            # set bounds
            self.pwa_capex["bounds"]["capex"] = tuple([(bound * fraction_year) for bound in self.pwa_capex["bounds"]["capex"]])
        # calculate capex of existing capacity
        self.capex_capacity_existing = self.calculate_capex_of_capacities_existing()

    def calculate_capex_of_single_capacity(self, capacity, index):
        """ this method calculates the annualized capex of a single existing capacity. """
        if capacity == 0:
            return 0
        # linear
        if not self.capex_is_pwa:
            capex = self.capex_specific[index[0]].iloc[0] * capacity
        else:
            capex = np.interp(capacity, self.pwa_capex["capacity"], self.pwa_capex["capex"])
        return capex

    ### --- getter/setter classmethods
    @classmethod
    def get_capex_conversion_factor_all_elements(cls, optimization_setup, variable_type, selectPWA, index_names=None):
        """ similar to Element.get_attribute_of_all_elements but only for capex and conversion_factor.
        If selectPWA, extract pwa attributes, otherwise linear.
        :param optimization_setup: The OptimizationSetup the element is part of
        :param variable_type: either capex or conversion_factor
        :param selectPWA: boolean if get attributes for pwa
        :return dict_of_attributes: returns dict of attribute values """
        _class_elements = optimization_setup.get_all_elements(cls)
        dict_of_attributes = {}
        if variable_type == "capex":
            _is_pwa_attribute = "capex_is_pwa"
            _attribute_name_pwa = "pwa_capex"
            _attribute_name_linear = "capex_specific"
        elif variable_type == "conversion_factor":
            _is_pwa_attribute = "conversion_factor_is_pwa"
            _attribute_name_pwa = "pwa_conversion_factor"
            _attribute_name_linear = "conversion_factor_linear"
        else:
            raise KeyError("Select either 'capex' or 'conversion_factor'")
        for _element in _class_elements:
            # extract for pwa
            if getattr(_element, _is_pwa_attribute) and selectPWA:
                dict_of_attributes, _ = optimization_setup.append_attribute_of_element_to_dict(_element, _attribute_name_pwa, dict_of_attributes)
            # extract for linear
            elif not getattr(_element, _is_pwa_attribute) and not selectPWA:
                dict_of_attributes, _ = optimization_setup.append_attribute_of_element_to_dict(_element, _attribute_name_linear, dict_of_attributes)
            if not dict_of_attributes:
                _, index_names = cls.create_custom_set(index_names, optimization_setup)
                return (dict_of_attributes, index_names)
        dict_of_attributes = pd.concat(dict_of_attributes, keys=dict_of_attributes.keys())
        if not index_names:
            logging.warning(f"Initializing a parameter ({variable_type}) without the specifying the index names will be deprecated!")
            return dict_of_attributes
        else:
            custom_set, index_names = cls.create_custom_set(index_names, optimization_setup)
            dict_of_attributes = optimization_setup.check_for_subindex(dict_of_attributes, custom_set)
            return (dict_of_attributes, index_names)

    ### --- classmethods to construct sets, parameters, variables, and constraints, that correspond to ConversionTechnology --- ###
    @classmethod
    def construct_sets(cls, optimization_setup):
        """ constructs the pe.Sets of the class <ConversionTechnology>
        :param optimization_setup: The OptimizationSetup the element is part of """
        model = optimization_setup.model
        # get input carriers
        _input_carriers = optimization_setup.get_attribute_of_all_elements(cls, "input_carrier")
        _output_carriers = optimization_setup.get_attribute_of_all_elements(cls, "output_carrier")
        _reference_carrier = optimization_setup.get_attribute_of_all_elements(cls, "reference_carrier")
        _dependent_carriers = {}
        for tech in _input_carriers:
            _dependent_carriers[tech] = _input_carriers[tech] + _output_carriers[tech]
            _dependent_carriers[tech].remove(_reference_carrier[tech][0])
        # input carriers of technology
        model.set_input_carriers = pe.Set(model.set_conversion_technologies, initialize=_input_carriers,
            doc="set of carriers that are an input to a specific conversion technology. Dimensions: set_conversion_technologies")
        # output carriers of technology
        model.set_output_carriers = pe.Set(model.set_conversion_technologies, initialize=_output_carriers,
            doc="set of carriers that are an output to a specific conversion technology. Dimensions: set_conversion_technologies")
        # dependent carriers of technology
        model.set_dependent_carriers = pe.Set(model.set_conversion_technologies, initialize=_dependent_carriers,
            doc="set of carriers that are an output to a specific conversion technology.\n\t Dimensions: set_conversion_technologies")

        # add pe.Sets of the child classes
        for subclass in cls.__subclasses__():
            if np.size(optimization_setup.system[subclass.label]):
                subclass.construct_sets(optimization_setup)

    @classmethod
    def construct_params(cls, optimization_setup):
        """ constructs the pe.Params of the class <ConversionTechnology>
        :param optimization_setup: The OptimizationSetup the element is part of """
        # slope of linearly modeled capex
        optimization_setup.parameters.add_parameter(name="capex_specific_conversion",
            data=cls.get_capex_conversion_factor_all_elements(optimization_setup, "capex", False, index_names=["set_conversion_technologies", "set_capex_linear", "set_nodes", "set_time_steps_yearly"]),
            doc="Parameter which specifies the slope of the capex if approximated linearly")
        # slope of linearly modeled conversion efficiencies
        optimization_setup.parameters.add_parameter(name="conversion_factor", data=cls.get_capex_conversion_factor_all_elements(optimization_setup, "conversion_factor", False,
                                                                                                                     index_names=["set_conversion_technologies", "set_conversion_factor_linear",
                                                                                                                                  "set_nodes", "set_time_steps_yearly"]),
            doc="Parameter which specifies the slope of the conversion efficiency if approximated linearly")

    @classmethod
    def construct_vars(cls, optimization_setup):
        """ constructs the pe.Vars of the class <ConversionTechnology>
        :param optimization_setup: The OptimizationSetup the element is part of """

        def flow_conversion_bounds(model, tech, carrier, node, time):
            """ return bounds of flow_conversion for bigM expression
            :param model: pe.ConcreteModel
            :param tech: tech index
            :param carrier: carrier index
            :param node: node index
            :param time: time index
            :return bounds: bounds of flow_conversion"""
            params = optimization_setup.parameters
            energy_system = optimization_setup.energy_system
            if optimization_setup.get_attribute_of_specific_element(cls, tech, "conversion_factor_is_pwa"):
                bounds = optimization_setup.get_attribute_of_specific_element(cls, tech, "pwa_conversion_factor")["bounds"][carrier]
            else:
                # convert operationTimeStep to time_step_year: operationTimeStep -> base_time_step -> time_step_year
                time_step_year = energy_system.time_steps.convert_time_step_operation2year(tech, time)
                if carrier == model.set_reference_carriers[tech].at(1):
                    _conversion_factor = 1
                else:
                    _conversion_factor = params.conversion_factor[tech, carrier, node, time_step_year]
                bounds = []
                for _bound in model.capacity[tech, "power", node, time_step_year].bounds:
                    if _bound is not None:
                        bounds.append(_bound * _conversion_factor)
                    else:
                        bounds.append(None)
                bounds = tuple(bounds)
            return (bounds)

        model = optimization_setup.model

        ## Flow variables
        # input flow of carrier into technology
        optimization_setup.variables.add_variable(model, name="flow_conversion_input", index_sets=cls.create_custom_set(["set_conversion_technologies", "set_input_carriers", "set_nodes", "set_time_steps_operation"], optimization_setup),
            domain=pe.NonNegativeReals, bounds=flow_conversion_bounds, doc='Carrier input of conversion technologies')
        # output flow of carrier into technology
        optimization_setup.variables.add_variable(model, name="flow_conversion_output", index_sets=cls.create_custom_set(["set_conversion_technologies", "set_output_carriers", "set_nodes", "set_time_steps_operation"], optimization_setup),
            domain=pe.NonNegativeReals, bounds=flow_conversion_bounds, doc='Carrier output of conversion technologies')

        ## pwa Variables - Capex
        # pwa capacity
        optimization_setup.variables.add_variable(model, name="capacity_approximation", index_sets=cls.create_custom_set(["set_conversion_technologies", "set_nodes", "set_time_steps_yearly"], optimization_setup), domain=pe.NonNegativeReals,
            doc='pwa variable for size of installed technology on edge i and time t')
        # pwa capex technology
        optimization_setup.variables.add_variable(model, name="capex_approximation", index_sets=cls.create_custom_set(["set_conversion_technologies", "set_nodes", "set_time_steps_yearly"], optimization_setup), domain=pe.NonNegativeReals,
            doc='pwa variable for capex for installing technology on edge i and time t')

        ## pwa Variables - Conversion Efficiency
        # pwa reference flow of carrier into technology
        optimization_setup.variables.add_variable(model, name="flow_approximation_reference",
            index_sets=cls.create_custom_set(["set_conversion_technologies", "set_dependent_carriers", "set_nodes", "set_time_steps_operation"], optimization_setup), domain=pe.NonNegativeReals,
            bounds=flow_conversion_bounds, doc='pwa of flow of reference carrier of conversion technologies')
        # pwa dependent flow of carrier into technology
        optimization_setup.variables.add_variable(model, name="flow_approximation_dependent",
            index_sets=cls.create_custom_set(["set_conversion_technologies", "set_dependent_carriers", "set_nodes", "set_time_steps_operation"], optimization_setup), domain=pe.NonNegativeReals,
            bounds=flow_conversion_bounds, doc='pwa of flow of dependent carriers of conversion technologies')

    @classmethod
    def construct_constraints(cls, optimization_setup):
        """ constructs the pe.Constraints of the class <ConversionTechnology>
        :param optimization_setup: The OptimizationSetup the element is part of """
        model = optimization_setup.model
        # add pwa constraints
        rules = ConversionTechnologyRules(optimization_setup)
        # capex
        set_pwa_capex = cls.create_custom_set(["set_conversion_technologies", "set_capex_pwa", "set_nodes", "set_time_steps_yearly"], optimization_setup)
        set_linear_capex = cls.create_custom_set(["set_conversion_technologies", "set_capex_linear", "set_nodes", "set_time_steps_yearly"], optimization_setup)
        if set_pwa_capex:
            # if set_pwa_capex contains technologies:
            pwa_breakpoints, pwa_values = cls.calculate_pwa_breakpoints_values(optimization_setup, set_pwa_capex[0], "capex")
            model.constraint_pwa_capex = pe.Piecewise(set_pwa_capex[0], model.capex_approximation, model.capacity_approximation, pw_pts=pwa_breakpoints, pw_constr_type="EQ", f_rule=pwa_values,
                                                      unbounded_domain_var=True, warn_domain_coverage=False, pw_repn="BIGM_BIN")
        if set_linear_capex[0]:
            # if set_linear_capex contains technologies:
            optimization_setup.constraints.add_constraint(model, name="constraint_linear_capex", index_sets=set_linear_capex, rule=rules.constraint_linear_capex_rule, doc="Linear relationship in capex")
        # Conversion Efficiency
        set_pwa_conversion_factor = cls.create_custom_set(["set_conversion_technologies", "set_conversion_factor_pwa", "set_nodes", "set_time_steps_operation"], optimization_setup)
        set_linear_conversion_factor = cls.create_custom_set(["set_conversion_technologies", "set_conversion_factor_linear", "set_nodes", "set_time_steps_operation"], optimization_setup)
        if set_pwa_conversion_factor:
            # if set_pwa_conversion_factor contains technologies:
            pwa_breakpoints, pwa_values = cls.calculate_pwa_breakpoints_values(optimization_setup, set_pwa_conversion_factor[0], "conversion_factor")
            model.constraint_pwa_conversion_factor = pe.Piecewise(set_pwa_conversion_factor[0], model.flow_approximation_dependent, model.flow_approximation_reference, pw_pts=pwa_breakpoints,
                                                                  pw_constr_type="EQ", f_rule=pwa_values, unbounded_domain_var=True, warn_domain_coverage=False, pw_repn="BIGM_BIN")
        if set_linear_conversion_factor[0]:
            # if set_linear_conversion_factor contains technologies:
            optimization_setup.constraints.add_constraint(model, name="constraint_linear_conversion_factor", index_sets=set_linear_conversion_factor, rule=rules.constraint_linear_conversion_factor_rule,
                doc="Linear relationship in conversion_factor")  # Coupling constraints
        # couple the real variables with the auxiliary variables
        optimization_setup.constraints.add_constraint(model, name="constraint_capex_coupling", index_sets=cls.create_custom_set(["set_conversion_technologies", "set_nodes", "set_time_steps_yearly"], optimization_setup),
            rule=rules.constraint_capex_coupling_rule, doc="couples the real capex variables with the approximated variables")
        # capacity
        optimization_setup.constraints.add_constraint(model, name="constraint_capacity_coupling", index_sets=cls.create_custom_set(["set_conversion_technologies", "set_nodes", "set_time_steps_yearly"], optimization_setup),
            rule=rules.constraint_capacity_coupling_rule, doc="couples the real capacity variables with the approximated variables")

        # flow coupling constraints for technologies, which are not modeled with an on-off-behavior
        # reference flow coupling
        optimization_setup.constraints.add_constraint(model, name="constraint_reference_flow_coupling",
            index_sets=cls.create_custom_set(["set_conversion_technologies", "set_no_on_off", "set_dependent_carriers", "set_location", "set_time_steps_operation"], optimization_setup),
            rule=rules.constraint_reference_flow_coupling_rule, doc="couples the real reference flow variables with the approximated variables")
        # dependent flow coupling
        optimization_setup.constraints.add_constraint(model, name="constraint_dependent_flow_coupling",
            index_sets=cls.create_custom_set(["set_conversion_technologies", "set_no_on_off", "set_dependent_carriers", "set_location", "set_time_steps_operation"], optimization_setup),
            rule=rules.constraint_dependent_flow_coupling_rule, doc="couples the real dependent flow variables with the approximated variables")

    # defines disjuncts if technology on/off
    @classmethod
    def disjunct_on_technology_rule(cls, optimization_setup, disjunct, tech, capacity_type, node, time):
        """definition of disjunct constraints if technology is On"""
        model = disjunct.model()
        # get parameter object
        params = optimization_setup.parameters
        energy_system = optimization_setup.energy_system
        reference_carrier = model.set_reference_carriers[tech].at(1)
        if reference_carrier in model.set_input_carriers[tech]:
            reference_flow = model.flow_conversion_input[tech, reference_carrier, node, time]
        else:
            reference_flow = model.flow_conversion_output[tech, reference_carrier, node, time]
        # get invest time step
        time_step_year = energy_system.time_steps.convert_time_step_operation2year(tech, time)
        # disjunct constraints min load
        disjunct.constraint_min_load = pe.Constraint(expr=reference_flow >= params.min_load[tech, capacity_type, node, time] * model.capacity[tech, capacity_type, node, time_step_year])
        # couple reference flows
        rules = ConversionTechnologyRules(optimization_setup)
        optimization_setup.constraints.add_constraint(disjunct, name=f"constraint_reference_flow_coupling_{'_'.join([str(tech), str(node), str(time)])}",
            index_sets=[[[tech], model.set_dependent_carriers[tech], [node], [time]], ["set_conversion_technologies", "setDependentCarriers", "set_nodes", "set_time_steps_operation"]],
            rule=rules.constraint_reference_flow_coupling_rule, doc="couples the real reference flow variables with the approximated variables", )
        # couple dependent flows
        optimization_setup.constraints.add_constraint(disjunct, name=f"constraint_dependent_flow_coupling_{'_'.join([str(tech), str(node), str(time)])}",
            index_sets=[[[tech], model.set_dependent_carriers[tech], [node], [time]], ["set_conversion_technologies", "setDependentCarriers", "set_nodes", "set_time_steps_operation"]],
            rule=rules.constraint_dependent_flow_coupling_rule, doc="couples the real dependent flow variables with the approximated variables", )

    @classmethod
    def disjunct_off_technology_rule(cls, disjunct, tech, capacity_type, node, time):
        """definition of disjunct constraints if technology is off"""
        model = disjunct.model()
        disjunct.constraint_no_load = pe.Constraint(expr=sum(model.flow_conversion_input[tech, input_carrier, node, time] for input_carrier in model.set_input_carriers[tech]) + sum(
            model.flow_conversion_output[tech, output_carrier, node, time] for output_carrier in model.set_output_carriers[tech]) == 0)

    @classmethod
    def calculate_pwa_breakpoints_values(cls, optimization_setup, setPWA, type_pwa):
        """ calculates the breakpoints and function values for piecewise affine constraint
        :param optimization_setup: The OptimizationSetup the element is part of
        :param setPWA: set of variable indices in capex approximation, for which pwa is performed
        :param type_pwa: variable, for which pwa is performed
        :return pwa_breakpoints: dict of pwa breakpoint values
        :return pwa_values: dict of pwa function values """
        pwa_breakpoints = {}
        pwa_values = {}

        # iterate through pwa variable's indices
        for index in setPWA:
            pwa_breakpoints[index] = []
            pwa_values[index] = []
            if len(index) > 1:
                tech = index[0]
            else:
                tech = index
            if type_pwa == "capex":
                # retrieve pwa variables
                pwa_parameter = optimization_setup.get_attribute_of_specific_element(cls, tech, f"pwa_{type_pwa}")
                pwa_breakpoints[index] = pwa_parameter["capacity"]
                pwa_values[index] = pwa_parameter["capex"]
            elif type_pwa == "conversion_factor":
                # retrieve pwa variables
                pwa_parameter = optimization_setup.get_attribute_of_specific_element(cls, tech, f"pwa_{type_pwa}")
                pwa_breakpoints[index] = pwa_parameter[optimization_setup.get_attribute_of_all_elements(cls, "reference_carrier")[tech][0]]
                pwa_values[index] = pwa_parameter[index[1]]

        return pwa_breakpoints, pwa_values


class ConversionTechnologyRules:
    """
    Rules for the ConversionTechnology class
    """

    def __init__(self, optimization_setup):
        """
        Inits the rules for a given EnergySystem
        :param optimization_setup: The OptimizationSetup the element is part of
        """

        self.optimization_setup = optimization_setup
        self.energy_system = optimization_setup.energy_system

    ### --- functions with constraint rules --- ###
    def constraint_linear_capex_rule(self, model, tech, node, time):
        """ if capacity and capex have a linear relationship"""
        # get parameter object
        params = self.optimization_setup.parameters
        return (model.capex_approximation[tech, node, time] == params.capex_specific_conversion[tech, node, time] * model.capacity_approximation[tech, node, time])

    def constraint_linear_conversion_factor_rule(self, model, tech, dependent_carrier, node, time):
        """ if reference carrier and dependent carrier have a linear relationship"""
        # get parameter object
        params = self.optimization_setup.parameters
        # get invest time step
        time_step_year = self.energy_system.time_steps.convert_time_step_operation2year(tech, time)
        return (model.flow_approximation_dependent[tech, dependent_carrier, node, time] == params.conversion_factor[tech, dependent_carrier, node, time_step_year] *
                model.flow_approximation_reference[tech, dependent_carrier, node, time])

    def constraint_capex_coupling_rule(self, model, tech, node, time):
        """ couples capex variables based on modeling technique"""
        return (model.cost_capex[tech, "power", node, time] == model.capex_approximation[tech, node, time])

    def constraint_capacity_coupling_rule(self, model, tech, node, time):
        """ couples capacity variables based on modeling technique"""
        return (model.capacity_addition[tech, "power", node, time] == model.capacity_approximation[tech, node, time])

    def constraint_reference_flow_coupling_rule(self, disjunct, tech, dependent_carrier, node, time):
        """ couples reference flow variables based on modeling technique"""
        model = disjunct.model()
        reference_carrier = model.set_reference_carriers[tech].at(1)
        if reference_carrier in model.set_input_carriers[tech]:
            return (model.flow_conversion_input[tech, reference_carrier, node, time] == model.flow_approximation_reference[tech, dependent_carrier, node, time])
        else:
            return (model.flow_conversion_output[tech, reference_carrier, node, time] == model.flow_approximation_reference[tech, dependent_carrier, node, time])

    def constraint_dependent_flow_coupling_rule(self, disjunct, tech, dependent_carrier, node, time):
        """ couples dependent flow variables based on modeling technique"""
        model = disjunct.model()
        if dependent_carrier in model.set_input_carriers[tech]:
            return (model.flow_conversion_input[tech, dependent_carrier, node, time] == model.flow_approximation_dependent[tech, dependent_carrier, node, time])
        else:
            return (model.flow_conversion_output[tech, dependent_carrier, node, time] == model.flow_approximation_dependent[tech, dependent_carrier, node, time])
