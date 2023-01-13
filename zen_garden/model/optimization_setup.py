"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      October-2021
Authors:      Jacob Mannhardt (jmannhardt@ethz.ch)
              Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:  Class defining the Concrete optimization model.
              The class takes as inputs the properties of the optimization problem. The properties are saved in the
              dictionaries analysis and system which are passed to the class. After initializing the Concrete model, the
              class adds carriers and technologies to the Concrete model and returns the Concrete optimization model.
              The class also includes a method to solve the optimization problem.
==========================================================================================================================================================================="""
import logging
import pyomo.environ as pe
from pyomo.core.expr.current import decompose_term
import os
import sys
import pandas as pd
import numpy as np
# import elements of the optimization problem
# technology and carrier classes from technology and carrier directory, respectively
from .objects.element import Element
from .objects.technology import *
from .objects.carrier import *
from .objects.energy_system import EnergySystem
from ..preprocess.functions.time_series_aggregation import TimeSeriesAggregation


class OptimizationSetup():

    def __init__(self, analysis, prepare):
        """setup Pyomo Concrete Model
        :param analysis: dictionary defining the analysis framework
        :param system: dictionary defining the system
        :param paths: dictionary defining the paths of the model
        :param solver: dictionary defining the solver"""
        self.prepare = prepare
        self.analysis = analysis
        self.system = prepare.system
        self.paths = prepare.paths
        self.solver = prepare.solver
        # step of optimization horizon
        self.step_horizon = 0
        # set optimization attributes (the five set above) to class <EnergySystem>
        EnergySystem.set_optimization_attributes(analysis, self.system, self.paths, self.solver)
        # set base scenario
        self.set_base_configuration()
        # add Elements to optimization
        self.add_elements()

    def add_elements(self):
        """This method sets up the parameters, variables and constraints of the carriers of the optimization problem.
        :param analysis: dictionary defining the analysis framework
        :param system: dictionary defining the system"""
        logging.info("\n--- Add elements to model--- \n")
        # add energy system to define system
        EnergySystem("energy_system")

        for element_name in EnergySystem.get_element_list():
            element_class = EnergySystem.dict_element_classes[element_name]
            element_name = element_class.label
            element_set = self.system[element_name]

            # before adding the carriers, get set_carriers and check if carrier data exists
            if element_name == "set_carriers":
                element_set = EnergySystem.get_attribute("set_carriers")
                self.system["set_carriers"] = element_set
                self.prepare.check_existing_carrier_data(self.system)

            # check if element_set has a subset and remove subset from element_set
            if element_name in self.analysis["subsets"].keys():
                element_subset = []
                for subset in self.analysis["subsets"][element_name]:
                    element_subset += [item for item in self.system[subset]]
                element_set = list(set(element_set) - set(element_subset))

            # add element class
            for item in element_set:
                element_class(item)
        if EnergySystem.get_solver()["analyze_numerics"]:
            EnergySystem.get_unit_handling().recommend_base_units(immutable_unit=EnergySystem.get_solver()["immutable_unit"], unit_exps=EnergySystem.get_solver()["rangeUnitExponents"])
        # conduct  time series aggregation
        TimeSeriesAggregation.conduct_tsa()

    def construct_optimization_problem(self):
        """ constructs the optimization problem """
        # create empty ConcreteModel
        self.model = pe.ConcreteModel()
        EnergySystem.set_pyomo_model(self.model)
        # define and construct components of self.model
        Element.construct_model_components()
        logging.info("Apply Big-M GDP ")
        # add transformation factory so that disjuncts are solved
        pe.TransformationFactory("gdp.bigm").apply_to(self.model)
        # find smallest and largest coefficient and RHS
        self.analyze_numerics()

    def get_optimization_horizon(self):
        """ returns list of optimization horizon steps """
        energy_system = EnergySystem.get_energy_system()
        # if using rolling horizon
        if self.system["use_rolling_horizon"]:
            self.years_in_horizon = self.system["years_in_rolling_horizon"]
            _time_steps_yearly = energy_system.set_time_steps_yearly
            self.steps_horizon = {year: list(range(year, min(year + self.years_in_horizon, max(_time_steps_yearly) + 1))) for year in _time_steps_yearly}
        # if no rolling horizon
        else:
            self.years_in_horizon = len(energy_system.set_time_steps_yearly)
            self.steps_horizon = {0: energy_system.set_time_steps_yearly}
        return list(self.steps_horizon.keys())

    def set_base_configuration(self, scenario="", elements={}):
        """set base configuration
        :param scenario: name of base scenario
        :param elements: elements in base configuration """
        if not hasattr(self, "base_scenario"):
            self.initial_scenario = scenario
            self.initial_configuration = elements
        else:
            self.overwrite_params(self.initial_scenario, self.initial_configuration)
        self.base_scenario = scenario
        self.base_configuration = elements

    def restore_base_configuration(self, scenario, elements):
        """restore default configuration
        :param scenario: scenario name
        :param elements: dictionary of scenario dependent elements and parameters"""
        if not scenario == self.base_scenario:
            # restore base configuration
            self.overwrite_params(self.base_scenario, self.base_configuration)
            # continuously update base_configuration so all parameters are reset to their base value after being changed
            for element_name, params in elements.items():
                if element_name not in self.base_configuration.keys():
                    self.base_configuration[element_name] = params
                else:
                    for param in params:
                        if param not in self.base_configuration[element_name]:
                            self.base_configuration[element_name].append(param)
                # continuously update initialConfiguration as well
                if element_name not in self.initial_configuration.keys():
                    self.initial_configuration[element_name] = params
                else:
                    for param in params:
                        if param not in self.initial_configuration[element_name]:
                            self.initial_configuration[element_name].append(param)

    def overwrite_params(self, scenario, elements):
        """overwrite scenario dependent parameters
        :param scenario: scenario name
        :param elements: dictionary of scenario dependent elements and parameters"""
        if scenario != "":
            scenario = "_" + scenario
        # list of parameters with raw_time_series
        conduct_tsa = False
        # overwrite scenario dependent parameter values for all elements
        for element_name, params in elements.items():
            if element_name == "EnergySystem":
                element = EnergySystem.get_energy_system()
            else:
                element = Element.get_element(element_name)
            # overwrite scenario dependent parameters
            for param in params:
                assert "pwa" not in param, "Scenarios are not implemented for piece-wise affine parameters."
                file_name = param
                column = None
                if type(param) is tuple:
                    file_name, column = param
                    param = param[1]
                if "yearly_variation" in param:
                    param    = param.replace("_yearly_variation","")
                    file_name = param
                # get old param value
                _old_param = getattr(element, param)
                _index_names = _old_param.index.names
                _index_sets = [index_set for index_set, index_name in element.datainput.index_names.items() if index_name in _index_names]
                _time_steps = None
                # if existing capacity is changed, setExistingTechnologies, existing lifetime, and capexExistingCapacity have to be updated as well
                if "set_existing_technologies" in _index_sets:
                    # update setExistingTechnologies and existingLifetime
                    _existing_technologies = element.datainput.extract_set_existing_technologies(scenario=scenario)
                    setattr(element, "set_existing_technologies", _existing_technologies)
                    _lifetime_existing_technologies = element.datainput.extract_lifetime_existing_technology(param, index_sets=_index_sets, scenario=scenario)
                    setattr(element, "lifetime_existing_technology", _lifetime_existing_technologies)
                # set new parameter value
                if hasattr(element, "raw_time_series") and param in element.raw_time_series.keys():
                    conduct_tsa = True
                    _time_steps = EnergySystem.get_energy_system().set_base_time_steps_yearly
                    element.raw_time_series[param] = element.datainput.extract_input_data(file_name, index_sets=_index_sets, column=column, time_steps=_time_steps, scenario=scenario)
                else:
                    assert isinstance(_old_param, pd.Series) or isinstance(_old_param, pd.DataFrame), f"Param values of '{param}' have to be a pd.DataFrame or pd.Series."
                    if "time" in _index_names:
                        _time_steps = list(_old_param.index.unique("time"))
                    _new_param = element.datainput.extract_input_data(file_name, index_sets=_index_sets, time_steps=_time_steps, scenario=scenario)
                    setattr(element, param, _new_param)
                    # if existing capacity is changed, capex_existing_capacity also has to be updated
                    if "existing_capacity" in param:
                        storage_energy = False
                        if element in EnergySystem.system["set_storage_technologies"]:
                            storage_energy = True
                        _capex_existing_capacities = element.calculate_capex_of_existing_capacities(storage_energy=storage_energy)
                        setattr(element, "capex_existing_capacity", _capex_existing_capacities)
        # if scenario contains timeSeries dependent params conduct tsa
        if conduct_tsa:
            TimeSeriesAggregation.conduct_tsa()

    def overwrite_time_indices(self, step_horizon):
        """ select subset of time indices, matching the step horizon
        :param step_horizon: step of the rolling horizon """
        energy_system = EnergySystem.get_energy_system()

        if self.system["use_rolling_horizon"]:
            _time_steps_yearly_horizon = self.steps_horizon[step_horizon]
            _base_time_steps_horizon = EnergySystem.decode_yearly_time_steps(_time_steps_yearly_horizon)
            # overwrite time steps of each element
            for element in Element.get_all_elements():
                element.overwrite_time_steps(_base_time_steps_horizon)
            # overwrite base time steps and yearly base time steps
            _new_base_time_steps_horizon = _base_time_steps_horizon.squeeze().tolist()
            if not isinstance(_new_base_time_steps_horizon, list):
                _new_base_time_steps_horizon = [_new_base_time_steps_horizon]
            energy_system.set_base_time_steps = _new_base_time_steps_horizon
            energy_system.set_time_steps_yearly = _time_steps_yearly_horizon

    def analyze_numerics(self):
        """ get largest and smallest matrix coefficients and RHS """
        if EnergySystem.get_solver()["analyze_numerics"]:
            largest_rhs = [None, 0]
            smallest_rhs = [None, np.inf]
            largest_coeff = [None, 0]
            smallest_coeff = [None, np.inf]

            for cons in self.model.component_objects(pe.Constraint):
                for idx in cons:
                    deco_lhs = decompose_term(cons[idx].expr.args[0])
                    deco_rhs = decompose_term(cons[idx].expr.args[1])
                    deco_comb = []
                    if deco_lhs[0]:
                        deco_comb += deco_lhs[1]
                    if deco_rhs[0]:
                        deco_comb += deco_rhs[1]
                    _RHS = 0
                    for item in deco_comb:
                        _abs = abs(item[0])
                        if _abs != 0:
                            if item[1] is not None:
                                if _abs > largest_coeff[1]:
                                    largest_coeff[0] = (cons[idx].name, item[1].name)
                                    largest_coeff[1] = _abs
                                if _abs < smallest_coeff[1]:
                                    smallest_coeff[0] = (cons[idx].name, item[1].name)
                                    smallest_coeff[1] = _abs
                            else:
                                _RHS += item[0]
                    _RHS = abs(_RHS)
                    if _RHS != 0:
                        if _RHS > largest_rhs[1]:
                            largest_rhs[0] = cons[idx].name
                            largest_rhs[1] = _RHS
                        if _RHS < smallest_rhs[1]:
                            smallest_rhs[0] = cons[idx].name
                            smallest_rhs[1] = _RHS
            logging.info(
                f"Numeric Range Statistics:\nLargest Matrix Coefficient: {largest_coeff[1]} in {largest_coeff[0]}\nSmallest Matrix Coefficient: {smallest_coeff[1]} in {smallest_coeff[0]}\nLargest RHS: {largest_rhs[1]} in {largest_rhs[0]}\nSmallest RHS: {smallest_rhs[1]} in {smallest_rhs[0]}")

    def solve(self, solver):
        """Create model instance by assigning parameter values and instantiating the sets
        :param solver: dictionary containing the solver settings """

        solver_name = solver["name"]
        # remove options that are None
        solver_options = {key: solver["solver_options"][key] for key in solver["solver_options"] if solver["solver_options"][key] is not None}

        logging.info(f"\n--- Solve model instance using {solver_name} ---\n")
        # disable logger temporarily
        logging.disable(logging.WARNING)
        # write an ILP file to print the IIS if infeasible
        # (gives Warning: unable to write requested result file ".//outputs//logs//model.ilp" if feasible)
        solver_parameters = f"ResultFile={os.path.dirname(solver['solver_options']['logfile'])}//infeasibleModelIIS.ilp"

        if solver_name == "gurobi_persistent":
            self.opt = pe.SolverFactory(solver_name, options=solver_options)
            self.opt.set_instance(self.model, symbolic_solver_labels=solver["useSymbolicLabels"])
            self.results = self.opt.solve(tee=solver["verbosity"], logfile=solver["solver_options"]["logfile"], options_string=solver_parameters)
        else:
            self.opt = pe.SolverFactory(solver_name)
            self.results = self.opt.solve(self.model, tee=solver["verbosity"], keepfiles=True, logfile=solver["solver_options"]["logfile"])
        # enable logger
        logging.disable(logging.NOTSET)

        # store the solution into the results
        self.model.solutions.store_to(self.results, skip_stale_vars=True)

    def add_newly_built_capacity(self, step_horizon):
        """ adds the newly built capacity to the existing capacity
        :param step_horizon: step of the rolling horizon """
        if self.system["use_rolling_horizon"]:
            _built_capacity = pd.Series(self.model.built_capacity.extract_values())
            _invest_capacity = pd.Series(self.model.invested_capacity.extract_values())
            _capex = pd.Series(self.model.capex.extract_values())
            _rounding_value = 10 ** (-EnergySystem.get_solver()["rounding_decimal_points"])
            _built_capacity[_built_capacity <= _rounding_value] = 0
            _invest_capacity[_invest_capacity <= _rounding_value] = 0
            _capex[_capex <= _rounding_value] = 0
            _base_time_steps = EnergySystem.decode_yearly_time_steps([step_horizon])
            Technology = getattr(sys.modules[__name__], "Technology")
            for tech in Technology.get_all_elements():
                # new capacity
                _built_capacity_tech = _built_capacity.loc[tech.name].unstack()
                _invested_capacity_tech = _invest_capacity.loc[tech.name].unstack()
                _capex_tech = _capex.loc[tech.name].unstack()
                tech.add_newly_built_capacity_tech(_built_capacity_tech, _capex_tech, _base_time_steps)
                tech.add_newly_invested_capacity_tech(_invested_capacity_tech, step_horizon)

    def add_carbon_emission_cumulative(self, step_horizon):
        """ overwrite previous carbon emissions with cumulative carbon emissions
        :param step_horizon: step of the rolling horizon """
        if self.system["use_rolling_horizon"]:
            energy_system = EnergySystem.get_energy_system()
            interval_between_years = EnergySystem.get_system()["interval_between_years"]
            _carbon_emissions_cumulative = self.model.carbon_emissions_cumulative.extract_values()[step_horizon]
            carbon_emissions = self.model.carbon_emissions_total.extract_values()[step_horizon]
            energy_system.previous_carbon_emissions = _carbon_emissions_cumulative + carbon_emissions * (interval_between_years - 1)
