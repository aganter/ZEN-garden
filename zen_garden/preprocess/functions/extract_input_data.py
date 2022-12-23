"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      January-2022
Authors:      Jacob Mannhardt (jmannhardt@ethz.ch)
              Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Functions to extract the input data from the provided input files
==========================================================================================================================================================================="""
import copy
import os
import logging
import warnings
import math
import numpy as np
import pandas as pd
from scipy.stats import linregress


class DataInput():

    def __init__(self, element, system, analysis, solver, energy_system, unit_handling):
        """ data input object to extract input data
        :param element: element for which data is extracted
        :param system: dictionary defining the system
        :param analysis: dictionary defining the analysis framework
        :param solver: dictionary defining the solver 
        :param energy_system: instance of class <EnergySystem> to define energy_system
        :param unit_handling: instance of class <UnitHandling> to convert units """
        self.element = element
        self.system = system
        self.analysis = analysis
        self.solver = solver
        self.energy_system = energy_system
        self.unit_handling = unit_handling
        # extract folder path
        self.folder_path = getattr(self.element, "input_path")

        # get names of indices
        # self.index_names     = {index_name: self.analysis['header_data_inputs'][index_name][0] for index_name in self.analysis['header_data_inputs']}
        self.index_names = self.analysis['header_data_inputs']

    def extract_input_data(self, file_name, index_sets, column=None, time_steps=None, scenario=""):
        """ reads input data and restructures the dataframe to return (multi)indexed dict
        :param file_name: name of selected file.
        :param index_sets: index sets of attribute. Creates (multi)index. Corresponds to order in pe.Set/pe.Param
        :param column: select specific column
        :param time_steps: specific time_steps of element
        :param scenario: scenario name
        :return dataDict: dictionary with attribute values """

        # generic time steps
        yearly_variation = False
        if not time_steps:
            time_steps = self.energy_system.set_base_time_steps
        # if time steps are the yearly base time steps
        elif time_steps is self.energy_system.set_base_time_steps_yearly:
            yearly_variation = True
            self.extract_yearly_variation(file_name, index_sets, column)

        # if existing capacities and existing capacities not used
        if (file_name == "existing_capacity" or file_name == "existing_capacity_energy") and not self.analysis["use_existing_capacities"]:
            df_output, *_ = self.create_default_output(index_sets, column, file_name=file_name, time_steps=time_steps, manual_default_value=0, scenario=scenario)
            return df_output
        else:
            df_output, default_value, index_name_list = self.create_default_output(index_sets, column, file_name=file_name, time_steps=time_steps, scenario=scenario)
        # set default_name
        if column:
            default_name = column
        else:
            default_name = file_name
        # read input file
        df_input = self.read_input_data(file_name + scenario)
        if scenario and yearly_variation and df_input is None:
            logging.info(f"{file_name}{scenario} is missing from {self.folder_path}. {file_name} is used as input file")
            df_input = self.read_input_data(file_name)

        assert (df_input is not None or default_value is not None), f"input file for attribute {default_name} could not be imported and no default value is given."
        if df_input is not None and not df_input.empty:
            df_output = self.extract_general_input_data(df_input, df_output, file_name, index_name_list, column, default_value)
        # save parameter values for analysis of numerics
        self.save_values_of_attribute(df_output=df_output, file_name=default_name)
        return df_output

    def extract_general_input_data(self, df_input, df_output, file_name, index_name_list, column, default_value):
        """ fills df_output with data from df_input
        :param df_input: raw input dataframe
        :param df_output: empty output dataframe, only filled with default_value
        :param file_name: name of selected file
        :param index_name_list: list of name of indices
        :param column: select specific column
        :param default_value: default for dataframe
        :return df_output: filled output dataframe """

        # select and drop scenario
        assert df_input.columns is not None, f"Input file '{file_name}' has no columns"
        # set index by index_name_list
        missing_index = list(set(index_name_list) - set(index_name_list).intersection(set(df_input.columns)))
        assert len(missing_index) <= 1, f"More than one the requested index sets ({missing_index}) are missing from input file for {file_name}"

        # no indices missing
        if len(missing_index) == 0:
            df_input = DataInput.extract_from_input_without_missing_index(df_input, index_name_list, column, file_name)
        else:
            missing_index = missing_index[0]
            # check if special case of existing Technology
            if "existingTechnology" in missing_index:
                if column:
                    default_name = column
                else:
                    default_name = file_name
                df_output = DataInput.extract_from_input_for_existing_capacities(df_input, df_output, index_name_list, default_name, missing_index)
                if isinstance(default_value, dict):
                    df_output = df_output * default_value["multiplier"]
                return df_output
            # index missing
            else:
                df_input = DataInput.extract_from_input_with_missing_index(df_input, df_output, copy.deepcopy(index_name_list), column, file_name, missing_index)

        # apply multiplier to input data
        df_input = df_input * default_value["multiplier"]
        # delete nans
        df_input = df_input.dropna()

        # get common index of df_output and df_input
        if not isinstance(df_input.index, pd.MultiIndex) and isinstance(df_output.index, pd.MultiIndex):
            index_list = df_input.index.to_list()
            if len(index_list) == 1:
                index_multi_index = pd.MultiIndex.from_tuples([(index_list[0],)], names=[df_input.index.name])
            else:
                index_multi_index = pd.MultiIndex.from_product([index_list], names=[df_input.index.name])
            df_input = pd.Series(index=index_multi_index, data=df_input.to_list())
        common_index = df_output.index.intersection(df_input.index)
        assert default_value is not None or len(common_index) == len(df_output.index), f"Input for {file_name} does not provide entire dataset and no default given in attributes.csv"
        df_output.loc[common_index] = df_input.loc[common_index]
        return df_output

    def read_input_data(self, input_file_name):
        """ reads input data and returns raw input dataframe
        :param input_file_name: name of selected file
        :return df_input: pd.DataFrame with input data """

        # append .csv suffix
        input_file_name += ".csv"

        # select data
        file_names = os.listdir(self.folder_path)
        if input_file_name in file_names:
            df_input = pd.read_csv(os.path.join(self.folder_path, input_file_name), header=0, index_col=None)
            return df_input
        else:
            return None

    def extract_attribute(self, attribute_name, skip_warning=False, scenario=""):
        """ reads input data and restructures the dataframe to return (multi)indexed dict
        :param attribute_name: name of selected attribute
        :param skip_warning: boolean to indicate if "Default" warning is skipped
        :param scenario: scenario name
        :return attribute_value: attribute value """
        filename = "attributes"
        df_input = self.read_input_data(filename + scenario)
        if df_input is not None:
            df_input = df_input.set_index("index").squeeze(axis=1)
            attribute_name = self.adapt_attribute_name(attribute_name, df_input, skip_warning)
        elif scenario != "":
            df_input = self.read_input_data(filename)
            df_input = df_input.set_index("index").squeeze(axis=1)
            attribute_name = self.adapt_attribute_name(attribute_name, df_input, skip_warning)
        # if df_input is None or attribute_name is None:
        #     df_input = self.read_input_data(filename)
        #     if df_input is not None:
        #         df_input = df_input.set_index("index").squeeze(axis=1)
        #     else:
        #         return None
        #     attribute_name = self.adapt_attribute_name(attribute_name,df_input,skip_warning)
        if attribute_name is not None:
            # get attribute
            attribute_value = df_input.loc[attribute_name, "value"]
            multiplier = self.unit_handling.get_unit_multiplier(df_input.loc[attribute_name, "unit"])
            try:
                attribute = {"value": float(attribute_value) * multiplier, "multiplier": multiplier}
                return attribute
            except:
                return attribute_value
        else:
            return None

    def adapt_attribute_name(self, attribute_name, df_input, skip_warning=False):
        """ check if attribute in index"""
        if attribute_name + "_default" not in df_input.index:
            if attribute_name not in df_input.index:
                return None
            elif not skip_warning:
                warnings.warn(f"Attribute names without '_default' suffix will be deprecated. \nChange for {attribute_name} of attributes in path {self.folder_path}", FutureWarning)
        else:
            attribute_name = attribute_name + "_default"
        return attribute_name

    def extract_yearly_variation(self, file_name, index_sets, column, scenario=""):
        """ reads the yearly variation of a time dependent quantity
        :param self.folder_path: path to input files
        :param file_name: name of selected file.
        :param index_sets: index sets of attribute. Creates (multi)index. Corresponds to order in pe.Set/pe.Param
        :param column: select specific column
        """
        # remove intrayearly time steps from index set and add interyearly time steps
        _index_sets = copy.deepcopy(index_sets)
        _index_sets.remove("set_time_steps")
        _index_sets.append("set_time_steps_yearly")
        # add YearlyVariation to file_name
        file_name += "YearlyVariation"
        # read input data
        df_input = self.read_input_data(file_name + scenario)
        if scenario and df_input is None:
            logging.info(f"{file_name}{scenario} is missing from {self.folder_path}. {file_name} is used as input file")
            df_input = self.read_input_data(file_name)
        if df_input is not None:
            if column is not None and column not in df_input:
                return
            df_output, default_value, index_name_list = self.create_default_output(_index_sets, column, file_name=file_name, manual_default_value=1)
            # set yearlyVariation attribute to df_output
            if column:
                _selected_column = column
                _name_yearly_variation = column + "YearlyVariation"
            else:
                _selected_column = None
                _name_yearly_variation = file_name
            df_output = self.extract_general_input_data(df_input, df_output, file_name, index_name_list, _selected_column, default_value)
            setattr(self, _name_yearly_variation, df_output)

    def extract_locations(self, extract_nodes=True):
        """ reads input data to extract nodes or edges.
        :param extractNodes: boolean to switch between nodes and edges """
        if extract_nodes == "country":
            set_country_nodes = self.read_input_data("set_nodes")["country"]
            set_country_nodes = set_country_nodes.drop_duplicates().to_list()
            return set_country_nodes
        elif extract_nodes:
            set_nodes_config  = self.system["set_nodes"]
            set_nodes_input   = self.read_input_data("set_nodes")["node"].to_list()
            # if no nodes specified in system, use all nodes
            if len(set_nodes_config) == 0 and not len(set_nodes_input) == 0:
                self.system["set_nodes"] = set_nodes_input
                set_nodes_config = set_nodes_input
            else:
                assert len(set_nodes_config) > 1, f"ZENx is a spatially distributed model. Please specify at least 2 nodes."
                _missing_nodes = list(set(set_nodes_config).difference(set_nodes_input))
                assert len(_missing_nodes) == 0, f"The nodes {_missing_nodes} were declared in the config but do not exist in the input file {self.folder_path + 'set_nodes'}"
            if not isinstance(set_nodes_config, list):
                set_nodes_config = set_nodes_config.to_list()
            set_nodes_config.sort()
            return set_nodes_config
        else:
            set_edges_input = self.read_input_data("set_edges")
            if set_edges_input is not None:
                set_edges = set_edges_input[(set_edges_input["node_from"].isin(self.energy_system.set_nodes)) & (set_edges_input["node_to"].isin(self.energy_system.set_nodes))]
                set_edges = set_edges.set_index("edge")
                return set_edges
            else:
                return None

    def extract_conversion_carriers(self):
        """ reads input data and extracts conversion carriers
        :return carrier_dict: dictionary with input and output carriers of technology """
        carrier_dict = {}
        # get carriers
        for _carrier_type in ["input_carrier", "output_carrier"]:
            _carrier_string = self.extract_attribute(_carrier_type, skip_warning=True)
            if type(_carrier_string) == str:
                _carrier_list = _carrier_string.strip().split(" ")
            else:
                _carrier_list = []
            carrier_dict[_carrier_type] = _carrier_list
        return carrier_dict

    def extract_set_existing_technologies(self, storage_energy=False, scenario=""):
        """ reads input data and creates setExistingCapacity for each technology
        :param storageEnergy: boolean if existing energy capacity of storage technology (instead of power)
        :return setExistingTechnologies: return set existing technologies"""
        #TODO merge changes in extract input data and optimization setup
        set_existing_technologies = np.array([0])
        if self.analysis["use_existing_capacities"]:
            if storage_energy:
                _energy_string = "Energy"
            else:
                _energy_string = ""

            df_input = self.read_input_data(f"existing_capacity{_energy_string}")
            if df_input is None:
                return [0]
            if self.element.name in self.system["set_transport_technologies"]:
                location = "edge"
            else:
                location = "node"
            _max_node_count = df_input[location].value_counts().max()
            if _max_node_count is not np.nan:
                setExistingTechnologies = np.arange(0, _max_node_count)

        return set_existing_technologies

    def extract_lifetime_existing_technology(self, file_name, index_sets, scenario=""):
        """ reads input data and restructures the dataframe to return (multi)indexed dict
        :param file_name:  name of selected file
        :param index_sets: index sets of attribute. Creates (multi)index. Corresponds to order in pe.Set/pe.Param
        :return df_output: return existing capacity and existing lifetime """
        column = "year_construction"
        index_list, index_name_list = self.construct_index_list(index_sets, None)
        multiidx = pd.MultiIndex.from_product(index_list, names=index_name_list)
        df_output = pd.Series(index=multiidx, data=0)
        # if no existing capacities
        if not self.analysis["use_existing_capacities"]:
            return df_output

        if f"{file_name}{scenario}.csv" in os.listdir(self.folder_path):
            df_input = self.read_input_data(file_name + scenario)
            # fill output dataframe
            df_output = self.extract_general_input_data(df_input, df_output, file_name, index_name_list, column, default_value=0)
            # get reference year
            reference_year = self.system["reference_year"]
            # calculate remaining lifetime
            df_output[df_output > 0] = - reference_year + df_output[df_output > 0] + self.element.lifetime
        return df_output

    def extract_pwa_data(self, variable_type):
        """ reads input data and restructures the dataframe to return (multi)indexed dict
        :param variable_type: technology approximation type
        :return pwa_dict: dictionary with pwa parameters """
        # attribute names
        if variable_type == "capex":
            _attribute_name = "capex_specific"
        elif variable_type == "conver_efficiency":
            _attribute_name = "conver_efficiency"
        else:
            raise KeyError(f"variable type {variable_type} unknown.")
        _index_sets = ["set_nodes", "set_time_steps"]
        _time_steps = self.energy_system.set_time_steps_yearly
        # import all input data
        df_input_nonlinear = self.read_pwa_files(variable_type, fileType="nonlinear_")
        df_input_breakpoints = self.read_pwa_files(variable_type, fileType="breakpoints_pwa_")
        df_input_linear = self.read_pwa_files(variable_type, fileType="linear_")
        df_linear_exist = self.exists_attribute(_attribute_name)
        assert (
                           df_input_nonlinear is not None and df_input_breakpoints is not None) or df_linear_exist or df_input_linear is not None, f"Neither pwa nor linear data exist for {variable_type} of {self.element.name}"
        # check if capex_specific exists
        if (df_input_nonlinear is not None and df_input_breakpoints is not None):
            # select data
            pwa_dict = {}
            # extract all data values
            nonlinear_values = {}

            if variable_type == "capex":
                # make absolute capex
                df_input_nonlinear["capex"] = df_input_nonlinear["capex"] * df_input_nonlinear["capacity"]
            for column in df_input_nonlinear.columns:
                nonlinear_values[column] = df_input_nonlinear[column].to_list()

            # assert that breakpoint variable (x variable in nonlinear input)
            assert df_input_breakpoints.columns[
                       0] in df_input_nonlinear.columns, f"breakpoint variable for pwa '{df_input_breakpoints.columns[0]}' is not in nonlinear variables [{df_input_nonlinear.columns}]"
            breakpoint_variable = df_input_breakpoints.columns[0]
            breakpoints = df_input_breakpoints[breakpoint_variable].to_list()

            pwa_dict[breakpoint_variable] = breakpoints
            pwa_dict["pwa_variables"] = []  # select only those variables that are modeled as pwa
            pwa_dict["bounds"] = {}  # save bounds of variables
            linear_dict = {}
            # min and max total capacity of technology
            min_capacity_tech, max_capacity_tech = (0, min(max(self.element.capacity_limit.values), max(breakpoints)))
            for value_variable in nonlinear_values:
                if value_variable == breakpoint_variable:
                    pwa_dict["bounds"][value_variable] = (min_capacity_tech, max_capacity_tech)
                else:
                    # conduct linear regress
                    linear_regress_object = linregress(nonlinear_values[breakpoint_variable], nonlinear_values[value_variable])
                    # calculate relative intercept (intercept/slope) if slope != 0
                    if linear_regress_object.slope != 0:
                        _relative_intercept = np.abs(linear_regress_object.intercept / linear_regress_object.slope)
                    else:
                        _relative_intercept = np.abs(linear_regress_object.intercept)
                    # check if to a reasonable degree linear
                    if _relative_intercept <= self.solver["linear_regression_check"]["eps_intercept"] and linear_regress_object.rvalue >= self.solver["linear_regression_check"]["epsRvalue"]:
                        # model as linear function
                        slope_lin_reg = linear_regress_object.slope
                        linear_dict[value_variable] = self.create_default_output(index_sets=_index_sets, column=column, time_steps=_time_steps, manual_default_value=slope_lin_reg)[0]
                    else:
                        # model as pwa function
                        pwa_dict[value_variable] = list(np.interp(breakpoints, nonlinear_values[breakpoint_variable], nonlinear_values[value_variable]))
                        pwa_dict["pwa_variables"].append(value_variable)
                        # save bounds
                        _values_between_bounds = [pwa_dict[value_variable][idxBreakpoint] for idxBreakpoint, breakpoint in enumerate(breakpoints) if
                                                  breakpoint >= min_capacity_tech and breakpoint <= max_capacity_tech]
                        _values_between_bounds.extend(list(np.interp([min_capacity_tech, max_capacity_tech], breakpoints, pwa_dict[value_variable])))
                        pwa_dict["bounds"][value_variable] = (min(_values_between_bounds), max(_values_between_bounds))
            # pwa
            if (len(pwa_dict["pwa_variables"]) > 0 and len(linear_dict) == 0):
                is_pwa = True
                return pwa_dict, is_pwa
            # linear
            elif len(linear_dict) > 0 and len(pwa_dict["pwa_variables"]) == 0:
                is_pwa = False
                linear_dict = pd.DataFrame.from_dict(linear_dict)
                linear_dict.columns.name = "carrier"
                linear_dict = linear_dict.stack()
                _conver_efficiency_levels = [linear_dict.index.names[-1]] + linear_dict.index.names[:-1]
                linear_dict = linear_dict.reorder_levels(_conver_efficiency_levels)
                return linear_dict, is_pwa
            # no dependent carrier
            elif len(nonlinear_values) == 1:
                is_pwa = False
                return None, is_pwa
            else:
                raise NotImplementedError(f"There are both linearly and nonlinearly modeled variables in {variable_type} of {self.element.name}. Not yet implemented")
        # linear
        else:
            is_pwa = False
            linear_dict = {}
            if variable_type == "capex":
                linear_dict["capex"] = self.extract_input_data(_attribute_name, index_sets=_index_sets, time_steps=_time_steps)
                return linear_dict, is_pwa
            else:
                _dependent_carrier = list(set(self.element.input_carrier + self.element.output_carrier).difference(self.element.reference_carrier))
                # TODO implement for more than 1 carrier
                if _dependent_carrier == []:
                    return None, is_pwa
                elif len(_dependent_carrier) == 1 and df_input_linear is None:
                    linear_dict[_dependent_carrier[0]] = self.extract_input_data(_attribute_name, index_sets=_index_sets, time_steps=_time_steps)
                else:
                    df_output, default_value, index_name_list = self.create_default_output(_index_sets, None, time_steps=_time_steps, manual_default_value=1)
                    assert (df_input_linear is not None), f"input file for linear_conver_efficiency could not be imported."
                    df_input_linear = df_input_linear.rename(columns={'year': 'time'})
                    for carrier in _dependent_carrier:
                        linear_dict[carrier] = self.extract_general_input_data(df_input_linear, df_output, "linearConverEfficiency", index_name_list, carrier, default_value).copy(deep=True)
                linear_dict = pd.DataFrame.from_dict(linear_dict)
                linear_dict.columns.name = "carrier"
                linear_dict = linear_dict.stack()
                _conver_efficiency_levels = [linear_dict.index.names[-1]] + linear_dict.index.names[:-1]
                linear_dict = linear_dict.reorder_levels(_conver_efficiency_levels)
                return linear_dict, is_pwa

    def read_pwa_files(self, variable_type, fileType):
        """ reads pwa Files
        :param variable_type: technology approximation type
        :param fileType: either breakpointsPWA, linear, or nonlinear
        :return df_input: raw input file"""
        df_input = self.read_input_data(fileType + variable_type)
        if df_input is not None:
            if "unit" in df_input.values:
                columns = df_input.iloc[-1][df_input.iloc[-1] != "unit"].dropna().index
            else:
                columns = df_input.columns
            df_input_units = df_input[columns].iloc[-1]
            df_input = df_input.iloc[:-1]
            _df_input_multiplier = df_input_units.apply(lambda unit: self.unit_handling.get_unit_multiplier(unit))
            df_input = df_input.apply(lambda column: pd.to_numeric(column, errors='coerce'))
            df_input[columns] = df_input[columns] * _df_input_multiplier
        return df_input

    def create_default_output(self, index_sets, column, file_name=None, time_steps=None, manual_default_value=None, scenario=""):
        """ creates default output dataframe
        :param file_name: name of selected file.
        :param index_sets: index sets of attribute. Creates (multi)index. Corresponds to order in pe.Set/pe.Param
        :param column: select specific column
        :param time_steps: specific time_steps of element
        :param scenario: investigated scenario
        :param manual_default_value: if given, use manual_default_value instead of searching for default value in attributes.csv"""
        # select index
        index_list, index_name_list = self.construct_index_list(index_sets, time_steps)
        # create pd.MultiIndex and select data
        if index_sets:
            index_multi_index = pd.MultiIndex.from_product(index_list, names=index_name_list)
        else:
            index_multi_index = pd.Index([0])
        if manual_default_value:
            default_value = {"value": manual_default_value, "multiplier": 1}
            default_name = None
        else:
            # check if default value exists in attributes.csv, with or without "Default" Suffix
            if column:
                default_name = column
            else:
                default_name = file_name
            default_value = self.extract_attribute(default_name, scenario=scenario)

        # create output Series filled with default value
        if default_value is None:
            df_output = pd.Series(index=index_multi_index, dtype=float)
        else:
            df_output = pd.Series(index=index_multi_index, data=default_value["value"], dtype=float)
        # save unit of attribute of element converted to base unit
        self.save_unit_of_attribute(default_name, scenario)
        return df_output, default_value, index_name_list

    def save_unit_of_attribute(self, file_name, scenario=""):
        """ saves the unit of an attribute, converted to the base unit """
        # if numerics analyzed
        if self.solver["analyze_numerics"]:
            attributes = "attributes"
            if scenario and os.path.exists(os.path.join(self.folder_path, "attributes" + scenario + ".csv")):
                attributes += scenario
            df_input = self.read_input_data(attributes).set_index("index").squeeze(axis=1)
            # get attribute
            if file_name:
                attribute_name = self.adapt_attribute_name(file_name, df_input)
                input_unit = df_input.loc[attribute_name, "unit"]
            else:
                input_unit = np.nan
            self.unit_handling.set_base_unit_combination(input_unit=input_unit, attribute=(self.element.name, file_name))

    def save_values_of_attribute(self, df_output, file_name):
        """ saves the values of an attribute """
        # if numerics analyzed
        if self.solver["analyze_numerics"]:
            if file_name:
                df_output_reduced = df_output[(df_output != 0) & (df_output.abs() != np.inf)]
                if not df_output_reduced.empty:
                    self.unit_handling.set_attribute_values(df_output=df_output_reduced, attribute=(self.element.name, file_name))

    def construct_index_list(self, index_sets, time_steps):
        """ constructs index list from index sets and returns list of indices and list of index names
        :param index_sets: index sets of attribute. Creates (multi)index. Corresponds to order in pe.Set/pe.Param
        :param time_steps: specific time_steps of element
        :return index_list: list of indices
        :return index_name_list: list of name of indices
        """
        index_list = []
        index_name_list = []
        # add rest of indices
        for index in index_sets:
            index_name_list.append(self.index_names[index])
            if index == "set_time_steps" and time_steps:
                index_list.append(time_steps)
            elif index == "set_existing_technologies":
                index_list.append(self.element.set_existing_technologies)
            elif index in self.system:
                index_list.append(self.system[index])
            elif hasattr(self.energy_system, index):
                index_list.append(getattr(self.energy_system, index))
            else:
                raise AttributeError(f"Index '{index}' cannot be found.")
        return index_list, index_name_list

    def exists_attribute(self, file_name, column=None):
        """ checks if default value or timeseries of an attribute exists in the input data
        :param file_name: name of selected file
        :param column: select specific column
        """
        # check if default value exists
        if column:
            default_name = column
        else:
            default_name = file_name
        default_value = self.extract_attribute(default_name)

        if default_value is None or math.isnan(default_value["value"]):  # if no default value exists or default value is nan
            _dfInput = self.read_input_data(file_name)
            return (_dfInput is not None)
        elif default_value and not math.isnan(default_value["value"]):  # if default value exists and is not nan
            return True
        else:
            return False

    @staticmethod
    def extract_from_input_without_missing_index(df_input, index_name_list, column, file_name):
        """ extracts the demanded values from Input dataframe and reformulates dataframe
        :param df_input: raw input dataframe
        :param index_name_list: list of name of indices
        :param column: select specific column
        :param file_name: name of selected file
        :return df_input: reformulated input dataframe
        """
        if index_name_list:
            df_input = df_input.set_index(index_name_list)
        if column:
            assert column in df_input.columns, f"Requested column {column} not in columns {df_input.columns.to_list()} of input file {file_name}"
            df_input = df_input[column]
        else:
            # check if only one column remaining
            assert len(df_input.columns) == 1, f"Input file for {file_name} has more than one value column: {df_input.columns.to_list()}"
            df_input = df_input.squeeze(axis=1)
        return df_input

    @staticmethod
    def extract_from_input_with_missing_index(df_input, df_output, index_name_list, column, file_name, missing_index):
        """ extracts the demanded values from Input dataframe and reformulates dataframe if the index is missing.
        Either, the missing index is the column of df_input, or it is actually missing in df_input.
        Then, the values in df_input are extended to all missing index values.
        :param df_input: raw input dataframe
        :param df_output: default output dataframe
        :param index_name_list: list of name of indices
        :param column: select specific column
        :param file_name: name of selected file
        :param missing_index: missing index in df_input
        :return df_input: reformulated input dataframe
        """
        index_name_list.remove(missing_index)
        df_input = df_input.set_index(index_name_list)
        # missing index values
        requested_index_values = set(df_output.index.get_level_values(missing_index))
        # the missing index is the columns of df_input
        _requested_index_values_in_columns = requested_index_values.intersection(df_input.columns)
        if _requested_index_values_in_columns:
            requested_index_values = _requested_index_values_in_columns
            df_input.columns = df_input.columns.set_names(missing_index)
            df_input = df_input[list(requested_index_values)].stack()
            df_input = df_input.reorder_levels(df_output.index.names)
        # the missing index does not appear in df_input
        # the values in df_input are extended to all missing index values
        else:
            # logging.info(f"Missing index {missing_index} detected in {file_name}. Input dataframe is extended by this index")
            _df_input_index_temp = pd.MultiIndex.from_product([df_input.index, requested_index_values], names=df_input.index.names + [missing_index])
            _df_input_temp = pd.Series(index=_df_input_index_temp, dtype=float)
            if column in df_input.columns:
                df_input = df_input[column].loc[_df_input_index_temp.get_level_values(df_input.index.names[0])].squeeze()
            else:
                if isinstance(df_input, pd.Series):
                    df_input = df_input.to_frame()
                if df_input.shape[1] == 1:
                    df_input = df_input.loc[_df_input_index_temp.get_level_values(df_input.index.names[0])].squeeze()
                else:
                    assert _df_input_temp.index.names[-1] != "time", f"Only works if columns contain time index and not for {_df_input_temp.index.names[-1]}"
                    df_input = _df_input_temp.to_frame().apply(lambda row: df_input.loc[row.name[0:-1], str(row.name[-1])], axis=1)
            df_input.index = _df_input_temp.index
            df_input = df_input.reorder_levels(order=df_output.index.names)
            if isinstance(df_input, pd.DataFrame):
                df_input = df_input.squeeze()
        return df_input

    @staticmethod
    def extract_from_input_for_existing_capacities(df_input, df_output, index_name_list, column, missing_index):
        """ extracts the demanded values from input dataframe if extracting existing capacities
        :param df_input: raw input dataframe
        :param df_output: default output dataframe
        :param index_name_list: list of name of indices
        :param column: select specific column
        :param missing_index: missing index in df_input
        :return df_output: filled output dataframe
        """
        index_name_list.remove(missing_index)
        df_input = df_input.set_index(index_name_list)
        set_location = df_input.index.unique()
        for location in set_location:
            if location in df_output.index.get_level_values(index_name_list[0]):
                values = df_input[column].loc[location].tolist()
                if isinstance(values, int) or isinstance(values, float):
                    index = [0]
                else:
                    index = list(range(len(values)))
                df_output.loc[location, index] = values
        return df_output
