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
import numpy  as np
import pandas as pd
from scipy.stats import linregress

class DataInput():

    def __init__(self,element,system,analysis,solver,energy_system,unit_handling):
        """ data input object to extract input data
        :param element: element for which data is extracted
        :param system: dictionary defining the system
        :param analysis: dictionary defining the analysis framework
        :param solver: dictionary defining the solver 
        :param energy_system: instance of class <EnergySystem> to define energy_system
        :param unit_handling: instance of class <UnitHandling> to convert units """
        self.element        = element
        self.system         = system
        self.analysis       = analysis
        self.solver         = solver
        self.energy_system   = energy_system
        self.unit_handling   = unit_handling
        # extract folder path
        self.folderPath = getattr(self.element,"input_path")

        # get names of indices
        # self.index_names     = {index_name: self.analysis['headerDataInputs'][index_name][0] for index_name in self.analysis['headerDataInputs']}
        self.index_names     = self.analysis['headerDataInputs']

    def extract_input_data(self,file_name,index_sets,column=None,time_steps=None,scenario=""):
        """ reads input data and restructures the dataframe to return (multi)indexed dict
        :param file_name: name of selected file.
        :param index_sets: index sets of attribute. Creates (multi)index. Corresponds to order in pe.Set/pe.Param
        :param column: select specific column
        :param time_steps: specific time_steps of element
        :return dataDict: dictionary with attribute values """

        # generic time steps
        if not time_steps:
            time_steps = self.energy_system.set_base_time_steps
        # if time steps are the yearly base time steps
        elif time_steps is self.energy_system.set_base_time_steps_yearly:
            self.extractYearlyVariation(file_name,index_sets,column)

        # if existing capacities and existing capacities not used
        if (file_name == "existingCapacity" or file_name == "existingCapacityEnergy") and not self.analysis["useExistingCapacities"]:
            dfOutput,*_ = self.createDefaultOutput(index_sets,column,file_name= file_name,time_steps=time_steps,manualDefaultValue=0,scenario=scenario)
            return dfOutput
        else:
            dfOutput, defaultValue, indexNameList = self.createDefaultOutput(index_sets,column,file_name =file_name, time_steps= time_steps,scenario=scenario)
        # set defaultName
        if column:
            defaultName = column
        else:
            defaultName = file_name
        # read input file
        dfInput = self.readInputData(file_name+scenario)

        assert(dfInput is not None or defaultValue is not None), f"input file for attribute {defaultName} could not be imported and no default value is given."
        if dfInput is not None and not dfInput.empty:
            dfOutput = self.extractGeneralInputData(dfInput,dfOutput,file_name,indexNameList,column,defaultValue)
        # save parameter values for analysis of numerics
        self.saveValuesOfAttribute(dfOutput=dfOutput,file_name=defaultName)
        return dfOutput

    def extractGeneralInputData(self,dfInput,dfOutput,file_name,indexNameList,column,defaultValue):
        """ fills dfOutput with data from dfInput
        :param dfInput: raw input dataframe
        :param dfOutput: empty output dataframe, only filled with defaultValue 
        :param file_name: name of selected file
        :param indexNameList: list of name of indices
        :param column: select specific column
        :param defaultValue: default for dataframe
        :return dfOutput: filled output dataframe """

        # select and drop scenario
        assert dfInput.columns is not None, f"Input file '{file_name}' has no columns"
        assert self.index_names["setScenarios"] not in dfInput.columns, f"the index '{self.index_names['setScenarios']}' is depreciated, but still found in input file '{file_name}'"
        # set index by indexNameList
        missingIndex = list(set(indexNameList) - set(indexNameList).intersection(set(dfInput.columns)))
        assert len(missingIndex) <= 1, f"More than one the requested index sets ({missingIndex}) are missing from input file for {file_name}"

        # no indices missing
        if len(missingIndex) == 0:
            dfInput = DataInput.extractFromInputWithoutMissingIndex(dfInput,indexNameList,column,file_name)
        else:
            missingIndex = missingIndex[0]
            # check if special case of existing Technology
            if "existingTechnology" in missingIndex:
                if column:
                    defaultName = column
                else:
                    defaultName = file_name
                dfOutput = DataInput.extractFromInputForExistingCapacities(dfInput,dfOutput,indexNameList,defaultName,missingIndex)
                if isinstance(defaultValue,dict):
                    dfOutput = dfOutput * defaultValue["multiplier"]
                return dfOutput
            # index missing
            else:
                dfInput = DataInput.extractFromInputWithMissingIndex(dfInput,dfOutput,copy.deepcopy(indexNameList),column,file_name,missingIndex)

        # apply multiplier to input data
        dfInput     = dfInput * defaultValue["multiplier"]
        # delete nans
        dfInput     = dfInput.dropna()

        # get common index of dfOutput and dfInput
        if not isinstance(dfInput.index, pd.MultiIndex):
            index_list               = dfInput.index.to_list()
            if len(index_list) == 1:
                indexMultiIndex     = pd.MultiIndex.from_tuples([(index_list[0],)], names=[dfInput.index.name])
            else:
                indexMultiIndex     = pd.MultiIndex.from_product([index_list], names=[dfInput.index.name])
            dfInput                 = pd.Series(index=indexMultiIndex, data=dfInput.to_list())
        commonIndex                 = dfOutput.index.intersection(dfInput.index)
        assert defaultValue is not None or len(commonIndex) == len(dfOutput.index), f"Input for {file_name} does not provide entire dataset and no default given in attributes.csv"
        dfOutput.loc[commonIndex]   = dfInput.loc[commonIndex]
        return dfOutput

    def readInputData(self,inputFileName):
        """ reads input data and returns raw input dataframe
        :param inputFileName: name of selected file
        :return dfInput: pd.DataFrame with input data """

        # append .csv suffix
        inputFileName += ".csv"

        # select data
        fileNames = os.listdir(self.folderPath)
        if inputFileName in fileNames:
            dfInput = pd.read_csv(os.path.join(self.folderPath, inputFileName), header=0, index_col=None)
            return dfInput
        else:
            return None

    def extractAttributeData(self,attribute_name,skipWarning = False,scenario=""):
        """ reads input data and restructures the dataframe to return (multi)indexed dict
        :param attribute_name: name of selected attribute
        :param skipWarning: boolean to indicate if "Default" warning is skipped
        :return attributeValue: attribute value """
        filename = "attributes"
        dfInput  = self.readInputData(filename+scenario)
        if dfInput is not None:
            dfInput = dfInput.set_index("index").squeeze(axis=1)
            name    = self.adaptAttributeName(attribute_name, dfInput, skipWarning)
        if dfInput is None or name is None:
            dfInput = self.readInputData(filename)
            if dfInput is not None:
                dfInput = dfInput.set_index("index").squeeze(axis=1)
            else:
                return None
        attribute_name = self.adaptAttributeName(attribute_name,dfInput,skipWarning)
        if attribute_name is not None:
            # get attribute
            attributeValue = dfInput.loc[attribute_name, "value"]
            multiplier = self.unit_handling.getUnitMultiplier(dfInput.loc[attribute_name, "unit"])
            try:
                attribute = {"value": float(attributeValue) * multiplier, "multiplier": multiplier}
                return attribute
            except:
                return attributeValue
        else:
            return None

    def adaptAttributeName(self,attribute_name,dfInput,skipWarning=False):
        """ check if attribute in index"""
        if attribute_name + "Default" not in dfInput.index:
            if attribute_name not in dfInput.index:
                return None
            elif not skipWarning:
                warnings.warn(
                    f"Attribute names without 'Default' suffix will be deprecated. \nChange for {attribute_name} of attributes in path {self.folderPath}",
                    FutureWarning)
        else:
            attribute_name = attribute_name + "Default"
        return attribute_name

    def extractYearlyVariation(self,file_name,index_sets,column):
        """ reads the yearly variation of a time dependent quantity
        :param self.folderPath: path to input files
        :param file_name: name of selected file.
        :param index_sets: index sets of attribute. Creates (multi)index. Corresponds to order in pe.Set/pe.Param
        :param column: select specific column
        """
        # remove intrayearly time steps from index set and add interyearly time steps
        _index_sets = copy.deepcopy(index_sets)
        _index_sets.remove("set_time_steps")
        _index_sets.append("set_time_steps_yearly")
        # add YearlyVariation to file_name
        file_name  += "YearlyVariation"
        # read input data
        dfInput         = self.readInputData(file_name)
        if dfInput is not None:
            if column is not None and column not in dfInput:
                return
            dfOutput, defaultValue, indexNameList = self.createDefaultOutput(_index_sets,column,file_name = file_name, manualDefaultValue=1)
            # set yearlyVariation attribute to dfOutput
            if column:
                _selectedColumn         = column
                _nameYearlyVariation    = column+"YearlyVariation"
            else:
                _selectedColumn         = None
                _nameYearlyVariation    = file_name
            dfOutput = self.extractGeneralInputData(dfInput, dfOutput, file_name, indexNameList, _selectedColumn,defaultValue)
            setattr(self,_nameYearlyVariation,dfOutput)

    def extract_locations(self,extractNodes = True):
        """ reads input data to extract nodes or edges.
        :param extractNodes: boolean to switch between nodes and edges """
        if extractNodes:
            setNodesConfig  = self.system["setNodes"]
            setNodesInput   = self.readInputData("setNodes")["node"].to_list()
            # if no nodes specified in system, use all nodes
            if len(setNodesConfig) == 0 and not len(setNodesInput) == 0:
                self.system["setNodes"] = setNodesInput
                setNodesConfig          = setNodesInput
            else:
                assert len(setNodesConfig) > 1, f"ZENx is a spatially distributed model. Please specify at least 2 nodes."
                _missingNodes   = list(set(setNodesConfig).difference(setNodesInput))
                assert len(_missingNodes) == 0, f"The nodes {_missingNodes} were declared in the config but do not exist in the input file {self.folderPath+'setNodes'}"
            if not isinstance(setNodesConfig, list):
                setNodesConfig = setNodesConfig.to_list()
            setNodesConfig.sort()
            return setNodesConfig
        else:
            set_edges_input = self.readInputData("setEdges")
            if set_edges_input is not None:
                setEdges        = set_edges_input[(set_edges_input["nodeFrom"].isin(self.energy_system.setNodes)) & (set_edges_input["nodeTo"].isin(self.energy_system.setNodes))]
                setEdges        = setEdges.set_index("edge")
                return setEdges
            else:
                return None

    def extractConversionCarriers(self):
        """ reads input data and extracts conversion carriers
        :param self.folderPath: path to input files
        :return carrierDict: dictionary with input and output carriers of technology """
        carrierDict = {}
        # get carriers
        for _carrierType in ["inputCarrier","outputCarrier"]:
            # TODO implement for multiple carriers
            _carrierString = self.extractAttributeData(_carrierType,skipWarning = True)
            if type(_carrierString) == str:
                _carrierList = _carrierString.strip().split(" ")
            else:
                _carrierList = []
            carrierDict[_carrierType] = _carrierList

        return carrierDict

    def extractSetExistingTechnologies(self, storageEnergy = False):
        """ reads input data and creates setExistingCapacity for each technology
        :param storageEnergy: boolean if existing energy capacity of storage technology (instead of power)
        :return setExistingTechnologies: return set existing technologies"""
        if self.analysis["useExistingCapacities"]:
            if storageEnergy:
                _energyString = "Energy"
            else:
                _energyString = ""

            dfInput = self.readInputData(f"existingCapacity{_energyString}")
            if dfInput is None:
                return  [0]

            if self.element.name in self.system["setTransportTechnologies"]:
                location = "edge"
            else:
                location = "node"
            maxNodeCount = dfInput[location].value_counts().max()
            setExistingTechnologies = np.arange(0, maxNodeCount)
        else:
            setExistingTechnologies = np.array([0])

        return setExistingTechnologies

    def extractLifetimeExistingTechnology(self, file_name, index_sets):
        """ reads input data and restructures the dataframe to return (multi)indexed dict
        :param file_name:  name of selected file
        :param index_sets: index sets of attribute. Creates (multi)index. Corresponds to order in pe.Set/pe.Param
        :return existingLifetimeDict: return existing capacity and existing lifetime """
        column   = "yearConstruction"
        dfOutput = pd.Series(index=self.element.existingCapacity.index,data=0)
        # if no existing capacities
        if not self.analysis["useExistingCapacities"]:
            return dfOutput

        if f"{file_name}.csv" in os.listdir(self.folderPath):
            index_list, indexNameList = self.constructIndexList(index_sets, None)
            dfInput                  = self.readInputData( file_name)
            # fill output dataframe
            dfOutput = self.extractGeneralInputData(dfInput, dfOutput, file_name, indexNameList, column, defaultValue = 0)
            # get reference year
            referenceYear            = self.system["referenceYear"]
            # calculate remaining lifetime
            dfOutput[dfOutput > 0]   = - referenceYear + dfOutput[dfOutput > 0] + self.element.lifetime

        return dfOutput

    def extractPWAData(self,variableType):
        """ reads input data and restructures the dataframe to return (multi)indexed dict
        :param variableType: technology approximation type
        :return PWADict: dictionary with PWA parameters """
        # attribute names
        if variableType == "Capex":
            _attributeName  = "capexSpecific"
        elif variableType == "ConverEfficiency":
            _attributeName  = "converEfficiency"
        else:
            raise KeyError(f"variable type {variableType} unknown.")
        _index_sets = ["setNodes", "set_time_steps"]
        _time_steps = self.energy_system.set_time_steps_yearly
        # import all input data
        dfInputNonlinear    = self.readPWAFiles(variableType, fileType="nonlinear")
        dfInputBreakpoints  = self.readPWAFiles(variableType, fileType="breakpointsPWA")
        dfInputLinear       = self.readPWAFiles(variableType, fileType="linear")
        ifLinearExist       = self.ifAttributeExists(_attributeName)
        assert (dfInputNonlinear is not None and dfInputBreakpoints is not None) \
               or ifLinearExist \
               or dfInputLinear is not None, \
            f"Neither PWA nor linear data exist for {variableType} of {self.element.name}"
        # check if capexSpecific exists
        if (dfInputNonlinear is not None and dfInputBreakpoints is not None):
            # select data
            PWADict = {}
            # extract all data values
            nonlinearValues     = {}

            if variableType == "Capex":
                # make absolute capex
                dfInputNonlinear["capex"] = dfInputNonlinear["capex"]*dfInputNonlinear["capacity"]
            for column in dfInputNonlinear.columns:
                nonlinearValues[column] = dfInputNonlinear[column].to_list()

            # assert that breakpoint variable (x variable in nonlinear input)
            assert dfInputBreakpoints.columns[0] in dfInputNonlinear.columns, f"breakpoint variable for PWA '{dfInputBreakpoints.columns[0]}' is not in nonlinear variables [{dfInputNonlinear.columns}]"
            breakpointVariable = dfInputBreakpoints.columns[0]
            breakpoints = dfInputBreakpoints[breakpointVariable].to_list()

            PWADict[breakpointVariable] = breakpoints
            PWADict["PWAVariables"]     = [] # select only those variables that are modeled as PWA
            PWADict["bounds"]           = {} # save bounds of variables
            LinearDict                  = {}
            # min and max total capacity of technology
            minCapacityTech,maxCapacityTech = (0,min(max(self.element.capacityLimit.values),max(breakpoints)))
            for valueVariable in nonlinearValues:
                if valueVariable == breakpointVariable:
                    PWADict["bounds"][valueVariable] = (minCapacityTech,maxCapacityTech)
                else:
                    # conduct linear regress
                    linearRegressObject = linregress(nonlinearValues[breakpointVariable],nonlinearValues[valueVariable])
                    # calculate relative intercept (intercept/slope) if slope != 0
                    if linearRegressObject.slope != 0:
                        _relativeIntercept = np.abs(linearRegressObject.intercept/linearRegressObject.slope)
                    else:
                        _relativeIntercept = np.abs(linearRegressObject.intercept)
                    # check if to a reasonable degree linear
                    if _relativeIntercept <= self.solver["linearRegressionCheck"]["epsIntercept"] and linearRegressObject.rvalue >= self.solver["linearRegressionCheck"]["epsRvalue"]:
                        # model as linear function
                        slopeLinReg = linearRegressObject.slope
                        LinearDict[valueVariable] = self.createDefaultOutput(index_sets=_index_sets, column=column, time_steps=_time_steps,
                                                 manualDefaultValue=slopeLinReg)[0]
                    else:
                        # model as PWA function
                        PWADict[valueVariable] = list(np.interp(breakpoints,nonlinearValues[breakpointVariable],nonlinearValues[valueVariable]))
                        PWADict["PWAVariables"].append(valueVariable)
                        # save bounds
                        _valuesBetweenBounds = [PWADict[valueVariable][idxBreakpoint] for idxBreakpoint,breakpoint in enumerate(breakpoints) if breakpoint >= minCapacityTech and breakpoint <= maxCapacityTech]
                        _valuesBetweenBounds.extend(list(np.interp([minCapacityTech,maxCapacityTech],breakpoints,PWADict[valueVariable])))
                        PWADict["bounds"][valueVariable] = (min(_valuesBetweenBounds),max(_valuesBetweenBounds))
            # PWA
            if (len(PWADict["PWAVariables"]) > 0 and len(LinearDict) == 0):
                isPWA = True
                return PWADict, isPWA
            # linear
            elif len(LinearDict) > 0 and len(PWADict["PWAVariables"]) == 0:
                isPWA = False
                LinearDict              = pd.DataFrame.from_dict(LinearDict)
                LinearDict.columns.name = "carrier"
                LinearDict              = LinearDict.stack()
                _converEfficiencyLevels = [LinearDict.index.names[-1]] + LinearDict.index.names[:-1]
                LinearDict              = LinearDict.reorder_levels(_converEfficiencyLevels)
                return LinearDict,  isPWA
            # no dependent carrier
            elif len(nonlinearValues) == 1:
                isPWA = False
                return None, isPWA
            else:
                raise NotImplementedError(f"There are both linearly and nonlinearly modeled variables in {variableType} of {self.element.name}. Not yet implemented")
        # linear
        else:
            isPWA = False
            LinearDict = {}
            if variableType == "Capex":
                LinearDict["capex"] = self.extract_input_data("capexSpecific", index_sets=_index_sets, time_steps=_time_steps)
                return LinearDict,isPWA
            else:
                _dependentCarrier = list(set(self.element.inputCarrier + self.element.outputCarrier).difference(self.element.referenceCarrier))
                # TODO implement for more than 1 carrier
                if _dependentCarrier == []:
                    return None, isPWA
                elif len(_dependentCarrier) == 1 and dfInputLinear is None:
                    LinearDict[_dependentCarrier[0]] = self.extract_input_data(_attributeName, index_sets=_index_sets, time_steps=_time_steps)
                else:
                    dfOutput,defaultValue,indexNameList = self.createDefaultOutput(_index_sets, None, time_steps=_time_steps, manualDefaultValue=1)
                    assert (dfInputLinear is not None), f"input file for linearConverEfficiency could not be imported."
                    dfInputLinear = dfInputLinear.rename(columns={'year': 'time'})
                    for carrier in _dependentCarrier:
                        LinearDict[carrier]        = self.extractGeneralInputData(dfInputLinear, dfOutput, "linearConverEfficiency", indexNameList, carrier, defaultValue).copy(deep=True)
                LinearDict = pd.DataFrame.from_dict(LinearDict)
                LinearDict.columns.name = "carrier"
                LinearDict = LinearDict.stack()
                _converEfficiencyLevels = [LinearDict.index.names[-1]] + LinearDict.index.names[:-1]
                LinearDict = LinearDict.reorder_levels(_converEfficiencyLevels)
                return LinearDict,isPWA

    def readPWAFiles(self,variableType,fileType):
        """ reads PWA Files
        :param variableType: technology approximation type
        :param fileType: either breakpointsPWA, linear, or nonlinear
        :return dfInput: raw input file"""
        dfInput             = self.readInputData(fileType+variableType)
        if dfInput is not None:
            if "unit" in dfInput.values:
                columns = dfInput.iloc[-1][dfInput.iloc[-1] != "unit"].dropna().index
            else:
                columns = dfInput.columns
            dfInputUnits        = dfInput[columns].iloc[-1]
            dfInput             = dfInput.iloc[:-1]
            dfInputMultiplier   = dfInputUnits.apply(lambda unit: self.unit_handling.getUnitMultiplier(unit))
            #dfInput[columns]    = dfInput[columns].astype(float
            dfInput             = dfInput.apply(lambda column: pd.to_numeric(column, errors='coerce'))
            dfInput[columns]    = dfInput[columns] * dfInputMultiplier
        return dfInput

    def createDefaultOutput(self,index_sets,column,file_name=None,time_steps=None,manualDefaultValue = None,scenario = ""):
        """ creates default output dataframe
        :param file_name: name of selected file.
        :param index_sets: index sets of attribute. Creates (multi)index. Corresponds to order in pe.Set/pe.Param
        :param column: select specific column
        :param time_steps: specific time_steps of element
        :param scenario: investigated scenario
        :param manualDefaultValue: if given, use manualDefaultValue instead of searching for default value in attributes.csv"""
        # select index
        index_list, indexNameList = self.constructIndexList(index_sets, time_steps)
        # create pd.MultiIndex and select data
        if index_sets:
            indexMultiIndex = pd.MultiIndex.from_product(index_list, names=indexNameList)
        else:
            indexMultiIndex = pd.Index([0])
        if manualDefaultValue:
            defaultValue = {"value":manualDefaultValue,"multiplier":1}
            defaultName  = None
        else:
            # check if default value exists in attributes.csv, with or without "Default" Suffix
            if column:
                defaultName = column
            else:
                defaultName = file_name
            defaultValue = self.extractAttributeData(defaultName,scenario=scenario)

        # create output Series filled with default value
        if defaultValue is None:
            dfOutput = pd.Series(index=indexMultiIndex, dtype=float)
        else:
            dfOutput = pd.Series(index=indexMultiIndex, data=defaultValue["value"], dtype=float)
        # save unit of attribute of element converted to base unit
        self.saveUnitOfAttribute(defaultName,scenario)
        return dfOutput,defaultValue,indexNameList

    def saveUnitOfAttribute(self,file_name,scenario=""):
        """ saves the unit of an attribute, converted to the base unit """
        # if numerics analyzed
        if self.solver["analyzeNumerics"]:
            if file_name:
                dfInput = self.readInputData("attributes" + scenario).set_index("index").squeeze(axis=1)
                # get attribute
                attribute_name = self.adaptAttributeName(file_name,dfInput)
                inputUnit = dfInput.loc[attribute_name, "unit"]
                self.unit_handling.setBaseUnitCombination(inputUnit=inputUnit,attribute=(self.element.name,file_name))

    def saveValuesOfAttribute(self,dfOutput,file_name):
        """ saves the values of an attribute """
        # if numerics analyzed
        if self.solver["analyzeNumerics"]:
            if file_name:
                dfOutputReduced = dfOutput[(dfOutput != 0) & (dfOutput.abs() != np.inf)]
                if not dfOutputReduced.empty:
                    self.unit_handling.setAttributeValues(dfOutput= dfOutputReduced,attribute=(self.element.name,file_name))

    def constructIndexList(self,index_sets,time_steps):
        """ constructs index list from index sets and returns list of indices and list of index names
        :param index_sets: index sets of attribute. Creates (multi)index. Corresponds to order in pe.Set/pe.Param
        :param time_steps: specific time_steps of element
        :return index_list: list of indices
        :return indexNameList: list of name of indices
        """
        index_list     = []
        indexNameList = []

        # add rest of indices
        for index in index_sets:
            indexNameList.append(self.index_names[index])
            if index == "set_time_steps" and time_steps:
                index_list.append(time_steps)
            elif index == "setExistingTechnologies":
                index_list.append(self.element.setExistingTechnologies)
            elif index in self.system:
                index_list.append(self.system[index])
            elif hasattr(self.energy_system,index):
                index_list.append(getattr(self.energy_system,index))
            else:
                raise AttributeError(f"Index '{index}' cannot be found.")
        return index_list,indexNameList

    def ifAttributeExists(self, file_name, column=None):
        """ checks if default value or timeseries of an attribute exists in the input data
        :param file_name: name of selected file
        :param column: select specific column
        """
        # check if default value exists
        if column:
            defaultName = column
        else:
            defaultName = file_name
        defaultValue = self.extractAttributeData(defaultName)

        if defaultValue is None or math.isnan(defaultValue["value"]): # if no default value exists or default value is nan
            _dfInput = self.readInputData(file_name)
            return (_dfInput is not None)
        elif defaultValue and not math.isnan(defaultValue["value"]): # if default value exists and is not nan
            return True
        else:
            return False

    @staticmethod
    def extractFromInputWithoutMissingIndex(dfInput,indexNameList,column,file_name):
        """ extracts the demanded values from Input dataframe and reformulates dataframe
        :param dfInput: raw input dataframe
        :param indexNameList: list of name of indices
        :param column: select specific column
        :param file_name: name of selected file
        :return dfInput: reformulated input dataframe
        """
        dfInput = dfInput.set_index(indexNameList)
        if column:
            assert column in dfInput.columns, f"Requested column {column} not in columns {dfInput.columns.to_list()} of input file {file_name}"
            dfInput = dfInput[column]
        else:
            # check if only one column remaining
            assert len(dfInput.columns) == 1,f"Input file for {file_name} has more than one value column: {dfInput.columns.to_list()}"
            dfInput = dfInput.squeeze(axis=1)
        return dfInput

    @staticmethod
    def extractFromInputWithMissingIndex(dfInput,dfOutput, indexNameList, column, file_name,missingIndex):
        """ extracts the demanded values from Input dataframe and reformulates dataframe if the index is missing.
        Either, the missing index is the column of dfInput, or it is actually missing in dfInput.
        Then, the values in dfInput are extended to all missing index values.
        :param dfInput: raw input dataframe
        :param dfOutput: default output dataframe
        :param indexNameList: list of name of indices
        :param column: select specific column
        :param file_name: name of selected file
        :param missingIndex: missing index in dfInput
        :return dfInput: reformulated input dataframe
        """
        indexNameList.remove(missingIndex)
        dfInput                 = dfInput.set_index(indexNameList)
        # missing index values
        requestedIndexValues    = set(dfOutput.index.get_level_values(missingIndex))
        # the missing index is the columns of dfInput
        _requestedIndexValuesInColumns  = requestedIndexValues.intersection(dfInput.columns)
        if _requestedIndexValuesInColumns:
            requestedIndexValues    = _requestedIndexValuesInColumns
            dfInput.columns         = dfInput.columns.set_names(missingIndex)
            dfInput                 = dfInput[list(requestedIndexValues)].stack()
            dfInput                 = dfInput.reorder_levels(dfOutput.index.names)
        # the missing index does not appear in dfInput
        # the values in dfInput are extended to all missing index values
        else:
            # logging.info(f"Missing index {missingIndex} detected in {file_name}. Input dataframe is extended by this index")
            _dfInputIndexTemp   = pd.MultiIndex.from_product([dfInput.index,requestedIndexValues],names=dfInput.index.names+[missingIndex])
            _dfInputTemp        = pd.Series(index=_dfInputIndexTemp, dtype=float)
            if column in dfInput.columns:
                dfInput = dfInput[column].loc[_dfInputIndexTemp.get_level_values(dfInput.index.names[0])].squeeze()
                # much slower than overwriting index:
                # dfInput         = _dfInputTemp.to_frame().apply(lambda row: dfInput.loc[row.name[0], column].squeeze(),axis=1)
            else:
                if isinstance(dfInput,pd.Series):
                    dfInput = dfInput.to_frame()
                if dfInput.shape[1] == 1:
                    dfInput         = dfInput.loc[_dfInputIndexTemp.get_level_values(dfInput.index.names[0])].squeeze()
                else:
                    assert _dfInputTemp.index.names[-1] != "time", f"Only works if columns contain time index and not for {_dfInputTemp.index.names[-1]}"
                    dfInput = _dfInputTemp.to_frame().apply(lambda row: dfInput.loc[row.name[0:-1],str(row.name[-1])],axis=1)
            dfInput.index = _dfInputTemp.index
            dfInput = dfInput.reorder_levels(order=dfOutput.index.names)
            if isinstance(dfInput,pd.DataFrame):
                dfInput = dfInput.squeeze()
        return dfInput

    @staticmethod
    def extractFromInputForExistingCapacities(dfInput,dfOutput, indexNameList, column, missingIndex):
        """ extracts the demanded values from input dataframe if extracting existing capacities
        :param dfInput: raw input dataframe
        :param dfOutput: default output dataframe
        :param indexNameList: list of name of indices
        :param column: select specific column
        :param missingIndex: missing index in dfInput
        :return dfOutput: filled output dataframe
        """
        indexNameList.remove(missingIndex)
        dfInput = dfInput.set_index(indexNameList)
        setLocation = dfInput.index.unique()
        for location in setLocation:
            if location in dfOutput.index.get_level_values(indexNameList[0]):
                values = dfInput[column].loc[location].tolist()
                if isinstance(values, int) or isinstance(values, float):
                    index = [0]
                else:
                    index = list(range(len(values)))
                dfOutput.loc[location, index] = values
        return dfOutput
