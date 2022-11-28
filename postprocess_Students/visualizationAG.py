"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich
Description:  Class is defining the postprocessing of the results.
              The class takes as inputs the optimization problem (model) and the system configurations (system).
              The class contains methods to read the results and save them in a result dictionary (resultDict).
==========================================================================================================================================================================="""
import os
import pickle
import copy
import numpy as np
import pandas as pd

import matplotlib.pyplot    as plt


class VisualizeResults:

    def __init__(self, dataset, scenario = "", pltShow = True):
        """postprocessing of the results of the optimization
        :param model:     optimization model
        :param pyoDict:   input data dictionary
        :param modelName: model name used for the directory to save the results in"""
        # set modelName
        self.dataset  = dataset
        self.scenario = scenario
        self.setModelName()
        self.nameDir  = f"outputs/{self.name}"
        # plot settings
        self.pltShow  = pltShow
        # init directories
        self.initDirectory("plots")
        self.initDirectory("files")
        # load results
        self.paramDict = self.loadResults("paramDict")
        self.varDict   = self.loadResults("varDict")
        self.analysis  = self.loadResults("Analysis")
        self.system    = self.loadResults("System")
        # get sets and set time-step duration and unitDict
        self.getSets()
        self.setTimeStepsDuration()
        self.setUnitDict()

    ## general methods
    def initDirectory(self,folder):
        """init directories to store plots and files"""
        if not os.path.exists(f"{self.nameDir}/{folder}"):
            os.makedirs(f"{self.nameDir}/{folder}")

    def setModelName(self):
        """set model name"""
        if self.scenario == str():
            self.name = self.dataset
        else:
            self.name = self.dataset + "_" + self.scenario

    def setTimeStepsDuration(self):
        """ set timesteps duration"""
        self.timeStepsCarrierDuration   = self.getDataframe("timeStepsOperationDuration", indexNames=["technology", "time"], type="param", subset=self.setCarriers)
        self.timeStepsOperationDuration = self.getDataframe("timeStepsOperationDuration", indexNames=["technology", "time"], type="param", subset=self.system["setTechnologies"])
        self.timeStepsInvestDuration    = self.getDataframe("timeStepsInvestDuration", indexNames=["technology", "year"], type="param")

    def setUnitDict(self):
        """set unit dictionary for unit conversion"""
        self.unitDict = {
            "energy": ["existingCapacity","builtCapacity", "capacity",],
            "power":  ["demandCarrier",
                       "inputFlow", "outputFlow", "carrierFlow",
                       "importCarrierFlow", "exportCarrierFlow"],
            #"length":  ["distance"],
            "mass":    ["carbonEmissions", "carbonEmissionsTotal","carbonBudget"
                        "carbon", "wet_biomass"],
            "cost":    ["importPriceCarrier","exportPriceCarrier",
                        "capex", "capexYearly",
                        "opex"]
        }

    def loadResults(self, name, nameDir = None):
        """ load results from results folder"""
        if not nameDir:
            nameDir = self.nameDir
        with open(f"{nameDir}/{name}.pickle", "rb") as file:
            output = pickle.load(file)
        return output

    def getDataframe(self, name, indexNames, type = "var", subset = None, dct = {}):
        """plot built capacity"""
        # check whether parameter or variable is extracted
        if dct != {}:
            dct = dct[name]
        elif type == "param":
            dct = self.paramDict[name]
        else:
            dct = self.varDict[name]
        # get dct and values
        keys, values   = zip(*dct.items())
        if  len(indexNames)>1: # check if index is mulitiindex
            idx            = pd.MultiIndex.from_tuples(keys)
        else:
            idx            = list(keys)
        # create series
        df             = pd.Series(values, index=idx)
        # set index names and column name
        df.index.names = indexNames
        df.name        = name
        # round values
        try:
            df = df.round(decimals=4)
        except:
            pass
        # select a subset
        if subset:
            df = df.loc[subset]
        return df

    def updateTimeIndex(self, df, index="time"):
        """update the time index from numeric values to actual timestamps:
        index: indicate name of time index that is updated"""
        baseYear  = 2020
        yearsDict = {}
        for year in df.index.unique(index):
            yearsDict[year] = baseYear+year
        df = df.rename(index=yearsDict)
        return df

    def getSets(self):
        """ get sets from system"""
        # carriers
        self.setCarriers = copy.deepcopy(self.system["setCarriers"])
        # conditioning technologies
        self.setConditioningTechnologies = copy.deepcopy(self.system["setConditioningTechnologies"])
        # conversion technologies
        self.setConversionTechnologies   = copy.deepcopy(self.system["setHydrogenConversionTechnologies"])
        self.setConversionTechnologies   = list(set(self.setConversionTechnologies) - set(self.setConditioningTechnologies))
        # electricity generation Technologies
        self.setElectricityGenerationTechnologies = copy.deepcopy(self.system["setElectricityGenerationTechnologies"])
        # transport technologies
        self.setTransportTechnologies = copy.deepcopy(self.system["setTransportTechnologies"])
        # storage technologies
        self.setStorageTechnologies = copy.deepcopy(self.system["setStorageTechnologies"])
        self.setStorageTechnologies.append("carbon_storage")

    def getUnitConversion(self, name, carrier=None, returnName = True):
        """get unit conversion factor"""
        if name in self.unitDict["energy"]:
            conver = 1e-6 # MW to TW
            type   = "MW"
        elif name in self.unitDict["power"]:
            conver = 1e-6  # MW to TW
            type   = "MW"
        elif name in self.unitDict["mass"]:
            conver = 1e-6  # t to Mt
            type   = "MW"
        elif name in self.unitDict["cost"]:
            conver = 1e-3  # kiloEuro to MEuro
            type   = "MW"
        else:
            raise ValueError(f"Unit for {name} not specified in unitDict")

        if carrier in self.unitDict["mass"]:
            conver = 1e-6  # t to Mt
            type   = "Mt"

        if returnName:
            return conver, type
        else:
            return conver

    def barplot(self, title, df, stacked = False, ylabel=None, xlabel=None):
        """ stacked barplot"""
        if df.empty:
            print(f"{title} is empty.")
        elif df[df>0].isna().all().all():
            print(f"{title} all values are 0")
        else:
            fig, axs = plt.subplots()
            df.plot.bar(ax=axs, stacked=stacked)
            axs.set_title(title)
            axs.set_xlabel(xlabel)
            axs.set_ylabel(ylabel)
            fig.savefig(f"{self.nameDir}/plots/{title}.png")
            if self.pltShow:
                fig.show()
            plt.close(fig)

    def areaplot(self, name, df, ylabel=None, xlabel=None):
        """ area plot of dataframe"""
        if df.empty:
            print(f"{name} is empty.")
        elif df[df>0].isna().all().all():
            print(f"{name} all values are 0")
        else:
            fig, axs = plt.subplots()
            axs.set_title(name)
            df.plot.area(ax=axs)
            axs.set_xlabel(xlabel)
            axs.set_ylabel(ylabel)
            fig.savefig(f"{self.nameDir}/plots/{name}.png")
            if self.pltShow:
                fig.show()
            plt.close(fig)

    ## plot results
    def evaluateHydrogenDemand(self):
        """plot hydrogen demand"""
        conver, name = self.getUnitConversion("demandCarrier")

        demand = self.getDataframe("demandCarrier",["carrier", "node", "time"], type="param")

        demand = demand.unstack("node")
        demand = demand.apply(lambda row: row*self.timeStepsCarrierDuration)
        demand = demand.apply(lambda row: row*conver) #conversion from GWh in TWh
        # total hydrogen demand per country
        demandNodes = demand.loc["hydrogen"].sum()
        demandNodes = demandNodes[demandNodes >= 0.1 * demandNodes.max()]
        self.barplot("totalHydrogenDemandPerCountry", demandNodes, ylabel = f"Hydrogen Demand in {name}" , xlabel= "years")
        # total hydrogen demand per country
        demandEvolution = demand.loc["hydrogen"].stack().groupby("time").sum()
        demandEvolution = demandEvolution[demandEvolution >= 0.1 * demandEvolution.max()]
        demandEvolution = self.updateTimeIndex(demandEvolution, index="time")
        self.barplot("totalHydrogenDemandTime", demandEvolution, ylabel = f"Hydrogen Demand in {name}", xlabel = "years")

    def evaluateBuiltCapacity(self):
        """plot built capacity"""
        conver, name  = self.getUnitConversion("builtCapacity")
        ylabel        = f"built capacity [{name}]"
        builtCapacity = self.getDataframe("builtCapacity",["technology", "capacityType", "location", "time"]) * conver

        # conversion technologies
        totalBuiltCapacity = builtCapacity[self.setConversionTechnologies].groupby(level=["technology","time"]).sum()
        self.barplot("totalBuiltCapacityConversion", totalBuiltCapacity.unstack("technology"), stacked=True)

        # electricity generation technologies
        totalBuiltCapacity = builtCapacity.loc[self.setElectricityGenerationTechnologies].groupby(level=["technology", "time"]).sum()
        self.barplot("totalBuiltCapacityElectricity", totalBuiltCapacity.unstack("technology"), stacked=True, ylabel=ylabel)

        # conditioning technologies
        totalBuiltCapacity = builtCapacity.loc[self.setConditioningTechnologies].groupby(level=["technology","time"]).sum()
        self.barplot("totalBuiltCapacityConditioning", totalBuiltCapacity.unstack("technology"), stacked=True, ylabel=ylabel)

        # transport technologies
        totalBuiltCapacity = builtCapacity.loc[self.setTransportTechnologies].groupby(level=["technology","time"]).sum()
        self.barplot("totalBuiltCapacityTransport", totalBuiltCapacity.unstack("technology"), stacked=True, ylabel=ylabel)

        # storage technologies
        conver2, name2     = self.getUnitConversion("builtCapacity", "carbon")
        totalBuiltCapacity = builtCapacity.loc[self.setStorageTechnologies].groupby(level=["technology","time"]).sum() / conver * conver2
        self.barplot("totalBuiltCapacityStorage", totalBuiltCapacity.unstack("technology"), stacked=True, ylabel=f"built capacity [{name2}]")

    def evaluateCapacity(self):
        """plot installed capacity"""
        conver, name = self.getUnitConversion("capacity")
        ylabel       = f"total capacity [{name}]"
        capacity     = self.getDataframe("capacity", ["technology", "capacityType", "location", "time"]).round(decimals=4) * conver

        # conversion technologies
        totalCapacity = capacity.loc[self.setConversionTechnologies].groupby(level=["technology","time"]).sum()
        self.barplot("totalCapacityConversion", totalCapacity.unstack("technology"), stacked=True, ylabel=ylabel)
        totalCapacity = capacity.loc[self.setConversionTechnologies].groupby(level=["technology", "location"]).sum()
        totalCapacity = totalCapacity[totalCapacity >= 0.1 * totalCapacity.max()]
        self.barplot("totalCapacityConversionNodes", totalCapacity.unstack("technology"), stacked=True, ylabel=ylabel)

        # electricity generation technologies
        totalCapacity = capacity.loc[self.setElectricityGenerationTechnologies].groupby(level=["technology", "time"]).sum()
        self.barplot("totalCapacityElectricity", totalCapacity.unstack("technology"), stacked=True, ylabel=ylabel)
        totalCapacity = capacity.loc[self.setElectricityGenerationTechnologies].groupby(level=["technology", "location"]).sum()
        totalCapacity = totalCapacity[totalCapacity>= 0.1*totalCapacity.max()]
        self.barplot("totalCapacityElectricityNodes", totalCapacity.unstack("technology"), stacked=True, ylabel=ylabel)

        # transport technologies
        totalCapacity = capacity.loc[self.setTransportTechnologies].groupby(level=["technology","time"]).sum()
        self.barplot("totalCapacityTransport", totalCapacity.unstack("technology"), stacked=True, ylabel=ylabel)

        # storage technologies
        conver2, name2 = self.getUnitConversion("capacity", carrier="carbon")
        totalCapacity  = capacity.loc[self.setStorageTechnologies].groupby(level=["technology","time"]).sum() / conver * conver2
        self.barplot("totalCapacityStorage", totalCapacity.unstack("technology"), stacked=True, ylabel=f"total capacity [{name2}]")

    def evaluateCarrierImports(self):
        """plot carrier imports"""
        carrierImports = self.getDataframe("importCarrierFlow", ["carrier", "location", "time"])
        carrierImports = carrierImports.reorder_levels(["carrier", "time", "location"]).unstack()
        carrierImports = carrierImports.apply(lambda row: row*self.timeStepsCarrierDuration)
        carrierImports = carrierImports.reorder_levels(["carrier", "time", "technology"])
        self.areaplot(f"carrierImports", carrierImports.stack().groupby(["carrier","time"]).sum().unstack("carrier"))
        # electricity and natural gas imports
        for carrier in carrierImports.index.unique("carrier"):
            conver, name = self.getUnitConversion("importCarrierFlow", carrier=carrier)
            imports = carrierImports.loc[carrier].sum() * conver
            imports = imports[imports >= 0.1 * imports.max()]
            self.barplot(f"{carrier}Imports", imports, stacked=False, ylabel =  f"{carrier} import [{name}]")

    def checkCarrierExports(self):
        """check if carrier exports are zero"""
        carrierExports = self.getDataframe("exportCarrierFlow", ["carrier", "location", "time"])
        carrierExports = carrierExports.reorder_levels(["carrier", "time", "location"]).unstack()
        carrierExports = carrierExports.apply(lambda row: row * self.timeStepsCarrierDuration)
        if carrierExports.sum(axis=0).sum().round(2) != 0:
            print("Carrier exports are not 0.")

    def evaluateInputFlow(self):
        """plot carrier flow"""
        inputFlow    = self.getDataframe("inputFlow", ["technology","carrier","location","time"])
        inputFlow    = inputFlow * self.timeStepsCarrierDuration
        inputFlow    = inputFlow.reorder_levels(["technology", "carrier", "location", "time"])

        # inputFlows Conversion
        inputFlowConversion = inputFlow.loc[self.setConversionTechnologies]
        inputFlowConversion = inputFlowConversion.unstack("carrier")
        inputFlowConversion = self.updateTimeIndex(inputFlowConversion, index="time")
        conver, name        = self.getUnitConversion("inputFlow")
        self.areaplot(f"inputFlowsConversion", inputFlowConversion.groupby("time").sum()*conver, ylabel = "Input flow [{name}]")
        for carrier in inputFlowConversion.columns:
            conver, name = self.getUnitConversion("inputFlow", carrier=carrier)
            tmp          = inputFlowConversion[carrier].groupby(["technology","time"]).sum() * conver
            self.barplot(f"{carrier}InputFlowsConversion", tmp.unstack("technology"), stacked=True, ylabel=f"input flow [{name}]")

        # inputFlows Conditioning
        inputFlowCodnitioning = inputFlow.loc[self.setConditioningTechnologies].unstack("carrier")
        for carrier in inputFlowCodnitioning.columns:
            conver, name = self.getUnitConversion("inputFlow", carrier=carrier)
            tmp          = inputFlowCodnitioning[carrier].groupby(["technology", "time"]).sum()*conver
            self.barplot(f"{carrier}InputFlowsConditioning", tmp.unstack("technology"), stacked=True, ylabel=f"input flow [{name}]")

        # inputFlows Storage
        inputFlowStorage = inputFlow.loc[self.setStorageTechnologies].unstack("carrier")
        for carrier in inputFlowStorage.columns:
            conver, name = self.getUnitConversion("inputFlow", carrier=carrier)
            tmp          = inputFlowStorage[carrier].groupby(["technology", "time"]).sum() * conver
            self.barplot(f"{carrier}InputFlowsStorage", tmp.unstack("technology"), stacked=True, ylabel=f"input flow [{name}]")

    def evaluateOutputFlow(self):
        """evaluation output flows"""
        outputFlow = self.getDataframe("outputFlow", ["technology", "carrier", "location", "time"])
        tsDuration = self.timeStepsOperationDuration.loc[self.setConversionTechnologies]
        # inputFlows Conversion
        outputFlow = outputFlow.loc[self.setConversionTechnologies]
        outputFlow = outputFlow.reorder_levels(["carrier", "technology", "location", "time"])
        for carrier in outputFlow.index.unique("carrier"):
            conver, name = self.getUnitConversion("outputFlow", carrier=carrier)
            output       = outputFlow.loc[carrier].groupby(["technology", "time"]).sum() * tsDuration * conver
            self.barplot(f"{carrier}OutputFlowsConversion", output.unstack("technology"), stacked=True, ylabel=f"output flow [{name}]")

    def evaluateCarrierFlowsTransport(self):
        """evaluate carrier flows transport techs"""
        carrierFlow = self.getDataframe("carrierFlow", ["technology", "location", "time"])
        for carrier in ["hydrogen", "carbon", "electricity","dry_biomass"]:
            conver, name = self.getUnitConversion("carrierFlow", carrier=carrier)
            tmp          = [tech for tech in carrierFlow.index.unique("technology") if carrier in tech]
            tmp          = carrierFlow.loc[tmp].groupby(["technology", "time"]).sum()
            tmp          = tmp * self.timeStepsOperationDuration.loc[tmp.index.unique("technology")] * conver
            self.barplot(f"{carrier}FlowsTransport", tmp.unstack("technology"), stacked=True, ylabel=f"{carrier} transport [{name}]")

    def evaluateCarbonEmissions(self):
        """plot carbon emissions"""

        carbonEmissions = self.getDataframe("carbonEmissionsTotal", ["year"])
        conver, name    = self.getUnitConversion("carbonEmissionsTotal", carrier="carbon")
        self.barplot("carbonEmissionsYearly", carbonEmissions*conver,ylabel=f"Carbon emissions [{name}]")


## method to run visualization
def runVisualization(dataset, scenario="", pltShow=False):
    """visualize and evaluate results
    :param scenario: specify scenario name
    :param pltShow: if true, display plots before saving"""
    visResults = VisualizeResults(dataset, scenario, pltShow=pltShow)
    ## params
    visResults.evaluateHydrogenDemand()
    ## vars
    # installed capacities
    visResults.evaluateBuiltCapacity()
    # visResults.evaluateCapacity()
    visResults.evaluateCarbonEmissions()
    # carrier flows
    visResults.evaluateCarrierImports()
    # visResults.checkCarrierExports()
    visResults.evaluateInputFlow()
    # visResults.evaluateOutputFlow()
    visResults.evaluateCarrierFlowsTransport()
    ## compute levelized cost
    visResults.computeLevelizedCost("electricity")
    visResults.computeLevelizedCost("hydrogen")
    ## compute invested capacities
    # visResults.determineExistingCapacities()


if __name__ == "__main__":
    os.chdir("..") # make sure you are in the correct directory so dataset is found
    dataset = "HSC_NUTS2" # dataset name
    pltShow = False  # True or False

    runVisualization(dataset, pltShow=pltShow)
