"""===========================================================================================================================================================================
Title:          ZEN-GARDEN
Created:        March-2022
Authors:        Alissa Ganter (aganter@ethz.ch)
Organization:   Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:    Class defining the parameters, variables and constraints of the conditioning technologies.
                The class takes the abstract optimization model as an input, and adds parameters, variables and
                constraints of the conversion technologies.
==========================================================================================================================================================================="""
import logging
import pyomo.environ                                as pe
import pandas                                       as pd
import numpy                                        as np
from ..energy_system                    import EnergySystem
from .conversion_technology import ConversionTechnology

class ConditioningTechnology(ConversionTechnology):
    # set label
    label = "setConditioningTechnologies"
    # empty list of elements
    listOfElements = []

    def __init__(self, tech):
        """init conditioning technology object
        :param tech: name of added technology"""

        logging.info(f'Initialize conditioning technology {tech}')
        super().__init__(tech)
        # store input data
        self.storeInputData()
        # add ConversionTechnology to list
        ConditioningTechnology.addElement(self)

    def storeInputData(self):
        """ retrieves and stores input data for element as attributes. Each Child class overwrites method to store different attributes """
        # get attributes from class <Technology>
        super().storeInputData()
        self.setConditioningCarriers()

    def setConditioningCarriers(self):
        """add conditioning carriers to system"""
        subset   = "setConditioningCarriers"
        analysis = EnergySystem.getAnalysis()
        system   = EnergySystem.getSystem()
        # add setConditioningCarriers to analysis and indexingSets
        if subset not in analysis["subsets"]["setCarriers"]:
            analysis["subsets"]["setCarriers"].append(subset)
        # add setConditioningCarriers to system
        if subset not in system.keys():
            system[subset] = []
        if self.outputCarrier[0] not in system[subset]:
            system[subset] += self.outputCarrier

    def getConverEfficiency(self):
        """retrieves and stores converEfficiency for <ConditioningTechnology>.
        Create dictionary with input parameters with the same format as PWAConverEfficiency"""
        #self.dataInput.extractAttributeData("minBuiltCapacity")["value"]
        self.specificHeat         = self.dataInput.extractAttributeData("specificHeat")["value"]
        self.specificHeatRatio    = self.dataInput.extractAttributeData("specificHeatRatio")["value"]
        self.pressureIn           = self.dataInput.extractAttributeData("pressureIn")["value"]
        self.pressureOut          = self.dataInput.extractAttributeData("pressureOut")["value"]
        self.temperatureIn        = self.dataInput.extractAttributeData("temperatureIn")["value"]
        self.isentropicEfficiency = self.dataInput.extractAttributeData("isentropicEfficiency")["value"]

        # calculate energy consumption
        _pressureRatio     = self.pressureOut / self.pressureIn
        _exponent          = (self.specificHeatRatio - 1) / self.specificHeatRatio
        if self.dataInput.ifAttributeExists("lowerHeatingValue", column=None):
            _lowerHeatingValue = self.dataInput.extractAttributeData("lowerHeatingValue")["value"]
            self.specificHeat  = self.specificHeat / _lowerHeatingValue
        _energyConsumption = self.specificHeat * self.temperatureIn / self.isentropicEfficiency \
                            * (_pressureRatio ** _exponent - 1)

        # check input and output carriers
        _inputCarriers = self.inputCarrier.copy()
        if self.referenceCarrier[0] in _inputCarriers:
            _inputCarriers.remove(self.referenceCarrier[0])
        assert len(_inputCarriers) == 1, f"{self.name} can only have 1 input carrier besides the reference carrier."
        assert len(self.outputCarrier) == 1, f"{self.name} can only have 1 output carrier."
        # create dictionary
        self.converEfficiencyIsPWA                            = False
        self.converEfficiencyLinear                           = dict()
        self.converEfficiencyLinear[self.outputCarrier[0]]    = self.dataInput.createDefaultOutput(indexSets=["setNodes","setTimeSteps"],
                                                                                                   column=None,
                                                                                                   timeSteps=self.setTimeStepsInvest,
                                                                                                   manualDefaultValue = 1)[0] # TODO losses are not yet accounted for
        self.converEfficiencyLinear[_inputCarriers[0]]        = self.dataInput.createDefaultOutput(indexSets=["setNodes", "setTimeSteps"],
                                                                                                   column=None,
                                                                                                   timeSteps=self.setTimeStepsInvest,
                                                                                                   manualDefaultValue=_energyConsumption)[0]
        # dict to dataframe
        self.converEfficiencyLinear              = pd.DataFrame.from_dict(self.converEfficiencyLinear)
        self.converEfficiencyLinear.columns.name = "carrier"
        self.converEfficiencyLinear              = self.converEfficiencyLinear.stack()
        _converEfficiencyLevels                  = [self.converEfficiencyLinear.index.names[-1]] + self.converEfficiencyLinear.index.names[:-1]
        self.converEfficiencyLinear              = self.converEfficiencyLinear.reorder_levels(_converEfficiencyLevels)

    @classmethod
    def constructSets(cls):
        """ constructs the pe.Sets of the class <ConditioningTechnology> """
        model = EnergySystem.getConcreteModel()
        # get parent carriers
        _outputCarriers    = cls.getAttributeOfAllElements("outputCarrier")
        _referenceCarriers = cls.getAttributeOfAllElements("referenceCarrier")
        _parentCarriers    = list()
        _childCarriers     = dict()
        for tech, carrierRef in _referenceCarriers.items():
            if carrierRef[0] not in _parentCarriers:
                _parentCarriers += carrierRef
                _childCarriers[carrierRef[0]] = list()
            if _outputCarriers[tech] not in _childCarriers[carrierRef[0]]:
                _childCarriers[carrierRef[0]] +=_outputCarriers[tech]
                _conditioningCarriers = list()
        _conditioningCarriers = _parentCarriers+[carrier[0] for carrier in _childCarriers.values()]

        # update indexing sets
        EnergySystem.indexingSets.append("setConditioningCarriers")
        EnergySystem.indexingSets.append("setConditioningCarrierParents")

        # set of conditioning carriers
        model.setConditioningCarriers = pe.Set(
            initialize=_conditioningCarriers,
            doc="set of conditioning carriers. Dimensions: setConditioningCarriers"
        )
        # set of parent carriers
        model.setConditioningCarrierParents = pe.Set(
            initialize=_parentCarriers,
            doc="set of parent carriers of conditioning. Dimensions: setConditioningCarrierParents"
        )
        # set that maps parent and child carriers
        model.setConditioningCarrierChildren = pe.Set(
            model.setConditioningCarrierParents,
            initialize=_childCarriers,
            doc="set of child carriers associated with parent carrier used in conditioning. Dimensions: setConditioningCarrierChildren"
        )
