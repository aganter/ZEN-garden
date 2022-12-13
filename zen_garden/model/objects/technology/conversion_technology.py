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
import pyomo.environ as pe
import numpy as np
import pandas as pd
from .technology import Technology
from ..energy_system import EnergySystem
from ..component import Parameter,Variable,Constraint

class ConversionTechnology(Technology):
    # set label
    label           = "setConversionTechnologies"
    locationType    = "setNodes"
    # empty list of elements
    listOfElements = []

    def __init__(self, tech):
        """init conversion technology object
        :param tech: name of added technology"""

        logging.info(f'Initialize conversion technology {tech}')
        super().__init__(tech)
        # store input data
        self.storeInputData()
        # add ConversionTechnology to list
        ConversionTechnology.addElement(self)

    def storeInputData(self):
        """ retrieves and stores input data for element as attributes. Each Child class overwrites method to store different attributes """   
        # get attributes from class <Technology>
        super().storeInputData()
        # define input and output carrier
        self.inputCarrier               = self.dataInput.extractConversionCarriers()["inputCarrier"]
        self.outputCarrier              = self.dataInput.extractConversionCarriers()["outputCarrier"]
        EnergySystem.setTechnologyOfCarrier(self.name, self.inputCarrier + self.outputCarrier)
        # check if reference carrier in input and output carriers and set technology to correspondent carrier
        assert self.referenceCarrier[0] in (self.inputCarrier + self.outputCarrier), f"reference carrier {self.referenceCarrier} of technology {self.name} not in input and output carriers {self.inputCarrier + self.outputCarrier}"
        # get conversion efficiency and capex
        self.getConverEfficiency()
        self.getAnnualizedCapex()
        if EnergySystem.getAnalysis()["capacityLimitCountry"]:
            self.capacityLimitCountry = self.dataInput.extractInputData("capacityLimitCountry",indexSets=["setCountryNodes", "setTimeSteps"],timeSteps=EnergySystem.getEnergySystem().setTimeStepsYearly)

    def getConverEfficiency(self):
        """retrieves and stores converEfficiency for <ConversionTechnology>.
        Each Child class overwrites method to store different converEfficiency """
        #TODO read PWA Dict and set Params
        _PWAConverEfficiency,self.converEfficiencyIsPWA   = self.dataInput.extractPWAData("ConverEfficiency")
        if self.converEfficiencyIsPWA:
            self.PWAConverEfficiency    = _PWAConverEfficiency
        else:
            self.converEfficiencyLinear = _PWAConverEfficiency

    def getAnnualizedCapex(self):
        """ this method retrieves the total capex and converts it to annualized capex """
        _PWACapex,self.capexIsPWA = self.dataInput.extractPWAData("Capex")
        # annualize capex
        fractionalAnnuity   = self.calculateFractionalAnnuity()
        system              = EnergySystem.getSystem()
        _fractionOfYear     = system["unaggregatedTimeStepsPerYear"] / system["totalHoursPerYear"]
        if not self.capexIsPWA:
            self.capexSpecific = _PWACapex["capex"] * fractionalAnnuity + self.fixedOpexSpecific*_fractionOfYear
        else:
            self.PWACapex          = _PWACapex
            assert (self.fixedOpexSpecific==self.fixedOpexSpecific).all(), "PWACapex is only implemented for constant values of fixed Opex"
            self.PWACapex["capex"] = [(value * fractionalAnnuity + self.fixedOpexSpecific[0]*_fractionOfYear) for value in self.PWACapex["capex"]]
            # set bounds
            self.PWACapex["bounds"]["capex"] = tuple([(bound * fractionalAnnuity + self.fixedOpexSpecific[0]*_fractionOfYear) for bound in self.PWACapex["bounds"]["capex"]])
        # calculate capex of existing capacity
        self.capexExistingCapacity = self.calculateCapexOfExistingCapacities()

    def calculateCapexOfSingleCapacity(self,capacity,index):
        """ this method calculates the annualized capex of a single existing capacity. """
        if capacity == 0:
            return 0
        # linear
        if not self.capexIsPWA:
            capex   = self.capexSpecific[index[0]].iloc[0]*capacity
        else:
            capex   = np.interp(capacity,self.PWACapex["capacity"],self.PWACapex["capex"])
        return capex

    ### --- getter/setter classmethods
    @classmethod
    def getCapexConverEfficiencyOfAllElements(cls, variableType, selectPWA,indexNames = None):
        """ similar to Element.getAttributeOfAllElements but only for capex and converEfficiency.
        If selectPWA, extract PWA attributes, otherwise linear.
        :param variableType: either capex or converEfficiency
        :param selectPWA: boolean if get attributes for PWA
        :return dictOfAttributes: returns dict of attribute values """
        _classElements      = cls.getAllElements()
        dictOfAttributes    = {}
        if variableType == "capex":
            _isPWAAttribute         = "capexIsPWA"
            _attributeNamePWA       = "PWACapex"
            _attributeNameLinear    = "capexSpecific"
        elif variableType == "converEfficiency":
            _isPWAAttribute         = "converEfficiencyIsPWA"
            _attributeNamePWA       = "PWAConverEfficiency"
            _attributeNameLinear    = "converEfficiencyLinear"
        else:
            raise KeyError("Select either 'capex' or 'converEfficiency'")
        for _element in _classElements:
            # extract for PWA
            if getattr(_element,_isPWAAttribute) and selectPWA:
                dictOfAttributes,_ = cls.appendAttributeOfElementToDict(_element, _attributeNamePWA, dictOfAttributes)
            # extract for linear
            elif not getattr(_element,_isPWAAttribute) and not selectPWA:
                dictOfAttributes,_ = cls.appendAttributeOfElementToDict(_element, _attributeNameLinear, dictOfAttributes)
            if not dictOfAttributes:
                _, indexNames = cls.createCustomSet(indexNames)
                return (dictOfAttributes,indexNames)
        dictOfAttributes = pd.concat(dictOfAttributes,keys=dictOfAttributes.keys())
        if not indexNames:
            logging.warning(f"Initializing a parameter ({variableType}) without the specifying the index names will be deprecated!")
            return dictOfAttributes
        else:
            customSet,indexNames = cls.createCustomSet(indexNames)
            dictOfAttributes    = EnergySystem.checkForSubindex(dictOfAttributes, customSet)
            return (dictOfAttributes,indexNames)

    ### --- classmethods to construct sets, parameters, variables, and constraints, that correspond to ConversionTechnology --- ###
    @classmethod
    def constructSets(cls):
        """ constructs the pe.Sets of the class <ConversionTechnology> """
        model = EnergySystem.getConcreteModel()
        # get input carriers
        _inputCarriers      = cls.getAttributeOfAllElements("inputCarrier")
        _outputCarriers     = cls.getAttributeOfAllElements("outputCarrier")
        _referenceCarrier   = cls.getAttributeOfAllElements("referenceCarrier")
        _dependentCarriers  = {}
        for tech in _inputCarriers:
            _dependentCarriers[tech] = _inputCarriers[tech]+_outputCarriers[tech]
            _dependentCarriers[tech].remove(_referenceCarrier[tech][0])
        # input carriers of technology
        model.setInputCarriers = pe.Set(
            model.setConversionTechnologies,
            initialize = _inputCarriers,
            doc = "set of carriers that are an input to a specific conversion technology. Dimensions: setConversionTechnologies"
        )
        # output carriers of technology
        model.setOutputCarriers = pe.Set(
            model.setConversionTechnologies,
            initialize = _outputCarriers,
            doc = "set of carriers that are an output to a specific conversion technology. Dimensions: setConversionTechnologies"
        )
        # dependent carriers of technology
        model.setDependentCarriers = pe.Set(
            model.setConversionTechnologies,
            initialize = _dependentCarriers,
            doc = "set of carriers that are an output to a specific conversion technology. Dimensions: setConversionTechnologies"
        )

        # add pe.Sets of the child classes
        for subclass in cls.getAllSubclasses():
            if np.size(EnergySystem.getSystem()[subclass.label]):
                subclass.constructSets()

    @classmethod
    def constructParams(cls):
        """ constructs the pe.Params of the class <ConversionTechnology> """
        # slope of linearly modeled capex
        Parameter.addParameter(
            name="capexSpecificConversion",
            data= cls.getCapexConverEfficiencyOfAllElements("capex",False,indexNames=["setConversionTechnologies","setCapexLinear","setNodes","setTimeStepsYearly"]),
            doc = "Parameter which specifies the slope of the capex if approximated linearly"
        )
        # slope of linearly modeled conversion efficiencies
        Parameter.addParameter(
            name="converEfficiencySpecific",
            data= cls.getCapexConverEfficiencyOfAllElements("converEfficiency",False,indexNames=["setConversionTechnologies","setConverEfficiencyLinear","setNodes","setTimeStepsYearly"]),
            doc = "Parameter which specifies the slope of the conversion efficiency if approximated linearly"
        )
        # country specific capacity limit
        if EnergySystem.getAnalysis()["capacityLimitCountry"]:
            Parameter.addParameter(
                name= "capacityLimitCountry",
                data= EnergySystem.initializeComponent(cls,"capacityLimitCountry",indexNames=["setConversionTechnologies", "setCapacityTypes", "setCountryNodes", "setTimeStepsYearly"], capacityTypes = True),
                doc = "Parameter which specifies the country specific capacity limit of a conversion technology"
            )

    @classmethod
    def constructVars(cls):
        """ constructs the pe.Vars of the class <ConversionTechnology> """
        def carrierFlowBounds(model, tech, carrier, node, time):
            """ return bounds of carrierFlow for bigM expression
            :param model: pe.ConcreteModel
            :param tech: tech index
            :param carrier: carrier index
            :param node: node index
            :param time: time index
            :return bounds: bounds of carrierFlow"""
            params = Parameter.getComponentObject()
            if cls.getAttributeOfSpecificElement(tech,"converEfficiencyIsPWA"):
                bounds = cls.getAttributeOfSpecificElement(tech,"PWAConverEfficiency")["bounds"][carrier]
            else:
                # convert operationTimeStep to timeStepYear: operationTimeStep -> baseTimeStep -> timeStepYear
                timeStepYear = EnergySystem.convertTimeStepOperation2Invest(tech,time)
                if carrier == model.setReferenceCarriers[tech].at(1):
                    _converEfficiency = 1
                else:
                    _converEfficiency = params.converEfficiencySpecific[tech,carrier,node,timeStepYear]
                bounds = []
                for _bound in model.capacity[tech, "power", node, timeStepYear].bounds:
                    if _bound is not None:
                        bounds.append(_bound*_converEfficiency)
                    else:
                        bounds.append(None)
                bounds = tuple(bounds)
            return (bounds)

        model = EnergySystem.getConcreteModel()
        
        ## Flow variables
        # input flow of carrier into technology
        Variable.addVariable(
            model,
            name="inputFlow",
            indexSets= cls.createCustomSet(["setConversionTechnologies","setInputCarriers","setNodes","setTimeStepsOperation"]),
            domain = pe.NonNegativeReals,
            bounds = carrierFlowBounds,
            doc = 'Carrier input of conversion technologies')
        # output flow of carrier into technology
        Variable.addVariable(
            model,
            name="outputFlow",
            indexSets= cls.createCustomSet(["setConversionTechnologies","setOutputCarriers","setNodes","setTimeStepsOperation"]),
            domain = pe.NonNegativeReals,
            bounds = carrierFlowBounds,
            doc = 'Carrier output of conversion technologies')
        
        ## PWA Variables - Capex
        # PWA capacity
        Variable.addVariable(
            model,
            name="capacityApproximation",
            indexSets= cls.createCustomSet(["setConversionTechnologies","setNodes","setTimeStepsYearly"]),
            domain = pe.NonNegativeReals,
            doc = 'PWA variable for size of installed technology on edge i and time t')
        # PWA capex technology
        Variable.addVariable(
            model,
            name="capexApproximation",
            indexSets= cls.createCustomSet(["setConversionTechnologies","setNodes","setTimeStepsYearly"]),
            domain = pe.NonNegativeReals,
            doc = 'PWA variable for capex for installing technology on edge i and time t')

        ## PWA Variables - Conversion Efficiency
        # PWA reference flow of carrier into technology
        Variable.addVariable(
            model,
            name="referenceFlowApproximation",
            indexSets= cls.createCustomSet(["setConversionTechnologies","setDependentCarriers","setNodes","setTimeStepsOperation"]),
            domain = pe.NonNegativeReals,
            bounds = carrierFlowBounds,
            doc = 'PWA of flow of reference carrier of conversion technologies')
        # PWA dependent flow of carrier into technology
        Variable.addVariable(
            model,
            name="dependentFlowApproximation",
            indexSets= cls.createCustomSet(["setConversionTechnologies","setDependentCarriers","setNodes","setTimeStepsOperation"]),
            domain = pe.NonNegativeReals,
            bounds = carrierFlowBounds,
            doc = 'PWA of flow of dependent carriers of conversion technologies')

    @classmethod
    def constructConstraints(cls):
        """ constructs the pe.Constraints of the class <ConversionTechnology> """
        model = EnergySystem.getConcreteModel()
        # add PWA constraints
        # capex
        setPWACapex    = cls.createCustomSet(["setConversionTechnologies","setCapexPWA","setNodes","setTimeStepsYearly"])
        setLinearCapex = cls.createCustomSet(["setConversionTechnologies","setCapexLinear","setNodes","setTimeStepsYearly"])
        if setPWACapex: 
            # if setPWACapex contains technologies:
            PWABreakpoints,PWAValues = cls.calculatePWABreakpointsValues(setPWACapex[0],"Capex")
            model.constraintPWACapex = pe.Piecewise(setPWACapex[0],
                model.capexApproximation,model.capacityApproximation,
                pw_pts = PWABreakpoints,pw_constr_type = "EQ", f_rule = PWAValues,unbounded_domain_var = True, warn_domain_coverage =False,pw_repn="BIGM_BIN")
        if setLinearCapex[0]:
            # if setLinearCapex contains technologies:
            Constraint.addConstraint(
                model,
                name="constraintLinearCapex",
                indexSets= setLinearCapex,
                rule = constraintLinearCapexRule,
                doc = "Linear relationship in capex"
            )
        # Conversion Efficiency
        setPWAConverEfficiency      = cls.createCustomSet(["setConversionTechnologies","setConverEfficiencyPWA","setNodes","setTimeStepsOperation"])
        setLinearConverEfficiency   = cls.createCustomSet(["setConversionTechnologies","setConverEfficiencyLinear","setNodes","setTimeStepsOperation"])
        if setPWAConverEfficiency:
            # if setPWAConverEfficiency contains technologies:
            PWABreakpoints,PWAValues = cls.calculatePWABreakpointsValues(setPWAConverEfficiency[0],"ConverEfficiency")
            model.constraintPWAConverEfficiency = pe.Piecewise(setPWAConverEfficiency[0],
                model.dependentFlowApproximation,model.referenceFlowApproximation,
                pw_pts = PWABreakpoints,pw_constr_type = "EQ", f_rule = PWAValues,unbounded_domain_var = True, warn_domain_coverage =False,pw_repn="BIGM_BIN")
        if setLinearConverEfficiency[0]:
            # if setLinearConverEfficiency contains technologies:
            Constraint.addConstraint(
                model,
                name="constraintLinearConverEfficiency",
                indexSets= setLinearConverEfficiency,
                rule = constraintLinearConverEfficiencyRule,
                doc = "Linear relationship in ConverEfficiency"
            )    
        # Coupling constraints
        # couple the real variables with the auxiliary variables
        Constraint.addConstraint(
            model,
            name="constraintCapexCoupling",
            indexSets= cls.createCustomSet(["setConversionTechnologies","setNodes","setTimeStepsYearly"]),
            rule = constraintCapexCouplingRule,
            doc = "couples the real capex variables with the approximated variables")
        # capacity
        Constraint.addConstraint(
            model,
            name="constraintCapacityCoupling",
            indexSets= cls.createCustomSet(["setConversionTechnologies","setNodes","setTimeStepsYearly"]),
            rule = constraintCapacityCouplingRule,
            doc = "couples the real capacity variables with the approximated variables")
        
        # flow coupling constraints for technologies, which are not modeled with an on-off-behavior
        # reference flow coupling
        Constraint.addConstraint(
            model,
            name="constraintReferenceFlowCoupling",
            indexSets= cls.createCustomSet(["setConversionTechnologies","setNoOnOff","setDependentCarriers","setLocation","setTimeStepsOperation"]),
            rule = constraintReferenceFlowCouplingRule,
            doc = "couples the real reference flow variables with the approximated variables")
        # dependent flow coupling
        Constraint.addConstraint(
            model,
            name="constraintDependentFlowCoupling",
            indexSets= cls.createCustomSet(["setConversionTechnologies","setNoOnOff","setDependentCarriers","setLocation","setTimeStepsOperation"]),
            rule = constraintDependentFlowCouplingRule,
            doc = "couples the real dependent flow variables with the approximated variables")
        # country-specific capacity limit for conversion technologies
        if EnergySystem.getAnalysis()["capacityLimitCountry"]:
            Constraint.addConstraint(
                model,
                name="constraintCapacityLimitCountry",
                indexSets=cls.createCustomSet(["setConversionTechnologies", "setCapacityTypes", "setCountryNodes","setTimeStepsYearly"]),
                rule=constraintCapacityLimitCountryRule,
                doc="country specifc capacity limit for each conversion technology")


    # defines disjuncts if technology on/off
    @classmethod
    def disjunctOnTechnologyRule(cls,disjunct, tech,capacityType, node, time):
        """definition of disjunct constraints if technology is On"""
        model = disjunct.model()
        # get parameter object
        params = Parameter.getComponentObject()
        referenceCarrier = model.setReferenceCarriers[tech].at(1)
        if referenceCarrier in model.setInputCarriers[tech]:
            referenceFlow = model.inputFlow[tech,referenceCarrier,node,time]
        else:
            referenceFlow = model.outputFlow[tech,referenceCarrier,node,time]
        # get invest time step
        timeStepYear = EnergySystem.convertTimeStepOperation2Invest(tech,time)
        # disjunct constraints min load
        disjunct.constraintMinLoad = pe.Constraint(
            expr=referenceFlow >= params.minLoad[tech,capacityType,node,time] * model.capacity[tech,capacityType,node, timeStepYear]
        )
        # couple reference flows
        disjunct.constraintReferenceFlowCoupling = pe.Constraint(
            [tech],
            model.setDependentCarriers[tech],
            [node],
            [time],
            rule = constraintReferenceFlowCouplingRule,
            doc = "couples the real reference flow variables with the approximated variables. Dimension: tech, setDependentCarriers[tech], node, time.")
        # couple dependent flows
        disjunct.constraintDependentFlowCoupling = pe.Constraint(
            [tech],
            model.setDependentCarriers[tech],
            [node],
            [time],
            rule = constraintDependentFlowCouplingRule,
            doc = "couples the real dependent flow variables with the approximated variables. Dimension: tech, setDependentCarriers[tech], node, time.")

    @classmethod
    def disjunctOffTechnologyRule(cls,disjunct, tech, capacityType, node, time):
        """definition of disjunct constraints if technology is off"""
        model = disjunct.model()
        disjunct.constraintNoLoad = pe.Constraint(
            expr=
            sum(model.inputFlow[tech,inputCarrier,node,time]     for inputCarrier  in model.setInputCarriers[tech]) +
            sum(model.outputFlow[tech,outputCarrier,node,time]   for outputCarrier in model.setOutputCarriers[tech])
            == 0
        )
            
    @classmethod
    def calculatePWABreakpointsValues(cls,setPWA,typePWA):
        """ calculates the breakpoints and function values for piecewise affine constraint
        :param setPWA: set of variable indices in capex approximation, for which PWA is performed
        :param typePWA: variable, for which PWA is performed
        :return PWABreakpoints: dict of PWA breakpoint values
        :return PWAValues: dict of PWA function values """
        PWABreakpoints = {}
        PWAValues = {}

        # iterate through PWA variable's indices
        for index in setPWA:
            PWABreakpoints[index] = []
            PWAValues[index] = []
            if len(index) > 1:
                tech = index[0]
            else:
                tech = index
            # retrieve PWA variables
            PWAParameter = cls.getAttributeOfSpecificElement(tech,f"PWA{typePWA}")
            if typePWA == "Capex":
                PWABreakpoints[index] = PWAParameter["capacity"]
                PWAValues[index] = PWAParameter["capex"]
            elif typePWA == "ConverEfficiency":
                PWABreakpoints[index] = PWAParameter[cls.getAttributeOfAllElements("referenceCarrier")[tech][0]]
                PWAValues[index] = PWAParameter[index[1]]

        return PWABreakpoints,PWAValues

### --- functions with constraint rules --- ###
def constraintLinearCapexRule(model,tech,node,time):
    """ if capacity and capex have a linear relationship"""
    # get parameter object
    params = Parameter.getComponentObject()
    return(model.capexApproximation[tech,node,time] == params.capexSpecificConversion[tech,node,time]*model.capacityApproximation[tech,node,time])

def constraintLinearConverEfficiencyRule(model,tech,dependentCarrier,node,time):
    """ if reference carrier and dependent carrier have a linear relationship"""
    # get parameter object
    params = Parameter.getComponentObject()
    # get invest time step
    timeStepYear = EnergySystem.convertTimeStepOperation2Invest(tech,time)
    return(
        model.dependentFlowApproximation[tech,dependentCarrier,node,time] 
        == params.converEfficiencySpecific[tech,dependentCarrier, node,timeStepYear]*model.referenceFlowApproximation[tech,dependentCarrier,node,time]
    )

def constraintCapexCouplingRule(model,tech,node,time):
    """ couples capex variables based on modeling technique"""
    return(model.capex[tech,"power",node,time] == model.capexApproximation[tech,node,time])

def constraintCapacityCouplingRule(model,tech,node,time):
    """ couples capacity variables based on modeling technique"""
    return(model.builtCapacity[tech,"power",node,time] == model.capacityApproximation[tech,node,time])

def constraintReferenceFlowCouplingRule(disjunct,tech,dependentCarrier,node,time):
    """ couples reference flow variables based on modeling technique"""
    model = disjunct.model()
    referenceCarrier = model.setReferenceCarriers[tech].at(1)
    if referenceCarrier in model.setInputCarriers[tech]:
        return(model.inputFlow[tech,referenceCarrier,node,time] == model.referenceFlowApproximation[tech,dependentCarrier,node,time])
    else:
        return(model.outputFlow[tech,referenceCarrier,node,time] == model.referenceFlowApproximation[tech,dependentCarrier,node,time])

def constraintDependentFlowCouplingRule(disjunct,tech,dependentCarrier,node,time):
    """ couples output flow variables based on modeling technique"""
    model = disjunct.model()
    if dependentCarrier in model.setInputCarriers[tech]:
        return(model.inputFlow[tech,dependentCarrier,node,time] == model.dependentFlowApproximation[tech,dependentCarrier,node,time])
    else:
        return(model.outputFlow[tech,dependentCarrier,node,time] == model.dependentFlowApproximation[tech,dependentCarrier,node,time])

def constraintCapacityLimitCountryRule(model,tech,capacityType,country,time):
    """country specific capacity limit"""
    params = Parameter.getComponentObject()
    if params.capacityLimitCountry[tech,capacityType,country,time] == np.inf:
        return pe.Constraint.Skip
    else:
        return(sum(model.capacity[tech,capacityType, node, time] for node in model.setNodes if node[:2]==country)
               <= params.capacityLimitCountry[tech,capacityType,country,time])