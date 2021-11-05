"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Labratory of Risk and Reliability Engineering, ETH Zurich

Description:  Class defining the abstract optimization model.
              The class takes as inputs the properties of the optimization problem. The properties are saved in the
              dictionaries analysis and system which are passed to the class. After initializing the abstract model, the
              class adds carriers and technologies to the abstract model and returns the abstract optimization model.
              The class also includes a method to solve the optimization problem.
==========================================================================================================================================================================="""
import logging
import pyomo.environ as pe
from pyomo.opt import SolverStatus, TerminationCondition
from objects.carrier import *
from objects.technology import *

class Model:

    def __init__(self, analysis, system):
        """
        create Pyomo Abstract Model
        :param analysis: dictionary defining the analysis framework
        :param system: dictionary defining the system
        """

        self.analysis = analysis  # analysis structure
        self.solver = analysis.solver  # solver structure
        self.constraints = dict() # dictionary containing the constraints

        self.model = pe.AbstractModel()
        self.addSets(self)
        self.addElements()

    def addSets(self):
        """
        This method sets up the sets of the optimization problem.
        Some sets are initialized with default values, if the value is not specified by the input data
        """

        ## FYI
        #(1) model.ct = Set(model.t)  # model.ct is “dictionary” of sets, i.e., model.ct[i] = Set() for all i in model.t
        #(2) model.ct = Set(within=model.t)  # model.ct is a subset of model.t, Pyomo will do the verification of this
        #(3) model.i = Set(initialize=model.t)  # makes a copy of whatever is in model.t during the time of construction
        #(4) model.i = SetOf(model.t)  # references whatever is in model.t at runtime (alias)
        # 'setTransport': 'Set of all transport technologies. Subset: setTechnologies'
        # 'setProduction': 'Set of all production technologies. Subset: setTechnologies'

        sets = {'setCarriers':      'Set of carriers',     # Subsets: setInputCarriers, setOutputCarriers
                'setTechnologies':  'Set of technologies', # Subsets: setTransportTechnologies, setProductionTechnologies
                'setTimeSteps':     'Set of time-steps',
                'setNodes':         'Set of nodes'}

        for set, setProperties in sets.items():
            peSet = pe.Set(doc=setProperties)
            setattr(self.model, set, peSet)

    def addElements(self, analysis):
        """
        This method sets up the parameters, variables and constraints of the carriers of the optimization problem.
        :param analysis: dictionary defining the analysis framework
        :param system: dictionary defining the system
        """
        # TODO create list of carrier types, only add relevant types
        # TODO to create list of carrier tpyes (e.g. general, CO2,...) write a function getCarrierTypes

        Carrier(self.model)

        if 'production' in analysis['technologies']:
            ProductionTechnology(self.model)
        if 'transport' in analysis['technologies']:
            ProductionTechnology(self.model)
        if 'storage' in analysis['technologies']:
            ProductionTechnology(self.model)

        for carrier in list_carriers:
            c = Carrier()
            c.set(data)
            c.getCarrieravailability()

    def addTechnologies(self):
        """
        This method sets up the parameters, variables and constraints of the technologies of the optimization problem.
        """

        # distinguish production, storage, and transport technologies

        technology = Technology(self.model)
        technology.addTechnologySets()
        technology.addTechnologyParams()
        technology.addTechnologyVars()
        technology.addTechnologyConstr()

    def addObjecctive(self):
        """
        :return:
        """

        # TODO figure out a way to formulate objective function using pyomo?

    def addConstr(self, listConstraints):
        """
        :param modelConstraints:
        :return:
        """

    # TODO add Constr from csv/txt files
    for constr in listConstraints:
        name = constr['name']
        forEach = constr['forEach']
        rule = constr['rule']

        exec
        '@staticmethod def {0}({1}): return {2}'.format(name, forEach, rule)


    def solve(self):
        """
        :return:
        """

        options = solveroptions(self)

        # TODO specify which objective function should be used depending on the settings

    def save(self, diagnostic, sepc):
        """
        :param diagnostic:
        :param sepc:
        :return:
        """

        #TODO save results

