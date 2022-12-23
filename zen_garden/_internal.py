"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
              Davide Tonelli (davidetonelli@outlook.com)
              Jacob Mannhardt (jmannhardt@ethz.ch)
Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:  Compilation  of the optimization problem.
==========================================================================================================================================================================="""
import os
import sys
import logging
import importlib.util
import pkg_resources

from shutil import rmtree

from   .preprocess.prepare             import Prepare
from   .model.optimization_setup       import OptimizationSetup
from   .postprocess.postprocess        import Postprocess


def main(config, dataset_path=None):
    """
    This function runs the compile.py script that was used in ZEN-Garden prior to the package build, it is executed
    in the __main__.py script
    :param config: A config instance used for the run
    :param dataset_path: If not None, used to overwrite the config.analysis["dataset"]
    """
    # SETUP LOGGER
    log_format = '%(asctime)s %(filename)s: %(message)s'
    log_path = os.path.join('outputs', 'logs')
    os.makedirs(log_path, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_path, 'valueChain.log'), level=logging.INFO,
                        format=log_format, datefmt='%Y-%m-%d %H:%M:%S')
    logging.captureWarnings(True)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)

    # print the version
    version = pkg_resources.require("zen_garden")[0].version
    logging.info(f"Running ZEN-Garden version: {version}")

    # prevent double printing
    logging.propagate = False

    # overwrite the path if necessary
    if dataset_path is not None:
        logging.info(f"Overwriting dataset to: {dataset_path}")
        config.analysis["dataset"] = dataset_path
    # get the abs path to avoid working dir stuff
    config.analysis["dataset"] = os.path.abspath(config.analysis['dataset'])
    config.analysis["folderOutput"] = os.path.abspath(config.analysis['folderOutput'])

    ### System - load system configurations
    system_path = os.path.join(config.analysis['dataset'], "system.py")
    if not os.path.exists(system_path):
        raise FileNotFoundError(f"system.py not found in dataset: {config.analysis['dataset']}")
    spec    = importlib.util.spec_from_file_location("module", system_path)
    module  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    system  = module.system
    config.system.update(system)

    ### overwrite default system and scenario dictionaries
    if config.system["conductScenarioAnalysis"]:
        scenarios_path = os.path.abspath(os.path.join(config.analysis['dataset'], "scenarios.py"))
        if not os.path.exists(scenarios_path):
            raise FileNotFoundError(f"scenarios.py not found in dataset: {config.analysis['dataset']}")
        spec        = importlib.util.spec_from_file_location("module", scenarios_path)
        module      = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        scenarios   = module.scenarios
        # config.scenarios.update(scenarios)
        config.scenarios = scenarios

    # create a dictionary with the paths to access the model inputs and check if input data exists
    prepare = Prepare(config)
    # check if all data inputs exist and remove non-existent
    prepare.checkExistingInputData()

    # FORMULATE THE OPTIMIZATION PROBLEM
    # add the elements and read input data
    optimizationSetup           = OptimizationSetup(config.analysis, prepare)
    # get rolling horizon years
    stepsOptimizationHorizon    = optimizationSetup.getOptimizationHorizon()

    # get the name of the dataset
    modelName = os.path.basename(config.analysis["dataset"])
    if os.path.exists(out_folder := os.path.join(config.analysis["folderOutput"], modelName)):
        logging.warning(f"The output folder '{out_folder}' already exists")
        if config.analysis["overwriteOutput"]:
            logging.warning("Existing files will be overwritten!")

    # determine base scenarios
    if "useBaseScenarios" in config.system.keys() and config.system["useBaseScenarios"]:
        baseScenarios = dict()
        for scenario in config.scenarios.keys():
            baseScenarios[scenario] = {tech: ["existingCapacity"] for tech in config.system["setTechnologies"]}
    else:
        baseScenarios = {"": {}}

    # update base scenario
    for baseScenario, elements in baseScenarios.items():
        optimizationSetup.setBaseConfiguration(baseScenario, elements)

        # update input data
        for scenario, elements in config.scenarios.items():
            if scenario != baseScenario or baseScenario == "":
                optimizationSetup.restoreBaseConfiguration(scenario, elements)  # per default scenario="" is used as base configuration. Use setBaseConfiguration(scenario, elements) if you want to change that
                optimizationSetup.overwriteParams(scenario, elements)
                # iterate through horizon steps
                for stepHorizon in stepsOptimizationHorizon:
                    if len(stepsOptimizationHorizon) == 1:
                        logging.info(f"\n--- Conduct optimization for perfect foresight {baseScenario} {scenario} --- \n")
                    else:
                        logging.info(f"\n--- Conduct optimization for rolling horizon step {stepHorizon} of {max(stepsOptimizationHorizon)}--- \n")
                    # overwrite time indices
                    optimizationSetup.overwriteTimeIndices(stepHorizon)
                    # create optimization problem
                    optimizationSetup.constructOptimizationProblem()
                    # SOLVE THE OPTIMIZATION PROBLEM
                    optimizationSetup.solve(config.solver)
                    # add newly builtCapacity of first year to existing capacity
                    optimizationSetup.addNewlyBuiltCapacity(stepHorizon)
                    # add cumulative carbon emissions to previous carbon emissions
                    optimizationSetup.addCarbonEmissionsCumulative(stepHorizon)
                    # EVALUATE RESULTS
                    subfolder = ""
                    scenario_name=None
                    if config.system["conductScenarioAnalysis"]:
                        # handle scenarios
                        if baseScenario != "":
                            subfolder += f"scenario_{baseScenario}_{scenario}"
                            scenario_name = subfolder
                        else:
                            subfolder += f"scenario_{scenario}"
                            scenario_name = subfolder
                    # handle myopic foresight
                    if len(stepsOptimizationHorizon) > 1:
                        if subfolder != "":
                            subfolder += f"_"
                        subfolder += f"MF_{stepHorizon}"
                    # write results
                    evaluation = Postprocess(optimizationSetup, scenarios=config.scenarios, subfolder=subfolder,
                                             modelName=modelName, scenario_name=scenario_name)
