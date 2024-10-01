"""
:Title:        ZEN-GARDEN
:Created:      October-2021
:Authors:      Alissa Ganter (aganter@ethz.ch),
               Jacob Mannhardt (jmannhardt@ethz.ch)
:Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Compilation  of the optimization problem.
"""

import importlib
import logging
import os
import json

from zen_garden.model.optimization_setup import OptimizationSetup
from zen_garden.model.objects.benders_decomposition.benders import BendersDecomposition
from zen_garden.model.objects.mga import ModelingToGenerateAlternatives
from zen_garden.utils import setup_logger, InputDataChecks, StringUtils, ScenarioUtils

# We setup the logger here
setup_logger()


def main(config, dataset_path=None, job_index=None):
    """
    This function runs ZEN garden,
    it is executed in the __main__.py script

    :param config: A config instance used for the run
    :param dataset_path: If not None, used to overwrite the config.analysis["dataset"]
    :param job_index: The index of the scenario to run or a list of indices, if None, all scenarios are run in sequence
    """

    # Print the version
    version = importlib.metadata.version("zen-garden")
    logging.info("Running ZEN-Garden version: %s", version)

    # Prevent double printing
    logging.propagate = False

    # Overwrite the path if necessary
    if dataset_path is not None:
        # logging.info(f"Overwriting dataset to: {dataset_path}")
        config.analysis["dataset"] = dataset_path
    logging.info("Optimizing for dataset %s", config.analysis["dataset"])
    # Get the abs path to avoid working dir stuff
    config.analysis["dataset"] = os.path.abspath(config.analysis["dataset"])
    config.analysis["folder_output"] = os.path.abspath(config.analysis["folder_output"])

    # SYSTEM CONFIGURATION
    input_data_checks = InputDataChecks(config=config, optimization_setup=None)
    input_data_checks.check_dataset()
    input_data_checks.read_system_file(config)
    input_data_checks.check_technology_selections()
    input_data_checks.check_year_definitions()

    # check if optimal solutions for scenarios should be computed
    if config.mga.benders.compute_optimal_solutions:
        logging.info("--- Computing cost-optimal solutions for benders subproblems ---")
        # save the current system and scenario dictionaries
        system_conduct_scenario_analysis = config.system.conduct_scenario_analysis
        # temporary overwrite the system and scenario dictionaries
        config.system.conduct_scenario_analysis = True
        config.analysis.folder_output_subfolder = "cost_optimal_solutions"
        scenarios, elements = ScenarioUtils.get_scenarios(
            config=config.mga, scenario_script_name="benders_scenarios", job_index=None)
        # Get the name of the dataset and clean sub-scenarios if necessary
        model_name, out_folder = StringUtils.get_model_name(config.analysis, config.system)
        ScenarioUtils.clean_scenario_folder(config, out_folder)
        for scenario, scenario_dict in zip(scenarios, elements):
            optimization_setup = OptimizationSetup(
                config,
                solver=config.solver,
                model_name=model_name,
                scenario_name=scenario,
                scenario_dict=scenario_dict,
                input_data_checks=input_data_checks,
            )
            optimization_setup.fit()
        config.system.conduct_scenario_analysis = system_conduct_scenario_analysis
        config.benders.compute_optimal_solutions = False
        config.analysis.folder_output_subfolder = ""

    # FORMULATE AND SOLVE THE OPTIMIZATION PROBLEM
    # Get the name of the dataset and clean sub-scenarios if necessary
    model_name, out_folder = StringUtils.get_model_name(config.analysis, config.system)
    ScenarioUtils.clean_scenario_folder(config, out_folder)
    optimization_setup = OptimizationSetup(
        config,
        solver=config.solver,
        model_name=model_name,
        scenario_name="",
        scenario_dict=dict(),
        input_data_checks=input_data_checks,
    )
    # Fit the optimization problem for every steps defined in the config and save the results
    optimization_setup.fit()
    logging.info("")

    # ITERATE THROUGH SCENARIOS
    # MODELING TO GENERATE ALTERNATIVES
    if config.mga["modeling_to_generate_alternatives"]:
        # Overwrite default system and scenario dictionaries
        mga_output_folder = StringUtils.get_output_folder(
            analysis=config.mga.analysis,
            system=config.mga["system"],
            folder_output=config.mga.analysis["folder_output"],
        )
        config.mga.analysis["dataset"] = os.path.abspath(config.analysis["dataset"])
        config.mga.analysis["folder_output"] = os.path.abspath(config.mga.analysis["folder_output"])
        config.mga.system.update(config.system)
        config.mga.system.update(config.mga.immutable_system_elements)
        scenarios, elements = ScenarioUtils.get_scenarios(
            config=config.mga, scenario_script_name="mga_iterations", job_index=job_index
        )
        ScenarioUtils.clean_scenario_folder(config.mga, mga_output_folder)
        for mga_scenario, mga_scenario_dict in zip(scenarios, elements):
            logging.info("--- Modeling to Generate Alternatives accessed to generate near-optimal solutions ---")

            # The scenario will create a double level folder structure: the first to be the different MGA objectives
            # and the second to be the iterations for each objective
            parts = mga_scenario.split("_")
            mga_scenario_name = "_".join(parts[:-2])
            iteration = int(parts[-1])
            logging.info("--- MGA for: %s ---", mga_scenario_name)
            logging.info("--- Iteration %s ---", iteration + 1)
            logging.info("")
            mga_iterations = ModelingToGenerateAlternatives(
                config=config,
                optimized_setup=optimization_setup,
                scenario_name=mga_scenario,
                scenario_dict=mga_scenario_dict,
                iteration=iteration
            )
            mga_iterations.fit()

            logging.info("--- MGA run finished ---")

    return optimization_setup
