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
    # Overwrite default system and scenario dictionaries
    scenarios, elements = ScenarioUtils.get_scenarios(
        config=config, scenario_script_name="scenarios", job_index=job_index
    )
    # Get the name of the dataset
    model_name, out_folder = StringUtils.get_model_name(config.analysis, config.system)
    # Clean sub-scenarios if necessary
    ScenarioUtils.clean_scenario_folder(config, out_folder)
    # ITERATE THROUGH SCENARIOS
    for scenario, scenario_dict in zip(scenarios, elements):
        # FORMULATE THE OPTIMIZATION PROBLEM
        optimization_setup = OptimizationSetup(
            config,
            model_name=model_name,
            scenario_name=scenario,
            scenario_dict=scenario_dict,
            input_data_checks=input_data_checks,
        )
        # Fit the optimization problem for every steps defined in the config and save the results
        optimization_setup.fit()
        logging.info("")

        # BENDERS DECOMPOSITION
        if config.benders.benders_decomposition:
            logging.info("--- Benders Decomposition accessed ---")
            benders_decomposition = BendersDecomposition(
                config=config,
                analysis=config.analysis,
                monolithic_model=optimization_setup,
                scenario_name=scenario,
                use_monolithic_solution=config.benders.use_monolithic_solution,
            )
            benders_decomposition.fit()

        # MODELING TO GENERATE ALTERNATIVES
        if config.mga["modeling_to_generate_alternatives"]:
            logging.info("--- Modeling to Generate Alternatives accessed to generate near-optimal solutions ---")
            config.mga.analysis["dataset"] = os.path.abspath(config.analysis["dataset"])
            config.mga.analysis["folder_output"] = os.path.abspath(config.mga.analysis["folder_output"])
            mga_output_folder = StringUtils.get_output_folder(
                analysis=config.mga.analysis,
                system=config.mga["system"],
                folder_output=config.mga.analysis["folder_output"],
            )
            config.mga.system.update(config.system)
            config.mga.system.update(config.mga.immutable_system_elements)

            scenarios, elements = ScenarioUtils.get_scenarios(
                config=config.mga, scenario_script_name="mga_iterations", job_index=job_index
            )

            ScenarioUtils.clean_scenario_folder(config.mga, mga_output_folder)

            # The scenario will create a double level folder structure: the first to be the different MGA objectives
            # and the second to be the iterations for each objective
            for scenario, scenario_dict in zip(scenarios, elements):
                parts = scenario.split("_")
                scenario_name = "_".join(parts[:-2])
                iteration = int(parts[-1])
                logging.info("--- MGA for: %s ---", scenario_name)
                logging.info("--- Iteration %s ---", iteration + 1)
                logging.info("")
                mga_iterations = ModelingToGenerateAlternatives(
                    config=config,
                    optimized_setup=optimization_setup,
                    scenario_name=scenario,
                    scenario_dict=scenario_dict,
                )
                mga_iterations.fit()

            logging.info("--- MGA run finished ---")

    return optimization_setup
