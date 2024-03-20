"""
:Title:        ZEN-GARDEN
:Created:      October-2021
:Authors:      Alissa Ganter (aganter@ethz.ch),
               Jacob Mannhardt (jmannhardt@ethz.ch)
:Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Compilation  of the optimization problem.
"""
import importlib.util
import logging
import os
from collections import defaultdict
import importlib

from .model.optimization_setup import OptimizationSetup
from .postprocess.postprocess import Postprocess
from .utils import setup_logger, InputDataChecks, StringUtils, ScenarioUtils

# we setup the logger here
setup_logger()

def main(config, dataset_path=None, job_index=None):
    """
    This function runs the compile.py script that was used in ZEN-Garden prior to the package build, it is executed
    in the __main__.py script

    :param config: A config instance used for the run
    :param dataset_path: If not None, used to overwrite the config.analysis["dataset"]
    :param job_index: The index of the scenario to run or a list of indices, if None, all scenarios are run in sequence
    """

    if system['calculation_mode'] == 'ZEN-GARDEN':
        optimization_setup = main_zen(config, dataset_path, job_index)

    elif system['calculation_mode'] == 'ALGORITHM':

        calculation_flag = 'design'
        optimization_setup = main_algor(config, dataset_path, job_index, calculation_flag)


def main_zen(config, dataset_path, job_index):
    """
    This function runs ZEN garden,
    it is executed in the __main__.py script

    :param config: A config instance used for the run
    :param dataset_path: If not None, used to overwrite the config.analysis["dataset"]
    :param job_index: The index of the scenario to run or a list of indices, if None, all scenarios are run in sequence
    """

    # print the version
    version = importlib.metadata.version("zen-garden")
    logging.info(f"Running ZEN-Garden version: {version}")

    # prevent double printing
    logging.propagate = False

    # overwrite the path if necessary
    if dataset_path is not None:
        # logging.info(f"Overwriting dataset to: {dataset_path}")
        config.analysis["dataset"] = dataset_path
    logging.info(f"Optimizing for dataset {config.analysis['dataset']}")
    # get the abs path to avoid working dir stuff
    config.analysis["dataset"] = os.path.abspath(config.analysis['dataset'])
    config.analysis["folder_output"] = os.path.abspath(config.analysis['folder_output'])

    ### SYSTEM CONFIGURATION
    input_data_checks = InputDataChecks(config=config, optimization_setup=None)
    input_data_checks.check_dataset()
    system_path = os.path.join(config.analysis['dataset'], "system.py")
    spec = importlib.util.spec_from_file_location("module", system_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    system = module.system
    config.system.update(system)
    input_data_checks.check_technology_selections()
    input_data_checks.check_year_definitions()
    # overwrite default system and scenario dictionaries
    scenarios,elements = ScenarioUtils.get_scenarios(config,job_index)
    # get the name of the dataset
    model_name, out_folder = StringUtils.get_model_name(config.analysis,config.system)
    # clean sub-scenarios if necessary
    ScenarioUtils.clean_scenario_folder(config,out_folder)
    ### ITERATE THROUGH SCENARIOS
    for scenario, scenario_dict in zip(scenarios, elements):
        # FORMULATE THE OPTIMIZATION PROBLEM
        # add the scenario_dict and read input data
        optimization_setup = OptimizationSetup(config, scenario_dict=scenario_dict, input_data_checks=input_data_checks)
        # get rolling horizon years
        steps_horizon = optimization_setup.get_optimization_horizon()
        # iterate through horizon steps
        for step in steps_horizon:
            StringUtils.print_optimization_progress(scenario,steps_horizon,step)
            # overwrite time indices
            optimization_setup.overwrite_time_indices(step)
            # create optimization problem
            optimization_setup.construct_optimization_problem()
            # SOLVE THE OPTIMIZATION PROBLEM
            optimization_setup.solve()
            # break if infeasible
            if not optimization_setup.optimality:
                # write IIS
                optimization_setup.write_IIS()
                break
            # save new capacity additions and cumulative carbon emissions for next time step
            optimization_setup.add_results_of_optimization_step(step)
            # EVALUATE RESULTS
            # create scenario name, subfolder and param_map for postprocessing
            scenario_name, subfolder, param_map = StringUtils.generate_folder_path(
                config = config,scenario = scenario,scenario_dict=scenario_dict,steps_horizon=steps_horizon,step=step
            )
            # write results
            Postprocess(optimization_setup, scenarios=config.scenarios, subfolder=subfolder,
                            model_name=model_name, scenario_name=scenario_name, param_map=param_map)
    logging.info("--- Optimization finished ---")
    return optimization_setup

def main_algor(config, dataset_path, job_index, calculation_flag):
    """
    This function runs ZEN garden,
    it is executed in the __main__.py script

    :param config: A config instance used for the run
    :param dataset_path: If not None, used to overwrite the config.analysis["dataset"]
    :param job_index: The index of the scenario to run or a list of indices, if None, all scenarios are run in sequence
    """

    # print the version
    version = importlib.metadata.version("zen-garden")
    logging.info(f"Running ZEN-Garden version: {version}")

    # prevent double printing
    logging.propagate = False

    # overwrite the path if necessary
    if dataset_path is not None:
        # logging.info(f"Overwriting dataset to: {dataset_path}")
        config.analysis["dataset"] = dataset_path
    logging.info(f"Optimizing for dataset {config.analysis['dataset']}")
    # get the abs path to avoid working dir stuff
    config.analysis["dataset"] = os.path.abspath(config.analysis['dataset'])
    config.analysis["folder_output"] = os.path.abspath(config.analysis['folder_output'])

    ### SYSTEM CONFIGURATION
    input_data_checks = InputDataChecks(config=config, optimization_setup=None)
    input_data_checks.check_dataset()
    system_path = os.path.join(config.analysis['dataset'], "system.py")
    spec = importlib.util.spec_from_file_location("module", system_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    system = module.system
    config.system.update(system)
    input_data_checks.check_technology_selections()
    input_data_checks.check_year_definitions()

    # Set scenario flag to False, if the design is being calculated
    if calculation_flag == 'design':
        config.system["conduct_scenario_analysis"] = False
    else:
        config.system["conduct_scenario_analysis"] = True

    # overwrite default system and scenario dictionaries
    scenarios,elements = ScenarioUtils.get_scenarios(config,job_index)
    # get the name of the dataset
    model_name, out_folder = StringUtils.get_model_name(config.analysis,config.system)
    # clean sub-scenarios if necessary
    ScenarioUtils.clean_scenario_folder(config,out_folder)
    ### ITERATE THROUGH SCENARIOS
    for scenario, scenario_dict in zip(scenarios, elements):
        # FORMULATE THE OPTIMIZATION PROBLEM
        # add the scenario_dict and read input data
        optimization_setup = OptimizationSetup(config, scenario_dict=scenario_dict, input_data_checks=input_data_checks)
        # get rolling horizon years
        steps_horizon = optimization_setup.get_optimization_horizon()
        # iterate through horizon steps
        for step in steps_horizon:
            StringUtils.print_optimization_progress(scenario,steps_horizon,step)
            # overwrite time indices
            optimization_setup.overwrite_time_indices(step)
            # create optimization problem
            optimization_setup.construct_optimization_problem()
            # SOLVE THE OPTIMIZATION PROBLEM
            optimization_setup.solve()
            # break if infeasible
            if not optimization_setup.optimality:
                # write IIS
                optimization_setup.write_IIS()
                break
            # save new capacity additions and cumulative carbon emissions for next time step
            optimization_setup.add_results_of_optimization_step(step)
            # EVALUATE RESULTS
            # create scenario name, subfolder and param_map for postprocessing
            scenario_name, subfolder, param_map = StringUtils.generate_folder_path(
                config = config,scenario = scenario,scenario_dict=scenario_dict,steps_horizon=steps_horizon,step=step
            )
            # write results
            Postprocess(optimization_setup, scenarios=config.scenarios, subfolder=subfolder,
                            model_name=model_name, scenario_name=scenario_name, param_map=param_map)
    logging.info("--- Optimization finished ---")
    return optimization_setup

def space_generation_bayesian(flow_at_nodes, dummy_edges, years):
    """
    This function creates the space/range for the all (input-)variables to vary in the bayesian optimization based on the data
    in the flow_at_nodes dict.

    Parameters:
        flow_at_nodes (dict): Dict containing the data for every edge from the design calculation to define the space
        dummy_edges (list): List with the edges involved in the bayesian optimization
        years (list): (Nested) List with the years to be optimized in the run

    Returns:
        space (list): List containing the space for every variable involved in the bayesian optimization
        names (list): List containing the variable names for every variable involved in the bayesian optimization
    """


    # Filter the dicts if there is no flow over the edge for specific transport type.
    # import_at_nodes = {key: val for key, val in import_at_nodes.items() if sum(sum(node_imp) for node_imp in val.values()) != 0}
    # export_at_nodes = {key: val for key, val in export_at_nodes.items() if sum(sum(node_exp) for node_exp in val.values()) != 0}

    # Filter the dicts if there is no flow over the edge for specific transport type.
    # sum_flow_import = dict()
    # sum_flow_export = dict()
    # for key_transport in import_at_nodes:
    #     sum_flow_import[key_transport] = []
    #     sum_flow_export[key_transport] = []
    #
    #     for key_node in import_at_nodes[key_transport]:
    #         sum_node_imp = import_at_nodes[key_transport][key_node]
    #         sum_flow_import[key_transport].append(sum(sum_node_imp))
    #         sum_node_exp = export_at_nodes[key_transport][key_node]
    #         sum_flow_export[key_transport].append(sum(sum_node_exp))
    #
    # for key_zero_imp, key_zero_ex in zip(sum_flow_import, sum_flow_export):
    #     if sum(sum_flow_import[key_zero_imp]) == 0:
    #         del import_at_nodes[key_zero_imp]
    #     if sum(sum_flow_export[key_zero_ex]) == 0:
    #         del export_at_nodes[key_zero_ex]

    # With the values from the dict, define the space for each edge and suproblem
    space = []
    names = []
    for transport in flow_at_nodes:
        for year in range(len(years)):
            for edge in dummy_edges:
                nodes = edge.split('-')
                # node_import = nodes[0] + 'dummy'
                # node_export = nodes[1] + 'dummy'

                min_value = 0 #min(import_at_nodes[transport][node_import][year], export_at_nodes[transport][node_export][year])
                max_value = flow_at_nodes[transport][edge][year]

                # Define space for the import_availability side and the demand side
                name_import = str(edge) + '.' + str(transport) + '.' + str(year) + '.import'
                name_demand = str(edge) + '.' + str(transport) + '.' + str(year) + '.demand'

                name_to_append = str(edge) + '.' + str(transport) + '.' + str(year)

                if min_value != max_value:
                    # Define space for the variation of the availability_import and the demand for the specific edge
                    space_to_append = [Real(min_value, max_value, name=name_import), Real(min_value, max_value, name=name_demand)]
                    space.append(space_to_append)
                    names.append(name_to_append)

    return space, names


def energy_model(sample_points, nodes_scenarios):
    """
    Function to set up two dictionaries, which contain the information to modifiy the files (availability_import, demand)
    for the next iteration.

    Parameters:
        sample_points (dict): Dict with Key: edge and Value: value for availability_import and demand for specifc edge (needed later to modify files)
        nodes_scenarios (list): (Nested) list containing lists with the specific nodes of every scenario

    Returns:
        avail_import_data (dict): Dict with the information to the availability import for all nodes, all carriers and all years.
        demand_data (dict): Dict with the information to the demand for all nodes, all carriers and all years.
    """

    avail_import_dict = dict()
    demand_dict = dict()
    for key_sample in sample_points:

        # Get infos of key_sample
        edge, transport_type, year = key_sample.split('.')

        # Amount of availability_import and demand
        avail_import = sample_points[key_sample][0]
        demand = sample_points[key_sample][1]

        import_node = edge.split('-')[0] + 'dummy'
        demand_node = edge.split('-')[1] + 'dummy'

        # List with edges to analyze
        edges_to_analyze = [f"{import_node}-{edge.split('-')[1]}", f"{edge.split('-')[0]}-{demand_node}"]

        # Check in which scenario the edge is
        for edge_scen in edges_to_analyze:

            # Differentiate between availability_import and demand to be modified

            if 'dummy' in edge_scen.split('-')[0]:

                # availability_import case
                node_to_modify = edge_scen.split('-')[0]
                node_helper = edge_scen.split('-')[1]
                for idx, scenario in enumerate(nodes_scenarios):
                    if node_helper in scenario and node_to_modify in scenario:
                        scenario_index = idx

                if scenario_index not in avail_import_dict:
                    avail_import_dict[scenario_index] = dict()

                if transport_type not in avail_import_dict[scenario_index]:
                    avail_import_dict[scenario_index][transport_type] = dict()

                if node_to_modify not in avail_import_dict[scenario_index][transport_type]:
                    avail_import_dict[scenario_index][transport_type][node_to_modify] = dict()

                if year not in avail_import_dict[scenario_index][transport_type][node_to_modify]:
                    avail_import_dict[scenario_index][transport_type][node_to_modify][year] = []
                    avail_import_dict[scenario_index][transport_type][node_to_modify][year].append(avail_import)
                else:
                    avail_import_dict[scenario_index][transport_type][node_to_modify][year].append(avail_import)


            elif 'dummy' in edge_scen.split('-')[1]:

                # demand case
                node_to_modify = edge_scen.split('-')[1]
                node_helper = edge_scen.split('-')[0]
                for idx, scenario in enumerate(nodes_scenarios):
                    if node_helper in scenario and node_to_modify in scenario:
                        scenario_index = idx

                if scenario_index not in demand_dict:
                    demand_dict[scenario_index] = dict()

                if transport_type not in demand_dict[scenario_index]:
                    demand_dict[scenario_index][transport_type] = dict()

                if node_to_modify not in demand_dict[scenario_index][transport_type]:
                    demand_dict[scenario_index][transport_type][node_to_modify] = dict()

                if year not in demand_dict[scenario_index][transport_type][node_to_modify]:
                    demand_dict[scenario_index][transport_type][node_to_modify][year] = []
                    demand_dict[scenario_index][transport_type][node_to_modify][year].append(demand)
                else:
                    demand_dict[scenario_index][transport_type][node_to_modify][year].append(demand)


    return avail_import_dict, demand_dict


def setup_logger(name, log_file, level=logging.INFO):
    """
    Function to setup a logger with a file handler and formatter

    Parameters:
        name (str): String defining the name of the .log-file
        log_file (str): String defining the path of the .log-file
        level: logging.INFO

    Returns:
        logger
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)
    file_handler = logging.FileHandler(log_file)
    logger.addHandler(file_handler)

    return logger


def create_files(avail_import_data, demand_data, specific_carrier_path, set_carrier_folder, years, all_nodes, nodes_scenarios):
    """
    Create files for the calculation of the scenarios (availability_import, demand, price_import)

    Parameters:
        avail_import_data (dict): Dict with the information to the availability import for all nodes, all carriers and all years.
        demand_data (dict): Dict with the information to the demand for all nodes, all carriers and all years.
        specific_carrier_path (list): List containing the paths to all carriers.
        set_carrier_folder (list): List containing the paths to all carriers.
        years (list): List with the years defined to optimize.
        all_nodes (list): List containing all nodes present in the specific run.
        nodes_scenarios (list): (Nested) list with lists containing the nodes of the different scenarios.

    Returns:
        None
    """

    create_new_import_files_bayesian(avail_import_data, specific_carrier_path, years, all_nodes, nodes_scenarios)
    create_new_demand_files_bayesian(demand_data, set_carrier_folder, years)
    create_new_priceimport_file_bayesian(avail_import_data, set_carrier_folder, years)

    return None