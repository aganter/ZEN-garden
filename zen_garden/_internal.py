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
import numpy as np
import time
import pickle
import shutil
import importlib

from skopt import Optimizer
from skopt.space import Real
from contextlib import contextmanager

from .model.optimization_setup import OptimizationSetup
from .postprocess.postprocess import Postprocess
from .postprocess.results import Results
from .utils import setup_logger, InputDataChecks, StringUtils, ScenarioUtils
from .helper_functions import *
from .community_detection import clustering_performance

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

    # Get calculation mode from config
    calculation_mode = config.system['calculation_mode']

    if calculation_mode == 'ZEN-GARDEN':

        optimization_setup = main_zen(config, dataset_path, job_index)

        return optimization_setup


    elif calculation_mode == 'ALGORITHM':

        # Preprocess
        remove_dummy_nodes_and_edges(config)

        # Perform design calculation and measure time
        calculation_flag = 'design'
        with timed_operation("Design calculation"):
            optimization_setup = main_algor(config, dataset_path, job_index, calculation_flag)

        # Copy the results to the protocol folder
        destination_folder = copy_resultsfolder(calculation_flag, config)

        # Delete old files
        delete_old_files(config)

        # List should be emtpy for clustering process. Check it and force it
        if len(config.system.set_cluster_nodes) != 0:
            config.system.set_cluster_nodes = []

        # Do the clustering and define the cluster_nodes variable in the updated config file
        config = clustering_performance(destination_folder, config)

        # Function to modify all the needed files/configurations, based on results from design
        flow_at_nodes, dummy_edges, nodes_scenarios = modify_configs(config, destination_folder)

        # Get results object from design calculation
        run_path = config.analysis['dataset']
        result = Results(destination_folder)

        # Get needed information
        all_nodes = config.system["set_nodes"]
        reference_year = config.system['reference_year']
        optimized_years = config.system['optimized_years']
        interval = config.system['interval_between_years']

        # Get actual years (not only indexes)
        years = [reference_year + year for year in range(0, optimized_years * interval, interval)]

        # Get paths to all carriers
        set_carrier_folder = os.path.join(run_path, 'set_carriers')
        all_carriers = result.solution_loader.scenarios['none'].system.set_carriers
        specific_carrier_path = [os.path.join(set_carrier_folder, carrier) for carrier in all_carriers]

        # Create the parameter space for all edges
        space, names, flag_seperate, space_for_adaption = space_generation_bayesian(flow_at_nodes, dummy_edges, years)

        # Check if no optimization with the bayesian is needed
        if flag_seperate == True:
            logging.info('There is no flow among the scenarios. Do a separate optimization for each of the scenarios')
            cluster_nodes = config.system.model_extra['set_cluster_nodes']

            # The length of the cluster_nodes is the number of the scenarios to calculate
            new_scenario = create_scenario_dict(cluster_nodes)
            calculation_flag = 'individual'
            optimization_setup = main_algor(config, dataset_path, job_index, calculation_flag, scenarios_ind=new_scenario)

            return optimization_setup


        # Next step: Operational calculation
        calculation_flag = 'loop'

        # Create loggers
        loggers = create_logs(destination_folder, names, nodes_scenarios, years)

        # Flag to check if it is the first iteration in the loop later
        flag_iter = True

        # Number of iterations for the optimization loop ¦¦ fixed configuration based on empirical values
        agg_ts_list = [4, 5]
        n_iterations = [12, 6]

        protocol_flow = dict()
        protocol_imp_dem = dict()
        old_spaces_protocol = dict()
        variable_information = dict()
        return_information = dict()

        # Initialize the optimizer with the Gaussian process estimator
        optimizer_edge = dict()
        for edge, name in zip(space, names):
            optimizer_edge[name] = Optimizer(dimensions=edge, base_estimator="gp", n_initial_points=10,
                                             acq_func="gp_hedge", random_state=42)

        # Keep track of iteration number
        actual_iteration = 0
        for n_impr in range(len(n_iterations)):

            for n_iter in range(n_iterations[n_impr]):

                # Ask the optimizer for the next point to sample
                sample_points = {}
                for key_edge, opt in optimizer_edge.items():
                    sample_points[key_edge] = opt.ask()

                # Modify input csv files with the new sample points
                avail_import_data, demand_data = energy_model(sample_points, nodes_scenarios)
                create_files(avail_import_data, demand_data, specific_carrier_path, set_carrier_folder, years, all_nodes,
                             nodes_scenarios, result, flag_iter, config)

                # Start optimization
                agg_ts = agg_ts_list[n_impr]
                with timed_operation("Operation calculation"):
                    optimization_setup = main_algor(config, dataset_path, job_index, calculation_flag, adapted_agg_ts=agg_ts)

                # Results object
                destination_folder = copy_resultsfolder(calculation_flag, config, iteration=actual_iteration)
                res = Results(destination_folder)

                # Analyze all edges
                flows_in_out_protocol = []
                for key_edge in optimizer_edge:
                    if key_edge not in protocol_flow and key_edge not in protocol_imp_dem:
                        protocol_flow[key_edge] = []
                        protocol_imp_dem[key_edge] = []

                    edge, transport_type, year_str = key_edge.split('.')
                    year = int(year_str)
                    node_in, node_out = (node + 'dummy' for node in edge.split('-'))

                    # List with edges to analyze
                    edges_to_analyze = [f"{node_in}-{edge.split('-')[1]}", f"{edge.split('-')[0]}-{node_out}"]
                    flows = []

                    # Check which scenario the edge is in
                    for edge_scen in edges_to_analyze:
                        node_to, node_reverso = edge_scen.split('-')
                        for idx, scenario in enumerate(nodes_scenarios):
                            if node_to in scenario and node_reverso in scenario:
                                scenario_index = idx

                        scenario_name = 'scenario_' + str(scenario_index)

                        # Read out the flow on the specific edge
                        val = res.get_total('flow_transport', scenario_name=scenario_name).round(3).loc[transport_type].loc[edge_scen][year]
                        flows.append(val)
                        flows_in_out_protocol.append(val)

                    protocol_imp_dem[key_edge].append(sample_points[key_edge])

                    if len(flows) == 2:
                        # Calculate the difference
                        flow_diff = abs(flows[0] - flows[1])
                        protocol_flow[key_edge].append(flow_diff)

                        # Tell the optimizer about the objective function value at the sampled point
                        optimizer_edge[key_edge].tell(sample_points[key_edge], flow_diff)

                    else:
                        error_msg = 'Error while reading out results object.'
                        raise RuntimeError(error_msg)

                # Flow difference to logger
                flows_protocol = [protocol_flow[edge_prot][actual_iteration] for edge_prot in protocol_flow]
                flows_str = ': '.join(map(str, flows_protocol))
                loggers['difference_flows'].info(flows_str)

                # Actual flows in both directions to logger
                flows_in_out_protocol_str = ': '.join(map(str, flows_in_out_protocol))
                loggers['actual_flows'].info(flows_in_out_protocol_str)

                # Costs to logger
                costs = []
                for idx_scen, _ in enumerate(nodes_scenarios):
                    for idx_year, _ in enumerate(years):
                        scenario_act = 'scenario_' + str(idx_scen)
                        cost_temp = res.get_total('net_present_cost', scenario_name=scenario_act).round(3).loc[idx_year]
                        costs.append(cost_temp)

                costs_str = ': '.join(map(str, costs))
                loggers['costs'].info(costs_str)

                # Import and demand values to logger
                temp_list = []
                for key_prot in protocol_imp_dem:
                    avail_imp = protocol_imp_dem[key_prot][actual_iteration][0]
                    demand = protocol_imp_dem[key_prot][actual_iteration][1]
                    temp_list.append(avail_imp)
                    temp_list.append(demand)

                attrs_str = ': '.join(map(str, temp_list))
                loggers['import_demand'].info(attrs_str)

                # Change flag to False, since not first iteraion an
                flag_iter = False

                # Keep track of iteration
                actual_iteration = actual_iteration + 1

                # Delete folder due to memory issues
                shutil.rmtree(destination_folder)



            # Save optimizer object
            notebooks_path = os.path.abspath(os.path.join(destination_folder, '..', '..'))
            for name in names:
                filename = os.path.join(notebooks_path, 'optimizer_objects', f'opt_{name}.pkl')
                with open(filename, 'wb') as f:
                    pickle.dump(optimizer_edge[name], f)

            # Prepare for space adaption
            pushing_lp_imp, pushing_up_imp, pushing_lp_exp, pushing_up_exp, adaption_of_bound = pre_spaceadaption(optimizer_edge, space_for_adaption)

            # Create adapted spaces
            if n_impr == 0:
                space, names, adapted_space = space_generation_bayesian_adaption(space_for_adaption, pushing_lp_imp, pushing_up_imp, pushing_lp_exp, pushing_up_exp, adaption_of_bound)
                old_spaces_protocol[n_impr] = adapted_space
            else:
                space_for_adaption = old_spaces_protocol[n_impr-1]
                space, names, adapted_space = space_generation_bayesian_adaption(space_for_adaption, pushing_lp_imp, pushing_up_imp, pushing_lp_exp, pushing_up_exp, adaption_of_bound)
                old_spaces_protocol[n_impr] = adapted_space

            # Save Optimizer information for actual iteration
            returns = dict()
            variables = dict()
            for key_edge in names:
                returns[key_edge] = optimizer_edge[key_edge].yi
                variables[key_edge] = optimizer_edge[key_edge].Xi

            # Add information from actual iteration to the list, where all values are saved from all previous iterations
            for key_edge in names:
                if key_edge not in variable_information:
                    variable_information[key_edge] = []
                    variable_information[key_edge] = variable_information[key_edge] + variables[key_edge]
                else:
                    variable_information[key_edge] = variable_information[key_edge] + variables[key_edge]

                if key_edge not in return_information:
                    return_information[key_edge] = []
                    return_information[key_edge] = return_information[key_edge] + returns[key_edge]
                else:
                    return_information[key_edge] = return_information[key_edge] + returns[key_edge]

            # Initialize the Optimizer with the Gaussian process estimator
            optimizer_edge = dict()
            for edge, name in zip(space, names):
                optimizer_edge[name] = Optimizer(dimensions=edge, base_estimator="gp", n_initial_points=10,
                                                 acq_func="gp_hedge", random_state=42)

            # Tell the Optimizer about the previous results.
            for key_edge in names:
                for variable_val, return_val in zip(variable_information[key_edge], return_information[key_edge]):

                    if (adapted_space[key_edge][0][0] < variable_val[0] < adapted_space[key_edge][0][1]) and (
                            adapted_space[key_edge][1][0] < variable_val[1] < adapted_space[key_edge][1][1]):

                        optimizer_edge[key_edge].tell(variable_val, return_val)


        return optimization_setup

    else:
        error_msg = f'The specified calculation mode "{calculation_mode}" is not allowed. Please choose between "ALGORITHM" and "ZEN-GARDEN".'
        raise RuntimeError(error_msg)



def main_zen(config, dataset_path, job_index, calculation_flag=None, adapted_agg_ts=None, scenarios_new=None):
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
    scenarios,elements = ScenarioUtils.get_scenarios(config,job_index, calculation_flag, adapted_agg_ts, scenarios_new)
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

def main_algor(config, dataset_path, job_index, calculation_flag, adapted_agg_ts=None, scenarios_ind=None):
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
        config.system.aggregated_time_steps_per_year = 1
    else:
        config.system["conduct_scenario_analysis"] = True

    # overwrite default system and scenario dictionaries
    scenarios,elements = ScenarioUtils.get_scenarios(config,job_index,calculation_flag, adapted_agg_ts, scenarios_ind)
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
    This function creates the optimization space for the all variables involved in the bayesian optimization based
    on the data in the flow_at_nodes dict.

    Parameters:
        flow_at_nodes (dict): Dict containing the info for every edge from the design calculation to define the space.
        dummy_edges (list): List with all the dummy edges.
        years (list): (Nested) List with the years to be optimized in the run.

    Returns:
        space (list): List containing the space for every variable involved in the bayesian optimization
        names (list): List containing the variable names of every variable involved in the bayesian optimization
        flag_separate (bool): Flag to check if the scenarios can be calculated separately without any
        bayesian optimization (True) or not (False)
        space_for_adaption (dict): Dict with the name of the variable as the key, and the corresponding optimization
        space as the value (needed for the dynamic space refinement).
    """

    # With the values from the dict, define the space for each edge and suproblem
    space = []
    names = []

    space_for_adaption = dict()

    for transport in flow_at_nodes:
        for year in range(len(years)):
            for edge in dummy_edges:

                min_value = 0
                max_value = flow_at_nodes[transport][edge][year]

                # Define space for the import_availability side and the demand (export) side
                name_import = f'{edge}.{transport}.{year}.import'
                name_demand = f'{edge}.{transport}.{year}.demand'

                name_to_append = f'{edge}.{transport}.{year}'

                if min_value != max_value:
                    # Define space for the variation of the availability_import and the demand (export) for the specific edge
                    diff = max_value - min_value
                    min_value = min_value + diff*0.5
                    max_value = max_value + diff*0.5

                    space_to_append = [Real(min_value, max_value, name=name_import), Real(min_value, max_value, name=name_demand)]
                    space.append(space_to_append)
                    names.append(name_to_append)

                    # Dictionary for later adaption of the space
                    space_for_adaption[name_to_append] = [[min_value, max_value], [min_value, max_value]]

    # Check if list is empty
    if len(space) == 0:
        logging.info('No edge to be optimized. Subproblems can be calculated separately.')
        flag_seperate = True
    else:
        flag_seperate = False

    return space, names, flag_seperate, space_for_adaption

@contextmanager
def timed_operation(operation_name):
    """Context manager for timing operations."""
    start_time = time.time()
    yield
    end_time = time.time()
    logging.info(f"{operation_name} took {end_time - start_time} seconds.")

def space_generation_bayesian_adaption(old_space, pushing_lp_imp, pushing_up_imp, pushing_lp_exp, pushing_up_exp, adaption_of_bound):
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


    # With the values from the dict, define the space for each edge and suproblem
    space = []
    names = []
    adapted_space = dict()

    for key_edge in old_space:

        # Get min and max of bound for import variable
        min_val_param_imp = old_space[key_edge][0][0]
        max_val_param_imp = old_space[key_edge][0][1]
        bound_imp = max_val_param_imp - min_val_param_imp

        # Get min and max of bound for export variable
        min_val_param_exp = old_space[key_edge][1][0]
        max_val_param_exp = old_space[key_edge][1][1]
        bound_exp = max_val_param_exp - min_val_param_exp


        # Define new space for the import
        if pushing_lp_imp[key_edge] == True and adaption_of_bound == True:
            min_val_param_imp_new = min_val_param_imp - bound_imp*0.3
            if min_val_param_imp_new < 0:
                min_val_param_imp_new = 0
        else:
            min_val_param_imp_new = min_val_param_imp #max(min_val_param_imp, min_val_param_imp + (abs(best_param_imp - min_val_param_imp) / bound_imp) * (bound_imp / learning_rate))

        if pushing_up_imp[key_edge] == True and adaption_of_bound == True:
            max_val_param_imp_new = max_val_param_imp + bound_imp*0.3
        else:
            max_val_param_imp_new = max_val_param_imp #min(max_val_param_imp, max_val_param_imp - (abs(max_val_param_imp - best_param_imp) / bound_imp) * (bound_imp / learning_rate))

        # Define new space for the export
        if pushing_lp_exp[key_edge] == True and adaption_of_bound == True:
            min_val_param_exp_new = min_val_param_exp - bound_exp*0.3
            if min_val_param_exp_new < 0:
                min_val_param_exp_new = 0
        else:
            min_val_param_exp_new = min_val_param_exp #max(min_val_param_exp, min_val_param_exp + (abs(best_param_exp - min_val_param_exp) / bound_exp) * (bound_exp / learning_rate))

        if pushing_up_exp[key_edge] == True and adaption_of_bound == True:
            max_val_param_exp_new = max_val_param_exp + bound_exp*0.3
        else:
            max_val_param_exp_new = max_val_param_exp #min(max_val_param_exp, max_val_param_exp - (abs(max_val_param_exp - best_param_exp) / bound_exp) * (bound_exp / learning_rate))

        name_import = key_edge + '.import'
        name_demand = key_edge + '.demand'

        space_to_append = [Real(min_val_param_imp_new, max_val_param_imp_new, name=name_import),
                           Real(min_val_param_exp_new, max_val_param_exp_new, name=name_demand)]
        space.append(space_to_append)
        names.append(key_edge)

        # Dictionary for later adaption of the space
        adapted_space[key_edge] = [[min_val_param_imp_new, max_val_param_imp_new], [min_val_param_exp_new, max_val_param_exp_new]]

    return space, names, adapted_space



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

def pre_spaceadaption(optimizer_edge, space_for_adaption):
    """
    Function to analyse the iterations for the preparation of the space adaption process. Checking best values and
    potential problems with bounds

    Parameters:
        optimizer_edge (dict): Dict with optimizer objects
        space_for_adaption (dict): Dict with defined variable space for every edge

    Returns:
        pushing_lp_imp (bool): Check if variable pushes against lower bound (import)
        pushing_up_imp (bool): Check if variable pushes against upper bound (import)
        pushing_lp_exp (bool): Check if variable pushes against lower bound (export)
        pushing_up_exp (bool): Check if variable pushes against upper bound (export)
        adaption_of_bound (bool): Check if adaption is needed
    """

    pushing_lp_imp = dict()
    pushing_up_imp = dict()
    pushing_lp_exp = dict()
    pushing_up_exp = dict()
    adaption_of_bound = False
    amount_to_check = 15
    for key_edge in optimizer_edge:

        # Get min and max of bound for import variable and export variable
        min_imp, max_imp = space_for_adaption[key_edge][0][0], space_for_adaption[key_edge][0][1]
        min_exp, max_exp = space_for_adaption[key_edge][1][0], space_for_adaption[key_edge][1][1]

        # Get the smallest numbers
        best_objective = sorted((num, idx) for idx, num in enumerate(optimizer_edge[key_edge].yi))[:amount_to_check]

        smallest_numbers = [num for num, idx in best_objective]
        smallest_indices = [idx for num, idx in best_objective]

        # Converting the list of optimization variables to a NumPy array for easy calculations
        optvars_np = np.array(optimizer_edge[key_edge].Xi)

        # Selecting the specific indices
        selected_optvars = optvars_np[smallest_indices, :]


        # Check if objective of 0 is reached, if not, adapt space
        mean_obj_func = sum(smallest_numbers) / len(smallest_numbers)
        if mean_obj_func > 10:
            adaption_of_bound = True


        # Import variable
        threshold = (max_imp - min_imp) * 0.05
        close_to_lb = [value[0] for value in selected_optvars if value[0] - min_imp < threshold]
        close_to_up = [value[0] for value in selected_optvars if max_imp - value[0] < threshold]

        pushing_lp_imp[key_edge] = len(close_to_lb) > len(selected_optvars) * 0.5  # More than 50% of the values are close
        pushing_up_imp[key_edge] = len(close_to_up) > len(selected_optvars) * 0.5  # More than 50% of the values are close


        # Export variable
        threshold = (max_exp - min_exp) * 0.05
        close_to_lb = [value[1] for value in selected_optvars if value[1] - min_exp < threshold]
        close_to_up = [value[1] for value in selected_optvars if max_exp - value[1] < threshold]

        pushing_lp_exp[key_edge] = len(close_to_lb) > len(selected_optvars) * 0.5  # More than 50% of the values are close
        pushing_up_exp[key_edge] = len(close_to_up) > len(selected_optvars) * 0.5  # More than 50% of the values are close

    return pushing_lp_imp, pushing_up_imp, pushing_lp_exp, pushing_up_exp, adaption_of_bound


def create_files(avail_import_data, demand_data, specific_carrier_path, set_carrier_folder, years, all_nodes, nodes_scenarios, result, flag_iter, config):
    """
    Create files for the calculation of the scenarios (availability_import, demand, price_import, price_export)

    Parameters:
        avail_import_data (dict): Dict with the information to the availability import for all nodes, all carriers and all years.
        demand_data (dict): Dict with the information to the demand for all nodes, all carriers and all years.
        specific_carrier_path (list): List containing the paths to all carriers.
        set_carrier_folder (str): Path to 'set_carrier' folder.
        years (list): List with the years defined to optimize.
        all_nodes (list): List containing all nodes present in the specific run.
        nodes_scenarios (list): (Nested) list with lists containing the nodes of the different scenarios.
        result (object): Result object from design calculation
        flag_iter (bool): Bool to check if it is the first time in this function
        config (): config instance of this run

    Returns:
        None
    """

    if flag_iter == True:
        # Availability Import/Export files
        create_new_import_files_bayesian(avail_import_data, specific_carrier_path, years, all_nodes, nodes_scenarios)
        create_new_export_files_bayesian(demand_data, specific_carrier_path, years, all_nodes, nodes_scenarios)

        # Price Import/Export files
        create_new_priceimport_file_bayesian(avail_import_data, set_carrier_folder, all_nodes, years)
        create_new_priceexport_file_bayesian(demand_data, set_carrier_folder, all_nodes, years, result, config)

    elif flag_iter == False:
        # Availability Import/Export files
        create_new_import_files_bayesian(avail_import_data, specific_carrier_path, years, all_nodes, nodes_scenarios)
        create_new_export_files_bayesian(demand_data, specific_carrier_path, years, all_nodes, nodes_scenarios)


    return None


def create_scenario_dict(cluster_nodes):
    """
    Creates the dict with the scenarios defined.

    Parameters:
        cluster_nodes (list): Nested list with nodes from each individual cluster.

    Returns:
        scenario_dict (dict): Dictionary with the defined scenarios.
    """
    scenario_dict = dict()

    for idx_cl, cluster_info in enumerate(cluster_nodes):
        scenario_dict[str(idx_cl)] = {'system': {'set_nodes': cluster_info,
                                                 'aggregated_time_steps_per_year': 380}
                                      }
    return scenario_dict