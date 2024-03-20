import os
import csv
import copy
import shutil
import warnings

from zen_garden.postprocess.results import Results

# Control of function to import in _internal.py
__all__ = ['modify_configs', 'copy_resultsfolder', 'create_new_import_files_bayesian', 'create_new_demand_files_bayesian', 'create_new_priceimport_file_bayesian']

def modify_configs(config, destination_folder):

    run_path = config.analysis['dataset']
    result = Results(destination_folder)

    # Get all the carrier paths
    set_carrier_folder = os.path.join(run_path, 'set_carriers')
    all_carriers = result.results[None]['system']['set_carriers']
    specific_carrier_path = [os.path.join(set_carrier_folder, carrier) for carrier in all_carriers]

    # Check set_nodes and set_edges file for completeness
    dummy_nodes, dummy_edges, nodes_scenarios = new_setnodes_setedges_file(result, run_path)
    # Analyze the flow over the edges for the global problem (design)
    # import_at_nodes, export_at_nodes = analyze_imports_exports(result, dummy_nodes, dummy_edges)
    flow_at_nodes = analyze_imports_exports_bayesian_modified(result, dummy_edges)

    # create_new_import_files(specific_carrier_path, result, dummy_nodes, import_at_nodes)
    # create_new_priceimport_file(specific_carrier_path, import_at_nodes)
    # create_new_demand_file(specific_carrier_path, result, export_at_nodes)
    modifiy_attribute_file(specific_carrier_path)

    return flow_at_nodes, dummy_edges, nodes_scenarios

def create_dummy_nodes(clustering_nodes, set_nodes):

    temp_cluster = copy.deepcopy(clustering_nodes)

    for list in temp_cluster:
        for node in set_nodes:
                if node not in list:
                    dummy_node = node + 'dummy'
                    list.append(dummy_node)

    return temp_cluster


def copy_resultsfolder(calculation_flag, config, iteration=None):
    """
    This function copies the results folder from the original place to another folder.

    Parameters:
        calculation_flag (str): Type of calculation in algorithm (design or operation)
        iteration (int): Natural number (number of interation in loop)

    Returns:
        None
    """

    # Define folder paths
    ZEN_garden = os.path.abspath(os.path.join(config.analysis["folder_output"], '..', '..'))
    folder_outputs = config.analysis['folder_output']
    save_folder = os.path.join(ZEN_garden, 'notebooks', 'protocol_results')

    model_name = os.path.basename(config.analysis["dataset"])
    folder_outputs_spec = os.path.join(folder_outputs, model_name)



    # Distinguish between design and operational calculation
    if calculation_flag == 'design':
        destination_folder = os.path.join(save_folder, model_name)

        if os.path.exists(destination_folder):
            warnings.warn(f"The folder {destination_folder} already exists and is being deleted/overwritten.")
            shutil.rmtree(destination_folder)

        shutil.copytree(folder_outputs_spec, destination_folder)

        return destination_folder

    else:
        root_folder = os.path.join(save_folder, model_name + '_loop')
        destination_folder = os.path.join(root_folder, str(iteration))

        if os.path.exists(destination_folder):
            warnings.warn(f"The folder {destination_folder} already exists and is being deleted/overwritten.")
            shutil.rmtree(destination_folder)

        shutil.copytree(folder_outputs_spec, destination_folder)

        return destination_folder

#copy_resultsfolder('design')

def new_setnodes_setedges_file(result, run_path):
    """
    This function creates a list of the dummy nodes and a list of the dummy edges and returns both.
    It also checks, if the specific dummy nodes and edges are present in the set_nodes.csv and the set_edges.csv files.

    Parameters:
        result (object): Result object to analyze results
        run_path (string): Path to the data

    Returns:
        dummy_nodes (list): List with the dummy nodes
        dummy_edges (list): List with the dummy edges
    """

    # Initialize dataframe to get the existing edges.
    df = result.get_total('flow_transport')

    # Get all the edges of the global problem.
    edges = []
    for index in range(len(df)):
        nodes_connect = df.index[index][1]
        if nodes_connect not in edges:
            edges.append(nodes_connect)

    # Create the node-configuration for every scenario.
    nodes = copy.deepcopy(result.results[None]['system']['set_nodes'])
    cluster_nodes_temp = copy.deepcopy(result.results[None]['system']['set_cluster_nodes'])
    nodes_scenarios = create_dummy_nodes(cluster_nodes_temp, nodes)

    # Create the possible connections between normal nodes and dummy nodes.
    dummy_edges = []
    for nodes_scenario in nodes_scenarios:
        for edge in edges:
            if ((edge.split('-')[0] + 'dummy' in nodes_scenario) and (edge.split('-')[1] + 'dummy' not in nodes_scenario)) or ((edge.split('-')[0] + 'dummy' not in nodes_scenario) and (edge.split('-')[1] + 'dummy' in nodes_scenario)):
                if edge not in dummy_edges:
                    dummy_edges.append(edge)

    # Create the dummy nodes (every normal node will also be used as a dummy node in one of the subproblems).
    dummy_nodes = [dummynode + 'dummy' for dummynode in nodes]


    # Check if all new dummy nodes are in the set_nodes.csv. Create e new (updated) file if it is not the case.
    set_nodes_path = os.path.join(run_path, 'system_specification', 'set_nodes.csv')
    if os.path.exists(set_nodes_path):
        with open(set_nodes_path, 'r', newline='') as set_nodes_file:
            csv_reader = csv.reader(set_nodes_file)
            nodes_existing = [row for row in csv_reader]
    else:
        raise RuntimeError('Problem with opening the set_node.csv-file. Check if the file exists in the folder: ' + set_nodes_path)

    # Check if all dummy nodes are in the file. If the list is empty, all dummy nodes are missing; To be added to the set_nodes.csv-file.
    check_dummies = [dummynode for dummynode in dummy_nodes for node_exist in nodes_existing if dummynode in node_exist[0]]

    if set(check_dummies) != set(dummy_nodes):

        # Check, which dummy nodes are missing
        missing_dummy_nodes = [dummynode for dummynode in dummy_nodes if dummynode not in check_dummies]

        # Arrange the missing dummy nodes, s.t. they can be added to the set_nodes.csv file in the same structure.
        nodes_to_be_added = []
        for missing_node in missing_dummy_nodes:
            for node in nodes_existing:
                if missing_node[:2] == node[0]:
                    x_coord = node[1]
                    y_coord = node[2]
                    dummy_node_finish = [missing_node, x_coord, y_coord, missing_node]
                    nodes_to_be_added.append(dummy_node_finish)

        # Add the missing dummy nodes to the set_nodes.csv-file.
        with open(set_nodes_path, 'w', newline='') as set_nodes_file_mod:
            writer = csv.writer(set_nodes_file_mod)
            data = nodes_existing + nodes_to_be_added
            writer.writerows(data)


    # Check if all new dummy edges are in the set_edges.csv and create a new (updated) file if it is not the case.
    set_edges_path = os.path.join(run_path, 'system_specification', 'set_edges.csv')
    if os.path.exists(set_edges_path):
        with open(set_edges_path, 'r', newline='') as set_edges_file:
            csv_reader = csv.reader(set_edges_file)
            edges_existing = [row for row in csv_reader]
    else:
        raise RuntimeError('Problem to open the set_edges.csv file. Please check if the file exists in the folder: ' + set_edges_path)

    # Create list with all the needed edges in the subproblems.
    dummy_edges_list = []
    for node in nodes:
        for edge in dummy_edges:
            if node == edge.split('-')[0]:
                node_from = edge.split('-')[0] + 'dummy'
                node_to = edge.split('-')[1]
                edge_temp = node_from + '-' + node_to
                new_edge = [edge_temp, node_from, node_to]
                edge_temp_reverse = node_to + '-' + node_from
                new_edge_reverse = [edge_temp_reverse, node_to, node_from]
                dummy_edges_list.append(new_edge)
                dummy_edges_list.append(new_edge_reverse)

    # Check if all edges are in the set_edges.csv-file.
    check_dummy_edges = [dummyedge for dummyedge in dummy_edges_list if dummyedge in edges_existing]

    # Find the missing edges (if there are some).
    flat_list_1 = [item for sublist in dummy_edges_list for item in sublist]
    flat_list_2 = [item for sublist in check_dummy_edges for item in sublist]
    elements_not_in_2 = [element for element in flat_list_1 if element not in flat_list_2]
    elements_not_in_1 = [element for element in flat_list_2 if element not in flat_list_1]

    # If edges missing, the if-statement is true. Add edges to the set_edges.csv-file.
    if set(elements_not_in_2) != set(elements_not_in_1):
        dummy_edges_to_be_added = []
        for missingedge in elements_not_in_2:
            if '-' in missingedge:
                split_edge = missingedge.split('-')
                missing_edge = [missingedge, split_edge[0], split_edge[-1]]
                dummy_edges_to_be_added.append(missing_edge)

        with open(set_edges_path, 'w', newline='') as set_edges_file_mod:
            writer = csv.writer(set_edges_file_mod)
            data = edges_existing + dummy_edges_to_be_added
            writer.writerows(data)

    return dummy_nodes, dummy_edges, nodes_scenarios


def analyze_imports_exports(result, dummy_nodes, dummy_edges):
    """
    This function checks the yearly flow_transport for the big problem and checks only the flows for edges that arise
    when creating the subproblem(s).

    Parameters:
        result (object): Result object to analyze results
        dummy_nodes (list): List with the dummy nodes
        dummy_edges (list): List with the dummy edges

    Returns:
        import_at_nodes (dict): Dictionary with the amount imported of specific carrier for every node.
        export_at_node (dict): Dictionary with the amount exported of specific carrier for every node.
    """

    # Get some information from the result object
    df = result.get_total('flow_transport').round(3)
    transport_types = result.results[None]['system']['set_transport_technologies']

    # Analyze the imports and exports for the dummy nodes
    import_at_nodes = dict()
    export_at_nodes = dict()
    for transport in transport_types:
        import_at_nodes[transport] = dict()
        export_at_nodes[transport] = dict()
        existing_edges = df.loc[transport].index
        for node in dummy_nodes:
            import_at_nodes[transport][node] = []
            export_at_nodes[transport][node] = []
            for ex_edge in existing_edges:
                if ex_edge.split('-')[0] == node[:2] and ex_edge in dummy_edges:
                    import_at_nodes[transport][node].append(list(df.loc[transport].loc[ex_edge]))
                if ex_edge.split('-')[-1] == node[:2] and ex_edge in dummy_edges:
                    export_at_nodes[transport][node].append(list(df.loc[transport].loc[ex_edge]))

    # Do the summation of every source
    for transport_key in import_at_nodes:
        for node in dummy_nodes:
            copied_list_imp = import_at_nodes[transport_key][node].copy()
            copied_list_exp = export_at_nodes[transport_key][node].copy()
            import_at_nodes[transport_key][node] = [sum(values) for values in zip(*copied_list_imp)]
            export_at_nodes[transport_key][node] = [sum(values) for values in zip(*copied_list_exp)]

    return import_at_nodes, export_at_nodes


def analyze_imports_exports_bayesian_modified(result, dummy_edges):
    """
    This function checks the yearly flow_transport for the big problem and checks only the flows for edges that arise
    when creating the subproblem(s).

    Parameters:
        result (object): Result object to analyze results
        dummy_nodes (list): List with the dummy nodes
        dummy_edges (list): List with the dummy edges

    Returns:
        import_at_nodes (dict): Dictionary with the amount imported of specific carrier for every node.
        export_at_node (dict): Dictionary with the amount exported of specific carrier for every node.
    """

    # Get some information from the result object
    df = result.get_total('flow_transport').round(3)
    transport_types = result.results[None]['system']['set_transport_technologies']

    # Analyze the imports and exports for the dummy nodes
    flow_at_nodes = dict()
    for transport in transport_types:
        flow_at_nodes[transport] = dict()

        for edge in dummy_edges:
            flow_at_nodes[transport][edge] = list(df.loc[transport].loc[edge])


    return flow_at_nodes


def create_new_import_files(specific_carrier_path, result, dummy_nodes, import_at_nodes):
    """
    This function creates new availability_import_yearly files for the carriers that are being transported.

    Parameters:
        specific_carrier_path (list): List with containing the paths to the carriers in the set_carrier-folder
        result (object): Result object to analyze results
        dummy_nodes (list): List with the dummy nodes
        import_at_nodes (dict): Dictionary with the amount imported of specific carrier for every node.

    Returns:
        None
    """

    # Get info needed from result object
    reference_year = result.results[None]['system']['reference_year']
    interval_years = result.results[None]['system']['interval_between_years']
    opt_years = result.results[None]['system']['optimized_years']
    years = [reference_year + year for year in range(0, opt_years * interval_years, interval_years)]

    # Define the carrier for every transport technology
    transporttype_to_carrier = {'biomethane_transport': 'biomethane',
                                'carbon_pipeline': 'carbon',
                                'dry_biomass_truck': 'dry_biomass',
                                'hydrogen_pipeline': 'hydrogen',
                                'power_line': 'electricity'}

    # Write the new availability_import files
    for transport_type in transporttype_to_carrier:
        for carrier_path in specific_carrier_path:

            # Specify needed paths
            avail_import_path = os.path.join(carrier_path, 'availability_import.csv')
            avail_import_yearly_path = os.path.join(carrier_path, 'availability_import_yearly_variation.csv')
            avail_import_yearly_path_new = os.path.join(carrier_path, 'availability_import_yearly_new.csv')

            carrier = os.path.basename(carrier_path)
            if carrier == transporttype_to_carrier[transport_type]:

                # Check if 'availability_import.csv'-file exists and read the data
                if os.path.exists(avail_import_path):
                    with open(avail_import_path, 'r', newline='') as input_file:
                        csv_reader = csv.reader(input_file)
                        avail_import = [row for row in csv_reader]
                else:
                    avail_import = []

                # Check if 'availability_import_yearly.csv'-file exists and read the data
                if os.path.exists(avail_import_yearly_path):
                    with open(avail_import_yearly_path, 'r', newline='') as yearly_var_file:
                        csv_reader = csv.reader(yearly_var_file)
                        avail_import_yearly = [row for row in csv_reader]
                else:
                    avail_import_yearly = []

                # If both files exist
                if len(avail_import_yearly) != 0 and len(avail_import) != 0:

                    # Change the already existing (hourly) values to yearly values
                    for import_aval in avail_import:
                        for import_vari in avail_import_yearly:
                            if import_aval[0] == import_vari[0] and avail_import_yearly.index(import_vari) != 0 and avail_import.index(import_aval) != 0:
                                new_val = float(import_aval[-1]) * 8760 * float(import_vari[-1])
                                import_vari[-1] = str(round(new_val, 3))

                    # Get the values from the 'import_at_nodes' dict in the correct structure for the new .csv-file
                    add_dummy_nodes = []
                    for dummynode in dummy_nodes:
                        for index in range(len(import_at_nodes[transport_type][dummynode])):
                            yearly_imp = round(float(import_at_nodes[transport_type][dummynode][index]), 3)
                            year = years[index]
                            new_row = [dummynode, year, yearly_imp]
                            add_dummy_nodes.append(new_row)

                    # Combine both data for the new 'availability_import_yearly'-file
                    new_data = avail_import_yearly + add_dummy_nodes
                    with open(avail_import_yearly_path_new, mode="w", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerows(new_data)

                else:
                    with open(avail_import_yearly_path_new, mode="w", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow(['node', 'year', 'availability_import'])

                        realnodes = ['CH', 'FR', 'DE', 'IT', 'AT', 'PL', 'CZ']
                        for realnode in realnodes:
                            for year in years:
                                writer.writerow([realnode, year, 0])

                        for dummynode in dummy_nodes:
                            for index in range(len(import_at_nodes[transport_type][dummynode])):
                                yearly_imp = round(float(import_at_nodes[transport_type][dummynode][index]), 4)
                                year = years[index]
                                new_row = [dummynode, year, yearly_imp]
                                writer.writerow(new_row)
            # else:
            #     with open(avail_import_yearly_path_new, mode="w", newline="") as file:
            #         writer = csv.writer(file)
            #         writer.writerow(['node', 'year', 'availability_import'])
            #
            #         allnodes = ['CH', 'FR', 'DE', 'IT', 'AT', 'PL', 'CZ', 'CHdummy', 'FRdummy', 'DEdummy', 'ITdummy', 'ATdummy', 'PLdummy', 'CZdummy']
            #         for allnode in allnodes:
            #             for year in years:
            #                 writer.writerow([allnode, year, 0])

    return None

def create_new_export_files(specific_carrier_path, result, dummy_nodes, export_at_nodes):
    """
    This function creates new availability_export_yearly files for the carriers that are being transported.

    Parameters:
        specific_carrier_path (list): List with containing the paths to the carriers in the set_carrier-folder
        result (object): Result object to analyze results
        dummy_nodes (list): List with the dummy nodes
        export_at_nodes (dict): Dictionary with the amount exported of specific carrier for every node.

    Returns:
        None
    """

    # Get info needed from result object
    reference_year = result.results[None]['system']['reference_year']
    interval_years = result.results[None]['system']['interval_between_years']
    opt_years = result.results[None]['system']['optimized_years']
    years = [reference_year + year for year in range(0, opt_years * interval_years, interval_years)]

    # Define the carrier for every transport technology
    transporttype_to_carrier = {'biomethane_transport': 'biomethane',
                                'carbon_pipeline': 'carbon',
                                'dry_biomass_truck': 'dry_biomass',
                                'hydrogen_pipeline': 'hydrogen',
                                'power_line': 'electricity'}

    # Write the new availability_export files
    for transport_type in transporttype_to_carrier:
        for carrier_path in specific_carrier_path:

            # Specify needed paths
            avail_export_path = os.path.join(carrier_path, 'availability_export.csv')
            avail_export_yearly_path = os.path.join(carrier_path, 'availability_export_yearly_variation.csv')
            avail_export_yearly_path_new = os.path.join(carrier_path, 'availability_export_yearly_new.csv')

            carrier = os.path.basename(carrier_path)
            if carrier == transporttype_to_carrier[transport_type]:

                # Check if 'availability_export.csv'-file exists and read the data
                if os.path.exists(avail_export_path):
                    with open(avail_export_path, 'r', newline='') as input_file:
                        csv_reader = csv.reader(input_file)
                        avail_export = [row for row in csv_reader]
                else:
                    avail_export = []

                # Check if 'availability_export_yearly.csv'-file exists and read the data
                if os.path.exists(avail_export_yearly_path):
                    with open(avail_export_yearly_path, 'r', newline='') as yearly_var_file:
                        csv_reader = csv.reader(yearly_var_file)
                        avail_export_yearly = [row for row in csv_reader]
                else:
                    avail_export_yearly = []

                # If both files exist
                if len(avail_export_yearly) != 0 and len(avail_export) != 0:

                    # Change the already existing (hourly) values to yearly values
                    for export_aval in avail_export:
                        for export_vari in avail_export_yearly:
                            if export_aval[0] == export_vari[0] and avail_export_yearly.index(export_vari) != 0 and avail_export.index(export_aval) != 0:
                                new_val = float(export_aval[-1]) * 8760 * float(export_vari[-1])
                                export_vari[-1] = str(round(new_val, 3))

                    # Get the values from the 'export_at_nodes' dict in the correct structure for the new .csv-file
                    add_dummy_nodes = []
                    for dummynode in dummy_nodes:
                        for index in range(len(export_at_nodes[transport_type][dummynode])):
                            yearly_imp = round(float(export_at_nodes[transport_type][dummynode][index]) * 8760, 3)
                            year = years[index]
                            new_row = [dummynode, year, yearly_imp]
                            add_dummy_nodes.append(new_row)

                    # Combine both data for the new 'availability_export_yearly'-file
                    new_data = avail_export_yearly + add_dummy_nodes
                    with open(avail_export_yearly_path_new, mode="w", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerows(new_data)

                else:

                    with open(avail_export_yearly_path_new, mode="w", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow(['node', 'year', 'availability_export'])

                        realnodes = ['CH', 'FR', 'DE', 'IT', 'AT']
                        for realnode in realnodes:
                            for year in years:
                                writer.writerow([realnode, year, 0])

                        for dummynode in dummy_nodes:
                            for index in range(len(export_at_nodes[transport_type][dummynode])):
                                yearly_imp = round(float(export_at_nodes[transport_type][dummynode][index]), 4)
                                year = years[index]
                                new_row = [dummynode, year, yearly_imp]
                                writer.writerow(new_row)

            else:
                with open(avail_export_yearly_path_new, mode="w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(['node', 'year', 'availability_export'])

                    allnodes = ['CH', 'FR', 'DE', 'IT', 'AT', 'CHdummy', 'FRdummy', 'DEdummy', 'ITdummy', 'ATdummy']
                    for allnode in allnodes:
                        for year in years:
                            writer.writerow([allnode, year, 0])


def create_new_priceimport_file(specific_carrier_path, import_at_nodes):
    """
    This function creates new price_import files for the carriers that are being imported.
    The price is set to 0 for the first implementation

    Parameters:
        specific_carrier_path (list): List with containing the paths to the carriers in the set_carrier-folder
        import_at_nodes (dict): Dictionary with the amount imported of specific carrier for every node.

    Returns:
        None
    """

    # Define the carrier for every transport technology
    transporttype_to_carrier = {'biomethane_transport': 'biomethane',
                                'carbon_pipeline': 'carbon',
                                'dry_biomass_truck': 'dry_biomass',
                                'hydrogen_pipeline': 'hydrogen',
                                'power_line': 'electricity'}

    for transport_type in transporttype_to_carrier:
        for carrier_path in specific_carrier_path:

            carrier = os.path.basename(carrier_path)
            if carrier == transporttype_to_carrier[transport_type]:

                # Define paths to price_import-files
                price_import_file_new = os.path.join(carrier_path, 'price_import_new.csv')
                price_import_file = os.path.join(carrier_path, 'price_import.csv')
                price_import_var_file_new = os.path.join(carrier_path, 'price_import_yearly_variation_new.csv')
                price_import_var_file = os.path.join(carrier_path, 'price_import_yearly_variation.csv')

                # Check if price_import_variation file exists
                if os.path.exists(price_import_var_file):
                    with open(price_import_var_file, 'r', newline='') as input_file:
                        csv_reader = csv.reader(input_file)
                        price_imp_var = [row for row in csv_reader]
                else:
                    price_imp_var = []

                # Check if price_import file exists
                if os.path.exists(price_import_file):
                    with open(price_import_file, 'r', newline='') as input_file:
                        csv_reader = csv.reader(input_file)
                        price_imp = [row for row in csv_reader]
                else:
                    price_imp = []


                # First implementation: set price to 0 if import is available
                nonzero_flow = []
                nonzero_flow_var = []
                for node in import_at_nodes[transport_type]:
                    if not all(element == 0 for element in import_at_nodes[transport_type][node]):

                        if len(price_imp) != 0:
                            if price_imp[0][1] == 'time':
                                nonzero_flow = nonzero_flow + [[node, str(i), '1'] for i in range(0, 8760)]
                            else:
                                nonzero_flow.append([node, '1'])
                        else:
                            if ['node', 'price_import'] not in nonzero_flow:
                                nonzero_flow.append(['node', 'price_import'])
                                nonzero_flow.append([node, '1'])
                            else:
                                nonzero_flow.append([node, '1'])

                        if len(price_imp_var) != 0:
                            for j in range(2040,2051):
                                nonzero_flow_var.append([node, str(j), '0'])
                        else:
                            if ['node', 'year', 'price_import_yearly_variation'] not in nonzero_flow_var:
                                nonzero_flow_var.append(['node', 'year', 'price_import_yearly_variation'])
                                for j in range(2040,2051):
                                    nonzero_flow_var.append([node, str(j), '0'])
                            else:
                                for j in range(2040,2051):
                                    nonzero_flow_var.append([node, str(j), '0'])


                new_prices = price_imp + nonzero_flow
                new_prices_var = price_imp_var + nonzero_flow_var

                # Only create new files if there is actually a flow of the specific carrier. Else, use still the old (normal) ones.
                if len(new_prices) != 0:
                    if os.path.exists(price_import_file_new):
                        os.remove(price_import_file_new)
                    with open(price_import_file_new, 'w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerows(new_prices)

                if len(new_prices_var) != 0:
                    if os.path.exists(price_import_var_file_new):
                        os.remove(price_import_var_file_new)
                    with open(price_import_var_file_new, 'w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerows(new_prices_var)

    return None


def create_new_demand_file(specific_carrier_path, result, export_at_nodes):
    """
        This function creates new demand files for the carriers and dummy nodes in the subproblems.
        This way the export should be enforced.

        Parameters:
            specific_carrier_path (list): List with containing the paths to the carriers in the set_carrier-folder
            result (object): Result object to analyze results
            export_at_nodes (dict): Dictionary with the amount exported of specific carrier for every node.

        Returns:
            None
        """

    # Get info needed from result object
    reference_year = result.results[None]['system']['reference_year']
    interval_years = result.results[None]['system']['interval_between_years']
    opt_years = result.results[None]['system']['optimized_years']
    years = [reference_year + year for year in range(0, opt_years * interval_years, interval_years)]

    # Define the carrier for every transport technology
    transporttype_to_carrier = {'biomethane_transport': 'biomethane',
                                'carbon_pipeline': 'carbon',
                                'dry_biomass_truck': 'dry_biomass',
                                'hydrogen_pipeline': 'hydrogen',
                                'power_line': 'electricity'}

    set_edges_filepath = r'C:\Users\bekaj\Documents\ETH\Master\Masterarbeit\Model_Code\Software\ZEN-garden\data\import_export_baseKopie2\system_specification\set_edges.csv'
    with open(set_edges_filepath, 'r', newline='') as csvfile:
        # Create a CSV reader object
        csv_reader = csv.reader(csvfile)
        edges = [row for row in csv_reader]

    cluster_nodes = result.results[None]['system']['set_cluster_nodes']
    cluster_nodes_new = []
    for i in range(len(cluster_nodes)):
        # List with entries containing 'dummy'
        with_dummy = [item for item in cluster_nodes[i] if 'dummy' in item]
        # List with entries not containing 'dummy'
        without_dummy = [item for item in cluster_nodes[i] if 'dummy' not in item]
        for dummy in with_dummy:
            check_edge = []
            for not_dummy in without_dummy:

                edge = dummy + '-' + not_dummy

                for ex_edge in edges:
                    if edge not in ex_edge:
                        check_edge.append(False)
                    else:
                        check_edge.append(True)

            if True not in check_edge:
                cluster_nodes[i].remove(dummy)
        cluster_nodes_new.append(cluster_nodes[i])


    # Write the new demand files
    for scen in cluster_nodes_new:
        for transport_type in transporttype_to_carrier:
            for carrier_path in specific_carrier_path:

                carrier = os.path.basename(carrier_path)
                if carrier == transporttype_to_carrier[transport_type]:

                    # Specify needed paths
                    demand_file_path_new = os.path.join(carrier_path, 'demand_' + str(cluster_nodes_new.index(scen)) + '.csv')
                    demand_yearly_file_path_new = os.path.join(carrier_path, 'demand_yearly_variation_' + str(cluster_nodes_new.index(scen)) + '.csv')

                    # Default, only hydrogen has a demand file. Do not modify it. Only create new demand files needed for the other carriers
                    if carrier != 'hydrogen':

                        # Get the hourly demand for all the years
                        new_hourly_demand = []
                        for key_node in export_at_nodes[transport_type]:
                            if key_node in scen:
                                new_entry = [key_node, str(1)]
                                new_hourly_demand.append(new_entry)

                        with open(demand_file_path_new, 'w', newline='') as file:
                            writer = csv.writer(file)
                            first_column = ['node', 'demand']
                            writer.writerow(first_column)
                            writer.writerows(new_hourly_demand)

                        # Get the yearly demand variation for all the years
                        new_yearly_demand = []
                        for key_node in export_at_nodes[transport_type]:
                            if key_node in scen:
                                yearly_demand_amount = [0 if yearly_dem == 0 else round(yearly_dem / 8760, 3) for yearly_dem
                                                        in export_at_nodes[transport_type][key_node]]
                                final_entry = [key_node] + yearly_demand_amount
                                new_yearly_demand.append(final_entry)

                        with open(demand_yearly_file_path_new, 'w', newline='') as file:
                            writer = csv.writer(file)
                            first_column = ['node'] + years
                            writer.writerow(first_column)
                            writer.writerows(new_yearly_demand)

    return None

def modifiy_attribute_file(specific_carrier_path):

    # Define the carrier for every transport technology
    transporttype_to_carrier = {'biomethane_transport': 'biomethane',
                                'carbon_pipeline': 'carbon',
                                'dry_biomass_truck': 'dry_biomass',
                                'hydrogen_pipeline': 'hydrogen',
                                'power_line': 'electricity'}

    for transport_type in transporttype_to_carrier:
        for carrier_path in specific_carrier_path:

            carrier = os.path.basename(carrier_path)
            if carrier == transporttype_to_carrier[transport_type]:

    #for carrier_path in specific_carrier_path:

                # Modify all carrier attributes    ### attributes_new.csv is needed for every carrier if defined in scenario dict
                attribute_file = os.path.join(carrier_path, 'attributes.csv')
                attribute_file_new = os.path.join(carrier_path, 'attributes_new.csv')
                with open(attribute_file, 'r') as file:
                    reader = csv.reader(file)
                    data = list(reader)

                new_data = []
                for row in data:
                    if len(row) != 0:
                        if row[0] == 'availability_import_default':
                            row[1] = 'inf'
                        elif row[0] == 'availability_export_default':
                            row[1] = 'inf'
                        elif row[0] == 'availability_import_yearly_default':
                            row[1] = '0'
                        elif row[0] == 'availability_export_yearly_default':
                            row[1] = '0'
                        new_data.append(row)

                # Save the modified data back to new CSV file
                with open(attribute_file_new, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows(new_data)


def create_new_import_files_bayesian(avail_import_data, specific_carrier_path, years, all_nodes, nodes_scenarios):
    """
    This function creates new availability_import_yearly files for the carriers that are being transported.

    Parameters:
        specific_carrier_path (list): List with containing the paths to the carriers in the set_carrier-folder
        result (object): Result object to analyze results
        dummy_nodes (list): List with the dummy nodes
        import_at_nodes (dict): Dictionary with the amount imported of specific carrier for every node.

    Returns:
        None
    """

    # Define the carrier for every transport technology
    transporttype_to_carrier = {'biomethane_transport': 'biomethane',
                                'carbon_pipeline': 'carbon',
                                'dry_biomass_truck': 'dry_biomass',
                                'hydrogen_pipeline': 'hydrogen',
                                'power_line': 'electricity'}

    reference_year = years[0]

    # Base path
    set_carrier_folder = os.path.dirname(specific_carrier_path[0])

    missing_scenarios = []
    for idx, scenario_nodes in enumerate(nodes_scenarios):
        if idx not in avail_import_data:
            missing_scenarios.append(idx)

    for missing_scen in missing_scenarios:
        nodes_involved = nodes_scenarios[missing_scen]

        # Specify needed paths
        for carrier_path in specific_carrier_path:
            avail_import_path = os.path.join(carrier_path, 'availability_import.csv')
            avail_import_yearly_path = os.path.join(carrier_path, 'availability_import_yearly_variation.csv')
            avail_import_yearly_path_new = os.path.join(carrier_path, 'availability_import_yearly_new_' + str(missing_scen) + '.csv')

            # Check if 'availability_import.csv'-file exists and read the data
            if os.path.exists(avail_import_path):
                with open(avail_import_path, 'r', newline='') as input_file:
                    csv_reader = csv.reader(input_file)
                    avail_import = [row for row in csv_reader]
            else:
                avail_import = []

            # Check if 'availability_import_yearly.csv'-file exists and read the data
            if os.path.exists(avail_import_yearly_path):
                with open(avail_import_yearly_path, 'r', newline='') as yearly_var_file:
                    csv_reader = csv.reader(yearly_var_file)
                    avail_import_yearly = [row for row in csv_reader]
            else:
                avail_import_yearly = []

            # Check which countries are contained in the file
            countries_exist = []
            for check_county in avail_import_yearly:
                if check_county[0] != 'node':
                    if check_county[0] not in countries_exist:
                        countries_exist.append(check_county[0])

            countries_to_add = []
            for country in all_nodes:
                if country not in countries_exist:
                    countries_to_add.append(country)

            countries_to_file = []
            for country_file in countries_to_add:
                for year in years:
                    countries_to_file.append([country_file, year, 0])

            # If both files exist
            if len(avail_import_yearly) != 0 and len(avail_import) != 0:

                # Change the already existing (hourly) values to yearly values
                for import_aval in avail_import:
                    for import_vari in avail_import_yearly:
                        if import_aval[0] == import_vari[0] and avail_import_yearly.index(
                                import_vari) != 0 and avail_import.index(import_aval) != 0:
                            new_val = float(import_aval[-1]) * 8760 * float(import_vari[-1])
                            import_vari[-1] = str(round(new_val, 3))


                # Add info for the dummy nodes.
                dummy_nodes = []
                for node in nodes_involved:
                    if 'dummy' in node:
                        for year in years:
                            dummy_nodes.append([node, year, 0])

                # Combine both data for the new 'availability_import_yearly'-file
                new_data = avail_import_yearly + countries_to_file + dummy_nodes
                with open(avail_import_yearly_path_new, mode="w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerows(new_data)

            else:
                if os.path.exists(avail_import_yearly_path_new):
                    os.remove(avail_import_yearly_path_new)

                with open(avail_import_yearly_path_new, mode="w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(['node', 'year', 'availability_import'])

                    for realnode in all_nodes:
                        for year in years:
                            writer.writerow([realnode, year, 0])

                    # Add info for the dummy nodes.
                    dummy_nodes = []
                    for node in nodes_involved:
                        if 'dummy' in node:
                            for year in years:
                                dummy_nodes.append([node, year, 0])

                    for new_row in dummy_nodes:
                        writer.writerow(new_row)




    for scenario in avail_import_data:

        for transport_type in avail_import_data[scenario]:

            carrier = transporttype_to_carrier[transport_type]
            carrier_path = os.path.join(set_carrier_folder, carrier)

            # Specify needed paths
            avail_import_path = os.path.join(carrier_path, 'availability_import.csv')
            avail_import_yearly_path = os.path.join(carrier_path, 'availability_import_yearly_variation.csv')
            avail_import_yearly_path_new = os.path.join(carrier_path, 'availability_import_yearly_new_' + str(scenario) + '.csv')

            # Check if 'availability_import.csv'-file exists and read the data
            if os.path.exists(avail_import_path):
                with open(avail_import_path, 'r', newline='') as input_file:
                    csv_reader = csv.reader(input_file)
                    avail_import = [row for row in csv_reader]
            else:
                avail_import = []

            # Check if 'availability_import_yearly.csv'-file exists and read the data
            if os.path.exists(avail_import_yearly_path):
                with open(avail_import_yearly_path, 'r', newline='') as yearly_var_file:
                    csv_reader = csv.reader(yearly_var_file)
                    avail_import_yearly = [row for row in csv_reader]
            else:
                avail_import_yearly = []

            # Check which countries are contained in the file
            countries_exist = []
            for check_county in avail_import_yearly:
                if check_county[0] != 'node':
                    if check_county[0] not in countries_exist:
                        countries_exist.append(check_county[0])

            countries_to_add = []
            for country in all_nodes:
                if country not in countries_exist:
                    countries_to_add.append(country)

            countries_to_file = []
            for country_file in countries_to_add:
                for year in years:
                    countries_to_file.append([country_file, year, 0])

            # If both files exist
            if len(avail_import_yearly) != 0 and len(avail_import) != 0:

                # Change the already existing (hourly) values to yearly values
                for import_aval in avail_import:
                    for import_vari in avail_import_yearly:
                        if import_aval[0] == import_vari[0] and avail_import_yearly.index(
                                import_vari) != 0 and avail_import.index(import_aval) != 0:
                            new_val = float(import_aval[-1]) * 8760 * float(import_vari[-1])
                            import_vari[-1] = str(round(new_val, 3))

                # Do summation of all available imports ¦¦Get the values from the 'import_at_nodes' dict in the correct structure for the new .csv-file
                add_dummy_nodes = []
                for dummynode in avail_import_data[scenario][transport_type]:
                    for year in avail_import_data[scenario][transport_type][dummynode]:
                        actual_year = int(year) + reference_year
                        value = max(avail_import_data[scenario][transport_type][dummynode][year])
                        entry_to_add = [dummynode, actual_year, value]
                        add_dummy_nodes.append(entry_to_add)

                # Add info for the missing years of the existing dummy nodes in the dict.
                nodes_to_check = []
                for dummynode in avail_import_data[scenario][transport_type]:
                    for idx, year in enumerate(years):
                        if str(idx) not in avail_import_data[scenario][transport_type][dummynode]:
                            nodes_to_check.append([dummynode, year, 0])

                # Add info for the missing dummy nodes.
                other_dummy_nodes = []
                for node in nodes_scenarios[scenario]:
                    if 'dummy' in node and node not in avail_import_data[scenario][transport_type]:
                        for year in years:
                            other_dummy_nodes.append([node, year, 0])


                # Combine both data for the new 'availability_import_yearly'-file
                new_data = avail_import_yearly + countries_to_file + add_dummy_nodes + nodes_to_check + other_dummy_nodes
                with open(avail_import_yearly_path_new, mode="w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerows(new_data)

            else:
                if os.path.exists(avail_import_yearly_path_new):
                    os.remove(avail_import_yearly_path_new)

                with open(avail_import_yearly_path_new, mode="w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(['node', 'year', 'availability_import'])

                    for realnode in all_nodes:
                        for year in years:
                            writer.writerow([realnode, year, 0])

                    # Get the max value from the list
                    add_dummy_nodes = []
                    for dummynode in avail_import_data[scenario][transport_type]:
                        for year in avail_import_data[scenario][transport_type][dummynode]:
                            actual_year = int(year) + reference_year
                            value = max(avail_import_data[scenario][transport_type][dummynode][year])
                            entry_to_add = [dummynode, actual_year, value]
                            add_dummy_nodes.append(entry_to_add)

                    # Add info for the missing years of the existing dummy nodes in the dict.
                    nodes_to_check = []
                    for dummynode in avail_import_data[scenario][transport_type]:
                        for idx, year in enumerate(years):
                            if str(idx) not in avail_import_data[scenario][transport_type][dummynode]:
                                nodes_to_check.append([dummynode, year, 0])

                    # Add info for the missing dummy nodes.
                    other_dummy_nodes = []
                    for node in nodes_scenarios[scenario]:
                        if 'dummy' in node and node not in avail_import_data[scenario][transport_type]:
                            for year in years:
                                other_dummy_nodes.append([node, year, 0])

                    all_dummy_nodes = add_dummy_nodes + nodes_to_check + other_dummy_nodes

                    for new_row in all_dummy_nodes:
                        writer.writerow(new_row)


    for scenario in avail_import_data:
        for carrier_path in specific_carrier_path:

            carrier = os.path.basename(carrier_path)
            active_transports = [transporttype_to_carrier[key_trans] for key_trans in avail_import_data[scenario]]

            if carrier in active_transports:
                continue
            else:
                nodes_involved = nodes_scenarios[scenario]

                avail_import_path = os.path.join(carrier_path, 'availability_import.csv')
                avail_import_yearly_path = os.path.join(carrier_path, 'availability_import_yearly_variation.csv')
                avail_import_yearly_path_new = os.path.join(carrier_path, 'availability_import_yearly_new_' + str(scenario) + '.csv')

                # Check if 'availability_import.csv'-file exists and read the data
                if os.path.exists(avail_import_path):
                    with open(avail_import_path, 'r', newline='') as input_file:
                        csv_reader = csv.reader(input_file)
                        avail_import = [row for row in csv_reader]
                else:
                    avail_import = []

                # Check if 'availability_import_yearly.csv'-file exists and read the data
                if os.path.exists(avail_import_yearly_path):
                    with open(avail_import_yearly_path, 'r', newline='') as yearly_var_file:
                        csv_reader = csv.reader(yearly_var_file)
                        avail_import_yearly = [row for row in csv_reader]
                else:
                    avail_import_yearly = []

                # Check which countries are contained in the file
                countries_exist = []
                for check_county in avail_import_yearly:
                    if check_county[0] != 'node':
                        if check_county[0] not in countries_exist:
                            countries_exist.append(check_county[0])

                countries_to_add = []
                for country in all_nodes:
                    if country not in countries_exist:
                        countries_to_add.append(country)

                countries_to_file = []
                for country_file in countries_to_add:
                    for year in years:
                        countries_to_file.append([country_file, year, 0])

                # If both files exist
                if len(avail_import_yearly) != 0 and len(avail_import) != 0:

                    # Change the already existing (hourly) values to yearly values
                    for import_aval in avail_import:
                        for import_vari in avail_import_yearly:
                            if import_aval[0] == import_vari[0] and avail_import_yearly.index(
                                    import_vari) != 0 and avail_import.index(import_aval) != 0:
                                new_val = float(import_aval[-1]) * 8760 * float(import_vari[-1])
                                import_vari[-1] = str(round(new_val, 3))

                    # Add info for the dummy nodes.
                    dummy_nodes = []
                    for node in nodes_involved:
                        if 'dummy' in node:
                            for year in years:
                                dummy_nodes.append([node, year, 0])

                    # Combine both data for the new 'availability_import_yearly'-file
                    new_data = avail_import_yearly + countries_to_file + dummy_nodes
                    with open(avail_import_yearly_path_new, mode="w", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerows(new_data)

                else:
                    if os.path.exists(avail_import_yearly_path_new):
                        os.remove(avail_import_yearly_path_new)

                    with open(avail_import_yearly_path_new, mode="w", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow(['node', 'year', 'availability_import'])

                        for realnode in all_nodes:
                            for year in years:
                                writer.writerow([realnode, year, 0])

                        # Add info for the dummy nodes.
                        dummy_nodes = []
                        for node in nodes_involved:
                            if 'dummy' in node:
                                for year in years:
                                    dummy_nodes.append([node, year, 0])

                        for new_row in dummy_nodes:
                            writer.writerow(new_row)

    return None


def create_new_demand_files_bayesian(demand_data, set_carrier_folder, years):
    """
        This function creates new demand files for the carriers and dummy nodes in the subproblems.
        This way the export should be enforced.

        Parameters:
            specific_carrier_path (list): List with containing the paths to the carriers in the set_carrier-folder
            result (object): Result object to analyze results
            export_at_nodes (dict): Dictionary with the amount exported of specific carrier for every node.

        Returns:
            None
        """

    # Define the carrier for every transport technology
    transporttype_to_carrier = {'biomethane_transport': 'biomethane',
                                'carbon_pipeline': 'carbon',
                                'dry_biomass_truck': 'dry_biomass',
                                'hydrogen_pipeline': 'hydrogen',
                                'power_line': 'electricity'}

    reference_year = years[0]

    for scenario in demand_data:

        for transport_type in demand_data[scenario]:
            carrier = transporttype_to_carrier[transport_type]
            carrier_path = os.path.join(set_carrier_folder, carrier)

            # Specify needed paths
            demand_file_path_new = os.path.join(carrier_path, 'demand_new_' + str(scenario) + '.csv')
            demand_yearly_file_path_new = os.path.join(carrier_path, 'demand_yearly_variation_new_' + str(scenario) + '.csv')

            # Default, only hydrogen has a demand file. Do not modify it. Only create new demand files needed for the other carriers
            if carrier != 'hydrogen':

                # Get the hourly demand for all the years
                nodes_involved = []
                new_hourly_demand = []
                for key_node in demand_data[scenario][transport_type]:
                    new_entry = [key_node, str(1)]
                    if new_entry not in new_hourly_demand:
                        new_hourly_demand.append(new_entry)
                        nodes_involved.append(key_node)

                with open(demand_file_path_new, 'w', newline='') as file:
                    writer = csv.writer(file)
                    first_column = ['node', 'demand']
                    writer.writerow(first_column)
                    writer.writerows(new_hourly_demand)

                # Get the yearly demand variation for all the years
                new_yearly_demand = []
                for node in demand_data[scenario][transport_type]:
                    node_demand = []
                    for year in demand_data[scenario][transport_type][node]:
                        value_year = float(max(demand_data[scenario][transport_type][node][year]))/8760
                        node_demand.append(value_year)

                    entry_to_add = [node] + node_demand
                    new_yearly_demand.append(entry_to_add)

                with open(demand_yearly_file_path_new, 'w', newline='') as file:
                    writer = csv.writer(file)
                    first_column = ['node'] + years
                    writer.writerow(first_column)
                    writer.writerows(new_yearly_demand)

    return None


def create_new_priceimport_file_bayesian(avail_import_data, set_carrier_folder, years):
    """
    This function creates new price_import files for the carriers that are being imported.
    The price is set to 0 for the first implementation

    Parameters:
        specific_carrier_path (list): List with containing the paths to the carriers in the set_carrier-folder
        import_at_nodes (dict): Dictionary with the amount imported of specific carrier for every node.

    Returns:
        None
    """
    # Define the carrier for every transport technology
    transporttype_to_carrier = {'biomethane_transport': 'biomethane',
                                'carbon_pipeline': 'carbon',
                                'dry_biomass_truck': 'dry_biomass',
                                'hydrogen_pipeline': 'hydrogen',
                                'power_line': 'electricity'}

    for scenario in avail_import_data:

        for transport_type in avail_import_data[scenario]:
            carrier = transporttype_to_carrier[transport_type]
            carrier_path = os.path.join(set_carrier_folder, carrier)

            # Specify needed paths
            avail_import_yearly_path_new = os.path.join(carrier_path, 'availability_import_yearly_new_' + str(scenario) + '.csv')

            # Check if 'availability_import_yearly.csv'-file exists and read the data
            if os.path.exists(avail_import_yearly_path_new):
                with open(avail_import_yearly_path_new, 'r', newline='') as yearly_var_file:
                    csv_reader = csv.reader(yearly_var_file)
                    avail_import_yearly = [row for row in csv_reader]

                avail_dummy = [row[0] for row in avail_import_yearly if 'dummy' in row[0] if row[-1] != str(0)]
                avail_dummy = list(set(avail_dummy))

                # Define paths to price_import-files
                price_import_file_new = os.path.join(carrier_path, 'price_import_new_' + str(scenario) + '.csv')
                price_import_file = os.path.join(carrier_path, 'price_import.csv')
                price_import_var_file_new = os.path.join(carrier_path, 'price_import_yearly_variation_new_' + str(scenario) + '.csv')
                price_import_var_file = os.path.join(carrier_path, 'price_import_yearly_variation.csv')

                # Check if price_import_variation file exists
                if os.path.exists(price_import_var_file):
                    with open(price_import_var_file, 'r', newline='') as input_file:
                        csv_reader = csv.reader(input_file)
                        price_imp_var = [row for row in csv_reader]
                else:
                    price_imp_var = []

                # Check if price_import file exists
                if os.path.exists(price_import_file):
                    with open(price_import_file, 'r', newline='') as input_file:
                        csv_reader = csv.reader(input_file)
                        price_imp = [row for row in csv_reader]
                else:
                    price_imp = []

                nonzero_flow = []
                nonzero_flow_var = []
                for node in avail_dummy:

                    if len(price_imp) != 0:
                        if price_imp[0][1] == 'time':
                            nonzero_flow = nonzero_flow + [[node, str(i), '1'] for i in range(0, 8760)]
                        else:
                            nonzero_flow.append([node, '1'])
                    else:
                        if ['node', 'price_import'] not in nonzero_flow:
                            nonzero_flow.append(['node', 'price_import'])
                            nonzero_flow.append([node, '1'])
                        else:
                            nonzero_flow.append([node, '1'])

                    if len(price_imp_var) != 0:
                        for j in years:
                            nonzero_flow_var.append([node, str(j), '0'])
                    else:
                        if ['node', 'year', 'price_import_yearly_variation'] not in nonzero_flow_var:
                            nonzero_flow_var.append(['node', 'year', 'price_import_yearly_variation'])
                            for j in years:
                                nonzero_flow_var.append([node, str(j), '0'])
                        else:
                            for j in years:
                                nonzero_flow_var.append([node, str(j), '0'])

                new_prices = price_imp + nonzero_flow
                new_prices_var = price_imp_var + nonzero_flow_var

                # Only create new files if there is actually a flow of the specific carrier. Else, use still the old (normal) ones.
                if len(new_prices) != 0:
                    if os.path.exists(price_import_file_new):
                        os.remove(price_import_file_new)
                    with open(price_import_file_new, 'w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerows(new_prices)

                if len(new_prices_var) != 0:
                    if os.path.exists(price_import_var_file_new):
                        os.remove(price_import_var_file_new)
                    with open(price_import_var_file_new, 'w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerows(new_prices_var)


    return None