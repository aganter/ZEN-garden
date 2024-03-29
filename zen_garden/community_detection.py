from zen_garden.postprocess.results import Results
import numpy as np
import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt
import csv
import os

def clustering_performance(destination_folder, config):
    """
    Create files for the calculation of the scenarios (availability_import, demand, price_import)

    Parameters:
        destination_folder (str): Path to the results in the protocol folder.
        config (): A config instance used for the run.

    Returns:
        None
    """

    # Get path and results object from design calculation
    run_path = config.analysis['dataset']
    res = Results(destination_folder)

    # Check nodes available
    # Test df
    test_df = res.get_total('availability_import').round(3).loc['dry_biomass']
    nodes_names = [name for name in test_df.T.columns if 'dummy' not in name]
    #config.system.set_nodes

    # Flow transport
    flow_tran_df_T = res.get_total('flow_transport').round(3).T
    # Remove columns with only zero entries
    zero_columns = flow_tran_df_T.columns[flow_tran_df_T.eq(0).all()]
    flow_tran_df_T_nonzero = flow_tran_df_T.drop(zero_columns, axis=1)

    # Check nodes where there is a flow transport and add them to list
    edges_w_flow = []
    for column_flow in flow_tran_df_T_nonzero.columns:
        if column_flow[1] not in edges_w_flow:
            edges_w_flow.append(column_flow[1])



    # Get connections
    path_to_edges = os.path.join(run_path, 'energy_system', 'set_edges.csv') #r'C:\Users\bekaj\Documents\ETH\Master\Masterarbeit\Model_Code\Software\ZEN-garden\data\HSC_solution_alogrithm_community\energy_system\set_edges.csv'
    with open(path_to_edges, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        data_edges = [row for row in csv_reader]

    # Analyze only edges that are present in the run case
    data_edges_avail = [edge for edge in data_edges if edge[1] in nodes_names if edge[2] in nodes_names]

    test_nodes = []
    for test_node in nodes_names:
        for test_edge in data_edges_avail:
            if test_node in test_edge and test_node not in test_nodes:
                test_nodes.append(test_node)

    connected_edges_temp = []
    for node in test_nodes:
        node_adjacency = []
        for other_node in test_nodes:
            connect = node + '-' + other_node
            check_list = []

            # Check only edges with flow
            # if connect in edges_w_flow:
            #     check_list.append(True)
            # else:
            #     check_list.append(False)

            for row in data_edges_avail:
                if connect in row:
                    check_list.append(True)
                elif connect not in row and node!=other_node:
                    check_list.append(False)
                elif node==other_node:
                    check_list.append(True)

            if True in check_list:
                node_adjacency.append(1)
            else:
                node_adjacency.append(0)
        connected_edges_temp.append(node_adjacency)

    connected_edges = np.array(connected_edges_temp)

    nodes_names = test_nodes


    # Availability import
    avail_imp_df = res.get_total('availability_import').round(3)

    # Availability import dry_biomass
    avail_imp_dry_biomass_df = avail_imp_df.loc['dry_biomass']
    avail_imp_dry_biomass_raw = avail_imp_dry_biomass_df.T[nodes_names].T
    X_avail_imp_dry_biomass = avail_imp_dry_biomass_raw.values

    # Availability imprt wet_biomass
    avail_imp_wet_biomass_df = avail_imp_df.loc['wet_biomass']
    avail_imp_wet_biomass_raw = avail_imp_wet_biomass_df.T[nodes_names].T
    X_avail_imp_wet_biomass = avail_imp_wet_biomass_raw.values

    # Photovoltaics capacity
    photovoltaic_series = res.get_total('capacity_limit').round(3).loc['photovoltaics'].loc['power']
    # photovoltaic_df = photovoltaic_series.to_frame()
    photovoltaic_df_filter = photovoltaic_series.T[nodes_names].T
    X_photovoltaic = photovoltaic_df_filter.values

    # Wind onshore capacity
    wind_onshore_series = res.get_total('capacity_limit').round(3).loc['wind_onshore'].loc['power']
    # wind_onshore_df = wind_onshore_series.to_frame()
    wind_onshore_df_filter = wind_onshore_series.T[nodes_names].T
    X_wind_onshore = wind_onshore_df_filter.values

    # Wind onshore capacity
    wind_offshore_series = res.get_total('capacity_limit').round(3).loc['wind_offshore'].loc['power']
    # wind_offshore_df = wind_offshore_series.to_frame()
    wind_offshore_df_filter = wind_offshore_series.T[nodes_names].T
    X_wind_offshore = wind_offshore_df_filter.values

    # Demand hydrogen
    demand_df = res.get_total('demand').round(3)
    demand_df_filter = demand_df.loc['hydrogen']
    demand_df_h2 = demand_df_filter.T[nodes_names].T

    dict_from_df = demand_df_h2.to_dict()
    dict_end = dict_from_df[0]  # Für Jahr 0 momentan

    # Sample graph, adding edges
    G = nx.Graph()

    trans_dict = res.get_total('flow_transport').round(2).loc['biomethane_transport'].to_dict()[0]
    edges_graph = []
    for edge in data_edges_avail:
        if edge[0] in trans_dict:
            edges_graph.append((str(edge[1]), str(edge[2]), str(trans_dict[edge[0]])))
            G.add_edge(str(edge[1]), str(edge[2]), capacity=trans_dict[edge[0]])
        else:
            edges_graph.append((str(edge[1]), str(edge[2]), str(0)))

    partition = community_louvain.best_partition(G, weight='weight')

    # # Visualization (optional)
    # pos = nx.spring_layout(G)
    # cmap = plt.get_cmap('viridis')
    # nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=700,
    #                        cmap=cmap, node_color=list(partition.values()))
    # nx.draw_networkx_edges(G, pos, alpha=0.5)
    # nx.draw_networkx_labels(G, pos)
    # plt.show()



    # G.add_edges_from(edges_graph)

    # Perform the minimum cut
    # cut_value, partition = nx.stoer_wagner(G)
    #
    # # partition gives you two sets of nodes representing the two clusters
    # print("Minimum cut value:", cut_value)
    # print("Partition:", partition)

    # hydrogen consumption for each node
    # H2_consumption = dict_end
    #
    # # Normalize hydrogen consumption for similarity calculation
    # max_consumption = max(H2_consumption.values())
    # normalized_consumption = {node: consumption / max_consumption for node, consumption in H2_consumption.items()}
    #
    # # Adjust edge weights based on hydrogen consumption similarity
    # for edge in G.edges():
    #     node1, node2 = edge
    #     consumption_similarity = 1 - abs(normalized_consumption[node1] - normalized_consumption[node2])
    #     G[edge[0]][edge[1]]['weight'] = consumption_similarity
    #
    # # Community detection with the modified graph
    # partition = community_louvain.best_partition(G, weight='weight')

    # # Visualization (optional)
    # pos = nx.spring_layout(G)
    # cmap = plt.get_cmap('viridis')
    # nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=700,
    #                        cmap=cmap, node_color=list(partition.values()))
    # nx.draw_networkx_edges(G, pos, alpha=0.5)
    # nx.draw_networkx_labels(G, pos)
    # plt.show()

    cluster_nodes_dict = dict()
    for key_node in partition:
        if partition[key_node] in cluster_nodes_dict:
            cluster_nodes_dict[partition[key_node]].append(key_node)
        else:
            cluster_nodes_dict[partition[key_node]] = []
            cluster_nodes_dict[partition[key_node]].append(key_node)

    cluster_nodes_list = [cluster_nodes_dict[key_cluster] for key_cluster in cluster_nodes_dict]

    config.system.set_cluster_nodes = cluster_nodes_list

    return config





# import networkx as nx
# import community as community_louvain
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Sample graph
# G = nx.Graph()
# G.add_edges_from([('A', 'B'), ('A', 'E'), ('A', 'D'),
#                   ('B', 'A'), ('B', 'E'), ('B', 'D'), ('B', 'C'),
#                   ('C', 'E'), ('C', 'B'), ('C', 'D'),
#                   ('D', 'A'), ('D', 'B'), ('D', 'B'),
#                   ('E', 'A'), ('E', 'B'), ('E', 'C')])  # Add your edges
#
# # Electricity consumption for each node
# electricity_consumption = {'A': 1000, 'B': 100, 'C': 110, 'D': 1100, 'E': 120}
#
# # Normalize electricity consumption for similarity calculation
# max_consumption = max(electricity_consumption.values())
# normalized_consumption = {node: consumption / max_consumption for node, consumption in electricity_consumption.items()}
#
# # Adjust edge weights based on electricity consumption similarity
# for edge in G.edges():
#     node1, node2 = edge
#     consumption_similarity = 1 - abs(normalized_consumption[node1] - normalized_consumption[node2])
#     G[edge[0]][edge[1]]['weight'] = consumption_similarity
#
# # Community detection with the modified graph
# partition = community_louvain.best_partition(G, weight='weight')
#
# # Visualization (optional)
# pos = nx.spring_layout(G)
# cmap = plt.get_cmap('viridis')
# nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=700,
#                        cmap=cmap, node_color=list(partition.values()))
# nx.draw_networkx_edges(G, pos, alpha=0.5)
# nx.draw_networkx_labels(G, pos)
# plt.show()
#
# print("Partition considering electricity consumption:", partition)
#


# import networkx as nx
# import community as community_louvain
# import matplotlib.pyplot as plt
# import csv
#
#
# pathedge = r'C:\Users\bekaj\Documents\ETH\Master\Masterarbeit\Model_Code\Software\ZEN-garden\data\HSC_solution_alogrithm_community\energy_system\set_edges.csv'
# with open(pathedge, 'r', newline='') as input_file:
#     csv_reader = csv.reader(input_file)
#     edges = [row for row in csv_reader]
#
# edges_graph = [(str(edge[1]), str(edge[2])) for edge in edges if 'edge' not in edge [0]]
#
# # Create a graph
# G = nx.Graph()
# G.add_edges_from(edges_graph)  # Example edges
#
# # Assuming you have a dictionary of electricity consumption for each node
# electricity_consumption = {'A': 100, 'B': 100, 'C': 100, 'D': 100, 'E': 100, 'F': 100}
#
# # Here, we simply use the Louvain method based on structural properties
# partition = community_louvain.best_partition(G)
#
# # Visualization (optional)
# pos = nx.spring_layout(G)
# cmap = plt.get_cmap('viridis')
# nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=700,
#                        cmap=cmap, node_color=list(partition.values()))
# nx.draw_networkx_edges(G, pos, alpha=0.5)
# nx.draw_networkx_labels(G, pos)
# plt.show()
#
# print("Partition (without considering electricity consumption):", partition)




# import networkx as nx
# import csv
#
#
# pathedge = r'C:\Users\bekaj\Documents\ETH\Master\Masterarbeit\Model_Code\Software\ZEN-garden\data\HSC_solution_alogrithm_community\energy_system\set_edges.csv'
# with open(pathedge, 'r', newline='') as input_file:
#     csv_reader = csv.reader(input_file)
#     edges = [row for row in csv_reader]
#
# edges_graph = [(str(edge[1]), str(edge[2])) for edge in edges if 'edge' not in edge [0]]
#
#
# # Example for an undirected graph
# G_undirected = nx.Graph()
# G_undirected.add_edges_from([('A', 'B'), ('B', 'C'), ('D', 'E')])  # Add more edges as needed
#
# connected_components = list(nx.connected_components(G_undirected))
# print("Connected components (Undirected):", connected_components)
#
# # Example for a directed graph
# G_directed = nx.DiGraph()
# G_directed.add_edges_from(edges_graph)  # Add more edges as needed
#
# strongly_connected_components = list(nx.strongly_connected_components(G_directed))
# print("Strongly connected components (Directed):", strongly_connected_components)
#
# c = 0