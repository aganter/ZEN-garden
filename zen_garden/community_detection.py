import os
import csv
import pandas as pd
import networkx as nx

from zen_garden.postprocess.results import Results

def clustering_performance(destination_folder, config):
    """
    Performing the clustering based on the louvain method and updates the config instance
    (with the new clustered nodes).

    Parameters:
        destination_folder (str): Path to the results in the protocol folder.
        config: A config instance used for the run.

    Returns:
        config: Updated config instance.
    """

    # Get path and results object from design calculation
    run_path = config.analysis['dataset']
    res = Results(destination_folder)

    # Get connections
    path_to_edges = os.path.join(run_path, 'energy_system', 'set_edges.csv')
    with open(path_to_edges, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        data_edges = [row for row in csv_reader]
    data_edges_avail = [edge for edge in data_edges if edge[0]!='edge']

    # Initialize graph and add edges
    G = nx.Graph()

    # Get total of all transport technologies for every edge
    transport_techs = res.solution_loader.scenarios['none'].system.set_transport_technologies
    trans_df = []
    for tech in transport_techs:
        df_temp = res.get_total('flow_transport').round(2).loc[tech]
        trans_df.append(df_temp)

    summarized_df = pd.concat(trans_df).groupby(level=0).sum()
    sum_years = summarized_df.sum(axis=1)
    summarized_dict = sum_years.to_dict()

    edges_graph = []
    for edge in data_edges_avail:
        if edge[0] in summarized_dict:
            edges_graph.append((str(edge[1]), str(edge[2]), str(summarized_dict[edge[0]])))
            G.add_edge(str(edge[1]), str(edge[2]), capacity=summarized_dict[edge[0]])
        else:
            edges_graph.append((str(edge[1]), str(edge[2]), str(0)))

    partitions = nx.community.louvain_communities(G, weight='weight')

    # Update the config instance with the new partitions
    cluster_nodes_list = [list(partition) for partition in partitions]
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