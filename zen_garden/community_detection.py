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