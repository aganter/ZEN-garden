import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from zen_garden.postprocess.results import Results
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import csv
from sklearn.preprocessing import StandardScaler


path_to_data = r'C:\Users\bekaj\Documents\ETH\Master\Masterarbeit\Model_Code\Software\ZEN-garden\data\outputs\import_export_baseKopie2_cluster'
res = Results(path_to_data)

# Check nodes available
# Test df
test_df = res.get_total('availability_import').round(3).loc['dry_biomass']
nodes_names = [name for name in test_df.T.columns if 'dummy' not in name]

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
path_to_edges = r'C:\Users\bekaj\Documents\ETH\Master\Masterarbeit\Model_Code\Software\ZEN-garden\data\import_export_baseKopie2_cluster\system_specification\set_edges.csv'
with open(path_to_edges, 'r', newline='') as csvfile:
    csv_reader = csv.reader(csvfile)
    data_edges = [row for row in csv_reader]

test_nodes = []
for test_node in nodes_names:
    for test_edge in data_edges:
        if test_node in test_edge and test_node not in test_nodes:
            test_nodes.append(test_node)

connected_edges_temp = []
for node in test_nodes:
    node_adjacency = []
    for other_node in test_nodes:
        connect = node + '-' + other_node
        check_list = []

        # Check only edges with flow
        if connect in edges_w_flow:
            check_list.append(True)
        else:
            check_list.append(False)

        # for row in data_edges:
        #     if connect in row:
        #         check_list.append(True)
        #     elif connect not in row and node!=other_node:
        #         check_list.append(False)
        #     elif node==other_node:
        #         check_list.append(True)

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
photovoltaic_series = res.get_total('capacity_limit').round(3).loc['photovoltaics']['power']
photovoltaic_df = photovoltaic_series.to_frame()
photovoltaic_df_filter = photovoltaic_df.T[nodes_names].T
X_photovoltaic = photovoltaic_df_filter.values

# Wind onshore capacity
wind_onshore_series = res.get_total('capacity_limit').round(3).loc['wind_onshore']['power']
wind_onshore_df = wind_onshore_series.to_frame()
wind_onshore_df_filter = wind_onshore_df.T[nodes_names].T
X_wind_onshore = wind_onshore_df_filter.values

# Wind onshore capacity
wind_offshore_series = res.get_total('capacity_limit').round(3).loc['wind_offshore']['power']
wind_offshore_df = wind_offshore_series.to_frame()
wind_offshore_df_filter = wind_offshore_df.T[nodes_names].T
X_wind_offshore = wind_offshore_df_filter.values

# Demand hydrogen
demand_df = res.get_total('demand').round(3)
demand_df_filter = demand_df.loc['hydrogen']
demand_df_h2 = demand_df_filter.T[nodes_names].T
X_h2 = demand_df_h2.values



# Get x- and y-coordinates of nodes
path_to_nodes = r'C:\Users\bekaj\Documents\ETH\Master\Masterarbeit\Model_Code\Software\ZEN-garden\data\import_export_baseKopie2_cluster\system_specification\set_nodes.csv'
with open(path_to_nodes, 'r', newline='') as csvfile:
    csv_reader = csv.reader(csvfile)
    data_nodes = [row for row in csv_reader]

node_coordinate = dict()
for node_loc in data_nodes:
    if node_loc[0] in nodes_names:
        x_coord = float(node_loc[1])
        y_coord = float(node_loc[2])
        node_coordinate[node_loc[0]] = [x_coord, y_coord]



# Standardize the data
scaler = StandardScaler()

X_demand_scaled = scaler.fit_transform(X_h2)
X_pv_capacity_scaled = scaler.fit_transform(X_photovoltaic)
X_connected_scaled = scaler.fit_transform(connected_edges)
X_avail_imp_dry_biomass_scaled = scaler.fit_transform(X_avail_imp_dry_biomass)
X_avail_imp_wet_biomass_scaled = scaler.fit_transform(X_avail_imp_wet_biomass)

# Combine standardized features into one matrix
X_combined = np.hstack((X_avail_imp_dry_biomass, connected_edges))


silhouette_scores = []
for num_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_combined)
    silhouette_avg = silhouette_score(X_combined, clusters)
    silhouette_scores.append(silhouette_avg)

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o', linestyle='-')
plt.title('Silhouette Analysis for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.xticks(range(2, 11))
plt.grid(True)
plt.show()

# Compute Calinski-Harabasz scores for different numbers of clusters
calinski_scores = []
for num_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_combined)
    calinski_score = calinski_harabasz_score(X_combined, clusters)
    calinski_scores.append(calinski_score)

# Plot Calinski-Harabasz scores
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), calinski_scores, marker='o', linestyle='-')
plt.title('Calinski-Harabasz Index for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Calinski-Harabasz Score')
plt.xticks(range(2, 11))
plt.grid(True)
plt.show()


# Specify the number of clusters
num_clusters = 5

# Initialize KMeans object
kmeans = KMeans(n_clusters=num_clusters, random_state=42)

# Fit KMeans clustering model
kmeans.fit(X_combined)

# Get cluster labels
cluster_labels = kmeans.labels_
cluster_access = dict()
for name in nodes_names:
    cluster_access[name] = cluster_labels[nodes_names.index(name)]


# Generate random data

x_data = [node_coordinate[key_node][0] for key_node in node_coordinate]
y_data = [node_coordinate[key_node][1] for key_node in node_coordinate]

# num_countries = len(x_data)
# colors = plt.cm.viridis(np.linspace(0, 1, num_countries))

colors = []
colors_matplotlib = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white', 'orange', 'purple', 'brown']

for label in cluster_labels[:37]:
    colors.append(colors_matplotlib[label])


# Create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, color=colors, s=40)

# Add labels to points
for key_node in (node_coordinate):
    plt.text(node_coordinate[key_node][0], node_coordinate[key_node][1], key_node, fontsize=9, ha='right')

# Set plot title and labels
plt.title('Countries Scatter Plot')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')

# Display plot
plt.grid(True)
plt.tight_layout()
plt.show()



x = 0