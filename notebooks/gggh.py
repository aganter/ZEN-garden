import numpy as np
from sklearn.cluster import KMeans
import networkx as nx
from sklearn.preprocessing import StandardScaler
import time
import logging
import pandas as pd

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Configure logging
actual_flows_path = r'C:\Users\bekaj\Documents\ETH\Master\Masterarbeit\euler_calcs\bayesian_prot_19032024\protocol_actual_flows.log'
imp_dem_path = r'C:\Users\bekaj\Documents\ETH\Master\Masterarbeit\euler_calcs\bayesian_prot_19032024\protocol_attr.log'
costs_path = r'C:\Users\bekaj\Documents\ETH\Master\Masterarbeit\euler_calcs\bayesian_prot_19032024\protocol_costs.log'
diff_flows_path = r'C:\Users\bekaj\Documents\ETH\Master\Masterarbeit\euler_calcs\bayesian_prot_19032024\protocol_diff_flows.log'


df_act_flows = pd.read_csv(actual_flows_path, delimiter=':')
df_imp_dem = pd.read_csv(imp_dem_path, delimiter=':')
df_costs = pd.read_csv(costs_path, delimiter=':')
df_diff_flows = pd.read_csv(diff_flows_path, delimiter=':')

# Cost plots
for column in df_costs.columns:
    plt.figure()
    plt.plot(df_costs.index, df_costs[column], marker='o')
    plt.title(column)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.xlim(xmin=0)
    plt.ylim(ymin=0, ymax=5000)
    plt.grid(True)
    plt.show()

# Difference flows plots
for column in df_diff_flows.columns:
    plt.figure(figsize=(14, 8))
    plt.plot(df_diff_flows.index, df_diff_flows[column], marker='o')
    plt.title(column)
    plt.xlabel('Iteration')
    plt.ylabel('Difference in flow')
    plt.xlim(xmin=0)
    plt.ylim(ymin=0, ymax=1500) # max(df_diff_flows[column]) - 0.5*max(df_diff_flows[column])
    plt.grid(True)
    plt.show()

y = 0

# Import and demand values
# cols_checked = []
# for column in df_imp_dem.columns:
#     for col_com in df_imp_dem.columns:
#
#         if column.rsplit('.', 1)[0] == col_com.rsplit('.', 1)[0] and column.split('.')[-1] != col_com.split('.')[-1] and column not in cols_checked and column not in cols_checked:
#
#             plt.figure(figsize=(14, 8))
#             plt.plot(df_imp_dem.index, df_imp_dem[column], marker='o', label=column)
#             plt.plot(df_imp_dem.index, df_imp_dem[col_com], marker='o', label=col_com)
#             plt.title(column.rsplit('.', 1)[0])
#             plt.xlabel('Iteration')
#             plt.ylabel('Value of import and demand')
#             plt.xlim(xmin=0)
#             plt.ylim(ymin=0, ymax=max(max(df_imp_dem[column]), max(df_imp_dem[col_com])) + 0.1*max(max(df_imp_dem[column]), max(df_imp_dem[col_com]))) # max(df_diff_flows[column]) - 0.5*max(df_diff_flows[column])
#             plt.grid(True)
#             plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
#             plt.show()
#
#             cols_checked.append(column)
#             cols_checked.append(col_com)

# Actual flows
# cols_checked = []
# for column in df_act_flows.columns:
#     for col_com in df_act_flows.columns:
#
#         if column.rsplit('.', 1)[0] == col_com.rsplit('.', 1)[0] and column.split('.')[-1] != col_com.split('.')[-1] and column not in cols_checked and column not in cols_checked:
#
#             plt.figure(figsize=(14, 8))
#             plt.plot(df_act_flows.index, df_act_flows[column], marker='o', label=column)
#             plt.plot(df_act_flows.index, df_act_flows[col_com], marker='o', label=col_com)
#             plt.title(column.rsplit('.', 1)[0])
#             plt.xlabel('Iteration')
#             plt.ylabel('Value of import and demand')
#             plt.xlim(xmin=0)
#             plt.ylim(ymin=0, ymax=max(max(df_act_flows[column]), max(df_act_flows[col_com])) + 0.1*max(max(df_act_flows[column]), max(df_act_flows[col_com]))) # max(df_diff_flows[column]) - 0.5*max(df_diff_flows[column])
#             plt.grid(True)
#             plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
#             plt.show()
#
#             cols_checked.append(column)
#             cols_checked.append(col_com)

y=2

# # Generating a synthetic dataset
# np.random.seed(42)  # For reproducible results
# n_samples = 100
# production_capacity = np.random.uniform(100, 400, n_samples)
# demand = np.random.uniform(50, 350, n_samples)
# connection_capacity = np.random.uniform(80, 300, n_samples)
# price_factor = np.random.uniform(0.8, 1.2, n_samples)
# regulatory_factor = np.random.choice([0.9, 1.0, 1.1], n_samples)
#
# # Assuming a complex relationship between the features and the target variable
# flow_rate = (production_capacity * 0.6 + demand * 0.4) * connection_capacity * price_factor * regulatory_factor + np.random.normal(0, 10, n_samples)
#
# # Creating a DataFrame
# data = pd.DataFrame({
#     'production_capacity': production_capacity,
#     'demand': demand,
#     'connection_capacity': connection_capacity,
#     'price_factor': price_factor,
#     'regulatory_factor': regulatory_factor,
#     'flow_rate': flow_rate
# })
#
# # Features and target variable
# X = data.drop('flow_rate', axis=1)
# y = data['flow_rate']
#
# # Splitting the dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Model training
# model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
# model.fit(X_train, y_train)
#
# # Predictions and model evaluation
# y_pred = model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
#
# print(f'Mean Squared Error: {mse:.2f}')
# print(f'R^2 Score: {r2:.2f}')
#
# # Plotting feature importance
# feature_importance = model.feature_importances_
# sorted_idx = np.argsort(feature_importance)
# pos = np.arange(sorted_idx.shape[0]) + .5
# plt.barh(pos, feature_importance[sorted_idx], align='center')
# plt.yticks(pos, np.array(X.columns)[sorted_idx])
# plt.xlabel('Relative Importance')
# plt.title('Feature Importance')
# plt.show()

x = 0
#
#
# # Configure logging
# file_path=r'C:\Users\bekaj\Documents\ETH\Master\Masterarbeit\Model_Code\Software\ZEN-garden\notebooks\protocol_results\protocol_attr.log'
#
#
# df = pd.read_csv(file_path, delimiter=':')
#
#
# # Sample electricity consumption data for each country over 3 years
# # This data is represented as a 2D array where each row represents a country and each column represents a year
# electricity_consumption = np.array([[180, 184, 182],
#                                     [180, 190, 180],
#                                     [180, 185, 181],
#                                     [182, 182, 185]])
#
# # Sample adjacency matrix representing the connectivity between countries
# # This is a symmetric matrix where each entry represents the connectivity between two countries
# # For simplicity, let's assume higher values indicate stronger connectivity
# adjacency_matrix = np.array([[1, 0, 0, 1],
#                              [0, 1, 0, 0],
#                              [0, 0, 1, 1],
#                              [1, 0, 1, 1]])
#
# # Calculating the clustering based on electricity consumption and connectivity
# # Reshape electricity consumption to have one row per country and one column per year
# X_elec = electricity_consumption.reshape(len(electricity_consumption), -1)
#
# scaler = StandardScaler()
#
# X_demand_scaled = scaler.fit_transform(X_elec)
# X_connected_scaled = scaler.fit_transform(adjacency_matrix)
#
# # Concatenate the adjacency matrix with the reshaped electricity consumption data
# # X = np.hstack((X_elec, adjacency_matrix))
# X = np.hstack((X_demand_scaled, X_connected_scaled))
#
# # Define the number of clusters
# n_clusters = 2
#
# # Perform KMeans clustering
# kmeans = KMeans(n_clusters=n_clusters, random_state=42)
# kmeans.fit(X)
#
# # Get cluster labels
# cluster_labels = kmeans.labels_
#
# # Print the clustering results
# print("Country Clusters:")
# for country, cluster_label in enumerate(cluster_labels):
#     print(f"Country {country + 1}: Cluster {cluster_label + 1}")
#
# # Visualize the graph
# x = 0

# import matplot