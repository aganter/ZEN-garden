from zen_garden.postprocess.results import Results
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


data_folder_0 = r'C:\Users\bekaj\Documents\ETH\Master\Masterarbeit\Neuer Ordner\outputs\import_export_base'
data_path_0 = os.path.join(data_folder_0)
res_0 = Results(data_path_0)

df_flow_tr_T = res_0.get_total('flow_transport').round(0).T

# Sum each column
column_sums = df_flow_tr_T.sum()
# Select columns with a sum greater than zero
non_zero_columns = column_sums[column_sums != 0].index
# Create a new DataFrame with only non-zero columns
df_non_zero = df_flow_tr_T[non_zero_columns].T

flow_trans_dry_bio = res_0.get_total('flow_transport').round(1)[0].loc['dry_biomass_truck']

# Creating a for loop to generate variables and save them in a dictionary with specified keys
# Removing the f-string formatting from the right side of the assignments and presenting the corrected code structure

edges = [i for i in flow_trans_dry_bio.index]
node_in = [edge[:2] for edge in edges]
node_out = [edge[3:] for edge in edges]


results = {}
for i in range(11):
    prefix = '' if i == 0 else str(i)
    results[f'flow_transport{prefix}'] = res_0.get_total('flow_transport').round(1)[i].loc['dry_biomass_truck']
    results[f'price_imp_dry_bio{prefix}'] = res_0.get_total('price_import').round(1)[i].loc['dry_biomass']
    results[f'avail_imp_dry_bio{prefix}'] = res_0.get_total('availability_import').round(1)[i].loc['dry_biomass']
    results[f'cap_limit_photovoltaics{prefix}'] = res_0.get_total('capacity_limit').round(1).loc['photovoltaics']['power']
    results[f'cap_limit_wind_offshore{prefix}'] = res_0.get_total('capacity_limit').round(1).loc['wind_offshore']['power']
    results[f'cap_limit_wind_onshore{prefix}'] = res_0.get_total('capacity_limit').round(1).loc['wind_onshore']['power']
    results[f'cap_limit_power_line{prefix}'] = res_0.get_total('capacity_limit').round(1).loc['power_line']['power']
    results[f'capex_spec_trans_hydro_pipeline{prefix}'] = res_0.get_total('capex_specific_transport').round(1)[i].loc['hydrogen_pipeline']
    results[f'capex_spec_trans_power_line{prefix}'] = res_0.get_total('capex_specific_transport').round(1)[i].loc['power_line']
    results[f'carbon_emission_limit_year{prefix}'] = res_0.get_total('carbon_emissions_annual_limit').round(1).loc[i]
    results[f'demand_hydrogen{prefix}'] = res_0.get_total('demand').round(1)[i].loc['hydrogen']
    results[f'distance_trans_dry_biomass{prefix}'] = res_0.get_total('distance').round(1).loc['dry_biomass_truck']
    results[f'max_load_photovoltaics{prefix}'] = res_0.get_total('max_load').round(1)[i].loc['photovoltaics']['power']
    results[f'max_load_wind_offshore{prefix}'] = res_0.get_total('max_load').round(1)[i].loc['wind_offshore']['power']
    results[f'max_load_wind_onshore{prefix}'] = res_0.get_total('max_load').round(1)[i].loc['wind_onshore']['power']
    results[f'price_imp_dry_biomass{prefix}'] = res_0.get_total('price_import').round(1)[i].loc['dry_biomass']
    results[f'price_imp_electricity{prefix}'] = res_0.get_total('price_import').round(1)[i].loc['electricity']
    results[f'price_imp_natural_gas{prefix}'] = res_0.get_total('price_import').round(1)[i].loc['natural_gas']
    print('----------------------')
    print(i)
    print('----------------------')


print('----------------------')
print('Dict ready')
print('----------------------')




# Assuming res_0, node_in, and node_out are defined
# Initialize lists for DataFrame columns
price_import_dry_bio_in, price_import_dry_bio_out = [], []
avail_imp_dry_bio_in, avail_imp_dry_bio_out = [], []
demand_in, demand_out = [], []
max_load_photo_in, max_load_photo_out = [], []
max_load_windon_in, max_load_windon_out = [], []
flow_rate = []

# Loop to populate the lists
for i in range(11):
    prefix = '' if i == 0 else str(i)

    # Assuming you have a way to get 'node_in' and 'node_out' lists
    for node in node_in:
        price_import_dry_bio_in.append(results[f'price_imp_dry_bio{prefix}'].loc[node])
        avail_imp_dry_bio_in.append(results[f'avail_imp_dry_bio{prefix}'].loc[node])
        demand_in.append(results[f'demand_hydrogen{prefix}'].loc[node])
        max_load_photo_in.append(results[f'max_load_photovoltaics{prefix}'].loc[node])
        max_load_windon_in.append(results[f'max_load_wind_onshore{prefix}'].loc[node])

    for node in node_out:
        price_import_dry_bio_out.append(results[f'price_imp_dry_bio{prefix}'].loc[node])
        avail_imp_dry_bio_out.append(results[f'avail_imp_dry_bio{prefix}'].loc[node])
        demand_out.append(results[f'demand_hydrogen{prefix}'].loc[node])
        max_load_photo_out.append(results[f'max_load_photovoltaics{prefix}'].loc[node])
        max_load_windon_out.append(results[f'max_load_wind_onshore{prefix}'].loc[node])

    # Adding flow rate data
    flow_rate = flow_rate + list(results[f'flow_transport{prefix}'])

    print('----------------------')
    print(i)
    print('----------------------')

print('----------------------')
print('Dataframe ready')
print('----------------------')

# Create DataFrame
data = pd.DataFrame({
    'price_import_dry_bio_in': price_import_dry_bio_in,
    'price_import_dry_bio_out': price_import_dry_bio_out,
    'avail_imp_dry_bio_in': avail_imp_dry_bio_in,
    'avail_imp_dry_bio_out': avail_imp_dry_bio_out,
    'demand_in': demand_in,
    'demand_out': demand_out,
    'max_load_photo_in': max_load_photo_in,
    'max_load_photo_out': max_load_photo_out,
    'max_load_windon_in': max_load_windon_in,
    'max_load_windon_out': max_load_windon_out,
    'flow_rate': flow_rate
})

# data.to_csv('results_data.csv', index=False)
# df_loaded = pd.read_csv('sample_data.csv')


# Features and target variable
X = data.drop('flow_rate', axis=1)
y = data['flow_rate']

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Model training
model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Predictions and model evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Loop for the best number
errors = [ mean_squared_error(y_test, preds) for preds in model.staged_predict(X_test)]
# Optimal number of estimators
optimal_num_estimators = np.argmin(errors) + 1

#Plot
plt.figure(figsize=(14, 8))
plt.plot([i for i in range(500)], errors, marker='o')
plt.title('Optimal number of estimators')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.xlim(xmin=0)
plt.ylim(ymin=0, ymax=max(errors)) # max(df_diff_flows[column]) - 0.5*max(df_diff_flows[column])
plt.grid(True)
plt.show()


print(f'Mean Squared Error: {mse:.2f}')
print(f'R^2 Score: {r2:.2f}')

# Retrain with new estimatoe number
opt_num = optimal_num_estimators
model_mod = GradientBoostingRegressor(n_estimators=opt_num, learning_rate=0.1, max_depth=3, random_state=42)
model_mod.fit(X_train, y_train)
y_pred_mod = model_mod.predict(X_test)

# Plotting feature importance
feature_importance = model.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, np.array(X.columns)[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Feature Importance')
plt.show()


# Plotting feature importance
feature_importance = model_mod.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, np.array(X.columns)[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Feature Importance - mod')
plt.show()

y = 0