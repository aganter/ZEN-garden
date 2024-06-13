import os
import pandas as pd
import chardet
import geopandas as gpd
from shapely.geometry import Point
import numpy as np

def generate_existing_capacities(file_demand, lifetime, technology, max_load):
    capacity_existing = pd.read_csv(file_demand)

    capacity_existing['year_construction'] = np.nan
    new_rows = []
    capacity_existing['capacity_existing'] = capacity_existing['demand'] / max_load / (lifetime - 1)

    for index, row in capacity_existing.iterrows():
        for year in range(0, lifetime - 1):
            new_row = row.copy()
            new_row['year_construction'] = 2023 - year
            new_rows.append(new_row)

    expanded_df = pd.DataFrame(new_rows)
    expanded_df['capacity_existing'] = expanded_df['capacity_existing'].round(4)

    expanded_df = expanded_df.drop(['demand'], axis=1)
    expanded_df.to_csv(f'../data/hard_to_abate/set_technologies/set_conversion_technologies/{technology}/capacity_existing_seconds.csv', index=False)

    return expanded_df

def generate_existing_capacity_ASU(file_demand, lifetime, technology, max_load):
    capacity_existing = pd.read_csv(file_demand)

    capacity_existing['year_construction'] = np.nan
    new_rows = []

    capacity_existing['capacity_existing'] = capacity_existing['demand'] * 0.823 / max_load / (lifetime - 1)

    for index, row in capacity_existing.iterrows():
        for year in range(0, lifetime - 1):
            new_row = row.copy()
            new_row['year_construction'] = 2023 - year
            new_rows.append(new_row)

    expanded_df = pd.DataFrame(new_rows)
    expanded_df['capacity_existing'] = expanded_df['capacity_existing'].round(4)

    expanded_df = expanded_df.drop(['demand'], axis=1)
    expanded_df.to_csv(f'../data/hard_to_abate/set_technologies/set_conversion_technologies/{technology}/capacity_existing_seconds.csv', index=False)

    return expanded_df

def generate_existing_capacities_steel(file_demand, lifetime_BF_BOF, lifetime_EAF, lifetime_DRI, max_load_BF_BOF, max_load_EAF, max_load_DRI):
    capacity_existing = pd.read_csv(file_demand)

    capacity_existing['year_construction'] = np.nan
    capacity_existing_BF_BOF = capacity_existing.copy()
    capacity_existing_BF_BOF['capacity_existing'] = capacity_existing_BF_BOF['demand'] * 0.7 / max_load_BF_BOF / (lifetime_BF_BOF - 1)

    new_rows = []

    for index, row in capacity_existing_BF_BOF.iterrows():
        for year in range(0, lifetime_BF_BOF - 1):
            new_row = row.copy()
            new_row['year_construction'] = 2023 - year
            new_rows.append(new_row)

    expanded_df_BF_BOF = pd.DataFrame(new_rows)
    expanded_df_BF_BOF['capacity_existing'] = expanded_df_BF_BOF['capacity_existing'].round(4)

    expanded_df_BF_BOF = expanded_df_BF_BOF.drop(['demand'], axis=1)
    expanded_df_BF_BOF.to_csv(
        f'../data/hard_to_abate/set_technologies/set_conversion_technologies/BF_BOF/capacity_existing_seconds.csv', index=False)

    capacity_existing_EAF = capacity_existing.copy()
    capacity_existing_EAF['capacity_existing'] = capacity_existing_EAF['demand'] * 0.3 / max_load_EAF / (lifetime_EAF -1)

    new_rows = []

    for index, row in capacity_existing_EAF.iterrows():
        for year in range(0, lifetime_EAF - 1):
            new_row = row.copy()
            new_row['year_construction'] = 2023 - year
            new_rows.append(new_row)

    expanded_df_EAF = pd.DataFrame(new_rows)
    expanded_df_EAF['capacity_existing'] = expanded_df_EAF['capacity_existing'].round(4)

    expanded_df_EAF = expanded_df_EAF.drop(['demand'], axis=1)

    expanded_df_EAF.to_csv(
        f'../data/hard_to_abate/set_technologies/set_conversion_technologies/EAF/capacity_existing_seconds.csv', index=False)

    capacity_existing_DRI = capacity_existing.copy()
    capacity_existing_DRI['capacity_existing'] = capacity_existing_EAF['demand'] * 0.22 / max_load_DRI / (lifetime_EAF - 1)

    new_rows = []

    for index, row in capacity_existing_DRI.iterrows():
        for year in range(0, lifetime_DRI - 1):
            new_row = row.copy()
            new_row['year_construction'] = 2023 - year
            new_rows.append(new_row)

    expanded_df_DRI = pd.DataFrame(new_rows)
    expanded_df_DRI['capacity_existing'] = expanded_df_DRI['capacity_existing'].round(4)

    expanded_df_DRI = expanded_df_DRI.drop(['demand'], axis=1)
    expanded_df_DRI.to_csv(
        f'../data/hard_to_abate/set_technologies/set_conversion_technologies/DRI/capacity_existing_seconds.csv',
        index=False)



def adjust_year(year, lifetime):
    if year == 0:
        return year
    while year + lifetime <= 2024:
        year += lifetime
    return year

def existing_capacities_hydrogen(file_path):
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read(100000))
        encoding = result['encoding']

    existing_capa = pd.read_csv(file_path, delimiter=';', encoding=encoding)
    existing_capa_filtered = existing_capa[['eigl_text_lon', 'eigl_text_lat', 'start_year', 'eigl_process', 'capacity_mwe']].copy()

    existing_capa_filtered['geometry'] = [Point(xy) for xy in zip(existing_capa_filtered.eigl_text_lon, existing_capa_filtered.eigl_text_lat)]
    geo_df = gpd.GeoDataFrame(existing_capa_filtered, geometry='geometry')
    geo_df.set_crs(epsg=4326, inplace=True)

    nuts2_regions = gpd.read_file('nuts_data/NUTS_RG_20M_2021_4326.shp')
    nuts2_regions = nuts2_regions.to_crs(epsg=4326)

    joined = gpd.sjoin(geo_df, nuts2_regions, how="inner", predicate='intersects')

    filtered_df = joined[joined['LEVL_CODE'] == 2]
    filtered_df = filtered_df[['NUTS_ID', 'start_year', 'eigl_process', 'capacity_mwe']]

    filtered_df['start_year'] = filtered_df['start_year'].fillna('0')
    filtered_df['start_year'] = pd.to_numeric(filtered_df['start_year'], errors='coerce')
    filtered_df = filtered_df[filtered_df['start_year'] <= 2024]
    filtered_df['start_year'] = filtered_df['start_year'].astype(int)

    filtered_df['capacity_mwe'] = filtered_df['capacity_mwe'] / 1000 # in GW

    filtered_df = filtered_df.rename(columns={'NUTS_ID': 'node', 'start_year': 'year_construction', 'capacity_mwe': 'capacity_existing'})

    electrolysis_capacity = filtered_df[filtered_df['eigl_process'].str.contains('electrolysis', case=False, na=False)]
    electrolysis_capacity = electrolysis_capacity.drop('eigl_process', axis=1)
    electrolysis_capacity['year_construction'] = electrolysis_capacity['year_construction'].apply(lambda year: adjust_year(year, lifetime=10))
    electrolysis_capacity = electrolysis_capacity.groupby(['node', 'year_construction'], as_index=False).agg({'capacity_existing': 'sum'})
    electrolysis_capacity.to_csv('../data/hard_to_abate/set_technologies/set_conversion_technologies/electrolysis/capacity_existing.csv', index=False)

    SMR_capacity = filtered_df[filtered_df['eigl_process'].str.contains('other or unknown', case=False, na=False)]
    SMR_capacity = SMR_capacity.drop('eigl_process', axis=1)
    SMR_capacity.to_csv('../data/hard_to_abate/set_technologies/set_conversion_technologies/SMR/capacity_existing.csv', index=False)

    SMR_CCS_capacity = filtered_df[filtered_df['eigl_process'].str.contains('CCS', case=False, na=False)]
    SMR_CCS_capacity = SMR_CCS_capacity.drop('eigl_process', axis=1)
    SMR_CCS_capacity['year_construction'] = SMR_CCS_capacity['year_construction'].apply(lambda year: adjust_year(year, lifetime=25))
    SMR_CCS_capacity = SMR_CCS_capacity.groupby(['node', 'year_construction'], as_index=False).agg({'capacity_existing': 'sum'})
    SMR_CCS_capacity.to_csv('../data/hard_to_abate/set_technologies/set_conversion_technologies/SMR_CCS/capacity_existing.csv', index=False)


if __name__ == "__main__":

    industries = ['ammonia', 'methanol', 'oil_products', 'cement',
                   ]

    industry_data = {
        'ammonia': {
            'lifetime': 30,
            'technology': 'haber_bosch',
            'max_load': 0.95
        },
        'steel': {
            'lifetime': 40,
            'technology': 'BF_BOF',
            'max_load': 0.95
        },
        'methanol': {
            'lifetime': 25,
            'technology': 'methanol_synthesis',
            'max_load': 0.95
        },
        'oil_products': {
            'lifetime': 30,
            'technology': 'refinery',
            'max_load': 0.95
        },
        'cement': {
            'lifetime': 25,
            'technology': 'cement_plant',
            'max_load': 0.8
        }
    }

    for industry in industries:
        data = industry_data[industry]
        lifetime = data['lifetime']
        technology = data['technology']
        max_load = data['max_load']
        file_demand = f"../data/hard_to_abate/set_carriers/{industry}/demand1.csv"
        generate_existing_capacities(file_demand, lifetime, technology, max_load)

    file_demand = "../data/hard_to_abate/set_carriers/ammonia/demand1.csv"
    lifetime = 30
    technology = 'ASU'
    max_load = 1.0
    #generate_existing_capacity_ASU(file_demand, lifetime, technology, max_load)

    file_demand = "../data/hard_to_abate/set_carriers/steel/demand1.csv"
    lifetime_BF_BOF = 40
    max_load_BF_BOF = 0.95
    lifetime_EAF = 25
    max_load_EAF = 0.9
    lifetime_DRI = 25
    max_load_DRI = 0.9
    #generate_existing_capacities_steel(file_demand, lifetime_BF_BOF, lifetime_EAF, lifetime_DRI, max_load_BF_BOF, max_load_EAF, max_load_DRI)

    file_path_hydrogen = "existing_capacities_input/existing_capacity_hydrogen.csv"
    #existing_capacities_hydrogen(file_path_hydrogen)


