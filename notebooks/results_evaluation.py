import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib
from zen_garden.postprocess.results import Results
import seaborn as sns
import os
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.ticker as ticker
import plotly.io as pio
import folium
from folium import plugins
import geopandas as gpd
import json
import webbrowser
import tempfile
import matplotlib.patches as mpatches
from matplotlib.patches import Wedge
from folium.plugins import MarkerCluster
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
import networkx as nx
from shapely.geometry import Point
from functools import partial
from functools import partial
import pyproj
from shapely.ops import transform
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter

plt.rcParams.update({'font.size': 22})
res_scenario = Results("../outputs/hard_to_abate_scenarios_070524/")

emissions_limit = res_scenario.get_total("carbon_emissions_annual_limit").xs("scenario_")
print(emissions_limit)
emissions = res_scenario.get_total("carbon_emissions_annual").xs("scenario_")
print(emissions)
def get_emissions(scenario):

    df_emissions = res_scenario.get_df("carbon_emissions_cumulative")
    df_emissions_scenario = df_emissions[scenario]
    print(df_emissions_scenario)

def save_total(folder_path, scenario):
    os.makedirs(folder_path, exist_ok=True)

    flow_conversion_input = res_scenario.get_total("flow_conversion_input")
    scenario_data_input = flow_conversion_input.xs(scenario)
    scenario_data_input = scenario_data_input.groupby(['technology', 'carrier']).sum()
    file_path = os.path.join(folder_path, f"flow_conversion_input_{scenario}.csv")
    scenario_data_input.to_csv(file_path)

    flow_conversion_output = res_scenario.get_total("flow_conversion_output")
    scenario_data_output = flow_conversion_output.xs(scenario)
    scenario_data_output = scenario_data_output.groupby(['technology', 'carrier']).sum()
    file_path = os.path.join(folder_path, f"flow_conversion_output_{scenario}.csv")
    scenario_data_output.to_csv(file_path)

def save_imports_exports(folder_path, scenario):

    os.makedirs(folder_path, exist_ok=True)
    imports = res_scenario.get_total("flow_import")
    imports = imports.xs(scenario)
    imports_grouped = imports.groupby(['carrier']).sum()
    file_path = os.path.join(folder_path, f"imports_grouped_{scenario}.csv")
    imports_grouped.to_csv(file_path)

    exports = res_scenario.get_total("flow_export")
    exports = exports.xs(scenario)
    exports_grouped = exports.groupby(['carrier']).sum()
    file_path = os.path.join(folder_path, f"exports_grouped_{scenario}.csv")
    exports_grouped.to_csv(file_path)

def energy_carrier(scenario):

    inputs = res_scenario.get_total("flow_conversion_input").xs(scenario).reset_index()

    outputs = res_scenario.get_total("flow_conversion_output").xs(scenario).reset_index()

    inputs['node'] = inputs['node'].str.slice(0, 2)
    inputs_grouped = inputs.groupby(['technology', 'carrier', 'node']).sum().reset_index()
    energy_carriers_inputs = ['coal', 'coal_for_cement', 'natural_gas', 'electricity', 'dry_biomass',
                        'biomethane', 'biomass_cement']
    energy_carriers_outputs = ['coal', 'coal_for_cement', 'natural_gas', 'electricity', 'dry_biomass',
                              'biomethane', 'biomass_cement']

    inputs_grouped = inputs_grouped[inputs_grouped['carrier'].isin(energy_carriers_inputs)]

    outputs['node'] = outputs['node'].str.slice(0, 2)
    outputs_grouped = outputs.groupby(['technology', 'carrier', 'node']).sum().reset_index()
    outputs_grouped = outputs_grouped[outputs_grouped['carrier'].isin(energy_carriers_outputs)]

    natural_gas = outputs_grouped[(outputs_grouped['technology'].isin(['biomethane_conversion', 'h2_to_ng']))]
    natural_gas = natural_gas.drop('technology', axis=1)
    summed_natural_gas = natural_gas.groupby(['node', 'carrier']).sum().reset_index()

    coal = outputs_grouped[(outputs_grouped['technology'].isin(['scrap_conversion_BF_BOF', 'BF_BOF_CCS']))]
    coal = coal.drop('technology', axis=1)
    summed_coal = coal.groupby(['node', 'carrier']).sum().reset_index()

    electricity = outputs_grouped[(outputs_grouped['technology'].isin(['SMR', 'SMR_CCS', 'gasification_methanol', 'pv_ground', 'wind_offshore', 'wind_onshore']))]
    electricity = electricity.drop('technology', axis=1)
    summed_electricity = electricity.groupby(['node', 'carrier']).sum().reset_index()

    coal_for_cement = outputs_grouped[(outputs_grouped['technology'].isin(['hydrogen_for_cement_conversion', 'biomass_to_coal_conversion']))]
    coal_for_cement = coal_for_cement.drop('technology', axis=1)
    summed_coal_for_cement = coal_for_cement.groupby(['node', 'carrier']).sum().reset_index()

    renewables = outputs_grouped[(outputs_grouped['technology'].isin(['pv_ground', 'wind_offshore', 'wind_onshore']))]
    renewables = renewables.drop('technology', axis=1)
    summed_renewables = renewables.groupby(['node', 'carrier']).sum().reset_index()
    summed_renewables.loc[summed_renewables['carrier'] == 'electricity', 'carrier'] = 'renewable_electricity'

    #for i in range(1, 14):
    for i in range(1, 27):
        summed_renewables[i] = summed_renewables[i].round(4)

    inputs_grouped_new = inputs_grouped.drop('technology', axis=1)
    inputs_grouped_new = inputs_grouped_new.groupby(['node', 'carrier']).sum().reset_index()

    dfs = {
        'inputs_grouped_new': inputs_grouped_new,
        'coal': summed_coal,
        'natural_gas': summed_natural_gas,
        'electricity': summed_electricity,
        'coal_for_cement': summed_coal_for_cement
    }

    for df_name, df in dfs.items():
        if df_name != 'inputs_grouped_new':
            filtered_df = df[(df['carrier'].isin(inputs_grouped_new['carrier'])) & (df['node'].isin(inputs_grouped_new['node']))]

            for index, row in filtered_df.iterrows():
                carrier = row['carrier']
                node = row['node']

                #inputs_grouped_new.loc[(inputs_grouped_new['carrier'] == carrier) & (inputs_grouped_new['node'] == node), 0:13] -= row[0:13]
                inputs_grouped_new.loc[(inputs_grouped_new['carrier'] == carrier) & (inputs_grouped_new['node'] == node), 0:26] -= row[0:26]

    #inputs_grouped_new[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]] = inputs_grouped_new[
     #   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]].apply(pd.to_numeric, errors='coerce')

    inputs_grouped_new[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]] = inputs_grouped_new[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]].apply(pd.to_numeric, errors='coerce')

    #for i in range(1, 14):
    for i in range(1, 27):
        inputs_grouped_new[i] = inputs_grouped_new[i].round(4)

    energy_carrier_country = pd.concat([inputs_grouped_new, summed_renewables])

    energy_carrier = energy_carrier_country.groupby(['carrier']).sum().reset_index()
    energy_carrier = energy_carrier.drop(['node'], axis=1)
    #energy_carrier.loc[energy_carrier['carrier'] == 'coal', 0:13] /= (27.35/3.6)
    energy_carrier.loc[energy_carrier['carrier'] == 'coal', 0:26] /= (27.35 / 3.6)
    energy_carrier_country = energy_carrier_country.fillna(0)

    return energy_carrier_country
def draw_wedges_on_map(folder_path, shapefile_path, year, scenario, radius_factor, figsize=(13, 15)):

    df = energy_carrier(scenario)
    df['production'] = df[year]

    gdf = gpd.read_file(shapefile_path).to_crs('EPSG:3035')
    level = [0]
    gdf = gdf[gdf['LEVL_CODE'].isin(level)]

    country_color = 'ghostwhite'
    border_color = 'dimgrey'

    countries_to_exclude = ['IS', 'TR']
    gdf = gdf[~gdf['CNTR_CODE'].isin(countries_to_exclude)]

    carrier_color_map = {
        'biomass_cement': 'forestgreen',
        'biomethane': 'darkkhaki',
        'coal': 'lightgray',
        'coal_for_cement': 'lightgrey',
        'dry_biomass': 'olivedrab',
        'natural_gas': 'dimgrey',
        'electricity': 'teal',
        'renewable_electricity': 'cyan'
    }

    france_centroid = (3713381.55, 2686876.92)

    plt.rcParams['hatch.color'] = 'grey'
    plt.rcParams['hatch.linewidth'] = 0.4

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    gdf.plot(ax=ax, color=country_color, edgecolor=border_color)

    for country_code in ['NO', 'UK', 'CH']:
        specific_gdf = gdf[gdf['CNTR_CODE'] == country_code]
        specific_gdf.plot(ax=ax, facecolor='lightgrey', hatch="\\\\\\", edgecolor='dimgrey', linewidth=0.8)

    legend_patches = []
    for node, group in df.groupby('node'):
        total_production = group['production'].sum()
        if total_production <= 0:
            continue


        radius = np.sqrt(total_production) * radius_factor * 100000

        if node == 'FR':
            centroid = france_centroid
        else:
            if node not in gdf['NUTS_ID'].values:
                print(f"Keine geografischen Daten für {node} gefunden.")
                continue
            country_geom = gdf.loc[gdf['NUTS_ID'] == node, 'geometry'].iloc[0]
            centroid = (country_geom.centroid.x, country_geom.centroid.y)

        start_angle = 0
        for _, row in group.iterrows():
            carrier = row['carrier']
            production = row['production']
            if production <= 0:
                continue

            angle = (production / total_production) * 360
            color = carrier_color_map.get(carrier, 'white')

            wedge = Wedge(centroid, radius, start_angle, start_angle + angle,
                          edgecolor='black', facecolor=color, linewidth=0.8)
            ax.add_patch(wedge)

            start_angle += angle

    ax.set_ylim(1.5e6, 5.5e6)
    ax.set_xlim(2.5e6, 5.9e6)

    plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.axis('off')
    plt.savefig(f"{folder_path}/map_energy_carrier_{scenario}_{year}.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()


def update_multiple_combinations(df_inputs, df_outputs, combinations):
    df_outputs_updated = df_outputs.copy()
    df_inputs_updated = df_inputs.copy()

    for tech_output, carrier_output, tech_input, carrier_input in combinations:
        output_indices = df_outputs_updated[
            (df_outputs_updated['technology'] == tech_output) & (df_outputs_updated['carrier'] == carrier_output)].index
        input_indices = df_inputs_updated[
            (df_inputs_updated['technology'] == tech_input) & (df_inputs_updated['carrier'] == carrier_input)].index

        if not output_indices.empty and not input_indices.empty:
            for output_idx in output_indices:
                for input_idx in input_indices:
                    #df_outputs_updated.loc[output_idx, range(14)] -= df_inputs_updated.loc[input_idx, range(14)].values
                    df_outputs_updated.loc[output_idx, range(27)] -= df_inputs_updated.loc[input_idx, range(27)].values

        output_indices = df_outputs_updated[
            (df_outputs_updated['technology'] == tech_input) & (df_outputs_updated['carrier'] == carrier_input)].index

        input_indices = df_inputs[
            (df_inputs_updated['technology'] == tech_output) & (df_inputs_updated['carrier'] == carrier_output)].index

        if not output_indices.empty and not input_indices.empty:
            #output_row = df_outputs_updated.loc[output_indices[0], range(14)]
            output_row = df_outputs_updated.loc[output_indices[0], range(27)]

            for idx in input_indices:
                #df_inputs_updated.loc[idx, range(14)] -= output_row.values
                df_inputs_updated.loc[idx, range(27)] -= output_row.values

    return df_outputs_updated, df_inputs_updated

def subtract_technology_pairs(df, tech_base, carrier_base, tech_compare, carrier_compare):
    idx_base = df[(df['technology'] == tech_base) & (df['carrier'] == carrier_base)].index
    idx_compare = df[(df['technology'] == tech_compare) & (df['carrier'] == carrier_compare)].index

    if not idx_base.empty and not idx_compare.empty:
        #df.loc[idx_base, range(1, 14)] -= df.loc[idx_compare, range(1, 14)].values
        df.loc[idx_base, range(1, 27)] -= df.loc[idx_compare, range(1, 27)].values

    return df

def generate_sankey_diagram(folder_path, scenario, target_technologies, intermediate_technologies, year, title, save_file):

    scenario_name_mapping = {
        'scenario_': 'Baseline Scenario',
        'scenario_high_demand': 'High Demand Scenario',
        'scenario_low_demand': 'Low Demand Scenario'
    }
    scenario_name = scenario_name_mapping.get(scenario, scenario)

    inputs_df = res_scenario.get_total("flow_conversion_input").xs(scenario).reset_index()
    inputs_df = inputs_df.drop('node', axis=1)
    inputs_df = inputs_df.groupby(['technology', 'carrier']).sum().reset_index()

    outputs_df = res_scenario.get_total("flow_conversion_output").xs(scenario).reset_index()
    outputs_df = outputs_df.drop('node', axis=1)
    outputs_df = outputs_df.groupby(['technology', 'carrier']).sum().reset_index()

    combination = [
        ('biomethane_DRI', 'natural_gas', 'DRI', 'natural_gas'),
        ('biomethane_haber_bosch', 'natural_gas', 'haber_bosch', 'natural_gas'),
        ('biomethane_SMR', 'natural_gas', 'SMR', 'natural_gas'),
        ('biomethane_SMR_CCS', 'natural_gas', 'SMR_CCS', 'natural_gas'),
        ('h2_to_ng', 'natural_gas', 'DRI', 'natural_gas'),
        ('scrap_conversion_EAF', 'direct_reduced_iron', 'EAF', 'direct_reduced_iron'),
        ('scrap_conversion_BF_BOF', 'coal', 'BF_BOF', 'coal'),
        ('scrap_conversion_BF_BOF', 'limestone', 'BF_BOF', 'limestone'),
        ('scrap_conversion_BF_BOF', 'iron_ore', 'BF_BOF', 'iron_ore'),
        ('hydrogen_for_cement_conversion', 'coal_for_cement', 'cement_plant', 'coal_for_cement'),
        ('biomass_to_coal_conversion', 'coal_for_cement', 'cement_plant', 'coal_for_cement'),
        ('carbon_conversion', 'carbon_methanol', 'methanol_synthesis', 'carbon_methanol')
    ]

    updated_outputs_df, updated_inputs_df = update_multiple_combinations(df_inputs = inputs_df, df_outputs = outputs_df, combinations = combination)

    technology_pairs = [
        ('DRI', 'natural_gas', 'biomethane_DRI', 'biomethane'),
        ('haber_bosch', 'natural_gas', 'biomethane_haber_bosch', 'biomethane'),
        ('SMR', 'natural_gas', 'biomethane_SMR', 'biomethane'),
        ('SMR_CCS', 'natural_gas', 'biomethane_SMR_CCS', 'biomethane'),
        ('DRI', 'natural_gas', 'h2_to_ng', 'hydrogen'),
        ('EAF', 'direct_reduced_iron', 'scrap_conversion_EAF', 'scrap'),
        #('BF_BOF', 'coal', 'scrap_conversion_BF_BOF', 'scrap'),
        #('BF_BOF', 'iron_ore', 'scrap_conversion_BF_BOF', 'scrap'),
        #('BF_BOF', 'limestone', 'scrap_conversion_BF_BOF', 'scrap'),
        ('cement_plant', 'coal_for_cement', 'hydrogen_for_cement_conversion', 'hydrogen'),
        ('cement_plant', 'coal_for_cement', 'biomass_to_coal_conversion', 'biomass_cement'),
        ('methanol_synthesis', 'carbon_methanol', 'carbon_conversion', 'carbon')

    ]

    for tech_base, carrier_base, tech_compare, carrier_compare in technology_pairs:
        updated_inputs_df = subtract_technology_pairs(updated_inputs_df, tech_base, carrier_base, tech_compare, carrier_compare)

    updated_inputs_df['technology'] = updated_inputs_df['technology'].str.replace('h2_to_ng', 'DRI')
    updated_inputs_df['technology'] = updated_inputs_df['technology'].str.replace('scrap_conversion_EAF', 'EAF')
    updated_inputs_df['technology'] = updated_inputs_df['technology'].str.replace('scrap_conversion_BF_BOF', 'BF_BOF')
    updated_inputs_df['technology'] = updated_inputs_df['technology'].str.replace('hydrogen_for_cement_conversion', 'cement_plant')
    updated_inputs_df['technology'] = updated_inputs_df['technology'].str.replace('biomass_to_coal_conversion',
                                                                                  'cement_plant')
    updated_inputs_df['technology'] = updated_inputs_df['technology'].str.replace('carbon_conversion',
                                                                                  'methanol_synthesis')
    technologies_to_remove = ['biomethane_SMR', 'biomethane_SMR_CCS', 'biomethane_haber_bosch', 'biomethane_DRI', 'scrap_conversion_EAF',
                              'hydrogen_for_cement_conversion', 'biomass_to_coal_conversion', 'carbon_conversion']
    updated_outputs_df = updated_outputs_df[~updated_outputs_df['technology'].isin(technologies_to_remove)]
    updated_inputs_df['technology'] = updated_inputs_df['technology'].str.replace('biomethane_', '')

    print(updated_inputs_df)

    input_techs_target = updated_inputs_df[updated_inputs_df['technology'].isin(target_technologies)]

    input_techs_intermediate = updated_inputs_df[updated_inputs_df['technology'].isin(intermediate_technologies)]

    output_techs_intermediate = updated_outputs_df[updated_outputs_df['technology'].isin(intermediate_technologies)]

    output_techs_target = updated_outputs_df[updated_outputs_df['technology'].isin(target_technologies)]

    input_sankey_target = pd.DataFrame({
        'source': input_techs_target['carrier'],
        'target': input_techs_target['technology'],
        'value': input_techs_target[year]
    })

    input_sankey_intermediate = pd.DataFrame({
        'source': input_techs_intermediate['carrier'],
        'target': input_techs_intermediate['technology'],
        'value': input_techs_intermediate[year]
    })

    output_sankey_intermediate = pd.DataFrame({
        'source': output_techs_intermediate['technology'],
        'target': output_techs_intermediate['carrier'],
        'value': output_techs_intermediate[year]
    })

    output_sankey_target = pd.DataFrame({
        'source': output_techs_target['technology'],
        'target': output_techs_target['carrier'],
        'value': output_techs_target[year]
    })

    links = pd.concat([input_sankey_target, output_sankey_intermediate, input_sankey_intermediate, output_sankey_target], axis=0)

    unique_source_target = list(pd.unique(links[['source', 'target']].values.ravel('K')))
    mapping_dict = {k:v for v, k in enumerate(unique_source_target)}
    inv_mapping_dict = {v: k for k, v in mapping_dict.items()}
    links['source'] = links['source'].map(mapping_dict)
    links['target'] = links['target'].map(mapping_dict)

    links_dict = links.to_dict(orient="list")

    color_mapping = {
        'steel': 'royalblue',
        'steel_BF_BOF': 'steelblue',
        'steel_DRI_EAF': 'skyblue',
        'scrap': 'cornflowerblue',
        'steel_inputs': 'mediumblue',
        'cement': 'darkslateblue',
        'ammonia': 'darkorchid',
        'hydrogen': 'sandybrown',
        'SMR': 'goldenrod',
        'SMR_CCS': 'darkgoldenrod',
        'methanol': 'fuchsia',
        'gasification_methanol': 'violet',
        'methanol_from_hydrogen': 'palevioletred',
        'refining': 'indianred',
        'electricity': 'aqua',
        'CCS': 'gainsboro',
        'other_techs': 'red',
        'natural_gas': 'dimgrey',
        'wet_biomass': 'yellowgreen',
        'dry_biomass': 'olivedrab',
        'biomass_cement': 'forestgreen',
        'biomethane': 'darkkhaki',
        'coal': 'gray',
        'electrolysis': 'darksalmon',
        'gasification': 'wheat',
        'gasification_CCS': 'peru',
        'nitrogen': 'mediumorchid',
        'oxygen': 'darkslateblue',
        'BF_BOF_CCS': 'lightsteelblue',
        'BF_BOF_CCS': 'lightsteelblue',
        'e_haber_bosch': 'darkviolet',
        'haber_bosch': 'mediumorchid',
        'ASU': 'mediumpurple',
        'fossil_fuel': 'black',
        'biomass': 'darkseagreen',
        'default_color': 'green'

    }

    category_mapping = {
        'BF_BOF': 'steel_BF_BOF',
        'EAF': 'steel_DRI_EAF',
        'DRI': 'steel_DRI_EAF',
        'scrap': 'scrap',
        'iron_ore': 'steel_inputs',
        'limestone': 'steel_inputs',
        'h2_to_ng': 'steel_inputs',
        'scrap_conversion_EAF': 'scrap',
        'scrap_conversion_BF_BOF': 'scrap',
        'steel': 'steel',
        'coal': 'coal',
        'natural_gas': 'natural_gas',
        'biomethane': 'biomethane',
        'biomethane_conversion': 'biomethane',
        'electricity': 'electricity',
        'direct_reduced_iron': 'steel_DRI_EAF',
        'carbon_liquid': 'CCS',
        'ASU': 'ASU',
        'DAC': 'CCS',
        'SMR': 'SMR',
        'SMR_CCS': 'SMR_CCS',
        'anaerobic_digestion': 'biomethane',
        'electrolysis': 'electrolysis',
        'gasification': 'gasification',
        'gasification_CCS': 'gasification_CCS',
        'wet_biomass': 'wet_biomass',
        'dry_biomass': 'dry_biomass',
        'hydrogen': 'hydrogen',
        'nitrogen': 'nitrogen',
        'oxygen': 'oxygen',
        'carbon': 'CCS',
        'oil_products': 'refining',
        'methanol': 'methanol',
        'ammonia': 'ammonia',
        'BF_BOF_CCS': 'BF_BOF_CCS',
        'carbon_liquefication': 'CCS',
        'carbon_removal': 'CCS',
        'carbon_storage': 'CCS',
        'e_haber_bosch': 'e_haber_bosch',
        'haber_bosch': 'haber_bosch',
        'gasification_methanol': 'gasification_methanol',
        'methanol_from_hydrogen': 'methanol_from_hydrogen',
        'methanol_synthesis': 'methanol',
        'refinery': 'refining',
        'cement_plant': 'cement',
        'cement_plant_oxy_combustion': 'CCS',
        'cement_plant_post_comb': 'CCS',
        'photovoltaics': 'electricity',
        'pv_ground': 'electricity',
        'pv_rooftop': 'electricity',
        'wind_offshore': 'electricity',
        'wind_onshore': 'electricity',
        'coal_for_cement': 'fossil_fuel',
        'biomass_cement': 'biomass',
        'biomass_to_coal_conversion': 'biomass',
        'cement': 'cement',
        'hydrogen_for_cement_conversion': 'hydrogen',
        'carbon_evaporation': 'CCS',
        'hydrogen_compressor_low': 'hydrogen',
        'hydrogen_decompressor': 'hydrogen',
        'carbon_methanol': 'methanol',
        'SMR_methanol': 'methanol',
        'gasification_methanol_h2': 'methanol',
        'carbon_conversion': 'methanol',
        'carbon_methanol_conversion': 'methanol',
        'biomethane_SMR': 'hydrogen',
        'biomethane_SMR_CCS': 'hydrogen',
        'biomethane_SMR_methanol': 'methanol',
        'biomethane_haber_bosch':'ammonia'

    }

    for cat in category_mapping.values():
        if cat not in color_mapping:
            print(f"Missing color mapping for category: {cat}")

    colors = [color_mapping.get(category_mapping.get(tech, 'other_techs')) for tech in unique_source_target]

    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "sankey"}]])
    #print([src for src in links['source']])
    fig.add_trace(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            #line=dict(color='black', width=0.5),
            label=unique_source_target,
            color=colors
        ),
        link=dict(
            #color=["rgba"+str(matplotlib.colors.to_rgba(color_mapping.get(category_mapping.get(inv_mapping_dict[link], 'other_techs')), alpha=0.6)) for link in links['source']],
            color=[
                "rgba" + str(matplotlib.colors.to_rgba(
                    color_mapping.get(category_mapping.get(inv_mapping_dict[link], 'other_techs'), 'default_color'),
                    # Use a default color if not found
                    alpha=0.6
                )) for link in links['source']
            ],
            #[str(matplotlib.colors.to_rgba(color_mapping[links_dict['source'][str(src)]])).replace("0.6", "1.0") for src in links_dict['source']],
            source=links_dict['source'],
            target=links_dict['target'],
            value=links_dict['value'],
            #text=[f"Value: {link}" for link in links_dict['value']],
            #hoverinfo='all'
            label=[f"{source} to {target}" for source, target in zip(links_dict['source'], links_dict['target'])],
            hovertemplate='%{value}'
        )
    ))

    if isinstance(year, str):
        year = int(year)
    #displayed_year = year * 2 + 2024
    displayed_year = year + 2024
    fig.update_layout(title_text=f"{title} {displayed_year} ({scenario_name})", font_size=18)

    fig.update_layout(font=dict(size=26))

    if save_file:
        subfolder_name = "sankey"
        subfolder_path = os.path.join(folder_path, subfolder_name)
        os.makedirs(subfolder_path, exist_ok=True)
        png_file_path = os.path.join(subfolder_path, f"{target_technologies}_{displayed_year}_{scenario}.png")

        fig.write_image(png_file_path, format="png", width=1600, height=1200)

    fig.show()


def plot_outputs(folder_path, scenario, carrier, save_file):

    df_output = res_scenario.get_total("flow_conversion_output").xs(scenario).reset_index()
    df_output = df_output.drop("node", axis=1)
    df_output = df_output.groupby(["technology", "carrier"]).sum().reset_index()

    df_input = res_scenario.get_total("flow_conversion_input").xs(scenario).reset_index()
    df_input = df_input.drop("node", axis=1)
    df_input = df_input.groupby(["technology", "carrier"]).sum().reset_index()

    filtered_rows = df_input[(df_input['technology'] == 'biomethane_SMR') & (df_input['carrier'] == 'biomethane')]
    filtered_rows_1 = df_input[(df_input['technology'] == 'biomethane_SMR_CCS') & (df_input['carrier'] == 'biomethane')]

    #columns_to_multiply = [year for year in range(0, 14)]
    columns_to_multiply = [year for year in range(0, 27)]
    filtered_rows[columns_to_multiply] = filtered_rows[columns_to_multiply].apply(lambda x: x / 1.2987)
    filtered_rows_1[columns_to_multiply] = filtered_rows_1[columns_to_multiply].apply(lambda x: x / 1.2987)

    new_row = filtered_rows[columns_to_multiply].mean().to_dict()
    new_row['technology'] = 'SMR_biomethane'
    new_row['carrier'] = 'hydrogen'

    df_output = pd.concat([df_output, pd.DataFrame([new_row])], ignore_index=True)

    new_row_1 = filtered_rows_1[columns_to_multiply].mean().to_dict()
    new_row_1['technology'] = 'SMR_CCS_biomethane'
    new_row_1['carrier'] = 'hydrogen'

    df_output = pd.concat([df_output, pd.DataFrame([new_row_1])], ignore_index=True)

    smr_hydrogen_row = df_output[(df_output['technology'] == 'SMR_biomethane') & (df_output['carrier'] == 'hydrogen')]
    #smr_values = smr_hydrogen_row.iloc[0][[i for i in range(14)]].values
    smr_values = smr_hydrogen_row.iloc[0][[i for i in range(27)]].values

    smr_biomethane_hydrogen_row = df_output[
        (df_output['technology'] == 'SMR') & (df_output['carrier'] == 'hydrogen')]
    smr_biomethane_hydrogen_index = smr_biomethane_hydrogen_row.index[0]

    #for i in range(14):
    for i in range(27):
        df_output.at[smr_biomethane_hydrogen_index, i] -= smr_values[i]

    smr_hydrogen_row = df_output[(df_output['technology'] == 'SMR_CCS_biomethane') & (df_output['carrier'] == 'hydrogen')]
    #smr_values = smr_hydrogen_row.iloc[0][[i for i in range(14)]].values
    smr_values = smr_hydrogen_row.iloc[0][[i for i in range(27)]].values

    smr_biomethane_hydrogen_row = df_output[
        (df_output['technology'] == 'SMR_CCS') & (df_output['carrier'] == 'hydrogen')]
    smr_biomethane_hydrogen_index = smr_biomethane_hydrogen_row.index[0]

    #for i in range(14):
    for i in range(27):
        df_output.at[smr_biomethane_hydrogen_index, i] -= smr_values[i]

    df_emissions_cumulative = res_scenario.get_total("carbon_emissions_cumulative").xs(scenario).reset_index()
    #df_emissions_cumulative['year'] = df_emissions_cumulative['year'].apply(lambda x: 2024 + 2 * x)
    df_emissions_cumulative['year'] = df_emissions_cumulative['year'].apply(lambda x: 2024 + x)

    grouped_df = df_output[df_output['carrier'] == carrier]

    #year_mapping = {i : (2024 + 2 * i) for i in range(14)}
    year_mapping = {i: (2024 + i) for i in range(27)}
    grouped_df.rename(columns=year_mapping, inplace=True)
    grouped_df.set_index('technology', inplace=True)
    grouped_df = grouped_df.dropna()
    grouped_df_values = grouped_df.drop(['carrier'], axis=1).transpose()

    desired_order = [
        'SMR', 'SMR_biomethane', 'SMR_CCS', 'SMR_CCS_biomethane',
        'electrolysis', 'gasification', 'gasification_CCS'
    ]

    available_technologies = [tech for tech in desired_order if tech in grouped_df_values.columns]

    grouped_df_values = grouped_df_values[available_technologies]

    if carrier == 'hydrogen':
        technology_colors = {
            'SMR': 'darkgray',
            'SMR_CCS': 'deeppink',
            'electrolysis': 'lime',
            'gasification': 'darkturquoise',
            'gasification_CCS': 'cyan',
            'SMR_biomethane': 'lightgrey',
            'SMR_CCS_biomethane': 'pink'
        }
        #palette = [technology_colors[tech] for tech in grouped_df_values.columns if tech in technology_colors]
    #else:
    #palette = sns.color_palette(n_colors=len(grouped_df_values.columns))

        filtered_technologies = [tech for tech in grouped_df_values.columns if tech in technology_colors]
        grouped_df_values = grouped_df_values[filtered_technologies]

        palette = [technology_colors[tech] for tech in grouped_df_values.columns]

    else:
        #palette = ['gray'] * len(grouped_df_values.columns)
        palette = plt.cm.tab20(np.linspace(0, 1, len(grouped_df_values.columns)))

    fig, ax1 = plt.subplots(figsize=(18, 10))
    ax2 = ax1.twinx()

    bottom = np.zeros(len(grouped_df_values))

    for idx, technology in enumerate(grouped_df_values.columns):
        ax1.bar(grouped_df_values.index, grouped_df_values[technology], bottom=bottom, color=palette[idx], label=technology, width=0.6)
        bottom += grouped_df_values[technology].values

    ax2.plot(df_emissions_cumulative['year'], df_emissions_cumulative[scenario], color='black', label='Cumulative Emissions', marker='o')
    emissions_budget = 2815967  # Put carbon budget [kt] here
    ax2.axhline(y=emissions_budget, color='red', linestyle='--', label='Emissions Budget')
    # Legends and adjustments
    ax1.legend(title='Technology', bbox_to_anchor=(1.08, 1), loc='upper left', frameon=False)
    ax2.legend(loc='upper left', bbox_to_anchor=(1.08, 0.535), frameon=False)

    plt.subplots_adjust(right=0.75)
    ax1.set_xlabel("Year", fontsize=22)

    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0, fontsize=14)

    y_labels_right = [f"{int(label / 1000)}" for label in ax2.get_yticks()]
    ax2.set_yticklabels(y_labels_right)

    ax2.set_ylabel("cumulative Emissions [Mt CO2]", fontsize=22)
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    ax1.set_ylabel("Yearly Hydrogen Production [TWh]" if carrier in ['hydrogen', 'electricity', 'natural_gas', 'biomethane'] else "Yearly Production [Mt]", fontsize=22)

    #ax1.set_ylim([0, 850000])
    #ax1.set_yticks(range(0, 850001, 100000))

    y_labels_left = [f"{label / 1000}" for label in ax1.get_yticks()]
    ax1.set_yticklabels(y_labels_left)

    if save_file:
        subfolder_name = "output_bar_charts"
        subfolder_path = os.path.join(folder_path, subfolder_name)
        os.makedirs(subfolder_path, exist_ok=True)
        png_file_path = os.path.join(subfolder_path, f"{carrier}_bar_chart_{scenario}.png")
        plt.savefig(png_file_path, bbox_inches='tight')

    plt.show()

def plot_outputs_carbon(scenario1, scenario2):

    df1 = res_scenario.get_total("flow_conversion_input", scenario=scenario1).xs('carbon_storage')
    grouped_df1 = df1.xs('carbon_liquid').sum()

    df2 = res_scenario.get_total("flow_conversion_input", scenario=scenario2).xs('carbon_storage')
    grouped_df2 = df2.xs('carbon_liquid').sum()

    years = list(range(2024, 2051, 2))

    bar_width = 0.45
    years1 = [x - bar_width / 2 for x in range(len(years))]
    years2 = [x + bar_width / 2 for x in range(len(years))]

    plt.figure(figsize=(12, 7))

    plt.bar(years1, grouped_df1.values, width=bar_width, color='cadetblue', label='Baseline')
    plt.bar(years2, grouped_df2.values, width=bar_width, color='lightskyblue', label='Biomass_high_price_10')

    plt.xlabel('Year')
    plt.ylabel('Stored carbon [Mt]')

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    formatter = FuncFormatter(lambda y, _: '{:,.0f}'.format(y / 1000))
    ax.yaxis.set_major_formatter(formatter)

    plt.xticks(range(len(years)), years)
    plt.legend()

    plt.axhline(y=82817, color='r', linestyle='--', linewidth=2)

    plt.show()

def plot_dataframe_on_map(df, df_input, technology_column, output_columns, shapefile_path=None, save_png=True):

    #columns = ['node', 'technology', 'carrier'] + [year for year in range(0, 14)]
    columns = ['node', 'technology', 'carrier'] + [year for year in range(0, 27)]
    df_output = df[df['carrier'] == 'hydrogen']

    #columns_to_operate = [year for year in range(0, 14)]
    columns_to_operate = [year for year in range(0, 27)]

    for node in df_input['node'].unique():
        filtered_rows = df_input[(df_input['technology'] == 'biomethane_SMR') &
                                 (df_input['carrier'] == 'biomethane') &
                                 (df_input['node'] == node)]

        filtered_rows_1 = df_input[(df_input['technology'] == 'biomethane_SMR_CCS') &
                                   (df_input['carrier'] == 'biomethane') &
                                   (df_input['node'] == node)]

        if not filtered_rows.empty:
            adjusted_values = (filtered_rows[columns_to_operate] / 1.2987).iloc[0].to_dict()
            new_row = {**{'node': node, 'technology': 'SMR_biomethane', 'carrier': 'hydrogen'}, **adjusted_values}
            df_output = pd.concat([df_output, pd.DataFrame([new_row])], ignore_index=True)

        if not filtered_rows_1.empty:
            adjusted_values_1 = (filtered_rows_1[columns_to_operate] / 1.2987).iloc[0].to_dict()
            new_row_1 = {**{'node': node, 'technology': 'SMR_CCS_biomethane', 'carrier': 'hydrogen'},
                         **adjusted_values_1}
            df_output = pd.concat([df_output, pd.DataFrame([new_row_1])], ignore_index=True)

    #columns_to_check = [i for i in range(14)]
    columns_to_check = [i for i in range(27)]

    df_output = df_output.loc[df_output[columns_to_check].sum(axis=1) > 0]

    #columns_to_operate = [i for i in range(14)]
    columns_to_operate = [i for i in range(27)]

    df_smr = df_output[df_output['technology'] == 'SMR'].set_index('node')
    df_smr_biomethane = df_output[df_output['technology'] == 'SMR_biomethane'].set_index('node')

    for node, row in df_smr.iterrows():
        if node in df_smr_biomethane.index:
            df_output.loc[(df_output['node'] == node) & (df_output['technology'] == 'SMR'), columns_to_operate] -= df_smr_biomethane.loc[
                node, columns_to_operate]

    #columns_to_operate = [i for i in range(14)]
    columns_to_operate = [i for i in range(27)]

    df_smr_ccs = df_output[df_output['technology'] == 'SMR_CCS'].set_index('node')
    df_smr_ccs_biomethane = df_output[df_output['technology'] == 'SMR_CCS_biomethane'].set_index('node')

    for node, row in df_smr_ccs.iterrows():
        if node in df_smr_ccs_biomethane.index:
            df_output.loc[
                (df_output['node'] == node) & (df_output['technology'] == 'SMR_CCS'), columns_to_operate] -= \
            df_smr_ccs_biomethane.loc[
                node, columns_to_operate]

    #columns_to_operate = [i for i in range(14)]
    columns_to_operate = [i for i in range(27)]
    df_output[columns_to_operate] = df_output[columns_to_operate].clip(lower=0)

    #df_output.to_csv('output_3.csv', index=False)

    gdf = gpd.read_file(shapefile_path).to_crs(epsg=3035)

    level = [0]
    country_gdf = gdf[gdf['LEVL_CODE'].isin(level)]

    countries_to_exclude = ['IS', 'TR']
    country_gdf = country_gdf[~country_gdf['CNTR_CODE'].isin(countries_to_exclude)]

    plt.rcParams['hatch.color'] = 'grey'
    plt.rcParams['hatch.linewidth'] = 0.4

    for year_col in output_columns:
        year = int(year_col)
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=300)

        country_gdf.plot(ax=ax, color='ghostwhite', edgecolor='dimgrey', linewidth=0.8)

        for country_code in ['NO', 'UK', 'CH']:
            specific_gdf = country_gdf[country_gdf['CNTR_CODE'] == country_code]
            specific_gdf.plot(ax=ax, facecolor='lightgrey', hatch="\\\\\\", edgecolor='dimgrey', linewidth=0.8)

        for index, row in df_output.iterrows():
            technology = row[technology_column]
            color = 'lime' if technology == 'electrolysis' else 'pink' if technology == 'SMR_CCS_biomethane' \
                else 'lightgrey' if technology == 'SMR_biomethane' else 'darkgray' if technology == 'SMR' \
                else 'deeppink' if technology == 'SMR_CCS' else 'darkturquoise' if technology == 'gasification' \
                else 'cyan' if technology == 'gasification_CCS' else 'None' if technology == 'hydrogen_decompressor' else 'None'
            output = row[year_col]
            if output == 0:
                continue
            radius = output * 0.01
            node = row['node']
            point = gdf.loc[gdf['NUTS_ID'] == node].to_crs(epsg=3035).geometry.centroid.iloc[0]
            ax.scatter(point.x, point.y, s=radius, color=color, label=technology, alpha=0.7)

        #ax.set_title(f"Hydrogen Production in Europe {year}")

        plt.axis('off')
        ax.set_xlim([2.5e6, 5.9e6])
        ax.set_ylim([1.5e6, 5.5e6])
        plt.tight_layout()
        plt.savefig(f"{folder_path}/hyrogen_production_map_{scenario}_{year}.png", dpi=300, bbox_inches='tight', pad_inches=0)

        plt.show()

def get_industry_capex_data(scenario):
    industries = ['ammonia', 'steel', 'cement', 'methanol', 'refining']
    df_industries = pd.DataFrame()

    for industry in industries:
        res_scenario = Results(f"../outputs/hard_to_abate_{industry}_130324/")
        df_industry = res_scenario.get_total("cost_capex_total", scenario)
        df_industry = df_industry.loc[scenario]
        df_industries[industry] = df_industry.squeeze()

    return df_industries

def get_total_industry_capex_data(scenario):
    res_scenario = Results("../outputs/hard_to_abate_scenarios_140324/")
    df_all_industries_total = res_scenario.get_total("cost_capex_total", scenario)
    df_all_industries_total = df_all_industries_total.loc[scenario]

    return df_all_industries_total

def get_industry_opex_data(scenario):
    industries = ['ammonia', 'steel', 'cement', 'methanol', 'refining']
    df_industries = pd.DataFrame()

    for industry in industries:
        res_scenario = Results(f"../outputs/hard_to_abate_{industry}_130324/")
        df_industry = res_scenario.get_total("cost_opex_total", scenario)
        df_industry = df_industry.loc["scenario_"]
        df_industries[industry] = df_industry.squeeze()

    return df_industries

def get_total_industry_opex_data(scenario):
    res_scenario = Results("../outputs/hard_to_abate_scenarios_140324/")
    df_all_industries_total = res_scenario.get_total("cost_opex_total", scenario)
    df_all_industries_total = df_all_industries_total.loc[scenario]

    return df_all_industries_total

def get_industry_carrier_costs_data(scenario):
    industries = ['ammonia', 'steel', 'cement', 'methanol', 'refining']
    df_all_industries = pd.DataFrame()

    for industry in industries:
        res_scenario = Results(f"../outputs/hard_to_abate_{industry}_130324/")
        df_industry = res_scenario.get_total("cost_carrier")
        df_industry = df_industry.loc[scenario]
        df_industry_sum = df_industry.sum()
        df_all_industries[industry] = df_industry_sum

    return df_all_industries

def get_total_industry_cost_carrier_data(scenario):
    res_scenario = Results("../outputs/hard_to_abate_scenarios_140324/")
    df_all_industries_total = res_scenario.get_total("cost_carrier")
    df_all_industries_total = df_all_industries_total.loc[scenario]
    total_carrier_costs_by_year = df_all_industries_total.sum()
    print(total_carrier_costs_by_year)

    return total_carrier_costs_by_year

def plot_costs_with_unique_colors(scenario):
    df_capex = get_industry_capex_data(scenario)
    df_opex = get_industry_opex_data(scenario)
    df_carrier_costs = get_industry_carrier_costs_data(scenario)
    total_all = get_total_industry_capex_data(scenario) + get_total_industry_opex_data(
        scenario) + get_total_industry_cost_carrier_data(scenario)

    years = np.arange(2024, 2051, 2)
    index = np.arange(len(years))
    bar_width = 0.35

    colors = {
        'ammonia': {'capex': 'mediumorchid', 'opex': '#9B30FF', 'carrier_costs': '#D15FEE'},
        'steel': {'capex': 'steelblue', 'opex': '#3B9F9F', 'carrier_costs': '#5CACEE'},
        'cement': {'capex': 'darkslateblue', 'opex': '#6A5ACD', 'carrier_costs': '#836FFF'},
        'methanol': {'capex': 'fuchsia', 'opex': '#E800E8', 'carrier_costs': '#FF77FF'},
        'refining': {'capex': 'firebrick', 'opex': '#CD2626', 'carrier_costs': '#FF3030'}
    }

    fig, ax = plt.subplots(figsize=(18, 10))

    legend_elements = []

    bottom = np.zeros(len(years))

    for industry, color_map in colors.items():
        for cost_type, color in color_map.items():
            if cost_type == 'capex':
                value = df_capex[industry]
            elif cost_type == 'opex':
                value = df_opex[industry]
            else:
                value = df_carrier_costs[industry]

            bars = ax.bar(index - bar_width / 2, value, bottom=bottom, color=color, width=bar_width,
                          label=f"{industry} {cost_type}")
            bottom += value.values

            if np.all(bottom == value.values):
                legend_elements.append(bars[0])

    ax.bar(index + bar_width / 2, total_all, color='grey', width=bar_width, label='totex integrated optimization')

    handles, labels = ax.get_legend_handles_labels()
    unique_labels, unique_handles = [], []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    #title_font = FontProperties(family='Times New Roman', size=12)
    ax.legend(unique_handles, unique_labels, loc='upper left', bbox_to_anchor=(1, 1),
              #title="Cost Types per Industry",
              frameon=False,
              prop={'size': 22})

    ax.set_xlabel('Year', fontsize=22)
    ax.set_ylabel('Yearly Costs in B€', fontsize=22)
    #ax.set_title('Cost Comparison: Individual Industries and Overall Optimization', fontname='Times New Roman')
    ax.set_xticks(index)
    ax.set_xticklabels(years)

    y_labels = [f"{int(label) / 1000:.2f}" for label in ax.get_yticks()]
    ax.set_yticklabels(y_labels)

    ax.tick_params(axis='x', which='major', labelsize=14, labelcolor='black', labelrotation=0)

    ax.tick_params(axis='y', which='major', labelsize=22, labelcolor='black', labelrotation=0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"cost_comparison_{scenario}.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()

def plot_npc(scenario, scenario_1):

    individual_capex_total = get_industry_capex_data(scenario).sum().sum()
    individual_opex_total = get_industry_opex_data(scenario).sum().sum()
    individual_carrier_costs_total = get_industry_carrier_costs_data(scenario).sum().sum()

    total_capex_total = get_total_industry_capex_data(scenario_1).sum()
    total_opex_total = get_total_industry_opex_data(scenario_1).sum()
    total_carrier_costs_total = get_total_industry_cost_carrier_data(scenario_1).sum().sum()

    categories = ['carrier costs', 'CAPEX', 'OPEX']
    colors = ['#A9A9A9', 'dimgray', '#505050']
    industrial_totals = np.array([individual_carrier_costs_total, individual_capex_total, individual_opex_total])
    total_totals = np.array([total_carrier_costs_total, total_capex_total, total_opex_total])

    fig, ax = plt.subplots(figsize=(9, 10))

    bar_width = 0.20
    bar_spacing = 0.07
    bar_positions = np.array([0, bar_width + bar_spacing])

    bottom_industrial = 0
    for i, color in enumerate(colors):
        ax.bar(bar_positions[0], industrial_totals[i], bottom=bottom_industrial, color=color, width=bar_width,
               label=categories[i])
        bottom_industrial += industrial_totals[i]

    bottom_total = 0
    for i, color in enumerate(colors):
        ax.bar(bar_positions[1], total_totals[i], bottom=bottom_total, color=color, width=bar_width)
        bottom_total += total_totals[i]

    y_labels = [f"{int(label) / 1000:.2f}" for label in ax.get_yticks()]
    ax.set_yticklabels(y_labels, fontsize=22)

    ax.set_xticks(bar_positions)
    ax.set_xticklabels(['individual', 'integrated'], fontsize=22)
    ax.set_ylabel('Totex 2024 - 2050 [B€]', fontsize=22)
    #ax.set_title('Cost Comparison',  fontname='Times New Roman')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False, prop={'size': 22})

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()

def plot_cost_comp_npc(scenario, scenario_1, scenario_2, scenario_3):

    total_capex_total = get_total_industry_capex_data(scenario).sum()
    total_opex_total = get_total_industry_opex_data(scenario).sum()
    total_carrier_costs_total = get_total_industry_cost_carrier_data(scenario).sum().sum()

    total_capex_total_1 = get_total_industry_capex_data(scenario_1).sum()
    total_opex_total_1 = get_total_industry_opex_data(scenario_1).sum()
    total_carrier_costs_total_1 = get_total_industry_cost_carrier_data(scenario_1).sum().sum()

    total_capex_total_2 = get_total_industry_capex_data(scenario_2).sum()
    total_opex_total_2 = get_total_industry_opex_data(scenario_2).sum()
    total_carrier_costs_total_2 = get_total_industry_cost_carrier_data(scenario_2).sum().sum()

    total_capex_total_3 = get_total_industry_capex_data(scenario_3).sum()
    total_opex_total_3 = get_total_industry_opex_data(scenario_3).sum()
    total_carrier_costs_total_3 = get_total_industry_cost_carrier_data(scenario_3).sum().sum()

    categories = ['CAPEX', 'OPEX', 'Carrier Costs']
    colors = ['dimgray', 'slategray', 'lightsteelblue']
    scenario_data = [
        [get_total_industry_capex_data(scenario).sum(), get_total_industry_opex_data(scenario).sum(),
         get_total_industry_cost_carrier_data(scenario).sum().sum()],
        [get_total_industry_capex_data(scenario_1).sum(), get_total_industry_opex_data(scenario_1).sum(),
         get_total_industry_cost_carrier_data(scenario_1).sum().sum()],
        [get_total_industry_capex_data(scenario_2).sum(), get_total_industry_opex_data(scenario_2).sum(),
         get_total_industry_cost_carrier_data(scenario_2).sum().sum()],
        [get_total_industry_capex_data(scenario_3).sum(), get_total_industry_opex_data(scenario_3).sum(),
         get_total_industry_cost_carrier_data(scenario_3).sum().sum()]
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.2
    n_scenarios = len(scenario_data)
    indices = np.arange(n_scenarios)

    for scenario_index, scenario_costs in enumerate(scenario_data):
        for category_index, cost in enumerate(scenario_costs):
            bar = ax.bar(scenario_index + bar_width * category_index, cost, bar_width, color=colors[category_index],
                         label=categories[category_index] if scenario_index == 0 else "")
            ax.text(scenario_index + bar_width * category_index, cost, f'{cost:.2f}', ha='center', va='bottom')

    ax.set_xlabel('Szenarien')
    ax.set_ylabel('Kosten')
    ax.set_title('Kostenvergleich der Szenarien')
    ax.set_xticks(indices + bar_width / 2 * (len(categories) - 1))
    ax.set_xticklabels(['Szenario 1', 'Szenario 2', 'Szenario 3', 'Szenario 4'])

    if n_scenarios == 1:
        ax.legend(categories, loc='upper right')

    plt.tight_layout()
    plt.grid(True)
    plt.show()
def draw_transport_arrows_carbon(csv_path, shapefile_path, year, scenario, figsize=(20, 20)):
    df = pd.read_csv(csv_path)
    df = df[df['Unnamed: 0'] == scenario]
    df = df[df['technology'] == 'carbon_pipeline']
    nuts_gdf = gpd.read_file(shapefile_path)
    nuts_gdf = nuts_gdf.to_crs('EPSG:3035')
    countries_to_exclude = ['UK', 'CH', 'IS', 'TR']
    nuts2_gdf = nuts_gdf[(nuts_gdf['LEVL_CODE'] == 2) & (~nuts_gdf['CNTR_CODE'].isin(countries_to_exclude))]
    nuts2_gdf['centroid'] = nuts2_gdf.geometry.centroid
    centroid_dict = nuts2_gdf.set_index('NUTS_ID')['centroid'].to_dict()

    background_gdf = nuts_gdf[(nuts_gdf['LEVL_CODE'] == 0) & (~nuts_gdf['CNTR_CODE'].isin(countries_to_exclude))]


    df[['source', 'target']] = df['edge'].str.split('-', expand=True)

    fig, ax = plt.subplots(figsize=figsize)
    background_gdf.plot(ax=ax, color='lightgrey', edgecolor='darkgray')
    background_gdf[background_gdf['CNTR_CODE'] == 'NO'].plot(ax=ax, color='darkgray', edgecolor='darkgray')
    nuts_gdf.plot(ax=ax, color='none', edgecolor='none')


    for _, row in df.iterrows():
        source = row['source']
        target = row['target']
        amount = row[year]
        tech = row['technology']

        color = 'forestgreen' if tech == 'biomethane_transport' else 'black' if tech =='carbon_pipeline' else 'red' if tech == 'hydrogen_pipeline' else 'olive' if tech == 'dry_biomass_truck' else 'purple'
        linewidth = np.sqrt(amount) * 0.03

        if source in centroid_dict and target in centroid_dict:
            source_point = centroid_dict[source]
            target_point = centroid_dict[target]

            ax.annotate('', xy=(target_point.x, target_point.y), xytext=(source_point.x, source_point.y),
                        arrowprops=dict(arrowstyle='->,head_width=0.2,head_length=0.4', color=color, lw=linewidth))

    plt.axis('off')
    plt.tight_layout()
    plt.show()


def draw_transport_and_capture(transport_data, capture_data, shapefile_path, year, scenario, figsize=(20, 20)):
    df_transport = transport_data
    transport_technologies = ['carbon_pipeline']
    df_transport = df_transport[df_transport['technology'].isin(transport_technologies)]

    nuts_gdf = gpd.read_file(shapefile_path)
    nuts_gdf = nuts_gdf.to_crs('EPSG:3035')
    countries_to_exclude = ['IS', 'TR']
    nuts2_gdf = nuts_gdf[(nuts_gdf['LEVL_CODE'] == 2) & (~nuts_gdf['CNTR_CODE'].isin(countries_to_exclude))]
    nuts2_gdf['centroid'] = nuts2_gdf.geometry.centroid
    centroid_dict = nuts2_gdf.set_index('NUTS_ID')['centroid'].to_dict()

    level = [0]
    countries_to_exclude = ['IS', 'TR']
    nuts_gdf_filtered = nuts_gdf[
        (nuts_gdf['LEVL_CODE'].isin(level)) & (~nuts_gdf['CNTR_CODE'].isin(countries_to_exclude))]

    df_capture = capture_data
    capture_technologies = [
        "BF_BOF_CCS", "gasification_CCS", "SMR_CCS",
        "cement_plant_oxy_combustion", "cement_plant_post_comb", "DAC"
    ]
    relevant_carriers = ["carbon", "carbon_liquid"]
    df_capture = df_capture[
        df_capture['technology'].isin(capture_technologies) &
        df_capture['carrier'].isin(relevant_carriers)
    ]
    df_capture_sum = df_capture.groupby('node')[year].sum().reset_index(name='Total_Carbon_Captured')
    nuts2_with_capture = nuts2_gdf.merge(df_capture_sum, left_on='NUTS_ID', right_on='node', how='left')
    nuts2_with_capture['Total_Carbon_Captured'].fillna(0, inplace=True)

    df_transport[['source', 'target']] = df_transport['edge'].str.split('-', expand=True)

    fig, ax = plt.subplots(figsize=figsize)

    vmin = nuts2_with_capture['Total_Carbon_Captured'].min()
    #vmax = nuts2_with_capture['Total_Carbon_Captured'].max()
    vmax = 3000
    sm = plt.cm.ScalarMappable(cmap='Blues', norm=Normalize(vmin=vmin, vmax=vmax))
    sm._A = []

    nuts2_with_capture.plot(column='Total_Carbon_Captured', ax=ax, cmap='Blues', norm=sm.norm,
                            edgecolor='lightgray', linewidth=0.8)

    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', shrink=0.75, aspect=30)

    level = [0]
    nuts_gdf = nuts_gdf[nuts_gdf['LEVL_CODE'].isin(level)]
    border_color = 'dimgray'

    nuts_gdf_filtered.plot(ax=ax, color="None", edgecolor=border_color)

    plt.rcParams['hatch.color'] = 'grey'
    plt.rcParams['hatch.linewidth'] = 0.4

    for country_code in ['UK', 'CH']:
        specific_gdf = nuts_gdf[nuts_gdf['CNTR_CODE'] == country_code]
        specific_gdf.plot(ax=ax, facecolor='lightgrey', hatch="\\\\\\", edgecolor='dimgrey', linewidth=0.8)


    norway = nuts_gdf[nuts_gdf['CNTR_CODE'] == 'NO']
    norway.plot(ax=ax, color='lightgrey', edgecolor=border_color)

    def format_by_thousand(x, pos):
        return '{}'.format(x / 1000)

    cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_by_thousand))
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label('Annual carbon capture rates [Mt]', fontsize=22)

    for _, row in df_transport.iterrows():
        source = row['source']
        target = row['target']

        filtered_df = df_transport[(df_transport['source'] == source) & (df_transport['target'] == target)]
        amount = filtered_df[year].sum()

        color = 'red'
        if amount > 0:
            linewidth = np.sqrt(amount) * 0.05
            if source in centroid_dict and target in centroid_dict:
                source_point = centroid_dict[source]
                target_point = centroid_dict[target]

                ax.annotate('', xy=(target_point.x, target_point.y), xytext=(source_point.x, source_point.y),
                            arrowprops=dict(arrowstyle='->, head_width=0.1, head_length=0.3', color=color,
                                            lw=linewidth))

    ax.set_xlim(2.5e6, 5.9e6)
    ax.set_ylim(1.5e6, 5.5e6)
    plt.axis('off')
    #plt.tight_layout
    plt.savefig(f"{folder_path}/carbon_transport_map_{scenario}_{year}.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()

def draw_hydrogen_pipelines(transport_data, capture_data, shapefile_path, year, scenario, figsize=(20, 20)):
    df_transport = transport_data
    df_transport = df_transport[df_transport['technology'] == 'hydrogen_pipeline']

    nuts_gdf = gpd.read_file(shapefile_path)
    nuts_gdf = nuts_gdf.to_crs('EPSG:3035')
    countries_to_exclude = ['IS', 'TR']
    nuts2_gdf = nuts_gdf[(nuts_gdf['LEVL_CODE'] == 2) & (~nuts_gdf['CNTR_CODE'].isin(countries_to_exclude))]
    nuts2_gdf['centroid'] = nuts2_gdf.geometry.centroid
    centroid_dict = nuts2_gdf.set_index('NUTS_ID')['centroid'].to_dict()

    level = [0]
    countries_to_exclude = ['IS', 'TR']
    nuts_gdf_filtered = nuts_gdf[
        (nuts_gdf['LEVL_CODE'].isin(level)) & (~nuts_gdf['CNTR_CODE'].isin(countries_to_exclude))]

    df_hydrogen = capture_data
    capture_technologies = [
        "SMR", "SMR_CCS", "gasification",
        "gasification_CCS", "electrolysis"
    ]
    relevant_carriers = ["hydrogen"]
    df_hydrogen = df_hydrogen[
        df_hydrogen['technology'].isin(capture_technologies) &
        df_hydrogen['carrier'].isin(relevant_carriers)
    ]
    df_capture_sum = df_hydrogen.groupby('node')[year].sum().reset_index(name='Total_Carbon_Captured')
    nuts2_with_capture = nuts2_gdf.merge(df_capture_sum, left_on='NUTS_ID', right_on='node', how='left')
    nuts2_with_capture['Total_Carbon_Captured'].fillna(0, inplace=True)

    df_transport[['source', 'target']] = df_transport['edge'].str.split('-', expand=True)

    fig, ax = plt.subplots(figsize=figsize)

    vmin = nuts2_with_capture['Total_Carbon_Captured'].min()
    #vmax = nuts2_with_capture['Total_Carbon_Captured'].max()
    vmax = 10000
    sm = plt.cm.ScalarMappable(cmap='YlOrBr', norm=Normalize(vmin=vmin, vmax=vmax))
    sm._A = []

    nuts2_with_capture.plot(column='Total_Carbon_Captured', ax=ax, cmap='YlOrBr', norm=sm.norm,
                            edgecolor='lightgray', linewidth=0.8)

    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', shrink=0.75, aspect=30)

    level = [0]
    nuts_gdf = nuts_gdf[nuts_gdf['LEVL_CODE'].isin(level)]
    border_color = 'dimgray'
    country_color = 'darkgrey'

    nuts_gdf_filtered.plot(ax=ax, color="None", edgecolor=border_color)

    plt.rcParams['hatch.color'] = 'grey'
    plt.rcParams['hatch.linewidth'] = 0.4

    for country_code in ['NO', 'UK', 'CH']:
        specific_gdf = nuts_gdf[nuts_gdf['CNTR_CODE'] == country_code]
        specific_gdf.plot(ax=ax, facecolor='lightgrey', hatch="\\\\\\", edgecolor='dimgrey', linewidth=0.8)

    def format_by_thousand(x, pos):
        return '{}'.format(x / 1000)

    cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_by_thousand))
    cbar.ax.tick_params(labelsize=18)

    cbar.set_label('Yearly hydrogen production [TWh]', fontsize=22)
    # Draw transport arrows
    for _, row in df_transport.iterrows():
        source = row['source']
        target = row['target']

        filtered_df = df_transport[(df_transport['source'] == source) & (df_transport['target'] == target)]

        amount = filtered_df[year].sum()
        color = 'blue'
        if amount > 0:
            linewidth = np.sqrt(amount) * 0.05
            if source in centroid_dict and target in centroid_dict:
                source_point = centroid_dict[source]
                target_point = centroid_dict[target]

                ax.annotate('', xy=(target_point.x, target_point.y), xytext=(source_point.x, source_point.y),
                            arrowprops=dict(arrowstyle='->, head_width=0.1, head_length=0.3', color=color,
                                            lw=linewidth))

    ax.set_xlim(2.5e6, 5.9e6)
    ax.set_ylim(1.5e6, 5.5e6)

    plt.axis('off')
    #plt.tight_layout
    plt.savefig(f"{folder_path}/hydrogen_transport_map_{scenario}_{year}.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()

def draw_transport_arrows_and_biomass_usage(transport_data, shapefile_path, year, scenario, figsize=(20, 20)):

    df = transport_data
    transport_technologies = ['biomethane_transport', 'dry_biomass_truck']
    df = df[df['technology'].isin(transport_technologies)]

    availability_dry_biomass_scenario = res_scenario.get_total("availability_import").xs(scenario)
    availability_dry_biomass = availability_dry_biomass_scenario.xs("dry_biomass", level = 'carrier')

    availability_wet_biomass_scenario = res_scenario.get_total("availability_import").xs(scenario)
    availability_wet_biomass = availability_wet_biomass_scenario.xs("wet_biomass",level='carrier')

    import_dry_biomass_scenario = res_scenario.get_total("flow_import").xs(scenario)
    import_dry_biomass = import_dry_biomass_scenario.xs("dry_biomass", level='carrier')

    import_wet_biomass_scenario = res_scenario.get_total("flow_import").xs(scenario)
    import_wet_biomass = import_wet_biomass_scenario.xs("wet_biomass", level='carrier')

    nuts_gdf = gpd.read_file(shapefile_path)
    nuts_gdf = nuts_gdf.to_crs('EPSG:3035')
    countries_to_exclude = ['IS', 'TR']

    nuts_gdf = nuts_gdf[~nuts_gdf['CNTR_CODE'].isin(countries_to_exclude)]
    nuts2_gdf = nuts_gdf[nuts_gdf['LEVL_CODE'] == 2]
    nuts2_gdf['centroid'] = nuts2_gdf.geometry.centroid
    centroid_dict = nuts2_gdf.set_index('NUTS_ID')['centroid'].to_dict()

    df[['source', 'target']] = df['edge'].str.split('-', expand=True)

    combined_dry_biomass = pd.merge(availability_dry_biomass[year], import_dry_biomass[year], left_on='node', right_on='node', how='outer')
    combined_dry_biomass['utilization'] = np.where(combined_dry_biomass[f'{year}_x'] > 0,
                                                      combined_dry_biomass[f'{year}_y'] / combined_dry_biomass[
                                                          f'{year}_x'], 0)

    combined_wet_biomass = pd.merge(availability_wet_biomass[year], import_wet_biomass[year], left_on='node', right_on='node', how='outer')
    combined_wet_biomass['utilization'] = np.where(combined_wet_biomass[f'{year}_x'] > 0,
                                                   combined_wet_biomass[f'{year}_y'] / combined_wet_biomass[
                                                       f'{year}_x'], 0)




    combined_biomass = pd.merge(combined_dry_biomass, combined_wet_biomass, left_on='node', right_on='node', how='outer')
    combined_biomass = combined_biomass.fillna(0)

    combined_biomass['utilization'] = (combined_biomass['utilization_x'] + combined_biomass['utilization_y']) / 2

    nuts2_gdf = nuts2_gdf.merge(combined_biomass, left_on='NUTS_ID', right_on='node', how='left').fillna(0)


    fig, ax = plt.subplots(figsize=figsize)
    vmin = nuts2_gdf['utilization'].min()
    vmax = 1.4
    norm = Normalize(vmin=vmin, vmax=vmax)
    nuts2_gdf.plot(column='utilization', ax=ax, cmap='Purples', edgecolor='lightgray', vmax=vmax, norm=norm)


    sm = plt.cm.ScalarMappable(cmap='Purples', norm=norm)
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', shrink=0.75, aspect=30)
    cbar.ax.set_ylim(0,1)
    level = [0]
    nuts_gdf = nuts_gdf[nuts_gdf['LEVL_CODE'].isin(level)]
    border_color = 'dimgray'
    country_color = 'darkgrey'

    nuts_gdf.plot(ax=ax, color="None", edgecolor=border_color)

    plt.rcParams['hatch.color'] = 'grey'
    plt.rcParams['hatch.linewidth'] = 0.4

    for country_code in ['NO', 'UK', 'CH']:
        specific_gdf = nuts_gdf[nuts_gdf['CNTR_CODE'] == country_code]
        specific_gdf.plot(ax=ax, facecolor='lightgrey', hatch="\\\\\\", edgecolor='dimgrey', linewidth=0.8)

    cbar.set_label('Usage of biomass potential', fontsize=22)

    cbar.ax.tick_params(labelsize=18)

    for _, row in df.iterrows():
        source = row['source']
        target = row['target']

        filtered_df = df[(df['source'] == source) & (df['target'] == target)]

        amount = filtered_df[year].sum()
        tech = row['technology']

        color = 'coral' if tech == 'biomethane_transport' else 'crimson' if tech == 'dry_biomass_truck' else 'purple'

        if pd.notnull(amount) and amount > 0:
            linewidth = np.sqrt(amount) * 0.05
            if source in centroid_dict and target in centroid_dict:
                source_point = centroid_dict[source]
                target_point = centroid_dict[target]

                ax.annotate('', xy=(target_point.x, target_point.y), xytext=(source_point.x, source_point.y),
                            arrowprops=dict(arrowstyle='->, head_width=0.2, head_length=0.4', color=color, lw=linewidth))

    ax.set_xlim(2.5e6, 5.9e6)
    ax.set_ylim(1.5e6, 5.5e6)

    plt.axis('off')
    plt.savefig(f"{folder_path}/biomass_transport_map_{scenario}_{year}.png", dpi=300, bbox_inches='tight', pad_inches=0)

    plt.show()

def calc_lco(scenario, discount_rate, carrier):
    capex_df = res_scenario.get_total("cost_capex_total").xs(scenario).reset_index()
    opex_df = res_scenario.get_total("cost_opex_total").xs(scenario).reset_index()
    carrier_cost_df = res_scenario.get_total("cost_carrier").xs(scenario).reset_index()

    # Filter and aggregate costs for the specific carrier
    carrier_cost_df = carrier_cost_df[carrier_cost_df['carrier'] == carrier]
    carrier_cost_df.drop(['node', 'carrier'], axis=1, inplace=True)
    carrier_cost_df = carrier_cost_df.sum()  # Sum up all costs per year for the selected carrier

    production_df = res_scenario.get_total("flow_conversion_output").xs(scenario).reset_index()
    production_df = production_df[production_df['carrier'] == carrier]
    production_df = production_df.drop(['node', 'carrier'], axis=1).sum()

    total_discounted_costs = 0
    total_discounted_production = 0

    #for year in range(14):
    for year in range(27):  # Assuming the year columns are labeled as integers from 0 to 13
        discount_factor = (1 + discount_rate) ** year
        discounted_costs = (capex_df.loc[year, 'scenario_'] +
                            opex_df.loc[year, 'scenario_'] +
                            carrier_cost_df[year]) / discount_factor

        discounted_production = production_df[year] / discount_factor

        total_discounted_costs += discounted_costs
        total_discounted_production += discounted_production

    lcoa = total_discounted_costs / total_discounted_production if total_discounted_production else float('inf')
    print("LCOA:", lcoa)
    return lcoa

def plot_capacity_addition(folder_path, scenario, technology, save_file):

    capacities = res_scenario.get_total("capacity_addition").xs(scenario).reset_index()
    capacities[capacities['technology'] == technology].to_csv(f"capacities_{technology}.csv", index=False)
    capacities_grouped = capacities.groupby(['technology', 'location']).sum().reset_index()
    print(capacities_grouped)
    year_mapping = {i: year for i, year in enumerate(range(2024, 2051, 2))}
    capacities_grouped.rename(columns=year_mapping, inplace=True)

    tech_data = capacities_grouped[capacities_grouped['technology'] == technology]

    year_columns = list(range(2024, 2051, 2))
    summed_data = tech_data[year_columns].sum() * 8760

    plt.figure(figsize=(10, 6))
    summed_data.plot(kind='bar')
    plt.title(f'Yearly capacity additions for {technology}')
    plt.xlabel('Year')
    plt.ylabel('Capacity [kt]')
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_file:
        plt.savefig(f"{folder_path}/{scenario}_{technology}_capacity_additions.png")

    plt.show()

def plot_existing_capacity(folder_path, scenario, technology, save_file):

    capacities = res_scenario.get_total("capacity").xs(scenario).reset_index()
    capacities[capacities['technology'] == technology].to_csv(f"existing_capacities_{technology}.csv", index=False)
    capacities_grouped = capacities.groupby(['technology', 'location']).sum().reset_index()
    capacities_grouped.to_csv("capacity_existing.csv", index=False)
    year_mapping = {i: year for i, year in enumerate(range(2024, 2051, 2))}
    capacities_grouped.rename(columns=year_mapping, inplace=True)

    tech_data = capacities_grouped[capacities_grouped['technology'] == technology]

    year_columns = list(range(2024, 2051, 2))
    summed_data = tech_data[year_columns].sum() * 8760

    plt.figure(figsize=(10, 6))
    summed_data.plot(kind='bar')
    plt.title(f'Yearly capacity for {technology}')
    plt.xlabel('Year')
    plt.ylabel('Capacity [kt]')
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_file:
        plt.savefig(f"{folder_path}/{scenario}_{technology}_capacity_additions.png")

    plt.show()

def plot_carbon_capture(scenario):
    carbon_techs = ['BF_BOF_CCS', 'SMR_CCS', 'cement_plant_oxy_combustion', 'gasification_CCS', 'DAC']
    outputs = ['carbon', 'carbon_liquid']

    def prepare_df(source_path):
        # Assume Results and other necessary imports are available
        # and properly defined here
        results = Results(source_path)
        df = results.get_total('flow_conversion_output').xs(scenario).reset_index()
        df = df[df['technology'].isin(carbon_techs)]
        df = df[df['carrier'].isin(outputs)]
        df = df.drop('node', axis=1)
        df = df.groupby(['technology', 'carrier']).sum()
        return df

    carbon_integrated = prepare_df("../outputs/hard_to_abate_scenarios_5%_290424/")
    carbon_ammonia = prepare_df("../outputs/hard_to_abate_ammonia_290424/")
    carbon_steel = prepare_df("../outputs/hard_to_abate_steel_300424/")
    carbon_cement = prepare_df("../outputs/hard_to_abate_cement_300424/")
    carbon_methanol = prepare_df("../outputs/hard_to_abate_methanol_300424")

    def melt_and_label(df, label):
        df_long = pd.melt(df.reset_index(), id_vars=['technology', 'carrier'], var_name='Year', value_name='Value')
        df_long['Source'] = label
        return df_long

    df_long_integrated = melt_and_label(carbon_integrated, 'Integrated')
    df_long_ammonia = melt_and_label(carbon_ammonia, 'Ammonia')
    df_long_steel = melt_and_label(carbon_steel, 'Steel')
    df_long_cement = melt_and_label(carbon_cement, 'Cement')
    df_long_methanol = melt_and_label(carbon_methanol, 'Methanol')

    combined = pd.concat([df_long_ammonia, df_long_steel, df_long_cement, df_long_methanol, df_long_integrated])

    combined['Group'] = combined['Source'].replace(
        {'Integrated': 'Integrated', 'Ammonia': 'Grouped', 'Steel': 'Grouped', 'Cement': 'Grouped',
         'Methanol': 'Grouped'})

    pivot_df = combined.pivot_table(index=['Year', 'Group'], columns=['technology', 'carrier'], values='Value',
                                    aggfunc='sum')

    print(pivot_df)
    fig, ax = plt.subplots(figsize=(20, 8))
    pivot_df.plot(kind='bar', stacked=True, ax=ax)
    plt.title('Comparison of Carbon Capture Technologies Over Years')
    plt.ylabel('Value')

    for patch in ax.patches:
        if patch.get_label() == 'Integrated':
            patch.set_hatch('//')

    years = range(2024, 2051, 2)
    x_positions = np.arange(len(years)) * 2 + 0.5
    plt.xticks(x_positions, years, rotation=0, fontsize=12)

    plt.xlabel('Year')

    ax.axhline(y=82817, color='r', linestyle='--', linewidth=2, label='Target Value')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels), title='Technology, Carrier', bbox_to_anchor=(1.05, 1),
              loc='upper left')

    plt.tight_layout()
    plt.show()

def plot_storage_utilization(scenario):
    #df_integrated = res_scenario.get_total("flow_conversion_input").xs(scenario).reset_index()
    storage = ['carbon_storage']
    output = ['carbon_liquid']
    def prepare_df(source_path):
        results = Results(source_path)
        df = results.get_total('flow_conversion_input').xs(scenario).reset_index()
        df = df[df['technology'].isin(storage)]
        df = df[df['carrier'].isin(output)]
        df = df.drop('node', axis=1)
        df = df.groupby(['technology', 'carrier']).sum()
        return df

    df_integrated = prepare_df("../outputs/hard_to_abate_scenarios_5%_290424/")
    df_ammonia = prepare_df("../outputs/hard_to_abate_ammonia_290424/")
    df_steel = prepare_df("../outputs/hard_to_abate_steel_300424/")
    df_cement = prepare_df("../outputs/hard_to_abate_cement_300424/")
    df_methanol = prepare_df("../outputs/hard_to_abate_methanol_300424")

    values = df_integrated.iloc[0, 2:14:2].values

    years = range(2024, 2051, 2)

    plt.plot(years, values, marker='o')
    plt.xlabel('Jahr')
    plt.ylabel('Wert')
    plt.title('Wertentwicklung von 2024 bis 2050')

    years = range(2024, 2051, 2)
    x_positions = np.arange(len(years)) * 2 + 0.5
    plt.xticks(x_positions, years, rotation=0, fontsize=12)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.tight_layout()
    plt.show()


def plot_biomass_and_transport(tech, source_path):
    df = Results(source_path)
    scenarios = ['scenario_', 'scenario_biomass_high_price_2', 'scenario_biomass_high_price_3',
                 'scenario_biomass_high_price_5',
                 'scenario_biomass_high_price_7', 'scenario_biomass_high_price_10']

    colors = plt.cm.jet_r([i / 13 for i in range(14)])

    marker = 'o'

    for year in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
        x_points = []
        y_points = []
        for i, scenario in enumerate(scenarios):
            y_point = df.get_total("flow_transport").xs(scenario).reset_index()
            y_point = y_point[y_point['technology'].isin([tech])]
            y_point = y_point.drop('edge', axis=1)
            y_point = y_point.groupby('technology').sum()

            x_point = (df.get_total("flow_import").xs(scenario).xs("wet_biomass", level='carrier') +
                       df.get_total("flow_import").xs(scenario).xs("dry_biomass", level='carrier') +
                       df.get_total("flow_import").xs(scenario).xs("biomass_cement", level='carrier')) / (
                                  df.get_total("availability_import").xs(scenario).xs('dry_biomass', level='carrier') +
                                  df.get_total("availability_import").xs(scenario).xs('wet_biomass',
                                                                                      level='carrier') + df.get_total(
                              "availability_import").xs(scenario).xs('biomass_cement', level='carrier'))

            x_points.append(x_point.mean()[year] * 100)
            y_points.append(y_point.iloc[0][year])

        for j in range(len(scenarios)):
            if j < len(scenarios) - 1:
                plt.plot([x_points[j], x_points[j + 1]], [y_points[j], y_points[j + 1]], color=colors[year],
                         marker=marker, linestyle='-')

    plt.xlabel('Usage of biomass potential (%)', fontsize=10)
    plt.ylabel(f'Ausbau von {tech}', fontsize=10)
    plt.title(f'Ausbau {tech} vs. Ausnutzung des Biomassepotentials', fontsize=10)
    #plt.legend(fontsize=10)
    #plt.grid(True)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.tight_layout()
    plt.show()

def plot_biomass_and_carbon_storage(source_path):
    df = Results(source_path)
    transport_tech = ['carbon_storage']
    output = ['carbon_liquid']
    scenarios = ['scenario_', 'scenario_biomass_high_price_2', 'scenario_biomass_high_price_3',
                 'scenario_biomass_high_price_5',
                 'scenario_biomass_high_price_7',
                 'scenario_biomass_high_price_10']

    colors = plt.cm.jet_r([i / 13 for i in range(14)])

    marker = 'o'

    for year in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                ]:
        x_points = []
        y_points = []
        for i, scenario in enumerate(scenarios):
            y_point = df.get_total("flow_conversion_input").xs(scenario).reset_index()
            y_point = y_point[y_point['technology'].isin(transport_tech)]
            y_point = y_point[y_point['carrier'].isin(output)]
            y_point = y_point.groupby(['technology', 'carrier']).sum()

            x_point = (df.get_total("flow_import").xs(scenario).xs("wet_biomass", level='carrier') +
                       df.get_total("flow_import").xs(scenario).xs("dry_biomass", level='carrier') +
                       df.get_total("flow_import").xs(scenario).xs("biomass_cement", level='carrier')) / (
                                  df.get_total("availability_import").xs(scenario).xs('dry_biomass', level='carrier') +
                                  df.get_total("availability_import").xs(scenario).xs('wet_biomass',
                                                                                      level='carrier') + df.get_total(
                              "availability_import").xs(scenario).xs('biomass_cement', level='carrier'))

            x_points.append(x_point.mean()[year] * 100)
            y_points.append(y_point.iloc[0][year])

        for j in range(len(scenarios)):
            if j < len(scenarios) - 1:
                plt.plot([x_points[j], x_points[j + 1]], [y_points[j], y_points[j + 1]], color=colors[year],
                         marker=marker, linestyle='-', label=f'Jahr {year}')

    plt.xlabel('Usage of biomass potential (%)', fontsize=10)
    plt.ylabel(f'{transport_tech}', fontsize=10)
    plt.title(f'{transport_tech} and biomass potential', fontsize=10)
    #plt.legend(fontsize=8)
    #plt.grid(True)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.tight_layout()
    plt.show()

def plot_biomass_and_carrier(carrier, source_path):
    df = Results(source_path)
    scenarios = ['scenario_', 'scenario_biomass_high_price_2', 'scenario_biomass_high_price_3',
                 'scenario_biomass_high_price_5',
                 'scenario_biomass_high_price_7', 'scenario_biomass_high_price_10'
                 ]

    colors = plt.cm.jet_r([i / 13 for i in range(14)])

    marker = 'o'

    for year in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
        x_points = []
        y_points = []
        for i, scenario in enumerate(scenarios):
            y_point = df.get_total("flow_conversion_input").xs(scenario).reset_index()
            y_point = y_point[y_point['carrier'].isin([carrier])]
            y_point = y_point.drop(['node', 'technology'], axis=1)
            y_point = y_point.groupby('carrier').sum()

            x_point = (df.get_total("flow_import").xs(scenario).xs("wet_biomass", level='carrier') +
                       df.get_total("flow_import").xs(scenario).xs("dry_biomass", level='carrier') +
                       df.get_total("flow_import").xs(scenario).xs("biomass_cement", level='carrier')) / (
                                  df.get_total("availability_import").xs(scenario).xs('dry_biomass', level='carrier') +
                                  df.get_total("availability_import").xs(scenario).xs('wet_biomass',
                                                                                      level='carrier') + df.get_total(
                              "availability_import").xs(scenario).xs('biomass_cement', level='carrier'))

            x_points.append(x_point.mean()[year] * 100)
            y_points.append(y_point.iloc[0][year])

        for j in range(len(scenarios)):
            if j < len(scenarios) - 1:
                plt.plot([x_points[j], x_points[j + 1]], [y_points[j], y_points[j + 1]], color=colors[year],
                         marker=marker, linestyle='-')

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.xlabel('Usage of biomass potential (%)', fontsize=10)
    plt.ylabel(f'Usage of {carrier}', fontsize=10)
    plt.title(f'Usage of {carrier} vs. usage of biomass potential', fontsize=10)
    #plt.legend(fontsize=10)
    #plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    folder_path = 'scenarios_070524'
    scenarios = [#'scenario_',
                 #'scenario_electrification',
                 #'scenario_hydrogen',
                 #'scenario_biomass',
                 #'scenario_CCS',
                 #'scenario_high_demand',
                 #'scenario_low_demand',
                 'scenario_biomass_high_price_2',
                 'scenario_biomass_high_price_3',
                 'scenario_biomass_high_price_5',
                 'scenario_biomass_high_price_7',
                 'scenario_biomass_high_price_10',
                 #'scenario_no_biomass'
                 ]
    carriers = [#'ammonia',
                #'steel',
                #'cement',
                #'methanol',
                #'oil_products',
                #'direct_reduced_iron',
                #'electricity',
                #'natural_gas',
                #'carbon',
                #'coal_for_cement',
                #'biomethane',
                'hydrogen',
                #'carbon_methanol',
                #'carbon_liquid'
                ]

    #for scenario in scenarios:
     #   get_emissions(scenario)

    #for scenario in scenarios:
     #   save_total(folder_path, scenario)

    #for scenario in scenarios:
     #   save_imports_exports(folder_path, scenario)

    # generate sankey diagram
    target_technologies = ['BF_BOF',
                           'BF_BOF_CCS',
                           'EAF',
                           'carbon_liquefication',
                           'carbon_storage',
                           'cement_plant',
                           'cement_plant_oxy_combustion', 'cement_plant_post_comb',
                           'e_haber_bosch', 'haber_bosch',
                           'gasification_methanol', 'methanol_from_hydrogen', 'methanol_synthesis',
                           'refinery',
                           'biomethane_SMR_methanol', 'biomethane_SMR', 'biomethane_SMR_CCS', #'biomethane_haber_bosch'
                           ]
    intermediate_technologies = [#'anaerobic_digestion',
        'biomethane_conversion', #'biomethane_haber_bosch',
        #'ASU',
        'biomass_to_coal_conversion', 'hydrogen_for_cement_conversion',
        'DAC',
        'DRI', 'h2_to_ng', 'scrap_conversion_EAF', 'scrap_conversion_BF_BOF',
        'SMR', 'SMR_CCS', 'gasification', 'gasification_CCS',
        'electrolysis',
        #'biomethane_SMR', 'biomethane_SMR_CCS',
        #'biomethane_DRI',
        #'biomethane_methanol_synthesis',
        'carbon_conversion', 'carbon_methanol_conversion',
        #'SMR_methanol', 'gasification_methanol_h2'
        #'photovoltaics', 'pv_ground', 'pv_rooftop', 'wind_offshore', 'wind_onshore',
        #'carbon_liquefication', 'carbon_removal',
        #'carbon_storage',
        #'carbon_evaporation'
    ]
    years = [0,
             #3,
             6,
             #8,
             #13,
             16,
             26
             ]
    #for year in years:
     #  for scenario in scenarios:
      #      generate_sankey_diagram(folder_path, scenario, target_technologies, intermediate_technologies, year, title="Process depiction in", save_file=False)

    years = [0, #8, 13
             6,
             16,
             26
             ]
    for scenario in scenarios:
        for year in years:
            shapefile_path = "nuts_data/NUTS_RG_20M_2021_4326.shp"
            #draw_wedges_on_map(folder_path, shapefile_path, year, radius_factor=0.004, scenario=scenario)

    # generate bar charts for industry outputs

    #for scenario in scenarios:
     #   for carrier in carriers:
      #      plot_outputs(folder_path, scenario, carrier, save_file=True)

    scenario = 'scenario_'
    scenario2 = 'scenario_low_demand'
    #plot_outputs_carbon(scenario, scenario2)

    for scenario in scenarios:
        df = res_scenario.get_total("flow_conversion_output").xs(scenario).reset_index()
        df_input = res_scenario.get_total("flow_conversion_input").xs(scenario).reset_index()

        #plot_dataframe_on_map(df, df_input, 'technology', [0, #'1', '2',
                                                 #3, #'4', '5',
                                                 #6,
                                                 #8, #'9', '10', '11', '12',
                                                 #13,
                                                 #16,
                                                # 26],
                              #'nuts_data/NUTS_RG_20M_2021_4326.shp', save_png=True)

    scenario_1 = "scenario_"
    scenario = "scenario_"
    #for scenario in scenarios:
    #plot_npc(scenario, scenario_1)
    #plot_costs_with_unique_colors(scenario)


    scenario_1 = "scenario_CCS"
    scenario_2 = "scenario_hydrogen"
    scenario_3 = "scenario_electrification"
    #plot_cost_comp_npc(scenario, scenario_1, scenario_2, scenario_3)

    for scenario in scenarios:
        transport_data = res_scenario.get_total("flow_transport").xs(scenario).reset_index()
        shapefile_path = "nuts_data/NUTS_RG_20M_2021_4326.shp"
        output_data = res_scenario.get_total("flow_conversion_output").xs(scenario).reset_index()
        years = [0,
                 7,
                 #8,
                 #13,
                 16,
                 26
                 ]
        #for year in years:
         #   draw_transport_and_capture(transport_data, output_data, shapefile_path, year, scenario, figsize=(20, 20))
          #  draw_hydrogen_pipelines(transport_data, output_data, shapefile_path, year, scenario, figsize=(20, 20))
           # draw_transport_arrows_and_biomass_usage(transport_data, shapefile_path, year, scenario, figsize=(20, 20))


    #calc_lco(scenario = "scenario_", discount_rate = 0.06, carrier="cement")

    technologies = ['ASU', 'haber_bosch', 'e_haber_bosch', 'EAF', 'BF_BOF', 'DRI', 'SMR', 'methanol_synthesis', 'refinery',
                    'cement_plant', 'DAC']
    #for technology in technologies:
     #   plot_capacity_addition(folder_path, scenario, technology, save_file=False)
      #  plot_existing_capacity(folder_path, scenario, technology, save_file=False)

    #plot_carbon_capture(scenario='scenario_')
    #plot_storage_utilization(scenario = 'scenario_')
    source_path = "../outputs/hard_to_abate_scenarios_070524/"
    transport_techs = ['carbon_pipeline', 'hydrogen_pipeline', 'biomethane_transport', 'dry_biomass_truck']
    for tech in transport_techs:
        plot_biomass_and_transport(tech, source_path)
    plot_biomass_and_carbon_storage(source_path)
    carriers = ['electricity',
                'hydrogen']
    for carrier in carriers:
        plot_biomass_and_carrier(carrier, source_path)
