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
import pycountry
from matplotlib.colors import LogNorm


plt.rcParams.update({'font.size': 22})

res_scenario = Results("../outputs/hard_to_abate_scenarios_biomass_070624/")

'''res_ammonia = Results("../outputs/hard_to_abate_ammonia_280524")
res_cement = Results("../outputs/hard_to_abate_cement_280524")
res_methanol = Results("../outputs/hard_to_abate_methanol_290524")
res_steel = Results("../outputs/hard_to_abate_steel_290524")
res_refining = Results("../outputs/hard_to_abate_refining_290524")'''

def save_total(folder_path, scenario):
    os.makedirs(folder_path, exist_ok=True)

    scenario_data_input = res_scenario.get_total("flow_conversion_input").xs(scenario).groupby(['technology', 'carrier']).sum()
    file_path = os.path.join(folder_path, f"flow_conversion_input_{scenario}.csv")
    scenario_data_input.to_csv(file_path)

    scenario_data_output = res_scenario.get_total("flow_conversion_output").xs(scenario).groupby(['technology', 'carrier']).sum()
    file_path = os.path.join(folder_path, f"flow_conversion_output_{scenario}.csv")
    scenario_data_output.to_csv(file_path)

def save_imports_exports(folder_path, scenario):

    os.makedirs(folder_path, exist_ok=True)
    imports = res_scenario.get_total("flow_import").xs(scenario).groupby(['carrier']).sum()
    imports_grouped = imports.groupby(['carrier']).sum()
    file_path = os.path.join(folder_path, f"imports_grouped_{scenario}.csv")
    imports_grouped.to_csv(file_path)

    exports = res_scenario.get_total("flow_export").xs(scenario)
    exports_grouped = exports.groupby(['carrier']).sum()
    file_path = os.path.join(folder_path, f"exports_grouped_{scenario}.csv")
    exports_grouped.to_csv(file_path)

def energy_carrier(scenario):

    # prepare data for usage in draw wedges on map
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

                inputs_grouped_new.loc[(inputs_grouped_new['carrier'] == carrier) & (inputs_grouped_new['node'] == node), 0:26] -= row[0:26]

    inputs_grouped_new[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]] = inputs_grouped_new[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]].apply(pd.to_numeric, errors='coerce')

    for i in range(1, 27):
        inputs_grouped_new[i] = inputs_grouped_new[i].round(4)

    energy_carrier_country = pd.concat([inputs_grouped_new, summed_renewables])

    energy_carrier = energy_carrier_country.groupby(['carrier']).sum().reset_index()
    energy_carrier = energy_carrier.drop(['node'], axis=1)
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

    # place centroid france in main land france
    france_centroid = (3713381.55, 2686876.92)

    plt.rcParams['hatch.color'] = 'grey'
    plt.rcParams['hatch.linewidth'] = 0.4

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    gdf.plot(ax=ax, color=country_color, edgecolor=border_color)

    # hatch non EU countries
    for country_code in ['NO', 'UK', 'CH']:
        specific_gdf = gdf[gdf['CNTR_CODE'] == country_code]
        specific_gdf.plot(ax=ax, facecolor='lightgrey', hatch="\\\\", edgecolor='dimgrey', linewidth=0.8)

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
                print(f"No geographic data for {node} found.")
                continue
            country_geom = gdf.loc[gdf['NUTS_ID'] == node, 'geometry'].iloc[0]
            centroid = (country_geom.centroid.x, country_geom.centroid.y)

        # draw wedges
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
    # prepare data for usage in sankey diagram
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
                    df_outputs_updated.loc[output_idx, range(27)] -= df_inputs_updated.loc[input_idx, range(27)].values

        output_indices = df_outputs_updated[
            (df_outputs_updated['technology'] == tech_input) & (df_outputs_updated['carrier'] == carrier_input)].index

        input_indices = df_inputs[
            (df_inputs_updated['technology'] == tech_output) & (df_inputs_updated['carrier'] == carrier_output)].index

        if not output_indices.empty and not input_indices.empty:
            output_row = df_outputs_updated.loc[output_indices[0], range(27)]

            for idx in input_indices:
                df_inputs_updated.loc[idx, range(27)] -= output_row.values

    return df_outputs_updated, df_inputs_updated

def subtract_technology_pairs(df, tech_base, carrier_base, tech_compare, carrier_compare):
    # prepare data for usage in sankey diagram and remove dummy techs and flows
    idx_base = df[(df['technology'] == tech_base) & (df['carrier'] == carrier_base)].index
    idx_compare = df[(df['technology'] == tech_compare) & (df['carrier'] == carrier_compare)].index

    if not idx_base.empty and not idx_compare.empty:
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

    # combinations of dummy techs and the actual combination without the dummy tech
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
    updated_inputs_df['technology'] = updated_inputs_df['technology'].str.replace('biomass_to_coal_conversion', 'cement_plant')
    updated_inputs_df['technology'] = updated_inputs_df['technology'].str.replace('carbon_conversion', 'methanol_synthesis')
    technologies_to_remove = ['biomethane_SMR', 'biomethane_SMR_CCS', 'biomethane_haber_bosch', 'biomethane_DRI', 'scrap_conversion_EAF',
                              'hydrogen_for_cement_conversion', 'biomass_to_coal_conversion', 'carbon_conversion']
    updated_outputs_df = updated_outputs_df[~updated_outputs_df['technology'].isin(technologies_to_remove)]
    updated_inputs_df['technology'] = updated_inputs_df['technology'].str.replace('biomethane_', '')

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
        'carbon_liquefaction': 'CCS',
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
    fig.add_trace(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            label=unique_source_target,
            color=colors
        ),
        link=dict(
            color=[
                "rgba" + str(matplotlib.colors.to_rgba(
                    color_mapping.get(category_mapping.get(inv_mapping_dict[link], 'other_techs'), 'default_color'),
                    # Use a default color if not found
                    alpha=0.6
                )) for link in links['source']
            ],
            source=links_dict['source'],
            target=links_dict['target'],
            value=links_dict['value'],
            label=[f"{source} to {target}" for source, target in zip(links_dict['source'], links_dict['target'])],
            hovertemplate='%{value}'
        )
    ))

    if isinstance(year, str):
        year = int(year)
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
    smr_values = smr_hydrogen_row.iloc[0][[i for i in range(27)]].values

    smr_biomethane_hydrogen_row = df_output[
        (df_output['technology'] == 'SMR') & (df_output['carrier'] == 'hydrogen')]
    smr_biomethane_hydrogen_index = smr_biomethane_hydrogen_row.index[0]

    for i in range(27):
        df_output.at[smr_biomethane_hydrogen_index, i] -= smr_values[i]

    smr_hydrogen_row = df_output[(df_output['technology'] == 'SMR_CCS_biomethane') & (df_output['carrier'] == 'hydrogen')]
    smr_values = smr_hydrogen_row.iloc[0][[i for i in range(27)]].values

    smr_biomethane_hydrogen_row = df_output[
        (df_output['technology'] == 'SMR_CCS') & (df_output['carrier'] == 'hydrogen')]
    smr_biomethane_hydrogen_index = smr_biomethane_hydrogen_row.index[0]

    for i in range(27):
        df_output.at[smr_biomethane_hydrogen_index, i] -= smr_values[i]

    df_emissions_cumulative = res_scenario.get_total("carbon_emissions_cumulative").xs(scenario).reset_index()
    df_emissions_cumulative['year'] = df_emissions_cumulative['year'].apply(lambda x: 2024 + x)

    grouped_df = df_output[df_output['carrier'] == carrier]

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

        filtered_technologies = [tech for tech in grouped_df_values.columns if tech in technology_colors]
        grouped_df_values = grouped_df_values[filtered_technologies]

        palette = [technology_colors[tech] for tech in grouped_df_values.columns]

    else:
        palette = plt.cm.tab20(np.linspace(0, 1, len(grouped_df_values.columns)))

    fig, ax1 = plt.subplots(figsize=(18, 10))
    ax2 = ax1.twinx()

    bottom = np.zeros(len(grouped_df_values))

    for idx, technology in enumerate(grouped_df_values.columns):
        ax1.bar(grouped_df_values.index, grouped_df_values[technology], bottom=bottom, color=palette[idx], label=technology, width=0.6)
        bottom += grouped_df_values[technology].values

    ax2.plot(df_emissions_cumulative['year'], df_emissions_cumulative[scenario], color='black', label='Cumulative Emissions', marker='o')
    emissions_budget = 2324170  # Put carbon budget [kt] here
    ax2.axhline(y=emissions_budget, color='red', linestyle='--', label='Emissions Budget')
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

    y_labels_left = [f"{label / 1000}" for label in ax1.get_yticks()]
    ax1.set_yticklabels(y_labels_left)

    if save_file:
        subfolder_name = "output_bar_charts"
        subfolder_path = os.path.join(folder_path, subfolder_name)
        os.makedirs(subfolder_path, exist_ok=True)
        png_file_path = os.path.join(subfolder_path, f"{carrier}_bar_chart_{scenario}.png")
        plt.savefig(png_file_path, bbox_inches='tight')

    plt.show()

def plot_dataframe_on_map(df, df_input, technology_column, output_columns, shapefile_path=None, save_png=True):

    columns = ['node', 'technology', 'carrier'] + [year for year in range(0, 27)]
    df_output = df[df['carrier'] == 'hydrogen']

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

    columns_to_check = [i for i in range(27)]

    df_output = df_output.loc[df_output[columns_to_check].sum(axis=1) > 0]

    columns_to_operate = [i for i in range(27)]

    df_smr = df_output[df_output['technology'] == 'SMR'].set_index('node')
    df_smr_biomethane = df_output[df_output['technology'] == 'SMR_biomethane'].set_index('node')

    for node, row in df_smr.iterrows():
        if node in df_smr_biomethane.index:
            df_output.loc[(df_output['node'] == node) & (df_output['technology'] == 'SMR'), columns_to_operate] -= df_smr_biomethane.loc[
                node, columns_to_operate]

    columns_to_operate = [i for i in range(27)]

    df_smr_ccs = df_output[df_output['technology'] == 'SMR_CCS'].set_index('node')
    df_smr_ccs_biomethane = df_output[df_output['technology'] == 'SMR_CCS_biomethane'].set_index('node')

    for node, row in df_smr_ccs.iterrows():
        if node in df_smr_ccs_biomethane.index:
            df_output.loc[
                (df_output['node'] == node) & (df_output['technology'] == 'SMR_CCS'), columns_to_operate] -= \
            df_smr_ccs_biomethane.loc[
                node, columns_to_operate]

    columns_to_operate = [i for i in range(27)]
    df_output[columns_to_operate] = df_output[columns_to_operate].clip(lower=0)

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

        plt.axis('off')
        ax.set_xlim([2.5e6, 5.9e6])
        ax.set_ylim([1.5e6, 5.5e6])
        plt.tight_layout()
        if save_png == True:
            plt.savefig(f"{folder_path}/hydrogen_production_map_{scenario}_{year}.png", dpi=300, bbox_inches='tight', pad_inches=0)

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
    plt.savefig(f"{folder_path}/hydrogen_transport_map_{scenario}_{year}.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()

def draw_transport_arrows_and_biomass_usage(transport_data, shapefile_path, year, scenario, figsize=(20, 20)):

    df = transport_data
    transport_technologies = ['biomethane_transport', 'dry_biomass_truck']
    df = df[df['technology'].isin(transport_technologies)]

    availability_dry_biomass = res_scenario.get_total("availability_import").xs("dry_biomass", level = 'carrier').xs(scenario)

    availability_wet_biomass = res_scenario.get_total("availability_import").xs("wet_biomass", level='carrier').xs(scenario)

    import_dry_biomass = res_scenario.get_total("flow_import").xs("dry_biomass", level='carrier').xs(scenario)

    import_wet_biomass = res_scenario.get_total("flow_import").xs("wet_biomass", level='carrier').xs(scenario)

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

    nuts_gdf.plot(ax=ax, color="None", edgecolor=border_color)

    plt.rcParams['hatch.color'] = 'grey'
    plt.rcParams['hatch.linewidth'] = 0.4

    for country_code in ['NO', 'UK', 'CH']:
        specific_gdf = nuts_gdf[nuts_gdf['CNTR_CODE'] == country_code]
        specific_gdf.plot(ax=ax, facecolor='lightgrey', hatch="\\\\", edgecolor='dimgrey', linewidth=0.8)

    cbar.set_label('Usage of biomass potential', fontsize=22)

    cbar.ax.tick_params(labelsize=18)

    for _, row in df.iterrows():
        source = row['source']
        target = row['target']
        tech = row['technology']

        # Filter the dataframe for each specific technology and edge combination
        filtered_df = df[(df['source'] == source) & (df['target'] == target) & (df['technology'] == tech)]

        amount = filtered_df[year].sum()

        color = 'yellow' if tech == 'biomethane_transport' else 'crimson' if tech == 'dry_biomass_truck' else 'purple'

        if pd.notnull(amount) and amount > 0:
            linewidth = np.sqrt(amount) * 0.05
            if source in centroid_dict and target in centroid_dict:
                source_point = centroid_dict[source]
                target_point = centroid_dict[target]

                ax.annotate('', xy=(target_point.x, target_point.y), xytext=(source_point.x, source_point.y),
                            arrowprops=dict(arrowstyle='->, head_width=0.2, head_length=0.4', color=color,
                                            lw=linewidth))

    ax.set_xlim(2.5e6, 5.9e6)
    ax.set_ylim(1.5e6, 5.5e6)

    plt.axis('off')
    plt.savefig(f"{folder_path}/biomass_transport_map_{scenario}_{year}.png", dpi=300, bbox_inches='tight', pad_inches=0)

    plt.show()

def calc_lco(scenario, discount_rate, carrier):
    capex_df = res_scenario.get_total("cost_capex_total").xs(scenario).reset_index()
    opex_df = res_scenario.get_total("cost_opex_total").xs(scenario).reset_index()
    carrier_cost_df = res_scenario.get_total("cost_carrier").xs(scenario).reset_index()

    carrier_cost_df = carrier_cost_df[carrier_cost_df['carrier'] == carrier]
    carrier_cost_df.drop(['node', 'carrier'], axis=1, inplace=True)
    carrier_cost_df = carrier_cost_df.sum()

    production_df = res_scenario.get_total("flow_conversion_output").xs(scenario).reset_index()
    production_df = production_df[production_df['carrier'] == carrier]
    production_df = production_df.drop(['node', 'carrier'], axis=1).sum()

    total_discounted_costs = 0
    total_discounted_production = 0

    for year in range(27):
        discount_factor = (1 + discount_rate) ** year
        discounted_costs = (capex_df.loc[year, scenario] +
                            opex_df.loc[year, scenario] +
                            carrier_cost_df[year]) / discount_factor

        discounted_production = production_df[year] / discount_factor

        total_discounted_costs += discounted_costs
        total_discounted_production += discounted_production

    lcoa = total_discounted_costs / total_discounted_production if total_discounted_production else float('inf')
    lcoa = lcoa * 1000
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

    carbon_techs = ['BF_BOF_CCS', 'SMR_CCS', 'cement_plant_oxy_combustion', 'gasification_CCS', #'DAC'
                    ]
    methanol_techs = ['carbon_conversion']
    DAC = ['DAC']
    outputs = ['carbon', #'carbon_liquid'
               ]
    methanol_outputs = ['carbon_methanol']
    output_DAC = ['carbon_liquid']

    def prepare_df(source_path):
        results = Results(source_path)
        df = results.get_total('flow_conversion_output').xs(scenario).reset_index()
        df_methanol = df[df['technology'].isin(methanol_techs) & df['carrier'].isin(methanol_outputs)]
        df_carbon = df[df['technology'].isin(carbon_techs) & df['carrier'].isin(outputs)]
        df_DAC = df[df['technology'].isin(DAC) & df['carrier'].isin(output_DAC)]
        df_carbon = df_carbon.drop('node', axis=1)
        df_methanol = df_methanol.drop('node', axis=1)
        df_DAC = df_DAC.drop('node', axis=1)
        df_carbon = df_carbon.groupby(['technology', 'carrier']).sum().reset_index()
        df_methanol = df_methanol.groupby(['technology', 'carrier']).sum().reset_index()
        df_DAC = df_DAC.groupby(['technology', 'carrier']).sum().reset_index()
        return df_carbon, df_methanol, df_DAC

    carbon_integrated, methanol_integrated, DAC_integrated = prepare_df("../outputs/hard_to_abate_scenarios_240524/")
    carbon_ammonia, methanol_ammonia, DAC_ammonia = prepare_df("../outputs/hard_to_abate_ammonia_280524/")
    carbon_steel, methanol_steel, DAC_steel = prepare_df("../outputs/hard_to_abate_steel_290524/")
    carbon_cement, methanol_cement, DAC_cement = prepare_df("../outputs/hard_to_abate_cement_280524/")
    carbon_methanol, methanol_methanol, DAC_methanol = prepare_df("../outputs/hard_to_abate_methanol_290524/")
    carbon_refining, methanol_refining, DAC_refining = prepare_df("../outputs/hard_to_abate_refining_290524/")
    def melt_and_label(df, label):
        df_long = pd.melt(df.reset_index(), id_vars=['technology', 'carrier'], var_name='Year', value_name='Value')
        if label == 'Integrated':
            industry_mapping = {
                'BF_BOF_CCS': 'Steel_integrated',
                'SMR_CCS': 'Hydrogen_integrated',
                'cement_plant_oxy_combustion': 'Cement_integrated',
                'gasification_CCS': 'Hydrogen_integrated',
                'DAC': 'DAC_integrated',
                'carbon_conversion': 'Methanol_integrated'
            }
            df_long['Source'] = df_long['technology'].map(industry_mapping)
        else:
            df_long['Source'] = label
        return df_long

    combined_integrated = melt_and_label(carbon_integrated, 'Integrated')

    combined_single = pd.concat([
        melt_and_label(carbon_ammonia, 'Ammonia'),
        melt_and_label(carbon_steel, 'Steel'),
        melt_and_label(carbon_cement, 'Cement'),
        melt_and_label(carbon_methanol, 'Methanol'),
        melt_and_label(carbon_refining, 'Refining')
    ])

    combined_DAC = pd.concat([
        melt_and_label(DAC_ammonia, 'Ammonia'),
        melt_and_label(DAC_steel, 'Steel'),
        melt_and_label(DAC_cement, 'Cement'),
        melt_and_label(DAC_methanol, 'Methanol'),
        melt_and_label(DAC_refining, 'Refining')
    ])

    combined_DAC_integrated = melt_and_label(DAC_integrated, 'Integrated')
    combined_methanol_integrated = melt_and_label(methanol_integrated, 'Integrated')

    pivot_df_integrated = combined_integrated.pivot_table(index='Year', columns=['Source'], values='Value', aggfunc='sum')
    pivot_df_DAC_integrated = combined_DAC_integrated.pivot_table(index='Year', columns=['Source'], values='Value', aggfunc='sum')
    pivot_df_single = combined_single.pivot_table(index='Year', columns=['Source'], values='Value', aggfunc='sum')
    pivot_df_DAC = combined_DAC.pivot_table(index='Year', columns=['Source'], values='Value', aggfunc='sum')
    pivot_df_DAC['DAC'] = pivot_df_DAC.sum(axis=1)
    pivot_df_methanol_integrated = combined_methanol_integrated.pivot_table(index='Year', columns=['Source'], values='Value', aggfunc='sum')

    color_map = {
        'Steel_integrated': 'tab:blue',
        'Steel': 'tab:blue',
        'Cement_integrated': 'tab:green',
        'Cement': 'tab:green',
        'Hydrogen_integrated': 'tab:cyan',
        'Ammonia': 'tab:pink',
        'Methanol': 'tab:olive',
        'Methanol_integrated': 'None',
        'DAC': 'tab:orange',
        'DAC_integrated': 'tab:orange',
        'Refining_integrated': 'tab:red',
        'Refining': 'tab:red'
    }

    results = Results("../outputs/hard_to_abate_scenarios_240524/")
    ratios_df = results.get_total("flow_conversion_input").xs(scenario).reset_index()
    ratios_df = ratios_df.drop('node', axis=1)
    ratios_df = ratios_df.groupby(['technology', 'carrier']).sum().reset_index()
    ratios_df = ratios_df[(ratios_df['carrier'] == 'hydrogen')]
    ratios_df = ratios_df.loc[ratios_df['technology'] != 'hydrogen_compressor_low']
    total_hydrogen_per_year = ratios_df.iloc[:, 2:].sum()

    industry_mapping = {
        'e_haber_bosch': 'Ammonia',
        'haber_bosch': 'Ammonia',
        'h2_to_ng': 'Steel',
        'hydrogen_for_cement_conversion': 'Cement',
        'methanol_synthesis': 'Methanol',
        'refinery': 'Refining'
    }

    ratios_df['industry'] = ratios_df['technology'].map(industry_mapping)
    industry_totals = ratios_df.groupby('industry').sum()

    ratios_df = industry_totals.div(total_hydrogen_per_year).transpose()
    ratios_df.reset_index(drop=True, inplace=True)
    ratios_df.columns.name = None
    ratios_df = ratios_df.dropna(how='all')

    ratios_df.reset_index(drop=True, inplace=True)
    new_order= ['Ammonia', 'Refining', 'Methanol', 'Cement', 'Steel']
    ratios_df = ratios_df[new_order]

    years = np.arange(2024, 2051, 2)
    start_year = 2024
    years = np.arange(start_year, start_year + 27, 2)
    indices = np.arange(0, 27, 2)

    pivot_df_DAC = pivot_df_DAC.loc[indices]
    pivot_df_integrated = pivot_df_integrated.loc[indices]
    pivot_df_DAC_integrated = pivot_df_DAC_integrated.loc[indices]
    pivot_df_single = pivot_df_single.loc[indices]
    pivot_df_methanol_integrated = pivot_df_methanol_integrated.loc[indices]

    ratios_df = ratios_df.loc[indices]

    fig, ax = plt.subplots(figsize=(20, 10))
    width = 0.35
    x = np.arange(len(years))

    industries = ['Hydrogen_integrated', 'Cement_integrated', 'Steel_integrated']
    bottoms = np.zeros(len(years))

    hydrogen_color_map = {
        'Ammonia': 'tab:pink',
        'Refining': 'tab:red',
        'Cement': 'tab:green',
        'Steel': 'tab:blue',
        'Methanol': 'tab:olive'
    }

    hydrogen_values = pivot_df_integrated['Hydrogen_integrated'].fillna(0)

    if 'index' in pivot_df_integrated.index:
        pivot_df_integrated = pivot_df_integrated.drop('index')

    if 'index' in pivot_df_DAC_integrated.index:
        pivot_df_DAC_integrated = pivot_df_DAC_integrated.drop('index')

    if 'index' in pivot_df_single.index:
        pivot_df_single = pivot_df_single.drop('index')

    if 'index' in pivot_df_DAC.index:
        pivot_df_DAC = pivot_df_DAC.drop('index')

    if 'index' in pivot_df_methanol_integrated.index:
        pivot_df_methanol_integrated = pivot_df_methanol_integrated.drop('index')

    for industry in industries:
        values = pivot_df_integrated[industry].fillna(0)
        ax.bar(x - width / 2, values, width,
               color=color_map[industry], bottom=bottoms)

        bottoms += values

    values_DAC = pivot_df_DAC_integrated['DAC_integrated'].fillna(0)
    ax.bar(x - width / 2, values_DAC, width,
           color=color_map['DAC_integrated'], bottom=bottoms)

    bottoms += values_DAC

    values_methanol = pivot_df_methanol_integrated['Methanol_integrated'].fillna(0)
    ax.bar(x - width / 2, values_methanol, width,
           color=color_map['Methanol_integrated'], bottom=(bottoms- values_methanol), hatch='//', edgecolor='darkblue')

    x = np.arange(len(ratios_df))
    bottoms_hydrogen = np.zeros(len(ratios_df))

    for idx in ratios_df.index:
        for industry in ratios_df.columns:
            ratio = ratios_df.loc[idx, industry]
            industry_values = hydrogen_values.loc[idx] * ratio
            color = hydrogen_color_map.get(industry, 'gray')
            ax.bar(x[idx // 2] - width / 2, industry_values, width, color=color, bottom=bottoms_hydrogen[idx // 2])
            bottoms_hydrogen[idx // 2] += industry_values

    ax.bar(x + width / 2, pivot_df_single['Ammonia'].fillna(0), width, label='Ammonia', color=color_map['Ammonia'], bottom=0)
    bottom_ammonia = pivot_df_single['Ammonia'].fillna(0)

    ax.bar(x + width / 2, pivot_df_single['Refining'].fillna(0), width, label='Refining', color=color_map['Refining'],
           bottom=bottom_ammonia)
    bottom_refining = bottom_ammonia + pivot_df_single['Refining'].fillna(0)

    ax.bar(x + width / 2, pivot_df_single['Methanol'].fillna(0), width, label='Methanol', color=color_map['Methanol'],
           bottom=bottom_refining)
    bottom_methanol = bottom_refining + pivot_df_single['Methanol'].fillna(0)

    ax.bar(x + width / 2, pivot_df_single['Cement'].fillna(0), width, label='Cement', color=color_map['Cement'],
           bottom=bottom_methanol)
    bottom_cement = bottom_methanol + pivot_df_single['Cement'].fillna(0)

    ax.bar(x + width / 2, pivot_df_single['Steel'].fillna(0), width, label='Steel', color=color_map['Steel'],
           bottom=bottom_cement)
    bottom_steel = bottom_cement + pivot_df_single['Steel'].fillna(0)

    ax.bar(x + width / 2, pivot_df_DAC['DAC'].fillna(0), width, label='DAC', color=color_map['DAC'], bottom=bottom_steel)

    ax.axhline(y=103604.52, color='r', linestyle='--', linewidth=2, label='Capacity limit carbon storage')

    x_positions = np.arange(len(years))
    plt.xticks(x_positions, years, rotation=0, fontsize=12)

    def format_func(value, tick_number):
        return int(value / 1000)

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_func))

    plt.xlabel('Year')
    ax.set_ylabel('Captured Carbon [Mt p. a.]')
    ax.set_title('Captured carbon by industry per year')
    ax.legend()

    plt.show()

    fig.savefig(f'captured_carbon_by_industry_{scenario}.svg', format='svg')

def plot_biomass_per_industry(scenario):

    results_integrated = Results("../outputs/hard_to_abate_scenarios_240524/")
    results_ammonia = Results("../outputs/hard_to_abate_ammonia_280524")
    results_steel = Results("../outputs/hard_to_abate_steel_290524")
    results_cement = Results("../outputs/hard_to_abate_cement_280524")
    results_methanol = Results("../outputs/hard_to_abate_methanol_290524")
    results_refining = Results("../outputs/hard_to_abate_refining_290524")
    def biomass_usage(results):
        df_dry_biomass = results.get_total('flow_conversion_input').xs(scenario).xs("dry_biomass", level='carrier').sum()
        df_wet_biomass = results.get_total('flow_conversion_input').xs(scenario).xs("wet_biomass", level='carrier').sum()
        return df_dry_biomass, df_wet_biomass

    def biomass_cement_usage(results):
        df_biomass_cement = results.get_total('flow_conversion_input').xs(scenario).xs("biomass_cement",level='carrier').sum()
        return df_biomass_cement

    def biomass_availability(results):
        df_dry_biomass_av = results.get_total('availability_import').xs(scenario).xs("dry_biomass", level='carrier').sum()
        df_wet_biomass_av = results.get_total('availability_import').xs(scenario).xs("wet_biomass", level='carrier').sum()
        return df_dry_biomass_av, df_wet_biomass_av

    def biomass_cement_availability(results):
        df_biomass_cement_av = results.get_total('availability_import').xs(scenario).xs("biomass_cement", level='carrier').sum()
        return df_biomass_cement_av

    dry_biomass_ammonia, wet_biomass_ammonia = biomass_usage(results_ammonia)
    dry_biomass_steel, wet_biomass_steel = biomass_usage(results_steel)
    dry_biomass_cement, wet_biomass_cement= biomass_usage(results_cement)
    dry_biomass_methanol, wet_biomass_methanol= biomass_usage(results_methanol)
    dry_biomass_refining, wet_biomass_refining = biomass_usage(results_refining)

    biomass_cement_cement = biomass_cement_usage(results_cement)

    dry_biomass_av_ammonia, wet_biomass_av_ammonia = biomass_availability(results_ammonia)
    dry_biomass_av_steel, wet_biomass_av_steel = biomass_availability(results_steel)
    dry_biomass_av_cement, wet_biomass_av_cement = biomass_availability(results_cement)
    dry_biomass_av_methanol, wet_biomass_av_methanol = biomass_availability(results_methanol)
    dry_biomass_av_refining, wet_biomass_av_refining = biomass_availability(results_refining)

    biomass_cement_av_cement = biomass_cement_availability(results_cement)

    # single industries
    biomass_cement = ((dry_biomass_cement / dry_biomass_av_cement) +
                    (wet_biomass_cement / wet_biomass_av_cement) +
                    (biomass_cement_cement / biomass_cement_av_cement)) / 3

    biomass_ammonia = ((dry_biomass_ammonia / dry_biomass_av_ammonia) +
                       (wet_biomass_ammonia / wet_biomass_av_ammonia)) / 2

    biomass_steel = ((dry_biomass_steel / dry_biomass_av_steel) +
                     (wet_biomass_steel / wet_biomass_av_steel)) / 2

    biomass_methanol = ((dry_biomass_methanol / dry_biomass_av_methanol) +
                        (wet_biomass_methanol / wet_biomass_av_methanol)) / 2

    biomass_refining = ((dry_biomass_refining / dry_biomass_av_refining) +
                        (wet_biomass_refining / wet_biomass_av_refining)) / 2

    print(biomass_steel)

    # integrated industry

    dry_biomass_integrated, wet_biomass_integrated = biomass_usage(results_integrated)
    dry_biomass_av_integrated, wet_biomass_av_integrated = biomass_availability(results_integrated)
    biomass_cement_integrated = biomass_cement_usage(results_integrated)
    biomass_cement_av_integrated = biomass_cement_availability(results_integrated)

    biomass_potential = (dry_biomass_integrated /dry_biomass_av_integrated + wet_biomass_integrated / wet_biomass_av_integrated + biomass_cement_integrated / biomass_cement_av_integrated) / 3
    #print(biomass_potential)

    biomass_potential_single = biomass_ammonia + biomass_methanol + biomass_refining + biomass_steel + biomass_cement
    #print('biomass_usage_single', biomass_potential_single)

    total_h2_dry_biomass = results_integrated.get_total('flow_conversion_output').xs(scenario).xs("gasification", level='technology').sum() + results_integrated.get_total('flow_conversion_output').xs(scenario).xs("gasification_CCS", level='technology').sum()
    total_h2 = results_integrated.get_total('flow_conversion_output').xs(scenario).xs("hydrogen", level='carrier').reset_index().drop('node', axis=1)
    total_h2 = total_h2.groupby('technology').sum()
    total_h2 = total_h2.query("technology != 'hydrogen_decompressor'")
    total_h2 = total_h2.sum()
    ratio_h2_dry_biomass = total_h2_dry_biomass / total_h2

    ratios_df = results_integrated.get_total("flow_conversion_input").xs(scenario).xs('hydrogen', level ='carrier').groupby('technology').sum().reset_index()

    ratios_df = ratios_df.loc[ratios_df['technology'] != 'hydrogen_compressor_low']
    total_hydrogen_per_year = ratios_df.sum().transpose()

    industry_mapping = {
        'e_haber_bosch': 'Ammonia',
        'haber_bosch': 'Ammonia',
        'h2_to_ng': 'Steel',
        'hydrogen_for_cement_conversion': 'Cement',
        'methanol_synthesis': 'Methanol',
        'refinery': 'Refining'
    }

    ratios_df['industry'] = ratios_df['technology'].map(industry_mapping)
    industry_totals = ratios_df.groupby('industry').sum()
    industry_totals = industry_totals.drop('technology', axis=1)
    ratios_hydrogen = industry_totals.divide(total_hydrogen_per_year, axis=1)
    ratios_df_dry_biomass = ratios_hydrogen * dry_biomass_integrated * ratio_h2_dry_biomass
    ratios_df_dry_biomass = ratios_df_dry_biomass / dry_biomass_av_integrated
    ratios_df_dry_biomass = ratios_df_dry_biomass.transpose()
    ratios_df_dry_biomass.reset_index(drop=True, inplace=True)
    ratios_df_dry_biomass.columns.name = None
    ratios_df_dry_biomass = ratios_df_dry_biomass.dropna(how='all')

    ratios_df_dry_biomass.reset_index(drop=True, inplace=True)

    # integrated industry wet biomass

    total_biomethane = results_integrated.get_total('flow_conversion_output').xs(scenario).xs("anaerobic_digestion", level='technology').reset_index().drop(['node', 'carrier'], axis=1).sum()
    ratio_biomethane = results_integrated.get_total("flow_conversion_input").xs(scenario).xs('biomethane', level='carrier').reset_index()
    ratio_biomethane = ratio_biomethane.groupby('technology').sum().reset_index()

    industry_mapping_biomethane = {
        'biomethane_DRI': 'Steel',
        'biomethane_SMR': 'Hydrogen',
        'biomethane_SMR_CCS': 'Hydrogen',
        'biomethane_haber_bosch': 'Ammonia',
    }

    ratio_biomethane['industry'] = ratio_biomethane['technology'].map(industry_mapping_biomethane)
    industry_totals_biomethane = ratio_biomethane.groupby('industry').sum()
    industry_totals_biomethane = industry_totals_biomethane.drop('technology', axis=1)
    ratios_df_biomethane = industry_totals_biomethane.divide(total_biomethane, axis=1)

    hydrogen_biomethane = ratios_df_biomethane.loc['Hydrogen']
    biomethane_distribution = pd.DataFrame()

    for industry in ratios_hydrogen.index:
        biomethane_distribution[industry] = hydrogen_biomethane * ratios_hydrogen.loc[industry]

    biomethane_distribution = biomethane_distribution.transpose()

    df_wet_biomass = pd.concat([biomethane_distribution, ratios_df_biomethane], axis=0)
    df_wet_biomass = df_wet_biomass.loc[df_wet_biomass.index != 'Hydrogen']
    df_wet_biomass = df_wet_biomass.groupby(df_wet_biomass.index).sum()
    ratios_df_wet_biomass = df_wet_biomass * wet_biomass_integrated
    ratios_df_wet_biomass = ratios_df_wet_biomass / wet_biomass_av_integrated

    ratios_df_wet_biomass = ratios_df_wet_biomass.transpose()
    ratios_df_wet_biomass.reset_index(drop=True, inplace=True)
    ratios_df_wet_biomass.columns.name = None
    ratios_df_wet_biomass = ratios_df_wet_biomass.dropna(how='all')

    ratios_df_wet_biomass.reset_index(drop=True, inplace=True)

    biomass_ammonia_integrated = (ratios_df_dry_biomass['Ammonia'] + ratios_df_wet_biomass['Ammonia']) /2
    biomass_steel_integrated = (ratios_df_dry_biomass['Steel'] + ratios_df_wet_biomass['Steel']) / 2
    biomass_methanol_integrated = (ratios_df_dry_biomass['Methanol'] + ratios_df_wet_biomass['Methanol']) / 2
    biomass_refining_integrated = (ratios_df_dry_biomass['Refining'] + ratios_df_wet_biomass['Refining']) / 2
    biomass_cement_integrated = (biomass_cement_integrated / biomass_cement_av_integrated + ratios_df_dry_biomass['Cement'] + ratios_df_wet_biomass['Cement']) / 3

    biomass_ammonia.name = 'Ammonia'
    biomass_cement.name = 'Cement'
    biomass_methanol.name = 'Methanol'
    biomass_steel.name = 'Steel'
    biomass_refining.name = 'Refining'

    biomass_ammonia_integrated.name = 'Ammonia_Integrated'
    biomass_cement_integrated.name = 'Cement_Integrated'
    biomass_methanol_integrated.name = 'Methanol_Integrated'
    biomass_steel_integrated.name = 'Steel_Integrated'
    biomass_refining_integrated.name = 'Refining_Integrated'

    non_integrated_frames = [biomass_ammonia, biomass_cement, biomass_methanol, biomass_steel, biomass_refining]
    non_integrated = pd.concat(non_integrated_frames, axis=1)
    non_integrated = non_integrated.loc[non_integrated.index != 'technology']

    integrated_frames = [biomass_ammonia_integrated, biomass_cement_integrated, biomass_methanol_integrated,
                         biomass_steel_integrated, biomass_refining_integrated]

    integrated = pd.concat(integrated_frames, axis=1)
    integrated = integrated.loc[integrated.index != 'technology']

    start_year = 2024
    years = np.arange(start_year, start_year + 27, 2)
    indices = np.arange(0, 27, 2)

    integrated = integrated.loc[indices]
    non_integrated = non_integrated.loc[indices]

    width = 0.35
    fig, ax = plt.subplots(figsize=(20, 10))
    x = np.arange(len(years))

    bottoms = np.zeros(len(years))

    color_map = {
        'Ammonia': 'tab:pink',
        'Refining': 'tab:red',
        'Cement': 'tab:green',
        'Steel': 'tab:blue',
        'Methanol': 'tab:olive'
    }
    industries = ['Ammonia', 'Cement', 'Methanol', 'Steel', 'Refining']

    for industry in industries:
        values = non_integrated[industry].fillna(0)
        ax.bar(x + width / 2, values, width,
               color=color_map[industry], bottom=bottoms)

        bottoms += values

    bottoms = np.zeros(len(years))

    for industry in industries:
        values = integrated[f"{industry}_Integrated"].fillna(0)
        ax.bar(x - width / 2, values, width,
               color=color_map[industry], bottom=bottoms, label=industry)

        bottoms += values

    x_positions = np.arange(len(years))
    plt.xticks(x_positions, years, rotation=0, fontsize=12)

    plt.xlabel('Year', fontsize=22)
    ax.set_ylabel('Usage of biomass potential per year', fontsize=22)
    ax.set_title('Usage of biomass potential by industry per year')
    ax.legend(frameon=False)

    plt.show()

    fig.savefig(f'usage_biomass_potential_{scenario}.svg', format='svg')

def prepare_scenarios():
    scenario_labels = {
        'scenario_': 'Baseline',
        'scenario_biomass_share_0.7': '70% import availability',
        'scenario_biomass_share_0.6': '60% import availability',
        'scenario_biomass_share_0.5': '50% import availability',
        'scenario_biomass_share_0.4': '40% import availability',
        'scenario_biomass_share_0.3': '30% import availability',
        'scenario_biomass_share_0.2': '20% import availability',
        'scenario_biomass_share_0.1': '10% import availability',
        'scenario_biomass_share_0.0': 'No import availability',
    }

    return scenario_labels

def plot_data(year_data, year_labels, marker_sizes, title, ylabel, xlabel):
    colors = plt.cm.jet_r(np.linspace(0, 1, len(scenarios)))
    fig, ax = plt.subplots()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    for year, data in year_data.items():
        if len(data['x']) > 1:
            ax.plot(data['x'], data['y'], 'k-', label=f'{year_labels[year]}', zorder=1)
            ax.annotate(year_labels[year], (data['x'][0], data['y'][0]),
                        textcoords="offset points", xytext=(25, -10), ha='right', fontsize=9, color='black')

        ax.scatter(data['x'], data['y'], s=marker_sizes[year], c=[colors[scenarios.index(s)] for s in data['scenario']],
                   zorder=2)

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10)

    custom_lines = [plt.Line2D([0], [0], color=colors[i], lw=4) for i in range(len(scenarios))]
    ax.legend(custom_lines, [prepare_scenarios()[s] for s in scenarios], fontsize=8, loc='best', frameon=False)

    plt.tight_layout()
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.show()

def plot_biomass_and_transport(tech, results, years):
    df = results

    special_years = [6, 16, 26]
    year_labels = {6: '2030', 16: '2040', 26: '2050'}
    marker_sizes = {6: 25, 16: 25, 26: 25}

    year_data = {year: {'x': [], 'y': [], 'scenario': []} for year in special_years}

    for scenario in scenarios:
        for year in years:
            y_point = df.get_total("flow_transport").xs(scenario).reset_index()
            y_point = y_point[y_point['technology'] == tech]
            y_point = y_point.groupby('technology').sum()

            availability_sum = (
                df.get_total("availability_import").xs(scenario).xs('dry_biomass', level='carrier') +
                df.get_total("availability_import").xs(scenario).xs('wet_biomass', level='carrier') +
                df.get_total("availability_import").xs(scenario).xs('biomass_cement', level='carrier')
            )
            total_availability = availability_sum.sum()
            if total_availability.sum() == 0:
                x_point[year] = 0
            else:
                x_point = (df.get_total("flow_import").xs(scenario).xs("wet_biomass", level='carrier') /
                           df.get_total("availability_import").xs('scenario_').xs('wet_biomass', level='carrier') +
                           df.get_total("flow_import").xs(scenario).xs("dry_biomass", level='carrier') /
                           df.get_total("availability_import").xs('scenario_').xs('dry_biomass', level='carrier') +
                           df.get_total("flow_import").xs(scenario).xs("biomass_cement", level='carrier') /
                           df.get_total("availability_import").xs('scenario_').xs('biomass_cement', level='carrier')) / 3

            if year in special_years:
                year_data[year]['x'].append(x_point.mean()[year] * 100)
                year_data[year]['y'].append(y_point.sum()[year])
                year_data[year]['scenario'].append(scenario)

    title = f'Comparison of {tech} transport by scenario and year'
    ylabel = f'Transported amounts via {tech} per year'
    xlabel = 'Usage of biomass potential (%)'
    plot_data(year_data, year_labels, marker_sizes, title, ylabel, xlabel)

def plot_biomass_and_carbon_storage(results, years):
    df = results
    tech = 'carbon_storage'
    output = ['carbon_liquid']

    special_years = [6, 16, 26]
    year_labels = {6: '2030', 16: '2040', 26: '2050'}
    marker_sizes = {6: 25, 16: 25, 26: 25}

    year_data = {year: {'x': [], 'y': [], 'scenario': []} for year in special_years}

    for scenario in scenarios:
        for year in years:
            y_point = df.get_total("flow_conversion_input").xs(scenario).reset_index()
            y_point = y_point[y_point['technology'].isin([tech])]
            y_point = y_point[y_point['carrier'].isin(output)]
            y_point = y_point.groupby(['technology', 'carrier']).sum()

            availability_sum = (
                    df.get_total("availability_import").xs(scenario).xs('dry_biomass', level='carrier') +
                    df.get_total("availability_import").xs(scenario).xs('wet_biomass', level='carrier') +
                    df.get_total("availability_import").xs(scenario).xs('biomass_cement', level='carrier')
            )
            total_availability = availability_sum.sum()
            print(total_availability)
            if total_availability.sum() == 0:
                x_point[year] = 0
            else:
                x_point = (df.get_total("flow_import").xs(scenario).xs("wet_biomass", level='carrier') /
                           df.get_total("availability_import").xs('scenario_').xs('wet_biomass', level='carrier') +
                           df.get_total("flow_import").xs(scenario).xs("dry_biomass", level='carrier') /
                           df.get_total("availability_import").xs('scenario_').xs('dry_biomass', level='carrier') +
                           df.get_total("flow_import").xs(scenario).xs("biomass_cement", level='carrier') /
                           df.get_total("availability_import").xs('scenario_').xs('biomass_cement', level='carrier')) / 3

            if year in special_years:
                year_data[year]['x'].append(x_point.mean()[year] * 100)
                year_data[year]['y'].append(y_point.sum()[year] / 1000)
                year_data[year]['scenario'].append(scenario)

    title = f'Usage of {tech} capacity and biomass potential from 2024 to 2050'
    ylabel = f'Usage of {tech} capacity [Mt p. a.]'
    xlabel = 'Usage of biomass potential (%)'
    plot_data(year_data, year_labels, marker_sizes, title, ylabel, xlabel)

def plot_biomass_and_captured_carbon(results, years):
    df = results
    tech = ['DAC', 'SMR_CCS', 'gasification_CCS', 'BF_BOF_CCS', 'cement_plant_oxy_combustion']
    output = ['carbon_liquid', 'carbon']

    special_years = [6, 16, 26]
    year_labels = {6: '2030', 16: '2040', 26: '2050'}
    marker_sizes = {6: 25, 16: 25, 26: 25}

    year_data = {year: {'x': [], 'y': [], 'scenario': []} for year in special_years}

    for scenario in scenarios:
        for year in years:
            y_point = df.get_total("flow_conversion_output").xs(scenario).reset_index()
            y_point = y_point[y_point['technology'].isin(tech)]
            y_point = y_point[y_point['carrier'].isin(output)]
            y_point = y_point.groupby(['technology', 'carrier']).sum()

            availability_sum = (
                    df.get_total("availability_import").xs('scenario_').xs('dry_biomass', level='carrier') +
                    df.get_total("availability_import").xs('scenario_').xs('wet_biomass', level='carrier') +
                    df.get_total("availability_import").xs('scenario_').xs('biomass_cement', level='carrier')
            )
            total_availability = availability_sum.sum()
            if total_availability.sum() == 0:
                x_point[year] = 0
            else:
                x_point = (df.get_total("flow_import").xs(scenario).xs("wet_biomass", level='carrier') /
                            df.get_total("availability_import").xs('scenario_').xs('wet_biomass', level='carrier') +
                            df.get_total("flow_import").xs(scenario).xs("dry_biomass", level='carrier') /
                            df.get_total("availability_import").xs('scenario_').xs('dry_biomass', level='carrier') +
                            df.get_total("flow_import").xs(scenario).xs("biomass_cement", level='carrier') /
                            df.get_total("availability_import").xs('scenario_').xs('biomass_cement', level='carrier')) / 3

            if year in special_years:
                year_data[year]['x'].append(x_point.mean()[year] * 100)
                year_data[year]['y'].append(y_point.sum()[year] / 1000)
                year_data[year]['scenario'].append(scenario)

    title = 'Captured carbon and biomass potential from 2024 to 2050'
    ylabel = f'Captured carbon [Mt p. a.]'
    xlabel = 'Usage of biomass potential (%)'
    plot_data(year_data, year_labels, marker_sizes, title, ylabel, xlabel)

def plot_biomass_and_hydrogen(carrier, technology, results, years):
    df = results

    special_years = [6, 16, 26]
    year_labels = {6: '2030', 16: '2040', 26: '2050'}
    marker_sizes = {6: 25, 16: 25, 26: 25}

    year_data = {year: {'x': [], 'y': [], 'scenario': []} for year in special_years}

    for scenario in scenarios:
        for year in years:
            y_point = df.get_total("flow_conversion_output").xs(scenario).reset_index()
            y_point = y_point[y_point['carrier'].isin([carrier])]
            y_point = y_point[y_point['technology'].isin(technology)]
            y_point = y_point.drop(['node', 'technology'], axis=1)
            y_point = y_point.groupby('carrier').sum()

            availability_sum = (
                    df.get_total("availability_import").xs(scenario).xs('dry_biomass', level='carrier') +
                    df.get_total("availability_import").xs(scenario).xs('wet_biomass', level='carrier') +
                    df.get_total("availability_import").xs(scenario).xs('biomass_cement', level='carrier')
            )
            total_availability = availability_sum.sum()
            if total_availability.sum() == 0:
                x_point[year] = 0
            else:
                x_point = (df.get_total("flow_import").xs(scenario).xs("wet_biomass", level='carrier') /
                           df.get_total("availability_import").xs('scenario_').xs('wet_biomass', level='carrier') +
                           df.get_total("flow_import").xs(scenario).xs("dry_biomass", level='carrier') /
                           df.get_total("availability_import").xs('scenario_').xs('dry_biomass', level='carrier') +
                           df.get_total("flow_import").xs(scenario).xs("biomass_cement", level='carrier') /
                           df.get_total("availability_import").xs('scenario_').xs('biomass_cement', level='carrier')) / 3

            if year in special_years:
                year_data[year]['x'].append(x_point.mean()[year] * 100)
                year_data[year]['y'].append(y_point.sum()[year] / 1000)
                year_data[year]['scenario'].append(scenario)

    title = f'Usage of {carrier} vs. usage of biomass potential from 2024 to 2050'
    ylabel = f'Usage of {carrier} [TWh p. a.]'
    xlabel = 'Usage of biomass potential (%)'
    plot_data(year_data, year_labels, marker_sizes, title, ylabel, xlabel)

def plot_biomass_and_electricity(carrier, results, years):
    df = results

    colors = plt.cm.jet_r(np.linspace(0, 1, len(scenarios)))
    special_years = [6, 16, 26]
    year_labels = {6: '2030', 16: '2040', 26: '2050'}
    marker_sizes = {6: 25, 16: 25, 26: 25}

    year_data = {year: {'x': [], 'y': [], 'scenario': []} for year in special_years}

    for scenario in scenarios:
        for year in years:
            y_point = df.get_total("flow_conversion_input").xs(scenario).reset_index()
            y_point = y_point[y_point['carrier'].isin([carrier])]
            y_point = y_point.drop(['node', 'technology'], axis=1)
            y_point = y_point.groupby('carrier').sum()

            availability_sum = (
                    df.get_total("availability_import").xs(scenario).xs('dry_biomass', level='carrier') +
                    df.get_total("availability_import").xs(scenario).xs('wet_biomass', level='carrier') +
                    df.get_total("availability_import").xs(scenario).xs('biomass_cement', level='carrier')
            )
            total_availability = availability_sum.sum()
            if total_availability.sum() == 0:
                x_point[year] = 0
            else:
                x_point = (df.get_total("flow_import").xs(scenario).xs("wet_biomass", level='carrier') /
                           df.get_total("availability_import").xs('scenario_').xs('wet_biomass', level='carrier') +
                           df.get_total("flow_import").xs(scenario).xs("dry_biomass", level='carrier') /
                           df.get_total("availability_import").xs('scenario_').xs('dry_biomass', level='carrier') +
                           df.get_total("flow_import").xs(scenario).xs("biomass_cement", level='carrier') /
                           df.get_total("availability_import").xs('scenario_').xs('biomass_cement', level='carrier')) / 3

            if year in special_years:
                year_data[year]['x'].append(x_point.mean()[year] * 100)
                year_data[year]['y'].append(y_point.sum()[year] / 1000)
                year_data[year]['scenario'].append(scenario)

    title = f'Usage of {carrier} vs. usage of biomass potential from 2024 to 2050'
    ylabel = f'Usage of {carrier} [TWh p. a.]'
    xlabel = 'Usage of biomass potential (%)'
    plot_data(year_data, year_labels, marker_sizes, title, ylabel, xlabel)

def prepare_el_data(file_path):
    el_data = pd.read_excel(file_path, sheet_name='Sheet 1', skiprows=12)
    el_data = el_data.drop('Unnamed: 2', axis=1)
    el_data.columns = ['Country', 'Electricity']

    el_data = el_data.dropna(subset=['Country', 'Electricity'])
    el_data = el_data[~el_data['Country'].isin(['Special value', ':'])]

    def get_country_code(country_name):
        try:
            return pycountry.countries.lookup(country_name).alpha_2
        except LookupError:
            return None

    el_data['Country Code'] = el_data['Country'].apply(get_country_code)
    el_data.loc[el_data['Country'] == 'Greece', 'Country Code'] = 'EL'
    print('now:', el_data)
    el_data = el_data.drop('Country', axis=1)
    el_data = el_data.rename(columns={'Electricity': 'Electricity_demand_2024'})

    return el_data
def draw_electricity_generation(electricity_data, shapefile_path, file_path, year, scenario, figsize=(20, 20)):
    nuts_gdf = gpd.read_file(shapefile_path)
    nuts_gdf = nuts_gdf.to_crs('EPSG:3035')
    countries_to_exclude = ['IS', 'TR', 'RS', 'MT', 'CY', 'NO', 'AL', 'ME', 'MK', 'LI']

    nuts2_gdf = nuts_gdf[(nuts_gdf['LEVL_CODE'] == 2) & (~nuts_gdf['CNTR_CODE'].isin(countries_to_exclude))]

    df_electricity = electricity_data[electricity_data['carrier'].isin(["electricity"])]
    df_capture_sum = df_electricity.groupby('node')[year].sum().reset_index(name='Electricity_demand')
    el_demand = df_electricity.groupby('node')[0].sum().reset_index(name='Electricity_2024')
    print('el demand:', el_demand)
    nuts2_with_capture = nuts2_gdf.merge(df_capture_sum, left_on='NUTS_ID', right_on='node', how='left')
    nuts2_with_capture = nuts2_with_capture.merge(el_demand, left_on='NUTS_ID', right_on='node', how='left')

    nuts2_with_capture[['Electricity_demand', 'Electricity_2024']].fillna(0, inplace=True)

    country_demand = nuts2_with_capture.groupby('CNTR_CODE')[['Electricity_demand', 'Electricity_2024']].sum().reset_index()
    el_demand_2024 = prepare_el_data(file_path)
    country_demand = country_demand.merge(el_demand_2024, left_on='CNTR_CODE', right_on='Country Code',
                                                  how='left')

    nuts0_gdf = nuts_gdf[nuts_gdf['LEVL_CODE'] == 0]
    nuts0_gdf = nuts0_gdf[~nuts0_gdf['CNTR_CODE'].isin(countries_to_exclude)]
    country_map = nuts0_gdf.merge(country_demand, left_on='CNTR_CODE', right_on='CNTR_CODE', how='left')
    country_map['Electricity_demand'].fillna(0, inplace=True)
    country_map['Electricity_demand_2024'].fillna(0, inplace=True)
    country_map['Electricity_2024'].fillna(0, inplace=True)
    country_map['percentage'] = ((country_map['Electricity_demand'] - country_map['Electricity_2024']) / country_map['Electricity_demand_2024'])+1
    country_map = country_map.dropna(subset=['percentage'])
    country_map['centroid'] = country_map.geometry.centroid

    fig, ax = plt.subplots(figsize=figsize)
    vmin, vmax = 0.98, 1.2
    sm = plt.cm.ScalarMappable(cmap='GnBu', norm=Normalize(vmin=vmin, vmax=vmax))
    sm._A = []

    country_map.plot(column='percentage', ax=ax, cmap='GnBu', norm=sm.norm,
                 linewidth=0.8)

    level = [0]
    nuts_gdf = nuts_gdf[nuts_gdf['LEVL_CODE'].isin(level)]

    nuts0_gdf.plot(ax=ax, color="None", edgecolor="None")

    plt.rcParams['hatch.color'] = 'grey'
    plt.rcParams['hatch.linewidth'] = 0.4
    countries_to_exclude = ['NO', 'UK', 'CH']

    for country_code in countries_to_exclude:
        specific_gdf = nuts_gdf[nuts_gdf['CNTR_CODE'] == country_code]
        specific_gdf.plot(ax=ax, facecolor='lightgrey', hatch="\\", linewidth=0.8)

    for index, row in country_map.iterrows():
        if row['CNTR_CODE'] not in countries_to_exclude and row['Electricity_demand_2024'] > 0:
            if row['CNTR_CODE'] == 'FR':
                x, y = 3713381.55, 2686876.92
            else:
                x, y = row['centroid'].x, row['centroid'].y

            demand_ratio = ((row['Electricity_demand'] - row['Electricity_2024']) / row['Electricity_demand_2024']) +1
            label_text = f"{demand_ratio:.2f}"

            plt.text(x=x, y=y, s=label_text,
                     horizontalalignment='center', fontsize=14, color='black', style='italic',
                     bbox=dict(facecolor='lightgray', alpha=0.5, edgecolor='none'))

    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', shrink=0.75, aspect=30)
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.2f}'))
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label('Change in electricity demand compared to net production in 2022', fontsize=22)

    ax.set_xlim(2.5e6, 5.9e6)
    ax.set_ylim(1.5e6, 5.5e6)
    plt.axis('off')
    plt.savefig(f"{folder_path}/electricity_demand_map_{scenario}_{year}.png", dpi=300, bbox_inches='tight',
               pad_inches=0)
    plt.show()


if __name__ == '__main__':

    folder_path = 'scenarios_biomass_070624'
    # scenarios to be analyzed
    scenarios = [#'scenario_',
                 #'scenario_electrification',
                 #'scenario_hydrogen',
                 #'scenario_biomass',
                 #'scenario_CCS',
                 #'scenario_high_demand',
                 #'scenario_low_demand',
                 #'scenario_biomass_high_price_2',
                 #'scenario_biomass_high_price_3',
                 #'scenario_biomass_high_price_5',
                 #'scenario_biomass_high_price_7',
                 #'scenario_biomass_high_price_10',
                 #'scenario_no_biomass',
                 'scenario_biomass_share_0.0',
                 #'scenario_biomass_share_0.1',
                 #'scenario_biomass_share_0.2',
                 #'scenario_biomass_share_0.3',
                 #'scenario_biomass_share_0.4',
                 #'scenario_biomass_share_0.5',
                 #'scenario_biomass_share_0.6',
                 #'scenario_biomass_share_0.7'
                 ]

    ## save csv files for specific variables
    #for scenario in scenarios:
     #   save_total(folder_path, scenario)
    #for scenario in scenarios:
     #   save_imports_exports(folder_path, scenario)

    ## generate sankey diagram
    # target technologies with only inputs
    target_technologies = ['BF_BOF',
                           'BF_BOF_CCS',
                           'EAF',
                           'carbon_liquefaction',
                           'carbon_storage',
                           'cement_plant',
                           'cement_plant_oxy_combustion', 'cement_plant_post_comb',
                           'e_haber_bosch', 'haber_bosch',
                           'gasification_methanol', 'methanol_from_hydrogen', 'methanol_synthesis',
                           'refinery',
                           'biomethane_SMR_methanol', 'biomethane_SMR', 'biomethane_SMR_CCS', 'biomethane_haber_bosch'
                           ]
    # intermediate technologies with inputs and outputs
    intermediate_technologies = [#'anaerobic_digestion',
        'biomethane_conversion',
        'ASU',
        'biomass_to_coal_conversion', 'hydrogen_for_cement_conversion',
        'DAC',
        'DRI', 'h2_to_ng', 'scrap_conversion_EAF', 'scrap_conversion_BF_BOF',
        'SMR', 'SMR_CCS', 'gasification', 'gasification_CCS',
        'electrolysis',
        'carbon_conversion', 'carbon_methanol_conversion',
        #'photovoltaics', 'pv_ground', 'pv_rooftop', 'wind_offshore', 'wind_onshore',
        'carbon_liquefaction', 'carbon_removal',
        'carbon_storage',
        'carbon_evaporation'
    ]

    # years to plot sankey diagrams for
    years = [0,
             #6,
             #8,
             #13,
             #16,
             26
             ]
    #for year in years:
     #  for scenario in scenarios:
      #      generate_sankey_diagram(folder_path, scenario, target_technologies, intermediate_technologies, year, title="Process depiction in", save_file=False)

    ## generate map with pie charts for biomass usage
    years = [0, #8, 13
             #6,
             16,
             26
             ]
    for scenario in scenarios:
        for year in years:
            shapefile_path = "nuts_data/NUTS_RG_20M_2021_4326.shp"
            #draw_wedges_on_map(folder_path, shapefile_path, year, radius_factor=0.004, scenario=scenario)

    ## generate bar charts for hydrogen production tech mix
    carrier = 'hydrogen'
    #for scenario in scenarios:
     #     plot_outputs(folder_path, scenario, carrier, save_file=True)

    for scenario in scenarios:
        df = res_scenario.get_total("flow_conversion_output").xs(scenario).reset_index()
        df_input = res_scenario.get_total("flow_conversion_input").xs(scenario).reset_index()
        years = [0, 6, 16, 26]
        # plot location and size of hydrogen production technologies
        #plot_dataframe_on_map(df, df_input, 'technology', years,'nuts_data/NUTS_RG_20M_2021_4326.shp', save_png=True)

    ## auxiliary plots to analyze results
    #calc_lco(scenario = "scenario_", discount_rate = 0.06, carrier="ammonia")

    technologies = ['ASU', 'haber_bosch', 'e_haber_bosch', 'EAF', 'BF_BOF', 'DRI', 'SMR', 'methanol_synthesis', 'refinery',
                    'cement_plant', 'DAC']

    #for technology in technologies:
     #   plot_capacity_addition(folder_path, scenario, technology, save_file=False)
      #  plot_existing_capacity(folder_path, scenario, technology, save_file=False)


    ####################################################################################################################
    ####################### PLOTS FOR PAPER ############################################################################
    ####################################################################################################################


    ## single vs. integrated industries bar plots (claim 1)
    #for scenario in scenarios:
    #    plot_carbon_capture(scenario) # double check computations on how to assign biomass feedstocks
    #    plot_biomass_per_industry(scenario)

    ## pareto frontiers for varying biomass avaialbilities (claim 2)
    results = res_scenario
    scenarios = [
        'scenario_', 'scenario_biomass_share_0.7',
        'scenario_biomass_share_0.6', 'scenario_biomass_share_0.5',
        'scenario_biomass_share_0.4', 'scenario_biomass_share_0.3',
        'scenario_biomass_share_0.2',
        'scenario_biomass_share_0.1', 'scenario_biomass_share_0.0'
    ]
    transport_techs = ['carbon_pipeline', 'hydrogen_pipeline']
    years = [0, 1, 2, 3, 4, 5,
             6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
             16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
             26]

    #for tech in transport_techs:
     #   plot_biomass_and_transport(tech, results, years)

    #plot_biomass_and_carbon_storage(results, years)

    #plot_biomass_and_captured_carbon(results, years)

    carrier = 'hydrogen'
    technology = ['SMR', 'SMR_CCS', 'gasification', 'gasification_CCS', 'electrolysis']
    #plot_biomass_and_hydrogen(carrier, technology, results, years)

    carrier = 'electricity'
    #plot_biomass_and_electricity(carrier, results, years)


    # plot transport and electricity generation maps (claim 3)
    for scenario in scenarios:
        transport_data = res_scenario.get_total("flow_transport").xs(scenario).reset_index()
        shapefile_path = "nuts_data/NUTS_RG_20M_2021_4326.shp"
        output_data = res_scenario.get_total("flow_conversion_output").xs(scenario).reset_index()
        electricity_data = res_scenario.get_total("flow_conversion_input").xs(scenario).reset_index()
        file_path = "el_generation_2022.xlsx"
        years = [0,
                 6,
                 16,
                 26
                 ]
        #for year in years:
         #   draw_transport_and_capture(transport_data, output_data, shapefile_path, year, scenario, figsize=(20, 20))
          #  draw_hydrogen_pipelines(transport_data, output_data, shapefile_path, year, scenario, figsize=(20, 20))
         #   draw_transport_arrows_and_biomass_usage(transport_data, shapefile_path, year, scenario, figsize=(20, 20))
            #draw_electricity_generation(electricity_data, shapefile_path, file_path, year, scenario, figsize=(20, 20)) ## claim 3


