import os
from zen_garden.postprocess.results import Results
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def merge_nutsdata(dataframe):
    """takes dataframe and returns NUTS2 merged dataframe
    :param dataframe: dataframe used for merging"""
    path_to_data = os.path.join("outputs", "NUTS_RG_01M_2016_3035", "NUTS_RG_01M_2016_3035.shp")
    gdf = gpd.read_file(path_to_data)
    gdf = gdf[["NUTS_ID", "geometry"]]
    gdf_merged = gdf.merge(dataframe, left_on="NUTS_ID", right_index=True)
    return gdf_merged


def excel_data(excel_name, sheet_name):
    """takes name of excel table, sheet and returns dataframe
    :param excel_name: name of excel table
    :param sheet_name: name of excel sheet"""
    path_to_data = os.path.join("employment_data", excel_name)
    df = pd.read_excel(path_to_data, sheet_name=sheet_name)
    return df


def plot_europe_map(dataframe, column, colormap, x_axis=None, y_axis=None, legend_bool=False, legend_label=None):
    """plots map of europe with correct x and y-limits, axes names
    :param dataframe: europe joined with data
    :param column: data to plot
    :param colormap: colormap for plotting
    :param x_axis: name of x_axis
    :param y_axis: name of y_axis"""
    dataframe.plot(column=column, figsize=(30, 20), cmap=colormap, legend=legend_bool, legend_kwds={'label': legend_label})
    x_lower_limit = 0.23 * 10 ** 7
    x_upper_limit = 0.6 * 10 ** 7
    y_lower_limit = 1.3 * 10 ** 6
    y_upper_limit = 5.6 * 10 ** 6
    plt.xlim([x_lower_limit, x_upper_limit])
    plt.ylim([y_lower_limit, y_upper_limit])
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.show()


class Visualization:

    def __init__(self, results_path, scenarios):
        """load results
        :param results_path: folder path results
        :param scenarios: list of scenarios"""
        if scenarios and not isinstance(scenarios, list):
            scenarios = [scenarios]
        self.scenarios = scenarios
        self.results = Results(results_path, scenarios=scenarios)
        self.system = self.results.results["system"]
        self.analysis = self.results.results["analysis"]

    def plot_output_flows(self, carrier, year, scenario=None, techs=None):
        """plot output flows for selected carriers"""

        output_flow = self.results.get_total("output_flow", scenario=scenario, year=year)

        if techs == "conversion":
            _techs = self.system["set_hydrogen_conversion_technologies"]
            output_flow = output_flow.loc[_techs]
        else:
            print(Warning, "Only conversion techs implemented")
        output_flow = output_flow.groupby(["node"]).sum()  # in GWh
        output_flow.name = "hydrogen_output"

        # Load NUTS2 data and joins with output_flow

        gdf_merged = merge_nutsdata(output_flow)

        # plot as a map with limited axes:

        plot_europe_map(gdf_merged, column='hydrogen_output', colormap='Reds')


    def plot_totaljobs_electrolysis(self, carrier, year, scenario=None, techs=None):
        """load data from excel sheets and plot total jobs in electrolysis for specific year"""

        # Read in data from the excel sheet
        df = excel_data("ETHZ_GESA_Fragen copy.xlsx", "Electrolysis")
        total_jobs = df.iloc[16:32, 2].sum()
        job_ratio = total_jobs / 5.9853  # jobs per GWh hydrogen

        # get data from HSC modeling results
        output_flow = self.results.get_total("output_flow", scenario=scenario, year=year).loc["electrolysis"]

        if techs == "conversion":
            _techs = self.system["set_hydrogen_conversion_technologies"]
            output_flow = output_flow.loc[_techs]
        else:
            print(Warning, "Only conversion techs implemented")
        output_flow = output_flow.groupby(["node"]).sum() * job_ratio  # total_number of jobs
        output_flow.name = "total_jobs"

        # Load NUTS2 data and join with output_flow
        path_to_data = os.path.join("outputs", "NUTS_RG_01M_2016_3035", "NUTS_RG_01M_2016_3035.shp")
        gdf = gpd.read_file(path_to_data)
        gdf = gdf[["NUTS_ID", "geometry"]]
        gdf_merged = gdf.merge(output_flow, left_on="NUTS_ID", right_index=True)

        # plot as a map with limited axes
        plot_europe_map(dataframe=gdf_merged, column='total_jobs', colormap='Reds', legend_bool=True, legend_label='Total Jobs')

    def plot_jobs_change_electrolysis(self, carrier, scenario=None, techs=None):
        """load data from excel sheets and plot temporal change in electrolysis for NUTS0 regions"""

        # Read in data from the excel sheet -> continue here
        path_to_data = os.path.join("employment_data", "ETHZ_GESA_Fragen copy.xlsx")
        df = pd.read_excel(path_to_data, sheet_name='Electrolysis')
        total_jobs = df.iloc[16:32, 2].sum()
        job_ratio = total_jobs / 5.9853  # jobs per GWh hydrogen

        # get data from HSC modeling results
        output_flow = self.results.get_total("output_flow", scenario=scenario).loc[techs]
        if techs == "conversion":
            _techs = self.system["set_hydrogen_conversion_technologies"]
            output_flow = output_flow.loc[_techs]
        else:
            print(Warning, "Only conversion techs implemented")

        # get the desired form of the data frame to plot
        output_flow = output_flow.groupby(["node"]).sum()  # total_number of jobs
        # Extract the first two initials of each row label
        initials = [label[:2] for label in output_flow.index]

        # Create a new DataFrame with the summed values grouped by the first two initials
        electrolysis_jobs = output_flow.groupby(initials).sum()
        electrolysis_jobs.index.name = "NUTS0"
        pivoted_data = electrolysis_jobs.transpose()

        pivoted_data.plot(figsize=(30, 20), cmap='gist_rainbow')
        plt.xlabel('Time')
        plt.ylabel('Jobs')
        plt.xticks(np.linspace(0, 15, 16, dtype=int), np.linspace(2020, 2050, 16, dtype=int))
        plt.show()

    def plot_carrier_flow(self, technology=None, scenario=None):
        """plot carrier flows for selected transport technologies"""

        carrier_flow = self.results.get_total("carrier_flow", scenario=scenario)
        carrier_flow = carrier_flow.loc[technology]

        # Load NUTS2 data and join with output_flow
        path_to_data = os.path.join("outputs", "NUTS_RG_01M_2016_3035", "NUTS_RG_01M_2016_3035.shp")
        gdf = gpd.read_file(path_to_data)
        gdf = gdf[["NUTS_ID", "geometry"]]

    def plot_demand_carrier(self, carrier, year, scenario=None):
        """plot carrier demand"""

        demand_carrier = self.results.get_total("demand_carrier", scenario=scenario, year=year)
        demand_carrier_hydrogen = demand_carrier[['hydrogen']]
        demand_carrier_hydrogen = demand_carrier_hydrogen.groupby(["node"]).sum()  # in GWh

    def plot_built_capacity(self, technology=None, scenario=None):
        """plot newly built capacity for selected technology"""
        built_capacity = self.results.get_total("built_capacity", scenario=scenario)
        built_capacity = built_capacity.groupby(["technology"]).sum()

    def plot_existing_capacity(self, technology=None, scenario=None):
        """plot existing capacity for selected technology"""
        existing_capacity = self.results.get_df("existing_capacity", scenario=scenario)
        existing_capacity = existing_capacity.groupby(["technology"]).sum().loc[technology]


if __name__ == "__main__":
    results_path = os.path.join("outputs", "HSC_NUTS2")
    scenario = "scenario_reference_h2push"  # to load more than one scenario, pass a list
    year = 10
    vis = Visualization(results_path, scenario)
    # these work:
    vis.plot_output_flows("hydrogen", scenario=scenario, techs="conversion", year=year)
    vis.plot_totaljobs_electrolysis("hydrogen", scenario=scenario, techs="electrolysis", year=year)
    vis.plot_jobs_change_electrolysis("hydrogen", scenario, techs="electrolysis")
    # these do not:
    vis.plot_carrier_flow(technology="hydrogen_pipeline", scenario=scenario)
    vis.plot_demand_carrier("hydrogen", scenario=scenario, year=year)
    vis.plot_built_capacity("SMR", scenario=scenario)
    vis.plot_existing_capacity("SMR", scenario=scenario)
