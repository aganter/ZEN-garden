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


def production_job_ratio(tech):
    """takes tech and returns corresponding job ratio in FTE/GWh
    :param tech: name of technology"""
    path_to_data = os.path.join("employment_data", "results_production.csv")
    df = pd.read_csv(path_to_data)
    filtered_df = df[df['production_method'] == tech]
    mean_job_ratio = filtered_df['job_ratio [FTE/GWh]'].mean()
    return mean_job_ratio


def plot_europe_map(dataframe, column, colormap='Reds', x_axis=None, y_axis=None, legend_bool=False, legend_label=None):
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

        plot_europe_map(gdf_merged, column='hydrogen_output', colormap='Reds', legend_bool=True, legend_label='Hydrogen Production in [GWh]')

    def plot_tech_jobs(self, year, techs, carrier='hydrogen', scenario=None):
        """load data from csv  and plot total jobs for specific year, tech"""

        # Read in data from csv file and use calculation method: mean

        job_ratio = production_job_ratio(tech=techs)  # jobs per GWh hydrogen

        # get data from HSC modeling results
        output_flow = self.results.get_total("output_flow", scenario=scenario, year=year, element_name=techs)
        total_jobs = output_flow.loc[carrier].groupby(["node"]).sum() * job_ratio  # total_number of jobs
        total_jobs.name = "total_jobs"

        # Load NUTS2 data and join with output_flow
        gdf_merged = merge_nutsdata(total_jobs)

        # plot as a map with limited axes
        plot_europe_map(dataframe=gdf_merged, column='total_jobs', colormap='Reds', legend_bool=True, legend_label=f'Total Jobs, {techs.capitalize()}')

    def plot_jobs_change(self, carrier, techs, scenario=None):
        """load data from csv and plot temporal change for specific tech for NUTS0 regions"""

        # Read in data from the csv

        job_ratio = production_job_ratio(tech=techs)

        # get data from HSC modeling results
        output_flow = self.results.get_total("output_flow", scenario=scenario, element_name=techs)

        # get the desired form of the data frame to plot
        output_flow = output_flow.loc[carrier].groupby(["node"]).sum() * job_ratio  # total_number of jobs
        # Extract the first two initials of each row label
        initials = [label[:2] for label in output_flow.index]

        # Create a new DataFrame with the summed values grouped by the first two initials
        tech_jobs = output_flow.groupby(initials).sum()
        tech_jobs.index.name = "NUTS0"
        pivoted_data = tech_jobs.transpose()

        # Create plot with temporal change
        pivoted_data.plot(figsize=(30, 20), cmap='gist_rainbow')
        plt.xlabel('Time')
        plt.ylabel(f'Jobs in {techs}')
        plt.xticks(np.linspace(0, 15, 16, dtype=int), np.linspace(2020, 2050, 16, dtype=int))
        plt.show()

    def plot_jobs_pipeline(self, year=0, technology='hydrogen_pipeline', scenario=None):

        # get data from Excel sheet
        df = excel_data(excel_name="ETHZ_GESA_Fragen copy.xlsx", sheet_name="Transportation")
        jobs_ratio = df.iloc[19:29, 3].sum() / 5.9853

        # get results for carrier flow and combine
        carrier_flow = self.results.get_total("carrier_flow", scenario=scenario, year=year)
        total_jobs = carrier_flow.loc[technology] * jobs_ratio
        initials = [label[:4] for label in total_jobs.index]
        pipeline_jobs = total_jobs.groupby(initials).sum()
        pipeline_jobs.name = "pipeline_jobs"

        # merge with NUTS2 dataframe
        gdf_merged = merge_nutsdata(pipeline_jobs)

        # plot as a map of Europe
        plot_europe_map(dataframe=gdf_merged, column='pipeline_jobs', legend_bool=True, legend_label='Total Jobs, Pipeline')


    def plot_jobs_truck_gas(self, year=7, technology='hydrogen_truck_gas', scenario=None):
        """plot jobs by truck in gaseous form"""

        # get results for carrier flow and combine
        carrier_flow = self.results.get_total("carrier_flow", scenario=scenario, year=year)
        total_jobs = carrier_flow.loc[technology] * jobs_ratio
        initials = [label[:4] for label in total_jobs.index]
        truck_gas_jobs = total_jobs.groupby(initials).sum()
        truck_gas_jobs.name = "truck_gas_jobs"

        # merge with NUTS2 dataframe
        gdf_merged = merge_nutsdata(truck_gas_jobs)

        # plot as a map of Europe
        plot_europe_map(dataframe=gdf_merged, column='truck_gas_jobs', legend_bool=True, legend_label='Total Jobs, Trucks')

    def plot_carrier_flow(self, technology=None, scenario=None):
        """plot carrier flows for selected transport technologies"""

        carrier_flow = self.results.get_total("carrier_flow", scenario=scenario)
        carrier_flow = carrier_flow.loc["hydrogen_truck_gas"]

        # Load NUTS2 data and join with carrier_flow
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
    scenario = "scenario_reference_bau"  # to load more than one scenario, pass a list
    year = 10
    vis = Visualization(results_path, scenario)
    # these work:
    vis.plot_output_flows("hydrogen", scenario=scenario, techs="conversion", year=year)
    vis.plot_tech_jobs(scenario=scenario, techs="electrolysis", year=year)
    vis.plot_tech_jobs(scenario=scenario, techs='SMR_CCS', year=year)
    vis.plot_jobs_change("hydrogen", scenario=scenario, techs="electrolysis")
    vis.plot_jobs_truck_gas(scenario=scenario)
    vis.plot_jobs_pipeline(scenario=scenario,year=year)
    # these do not:
    vis.plot_carrier_flow(scenario=scenario)
    vis.plot_demand_carrier("hydrogen", scenario=scenario, year=year)
    vis.plot_built_capacity("SMR", scenario=scenario)
    vis.plot_existing_capacity("SMR", scenario=scenario)
