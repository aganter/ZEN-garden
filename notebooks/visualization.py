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
    gdf = gdf[["NUTS_ID", "geometry", "LEVL_CODE"]]
    gdf_merged = gdf.merge(dataframe, left_on="NUTS_ID", right_index=True, how='left')
    return gdf_merged


def job_ratio_results(tech):
    """takes tech and returns corresponding mean job ratio in FTE/GW or FTE/(GW*km)
    :param tech: name of technology"""
    path_to_data = os.path.join("employment_data", "job_ratio_results.xlsx")
    df = pd.read_excel(path_to_data)
    filtered_df = df[df['technology'] == tech]
    mean_job_ratio = filtered_df['job_ratio'].mean()
    return mean_job_ratio


def plot_europe_map(dataframe, column, colormap='Reds', x_axis=None, y_axis=None, legend_bool=False, legend_label=None, year=None):
    """plots map of europe with correct x and y-limits, axes names
    :param dataframe: europe joined with data
    :param column: data to plot
    :param colormap: colormap for plotting
    :param x_axis: name of x_axis
    :param y_axis: name of y_axis"""

    dataframe.loc[(dataframe['LEVL_CODE'] == 2) & ~(dataframe['NUTS_ID'].str.startswith(('TR', 'IS'))), column] = \
        dataframe.loc[(dataframe['LEVL_CODE'] == 2) & ~(dataframe['NUTS_ID'].str.startswith(('TR', 'IS'))), column].fillna(0)
    dataframe.plot(column=column, figsize=(30, 20), cmap=colormap, legend=legend_bool, legend_kwds={'label': legend_label}, edgecolor='black')
    x_lower_limit = 0.23 * 10 ** 7
    x_upper_limit = 0.6 * 10 ** 7
    y_lower_limit = 1.3 * 10 ** 6
    y_upper_limit = 5.6 * 10 ** 6
    plt.xlim([x_lower_limit, x_upper_limit])
    plt.ylim([y_lower_limit, y_upper_limit])
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.axis('off')
    plt.title(2020 + 2*year, fontsize=40)
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

        plot_europe_map(gdf_merged, column='hydrogen_output', colormap='Reds', legend_bool=True, legend_label='Hydrogen Production in [GWh]', year=year)

    def plot_tech_jobs(self, year, techs, carrier='hydrogen', scenario=None):
        """load data from Excel file and plot total jobs for specific year, tech"""

        # Read in data from Excel file and use calculation method: mean

        job_ratio = job_ratio_results(tech=techs)  # jobs per GW hydrogen

        # get data from HSC modeling results and check with output_flow for active capacities
        capacity = self.results.get_total("capacity", scenario=scenario, year=year, element_name=techs)
        output_flow = self.results.get_total("output_flow", scenario=scenario, year=year, element_name=techs)
        check = output_flow.loc[carrier].groupby(["node"]).sum()
        total_jobs = capacity.droplevel(0) * job_ratio # total_number of jobs
        total_jobs[check == 0] = 0
        total_jobs.name = "total_jobs"

        # Load NUTS2 data and join with total_jobs
        gdf_merged = merge_nutsdata(total_jobs)

        # plot as a map with limited axes
        plot_europe_map(dataframe=gdf_merged, column='total_jobs', colormap='Reds', legend_bool=True, legend_label=f'Total Jobs, {techs.capitalize()}', year=year)

    def plot_jobs_change(self, carrier, techs, scenario=None):
        """load data from Excel and plot temporal change for specific tech for NUTS0 regions"""

        # Read in data from the Excel

        job_ratio = job_ratio_results(tech=techs)

        # get data from HSC modeling results
        capacity = self.results.get_total("capacity", scenario=scenario, element_name=techs)
        output_flow = self.results.get_total("output_flow", scenario=scenario, element_name=techs)
        check = output_flow.loc[carrier].groupby(["node"]).sum()

        # get the desired form of the data frame to plot
        total_jobs = capacity.droplevel(0) * job_ratio
        # output_flow = output_flow.loc[carrier].groupby(["node"]).sum() * job_ratio  # total_number of jobs
        total_jobs[check == 0] = 0
        # Extract the first two initials of each row label
        initials = [label[:2] for label in total_jobs.index]

        # Create a new DataFrame with the summed values grouped by the first two initials
        tech_jobs = total_jobs.groupby(initials).sum()
        tech_jobs.index.name = "NUTS0"
        pivoted_data = tech_jobs.transpose()

        # Create plot with temporal change
        pivoted_data.plot(figsize=(30, 20), cmap='gist_rainbow')
        plt.xlabel('Time')
        plt.ylabel(f'Jobs in {techs}')
        plt.xticks(np.linspace(0, 15, 16, dtype=int), np.linspace(2020, 2050, 16, dtype=int))
        plt.show()

    def plot_jobs_total(self, techs, year=0, carrier='hydrogen', scenario=None):
        """loop over all hydrogen producing technologies sum and plot total"""
        total_jobs = np.zeros_like(self.results.get_total("capacity", scenario=scenario, year=year, element_name='SMR').droplevel(0))
        for tech in techs:
            job_ratio = job_ratio_results(tech)
            capacity = self.results.get_total("capacity", scenario=scenario, year=year, element_name=tech)
            output_flow = self.results.get_total("output_flow", scenario=scenario, year=year, element_name=tech)
            check = output_flow.loc[carrier].groupby(["node"]).sum()
            tech_jobs = capacity.droplevel(0) * job_ratio  # total_number of jobs
            tech_jobs[check == 0] = 0
            total_jobs += tech_jobs

        # merge with NUTS2 data:
        total_jobs.name = 'total_jobs'
        gdf_merged = merge_nutsdata(total_jobs)

        # plot as a map with limited axes
        plot_europe_map(dataframe=gdf_merged, column='total_jobs', colormap='Reds', legend_bool=True,
                        legend_label='Total Jobs in Hydrogen', year=year)

    def plot_jobs_tr(self, tech, year, carrier='hydrogen', scenario=None):

        # get job ratio from results Excel
        job_ratio = job_ratio_results(tech)

        # get carrier flow results and distances between regions and combine
        carrier_flow = self.results.get_total("carrier_flow", scenario=scenario, year=year, element_name=tech)
        distance = self.results.get_df('distance').loc[tech]
        carrier_flow = carrier_flow.multiply(distance)
        total_jobs = carrier_flow * job_ratio

        # use origin region as job source

        initials = [label[:4] for label in total_jobs.index]
        tech_jobs = total_jobs.groupby(initials).sum()
        tech_jobs.name = "tech_jobs"

        # merge with NUTS2 dataframe
        gdf_merged = merge_nutsdata(tech_jobs)

        # plot as a map of Europe
        plot_europe_map(dataframe=gdf_merged, column='tech_jobs', legend_bool=True, legend_label=f'Total Jobs, {tech.capitalize()}', year=year)



if __name__ == "__main__":
    results_path = os.path.join("outputs", "HSC_NUTS2")
    scenario = "scenario_reference_bau"  # to load more than one scenario, pass a list
    techs = ['SMR', 'SMR_CCS', 'gasification', 'gasification_CCS', 'electrolysis']
    years = [0, 5, 15]
    vis = Visualization(results_path, scenario)
    vis.plot_jobs_tr('hydrogen_pipeline', year=10, scenario=scenario)
    vis.plot_jobs_total(scenario=scenario, techs=techs, year=15)
    # loop to plot all technologies:
    for tech in techs:
        vis.plot_jobs_change("hydrogen", scenario=scenario, techs=tech)
        for year in years:
            vis.plot_tech_jobs(scenario=scenario, techs=tech, year=year)

    vis.plot_output_flows("hydrogen", scenario=scenario, techs="conversion", year=year)
    vis.plot_tech_jobs(scenario=scenario, techs="electrolysis", year=year)
    vis.plot_tech_jobs(scenario=scenario, techs='SMR_CCS', year=year)
    vis.plot_jobs_change("hydrogen", scenario=scenario, techs="electrolysis")

