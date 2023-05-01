import os

from zen_garden.postprocess.results import Results
from eth_colors import ETHColors
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



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
        self.eth_colors = ETHColors()
        self.set_colormaps()

    def set_colormaps(self):
        """define colors for colormaps"""
        self.colormap_dict = {
            "SMR": "blue",
            "gasification": "green",
            "electrolysis": "bronze",
            "anaerobic_digestion": "petrol",
            "hydrogen_demand": "purple",
            "hydrogen_output": "red",
            "hydrogen_truck_gas": "blue",
            "hydrogen_truck_liquid": "petrol"
        }

    def merge_nutsdata(self, dataframe, nuts=2):
        """takes dataframe and returns NUTS2 merged dataframe
        :param dataframe: dataframe used for merging
        :param nuts: which NUTS level"""

        path_to_data = os.path.join("outputs", "NUTS_RG_01M_2016_3035", "NUTS_RG_01M_2016_3035.shp")
        gdf = gpd.read_file(path_to_data)
        gdf = gdf[["NUTS_ID", "geometry", "LEVL_CODE"]]
        # For NUTS0 calculations:
        if nuts == 0:
            initials = [label[:2] for label in dataframe.index]
            # Create a new DataFrame with the summed values grouped by the first two initials
            dataframe = dataframe.groupby(initials).sum()
            dataframe.index.name = "NUTS0"

        gdf_merged = gdf.merge(dataframe, left_on="NUTS_ID", right_index=True, how='left')
        return gdf_merged

    def job_ratio_results(self, tech, calc_method='mean'):
        """takes tech and returns corresponding job ratio in FTE/GW or FTE/(GW*km) with chosen calc method
        :param tech: name of technology
        :param calc_method: method of using employment data"""
        path_to_data = os.path.join("employment_data", "job_ratio_results.xlsx")
        df = pd.read_excel(path_to_data)
        filtered_df = df[df['technology'] == tech]
        job_ratio = filtered_df['job_ratio'].agg(calc_method)
        return job_ratio

    def combine_tech_CCS(self, total_jobs, tech, calc_method, scenario, year=0, carrier='hydrogen', time=False):
        """If gasification or SMR additionally add the CCS jobs:"""

        job_ratio_CCS = self.job_ratio_results(tech=f'{tech}_CCS', calc_method=calc_method)
        capacity_CCS = self.results.get_total("capacity", scenario=scenario, element_name=f'{tech}_CCS')
        total_CCS_jobs = capacity_CCS.droplevel(0) * job_ratio_CCS
        output_flow_CCS = self.results.get_total("output_flow", scenario=scenario, element_name=f'{tech}_CCS')
        check_CCS = output_flow_CCS.loc[carrier].groupby(["node"]).sum()
        total_CCS_jobs[check_CCS == 0] = 0
        # For single point in time:
        if time:
            total_jobs += total_CCS_jobs.iloc[:, year]
        else:
            total_jobs += total_CCS_jobs
        return total_jobs

    def plot_output_flows(self, carrier, year, scenario=None, techs=None, nuts=2):
        """plot output flows for selected carriers"""

        output_flow = self.results.get_total("output_flow", scenario=scenario, year=year)

        if techs == "conversion":
            _techs = self.system["set_hydrogen_conversion_technologies"]
            output_flow = output_flow.loc[_techs]
        else:
            print(Warning, "Only conversion techs implemented")
        output_flow = output_flow.loc[:, carrier].groupby(["node"]).sum()  # in GWh

        # Load NUTS2 data and joins with output_flow
        output_flow.name = "hydrogen_output"
        gdf_merged = self.merge_nutsdata(output_flow, nuts=nuts)

        # Plot as a map of Europe:
        self.plot_europe_map(gdf_merged, column=output_flow.name, legend_bool=True, legend_label='Hydrogen Production in [GWh]', year=year, nuts=nuts)

    def plot_demand(self, carrier, year, scenario=None, nuts=2):
        """plot hydrogen demand for selected carrier"""

        demand = self.results.get_total("demand_carrier", scenario=scenario, year=year)

        demand = demand.loc[carrier].groupby(["node"]).sum()  # in GWh

        # Load geographical data and join with demand
        demand.name = "hydrogen_demand"
        gdf_merged = self.merge_nutsdata(demand, nuts=nuts)

        # Plot as a map of Europe
        self.plot_europe_map(gdf_merged, column=demand.name, legend_bool=True, legend_label='Hydrogen Demand in [GWh]', year=year, nuts=nuts)

    def plot_tech_jobs(self, year, techs, carrier='hydrogen', scenario=None, calc_method='mean', nuts=2):
        """load data from Excel file and plot total jobs for specific year, tech"""

        # Read in data from Excel file and use calculation method
        job_ratio = self.job_ratio_results(tech=techs, calc_method=calc_method)  # jobs per GW hydrogen

        # get data from HSC modeling results
        capacity = self.results.get_total("capacity", scenario=scenario, year=year, element_name=techs)  # in GW
        output_flow = self.results.get_total("output_flow", scenario=scenario, year=year, element_name=techs)
        total_jobs = capacity.droplevel(0) * job_ratio

        # Check with output_flow for active capacities
        check = output_flow.loc[carrier].groupby(["node"]).sum()
        total_jobs[check == 0] = 0

        # If gasification or SMR additionally add the CCS jobs:
        if techs == 'SMR' or techs == 'gasification':
            total_jobs = self.combine_tech_CCS(total_jobs=total_jobs, tech=techs, calc_method=calc_method, year=year,
                                               scenario=scenario, time=True)

        # Load NUTS2 data and join with total_jobs
        total_jobs.name = techs
        gdf_merged = self.merge_nutsdata(total_jobs, nuts=nuts)

        # plot as a map with limited axes
        self.plot_europe_map(dataframe=gdf_merged, column=total_jobs.name, legend_bool=True, legend_label=f'Total Jobs, {techs.capitalize()}, {calc_method}', year=year, nuts=nuts)


    def plot_jobs_change(self,  tech, carrier='hydrogen', scenario=None, calc_method='median', nuts=2):
        """load data from Excel and plot temporal change for specific tech for NUTS0 regions"""

        # Read in data from the Excel
        job_ratio = self.job_ratio_results(tech=tech, calc_method=calc_method)

        # get data from HSC modeling results
        capacity = self.results.get_total("capacity", scenario=scenario, element_name=tech)
        output_flow = self.results.get_total("output_flow", scenario=scenario, element_name=tech)
        check = output_flow.loc[carrier].groupby(["node"]).sum()

        # get the desired form of the data frame to plot and check with output flow
        total_jobs = capacity.droplevel(0) * job_ratio
        total_jobs[check == 0] = 0

        # If gasification or SMR additionally add the CCS jobs:
        if tech == 'SMR' or tech == 'gasification':
            total_jobs = self.combine_tech_CCS(total_jobs=total_jobs, tech=tech, calc_method=calc_method,
                                               scenario=scenario)

        # Summation over NUTS0 label for country
        if nuts == 0:
            # Extract the first two initials of each row label
            initials = [label[:2] for label in total_jobs.index]

            # Create a new DataFrame with the summed values grouped by the first two initials
            total_jobs = total_jobs.groupby(initials).sum()
            total_jobs.index.name = "NUTS0"

        pivoted_data = total_jobs.transpose()

        # Create plot with temporal change
        color_dict = self.eth_colors.retrieve_colors_dict(pivoted_data.index.unique(), "country_max")
        pivoted_data.plot(figsize=(30, 20), cmap=color_dict)
        plt.xlabel('Time', fontsize=16)
        plt.ylabel(f'Jobs in {tech}', fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(prop={'size': 18})
        plt.title(f'{tech}, {calc_method}', fontsize=40)
        plt.xticks(np.linspace(0, 15, 16, dtype=int), np.linspace(2020, 2050, 16, dtype=int))
        plot_name = f'time_{tech}_{calc_method}.svg'
        plt.savefig(plot_name, format='svg', dpi=800)
        plt.show()

    def plot_jobs_total(self, techs, year=0, carrier='hydrogen', scenario=None, calc_method='mean', nuts=2):
        """loop over all hydrogen producing technologies sum and plot total"""

        total_jobs = np.zeros_like(self.results.get_total("capacity", scenario=scenario, year=year, element_name='SMR').droplevel(0))
        for tech in techs:
            job_ratio = self.job_ratio_results(tech, calc_method=calc_method)
            capacity = self.results.get_total("capacity", scenario=scenario, year=year, element_name=tech)
            output_flow = self.results.get_total("output_flow", scenario=scenario, year=year, element_name=tech)
            check = output_flow.loc[carrier].groupby(["node"]).sum()
            tech_jobs = capacity.droplevel(0) * job_ratio  # total_number of jobs
            tech_jobs[check == 0] = 0
            total_jobs += tech_jobs

        # merge with NUTS2 data:
        total_jobs.name = 'total_jobs'
        gdf_merged = self.merge_nutsdata(total_jobs, nuts=nuts)

        # plot as a map with limited axes
        self.plot_europe_map(dataframe=gdf_merged, column='total_jobs', legend_bool=True,
                        legend_label=f'Total Jobs in Hydrogen, {calc_method}', year=year, nuts=nuts)

    def plot_jobs_transport(self, tech, year, scenario=None, calc_method='mean', nuts=2):

        # get job ratio from results Excel
        job_ratio = self.job_ratio_results(tech, calc_method=calc_method)

        # get carrier flow results and distances between regions and combine
        carrier_flow = self.results.get_total("carrier_flow", scenario=scenario, year=year, element_name=tech)
        distance = self.results.get_df('distance').loc[tech]
        carrier_flow = carrier_flow.multiply(distance)
        total_jobs = carrier_flow * job_ratio

        # use origin region as job source

        initials = [label[:4] for label in total_jobs.index]
        tech_jobs = total_jobs.groupby(initials).sum()
        tech_jobs.name = tech

        # merge with NUTS2 dataframe
        gdf_merged = self.merge_nutsdata(tech_jobs, nuts=nuts)

        # plot as a map of Europe
        self.plot_europe_map(dataframe=gdf_merged, column=tech_jobs.name, legend_bool=True, legend_label=f'Total Jobs, {tech.capitalize()}, {calc_method}', year=year, nuts=nuts)

    def plot_europe_map(self, dataframe, column, legend_bool=False, legend_label=None, year=None, nuts=2):
        """plots map of europe with correct x and y-limits, axes names
        :param dataframe: europe joined with data
        :param column: data to plot
        :param colormap: colormap for plotting
        :param x_axis: name of x_axis
        :param y_axis: name of y_axis"""


        colormap = self.eth_colors.get_custom_colormaps(self.colormap_dict[column])
        dataframe.loc[(dataframe['LEVL_CODE'] == nuts) & ~(dataframe['NUTS_ID'].str.startswith(('TR', 'IS'))), column] = \
            dataframe.loc[(dataframe['LEVL_CODE'] == nuts) & ~(dataframe['NUTS_ID'].str.startswith(('TR', 'IS'))), column].fillna(0)

        ax = dataframe.plot(figsize=(30, 20), column=column, cmap=colormap, legend=legend_bool, legend_kwds={'label': legend_label}, edgecolor='black')
        fig = ax.figure
        cb_ax = fig.axes[1]
        cb_ax.tick_params(labelsize=30)
        x_lower_limit = 0.23 * 10 ** 7
        x_upper_limit = 0.6 * 10 ** 7
        y_lower_limit = 1.3 * 10 ** 6
        y_upper_limit = 5.6 * 10 ** 6
        plt.xlim([x_lower_limit, x_upper_limit])
        plt.ylim([y_lower_limit, y_upper_limit])
        plt.axis('off')
        plt.title(str(2020 + 2*year) + f', {legend_label}', fontsize=40)
        plot_name = f'europe_{nuts}_{legend_label}_{year}.svg'
        plt.savefig(plot_name, format='svg', dpi=800)
        plt.show()



if __name__ == "__main__":
    results_path = os.path.join("outputs", "HSC_NUTS2")
    scenario = "scenario_reference_bau"  # to load more than one scenario, pass a list
    techs = ['SMR', 'gasification', 'electrolysis']
    tr_techs = ['hydrogen_truck_liquid', 'hydrogen_truck_gas']
    years = [0, 5, 15]
    nutss = [0, 2]
    vis = Visualization(results_path, scenario)
    vis.plot_jobs_change(carrier="hydrogen", scenario=scenario, tech="electrolysis", nuts = 0)
    # Loop to plot all technologies for every year and nuts:
    for nuts in nutss:
        for year in years:
            for tech in techs:
                # vis.plot_jobs_change(scenario=scenario, tech=tech, calc_method='median', nuts=nuts)
                vis.plot_tech_jobs(scenario=scenario, techs=tech, year=year, nuts=nuts, calc_method='median')
            # for tech in tr_techs:
            #    vis.plot_jobs_transport(tech=tech, year=year, scenario=scenario, calc_method='median', nuts=nuts)
            vis.plot_demand("hydrogen", scenario=scenario, year=year, nuts=nuts)

    # Test functions for plotting:
    vis.plot_output_flows(carrier="hydrogen", scenario=scenario, techs="conversion", year=5)
    vis.plot_tech_jobs(scenario=scenario, techs='gasification', year=years[2], calc_method='max', nuts=2)
    vis.plot_demand("hydrogen", scenario=scenario, year=5, nuts=0)
    vis.plot_tech_jobs(scenario=scenario, techs='SMR', year=5, calc_method='max', nuts=0)
    vis.plot_demand("hydrogen", scenario=scenario, year=5, nuts=0)
    vis.plot_jobs_total(scenario=scenario, techs=techs, year=15, calc_method='max', nuts=0)
    vis.plot_jobs_change("hydrogen", scenario=scenario, tech="electrolysis")

