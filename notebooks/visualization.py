import os
from zen_garden.postprocess.results import Results
from eth_colors import ETHColors
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patheffects as mpe
from matplotlib import patches as mpatches
import networkx as nx
import pandas as pd
import numpy as np

# Adjust font to LaTeX font
font_params = {"ytick.color": "black",
                       "xtick.color": "black",
                       "axes.labelcolor": "black",
                       "axes.edgecolor": "black",
                       "text.usetex": True,
                       "font.family": "serif",
                       "font.serif": ["Computer Modern Serif"]}

plt.rcParams.update(font_params)


class Visualization:

    def __init__(self, results_path, scenarios):
        """load results
        :param results_path: folder path results
        :param scenarios: list of scenarios"""
        self.font_params = None
        if scenarios and not isinstance(scenarios, list):
            scenarios = [scenarios]
        self.scenarios = scenarios
        self.results = Results(results_path, scenarios=scenarios)
        self.system = self.results.results["system"]
        self.analysis = self.results.results["analysis"]
        self.eth_colors = ETHColors()
        self.set_colormaps()
        self.set_gdf()
        self.set_pts()
        self.set_population()
    def set_colormaps(self):
        """define colors for colormaps"""
        self.colormap_dict = {
            "SMR": "blue",
            "gasification": "green",
            "electrolysis": "bronze",
            "anaerobic_digestion": "petrol",
            "pv_ground": "red",
            "wind_onshore": "blue",
            "wind_offshore": "petrol",
            'renew_tot': "purple",
            "hydrogen_demand": "purple",
            "hydrogen_output": "red",
            "hydrogen_truck_gas": "blue",
            "hydrogen_truck_liquid": "petrol",
            "export": ("blue", "red"),
            "gini": ("blue", "red"),
            "jobs": "green",
            "dry_biomass": "green",
            "carbon": "red",
            "biomethane_share": ("blue", "green"),
            "biomethane_share2": ("blue", "green")
        }

    def combine_to_nuts0(self, dataframe):
        """combines from NUTS2 to NUTS0 level
        :param dataframe: dataframe containing information on NUTS2 level"""

        initials = [label[:2] for label in dataframe.index]
        dataframe = dataframe.groupby(initials).sum()
        dataframe.index.name = "NUTS0"
        return dataframe
    def gini_coefficient(self, dataframe, column):
        """Calculates Gini coefficient for given dataframe"""
        # Calculate cumulative shares and percentages with sorted dataframe
        sorted_data = dataframe.sort_values(column)
        cumulative_percentage = np.cumsum(sorted_data[column]) / sorted_data[column].sum()
        cumulative_share = np.cumsum(sorted_data['OBS_VALUE']) / sorted_data['OBS_VALUE'].sum()
        # Add the origin as the first coordinate
        cumulative_percentage = np.concatenate(([0], cumulative_percentage))
        cumulative_share = np.concatenate(([0], cumulative_share))
        # Calculate the area under the curve and total area
        auc = np.trapz(y=cumulative_percentage, x=cumulative_share)
        total = 0.5
        # Calculate Gini geometrically
        gini = (total - auc) / total
        return gini

    def set_gdf(self):
        """set geodataframe"""
        # Read in the geodataframe for Europe on NUTS2 resolution
        path_to_data = os.path.join("outputs", "NUTS_RG_01M_2016_3035", "NUTS_RG_01M_2016_3035.shp")
        gdf = gpd.read_file(path_to_data)
        gdf = gdf[["NUTS_ID", "geometry", "LEVL_CODE"]]
        gdf = gdf.set_index("NUTS_ID")
        self.gdf = gdf

    def set_population(self):
        """set population dataframe"""
        # Read in population number from csv sheet, source: EUROSTAT
        path_to_data = os.path.join("outputs", "demo_r_d2jan_page_linear.csv")
        df = pd.read_csv(path_to_data)
        population = df[['geo', 'OBS_VALUE']]
        population.set_index('geo', inplace=True)
        self.population = population

    def merge_nutsdata(self, dataframe, nuts=2):
        """takes dataframe and returns NUTS2 merged dataframe
        :param dataframe: dataframe used for merging
        :param nuts: which NUTS level"""

        gdf = self.gdf.copy(deep=True)
        # For NUTS0 calculations:
        if nuts == 0:
            dataframe = self.combine_to_nuts0(dataframe)
        gdf_merged = gdf.merge(dataframe, left_index=True, right_index=True, how='right')
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

    def combine_tech_CCS(self, total_jobs, tech, calc_method, scenario, year=0, carrier='hydrogen', time=False, country=None, capacity_only=False):
        """If gasification or SMR additionally add the CCS capacity or jobs"""

        job_ratio_CCS = self.job_ratio_results(tech=f'{tech}_CCS', calc_method=calc_method)
        capacity_CCS = self.results.get_total("capacity", scenario=scenario, element_name=f'{tech}_CCS')
        if capacity_only:
            total_CCS_jobs = capacity_CCS.droplevel(0)
        else:
            total_CCS_jobs = capacity_CCS.droplevel(0) * job_ratio_CCS
        output_flow_CCS = self.results.get_total("flow_conversion_output", scenario=scenario, element_name=f'{tech}_CCS')
        # Check if capacity installed is used
        check_CCS = output_flow_CCS.loc[carrier].groupby(["node"]).sum()
        total_CCS_jobs[check_CCS == 0] = 0
        # For single point in time:
        if time:
            total_jobs += total_CCS_jobs.iloc[:, year]
        # For single country:
        elif country:
            mask = total_CCS_jobs.index.str.startswith(country)
            total_CCS_jobs = total_CCS_jobs.loc[mask, :]
            total_jobs += total_CCS_jobs
        else:
            total_jobs += total_CCS_jobs

        return total_jobs

    def calc_rel(self, dataframe, nuts, column):
        """dataframe is divided by absolute value and returns relative value
        :param dataframe: dataframe that contains data
        :param nuts: specifies spatial resolution
        :param column: specifies which column of dataframe"""

        # Get population dataframe
        df = self.population
        if nuts == 0:
            df = self.combine_to_nuts0(df)
        # Merge with dataframe and calculate value per 100'000 capita
        gdf_merged = df.merge(dataframe, left_index=True, right_index=True, how='right')
        gdf_merged[column] = (gdf_merged[column] / gdf_merged['OBS_VALUE']) * 100000

        return gdf_merged

    def set_pts(self):
        """set dataframe with points"""
        if not hasattr(self, "gdf"):
            self.set_gdf()
        self.pts = self.gdf["geometry"].centroid

    def create_directed_graph(self, ax, values):
        """create directed Graph"""
        pts = self.set_pts()
        G = nx.DiGraph(ax=ax)
        posDict = {}
        for edge in values.index:
            nodeFrom, nodeTo = edge.split("-")
            # add nodes
            if nodeFrom not in list(G.nodes):
                position = (pts.loc[nodeFrom].geometry.x, pts.loc[nodeFrom].geometry.y)
                posDict[nodeFrom] = position
                G.add_node(nodeFrom, pos=position)
            if nodeTo not in list(G.nodes):
                position = (pts.loc[nodeTo].geometry.x, pts.loc[nodeTo].geometry.y)
                posDict[nodeTo] = position
                G.add_node(nodeTo, pos=position)
            # add edges
            weight = values.loc[edge, "linewidth"]
            arrowsize = values.loc[edge, "arrowsize"]
            color = self.get_edge_color(values.loc[edge, "edgeweight"])
            G.add_edge(nodeFrom, nodeTo, weight=weight, color=color, arrowsize=arrowsize)
        return G

    def plot_output_flows(self, carrier, year, scenario=None, techs=None, nuts=2):
        """plot output flows for selected carriers"""

        output_flow = self.results.get_total("flow_conversion_output", scenario=scenario, year=year)

        if techs == "conversion":
            _techs = self.system["set_hydrogen_conversion_technologies"]
            output_flow = output_flow.loc[_techs]
        else:
            print(Warning, "Only conversion techs implemented")
        output_flow = output_flow.loc[:, carrier].groupby(["node"]).sum()  # in GWh

        # Load NUTS2 data and joins with output_flow
        output_flow.name = "output_flow"
        gdf_merged = self.merge_nutsdata(output_flow, nuts=nuts)

        # Plot as a map of Europe:
        self.plot_europe_map(gdf_merged, column=output_flow.name, legend_label=
            f'{carrier.capitalize()} Production in [GWh]', year=year, nuts=nuts)

    def plot_conversion_input(self, carrier, year, scenario=None, nuts=2):
        """plot conversion input for selected carrier"""

        # Get available stock of selected carrier
        carrier_available = self.results.get_total("availability_import", scenario=scenario, year=year).groupby(["carrier", "node"]).mean().loc[carrier]
        carrier_used = self.results.get_total("flow_conversion_input", scenario=scenario, year=year).groupby(["carrier", "node"]).mean().loc[carrier]
        # Load NUTS2 data and join with carrier_available
        if carrier == 'carbon':
            potential_used = carrier_used
        else:
            potential_used = (carrier_used/carrier_available) * 100
        potential_used.name = carrier
        gdf_merged = self.merge_nutsdata(potential_used, nuts=nuts)

        # Plot as a map of Europe:
        self.plot_europe_map(gdf_merged, column=potential_used.name, legend_label=f'{carrier.capitalize()}',
                             year=year, nuts=nuts, title=False)

    def plot_demand(self, carrier, year, scenario=None, nuts=2):
        """plot hydrogen demand"""

        demand = self.results.get_total("demand", scenario=scenario, year=year)
        demand = demand.loc[carrier].groupby(["node"]).sum()/1000  # in TWh
        # Load geographical data and join with demand
        demand.name = f"{carrier}_demand"
        gdf_merged = self.merge_nutsdata(demand, nuts=nuts)

        # Plot as a map of Europe
        self.plot_europe_map(gdf_merged, column=demand.name, legend_label=f'{carrier.capitalize()} Demand in [TWh/year]'
                             , year=year, nuts=nuts, title=False)

    def plot_tech_jobs(self, year, techs, carrier='hydrogen', scenario=None, calc_method='mean', nuts=2, rel=False, max_val=0):
        """load data from Excel file and plot total jobs for specific year, tech"""

        # Read in data from Excel file and use calculation method
        job_ratio = self.job_ratio_results(tech=techs, calc_method=calc_method)  # jobs per GW hydrogen

        # get data from HSC modeling results
        capacity = self.results.get_total("capacity", scenario=scenario, year=year, element_name=techs)  # in GW
        output_flow = self.results.get_total("flow_conversion_output", scenario=scenario, year=year, element_name=techs)
        jobs = capacity.droplevel(0) * job_ratio

        # Check with output_flow for active capacities
        check = output_flow.loc[carrier].groupby(["node"]).sum()
        jobs[check == 0] = 0

        # If gasification or SMR additionally add the CCS jobs:
        if techs == 'SMR' or techs == 'gasification':
            jobs = self.combine_tech_CCS(total_jobs=jobs, tech=techs, calc_method=calc_method, year=year,
                                               scenario=scenario, time=True)

        # Load NUTS2 data and join with jobs
        jobs.name = techs

        if rel:
            jobs = self.calc_rel(jobs, column=techs, nuts=nuts)
            unit = "[FTE] per 100'000 capita"
        else:
            unit = '[FTE]'
        gdf_merged = self.merge_nutsdata(jobs, nuts=nuts)
        # plot as a map with limited axes
        self.plot_europe_map(dataframe=gdf_merged, column=techs, legend_label=f'Jobs, {techs.capitalize()} {unit}',
                             year=year, nuts=nuts, title=False, max_val=max_val)

        # compute percentage of non-zero and non-null values
        non_zero_non_null_pct = ((check != 0) & (~check.isnull())).sum() / len(check) * 100
        print(f"{techs.capitalize()} produces jobs in {non_zero_non_null_pct:.2f}% of regions")

    def plot_export_potential(self, carrier='hydrogen', scenario=None, year=0, nuts=2, diverging=True, title=False):
        """plot export potential for every NUTS0/2 region, production - demand"""

        # Get production of selected carrier per year
        production = self.results.get_total("flow_conversion_output", scenario=scenario, year=year)
        production = production.loc[:, carrier].groupby(["node"]).sum()

        # Get demand of selected carrier per year
        if carrier == 'hydrogen':
            demand = self.results.get_total("demand", scenario=scenario, year=year)
            demand = demand.loc[carrier].groupby(["node"]).sum()
        else:
            demand = self.results.get_total("flow_conversion_input", scenario=scenario, year=year).groupby(
                ["carrier", "node"]).mean().loc[carrier]

        # Calculate difference and merge with NUTS data
        export = (production - demand)/1000  # in TWh
        export.name = 'export'
        gdf_merged = self.merge_nutsdata(export, nuts=nuts)

        # Get the transport data for arrows
        arrows = self.results.get_total("flow_transport", scenario=scenario, year=year,
                                        element_name='hydrogen_pipeline')

        # Plot as a map of Europe
        self.plot_europe_map(dataframe=gdf_merged, column=export.name, legend_label=
        f'Production - Demand, {carrier.capitalize()} [TWh/year]', year=year, nuts=nuts, diverging=diverging, title=title)

    def plot_tech_change(self, tech, carrier='hydrogen', scenario=None, calc_method='median', nuts=2, time=0, title=False):
        """load data from Excel and plot temporal change for specific tech for NUTS0/2 regions"""

        # Read in job ratio for specified tech from the Excel
        job_ratio = self.job_ratio_results(tech=tech, calc_method=calc_method)

        # Get capacity and output flow from HSC modeling results
        capacity = self.results.get_total("capacity", scenario=scenario, element_name=tech)
        output_flow = self.results.get_total("flow_conversion_output", scenario=scenario, element_name=tech)
        check = output_flow.loc[carrier].groupby(["node"]).sum()

        # Get the desired form of the data frame to plot and check with output flow for active capacities
        total_jobs = capacity.droplevel(0) * job_ratio
        total_jobs[check == 0] = 0

        # If gasification or SMR additionally add the CCS jobs:
        if tech == 'SMR' or tech == 'gasification':
            total_jobs = self.combine_tech_CCS(total_jobs=total_jobs, tech=tech, calc_method=calc_method,
                                               scenario=scenario)

        # Summation over NUTS0 label for country
        if nuts == 0:
            total_jobs = self.combine_to_nuts0(total_jobs)
        # Hand-adjusted parameters:
        if tech == 'electrolysis':
            time = 15
        if tech == 'gasification':
            time = 10
        if tech == 'SMR':
            time = 15

        # Pick max, median and min from chosen time
        colorful = np.empty(3, dtype=object)
        colorful[0] = total_jobs.iloc[:, time].idxmax()
        max_val = total_jobs.iloc[:, time].max()
        min_val = total_jobs.iloc[:, time].min()
        median_value = (max_val + min_val) / 2
        colorful[1] = (total_jobs.iloc[:, time] - median_value).abs().idxmin()
        colorful[2] = total_jobs.iloc[:, time].idxmin()

        # Get colors to create color dictionary
        color_dict = {}
        for region in total_jobs.index:
            if region in colorful:
                if region == colorful[0]:
                    color_dict[region] = self.eth_colors.get_color('blue')
                elif region == colorful[1]:
                    color_dict[region] = self.eth_colors.get_color('green')
                else:
                    color_dict[region] = self.eth_colors.get_color('bronze')
            else:
                color_dict[region] = self.eth_colors.get_color('grey', 40)

        # Plot with as a timescale
        self.plot_time_scale(dataframe=total_jobs, color_dict=color_dict, carrier=carrier, colorful=colorful, title=title)

    def plot_total_change(self, techs, carrier='hydrogen', scenario=None, calc_method='mean', nuts=0, time=15, title=False):
        """Total jobs for every country for specified techs"""

        # Create dataframe to fill in total jobs
        total_jobs = self.results.get_total("capacity", scenario=scenario, element_name='SMR').droplevel(0) * 0

        # Loop over all specified techs and sum the total jobs
        for tech in techs:
            job_ratio = self.job_ratio_results(tech, calc_method=calc_method)
            capacity = self.results.get_total("capacity", scenario=scenario, element_name=tech)
            output_flow = self.results.get_total("flow_conversion_output", scenario=scenario, element_name=tech)
            if tech == 'anaerobic_digestion':
                check = output_flow.loc['biomethane'].groupby(['node']).sum()
            else:
                check = output_flow.loc[carrier].groupby(["node"]).sum()
            tech_jobs = capacity.droplevel(0) * job_ratio  # total_number of jobs
            tech_jobs[check == 0] = 0
            # If gasification or SMR additionally add the CCS jobs:
            if tech == 'SMR' or tech == 'gasification':
                tech_jobs = self.combine_tech_CCS(total_jobs=tech_jobs, tech=tech, calc_method=calc_method,
                                                   scenario=scenario)

            total_jobs += tech_jobs

        # Summation over NUTS0 label for country
        if nuts == 0:
            total_jobs = self.combine_to_nuts0(total_jobs)

        # Produce color_dict for 'Leaders' and 'Followers'
        colorful = np.empty(0)
        jobs_at_time = total_jobs.iloc[:, time]
        colorful = np.concatenate([colorful, jobs_at_time.nlargest(3).index.values])
        colorful = np.concatenate([colorful, jobs_at_time.nlargest(12).nsmallest(9).index.values])

        color_dict = {}
        for region in total_jobs.index:
            if region in colorful[:3]:
                color_dict[region] = self.eth_colors.get_color('purple')
            elif region in colorful[3:12]:
                color_dict[region] = self.eth_colors.get_color('blue')

        # Print with timescale
        self.plot_time_scale(dataframe=total_jobs, color_dict=color_dict, carrier=carrier, colorful=colorful, title=title)

    def country_jobs_change(self, country, techs, carrier='hydrogen', scenario=None, calc_method='mean', nuts=0):
        """plot change of jobs for all technologies for a country or NUTS2 regions in country"""

        # Read in job ratios from Excel sheet
        path_to_data = os.path.join("employment_data", "job_ratio_results.xlsx")
        df = pd.read_excel(path_to_data)

        grouped_df = df.groupby('technology')['job_ratio'].agg(calc_method)
        # Create a new DataFrame to store the results
        job_ratios_df = pd.DataFrame({'technology': grouped_df.index, 'job_ratio': grouped_df.values})
        # get data from HSC modeling results
        fig, ax = plt.subplots(figsize=(30, 20))

        # Loop over all techs and plot jobs over time
        for tech in techs:
            capacity = self.results.get_total("capacity", scenario=scenario, element_name=tech)
            mask = capacity.index.get_level_values(1).str.startswith(country)
            capacity_filtered = capacity.loc[mask, :]
            output_flow = self.results.get_total("flow_conversion_output", scenario=scenario, element_name=tech)
            if tech == 'anaerobic_digestion':
                check = output_flow.loc['biomethane'].groupby(['node']).sum()
            else:
                check = output_flow.loc[carrier].groupby(["node"]).sum()
            check = check.loc[mask, :]
            job_ratio = job_ratios_df.loc[job_ratios_df['technology'] == tech, 'job_ratio'].iloc[0]
            total_jobs = capacity_filtered.droplevel(0) * job_ratio
            total_jobs[check == 0] = 0

            # If gasification or SMR additionally add the CCS jobs:
            if tech == 'SMR' or tech == 'gasification':
                total_jobs = self.combine_tech_CCS(total_jobs=total_jobs, tech=tech, calc_method=calc_method,
                                                   scenario=scenario, country=country, capacity_only=True)
            # Sum over NUTS0
            if nuts == 0:
                total_jobs = self.combine_to_nuts0(total_jobs)

            pivoted_data = total_jobs.transpose()
            color = self.eth_colors.get_color(self.colormap_dict[tech])

            # Add the plot to the figure object
            pivoted_data.plot(ax=ax, color=color, linewidth=5, label=tech)

        # Plot details:
        ax.set_ylabel(f'Production [TWh]', fontsize=30)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xticks(np.linspace(0, 15, 16, dtype=int), np.linspace(2020, 2050, 16, dtype=int), fontsize=24)
        plt.yticks(fontsize=24)
        plt.xlabel('Time', fontsize=30)
        if nuts == 0:
            plt.legend(labels=techs, prop={'size': 40}, frameon=False)

        # Plot title and save plot:
        plt.title(f'{country.upper()}, {calc_method.capitalize()}', fontsize=40)
        plot_name = f'{country}_{calc_method}_{nuts}.svg'
        plt.savefig(plot_name, format='svg', dpi=600, transparent=True)
        plt.show()

    def plot_hydrogen_jobs(self, techs, year=0, carrier='hydrogen', scenario=None, calc_method='mean', nuts=2, rel=False, plot='europe', max_val=0):
        """loop over all hydrogen producing technologies sum and plot total"""

        # Loop over all hydrogen producing technologies and sum up jobs
        jobs = np.zeros_like(self.results.get_total("capacity", scenario=scenario, year=year, element_name='SMR').
                             droplevel(0))

        # Create list for pie chart
        pie_data = []

        for tech in techs:
            job_ratio = self.job_ratio_results(tech, calc_method=calc_method)
            capacity = self.results.get_total("capacity", scenario=scenario, year=year, element_name=tech)
            output_flow = self.results.get_total("flow_conversion_output", scenario=scenario, year=year, element_name=tech)
            if tech == 'anaerobic_digestion':
                check = output_flow.loc['biomethane'].groupby(['node']).sum()
            else:
                check = output_flow.loc[carrier].groupby(["node"]).sum()
            tech_jobs = capacity.droplevel(0) * job_ratio  # total_number of jobs
            tech_jobs[check == 0] = 0
            if tech == 'SMR' or tech == 'gasification':
                tech_jobs = self.combine_tech_CCS(total_jobs=tech_jobs, tech=tech, calc_method=calc_method,
                                                   scenario=scenario, year=year, time=True)

            jobs += tech_jobs
            # Append a tuple containing the tech name and the total jobs to pie_data
            pie_data.append((tech, tech_jobs.sum()))

        # Create a DataFrame from the list of tuples
        if plot == 'pie':
            pie_dataframe = pd.DataFrame(pie_data, columns=['Technology', 'Total Jobs'])
            self.plot_pie_chart(pie_dataframe, year=year)

        # Merge with NUTS2 data:
        jobs.name = 'jobs'
        unit = '[FTE]'

        # Make Lorenz Plot
        if plot == 'lorenz':
            self.lorenz_plot(dataframe=jobs, column='jobs', year=year)
        # Make GINI Plot
        if plot == 'gini':
            self.gini_plot(dataframe=jobs, column='jobs', year=year)
        # Calculate jobs per 100'000 capita
        if rel:
            jobs = self.calc_rel(jobs, column=jobs.name, nuts=nuts)
            unit = "[FTE] per 100'000 capita"

        # Merge with NUTS2 data:
        jobs = self.merge_nutsdata(jobs, nuts=nuts)
        jobs.name = 'Jobs'

        # Make Europe plot
        if plot == 'europe':
            self.plot_europe_map(dataframe=jobs, column='jobs',
                        legend_label=f'{jobs.name.capitalize()} in Hydrogen, {unit}', year=year, nuts=nuts, title=False, max_val=max_val)

        # Compute percentage of non-zero and non-null values
        non_zero_non_null_pct = ((jobs['jobs'] != 0) & (~jobs['jobs'].isnull())).sum() / len(jobs) * 100
        print(f"Hydrogen produces jobs in {non_zero_non_null_pct:.2f}% of regions")
    def plot_jobs_transport(self, tech, year, scenario=None, calc_method='mean', nuts=2):

        # Get job ratio from results Excel
        job_ratio = self.job_ratio_results(tech, calc_method=calc_method)

        # Get carrier flow results and distances between regions and combine
        carrier_flow = self.results.get_total("flow_transport", scenario=scenario, year=year, element_name=tech)
        distance = self.results.get_df('distance').loc[tech]
        carrier_flow = carrier_flow.multiply(distance)
        total_jobs = carrier_flow * job_ratio

        # Use origin region as job source
        initials = [label[:4] for label in total_jobs.index]
        tech_jobs = total_jobs.groupby(initials).sum()
        tech_jobs.name = tech

        # Merge with NUTS2 dataframe
        gdf_merged = self.merge_nutsdata(tech_jobs, nuts=nuts)

        # Plot as a map of Europe
        self.plot_europe_map(dataframe=gdf_merged, column=tech_jobs.name,
                             legend_label=f'Total Jobs, {tech.capitalize()}, {calc_method}', year=year, nuts=nuts)

    def plot_biomethane_share(self, year, scenario=None, nuts=2):
        """plot the share of biomethane as input in the SMR/SMRCCS plants for a specific year"""

        # Get total conversion input data
        input_flow = self.results.get_total("flow_conversion_input", scenario=scenario, year=year)

        # Get the data from the model for biomethane used for conversion
        biomethane_use = input_flow.loc['biomethane_conversion'].groupby(["node"]).sum()

        # Get the data from the model for NG(NG&biomethane) used in SMR/SMRCCS
        naturalgas_use = input_flow.loc['SMR'].groupby(['node']).sum() + input_flow.loc['SMR_CCS'].groupby(['node']).sum()

        # Share of biomethane in the total gas consumption for SMR/SMRCCS
        biomethane_share = (biomethane_use / naturalgas_use) * 100

        biomethane_share.name = 'biomethane_share'

        gdf_merged = self.merge_nutsdata(biomethane_share, nuts=nuts)

        self.plot_europe_map(dataframe=gdf_merged, column=biomethane_share.name, legend_label=
                    f'Biomethane share in percent', year=year, nuts=nuts, title=False, diverging=True)

    def plot_biomethane_shares(self, years, scenario=None, nuts=2):
        """plot the share of biomethane as input in the SMR/SMRCCS plants over specified years"""

        # Loop over years and calculate biomethane shares
        for i, year in enumerate(years):
            input_flow = self.results.get_total("flow_conversion_input", scenario=scenario, year=year)

            # Get the data from the model for biomethane used for conversion
            biomethane_use = input_flow.loc['biomethane_conversion'].groupby(["node"]).sum()

            # Get the data from the model for NG(NG&biomethane) used in SMR/SMRCCS
            naturalgas_use = input_flow.loc['SMR'].groupby(['node']).sum() + input_flow.loc['SMR_CCS'].groupby(
                ['node']).sum()

            # Share of biomethane in the total gas consumption for SMR/SMRCCS
            biomethane_share = (biomethane_use / naturalgas_use) * 100

            biomethane_share.name = f'biomethane_share{i}'

            if i == 0:
                gdf_merged = self.merge_nutsdata(biomethane_share, nuts=nuts)
            else:
                gdf_merged = pd.concat([gdf_merged, biomethane_share], axis=1)

        self.plot_europe_maps(dataframe=gdf_merged, column=biomethane_share.name,
                             legend_label=f'Biomethane share in percent', year=years, nuts=nuts, title=False,
                             diverging=True)

    def plot_renewables(self, year, scenario=None, nuts=2):
        """plot renewables availability"""

        # Technologies considered in model:
        techs = ["pv_ground", "wind_onshore"]

        renewable_tot = np.zeros_like(self.results.get_total("flow_conversion_output",scenario=scenario, year=year,
                                                             element_name='pv_ground').groupby(['node']).sum())
        # Loop over renewable techs and sum up output
        for tech in techs:
            renewable_prod = self.results.get_total("flow_conversion_output", scenario=scenario,
                                                    year=year, element_name=tech).groupby(['node']).sum()/1000
            renewable_prod.name = tech
            gdf_merged = self.merge_nutsdata(dataframe=renewable_prod, nuts=nuts)

            self.plot_europe_map(dataframe=gdf_merged, column=tech, legend_label=f'{tech.capitalize()}',
                                 year=year, nuts=nuts)
            renewable_tot += renewable_prod
        renewable_tot.name = 'renew_tot'
        gdf_merged = self.merge_nutsdata(renewable_tot, nuts=nuts)

        self.plot_europe_map(dataframe=gdf_merged, column=renewable_tot.name,
                             legend_label=f'Renewable production [TWh/year]', year=year, nuts=nuts, title=False)

    def plot_europe_map(self, dataframe, column, legend_label=None, year=0, nuts=2, diverging=False, title=False, max_val=0, arrows=None):
        """plots map of europe with correct x and y-limits, axes names
        :param dataframe: europe joined with data
        :param column: data to plot
        :param: legend_label: Label of legend
        :param nuts: region size
        :param diverging: for diverging cmap
        :param title: bool title
        :param max_val: maximum value for color-bar
        :param arrows: dataframe for arrow plotting"""
        min_val = 0
        # Adjust color-bar scale
        if max_val == 0:
            max_val = dataframe[column].max()
            min_val = dataframe[column].min()
            if max_val == min_val:
                max_val += 1
        # Get the colormap for plotting
        colormap = self.eth_colors.get_custom_colormaps(self.colormap_dict[column], diverging=diverging)
        grey = self.eth_colors.get_color('grey', 40)

        # Quick fix for balkan and NUTS0
        gdf_quickfix = self.gdf.copy(deep=True)
        gdf_quickfix[column] = pd.NA
        conditions = (gdf_quickfix['LEVL_CODE'] == 2) & (
            gdf_quickfix.index.str.startswith(('AL', 'ME', 'RS', 'MK')))
        gdf_quickfix.loc[conditions, column] = -50000.00000
        selected_rows = gdf_quickfix.loc[conditions].dropna(subset=[column])
        selected_rows[column] = pd.to_numeric(selected_rows[column])
        dataframe = dataframe.append(selected_rows)
        colormap.set_under(color=grey)

        # Plot figure, axis and the underlying data in a Europe map
        fig, ax = plt.subplots(figsize=(30, 20))
        dataframe.plot(column=column, cmap=colormap, legend=False, edgecolor='black', linewidth=0.5, ax=ax, vmin=min_val, vmax=max_val)
        # Outline countries additionally
        if nuts == 2:
            # Only take NUTS2 regions that have model values assigned
            mask = dataframe[column].notnull()
            # Get the first two characters of the index for rows where 'column' is not null
            idx_values = dataframe[mask].index.str[:2]
            # Filter the 'nuts0' dataframe based on the first two characters of the index and 'LEVL_CODE'
            nuts0 = gdf_quickfix[(gdf_quickfix.index.isin(idx_values)) & (gdf_quickfix['LEVL_CODE'] == 0)]
            nuts0.boundary.plot(edgecolor='black', linewidth=1, ax=ax)

        # Modify colormap and add frame and details to plot
        norm = plt.Normalize(vmin=min_val, vmax=max_val)
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=colormap, norm=norm), ax=ax)
        cbar.set_label(legend_label, size=50, labelpad=15)
        cbar.ax.tick_params(labelsize=40, length=8, width=4, pad=20)
        cbar.outline.set_linewidth(4)
        x_lower_limit = 0.23 * 10 ** 7
        x_upper_limit = 0.6 * 10 ** 7
        y_lower_limit = 1.3 * 10 ** 6
        y_upper_limit = 5.6 * 10 ** 6
        ax.set_xlim([x_lower_limit, x_upper_limit])
        ax.set_ylim([y_lower_limit, y_upper_limit])
        ax.set_axis_off()
        if arrows:
            linewidth = 2.5
            lwMin = 0.5
            lwMax = 2.5
            arrowsize_max = 5
            arrowsize_min = 2
            G = self.create_directed_graph(ax, arrows)
            position = nx.get_node_attributes(G, "pos")
            edge_colors = nx.get_edge_attributes(G, "color")
            weights = nx.get_edge_attributes(G, "weight")
            arrowsize = nx.get_edge_attributes(G, "arrowsize")
            for edge in list(G.edges):
                style_kwds["arrowstyle"] = mpatches.ArrowStyle.Simple(head_width=arrowsize[edge],
                                                                      head_length=arrowsize[edge],
                                                                      tail_width=weights[edge])
                # nx.draw_networkx(G, position, edgelist = [edge], edge_color=edge_colors[edge],
                # width=weights[edge], arrowsize=arrowsize[edge], **style_kwds)
                arrow = mpatches.FancyArrowPatch(posA=position[edge[0]], posB=position[edge[1]],
                                                 color=edge_colors[edge], **style_kwds)
                ax.add_patch(arrow)
            nx.draw_networkx_nodes(G, position, node_color="#575757", node_size=0.3, ax=ax)
            if time == 10 or add_legend_edges:
                carrier = plotName.split("_")[0]
                fig = self.add_legend_edges(fig, carrier, max_value, unitName, type, maxLinewidth=lwMax)

        if title:
            plt.title(str(2020 + 2*year) + f', {legend_label}', fontsize=40)
        # Save and display plot
        plot_name = f'europe_{nuts}_{column}_{year}'
        fig.savefig(f'{plot_name}.svg', format='svg', dpi=200, transparent=True)
        fig.savefig(f'{plot_name}.png', format='png', dpi=200, transparent=True)
        plt.show()

    def plot_time_scale(self, dataframe, color_dict, carrier, title=False, colorful=None):
        """plots dataframe over time, with colordict"""

        pivoted_data = dataframe.transpose()
        color = self.eth_colors.get_color('grey')
        # Create plot with temporal change
        ax = pivoted_data.plot(figsize=(30, 20), color=color, linewidth=3, legend=False)
        # Manual values for labelling the interesting lines in total jobs plot
        x_start = 15
        x_end = 15.5
        PAD = 0.1
        end_point = pd.DataFrame([[12799, 8995, 8012, 1034, 1334, 1634, 1934, 2234, 2534, 2930, 3582, 3933]])
        # Plot interesting lines above
        for idx, region in enumerate(colorful):
            if idx < 3:
                pivoted_data[region].plot(color=color_dict[region], linewidth=5, zorder=3.5, ax=ax)
            else:
                pivoted_data[region].plot(color=color_dict[region], linewidth=5, zorder=2.5, ax=ax)
            # Vertical start of line
            y_start = pivoted_data[region].loc[15]
            y_end = end_point.iloc[0, idx]
            # Add line
            ax.plot([x_start, (x_start + x_end - PAD) / 2, x_end - PAD], [y_start, y_end, y_end],
                    color=color_dict[region], ls='dashed')
            # Add text
            ax.text(x_end, y_end, region, color='black', fontsize=25, weight='bold', va='center')
        # Add shade between lines
        ax.fill_between(pivoted_data.index, pivoted_data['IT'], pivoted_data['DE'], interpolate=True,
                       color=color_dict['DE'], alpha=0.2)
        ax.fill_between(pivoted_data.index, pivoted_data['IT'], pivoted_data['ES'],
                        where=pivoted_data['IT'] > pivoted_data['ES'], interpolate=True, color=color_dict['DE'], alpha=0.2)
        ax.fill_between(pivoted_data.index, pivoted_data['DE'], pivoted_data['ES'],
                        where=pivoted_data['ES'] > pivoted_data['DE'], interpolate=True, color=color_dict['DE'],
                        alpha=0.2)
        ax.fill_between(pivoted_data.index, pivoted_data['PL'], pivoted_data['LT'],
                        where=pivoted_data['PL'] > pivoted_data['LT'], interpolate=True, color=color_dict['NL'],
                        alpha=0.2)
        ax.fill_between(pivoted_data.index, pivoted_data['PL'], pivoted_data['NL'],
                        where=pivoted_data['NL'] > pivoted_data['PL'], interpolate=True, color=color_dict['NL'],
                        alpha=0.2)
        # Plot details
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xlabel('Time', fontsize=40, fontdict=dict(weight='bold'))
        plt.ylabel('Total Jobs', fontsize=40, fontdict=dict(weight='bold'))
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        if title:
            plt.title(f'Total Jobs in {carrier.capitalize()}', fontsize=40)
        plt.xticks(np.linspace(0, 15, 16, dtype=int), np.linspace(2020, 2050, 16, dtype=int))
        plot_name = f'time_Total_{carrier}'
        plt.savefig(f'{plot_name}.svg', format='svg', dpi=600, transparent=True)
        plt.savefig(f'{plot_name}.png', format='png', dpi=200, transparent=True)
        plt.show()

    def plot_europe_maps(self, dataframe, column, legend_label=None, year=0, nuts=2, diverging=False, title=True, max_val=0):
        """plots maps of europe with correct x and y-limits, axes names
        :param dataframe: europe joined with data
        :param column: data to plot
        :param: legend_label: Label of legend
        :param nuts: region size
        :param diverging: for diverging cmap
        :param title: bool title
        :param max_val: maximum value for color-bar"""

        # Adjust color-bar scale
        min_val = 0
        if max_val == 0:
            max_val = dataframe[column].max()
            min_val = dataframe[column].min()
            if max_val == min_val:
                max_val += 1

        # Get colormap and tune it
        colormap = self.eth_colors.get_custom_colormaps(self.colormap_dict[column], diverging=diverging)
        norm = plt.Normalize(vmin=min_val, vmax=max_val)
        grey = self.eth_colors.get_color('grey', 20)
        colormap.set_under(color=grey)

        # Had to adjust column name to iterate:
        column = column[:-1]

        # Quick fix for balkan and NUTS0
        gdf_quickfix = self.gdf.copy(deep=True)
        gdf_quickfix[f'{column}{0}'] = pd.NA
        conditions = (gdf_quickfix['LEVL_CODE'] == 2) & (
            gdf_quickfix.index.str.startswith(('AL', 'ME', 'RS', 'MK'))
        )
        gdf_quickfix.loc[conditions, f'{column}{0}'] = -50000.00000
        selected_rows = gdf_quickfix.loc[conditions].dropna(subset=[f'{column}{0}'])
        selected_rows[f'{column}{0}'] = pd.to_numeric(selected_rows[f'{column}{0}'])
        dataframe = dataframe.append(selected_rows)

        # Count the number of plots needed
        num_axes = len([col for col in dataframe.columns if col.startswith(column)])

        fig, axs = plt.subplots(figsize=(33, 10), ncols=num_axes, nrows=1, squeeze=False)

        # Loop over the dataframe and plot the interesting columns with correct plot details
        for i in range(0, num_axes):
            if i != 0:
                dataframe.loc[conditions, f'{column}{i}'] = -50000.00000
            dataframe.loc[(dataframe['LEVL_CODE'] == nuts) & ~(dataframe.index.str.startswith(('TR', 'IS'))),
                          f'{column}{i}'] = dataframe.loc[(dataframe['LEVL_CODE'] == nuts) &
                                                          ~(dataframe.index.str.startswith(
                                                              ('TR', 'IS'))), f'{column}{i}'].fillna(-50000)
            dataframe.plot(column=f'{column}{i}', cmap=colormap, legend=False, edgecolor='black', linewidth=0.5,
                           ax=axs[0, i], norm=norm)
            if nuts == 2:
                # Only take NUTS2 regions that have model values assigned
                mask = dataframe[f'{column}{i}'].notnull()
                # Get the first two characters of the index for rows where 'column' is not null
                idx_values = dataframe[mask].index.str[:2]
                # Filter the 'nuts0' dataframe based on the first two characters of the index and 'LEVL_CODE'
                nuts0 = gdf_quickfix[(gdf_quickfix.index.isin(idx_values)) & (gdf_quickfix['LEVL_CODE'] == 0)]
                nuts0.boundary.plot(edgecolor='black', linewidth=1, ax=axs[0, i])

            x_lower_limit = 0.25 * 10 ** 7
            x_upper_limit = 0.6 * 10 ** 7
            y_lower_limit = 1.3 * 10 ** 6
            y_upper_limit = 5.6 * 10 ** 6
            axs[0, i].set_xlim([x_lower_limit, x_upper_limit])
            axs[0, i].set_ylim([y_lower_limit, y_upper_limit])
            axs[0, i].set_axis_off()

        # Add description and colorbar, save the plot
        axs[0, 0].set_title('2020', fontsize=50, fontdict=dict(weight='bold'))
        axs[0, 1].set_title('2030', fontsize=50, fontdict=dict(weight='bold'))
        axs[0, 2].set_title('2050', fontsize=50, fontdict=dict(weight='bold'))
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=colormap, norm=norm), ax=axs[0, num_axes - 1])
        cbar.set_label(legend_label, size=50, labelpad=15)
        cbar.ax.tick_params(labelsize=40, length=8, width=4, pad=20)
        cbar.outline.set_linewidth(4)
        if title:
            plt.title(str(2020 + 2*year) + f', {legend_label}', fontsize=40)
        plot_name = f'europecombo_{nuts}_{column}'
        fig.savefig(f'{plot_name}.svg', format='svg', dpi=600, transparent=True)
        fig.savefig(f'{plot_name}.png', format='png', dpi=200, transparent=True)
        plt.show()

    def plot_pie_chart(self, dataframe, year):
        """Make pie chart of dataframe"""
        # Assign color and label to each tech from dataframe
        colors = []
        labels = []
        numbers = []
        for index, row in dataframe.iterrows():
            tech = row[0]
            color = self.eth_colors.get_color(self.colormap_dict[tech], 80)
            colors.append(color)
            if tech == 'SMR':
                labels.append(f'{tech}')
            elif tech == 'anaerobic_digestion':
                labels.append('Anaerobic digestion')
            else:
                labels.append(f'{tech.capitalize()}')
            numbers.append(int(row[1]))  # Get the total number for each tech and convert to integer
        # Plot with legend of all techs and total jobs in each tech
        fig, ax = plt.subplots(figsize=(30, 20))
        wedges, _, _ = ax.pie(numbers, colors=colors, labels=labels, autopct='',
                              textprops={'fontsize': 60, 'weight': 'bold'})
        for wedge in wedges:
            wedge.set_edgecolor('black')
            wedge.set_linewidth(2)
        # Add total numbers as annotations inside the wedges
        for i, wedge in enumerate(wedges):
            angle = (wedge.theta2 - wedge.theta1) / 2.0 + wedge.theta1
            x = np.cos(np.deg2rad(angle))
            y = np.sin(np.deg2rad(angle))
            distance = 0.5  # adjust distance of the label from the center of the wedge
            text_x = x * distance
            text_y = y * distance
            # Flip the orientation of upside-down numbers
            if 90 < angle <= 270:

                ax.text(np.cos(np.deg2rad(180))*0.5, np.sin(np.deg2rad(180))*0.5, str(numbers[i]),
                        ha='center', va='center', fontsize=50, fontdict=dict(weight='bold'))
            else:
                ax.text(text_x, text_y, str(numbers[i]), ha='center', va='center', fontsize=50, rotation_mode='default',
                        rotation=angle, fontdict=dict(weight='bold'))
        plot_name = f'pie_chart_{year}'
        fig.savefig(f'{plot_name}.png', format='png', dpi=200, transparent=True)
        fig.savefig(f'{plot_name}.svg', format='svg', dpi=200, transparent=True)
        plt.show()

    def lorenz_plot(self, dataframe, column, year, title=False):
        """Lorenz curve plot of dataframe"""
        # Read in population number from csv sheet, source: EUROSTAT2018
        population = self.population

        # Combine dataframe to NUTS0 and NUTS2
        population0 = self.combine_to_nuts0(population)
        dataframe0 = pd.DataFrame(self.combine_to_nuts0(dataframe))
        dataframe2 = pd.DataFrame(dataframe)

        # Merge dataframes with population
        df_merged0 = dataframe0.merge(population0, left_index=True, right_index=True, how='left')
        df_merged2 = dataframe2.merge(population, left_index=True, right_index=True, how='left')

        # Sort the merged dataframes by the 'column' values in ascending order
        df_merged0.sort_values(by=column, inplace=True)
        df_merged2.sort_values(by=column, inplace=True)

        # Calculate cumulative percentage and share
        cumulative_percentage0 = np.cumsum(df_merged0[column]) / df_merged0[column].sum()
        cumulative_share0 = np.cumsum(df_merged0['OBS_VALUE']) / df_merged0['OBS_VALUE'].sum()
        cumulative_percentage2 = np.cumsum(df_merged2[column]) / df_merged2[column].sum()
        cumulative_share2 = np.cumsum(df_merged2['OBS_VALUE']) / df_merged2['OBS_VALUE'].sum()

        # Get the colors for each Lorenz curve
        color2 = self.eth_colors.get_color('green')
        color0 = self.eth_colors.get_color('blue')
        grey = self.eth_colors.get_color('grey', 60)

        # Create Lorenz curve
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.plot(cumulative_share2, cumulative_percentage2, color=color2, linewidth=5)
        ax.plot(cumulative_share0, cumulative_percentage0, color=color0, linewidth=5)
        plt.text(0.62, 0.4, 'Countries', size=40, rotation=64, color=color0,
                         ha="center", va="center", bbox=dict(ec='1', fc='1'))
        plt.text(0.87, 0.4, 'Regions', size=40, rotation=75, color=color2,
                         ha="center", va="center", bbox=dict(ec='1', fc='1'))
        ax.plot([0, 1], [0, 1], color=grey, linestyle='--', linewidth=5)
        plt.xlabel('Cumulative proportion of population', fontsize=40, fontdict=dict(weight='bold'))
        plt.gca().yaxis.set_label_position("right")
        plt.gca().yaxis.tick_right()
        plt.ylabel(f'Cumulative proportion of {column}', fontsize=40, fontdict=dict(weight='bold'))
        if title:
            plt.title(f'Lorenz Curve in {str(2020 + 2 * year)} for NUTS0/2 regions', fontsize=50,
                    fontdict=dict(weight='bold'))
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.savefig(f'lorenz_{year}.png', format='png', dpi=200, transparent=True)
        plt.savefig(f'lorenz_{year}.svg', format='svg', dpi=200, transparent=True)
        plt.show()

    def gini_plot(self, dataframe, column, year):
        """plot gini coefficient for all NUTS0 regions"""
        # Get population data
        dataframe = pd.DataFrame(dataframe)
        population = self.population

        # Get initials for all NUTS0 regions
        initials = [label[:2] for label in dataframe.index]

        # Merge dataframes
        df_merged = dataframe.merge(population, left_index=True, right_index=True, how='left')

        # Only taking unique values
        unique_initials = set(initials)

        # Calculate the GINI coefficient for every country
        gini_df = pd.DataFrame(columns=['Gini Coefficient'])
        for country in unique_initials:
            country_data = df_merged[df_merged.index.str.startswith(country)]
            # For countries with no jobs
            if country_data[column].sum == 0:
                gini = None
            else:
                gini = self.gini_coefficient(dataframe=country_data, column=column)
            # Append the Gini coefficient to the dataframe with the country as the index
            gini_df.loc[country] = [gini]

        # Calculate Gini coefficient for Europe
        gini_europe = self.gini_coefficient(dataframe=df_merged, column=column)
        # Drop NaN values and sort
        gini_df = gini_df.dropna()
        gini_df.sort_values('Gini Coefficient', ascending=False, inplace=True)
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 15))
        # Create diverging colormap with middle at Gini Europe
        cmap = self.eth_colors.get_custom_colormaps(self.colormap_dict['gini'], diverging=True)
        norm = colors.TwoSlopeNorm(vmin=gini_df['Gini Coefficient'].min(), vcenter=gini_europe,
                                      vmax=gini_df['Gini Coefficient'].max())
        # Iterate over the dataframe and plot each horizontal line separately
        for i, (country, gini_coefficient) in enumerate(gini_df.iterrows()):
             color = cmap(norm(gini_coefficient))
             plt.hlines(y=i / 2, xmin=gini_europe, xmax=gini_coefficient, colors=color, linewidth=25,
                       path_effects=[mpe.withStroke(foreground='black', linewidth=28)])

        # Plot the Gini for Europe
        ax.axvline(gini_europe, color='black', linestyle='--')
        ax.annotate('Gini Europe', xy=(gini_europe, 10), xytext=(gini_europe, ax.get_ylim()[1] + 0.2),
                    textcoords='data', ha='center', fontsize=25)
        # Invert the x-axis
        ax.invert_xaxis()
        # Set x-axis limits
        ax.set_xlim(1, 0)
        plt.yticks(np.arange(len(gini_df))/2, gini_df.index, fontsize=25, fontdict=dict(weight='bold'))
        plt.xticks(fontsize=25)
        plt.xlabel('Gini coefficient', fontsize=30, fontdict=dict(weight='bold'))
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.savefig(f'gini_{year}.png', format='png', dpi=200, transparent=True)
        plt.savefig(f'gini_{year}.svg', format='svg', dpi=200, transparent=True)
        plt.show()


if __name__ == "__main__":
    results_path = os.path.join("outputs", "HSC_NUTS2")
    scenario = "scenario_referenceBM_med"  # to load more than one scenario, pass a list
    techs = ['SMR', 'gasification', 'electrolysis', 'anaerobic_digestion']
    tr_techs = ['hydrogen_truck_liquid', 'hydrogen_truck_gas']
    countries = ['DE', 'RO', 'FR', 'IT', 'UK']
    years = [0, 5, 15]
    nutss = [0, 2]
    vis = Visualization(results_path, scenario)
    # Loop to plot all technologies for every year and nuts:
    vis.plot_hydrogen_jobs(techs=techs, scenario=scenario, year=15, plot='lorenz', nuts=2, max_val=2000)
    vis.plot_tech_jobs(techs='SMR', carrier='hydrogen', scenario=scenario, year=15, rel=True)
    vis.plot_export_potential(scenario=scenario, year=15)
    vis.plot_tech_jobs(techs='SMR', carrier='hydrogen', scenario=scenario, year=15, max_val=300)
    vis.plot_biomethane_shares(years=years, scenario=scenario)
    vis.plot_tech_jobs(techs='anaerobic_digestion', carrier='biomethane', scenario=scenario, year=15)
    vis.plot_total_change(techs=techs, scenario=scenario, time=15)
    vis.plot_carrier_usage(carrier='carbon', year=15, scenario=scenario)