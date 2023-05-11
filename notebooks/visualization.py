import os


from zen_garden.postprocess.results import Results
from eth_colors import ETHColors
import geopandas as gpd
import matplotlib.pyplot as plt
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
            "export": ("red", "blue"),
            "jobs": "green",
            "dry_biomass": "green",
            "biomethane_share": "green"
        }
    def combine_to_nuts0(self, dataframe):
        """combines from NUTS2 to NUTS0 level
        :param dataframe: dataframe containing information on NUTS2 level"""

        initials = [label[:2] for label in dataframe.index]
        dataframe = dataframe.groupby(initials).sum()
        dataframe.index.name = "NUTS0"
        return dataframe

    def merge_nutsdata(self, dataframe, nuts=2):
        """takes dataframe and returns NUTS2 merged dataframe
        :param dataframe: dataframe used for merging
        :param nuts: which NUTS level"""

        path_to_data = os.path.join("outputs", "NUTS_RG_01M_2016_3035", "NUTS_RG_01M_2016_3035.shp")
        gdf = gpd.read_file(path_to_data)
        gdf = gdf[["NUTS_ID", "geometry", "LEVL_CODE"]]
        # For NUTS0 calculations:
        if nuts == 0:
            dataframe = self.combine_to_nuts0(dataframe)

        gdf_merged = gdf.merge(dataframe, left_on="NUTS_ID", right_index=True, how='left')
        gdf_merged = gdf_merged.set_index('NUTS_ID')
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

    def combine_tech_CCS(self, total_jobs, tech, calc_method, scenario, year=0, carrier='hydrogen', time=False, country=None):
        """If gasification or SMR additionally add the CCS jobs:"""

        job_ratio_CCS = self.job_ratio_results(tech=f'{tech}_CCS', calc_method=calc_method)
        capacity_CCS = self.results.get_total("capacity", scenario=scenario, element_name=f'{tech}_CCS')
        total_CCS_jobs = capacity_CCS.droplevel(0) * job_ratio_CCS
        output_flow_CCS = self.results.get_total("flow_conversion_output", scenario=scenario, element_name=f'{tech}_CCS')
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
    def calc_rel(self, dataframe, nuts):
        """dataframe is divided by absolute value"""

        # Read in job ratios from csv sheet, source: EUROSTAT
        path_to_data = os.path.join("outputs", "demo_r_d2jan_page_linear.csv")
        df = pd.read_csv(path_to_data)
        df = df[['geo', 'OBS_VALUE']]

        if nuts == 0:
            df = self.combine_to_nuts0(df)

        gdf_merged = dataframe.merge(df, left_on='NUTS_ID', right_on='NUTS0', how='left')
        gdf_merged['jobs'] = (gdf_merged['jobs'] / gdf_merged['OBS_VALUE']) * 100000

        return gdf_merged

    def plot_output_flows(self, carrier, year, scenario=None, techs=None, nuts=2):
        """plot output flows for selected carriers"""

        output_flow = self.results.get_total("flow_conversion_output", scenario=scenario, year=year)

        # carrier availability
        # self.results.get_total("availability_carrier_import", scenario=scenario).groupby(["carrier", "node"]).mean().loc["dry_biomass"]
        # self.results.get_total("import_carrier_flow", scenario=scenario).loc["dry_biomass"]  # in

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
        self.plot_europe_map(gdf_merged, column=output_flow.name, legend_label=f'{carrier.capitalize()} Production in [GWh]', year=year, nuts=nuts)

    def plot_carrier_usage(self, carrier, year, scenario=None, nuts=2):
        """plot carrier availability and demand"""

        # Get available stock of selected carrier
        carrier_available = self.results.get_total("availability_import", scenario=scenario, year=year).groupby(["carrier", "node"]).mean().loc[carrier]
        carrier_used = self.results.get_total("flow_conversion_input", scenario=scenario, year=year).groupby(["carrier", "node"]).mean().loc[carrier]
        # Load NUTS2 data and join with carrier_available
        potential_used = (carrier_used/carrier_available) * 100
        potential_used.name = carrier
        gdf_merged = self.merge_nutsdata(potential_used, nuts=nuts)

        # Plot as a map of Europe:
        self.plot_europe_map(gdf_merged, column=potential_used.name, legend_label=f'Percent of {carrier} used', year=year, nuts=nuts, title=False)


    def plot_demand(self, carrier, year, scenario=None, nuts=2):
        """plot hydrogen demand"""

        demand = self.results.get_total("demand", scenario=scenario, year=year)

        demand = demand.loc[carrier].groupby(["node"]).sum()/1000  # in TWh

        # Load geographical data and join with demand
        demand.name = f"{carrier}_demand"
        gdf_merged = self.merge_nutsdata(demand, nuts=nuts)

        # Plot as a map of Europe
        self.plot_europe_map(gdf_merged, column=demand.name, legend_label=f'{carrier.capitalize()} Demand in [TWh/year]', year=year, nuts=nuts)

    def plot_tech_jobs(self, year, techs, carrier='hydrogen', scenario=None, calc_method='mean', nuts=2):
        """load data from Excel file and plot total jobs for specific year, tech"""

        # Read in data from Excel file and use calculation method
        job_ratio = self.job_ratio_results(tech=techs, calc_method=calc_method)  # jobs per GW hydrogen

        # get data from HSC modeling results
        capacity = self.results.get_total("capacity", scenario=scenario, year=year, element_name=techs)  # in GW
        output_flow = self.results.get_total("flow_conversion_output", scenario=scenario, year=year, element_name=techs)
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
        self.plot_europe_map(dataframe=gdf_merged, column=total_jobs.name, legend_label=f'Total Jobs, {techs.capitalize()}', year=year, nuts=nuts, title=False)

    def plot_export_potential(self, carrier='hydrogen', scenario=None, year=0, nuts=2, diverging=True):
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

        # Plot as a map of Europe
        self.plot_europe_map(dataframe=gdf_merged, column=export.name, legend_label=f'Excess Production, {carrier.capitalize()} [TWh/year]', year=year, nuts=nuts, diverging=diverging, title=False)


    def plot_tech_change(self, tech, carrier='hydrogen', scenario=None, calc_method='median', nuts=2, time=0, title=False):
        """load data from Excel and plot temporal change for specific tech for NUTS0/2 regions"""

        # Read in data from the Excel
        job_ratio = self.job_ratio_results(tech=tech, calc_method=calc_method)

        # get data from HSC modeling results
        capacity = self.results.get_total("capacity", scenario=scenario, element_name=tech)
        output_flow = self.results.get_total("flow_conversion_output", scenario=scenario, element_name=tech)
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

        self.plot_time_scale(dataframe=total_jobs, color_dict=color_dict, carrier=carrier, colorful=colorful, title=title)

    def plot_total_change(self, techs, carrier='hydrogen', scenario=None, calc_method='mean', nuts=0, time=0, title=False):
        """Total jobs for every country for specified techs"""
        total_jobs = np.zeros_like(
            self.results.get_total("capacity", scenario=scenario, element_name='SMR').droplevel(0))

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

        # Produce color_dict
        colorful = np.empty(0, dtype=object)
        jobs_at_time = total_jobs.iloc[:, time]
        colorful = np.concatenate([colorful, jobs_at_time.nlargest(3).index.values])
        colorful = np.concatenate([colorful, jobs_at_time.nlargest(7).nsmallest(4).index.values])
        colorful = np.concatenate([colorful, jobs_at_time.nsmallest(len(jobs_at_time) - 7).index.values])

        color_dict = {}
        for region in total_jobs.index:
            if region in colorful[:3]:
                color_dict[region] = self.eth_colors.get_color('bronze')
            elif region in colorful[3:7]:
                color_dict[region] = self.eth_colors.get_color('purple')
            else:
                color_dict[region] = self.eth_colors.get_color('blue')

        # Print with timescale
        self.plot_time_scale(dataframe=total_jobs, color_dict=color_dict, carrier=carrier, colorful=colorful, title=title)

    def country_jobs_change(self, country, techs, carrier='hydrogen', scenario=None, calc_method='median', nuts=0):
        """plot change of jobs for all technologies for a country or NUTS2 regions in country"""

        # Read in job ratios from Excel sheet
        path_to_data = os.path.join("employment_data", "job_ratio_results.xlsx")
        df = pd.read_excel(path_to_data)

        grouped_df = df.groupby('technology')['job_ratio'].agg(calc_method)
        # Create a new DataFrame to store the results
        job_ratios_df = pd.DataFrame({'technology': grouped_df.index, 'job_ratio': grouped_df.values})
        # get data from HSC modeling results
        fig, ax = plt.subplots(figsize=(30, 20))

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
                                                   scenario=scenario, country=country)
            # Sum over NUTS0
            if nuts == 0:
                total_jobs = self.combine_to_nuts0(total_jobs)

            pivoted_data = total_jobs.transpose()
            color = self.eth_colors.get_color(self.colormap_dict[tech])

            # Add the plot to the figure object
            pivoted_data.plot(ax=ax, color=color, linewidth=5, label=tech)

        # Plot details:
        ax.set_ylabel(f'Jobs', fontsize=30)
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


    def plot_hydrogen_jobs(self, techs, year=0, carrier='hydrogen', scenario=None, calc_method='mean', nuts=2, rel=False):
        """loop over all hydrogen producing technologies sum and plot total"""

        jobs = np.zeros_like(self.results.get_total("capacity", scenario=scenario, year=year, element_name='SMR').droplevel(0))
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
            pie_data.append((tech, tech_jobs.sum()))  # Append a tuple containing the tech name and the total jobs to pie_data

        # Create a DataFrame from the list of tuples
        if rel == False:
            pie_dataframe = pd.DataFrame(pie_data, columns=['Technology', 'Total Jobs'])
            self.plot_pie_chart(pie_dataframe, year=year)

        # merge with NUTS2 data:
        jobs.name = 'jobs'
        unit = '[FTE]'
        gdf_merged = self.merge_nutsdata(jobs, nuts=nuts)

        if rel:
            gdf_merged = self.calc_rel(gdf_merged, nuts=nuts)
            unit = "[FTE] per 100'000 capita"

        # plot as a map with limited axes
        self.plot_europe_map(dataframe=gdf_merged, column='jobs',
                        legend_label=f'{jobs.name.capitalize()} in Hydrogen, {unit}', year=year, nuts=nuts, title=False)

    def plot_jobs_transport(self, tech, year, scenario=None, calc_method='mean', nuts=2):

        # get job ratio from results Excel
        job_ratio = self.job_ratio_results(tech, calc_method=calc_method)

        # get carrier flow results and distances between regions and combine
        carrier_flow = self.results.get_total("flow_transport", scenario=scenario, year=year, element_name=tech)
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
        self.plot_europe_map(dataframe=gdf_merged, column=tech_jobs.name, legend_label=f'Total Jobs, {tech.capitalize()}, {calc_method}', year=year, nuts=nuts)

    def plot_biomethane_share(self, year, scenario=None, nuts=2):
        """plot the share of biomethane as input in the SMR/SMRCCS plants"""

        input_flow = self.results.get_total("flow_conversion_input", scenario=scenario, year=year)

        # Get the data from the model for biomethane used for conversion
        biomethane_use = input_flow.loc['biomethane_conversion'].groupby(["node"]).sum()

        # Get the data from the model for NG(NG&biomethane) used in SMR/SMRCCS
        naturalgas_use = input_flow.loc['SMR'].groupby(['node']).sum() + input_flow.loc['SMR_CCS'].groupby(['node']).sum()

        # Share of biomethane in the total gas consumption for SMR/SMRCCS
        biomethane_share = (biomethane_use / naturalgas_use) * 100

        biomethane_share.name = 'biomethane_share'

        gdf_merged = self.merge_nutsdata(biomethane_share, nuts=nuts)

        self.plot_europe_map(dataframe=gdf_merged, column=biomethane_share.name, legend_label=f'Biomethane share in percent', year=year, nuts=nuts, title=False)

    def plot_renewables(self, year, scenario=None, nuts=2):
        """plot renewables and share used for electrolysis"""

        techs = ["pv_ground", "wind_onshore"]

        renewable_tot = np.zeros_like(self.results.get_total("flow_conversion_output", scenario=scenario, year=year, element_name='pv_ground').groupby(['node']).sum())
        for tech in techs:
            renewable_prod = self.results.get_total("flow_conversion_output", scenario=scenario, year=year, element_name=tech).groupby(['node']).sum()/1000
            renewable_prod.name = tech
            gdf_merged = self.merge_nutsdata(dataframe=renewable_prod, nuts=nuts)

            self.plot_europe_map(dataframe=gdf_merged, column=tech, legend_label=f'{tech.capitalize()}', year=year, nuts=nuts)
            renewable_tot += renewable_prod
        renewable_tot.name = 'renew_tot'
        gdf_merged = self.merge_nutsdata(renewable_tot, nuts=nuts)

        self.plot_europe_map(dataframe=gdf_merged, column=renewable_tot.name, legend_label=f'Renewable production [TWh/year]', year=year, nuts=nuts, title=False)



    def plot_europe_map(self, dataframe, column, legend_label=None, year=None, nuts=2, diverging=False, title=True):
        """plots map of europe with correct x and y-limits, axes names
        :param dataframe: europe joined with data
        :param column: data to plot
        :param colormap: colormap for plotting
        :param nuts: region size
        :param diverging: for diverging cmap
        :param title: bool title"""

        min_val = 0
        # Get colors for technologies (is there a better way?)
        if column == 'export' and nuts == 2:
            min_val = -7
            max_val = 7
        elif column == 'export' and nuts == 0:
            min_val = -3
            max_val = 3
        elif column == 'demand' and nuts == 0:
            max_val = 40
        elif column == 'jobs' and nuts == 0:
            max_val = 60
        elif column == 'jobs' and nuts == 2:
            max_val = 500
        elif column == 'dry_biomass' and nuts == 2:
            max_val = 100
        elif column == 'SMR':
            max_val = 250
        elif column == 'electrolysis':
            max_val = 1300
        elif column == 'gasification':
            max_val = 400
        elif column == 'biomethane_share':
            max_val = 100
        elif column == 'renew_tot':
            max_val = 20
        else:
            max_val = dataframe[column].max()
            min_val = dataframe[column].min()
            if max_val == min_val:
                max_val += 1

        colormap = self.eth_colors.get_custom_colormaps(self.colormap_dict[column], diverging=diverging)
        norm = plt.Normalize(vmin=min_val, vmax=max_val)
        grey = self.eth_colors.get_color('grey', 40)

        dataframe.loc[(dataframe['LEVL_CODE'] == nuts) & ~(dataframe.index.str.startswith(('TR', 'IS'))), column] = \
           dataframe.loc[(dataframe['LEVL_CODE'] == nuts) & ~(dataframe.index.str.startswith(('TR', 'IS'))), column].fillna(-50000)
        colormap.set_under(color=grey)

        fig, ax = plt.subplots(figsize=(30, 20))

        dataframe.plot(column=column, cmap=colormap, legend=False, edgecolor='black', linewidth=0.5, ax=ax, norm=norm)
        if nuts == 2:
            # Only take NUTS2 regions that have model values assigned
            mask = dataframe[column].notnull()
            # Get the first two characters of the index for rows where 'column' is not null
            idx_values = dataframe[mask].index.str[:2]
            # Filter the 'nuts0' dataframe based on the first two characters of the index and 'LEVL_CODE'
            nuts0 = dataframe[(dataframe.index.isin(idx_values)) & (dataframe['LEVL_CODE'] == 0)]
            nuts0.boundary.plot(edgecolor='black', linewidth=1, ax=ax)

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
        if title:
            plt.title(str(2020 + 2*year) + f', {legend_label}', fontsize=40)
        plot_name = f'europe_{nuts}_{column}_{year}'
        # fig.savefig(f'{plot_name}.svg', format='svg', dpi=600, transparent=True)
        fig.savefig(f'{plot_name}.png', format='png', dpi=200, transparent=True)
        plt.show()

    def plot_time_scale(self, dataframe, color_dict, carrier, title=False, colorful=None):
        """plots dataframe over time, with colordict"""

        pivoted_data = dataframe.transpose()

        # Create plot with temporal change
        ax = pivoted_data.plot(figsize=(30, 20), color=color_dict, linewidth=3)
        # Plot interesting lines above
        for region in colorful:
            pivoted_data[region].plot(figsize=(30, 20), color=color_dict[region], linewidth=5, label='_nolegend_')

        # Plot details
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xlabel('Time', fontsize=40, fontdict=dict(weight='bold'))
        plt.ylabel('Total Jobs', fontsize=40, fontdict=dict(weight='bold'))
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.legend(prop={'size': 20}, frameon=False)
        if title:
            plt.title(f'Total Jobs in {carrier.capitalize()}', fontsize=40)
        plt.xticks(np.linspace(0, 15, 16, dtype=int), np.linspace(2020, 2050, 16, dtype=int))
        plot_name = f'time_Total_{carrier}'
        plt.savefig(f'{plot_name}.svg', format='svg', dpi=600, transparent=True)
        plt.savefig(f'{plot_name}.png', format='png', dpi=200, transparent=True)
        plt.show()

    def plot_pie_chart(self, dataframe, year):
        """make pie chart of dataframe"""
        # Assign color to each tech from dict
        colors = []
        for tech in dataframe.iloc[:, 0]:
            color = self.eth_colors.get_color(self.colormap_dict[tech])
            colors.append(color)
        # Plot with legend of all techs and total jobs in each tech
        fig, ax = plt.subplots(figsize=(40, 20))
        wedges = ax.pie(dataframe.iloc[:, 1], colors=colors)
        ax.legend(wedges[0], [f"{tech.upper()}: {int(round(value))} Jobs" for tech, value in zip(dataframe.iloc[:, 0],
                  dataframe.iloc[:, 1])], title="Technologies:", title_fontsize=60, prop={'size': 40}, frameon=False,
                  loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        for wedge in wedges[0]:
            wedge.set_edgecolor('black')
            wedge.set_linewidth(2)
        plot_name = f'pie_chart_{year}'
        fig.savefig(f'{plot_name}.png', format='png', dpi=200, transparent=True)
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
    vis.plot_total_change(techs=techs, scenario=scenario, time=15)
    vis.plot_hydrogen_jobs(carrier='hydrogen', techs=techs, year=15, scenario=scenario)
    vis.plot_demand(carrier='hydrogen', year=15, nuts=2, scenario=scenario)
    vis.plot_jobs_transport(tech='hydrogen_train', year=15, scenario=scenario, calc_method='mean', nuts=2)
    vis.plot_tech_change(tech='electrolysis', scenario=scenario, nuts=0, time=15)
    for year in years:
        vis.plot_biomethane_share(year=year, scenario=scenario, nuts=2)
        vis.plot_renewables(year=year, scenario=scenario)
    for year in years:
        for tech in techs:
            vis.plot_tech_jobs(year=year, scenario=scenario, techs=tech)

    # Test functions for plotting:

    vis.plot_output_flows(carrier='dry_biomass', year=15, scenario=scenario)
    vis.plot_export_potential(scenario=scenario, year=15)
    vis.country_jobs_change(carrier='hydrogen', techs=techs, scenario=scenario, country='DE', nuts=2)
    vis.plot_output_flows(carrier="hydrogen", scenario=scenario, techs="conversion", year=5)
    vis.plot_demand("hydrogen", scenario=scenario, year=5, nuts=0)
    vis.plot_demand("hydrogen", scenario=scenario, year=5, nuts=0)
    vis.plot_tech_jobs(scenario=scenario, techs='SMR', year=5, calc_method='max', nuts=0)

