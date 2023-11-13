import os
from zen_garden.postprocess.results import Results
from eth_colors import ETHColors
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patheffects as mpe
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import MultiPolygon
import networkx as nx
import pandas as pd
import numpy as np

# Adjust font to LaTeX font
mpl.rcParams.update(mpl.rcParamsDefault)
                    #font_params = {"ytick.color":
# "black","xtick.color": "black",
# "axes.labelcolor": "black",
                      # "axes.edgecolor": "black",
                      # "text.usetex": True,
                      # "font.family": "serif",
                      # "font.serif": ["Computer Modern Serif"]}

#plt.rcParams.update(font_params)
mpl.rcParams["font.size"] = 6
mpl.rcParams["axes.titlesize"] = 7



class Visualization:

    def __init__(self, dataset, scenarios):
        """load results
        :param results_path: folder path results
        :param scenarios: list of scenarios"""
        self.font_params = None
        if scenarios and not isinstance(scenarios, list):
            scenarios = [scenarios]
        self.dataset = dataset
        self.scenarios = scenarios
        results_path = os.path.join("outputs", dataset)
        self.results = Results(results_path, scenarios=scenarios)
        self.system = self.results.results["system"]
        self.analysis = self.results.results["analysis"]
        self.eth_colors = ETHColors()
        self.set_colormaps()
        self.set_gdf()
        self.set_pts()
        self.set_population()
        self.set_job_ratios()

    def set_folder_figures(self, scenario):
        # create folder for figures
        self.folder_figures = os.path.join("figures", self.dataset, scenario)
        if not os.path.exists(os.path.join("figures", self.dataset)):
            os.mkdir(os.path.join("figures", self.dataset))
        if not os.path.exists(self.folder_figures):
            os.mkdir(self.folder_figures)

    def set_colormaps(self):
        """define colors for colormaps"""
        self.colormap_dict = {
            "SMR": "bronze",
            "SMR_CCS": "bronze",
            "gasification": "green",
            "gasification_CCS": "green",
            "electrolysis": "blue",
            "anaerobic_digestion": "petrol",
            "pv_ground": "red",
            "pv": "red",
            "wind_onshore": "blue",
            "wind_offshore": "petrol",
            "wind": "petrol",
            'renew_tot': "purple",
            "hydrogen_demand": "purple",
            "hydrogen_output": "red",
            "hydrogen_truck_gas": "blue",
            "hydrogen_truck_liquid": "petrol",
            "export": ("red", "blue"),
            "gini": ("blue", "red"),
            "jobs": "purple",
            "dry_biomass": "green",
            "wet_biomass": "green",
            "carbon": "red",
            "hydrogen": "blue",
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
        cumulative_share = np.cumsum(sorted_data["population"]) / sorted_data["population"].sum()
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
        path_to_data = os.path.join("geodata", "NUTS_RG_10M_2016_3035.shp")
        gdf = gpd.read_file(path_to_data)
        gdf = gdf[["NUTS_ID", "geometry", "LEVL_CODE","NAME_LATN"]]
        gdf = gdf.set_index("NUTS_ID")
        gdf = gdf[gdf.LEVL_CODE.isin([0, 2])]
        # remove all areas that are smaller than 0.1% of Luxembourg, old threshold: 1.9e8
        exploded = gdf.explode(column="geometry", index_parts=True)
        for idx0 in gdf[(gdf.geometry.type == "MultiPolygon") & (gdf.LEVL_CODE==2)].index:
            for idx1, row in exploded.loc[idx0].iterrows():
                neighbors = len(exploded.loc[idx0][row.geometry.touches(exploded.loc[idx0].geometry)].index)
                if neighbors == 0 and row.geometry.area < 1.9e8 and not "London" in row.NAME_LATN:
                    exploded = exploded.drop((idx0, idx1))
            assert len(exploded.loc[idx0]) > 0, f"Multipolygon for {idx0} is empty"
            gdf.geometry.loc[idx0] = MultiPolygon(exploded.loc[idx0].geometry.values)
        self.gdf = gdf

    def set_population(self):
        """set population dataframe"""
        # Read in population number from csv sheet, source: EUROSTAT
        path_to_data = os.path.join("employment_data", "demo_r_d2jan_linear.csv.gz")
        df = pd.read_csv(path_to_data,compression='gzip')
        df = df[df["TIME_PERIOD"]==2016]
        df = df[df["age"]=="TOTAL"]
        df = df[df["sex"]=="T"]
        population = df[['geo', 'OBS_VALUE']]
        population.set_index('geo', inplace=True)
        index = self.gdf[self.gdf["LEVL_CODE"]==2].index.intersection(population.index)
        missing_index = population.index.difference(self.gdf.index)
        missing_index = missing_index.union(self.gdf[self.gdf["LEVL_CODE"]==2].index.difference(population.index))
        if len(missing_index)>1:
            print("The following indices are missing from gdf", missing_index)
        self.population = population.loc[index]
        self.population.index.names = ["node"]
        self.population = self.population.rename(columns={"OBS_VALUE":"population"})

    def merge_nutsdata(self, dataframe, nuts=2):
        """takes dataframe and returns NUTS2 merged dataframe
        :param dataframe: dataframe used for merging
        :param nuts: which NUTS level"""

        gdf = self.gdf.copy(deep=True)
        # For NUTS0 calculations:
        if nuts == 0:
            dataframe = self.combine_to_nuts0(dataframe)
        gdf.index.name = "node"
        gdf_merged = gdf.merge(dataframe, left_index=True, right_index=True, how='right')
        return gdf_merged

    def set_job_ratios(self, folder="employment_data", filename="job_ratio_results", type="om"):
        """set job ratios"""
        filename=filename+"_"+type+".xlsx"
        path_to_data = os.path.join(folder, filename)
        self.df_job_ratios = pd.read_excel(path_to_data)

    def get_job_ratio(self, tech, calc_method='mean', type="om"):
        """takes tech and returns corresponding job ratio in FTE/GW or FTE/(GW*km) with chosen calc method
        :param tech: name of technology
        :param calc_method: method of using employment data"""
        self.set_job_ratios(type=type)
        if tech.endswith("CCS"):
            tech = tech.split("_")[0]
        filtered_df = self.df_job_ratios[self.df_job_ratios['tech_code'] == tech]
        job_ratio = filtered_df['value'].agg(calc_method)
        return job_ratio

    def combine_tech_CCS(self, total_jobs, tech, calc_method, scenario, year=0, carrier='hydrogen', time=False,
                         country=None, capacity_only=False):
        """If gasification or SMR additionally add the CCS capacity or jobs"""

        job_ratio_CCS = self.get_job_ratio(tech=f'{tech}_CCS', calc_method=calc_method)
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
        gdf_merged[column] = (gdf_merged[column] / gdf_merged["population"]) * 100000

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
        output_flow = output_flow.loc[:, carrier].groupby(["node"]).sum() / 1000  # in TWh

        # Load NUTS2 data and joins with output_flow
        output_flow.name = carrier
        gdf_merged = self.merge_nutsdata(output_flow, nuts=nuts)

        # Plot as a map of Europe:
        self.plot_europe_map(gdf_merged, column=output_flow.name,
                             legend_label=f'{carrier.capitalize()} production [TWh]', year=year, nuts=nuts)

    def plot_conversion_input(self, carrier, year, scenario=None, nuts=2):
        """plot conversion input for selected carrier"""

        # Get available stock of selected carrier
        carrier_available = self.results.get_total("availability_import", scenario=scenario,
                                                   year=year).groupby(["carrier", "node"]).mean().loc[carrier]
        carrier_used = self.results.get_total("flow_conversion_input", scenario=scenario,
                                              year=year).groupby(["carrier", "node"]).mean().loc[carrier]
        # Load NUTS2 data and join with carrier_available
        if carrier == 'carbon':
            potential_used = carrier_used
        else:
            potential_used = (carrier_used/carrier_available) * 100
        potential_used.name = carrier
        gdf_merged = self.merge_nutsdata(potential_used, nuts=nuts)
        # Plot as a map of Europe:
        self.plot_europe_map(gdf_merged, column=potential_used.name,
                             legend_label=f'Available {carrier.capitalize()} used in percent',
                             year=year, nuts=nuts, title=False)

    def plot_demand(self, carrier, years, scenario=None, nuts=2):
        """plot carrier demand"""
        if not isinstance(years,list):
            years = list(years)
        demand = self.results.get_total("demand", scenario=scenario)[years]
        demand = demand.loc[carrier].groupby(["node"]).sum()/1000  # in TWh
        # print(f'Total hydrogen demand in Europe in the year {2020 + year*2} is {demand.sum()} TWh')
        # Load geographical data and join with demand
        gdf_merged = self.merge_nutsdata(demand, nuts=nuts)

        # Plot as a map of Europe
        self.plot_europe_map(gdf_merged, columns=years, cmap=f"{carrier}_demand", legend_label=f'{carrier.capitalize()} demand [TWh]',
                             years=years, nuts=nuts)

    def plot_capacity(self, tech, years, scenario=None, nuts=2):
        """plot carrier demand"""
        if not isinstance(years,list):
            years = list(years)
        capacity = self.results.get_total("capacity", scenario=scenario)[years]
        techs = [t for t in capacity.index.unique("technology") if tech in t]
        capacity = capacity.loc[techs].groupby(["location"]).sum()/1000  # in TWh
        capacity.index.names = ["node"]
        # print(f'Total hydrogen demand in Europe in the year {2020 + year*2} is {demand.sum()} TWh')
        # Load geographical data and join with demand
        gdf_merged = self.merge_nutsdata(capacity, nuts=nuts)

        # Plot as a map of Europe
        self.plot_europe_map(gdf_merged, columns=years, cmap=f"{tech}", legend_label=f'{tech.capitalize()} capacity [GW]',
                             years=years, nuts=nuts, name="capacity")

    def plot_tech_jobs(self, years, techs, scenario=None, calc_method='mean', nuts=2, rel=False, max_val=0):
        """load data from Excel file and plot total jobs for specific year, tech"""
        if not isinstance(years,list):
            year = list(years)

        jobs = list()
        for y in years:
            j = self.get_total_jobs(scenario, techs, year=y, per_technology=True, per_capacity=False, calc_method=calc_method)
            j.rename(columns={techs: y}, inplace=True)
            if rel:
                j = self.calc_rel(j, column=y, nuts=nuts)
                unit = "[FTE] per 100'000 capita"
            else:
                unit = '[FTE]'
            jobs.append(j)
        gdf_merged = self.merge_nutsdata(pd.concat(jobs, axis=1), nuts=nuts)
        # plot as a map with limited axes
        self.plot_europe_map(dataframe=gdf_merged, columns=years, cmap=techs, legend_label=f'Jobs, {techs.capitalize()} {unit}',
                             years=years, nuts=nuts, calc_method=calc_method)

        # compute percentage of non-zero and non-null values
        # non_zero_non_null_pct = ((check != 0) & (~check.isnull())).sum() / len(check) * 100
        # print(f"{techs.capitalize()} produces jobs in {non_zero_non_null_pct:.2f}% of regions")

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
        self.plot_europe_map(dataframe=gdf_merged, column=export.name,
                             legend_label=f'Production - Demand, {carrier.capitalize()} [TWh]',
                             year=year, nuts=nuts, diverging=diverging, title=title)

    def plot_tech_change(self, tech, carrier='hydrogen', scenario=None, calc_method='mean', nuts=2,
                         time=0, title=False):
        """load data from Excel and plot temporal change for specific tech for NUTS0/2 regions"""

        # Read in job ratio for specified tech from the Excel
        job_ratio = self.get_job_ratio(tech=tech, calc_method=calc_method)

        # Get capacity and output flow from HSC modeling results
        capacity = self.results.get_total("capacity", scenario=scenario, element_name=tech)
        output_flow = self.results.get_total("flow_conversion_output", scenario=scenario, element_name=tech)
        check = output_flow.loc[carrier].groupby(["node"]).sum()

        # Get the desired form of the data frame to plot and check with output flow for active capacities
        total_jobs = capacity.droplevel(0) * job_ratio
        total_jobs[check == 0] = 0

        # If gasification or SMR additionally add the CCS jobs:
        if tech == 'SMR' or tech == 'gasification':
            #todo check computations
            total_jobs +=  self.results.get_total("capacity", scenario=scenario, element_name=tech+"_CCS") * job_ratio

        # Summation over NUTS0 label for country
        if nuts == 0:
            total_jobs = self.combine_to_nuts0(total_jobs)
        # Hand-adjusted parameters:
        if tech == 'electrolysis':
            time = 14
        if tech == 'gasification':
            time = 14
        if tech == 'SMR':
            time = 14

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
        self.plot_time_scale(dataframe=total_jobs, color_dict=color_dict, carrier=carrier, title=title)

    def compute_jobs_per_tech(self, scenario, per_capacity=False, per_technology=True, calc_method="mean", year=None,):
        """ compute jobs per tech"""
        jobs = self.get_total_jobs(scenario, [], per_capacity, per_technology, calc_method, year)
        file_path = os.path.join(self.folder_figures, f"jobs_per_region_{calc_method}.csv")
        jobs.join(self.population, how='outer').to_csv(file_path, index=True)
        # add country index
        jobs["country"] = [idx[:2] for idx in jobs.index.get_level_values("node")]
        jobs = jobs.set_index("country",append=True)
        jobs["electrolysis"].groupby(["country", "year"]).sum().unstack("country").plot()
        plt.show()
        a=1

    def get_total_jobs(self, scenario, techs = list(), per_capacity=False, per_technology=True, calc_method="mean", year=None):
        """ for a given sceanrio, get total jobs for each technology based on capacity or output flow"""
        if not isinstance(techs,list):
            techs = [techs]
        if len(techs) == 0:
            techs = self.df_job_ratios["tech_code"].unique()
        # get raw data
        round = 2
        base = (self.results.get_total("capacity", scenario=scenario).stack()).round(round)
        base.index.names = base.index.names[:-1] + ["year"]
        base = base.groupby(["year", "location", "technology"]).sum()
        base.index.names = ["year", "node", "technology"]
        if not per_capacity:
            output = (self.results.get_total("flow_conversion_output", scenario=scenario).stack() / 8760).round(round)
            output.index.names = output.index.names[:-1] + ["year"]
            output = output.groupby(["technology","carrier", "year", "node"]).sum()
            reference_carriers=self.results._to_df(self.results.results[scenario][None]["sets"]["set_reference_carriers"]["dataframe"])
            for tech in output.index.unique("technology"):
                c = [c for c in output.loc[tech].index.unique("carrier") if c != reference_carriers.loc[tech].value]
                output.loc[tech, c] = 0
            output = output.groupby(["year", "node","technology"]).sum().round(4)
            transport = (self.results.get_total("flow_transport", scenario=scenario).stack() / 8760).round(round)
            transport.index.names = transport.index.names[:-1] + ["year"]
            transport = transport.groupby(["year", "edge", "technology"]).sum()
            transport.index.names = ["year", "node", "technology"]
            flows = pd.concat([output,transport]) #.round(3)
            base = flows
            base = base.loc[flows.index][flows>0]
        # compute total jobs per tech
        # base = base.to_frame()
        # base["country"] = [idx[:2] for idx in base.index.get_level_values("node")]
        # base = base.set_index("country",append=True)
        base = base.unstack("technology").fillna(0)
        techs = [tech for tech in techs if tech in base.columns]
        base = base[techs]
        for tech in techs:
            job_ratio = self.get_job_ratio(tech, calc_method=calc_method)
            base[tech] *= job_ratio  # total_number of jobs
        if not year is None:
            base = base.loc[year]
        if not per_technology:
            base = base.sum(axis=1)
        return base

    def plot_total_change(self, techs, carrier='hydrogen', scenario=None, calc_method='mean',
                          nuts=0, time=None, title=False):
        """Total jobs for every country for specified techs"""
        total_jobs = self.get_total_jobs(scenario, techs, per_capacity=False, per_technology=False, calc_method=calc_method) #TODO add calc method
        total_jobs = total_jobs.unstack("year")
        # Summation over NUTS0 label for country
        if nuts == 0:
            total_jobs = self.combine_to_nuts0(total_jobs) #Check funciton!!
        if time == None:
            time = total_jobs.columns.max()

        # Produce color_dict for 'Leaders' and 'Followers'
        jobs_at_time = total_jobs.iloc[:, time].sort_values(ascending=False)
        color_dict = {}
        i = 0
        for j, color in zip([3,8], ["purple","blue"]): # indicate which areas to fill
            color_dict.update({region: self.eth_colors.get_color(color) for region in list(jobs_at_time.index)[i:j]})
            i = j
        # Plot with timescale
        self.plot_time_scale(dataframe=total_jobs, color_dict=color_dict, carrier=carrier, title=title, total_jobs=True, calc_method=calc_method)

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
        ax.set_ylabel(f'Jobs [FTE]', fontsize=30)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xticks(np.linspace(0, 15, 16, dtype=int), np.linspace(2020, 2050, 16, dtype=int), fontsize=24)
        plt.yticks(fontsize=24)
        plt.xlabel('Time', fontsize=30)
        if nuts == 0:
            plt.legend(labels=techs, prop={'size': 40}, frameon=False)

        # Plot title and save plot:
        plt.title(f'{country.upper()}', fontsize=40)
        plot_name = f'{self.folder_figures}/{country}_{calc_method}_{nuts}.svg'
        plt.savefig(plot_name, format='svg', dpi=600, transparent=True, bbox_inches='tight',)
        # plt.show()

    def plot_hydrogen_jobs(self, techs, year=0, scenario=None, calc_method='mean', nuts=2, rel=False, plot='all'):
        """loop over all hydrogen producing technologies sum and plot total"""

        jobs = self.get_total_jobs(scenario, techs, per_capacity=False, calc_method=calc_method)

        # Create a DataFrame from the list of tuples
        if plot == 'pie' or plot == 'all':
            pie_data = jobs.groupby("year").sum().loc[year]
            pie_data.index.name = "Technology"
            pie_data.name = "Total Jobs"
            self.plot_pie_chart(pie_data.reset_index(), year=year, calc_method=calc_method)

        # Compute total jobs and merge with NUTS2 data:
        total_jobs = jobs.groupby("node").sum().sum(axis=1)
        total_jobs.name = 'jobs'
        unit = '[FTE]'

        # Make Lorenz Plot
        if plot == 'lorenz' or plot == 'all':
            self.lorenz_plot(dataframe=total_jobs, column='jobs', year=year, calc_method=calc_method)
        # Make GINI Plot
        # if plot == 'gini' or plot == 'all':
        #     self.gini_plot(dataframe=total_jobs, column='jobs', year=year, calc_method=calc_method)

        # Calculate jobs per 100'000 capita
        if rel:
            total_jobs = self.calc_rel(total_jobs, column=total_jobs.name, nuts=nuts)
            unit = "[FTE] per 100'000 capita"

        # Merge with NUTS2 data:
        jobs = self.merge_nutsdata(total_jobs, nuts=nuts)
        jobs.name = 'Jobs'

        if plot == 'bubble':
            self.bubble_plot(dataframe=jobs, column="jobs", calc_method=calc_method)

        # Make Europe plot
        if plot == 'europe':
            self.plot_europe_map(dataframe=jobs, columns='jobs', cmap='jobs',
                                 legend_label=f'{jobs.name.capitalize()} in Hydrogen, {unit}',
                                 years=year, nuts=nuts, title=False)

        # Compute percentage of non-zero and non-null values
        non_zero_non_null_pct = ((jobs['jobs'] != 0) & (~jobs['jobs'].isnull())).sum() / len(jobs) * 100
        print(f"Hydrogen produces jobs in {non_zero_non_null_pct:.2f}% of regions")

    def bubble_plot(self, dataframe, column, calc_method):
        """plot jobs, gdp and population per region"""
        filepath = os.path.join("employment_data", "nuts2_gdp_2016.xlsx")
        gdp = pd.read_excel(filepath, index_col=0).rename(columns={"Value": "gdp"})
        dataframe = dataframe.join(gdp["gdp"], how='inner')
        dataframe = dataframe.join(self.population, how='inner')
        dataframe["population"] *= 1e-6  # Million people
        dataframe["gdp"] *= 1e-3  # MEUR to GEUR
        _x = "population"
        _y = "gdp"
        _s = column+"_relative"
        dataframe[_s] = dataframe[column]/dataframe[column].max()*100
        dataframe.sort_values(by=_s, inplace=True, ascending=False)
        # create plot
        for q in [0.75,0.9]:
            fig, ax = plt.subplots()
            lb = -0.1
            dataframe[dataframe[column]<1][column] = 0
            dataframe[dataframe[_s]<5][_s]=5
            # add regions without jobs to plot
            idx = dataframe[dataframe[column] == 0].index
            dataframe.loc[idx, _s] = 5
            limit = dataframe[dataframe[column]>1][column].quantile(q)
            max = dataframe[column].max()
            for c, ub in zip(["grey", "petrol", "purple"], [0, limit, max]):
                tmp = dataframe[(dataframe[column]>lb) & (dataframe[column]<=ub)]
                lb = ub
                x = tmp[_x].values.astype(float)
                y = tmp[_y].values.astype(float)
                s = tmp[_s].values.astype(float)
                ax.scatter(x, y, s=s, color=self.eth_colors.get_color(c), alpha=0.6)
            ax.set_xlim(0,14)
            ax.set_ylim(0,800)
            ax.set_ylabel("GDP [GEUR]")
            ax.set_xlabel("Million people")
            # add labels for most jobs
            # regions = []
            # for i in [0,1,2]:
            #     x = dataframe.iloc[i][_x]
            #     y = dataframe.iloc[i][_y]#-dataframe.iloc[i][_y]*0.5
            #     ax.text(x, y, dataframe.iloc[i].name, va="bottom", ha="center",
            #             bbox={'fc': 'w', 'alpha': 0.5, 'pad': 0, "ec":"none"})
            #     ax.scatter(x, y, s=dataframe.iloc[i][_s], c="none", ec="k")
            #     regions.append(dataframe.iloc[i].name)
            # add labels for most jobs
            # dataframe.sort_values(by="population", inplace=True, ascending=False)
            # for i in [0,1]:
            #     if dataframe.iloc[i].name not in regions:
            #         x = dataframe.iloc[i][_x]
            #         y = dataframe.iloc[i][_y]#-dataframe.iloc[i][column]*0.5
            #         ax.text(x, y, dataframe.iloc[i].name, va="bottom", ha="center",
            #                 bbox={'fc': 'w', 'alpha': 0.5, 'pad': 0, "ec":"none"})
            #         ax.scatter(x, dataframe.iloc[i][_s], s=dataframe.iloc[i][column], c="none", ec="k")
            plt.savefig(os.path.join(self.folder_figures,f"bubble_plot_quantile_{q}_{calc_method}.pdf"), transparent=True, bbox_inches='tight',)
            plt.savefig(os.path.join(self.folder_figures, f"bubble_plot_quantile_{q}_{calc_method}.pdf"), transparent=True, bbox_inches='tight',)
            # plt.show()


    def plot_jobs_transport(self, tech, year, scenario=None, calc_method='mean', nuts=2):

        # Get job ratio from results Excel
        job_ratio = self.get_job_ratio(tech, calc_method=calc_method)

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
        naturalgas_use =\
            input_flow.loc['SMR'].groupby(['node']).sum() + input_flow.loc['SMR_CCS'].groupby(['node']).sum()

        # Share of biomethane in the total gas consumption for SMR/SMRCCS
        biomethane_share = (biomethane_use / naturalgas_use) * 100

        biomethane_share.name = 'biomethane_share'

        gdf_merged = self.merge_nutsdata(biomethane_share, nuts=nuts)

        self.plot_europe_map(dataframe=gdf_merged, column=biomethane_share.name,
                             legend_label=f'Biomethane share in percent', year=year,
                             nuts=nuts, title=False, diverging=True)

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

        renewable_tot = np.zeros_like(self.results.get_total("flow_conversion_output", scenario=scenario, year=year,
                                                             element_name='pv_ground').groupby(['node']).sum())
        # Loop over renewable techs and sum up output
        for tech in techs:
            renewable_prod = self.results.get_total("flow_conversion_output", scenario=scenario,
                                                    year=year, element_name=tech).groupby(['node']).sum()/1000
            renewable_prod.name = tech
            gdf_merged = self.merge_nutsdata(dataframe=renewable_prod, nuts=nuts)

            self.plot_europe_map(dataframe=gdf_merged, column=tech, legend_label=f'{tech.capitalize()} [TWh]',
                                 year=year, nuts=nuts)
            renewable_tot += renewable_prod
        renewable_tot.name = 'renew_tot'
        gdf_merged = self.merge_nutsdata(renewable_tot, nuts=nuts)

        self.plot_europe_map(dataframe=gdf_merged, column=renewable_tot.name, legend_label=f'Renewable production [TWh]',
                             year=year, nuts=nuts, title=False)

    def plot_europe_map(self, dataframe, columns, cmap, legend_label=None, years=0, nuts=2, diverging=False, name="", calc_method=""):
        """plots map of europe with correct x and y-limits, axes names
        :param dataframe: europe joined with data
        :param column: data to plot
        :param legend_label: Label of legend
        :param year: year
        :param nuts: region size
        :param diverging: for diverging cmap
        :param title: bool title
        :param max_val: maximum value for color-bar
        :param arrows: dataframe for arrow plotting"""
        # check for typeconsistency
        if not isinstance(years, list):
            years = [years]
        if not isinstance(columns, list):
            columns = [columns]

        # Get the colormap for plotting
        colormap = self.eth_colors.get_custom_colormaps(self.colormap_dict[cmap], diverging=diverging)
        grey = self.eth_colors.get_color('grey', 40)

        # Plot figure, axis and the underlying data in a Europe map
        if len(years)<3:
            width = 2.75
            height = 3
        else:
            width = 1.85
            height = 2
        fig, axs = plt.subplots(1, len(years), figsize=(width*len(years), height))

        # plot settings single and multi-year
        min_val = 0
        x_lower_limit = 0.23 * 10 ** 7
        x_upper_limit = 0.6 * 10 ** 7
        y_lower_limit = 1.3 * 10 ** 6
        y_upper_limit = 5.6 * 10 ** 6

        if not isinstance(axs, np.ndarray):
            axs = [axs]
            max_val = dataframe[columns].max()
        else:
            max_val = dataframe[columns].max().max()

        for y, a, c in zip(years, axs, columns):
            # Quick fix for balkan and NUTS0
            gdf_quickfix = self.gdf.copy(deep=True)
            gdf_quickfix[c] = pd.NA
            conditions = (gdf_quickfix['LEVL_CODE'] == 2) & (
                gdf_quickfix.index.str.startswith(('AL', 'ME', 'RS', 'MK')))
            gdf_quickfix.loc[conditions, c] = -50000.00000
            selected_rows = gdf_quickfix.loc[conditions].dropna(subset=[c])
            selected_rows[c] = pd.to_numeric(selected_rows[c])
            dataframe = pd.concat([dataframe, selected_rows])
            dataframe = dataframe.dropna()
            colormap.set_under(color=grey)
            # Quick fix to make nan values grey
            dataframe.loc[(dataframe['LEVL_CODE'] == nuts) & ~(dataframe.index.str.startswith(('TR', 'IS'))),c] = dataframe.loc[(dataframe['LEVL_CODE'] == nuts) & ~(dataframe.index.str.startswith(('TR', 'IS'))), c].fillna(-50000)
            dataframe.plot(ax=a, column=c, cmap=colormap, legend=False, edgecolor='black', linewidth=0, vmin=min_val, vmax=max_val)
            a.set_xlim([x_lower_limit, x_upper_limit])
            a.set_ylim([y_lower_limit, y_upper_limit])
            a.set_axis_off()
            if len(years)>1:
                a.set_title(str(2022 + 2 * y))
            # Outline countries additionally
            if nuts == 2:
                nuts0 = gdf_quickfix[(gdf_quickfix['LEVL_CODE'] == 0)]
                nuts0 = nuts0[~nuts0.index.str.startswith(('TR', 'IS'))]
                nuts0.boundary.plot(edgecolor=self.eth_colors.get_color("grey","dark"), linewidth=.2, ax=a)

        # Modify colormap and add frame and details to plot
        # norm = plt.Normalize(vmin=min_val, vmax=max_val)
        # cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=colormap, norm=norm), ax=ax)
        # cbar.set_label(legend_label, size=50, labelpad=14)
        # cbar.ax.tick_params(labelsize=40, length=8, width=4, pad=20)

        divider = make_axes_locatable(axs[-1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(axs[-1].collections[0], cax=cax, label=legend_label)
        cbar.outline.set_linewidth(.4)
        for l in cbar.ax.yaxis.get_major_ticks():
            l._width = 0.3
            # l.set_fontsize(16)

        # Save and display plot
        plot_name = f'{self.folder_figures}/europe_{nuts}_{name}_{cmap}_{years[0]}_{years[-1]}_{calc_method}'
        fig.savefig(f'{plot_name}.svg', format='svg', dpi=200, transparent=True, bbox_inches='tight',)
        fig.savefig(f'{plot_name}.pdf', format='pdf', dpi=200, transparent=True, bbox_inches='tight',)
        # plt.show()

    def plot_time_scale(self, dataframe, color_dict, carrier, title=False, total_jobs=False, calc_method=""):
        """plots dataframe over time, with colordict"""

        pivoted_data = dataframe.rename(columns={year: year*2+2022 for year in np.arange(0, 15, 1)}).transpose()
        color = self.eth_colors.get_color('grey')
        # Fix NL and ES --> why this drop in AD capacity??
        # if calc_method=="mean":
        #     for r, y in zip(["NL","ES"], [2042,2044]):
        #         pivoted_data.loc[y+2,r] = pivoted_data.loc[y,r]
        #         pivoted_data.loc[y+4,r] = pivoted_data.loc[y,r] + 0.5*(pivoted_data.loc[y+6,r]-pivoted_data.loc[y,r])
        # Create plot with temporal change
        ax = pivoted_data.plot(legend=False, color=color) #figsize=(30, 20),  linewidth=3,
        # Manual values for labelling the interesting lines in total jobs plot
        if total_jobs:
            x_start = pivoted_data.index.max()
            x_end = x_start+0.5
            PAD = 0.1
            # Plot interesting lines above
            for idx, region in enumerate(color_dict.keys()):
                if idx < 3:
                    pivoted_data[region].plot(color=color_dict[region], linewidth=1.2, zorder=3.5, ax=ax)
                else:
                    pivoted_data[region].plot(color=color_dict[region], linewidth=1.2, zorder=2.5, ax=ax)
                # Vertical start of line
                year = pivoted_data.index.max()
                y_start = pivoted_data[region].loc[year]
                y_end = pivoted_data.loc[x_start,region]
                # Add line
                ax.plot([x_start, (x_start + x_end - PAD) / 2, x_end - PAD], [y_start, y_end, y_end],
                        color=color_dict[region], ls='dashed')
                # Add text
                ax.text(x_end, y_end, region, color='black', weight='bold', va='center')
            # Add shade between lines
            # ax.fill_between(pivoted_data.index, pivoted_data['IT'], pivoted_data['DE'], interpolate=True,
            #                 color=color_dict['DE'], alpha=0.2)
            # ax.fill_between(pivoted_data.index, pivoted_data['IT'], pivoted_data['ES'],
            #                 where=pivoted_data['IT'] > pivoted_data['ES'], interpolate=True, color=color_dict['DE'],
            #                 alpha=0.2)
            # ax.fill_between(pivoted_data.index, pivoted_data['DE'], pivoted_data['ES'],
            #                 where=pivoted_data['ES'] > pivoted_data['DE'], interpolate=True, color=color_dict['DE'],
            #                 alpha=0.2)
            # ax.fill_between(pivoted_data.index, pivoted_data['PL'], pivoted_data['LT'],
            #                 where=pivoted_data['PL'] > pivoted_data['LT'], interpolate=True, color=color_dict['NL'],
            #                 alpha=0.2)
            # ax.fill_between(pivoted_data.index, pivoted_data['PL'], pivoted_data['NL'],
            #                 where=pivoted_data['NL'] > pivoted_data['PL'], interpolate=True, color=color_dict['NL'],
            #                 alpha=0.2)
        # Plot details
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xlabel("Time")
        plt.ylabel("Jobs [FTE]")
        max_jobs = pivoted_data.max().max()
        if len(str(int(max_jobs)))>3:
            digits = -3
        else:
            digits = -len(str(int(max_jobs)))-1
        scale = 10 ** digits
        max_jobs = np.ceil(max_jobs*scale)/scale
        ax.set_ylim(0,max_jobs.round(-digits))
        ax.set_xlim(2022,2050)
        if title:
            plt.title(f"total jobs in {carrier}")
        #ax.set_xticks(np.linspace(2022, 2050, 15, dtype=int))
        plot_name = f'{self.folder_figures}/time_total_{carrier}_{calc_method}'
        plt.savefig(f'{plot_name}.svg', transparent=True, bbox_inches='tight',)
        plt.savefig(f'{plot_name}.pdf', transparent=True, bbox_inches='tight',)
        # plt.show()

    def plot_europe_maps(self, dataframe, column, legend_label=None, year=0, nuts=2, diverging=False,
                         title=True, max_val=0, calc_method=""):
        """plots maps of europe with correct x and y-limits, axes names
        :param dataframe: europe joined with data
        :param column: data to plot
        :param legend_label: Label of legend
        :param year: year
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
        plot_name = f'{self.folder_figures}/europecombo_{nuts}_{column}_{calc_method}'
        fig.savefig(f'{plot_name}.svg', format='svg', dpi=600, transparent=True, bbox_inches='tight',)
        fig.savefig(f'{plot_name}.pdf', format='pdf', dpi=200, transparent=True, bbox_inches='tight',)
        # plt.show()

    def plot_pie_chart(self, dataframe, year, calc_method=""):
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
            numbers.append(round(float(row[1]), 2))  # Get the total number for each tech and convert to 2 fig integer
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

                ax.text(np.cos(np.deg2rad(180))*0.5, np.sin(np.deg2rad(180))*0.5,
                        '{:g}'.format(float('{:.2g}'.format(numbers[i]))),
                        ha='center', va='center', fontsize=50, fontdict=dict(weight='bold'))
            else:
                ax.text(text_x, text_y, '{:g}'.format(float('{:.2g}'.format(numbers[i]))), ha='center', va='center',
                        fontsize=50, rotation_mode='default', rotation=angle, fontdict=dict(weight='bold'))
        plot_name = f'{self.folder_figures}/pie_chart_{year}_{calc_method}'
        fig.savefig(f'{plot_name}.pdf', format='pdf', dpi=200, transparent=True, bbox_inches='tight',)
        fig.savefig(f'{plot_name}.svg', format='svg', dpi=200, transparent=True, bbox_inches='tight',)
        # plt.show()

    def lorenz_plot(self, dataframe, column, year, title=False, calc_method=""):
        """Lorenz curve plot of dataframe"""
        # Read in population number from csv sheet, source: EUROSTAT2018
        population = self.population
        dataframe = dataframe.fillna(0) #check computation of jobs

        # Combine dataframe to NUTS0 and NUTS2
        population0 = self.combine_to_nuts0(population)
        dataframe0 = pd.DataFrame(self.combine_to_nuts0(dataframe))
        dataframe2 = pd.DataFrame(dataframe).fillna(0)

        # Merge dataframes with population
        df_merged0 = dataframe0.merge(population0, left_index=True, right_index=True, how='left')
        df_merged2 = dataframe2.merge(population, left_index=True, right_index=True, how='left')

        # sort by column
        df_merged2.sort_values(by=column, inplace=True)
        df_merged0.sort_values(by=column, inplace=True)
        cumulative2 = np.cumsum(df_merged2[column]) / df_merged2[column].sum()
        cumulative0 = np.cumsum(df_merged0[column]) / df_merged0[column].sum()

        # Get the colors for each Lorenz curve
        color2 = self.eth_colors.get_color('green')
        color0 = self.eth_colors.get_color('blue')
        grey = self.eth_colors.get_color('grey', 60)

        # Create Lorenz curve
        fig, ax = plt.subplots(figsize=(20, 20))
        range0 = np.arange(0, len(cumulative0.index)) / (len(cumulative0.index) - 1)
        range2 = np.arange(0, len(cumulative2.index)) / (len(cumulative2.index) - 1)
        ax.plot(range2, cumulative2, color=color2, linewidth=5)
        ax.plot(range0, cumulative0, color=color0, linewidth=5)
        plt.text(0.62, 0.4, 'Countries', size=40, rotation=64, color=color0,
                 ha="center", va="center", bbox=dict(ec='1', fc='1'))
        plt.text(0.87, 0.4, 'Regions', size=40, rotation=75, color=color2,
                 ha="center", va="center", bbox=dict(ec='1', fc='1'))
        ax.plot([0, 1], [0, 1], color=grey, linestyle='--', linewidth=5)
        plt.xlabel('Share of regions', fontsize=40, fontdict=dict(weight='bold'))
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
        plt.savefig(f'{self.folder_figures}/lorenz_{year}_{calc_method}.pdf', format='pdf', dpi=200, transparent=True, bbox_inches='tight',)
        plt.savefig(f'{self.folder_figures}/lorenz_{year}_{calc_method}.svg', format='svg', dpi=200, transparent=True, bbox_inches='tight',)
        # plt.show()

    def gini_plot(self, dataframe, column, year, calc_method=""):
        """plot gini coefficient for all NUTS0 regions"""
        # Get population data
        dataframe = pd.DataFrame(dataframe) ##check computation of jobs --> why so many nan??
        population = self.population
        df_merged = dataframe.merge(population, left_index=True, right_index=True, how='left')

        # Get initials for all NUTS0 regions
        unique_initials = set([label[:2] for label in dataframe.index])

        # Calculate the GINI coefficient for every country
        gini_df = pd.DataFrame(columns=['Gini Coefficient'])
        for country in unique_initials:
            country_data = df_merged[df_merged.index.str.startswith(country)].dropna()
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
            color = cmap(gini_coefficient) #norm?!
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
        plt.savefig(f'{self.folder_figures}/gini_{year}_{calc_method}.pdf', format='pdf', dpi=200, transparent=True, bbox_inches='tight',)
        plt.savefig(f'{self.folder_figures}/gini_{year}_{calc_method}.svg', format='svg', dpi=200, transparent=True, bbox_inches='tight',)
        # plt.show()


if __name__ == "__main__":
    datasets = ["HSC_NUTS2_electrolysis"]
    scens = ["med", "low", "high",] # "min", "max"
    cms = ["mean"] #, "max","min",
    techs = ['electrolysis', ] #'SMR', 'gasification', 'anaerobic_digestion','SMR_CCS', 'gasification_CCS'
    years = [0, 4, 14]

    for dataset in datasets:
        vis = Visualization(dataset, [f"scenario_{h}" for h in scens])
        for scenario in [f"scenario_{h}" for h in scens]:
            vis.set_folder_figures(scenario)
            for cm in cms:

                ## Data
                vis.compute_jobs_per_tech(scenario, per_technology=True, calc_method=cm)

                ## PLOTS
                # job distribution over time for all techs considered
                vis.plot_total_change(techs=techs, scenario=scenario, time=14, calc_method=cm)
                #  the distribution of jobs for specified year as lorenz curve and gini distribution
                vis.plot_hydrogen_jobs(techs=techs, scenario=scenario, year=14, calc_method=cm)

                # total jobs in specified year as Europe map, can also be done with jobs/capita
                # vis.plot_hydrogen_jobs(techs=techs, scenario=scenario, year=14, plot='europe', nuts=2, rel=False, calc_method=cm)

                # jobs for technologies in specified year, can also be done with jobs/capita
                vis.plot_tech_jobs(techs='electrolysis', scenario=scenario, years=[4,14], rel=False, calc_method=cm)  # 2022-2050
                vis.plot_tech_jobs(techs='electrolysis', scenario=scenario, years=[14], rel=False, calc_method=cm) # 2050
                vis.plot_tech_jobs(techs='SMR', scenario=scenario, years=[0], rel=False, calc_method=cm) # 2022

                # plot res capacity
                vis.plot_capacity(tech='pv', scenario=scenario, years=[4, 14])  # 2022-2050
                vis.plot_capacity(tech='wind', scenario=scenario, years=[4, 14])  # 2022-2050

                # Plot carrier demand in TWh as Europe map
                vis.plot_demand(carrier='hydrogen', scenario=scenario, years=[0,4,14], nuts=2) # 2022-2050

            ## additional plots
            additional_plots = False
            if additional_plots:

                # total jobs, gdp and population per region
                vis.plot_hydrogen_jobs(techs=techs, scenario=scenario, year=14, calc_method=cm, plot="bubble")

                # Plot the production - demand
                vis.plot_export_potential(scenario=scenario, year=14, nuts=2)

                # Plot biomethane shares
                vis.plot_biomethane_shares(years=years, scenario=scenario)

                # Plot output flow for carrier
                vis.plot_output_flows(carrier='hydrogen', scenario=scenario, year=14, nuts=2)


                # Plot change of singular hydrogen producing tech over time
                vis.plot_tech_change(tech='SMR', scenario=scenario, nuts=0)

                # Plot change of all techs over time for country
                vis.country_jobs_change(country='DE', techs=techs, scenario=scenario)

                # Plot jobs in transport for tech as Europe map
                vis.plot_jobs_transport(tech='hydrogen_truck_liquid', scenario=scenario, year=14, nuts=2)

                # Plot renewable electricity production for specific year as Europe map
                vis.plot_renewables(year=14, scenario=scenario)

                # Plot relative feedstock used (wet & dry biomass, electricity, biomethane, natural gas) as Europe map
                vis.plot_conversion_input(carrier='dry_biomass', scenario=scenario, year=14, nuts=2)
