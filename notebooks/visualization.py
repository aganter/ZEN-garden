import os
import matplotlib
import numpy as np
import pandas as pd
import geopandas as gp
import matplotlib.pyplot as plt
from eth_colors import ETHColors
from zen_garden.postprocess.results import Results

class Visualization:

    def __init__(self, dataset, figures_folder=None, results_folder=None):
        """initialize visualization class
        :param dataset: name of the dataset
        :param figures_folder: folder to save figures
        :param results_folder: folder where results are located"""
        self.dataset = dataset
        self.load_results(dataset, results_folder, figures_folder)
        self.set_system()
        self.set_figures_folder(figures_folder)
        self.set_gdf()
        self.colors = ETHColors()

    def load_results(self, dataset, results_folder, figures_folder):
        """load results"""
        if results_folder is None:
            results_folder = os.path.join("..", "data", "outputs")
        self.results = Results(os.path.join(results_folder, dataset))

    def set_system(self):
        """set system"""
        self.system = self.results.get_system()

    def set_figures_folder(self, figures_folder):
        """set figures folder"""
        if figures_folder is None:
            figures_folder = os.path.join(f"figures_{self.dataset}")
        self.create_directory(figures_folder)
        self.figures_folder = figures_folder
        print(f"figures will be saved in {self.figures_folder}")

    def create_directory(self, folderpath):
        """create directory for results"""
        if not os.path.exists(folderpath):
            os.makedirs(folderpath)

    def get_reference_year(self, scenario):
        """get reference year"""
        return self.system[scenario]["reference_year"]

    def set_gdf(self):
        """load geodataframe to plot maps"""
        gdf = gp.read_file("NUTS_RG_20M_2016_4326.shp")
        gdf = gdf.set_index("NUTS_ID")
        gdf.index.name = "node"
        gdf = gdf.to_crs(epsg=3035)
        gdf["x"] = gdf.centroid.x
        gdf["y"] = gdf.centroid.y
        gdf.loc["FR", "x"] = max(gdf.loc["FR"].geometry.geoms, key=lambda a: a.area).centroid.x
        gdf.loc["FR", "y"] = max(gdf.loc["FR"].geometry.geoms, key=lambda a: a.area).centroid.x
        self.gdf = gdf

    def get_gdf(self, level):
        """ get geodataframe for given level"""
        gdf = self.gdf[self.gdf["LEVL_CODE"] == level][["geometry","x","y"]]
        return gdf.copy(deep=True)

    def map_plot_setting(self, axs):
        """plot background of maps"""
        unravel=False
        gdf = self.get_gdf(level=0)
        if not isinstance(axs, np.ndarray):
            unravel = True
            axs = [axs]
        for ax in axs:
            plt.rcParams['hatch.color'] = 'grey'
            plt.rcParams['hatch.linewidth'] = 0.4
            gdf.boundary.plot(ax=ax, color=self.colors.get_color("grey",60), linewidth=.4) # add country boarders
            ax.set_ylim(1.5e6, 5.5e6)
            ax.set_xlim(2.5e6, 5.9e6)
            ax.axis("off")
        if unravel:
            return axs[0]
        else:
            return axs

    def plot_capacity(self, techs, unit, scenario, name="capacity"):
        """stacked barplot of capacity for given reference carrier"""
        # get total returns the annual values of a variable for all technologies in their base units
        # the base units are defined in the dataset --> energy_system --> base_units.csv
        # the capacity depends on the technology reference carrier, i.e.
           # hydrogen is defined in [energy] and thus, the base unit is GW
           # carbon is defined in [mass] and thus, the base unit is kton
        capacity = self.results.get_total("capacity", scenario=scenario)
        # group by technology and sum values
        capacity = capacity.groupby("technology").sum()
        # select the subset of technologies we want to show
        capacity = capacity.loc[techs]
        # colordict
        color_dict =  self.colors.retrieve_colors_dict(techs, category="techs")
        # get a sinel color with self.colors.retrieve_specific_color(techs, category="techs")
        # for carriers use carrier name/ list of carriers and category = "carriers"
        color_dict =  {tech: color for tech, color in color_dict.items() if tech in capacity.index}
        # create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        # plot stacked barplot
        capacity.T.plot.bar(ax=ax, stacked=True, color=color_dict)
        # set labels
        ax.set_ylabel(f"capacity [{unit}]")
        # show plot
        plt.show()
        # save plot
        techs="_".join(techs)
        plt.savefig(os.path.join(self.figures_folder, f"capacity_{techs}_{scenario}.pdf")) # please use vector graphics
        plt.close()

    def plot_map(self, variable, years, scenario, carrier=None):
        """plot map of given variable for given year"""
        # get the values of the variable for the given year
        values = self.results.get_total(variable, scenario=scenario)
        values = values[years]
        values = values.groupby(["node", "carrier"]).sum()
        if carrier:
            values = values.xs(carrier, level="carrier")
        # merge values with geodataframe
        gdf = self.gdf.merge(values, left_index=True, right_index=True)
        # create plot
        fig, axs = plt.subplots(1,len(years))
        # plot map
        cmap = self.colors.get_custom_colormaps("petrol")
        for y, year in enumerate(years):
            real_year = self.get_reference_year(scenario) + year
            gdf.plot(column=year, ax=axs[y], legend=True, cmap=cmap)
            axs[y].set_title(f"Year {real_year}")
        # update plot settings
        axs = self.map_plot_setting(axs)
        # show plot
        plt.show()
        # save plot
        plt.savefig(os.path.join(self.figures_folder, f"{variable}_{carrier}_{scenario}.pdf"))


if __name__ == "__main__":
    dataset= "HSC_imports"
    vis = Visualization(dataset)
    vis.plot_capacity(["SMR", "gasification", "electrolysis", "gasification_CCS", "SMR_CCS"], "GW", scenario="scenario_")
    vis.plot_map("flow_conversion_output", years=[0,13], scenario="scenario_", carrier="hydrogen")

