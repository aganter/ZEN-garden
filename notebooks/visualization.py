import os
from zen_garden.postprocess.results import Results
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd


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

    def plot_output_flows(self, carrier, scenario=None, techs=None):
        """plot output flows for selected carriers"""

        output_flow = self.results.get_total("output_flow", scenario=scenario, year=0)

        if techs == "conversion":
            _techs = self.system["set_hydrogen_conversion_technologies"]
            output_flow = output_flow.loc[_techs]
        else:
            print(Warning, "Only conversion techs implemented")
        output_flow = output_flow.groupby(["node"]).sum()  # in GWh
        # idx_names = output_flow.index.names        # find out what this means

        #path_to_data = os.path.join("outputs", "NUTS_RG_20M_2016_3035")
        #gdf = geopandas.read_file(path_to_data)
        #gdf.plot()
        # Load NUTS2 data and join with output_flow
        path_to_data = os.path.join("outputs", "NUTS_RG_01M_2016_3035", "NUTS_RG_01M_2016_3035.shp")
        gdf = gpd.read_file(path_to_data)
        gdf = gdf[gdf["LEVL_CODE"] == 2]
        gdf = gdf[["NUTS_ID", "geometry"]]
        right = pd.Series(output_flow, name="hydrogen_output")
        gdf_merged = gdf.merge(right, left_on="NUTS_ID", right_index=True, how="inner")
        gdf_merged = gdf_merged[gdf_merged['hydrogen_output'].notna()]
        # plot as a map:
        gdf_merged.plot(column='hydrogen_output', legend=True, legend_kwds={"label": "Hydrogen Output in GWh", "orientation":"horizontal"})
        plt.show()

    def plot_carrier_flow(self, technology=None, scenario=None):
        """plot carrier flows for selected transport technologies"""

        carrier_flow = self.results.get_total("carrier_flow", scenario=scenario)
        carrier_flow = carrier_flow.groupby(["technology"]).sum().loc[technology]

    def plot_demand_carrier(self, carrier, scenario=None):
        """plot carrier demand"""

        demand_carrier = self.results.get_df("demand_carrier", scenario=scenario)
        demand_carrier = demand_carrier.groupby(["carrier"]).sum() # in GWh

    def plot_built_capacity(self, technology=None, scenario=None):
        """plot newly built capacity for selected technology"""
        built_capacity = self.results.get_total("built_capacity", scenario=scenario)
        built_capacity = built_capacity.sum()



if __name__ == "__main__":
    results_path = os.path.join("outputs", "HSC_NUTS2")
    scenario = "scenario_reference_bau"  # to load more than one scenario, pass a list
    vis = Visualization(results_path, scenario)

    vis.plot_output_flows("hydrogen", scenario=scenario, techs="conversion")
    vis.plot_carrier_flow("hydrogen_pipeline", scenario=scenario)
    vis.plot_demand_carrier("hydrogen", scenario=scenario)
    vis.plot_built_capacity("SMR", scenario=scenario)
