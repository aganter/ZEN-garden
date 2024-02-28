"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      January-2024
Authors:      Anya Xie (anyxie@ethz.ch)
Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:  Analysis of results of exog vs. endog model
==========================================================================================================================================================================="""

import os
import re
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from zen_garden._internal import main
from zen_garden.postprocess.results import Results

import matplotlib.pyplot as plt

# fixtures
##########

@pytest.fixture
def config():
    """
    :return: A new instance of the config
    """
    from config import config
    config.solver["keep_files"] = False
    return deepcopy(config)

@pytest.fixture
def folder_path():
    """
    :return: Returns the path of the testcase folder
    """
    return os.path.dirname(__file__)


# helper functions
##################

def str2tuple(string):
    """
    Extracts the values of a string tuple
    :param string: The string
    :return: A list of indices
    """
    indices = []
    for s in string.split(","):
        # string are between single quotes
        if "'" in s:
            indices.append(re.search("'([^']+)'", s).group(1))
        # if it is not a sting it is a int
        else:
            indices.append(int(re.search("\d+", s)[0]))
    return indices


def compare_variables(test_model, optimization_setup,folder_path):
    """ assertion test: compare model variables to desired values
    :param test_model: The model to test (name of the data set)
    :param optimization_setup: optimization setup with model of tested model
    :param folder_path: The path to the folder containing the file with the correct variables
    """
    # skip for models with scenario analysis
    if optimization_setup.system["conduct_scenario_analysis"]:
        return
    # import csv file containing selected variable values of test model collection
    test_variables = pd.read_csv(os.path.join(folder_path, 'test_variables_readable.csv'),header=0, index_col=None)
    # dictionary to store variable names, indices, values and test values of variables which don't match the test values
    failed_variables = defaultdict(dict)
    # iterate through dataframe rows
    for _,data_row in test_variables[test_variables["test"] == test_model].iterrows():
        # get variable attribute of optimization_setup object by using string of the variable's name (e.g. optimization_setup.model.variables["importCarrierFLow"])
        variable_attribute = optimization_setup.model.solution[data_row["variable_name"]]

        # extract the values
        index = str2tuple(data_row["index"])
        variable_value = variable_attribute.loc[*index].item()

        if not np.isclose(variable_value, data_row["value"], rtol=1e-3):
            failed_variables[data_row["variable_name"]][data_row["index"]] = {"computed_value": variable_value,
                                                          "test_value": data_row["value"]}
    assertion_string = str()
    for failed_var in failed_variables:
        assertion_string += f"\n{failed_var}{failed_variables[failed_var]}"

    assert len(failed_variables) == 0, f"The variables {assertion_string} don't match their test values"


def compare_variables_results(test_model: str, results: Results, folder_path: str):
    """
    Compares the variables of a Results object from the test run to precomputed values
    :param test_model: The model to test (name of the data set)
    :param results: The Results object
    :param folder_path: The path to the folder containing the file with the correct variables
    """
    # import csv file containing selected variable values of test model collection
    test_variables = pd.read_csv(os.path.join(folder_path, 'test_variables_readable.csv'),header=0, index_col=None)
    # dictionary to store variable names, indices, values and test values of variables which don't match the test values
    failed_variables = defaultdict(dict)
    # iterate through dataframe rows
    for _,data_row in test_variables[test_variables["test"] == test_model].iterrows():
        # get the corresponding data frame from the results
        if not results.has_scenarios:
            variable_df = results.get_df(data_row["variable_name"])
            added_str = ""
        else:
            variable_df = results.get_df(data_row["variable_name"],scenario=data_row["scenario"])
            added_str = f" ({data_row['scenario']})"
        # iterate through indices of current variable
        for variable_index, variable_value in variable_df.items():
            # ensure equality of dataRow index and variable index
            if str(variable_index) == data_row["index"]:
                # check if close
                if not np.isclose(variable_value, data_row["value"], rtol=1e-3):
                    failed_variables[data_row["variable_name"]+added_str][data_row["index"]] = {"computed_values": variable_value,
                                                                  "test_value": data_row["value"]}
    # create the string of all failed variables
    assertion_string = ""
    for failed_var, failed_value in failed_variables.items():
        assertion_string += f"\n{failed_var}: {failed_value}"

    assert len(failed_variables) == 0, f"The variables {assertion_string} don't match their test values"


def check_get_total_get_full_ts(results: Results, specific_scenario=False, year=None, element_name=None, discount_to_first_step=True, get_doc=False):
    """
    Tests the functionality of the Results methods get_total() and get_full_ts()

    :param get_doc:
    :param discount_to_first_step: Apply annuity to first year of interval or entire interval
    :param element_name: Specific element
    :param year: Specific year
    :param specific_scenario: Specific scenario
    :param results: Results instance of testcase function has been called from
    """
    test_variables = ["demand", "capacity", "storage_level", "capacity_limit"]
    scenario = None
    if specific_scenario:
        scenario = results.scenarios[0]
    for test_variable in test_variables:
        df_total = results.get_total(test_variable, scenario=scenario, year=year)
        if test_variable != "capacity_limit":
            df_full_ts = results.get_full_ts(test_variable, scenario=scenario, year=year, discount_to_first_step=discount_to_first_step)
        if element_name is not None:
            df_total = results.get_total(test_variable, element_name=df_total.index[0][0])
            if test_variable != "capacity_limit":
                df_full_ts = results.get_full_ts(test_variable, element_name=df_full_ts.index[0][0])
    if get_doc:
        results.get_doc(test_variables[0])

def empty_plot_with_text(text, background_color='white'):
    fig = plt.figure(facecolor=background_color)
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=16)
    plt.show()

# PRE-CHECK: Random variabes i like to look at
def pre_check(res, scenario=None):
    res.get_df("capacity_addition", scenario=scenario).groupby(['technology', 'capacity_type', 'year']).sum().reset_index()
    res.get_df("total_cost_pwa_cum_capacity_segment_position", scenario=scenario).groupby(['set_technologies', 'set_capacity_types', 'set_time_steps_yearly']).sum().reset_index()
    res.get_df("capacity_limit", scenario=scenario).groupby(['technology', 'capacity_type', 'year']).sum().reset_index()
    res.get_df("global_cumulative_capacity", scenario=scenario).groupby(['technology', 'capacity_type']).sum().reset_index()


# CHECK 1:  Compare it with the total cost obtained
# Calculate the PWA result of the total cost function
def check1(res, scenario=None):
    calc_total_cost = (res.get_df("total_cost_pwa_segment_selection", scenario=scenario)*res.get_df("total_cost_pwa_intersect", scenario=scenario) + res.get_df("total_cost_pwa_segment_selection", scenario=scenario)*res.get_df("total_cost_pwa_slope", scenario=scenario)).groupby(level=[0,1,3]).sum()
    diff_total_cost = (res.get_df("total_cost_pwa_global_cost", scenario=scenario) - calc_total_cost.rename_axis(index={'set_time_steps_yearly': 'year'})).sum()
    print(f"The difference between the variable total cost and the calculated total cost is {round(diff_total_cost,4)}.")



# CHECK 2: See if results on total cost function
# for each technology and each capacity type
def check2(res, scenario=None, save_fig=False, file_type=None):
    for tech in res.get_df("capacity", scenario=scenario).index.get_level_values("technology").unique():
        for capacity_type in res.get_df("capacity", scenario=scenario).loc[tech].index.get_level_values("capacity_type").unique():
            # get the global share factor
            gsf = res.get_df("global_share_factor", scenario=scenario).loc[tech]

            interpolated_q = res.get_df("total_cost_pwa_points_lower_bound", scenario=scenario).loc[tech, capacity_type, :].values
            interpolated_q = np.append(interpolated_q, res.get_df("total_cost_pwa_points_upper_bound", scenario=scenario).loc[tech, capacity_type, :].values[-1])

            interpolated_TC = res.get_df("total_cost_pwa_TC_lower_bound", scenario=scenario).loc[tech, capacity_type, :].values
            interpolated_TC = np.append(interpolated_TC, res.get_df("total_cost_pwa_TC_upper_bound", scenario=scenario).loc[tech, capacity_type, :].values[-1])


            res_capacity = (1/gsf)*res.get_df("capacity", scenario=scenario).groupby(level=[0,1,3]).sum().loc[tech, capacity_type, :].values
            cum_capacity = res.get_df("global_cumulative_capacity", scenario=scenario).loc[tech, capacity_type, :].values
            res_TC = res.get_df("total_cost_pwa_global_cost", scenario=scenario).loc[tech, capacity_type, :]

            initial_capacity = (1/gsf)*res.get_df("global_initial_capacity", scenario=scenario).loc[tech]

            # plot the total cost function
            plt.plot(interpolated_q, interpolated_TC, label=f'PWA: {tech}', color='red')
            plt.scatter(interpolated_q, interpolated_TC, color='red')
            plt.scatter(res_capacity, res_TC, label=f'Model Results {tech}', color='blue')
            plt.scatter(cum_capacity, res_TC, label=f'Cumulative Capacity {tech}', color='orange')
            plt.scatter(initial_capacity, res.get_df("total_cost_pwa_initial_global_cost", scenario=scenario).loc[tech], label=f'Initial Capacity {tech}', color='green')
            plt.legend()
            if save_fig:
                path = os.path.join(os.getcwd(), "outputs")
                path = os.path.join(path, os.path.basename(res.results[scenario]["analysis"]["dataset"]))
                path = os.path.join(path, "result_plots")
                if not os.path.exists(path):
                    os.makedirs(path)
                plt.savefig(os.path.join(path, tech + "_" + capacity_type + "_" + "." + file_type))
            plt.show()

    print("Plot of Total Cost calculations on the curve.")


# CHECK 3:  Check if cost capex equal to differenc between total cost in each step
# for each technology and each capacity type
def check3(res, scenario=None):
    for tech in res.get_df("capacity", scenario=scenario).index.get_level_values("technology").unique():
        for capacity_type in res.get_df("capacity", scenario=scenario).loc[tech].index.get_level_values("capacity_type").unique():
            calc_cost_capex = []
            for year in res.get_df("capacity", scenario=scenario).index.get_level_values("year").unique():
                if year==0:
                    calc_cost_capex.append(res.get_df("total_cost_pwa_global_cost", scenario=scenario).loc[tech, :, year] - res.get_df("total_cost_pwa_initial_global_cost", scenario=scenario).loc[tech])
                else:
                    calc_cost_capex.append(res.get_df("total_cost_pwa_global_cost", scenario=scenario).loc[tech, :,year] - res.get_df("total_cost_pwa_global_cost", scenario=scenario).loc[tech,:,year-1])

            diff_calc_cost_capex = (pd.concat(calc_cost_capex).values - res.get_df("cost_capex", scenario=scenario).loc[tech, capacity_type].values).sum().round(4)
            print(f"The difference between the calculated cost capex and the variable cost capex for"
                  f" {tech}-{capacity_type} is {diff_calc_cost_capex}.")


# CHECK 4: Check if capacity addition
def check4(res, scenario=None):
    for tech in res.get_df("capacity", scenario=scenario).index.get_level_values("technology").unique():
        for capacity_type in res.get_df("capacity", scenario=scenario).loc[tech].index.get_level_values("capacity_type").unique():
            calc_capacity_addition = []
            for year in res.get_df("capacity", scenario=scenario).index.get_level_values("year").unique():
                if year==0:
                    calc_capacity_addition.append(1/res.get_df("global_share_factor", scenario=scenario).loc[tech]*(res.get_df("total_cost_pwa_global_cost", scenario=scenario).loc[tech, :, year] - res.get_df("capacity_existing", scenario=scenario).loc[tech, capacity_type].sum()))
                else:
                    calc_capacity_addition.append(1/res.get_df("global_share_factor", scenario=scenario).loc[tech]*(res.get_df("global_cumulative_capacity", scenario=scenario).loc[tech, :,year] - res.get_df("global_cumulative_capacity", scenario=scenario).loc[tech,:,year-1]))

            diff_calc_capacity_addition = pd.concat(calc_capacity_addition).values - res.get_df("capacity_addition", scenario=scenario).loc[tech, capacity_type].groupby(level=[1]).sum().values.round(4)
            print(f"The difference between the calculated capacity addition and the variable capacity addition for"
                  f" {tech}-{capacity_type} is {diff_calc_capacity_addition}.")



# CHECK 5: Check if capacity meets demand
def check5(res, scenario=None):
    for carrier in res.get_df("demand", scenario=scenario).index.get_level_values("carrier").unique():
        for year in res.get_df("demand", scenario=scenario).index.get_level_values("time_operation").unique(): # will be an issue when intra-year demand
            calc_cum_capacity = 1/res.get_df("global_share_factor", scenario=scenario)[0]*res.get_df("global_cumulative_capacity", scenario=scenario).loc[:, :, year].sum() # simplified: assuming all techs supply this one carrier and global share equal for both = 1
            diff_demand = (res.get_df("demand", scenario=scenario).loc[carrier, :, year].sum() - calc_cum_capacity).round(4)

            print(f"The difference between the demand and installed capacity for year {year} and carrier {carrier} is {diff_demand}.")



# CHECK 7: Check if any technology reaches its capacity limit
def check7(res, endogenous=True, scenario=None):
    cap_installed= res.get_df("capacity").groupby(["technology","capacity_type", "location"]).tail(1).groupby(["technology","capacity_type"]).sum()
    global_cap_installed = res.get_df("global_cumulative_capacity").groupby(level=[0,1]).tail(1)
    global_cap_installed.index = global_cap_installed.index.droplevel(-1)
    if endogenous:
        cap_limit = res.get_df("total_cost_pwa_points_upper_bound",scenario=scenario).groupby(level=[0,1]).tail(1)
    else:
        cap_limit = res.get_df("capacity_limit").groupby(["technology","capacity_type"]).sum()
    cap_limit.index = cap_limit.index.droplevel(-1)
    for tech in cap_installed.index.get_level_values("technology").unique():
        for capacity_type in cap_installed.index.get_level_values("capacity_type").unique():
            diff = (global_cap_installed.loc[tech, capacity_type] - cap_limit.loc[tech, capacity_type]).round(4)
            if diff >= 0:
                print(f"The difference between the global cum. capacity and the capacity limit for"
                  f" {tech}-{capacity_type} is {diff}.")



# PLOT1: STANDARD PLOTS: Plot all standard plots for all scenarios
def standard_plots_AX(res, save_fig=False, file_type=None):
    # Create Standard Plots for all scenarios
    for scenario in res.scenarios:
        empty_plot_with_text(scenario, background_color='lightyellow')
        demand = res.get_df("demand", scenario=scenario)
        # remove carriers without demand
        demand = demand.loc[(demand != 0), :]
        for carrier in demand.index.levels[0].values:
            if carrier in demand:
                res.plot("capacity", yearly=True, tech_type="conversion", reference_carrier=carrier,
                         plot_strings={
                             "title": f"Capacities of {carrier.capitalize()} Generating Conversion Technologies",
                             "ylabel": "Capacity"}, save_fig=save_fig, file_type=file_type, scenario=scenario)
                res.plot("capacity_addition", yearly=True, tech_type="conversion", reference_carrier=carrier,
                         plot_strings={
                             "title": f"Capacity Addition of {carrier.capitalize()} Generating Conversion Technologies",
                             "ylabel": "Capacity"}, save_fig=save_fig, file_type=file_type, scenario=scenario)
                res.plot("flow_conversion_input", yearly=True, reference_carrier=carrier, plot_strings={
                    "title": f"Input Flows of {carrier.capitalize()} Generating Conversion Technologies",
                    "ylabel": "Input Flow"}, save_fig=save_fig, file_type=file_type, scenario=scenario)
                res.plot("flow_conversion_output", yearly=True, reference_carrier=carrier, plot_strings={
                    "title": f"Output Flows of {carrier.capitalize()} Generating Conversion Technologies",
                    "ylabel": "Output Flow"}, save_fig=save_fig, file_type=file_type, scenario=scenario)

        res.plot("capex_yearly", yearly=True, plot_strings={"title": "Total Capex", "ylabel": "Capex"},
                 save_fig=save_fig, file_type=file_type, scenario=scenario)
        res.plot("cost_carrier", yearly=True, plot_strings={"title": "Carrier Cost", "ylabel": "Cost"},
                 save_fig=save_fig, file_type=file_type, scenario=scenario)
        res.plot("carbon_emissions_annual", yearly=True, plot_strings={"title": "Annual Carbon Emissions", "ylabel": "Emissions"},
                 save_fig=save_fig, file_type=file_type, scenario=scenario)

# PLOT2: LEARNING CURVES: Plot all learning curves
def learning_curve_plots(res, tech_carrier, scenario=None):
    def fun_total_cost(u, c_initial: float, q_initial: float,
                       learning_rate: float) -> object:  # u is a vector
        """
        Total cumulative Cost for Learning Curve
        :param u: Cumulative Capacity
        :param c_initial: Initial Cost
        :param q_initial: Initital Capacity
        :param learning_rate: Learning Rate
        :return: Total cumulative cot
        """
        alpha = c_initial / np.power(q_initial, learning_rate)
        exp = 1 + learning_rate
        TC = alpha / exp * ( np.power(u, exp) )

        return TC

    def unit_cost(u, c_initial: float, q_initial: float, learning_rate: float) -> object: # u is a vector
        """
        Exponential regression for Learning Curve
        Input: Cumulative Capacity u
        Parameters: c_initial, q_initial, learning_rate
        Output: Unit cost of technology
            :rtype:
            """
        alpha = c_initial / np.power(q_initial, learning_rate)
        v = alpha * np.power(u, learning_rate)
        return v

    for tech in tech_carrier:
        for capacity_type in res.get_df("capacity", scenario=scenario).loc[tech].index.get_level_values("capacity_type").unique():

            # Plot interpolation points
            capacity_values = res.get_df("total_cost_pwa_points_lower_bound", scenario=scenario).loc[tech, capacity_type, :].values
            capacity_values = np.append(capacity_values, res.get_df("total_cost_pwa_points_upper_bound", scenario=scenario).loc[tech, capacity_type, :].values[-1])
            total_cost_values = res.get_df("total_cost_pwa_TC_lower_bound", scenario=scenario).loc[tech, capacity_type, :].values
            total_cost_values = np.append(total_cost_values, res.get_df("total_cost_pwa_TC_upper_bound", scenario=scenario).loc[tech, capacity_type, :].values[-1])

            plt.plot(capacity_values, total_cost_values, label=f'{tech}-{capacity_type}')
            plt.legend()
            plt.title('Total cost curve linearly approximated for all technologies')
            plt.xlabel('Capacity')
            plt.ylabel('Total Cost')

    plt.show()

    for tech in tech_carrier:
        for capacity_type in res.get_df("capacity", scenario=scenario).loc[tech].index.get_level_values("capacity_type").unique():
            # get lower bound of x values
            lb = res.get_df("total_cost_pwa_points_lower_bound", scenario=scenario).loc[tech, capacity_type, :].values[0]
            # get upper bound of x values
            ub = res.get_df("total_cost_pwa_points_upper_bound", scenario=scenario).loc[tech, capacity_type, :].values[-1]
            # get parameters of each tech and capacity type
            capacity_values = np.linspace(lb, ub, 1000)

            # get initial cost
            c_initial = res.get_df("total_cost_pwa_initial_unit_cost", scenario=scenario).loc[tech, capacity_type]
            # get initial capacity
            q_initial = res.get_df("global_initial_capacity", scenario=scenario).loc[tech]
            # get learning rate
            learning_rate = res.get_df("learning_rate", scenario=scenario).loc[tech]

            unit_cost_values = unit_cost(capacity_values, c_initial, q_initial, learning_rate)

            plt.plot(capacity_values, unit_cost_values, label=f'{tech}-{capacity_type}')
            plt.legend()
            plt.xlabel('Capacity')
            plt.ylabel('Unit Cost')
            plt.title('Unit cost curve for all technologies')
            # plt.ylim(0, 2000)
            plt.xlim(0,30)


    plt.show()


# PLOT 3: GLOBAL CAPACITY: Plot of global cum capacity compared to actual active capacity
def compare_global_capacity_plots(res, scenario=None):
    for tech in res.get_df("capacity", scenario=scenario).index.get_level_values("technology").unique():
        for capacity_type in res.get_df("capacity", scenario=scenario).loc[tech].index.get_level_values("capacity_type").unique():
            gsf = res.get_df("global_share_factor", scenario=scenario).loc[tech]
            active_capacity = (1/gsf)*res.get_df("capacity", scenario=scenario).loc[tech,capacity_type,:,:].groupby(level=[1]).sum()
            global_capacity = res.get_df("global_cumulative_capacity", scenario=scenario).loc[tech,capacity_type,:]

            plt.plot(active_capacity.values, label=f'Active capacity {tech}-{capacity_type}')
            plt.plot(global_capacity.values, label=f'Global cumulative capacity {tech}-{capacity_type}')
            plt.xlabel('Year')
            plt.ylabel('Capacity')
            plt.legend()
            plt.show()


# PLOT 4: SANKEY DIAGRAM: Plot of Sankey Diagram
import plotly.graph_objects as go
import plotly.express as px
def generate_sankey_diagram(scenario, target_technologies, intermediate_technologies, year, title):

    path = os.path.join(os.getcwd(), "outputs")
    path = os.path.join(path, os.path.basename(res.results[scenario]["analysis"]["dataset"]))
    path = os.path.join(path, "result_plots")

    file_path_input = os.path.join(path, f"flow_conversion_input_grouped_{scenario}.csv")
    inputs_df = pd.read_csv(file_path_input)
    file_path_output = os.path.join(path, f"flow_conversion_output_grouped_{scenario}.csv")
    outputs_df = pd.read_csv(file_path_output)

    input_techs_target = inputs_df[inputs_df['technology'].isin(target_technologies)]
    input_techs_intermediate = inputs_df[inputs_df['technology'].isin(intermediate_technologies)]
    output_techs_intermediate = outputs_df[outputs_df['technology'].isin(intermediate_technologies)]
    output_techs_target = outputs_df[outputs_df['technology'].isin(target_technologies)]

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

    links['source'] = links['source'].map(mapping_dict)
    links['target'] = links['target'].map(mapping_dict)

    links_dict = links.to_dict(orient="list")

    color_palette = px.colors.qualitative.Dark24

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color='black', width=0.5),
            label=unique_source_target,
            color=color_palette
        ),
        link=dict(
            source= links_dict['source'],
            target= links_dict['target'],
            value= links_dict['value'],
            label=[f"{source} to {target}" for source, target in zip(links_dict['source'], links_dict['target'])]
        )
    )]
    )
    if isinstance(year, str):
        year = int(year)
    displayed_year = year * 2 + 2024
    fig.update_layout(title_text=f"{title} {displayed_year}", font_size=18)

    fig.write_html(os.path.join(path, f"sankey_diagram_{year}.html"), auto_open=True)

def save_total(res, scenario=None):

    path = os.path.join(os.getcwd(), "outputs")
    path = os.path.join(path, os.path.basename(res.results[scenario]["analysis"]["dataset"]))
    path = os.path.join(path, "result_plots")

    os.makedirs(path, exist_ok=True)

    flow_conversion_input = res.get_total("flow_conversion_input", scenario=scenario)
    file_path = os.path.join(path, f"flow_conversion_input_{scenario}.csv")
    flow_conversion_input.to_csv(file_path)

    flow_conversion_output = res.get_total("flow_conversion_output", scenario=scenario)
    file_path = os.path.join(path, f"flow_conversion_output_{scenario}.csv")
    flow_conversion_output.to_csv(file_path)

    file_path = os.path.join(path, f"flow_conversion_input_{scenario}.csv")
    total_input = pd.read_csv(file_path)
    total_input = total_input.drop('node', axis=1)
    total_input = total_input.groupby(['technology', 'carrier']).sum()
    file_path = os.path.join(path, f"flow_conversion_input_grouped_{scenario}.csv")
    total_input.to_csv(file_path)

    file_path = os.path.join(path, f"flow_conversion_output_{scenario}.csv")
    total_output = pd.read_csv(file_path)
    total_output = total_output.drop('node', axis=1)
    total_output = total_output.groupby(['technology', 'carrier']).sum()
    file_path = os.path.join(path, f"flow_conversion_output_grouped_{scenario}.csv")
    total_output.to_csv(file_path)

# PLOT 5: COST CURVE: Plot of cost curve
def plot_average_unit_cost(res, carrier, scenario=None):
    data_total_cap_add = res.get_total("capacity_addition", scenario=scenario)
    data_extract_cap_add = res.extract_reference_carrier(data_total_cap_add, carrier, scenario)
    data_extract_cap_add = data_extract_cap_add.groupby(["technology", "capacity_type"]).sum().T

    data_total_cap = res.get_total("global_cumulative_capacity", scenario=scenario)
    data_extract_cap = res.extract_reference_carrier(data_total_cap, carrier, scenario)
    data_extract_cap = data_extract_cap.groupby(["technology", "capacity_type"]).sum().T

    data_total_cost = res.get_total("cost_capex", scenario=scenario)
    data_extract_cost = res.extract_reference_carrier(data_total_cost, carrier, scenario)
    data_extract_cost = data_extract_cost.groupby(["technology", "capacity_type"]).sum().T

    # Plot the unit cost over the years
    unit_cost_of_investment_T = data_extract_cost / data_extract_cap_add
    unit_cost_of_investment_T = unit_cost_of_investment_T.interpolate(limit_direction='both')
    plt.figure(figsize=(8, 6))
    plt.plot(unit_cost_of_investment_T, marker='.')
    plt.legend(unit_cost_of_investment_T.columns, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("Year")
    plt.ylabel("Unit Cost of Technology")
    plt.title("Cost Evolution of Technologies over the years")
    plt.xticks(range(len(unit_cost_of_investment_T.index)))
    plt.tight_layout()
    plt.show()

# PLOT 6: UNIT COST OVER TIME
def plot_unit_cost_over_time(res, carrier,scenario=None):

    def unit_cost(u, c_initial: float, q_initial: float, learning_rate: float) -> object: # u is a vector
        """
        Exponential regression for Learning Curve
        Input: Cumulative Capacity u
        Parameters: c_initial, q_initial, learning_rate
        Output: Unit cost of technology
            :rtype:
            """
        alpha = c_initial / np.power(q_initial, learning_rate)
        v = alpha * np.power(u, learning_rate)
        return v

    data_total = res.get_total("capacity", scenario=scenario)
    data_extract = res.extract_reference_carrier(data_total, carrier, scenario)
    data_extract = data_extract.groupby(["technology"]).mean().T
    tech_carrier = data_extract.columns

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    for tech in tech_carrier:
        for capacity_type in res.get_df("capacity", scenario=scenario).loc[tech].index.get_level_values(
                "capacity_type").unique():
            # get initial cost
            c_initial = res.get_df("total_cost_pwa_initial_unit_cost", scenario=scenario).loc[tech, capacity_type]
            # get initial capacity
            q_initial = res.get_df("global_initial_capacity", scenario=scenario).loc[tech]
            # get learning rate
            learning_rate = res.get_df("learning_rate", scenario=scenario).loc[tech]
            # PWA range
            pwa_lower_bound = res.get_df("total_cost_pwa_points_lower_bound", scenario=scenario).loc[tech, capacity_type, :].values[0]
            pwa_upper_bound = res.get_df("total_cost_pwa_points_upper_bound", scenario=scenario).loc[tech, capacity_type, :].values[-1]
            pwa_range = np.linspace(pwa_lower_bound, pwa_upper_bound, 1000)

            data_total_cap = res.get_total("global_cumulative_capacity", scenario=scenario)
            data_extract_cap = res.extract_reference_carrier(data_total_cap, carrier, scenario)
            data_extract_cap = data_extract_cap.groupby(["technology", "capacity_type"]).sum().T

            cap_tech = data_extract_cap.T.loc[tech, capacity_type]
            point_on_curve = unit_cost(cap_tech, c_initial, q_initial, learning_rate)

            learning_curve = unit_cost(pwa_range, c_initial, q_initial, learning_rate)

            ax1.plot(point_on_curve, marker='.', label=f'{tech}-{capacity_type}')
            ax1.legend()
            ax1.set_xlabel('Years')
            ax1.set_ylabel('Unit Cost')
            ax1.set_title('Cost Evolution of Technologies over the years')
            # plt.ylim(0, 2000)
            # plt.xlim(0, 60)

            ax2.scatter(cap_tech, point_on_curve, marker='.')
            ax2.plot(pwa_range, learning_curve, label=f'{tech}-{capacity_type}')
            ax2.legend()
            ax2.set_xlabel('Cum. Global Capacity')
            ax2.set_ylabel('Unit Cost')
            ax2.set_title('Cost Evolution of Technologies over cumulative global capacity')

    plt.show()






############################################## Result anaylsis ##############################################

# I. Read the results of the two models
folder_path = os.path.dirname(__file__)
data_set_name = "20240209_Hydrogen_endog"

res = Results(os.path.join(folder_path, data_set_name))

scenario = None
save_fig = True
file_type = "png"

############################################## PLOTS ##############################################

# Create custom standard_plots for all scenarios
standard_plots_AX(res, save_fig=save_fig, file_type=file_type)

# Do indivudal checks for the scenarios
check2(res, scenario, save_fig=save_fig, file_type=file_type)

# Create individual plots for scenarios
compare_global_capacity_plots(res, scenario=scenario)


# Plot the unit cost of technologies in each optimization year
# Maybe show it on the learning curve
demand = res.get_df("demand", scenario=scenario)
# remove carriers without demand
demand = demand.loc[(demand != 0), :]
for carrier in demand.index.levels[0].values:
    if carrier in demand:
        plot_average_unit_cost(res, carrier, scenario=scenario)

# Unit cost graph
plot_unit_cost_over_time(res, "hydrogen")

save_total(res, scenario=scenario)

import plotly.io as pio
target_technologies = ["electrolysis", "SMR", "SMR_CCS", "gasification", "gasification_CCS"]
intermediate_technologies = ["pv_ground", "biomethane_conversion", "anaerobic_digestion", "wind_onshore", "wind_offshore"]
year = "13"
title = data_set_name
generate_sankey_diagram(scenario, target_technologies, intermediate_technologies, year, title)



df_extact = res.extract_reference_carrier(res.get_df("capacity_addition", scenario=scenario), "hydrogen", scenario).groupby(["technology"]).sum()
plt.pie(df_extact, labels=df_extact.index, autopct='%1.1f%%')
plt.title("Shares of Hydrogen Generating Technologies in Capacity Additions over full horizon")
plt.show()


df_full = res.get_df("flow_conversion_input", scenario=scenario)
flow_biomethane = df_full.groupby(["carrier"]).sum().loc["biomethane"]
df_extract = res.extract_reference_carrier(df_full, "hydrogen", scenario).groupby(["carrier"]).sum()
df_extract["natural_gas"]= df_extract["natural_gas"] - flow_biomethane
df_extract=pd.concat([df_extract,pd.Series({"biomethane": flow_biomethane})])
plt.pie(df_extract, labels=df_extract.index, autopct='%1.1f%%')
plt.title("Shares of Hydrogen Generating Carriers over full horizon")
plt.show()

res.get_df("flow_conversion_input").groupby(["technology"]).sum()

plt.plot(res.get_total("flow_conversion_input").groupby(["carrier"]).sum().T)
plt.legend(res.get_total("flow_conversion_input").groupby(["carrier"]).sum().T.columns)
plt.show()


plt.plot(res.get_total("flow_conversion_output").groupby(["carrier"]).sum().T)
plt.legend(res.get_total("flow_conversion_output").groupby(["carrier"]).sum().T.columns)
plt.show()


# Need to minues the biomethane emissions
emissions_carrier = res.get_df("carbon_emissions_carrier").groupby(["carrier"]).sum()
plt.pie(emissions_carrier, labels=emissions_carrier.index, autopct='%1.1f%%')
plt.title("Shares of Emissions of Hydrogen Generating Carriers over full horizon")
plt.show()

# Need to minues the biomethane emissions
emissions_technology = res.get_df("carbon_emissions_technology").groupby(["technology"]).sum()
plt.pie(emissions_technology, labels=emissions_technology.index, autopct='%1.1f%%')
plt.title("Shares of Emissions of Hydrogen Generating Carriers over full horizon")
plt.show()

# Exogenous


############################################## EXPECTED OUTCOME ##############################################

# Look at demand of each carrier across all nodes
res.get_total("demand", scenario=scenario).groupby(["carrier"]).sum()

# EXOG Price
# Look at average price of conversion technologies that create same output carrier
capacity = res.get_df("capacity", scenario=scenario)
demand = res.get_df("demand", scenario=scenario)
# remove carriers without demand
demand = demand.loc[(demand != 0), :]
for carrier in demand.index.levels[0].values:
    if carrier in demand:
        data_total = res.get_total("capex_specific_conversion", scenario=scenario)
        data_extract = res.extract_reference_carrier(data_total, carrier, scenario)
        df_capex_specific = data_extract.groupby(["technology"]).mean().T
        plt.plot(df_capex_specific)
        plt.title(f"Capex specific of {carrier.capitalize()} Generating Conversion Technologies")
        plt.xlabel("Year")
        plt.ylabel("Capex")
        plt.legend(df_capex_specific.columns, loc='center right')
        plt.xticks(range(len(df_capex_specific.index)))
        plt.show()


# Look at the average price of imports of the carriers
res.get_total("price_import").groupby(["carrier"]).mean()
plt.plot(res.get_total("price_import").groupby(["carrier"]).mean().T)
plt.legend(res.get_total("price_import").groupby(["carrier"]).mean().T.columns, loc='center left')
plt.show()
#
# # Look at existing capacities
# res.get_df("capacity_existing", scenario=scenario).groupby(["technology","capacity_type"]).sum()
#
# ENDOG Learning Curve
demand = res.get_df("demand", scenario=scenario)
# remove carriers without demand
demand = demand.loc[(demand != 0), :]
for carrier in demand.index.levels[0].values:
    if carrier in demand:
        data_total = res.get_total("capacity", scenario=scenario)
        data_extract = res.extract_reference_carrier(data_total, carrier, scenario)
        data_extract = data_extract.groupby(["technology"]).mean().T
        tech_carrier = data_extract.columns
        learning_curve_plots(res, tech_carrier=tech_carrier, scenario=scenario)

plt.plot(res.get_total("price_import").groupby(["carrier"]).mean().T)
plt.legend(res.get_total("price_import").groupby(["carrier"]).mean().T.columns, loc='center left')
plt.show()

res.get_total("carbon_emissions_technology").groupby(["technology"]).sum()

plt.plot(res.get_total("carbon_intensity_carrier").groupby(["carrier"]).mean().T)
plt.legend(res.get_total("carbon_intensity_carrier").groupby(["carrier"]).mean().T.columns, loc='center left')
plt.show()

plt.plot(res.get_total("capacity").groupby(["technology"]).sum().loc[["carbon_pipeline", "dry_biomass_truck", "hydrogen_pipeline"]].T)
plt.legend(res.get_total("capacity").groupby(["technology"]).sum().loc[["carbon_pipeline", "dry_biomass_truck", "hydrogen_pipeline"]].T.columns, loc='center left')
plt.xlabel("Year")
plt.ylabel("Capacity")
plt.title("Capacity of Transport Technologies")
plt.show()
############################################## ACTUAL OUTCOME ##############################################

# 2. Look at conversion output flows of all technologies
res.get_total("flow_conversion_output", scenario=scenario).groupby(["carrier"]).sum()

# 3. Look at exports of carriers
res.get_total("flow_export", scenario=scenario).groupby(["carrier"]).sum()

#4. Look at emissions of technologies
res.get_df("carbon_emissions_annual", scenario=scenario)
#
# # Look at capacities of all technologies
res.get_df("capacity", scenario=scenario).groupby(["technology"]).sum()
#
# # Biomethane post-processing
res.get_df("flow_conversion_input", scenario=scenario).groupby(["carrier"]).sum()
res.get_df("flow_import", scenario=scenario).groupby(["carrier"]).sum()
# Check if the numbers add up
res.get_total("flow_import", scenario=scenario).groupby(["carrier"]).sum().loc["natural_gas"]
# If we substract the biomethane from the natural gas flow conversion input then we should get the "real" natural gas
# The "real" natural gas should correspond to the flow imports of natural gas
res.get_total("flow_conversion_input", scenario=scenario).groupby(["carrier"]).sum().loc["natural_gas"] - res.get_total("flow_conversion_output").groupby(["carrier"]).sum().loc["natural_gas"]


# Net Present Cost
res.get_df("net_present_cost").sum()
# Check Emissions
res.get_df("net_present_cost").sum()
res.get_df("carbon_emissions_annual").sum()
res.get_df("carbon_emissions_cumulative")

print("Done with result analysis")