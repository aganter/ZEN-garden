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

from itertools import cycle

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
def get_learning_curve_total_cost_values(res, tech, scenario=None):
    def unit_cost(u, c_initial: float, q_initial: float, learning_rate: float) -> object:  # u is a vector
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

    tc_pwa_x = res.get_df("total_cost_pwa_points_lower_bound", scenario=scenario).loc[tech, "power",:].values
    tc_pwa_x = np.append(tc_pwa_x, res.get_df("total_cost_pwa_points_upper_bound", scenario=scenario).loc[tech, "power", :].values[-1])
    tc_pwa_y = res.get_df("total_cost_pwa_TC_lower_bound", scenario=scenario).loc[tech, "power", :].values
    tc_pwa_y = np.append(tc_pwa_y, res.get_df("total_cost_pwa_TC_upper_bound", scenario=scenario).loc[tech, "power", :].values[-1])

    # get lower bound of x values
    lb = res.get_df("total_cost_pwa_points_lower_bound", scenario=scenario).loc[tech, "power", :].values[0]
    # get upper bound of x values
    ub = res.get_df("total_cost_pwa_points_upper_bound", scenario=scenario).loc[tech, "power", :].values[-1]
    # get parameters of each tech and capacity type
    capacity_values = np.linspace(lb, ub, 10000)

    # get initial cost
    c_initial = res.get_df("total_cost_pwa_initial_unit_cost", scenario=scenario).loc[tech, "power"]
    # get initial capacity
    q_initial = res.get_df("global_initial_capacity", scenario=scenario).loc[tech]
    # get learning rate
    learning_rate = res.get_df("learning_rate", scenario=scenario).loc[tech]

    unit_cost_values = unit_cost(capacity_values, c_initial, q_initial, learning_rate)

    return tc_pwa_x, tc_pwa_y, capacity_values, unit_cost_values

def plot_tc_curve(res, tech, scenario):

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
        TC = alpha / exp * (np.power(u, exp))

        return TC

    def unit_cost(u, c_initial: float, q_initial: float, learning_rate: float) -> object:  # u is a vector
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

    tc_pwa_x = res.get_df("total_cost_pwa_points_lower_bound", scenario=scenario).loc[tech, "power", :].values
    tc_pwa_x = np.append(tc_pwa_x,res.get_df("total_cost_pwa_points_upper_bound", scenario=scenario).loc[tech, "power",:].values[-1])
    tc_pwa_y = res.get_df("total_cost_pwa_TC_lower_bound", scenario=scenario).loc[tech, "power", :].values
    tc_pwa_y = np.append(tc_pwa_y, res.get_df("total_cost_pwa_TC_upper_bound", scenario=scenario).loc[tech, "power",:].values[-1])

    # get lower bound of x values
    lb = res.get_df("total_cost_pwa_points_lower_bound", scenario=scenario).loc[tech, "power", :].values[0]
    # get upper bound of x values
    ub = res.get_df("total_cost_pwa_points_upper_bound", scenario=scenario).loc[tech, "power", :].values[-1]
    # get parameters of each tech and capacity type
    capacity_values = np.linspace(lb, ub, 10000)

    # get initial cost
    c_initial = res.get_df("total_cost_pwa_initial_unit_cost", scenario=scenario).loc[tech, "power"]
    # get initial capacity
    q_initial = res.get_df("global_initial_capacity", scenario=scenario).loc[tech]
    # get learning rate
    learning_rate = res.get_df("learning_rate", scenario=scenario).loc[tech]

    tc = fun_total_cost(capacity_values, c_initial, q_initial, learning_rate)
    uc = unit_cost(capacity_values, c_initial, q_initial, learning_rate)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(capacity_values, uc, label='Unit cost', color='green')
    ax.scatter(q_initial, c_initial, marker='o')
    ax.text(q_initial, c_initial, r'$\overline{s}, \overline{\alpha}$', fontsize=12, verticalalignment='bottom')
    ax_right = ax.twinx()
    ax_right.plot(capacity_values, tc, label='Total cost', color='blue')
    ax_right.plot(tc_pwa_x, tc_pwa_y, marker='.', label='PWA total cost', color='red', linestyle="--")
    ax_right.set_ylabel('Total cost [€]')
    ax.legend()
    ax_right.legend()
    ax.set_ylim(0, 500)
    ax.set_xlabel('Capacity [GW]')
    ax.set_ylabel('Unit cost [€/kW]')

    # Filling the area
    fill_x = np.concatenate([[0], capacity_values[100]])
    fill_y = np.concatenate([[0], uc[100]])
    ax.fill(fill_x, fill_y, color='gray', alpha=0.2)

    plt.show()





def learning_curve_plots(res, carrier, scenario=None, save_fig=False, file_type=None):






    # for tech in tech_carrier:
    #     for capacity_type in res.get_df("capacity", scenario=scenario).loc[tech].index.get_level_values("capacity_type").unique():
    #
    #         # Plot interpolation points
    #         capacity_values = res.get_df("total_cost_pwa_points_lower_bound", scenario=scenario).loc[tech, capacity_type, :].values
    #         capacity_values = np.append(capacity_values, res.get_df("total_cost_pwa_points_upper_bound", scenario=scenario).loc[tech, capacity_type, :].values[-1])
    #         total_cost_values = res.get_df("total_cost_pwa_TC_lower_bound", scenario=scenario).loc[tech, capacity_type, :].values
    #         total_cost_values = np.append(total_cost_values, res.get_df("total_cost_pwa_TC_upper_bound", scenario=scenario).loc[tech, capacity_type, :].values[-1])
    #
    #         plt.plot(capacity_values, total_cost_values, label=f'{tech}')
    #         plt.legend()
    #         plt.title('Total cost curve linearly approximated for all technologies')
    #         plt.xlabel('Capacity')
    #         plt.ylabel('Total Cost')
    #
    # plt.show()

    fig, ax = plt.subplots()
    for tech in tech_carrier:
        for capacity_type in res.get_df("capacity", scenario=scenario).loc[tech].index.get_level_values("capacity_type").unique():


            ax.plot(capacity_values, unit_cost_values, label=f'{tech}', color=tech_colors[tech])
            ax.plot(q_initial, c_initial, marker='x', color=tech_colors[tech])
            ax.legend()
            ax.set_xlabel('Capacity [GW]')
            ax.set_ylabel('CAPEX [€/kW]')
            ax.set_title('Unit cost curve for all technologies')
            if carrier=="hydrogen":
                ax.set_xlim(0.1, 300)
                ax.set_ylim(0.1, 5000)
            else:
                ax.set_xlim(0, 1000)
                ax.set_ylim(0, 5000)
            ax.set_xscale('log')
            ax.set_yscale('log')

            print(f"Floor value of CAPEX for {tech}: {unit_cost_values[-1]}")


    if save_fig:
        path = os.path.join(os.getcwd(), "outputs")
        path = os.path.join(path, os.path.basename(res.results[scenario]["analysis"]["dataset"]))
        path = os.path.join(path, "result_plots")
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(os.path.join(path, "learning_curve_plot_" + carrier + "." + file_type))

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

# Cost Evolution of technologies
def plot_unit_cost_over_time(res, unit_cost, scenario, fig, ax,  plot_style,save_fig=False, file_type=None):
    if plot_style is not None:
        edgecolor = plot_style["edgecolor"]
        tech_colors = plot_style["tech_colors"]
        tech_hatches = plot_style["tech_hatches"]
        tech_markers = plot_style["tech_markers"]
        bar_width = plot_style["bar_width"]
    def get_change_point(res, scenario, tech):
        initial_capacity = res.get_df("global_initial_capacity", scenario=scenario).loc[tech]
        cum_capacity = res.get_df("global_cumulative_capacity", scenario=scenario).droplevel(1).loc[tech]
        cap_diff = cum_capacity.diff().fillna(cum_capacity.iloc[0] - initial_capacity)
        if cap_diff.iloc[-1] == 0:
            change_point = (cap_diff == 0).idxmax()
        else:
            change_point = len(cap_diff)
        return change_point
    # Plot
    for tech in unit_cost.index.levels[0]:
        if res.results[scenario]["system"]["use_endogenous_learning"]:
            change_point= get_change_point(res, scenario=scenario, tech=tech)
        else:
            change_point=len(unit_cost.loc[tech, "cost"])
        if change_point ==0:
            ax.plot(unit_cost.loc[tech, "cost"].index, unit_cost.loc[tech, "cost"], label=tech, color=tech_colors[tech], linestyle="--")
        else:
            ax.plot(unit_cost.loc[tech, "cost"].iloc[:change_point].index, unit_cost.loc[tech, "cost"].iloc[:change_point], label=tech, color=tech_colors[tech], linestyle="-")
            ax.plot(unit_cost.loc[tech, "cost"].iloc[change_point-1:].index, unit_cost.loc[tech, "cost"].iloc[change_point-1:],  color=tech_colors[tech], linestyle="--")
            ax.scatter(unit_cost.loc[tech, "cost"].iloc[:change_point].index, unit_cost.loc[tech, "cost"].iloc[:change_point],  color=tech_colors[tech],marker='.')
    xtick_labels = [year * 2 + 2024 for year in range(len(unit_cost.columns))]
    ax.set_xlabel("Years")
    ax.set_ylabel("Unit cost [EUR/kW]")
    ax.set_ylim(0, 2500)
    ax.set_xticks(range(len(unit_cost.columns)))
    ax.set_xticklabels(xtick_labels, rotation=90)
    ax.legend(loc="upper right", ncols=3)

    if save_fig:
        path = os.path.join(os.getcwd(), "outputs")
        path = os.path.join(path, os.path.basename(res.results[scenario]["analysis"]["dataset"]))
        path = os.path.join(path, "result_plots")
        if not os.path.exists(path):
            os.makedirs(path)
        try:
            file_path = os.path.join(path, scenario + "cost_over_time_" + carrier + "." + file_type)
        except:
            file_path = os.path.join(path, scenario + "cost_over_time_all_tech" + "." + file_type)
        print("Saving plot to:", file_path)  # Debug print statement
        fig.savefig(file_path)

    fig.show()


def calculate_unit_cost_over_time(res, carrier=None, scenario=None, forecast_scenario=None):

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
    if carrier != None:
        data_extract = res.extract_reference_carrier(data_total, carrier, scenario)
        data_extract = data_extract.groupby(["technology"]).mean().T
        tech_carrier = data_extract.columns
    else:
        tech_carrier = res.get_df("capacity", scenario=scenario).groupby(["technology"]).sum().index

    unit_cost_over_time = {}

    if res.results[scenario]["system"]["use_endogenous_learning"]:
        for tech in tech_carrier:
            for capacity_type in res.get_df("capacity", scenario=scenario).loc[tech].index.get_level_values("capacity_type").unique():
                # get initial cost
                # get initial cost
                c_initial = res.get_df("total_cost_pwa_initial_unit_cost", scenario=scenario).loc[tech, capacity_type]
                # get initial capacity
                q_initial = res.get_df("global_initial_capacity", scenario=scenario).loc[tech]
                # get learning rate
                learning_rate = res.get_df("learning_rate", scenario=scenario).loc[tech]

                if forecast_scenario is not None:
                    cap_add = res.get_total("capacity_addition", scenario=forecast_scenario).loc[tech].sum()
                    cum_cap = cap_add.cumsum() + q_initial
                else:
                    cum_cap = res.get_df("global_cumulative_capacity", scenario=scenario).loc[tech].values

                point_on_curve = unit_cost(cum_cap, c_initial, q_initial, learning_rate)
                unit_cost_over_time[tech] = {'cap': cum_cap, 'cost': point_on_curve}
    else:
        for tech in tech_carrier:
            try:
                point_on_curve = res.get_total("capex_specific_conversion", scenario=scenario).groupby(["technology"]).mean().loc[tech]
            except:
                point_on_curve = res.get_total("capex_specific_transport", scenario=scenario).groupby(["technology"]).mean().loc[tech]
            cap_tech = res.get_total("capacity", scenario=scenario).groupby(["technology"]).sum().loc[tech]
            unit_cost_over_time[tech] = {'cap': cap_tech, 'cost': point_on_curve}

    # Flatten the dictionary to create a DataFrame
    df_unit_cost_over_time = pd.DataFrame({(outerKey, innerKey): values for outerKey, innerDict in unit_cost_over_time.items() for innerKey, values in innerDict.items()}, index=range(14))

    # Rename the columns
    df_unit_cost_over_time.columns = pd.MultiIndex.from_tuples(df_unit_cost_over_time.columns, names=['tech', 'cap/cost'])

    return df_unit_cost_over_time.T

def plot_unit_cost_over_capacity(res, carrier= None,scenario=None):
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
    if carrier != None:
        data_extract = res.extract_reference_carrier(data_total, carrier, scenario)
        data_extract = data_extract.groupby(["technology"]).mean().T
        tech_carrier = data_extract.columns
    else:
        tech_carrier = res.get_df("capacity", scenario=scenario).groupby(["technology"]).sum().index

    fig2, ax2 = plt.subplots()
    colors = cycle(['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'])

    unit_cost_over_time = {}

    for tech in tech_carrier:
        for capacity_type in res.get_df("capacity", scenario=scenario).loc[tech].index.get_level_values(
                "capacity_type").unique():
            # get initial cost
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

            data_total_cap = res.get_df("global_cumulative_capacity", scenario=scenario).unstack()
            data_extract_cap = res.extract_reference_carrier(data_total_cap, carrier, scenario)
            data_extract_cap = data_extract_cap.groupby(["technology", "capacity_type"]).sum().T

            cap_tech = data_extract_cap.T.loc[tech, capacity_type]
            point_on_curve = unit_cost(cap_tech, c_initial, q_initial, learning_rate)

            unit_cost_over_time[tech] = {'cap': cap_tech, 'cost': point_on_curve}

            learning_curve = unit_cost(pwa_range, c_initial, q_initial, learning_rate)

            color = next(colors)

            ax2.scatter(cap_tech, point_on_curve, marker=tech_markers[tech], color=tech_colors[tech])
            ax2.plot(pwa_range, learning_curve, label=f'{tech}-{capacity_type}', color=tech_colors[tech])
            ax2.plot(q_initial, c_initial, marker='x', color=tech_colors[tech])
            ax2.legend()
            ax2.set_xlabel('Cum. Global Capacity')
            ax2.set_ylabel('Unit Cost')
            ax2.set_title('Cost Evolution of Technologies over cumulative global capacity')
            # ax2.set_xlim(0,1000)
        if save_fig:
            path = os.path.join(os.getcwd(), "outputs")
            path = os.path.join(path, os.path.basename(res.results[scenario]["analysis"]["dataset"]))
            path = os.path.join(path, "result_plots")
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(os.path.join(path, scenario + "cost_over_capacity" + "." + file_type))
        plt.show()


def plot_cost_reductions(res, df_unit_cost_over_time, plot_style=None,scenario=None, save_fig=False, file_type=None):
    if plot_style is not None:
        edgecolor = plot_style["edgecolor"]
        tech_colors = plot_style["tech_colors"]
        tech_hatches = plot_style["tech_hatches"]
        tech_markers = plot_style["tech_markers"]
        bar_width = plot_style["bar_width"]
    else:
        edgecolor = 'black'
        tech_colors = tech_colors_1
        tech_hatches = tech_hatches_1
        tech_markers = tech_markers
        bar_width = 0.3

    cost_reduction = {}
    for tech in df_unit_cost_over_time.index.levels[0]:
        try:
            c_initial = res.get_total("capex_specific_conversion", scenario=scenario).groupby(["technology"]).mean().loc[tech,0]
        except:
            c_initial = res.get_total("capex_specific_transport", scenario=scenario).groupby(["technology"]).mean().loc[tech,0]
        c_end = df_unit_cost_over_time.loc[tech, "cost"].iloc[-1]
        cost_reduction[tech] = (c_initial - c_end) / c_initial * 100

    # sort the cost reductions
    cost_reduction_df = pd.Series(cost_reduction)

    # Reverse the order of the DataFrame index
    cost_reduction_df = cost_reduction_df[::-1]

    # plot the cost reductions
    plt.figure(figsize=(6, 6))
    bars = plt.barh(cost_reduction_df.index, cost_reduction_df.values, color=[tech_colors[tech] for tech in cost_reduction_df.index], hatch= [tech_hatches[tech] for tech in cost_reduction_df.index])  # Using plt.barh() for horizontal bars
    plt.xlabel('Cost Reduction [%]')  # Adjusting x-label
    plt.ylabel('Technology')  # Adjusting y-label
    plt.title('Cost Reduction of Technologies from 2024 to 2050')

    # Add numbers inside the bars
    for bar, value in zip(bars, cost_reduction_df.values):
        plt.text(bar.get_width() - 5, bar.get_y() + bar.get_height() / 2, f'{value:.1f}%',
                 va='center', ha='center', color='white')

    plt.tight_layout()
    if save_fig:
        path = os.path.join(os.getcwd(), "outputs")
        path = os.path.join(path, os.path.basename(res.results[scenario]["analysis"]["dataset"]))
        path = os.path.join(path, "result_plots")
        if not os.path.exists(path):
            os.makedirs(path)
        file_path = os.path.join(path, scenario + "cost_reductions" + "." + file_type)
        print("Saving plot to:", file_path)  # Debug print statement
        plt.savefig(file_path)
    plt.show()

    return cost_reduction_df


def plot_component(res, component="capacity", scenario=None, plot_style=None, save_fig=False, file_type=None):
    if plot_style is not None:
        edgecolor = plot_style["edgecolor"]
        tech_colors = plot_style["tech_colors"]
        tech_hatches = plot_style["tech_hatches"]
        tech_markers = plot_style["tech_markers"]
        bar_width = plot_style["bar_width"]
    else:
        edgecolor = 'black'
        tech_colors = tech_colors_1
        tech_hatches = tech_hatches_1
        tech_markers = tech_markers
        bar_width = 0.3

    component_data = res.get_total(component, scenario=scenario)

    # Plot 2a: Plot of the hydrogen producing technologies
    fig, ax = plt.subplots(figsize=(10,8))
    cap_h2 = res.extract_reference_carrier(component_data, "hydrogen", scenario).groupby(["technology"]).sum()
    if 'biomethane_conversion' in component_data.index.levels[0]:
        hydrogen_output, df_gas, bio_share_SMR, bio_share_SMR_CCS = calculate_hydrogen_production(res, scenario=scenario)
        cap_h2 = add_biomethane_split(cap_h2, bio_share_SMR, bio_share_SMR_CCS)
        cap_h2 = cap_h2.reindex(h2_tech_meth).dropna()
    else:
        cap_h2 = cap_h2.reindex(h2_tech).dropna()
    # Plotting the stacked bar chart for cap_endog_transposed
    x = np.arange(len(cap_h2.columns))
    bottom = np.zeros(len(cap_h2.columns))
    opacity = 1.0
    for tech in cap_h2.index:
        # Calculate the x-positions for the bars in the current scenario
        ax.bar(x, cap_h2.loc[tech], bottom=bottom, alpha=opacity, width=bar_width, color=tech_colors[tech],hatch=tech_hatches[tech], edgecolor=edgecolor,label=tech)
        bottom += cap_h2.loc[tech]
    # Plot demand if capacity
    if component=="capacity":
        demand = res.get_total("demand", scenario=scenario).groupby(["carrier"]).sum().loc["hydrogen"] / 8760
        ax.plot(demand, label="H2 demand", color="black", linestyle="--")
    xtick_labels = [year * 2 + 2024 for year in range(len(cap_h2.columns))]
    ax.legend(loc='upper right', ncols=4)
    ax.set_xlabel("Years")
    ax.set_ylabel(f"{component.capitalize()} [GW]")
    ax.set_ylim(0, 40)
    ax.set_xticks(range(len(cap_h2.columns)))
    ax.set_xticklabels(xtick_labels, rotation=90)
    ax.set_title(f"{component.capitalize()} of Hydrogen Generating Conversion Technologies")

    fig.tight_layout()
    if save_fig:
        path = os.path.join(os.getcwd(), "outputs")
        path = os.path.join(path, os.path.basename(res.results[scenario]["analysis"]["dataset"]))
        path = os.path.join(path, "result_plots")
        if not os.path.exists(path):
            os.makedirs(path)
        fig.savefig(os.path.join(path,scenario + component +"_h2" + "." + file_type))
    fig.show()

    # Plot 2b: Plot for the CCS supporting technologies
    fig, ax = plt.subplots(figsize=(10,8))
    cap_ccs = res.extract_reference_carrier(component_data, "carbon", scenario).groupby(["technology"]).sum()
    # Plotting the stacked bar chart
    x = np.arange(len(cap_ccs.columns))
    bottom = np.zeros(len(cap_ccs.columns))
    opacity = 1.0
    for tech in cap_ccs.index:
        # Calculate the x-positions for the bars in the current scenario
        ax.bar(x, cap_ccs.loc[tech], bottom=bottom, alpha=opacity, width=bar_width, color=tech_colors[tech],hatch=tech_hatches[tech], edgecolor=edgecolor)
        bottom += cap_ccs.loc[tech]
    xtick_labels = [year * 2 + 2024 for year in range(len(cap_ccs.columns))]
    ax.legend(loc='upper right', labels=cap_ccs.index.values)
    ax.set_xlabel("Years")
    ax.set_ylabel(f"{component.capitalize()} [GW]")
    ax.set_ylim(0, 12)
    ax.set_xticks(range(len(cap_ccs.columns)))
    ax.set_xticklabels(xtick_labels, rotation=90)
    ax.set_title(f"{component.capitalize()} of Carbon Utilizing Technologies")

    fig.tight_layout()
    if save_fig:
        path = os.path.join(os.getcwd(), "outputs")
        path = os.path.join(path, os.path.basename(res.results[scenario]["analysis"]["dataset"]))
        path = os.path.join(path, "result_plots")
        if not os.path.exists(path):
            os.makedirs(path)
        fig.savefig(os.path.join(path, scenario + component + "_ccs" + "." + file_type))
    fig.show()


    # Plot 2c: Plot for the electricity producing technologies
    fig, ax = plt.subplots(figsize=(10,8))
    power_tech = ["pv_ground", "pv_rooftop", "wind_onshore", "wind_offshore"]
    cap_power = res.extract_reference_carrier(component_data, "electricity", scenario).groupby(["technology"]).sum()
    # Plotting the stacked bar chart
    x = np.arange(len(cap_power.columns))
    bottom = np.zeros(len(cap_power.columns))
    opacity = 1.0
    for tech in cap_power.index:
        # Calculate the x-positions for the bars in the current scenario
        ax.bar(x, cap_power.loc[tech], bottom=bottom, alpha=opacity, width=bar_width, color=tech_colors[tech],hatch=tech_hatches[tech], edgecolor=edgecolor)
        bottom += cap_power.loc[tech]
    xtick_labels = [year * 2 + 2024 for year in range(len(cap_power.columns))]
    ax.legend(loc='upper right', labels=cap_power.index.values)
    ax.set_xlabel("Years")
    ax.set_ylabel(f"{component.capitalize()} [GW]")
    ax.set_ylim(0, 50)
    ax.set_xticks(range(len(cap_power.columns)))
    ax.set_xticklabels(xtick_labels, rotation=90)
    ax.set_title(f"{component.capitalize()} of Power Generating Conversion Technologies")

    fig.tight_layout()
    if save_fig:
        path = os.path.join(os.getcwd(), "outputs")
        path = os.path.join(path, os.path.basename(res.results[scenario]["analysis"]["dataset"]))
        path = os.path.join(path, "result_plots")
        if not os.path.exists(path):
            os.makedirs(path)
        fig.savefig(os.path.join(path, scenario + component + "_power" + "." + file_type))
    fig.show()



    # Plot 2d: Pot for anaerobic digestion
    if "anaerobic_digestion" in component_data.index.levels[0]:
        fig, ax = plt.subplots(figsize=(10,8))
        cap_anaerobic = component_data.loc["anaerobic_digestion"].sum()
        x = cap_anaerobic.index
        bottom = np.zeros(len(cap_anaerobic.index))
        ax.bar(x, cap_anaerobic, bottom=bottom, alpha=opacity, width=bar_width, color=tech_colors["anaerobic_digestion"],hatch=tech_hatches["anaerobic_digestion"], edgecolor=edgecolor)
        xtick_labels = [year * 2 + 2024 for year in range(len(cap_anaerobic.index))]
        ax.legend(loc='upper right', labels=["Anaerobic Digestion"])
        ax.set_xlabel("Years")
        ax.set_ylabel(f"{component.capitalize()} [GW]")
        ax.set_ylim(0, 25)
        ax.set_xticks(range(len(cap_anaerobic.index)))
        ax.set_xticklabels(xtick_labels, rotation=90)
        ax.set_title(f"{component.capitalize()} of Anaerobic Digestion")

        fig.tight_layout()
        if save_fig:
            path = os.path.join(os.getcwd(), "outputs")
            path = os.path.join(path, os.path.basename(res.results[scenario]["analysis"]["dataset"]))
            path = os.path.join(path, "result_plots")
            if not os.path.exists(path):
                os.makedirs(path)
            fig.savefig(os.path.join(path, scenario + component + "_biometh" + "." + file_type))
        fig.show()


def calculate_hydrogen_production(res, scenario=None):
    # Plot 3a: Hydrogen production from technologies
    flow_conversion_output = res.get_total("flow_conversion_output", scenario=scenario).groupby(["technology", "carrier"]).sum()
    hydrogen_output = flow_conversion_output[flow_conversion_output.index.get_level_values(1) == "hydrogen"].droplevel(1)
    # Plot 3b: Input natural gas
    flow_input_gas = res.get_total("flow_conversion_input", scenario=scenario).groupby(["carrier"]).sum().loc["natural_gas"]

    if 'biomethane_conversion' in flow_conversion_output.index.levels[0]:
        # Natural gas coming out of biomethane conversion, which is actually biomethane
        biomethane_input = flow_conversion_output.loc["biomethane_conversion", "natural_gas"].values
    else:
        biomethane_input = np.zeros(len(hydrogen_output.columns))

    natural_gas_input = flow_input_gas.values - biomethane_input
    df_gas = pd.DataFrame()
    df_gas["natural_gas"] = natural_gas_input
    df_gas["biomethane"] = biomethane_input
    # Share of hydrogen production
    hydrogen_output / hydrogen_output.sum()
    df_gas.T / df_gas.T.sum(axis=0)

    # Production share of biomethane
    # V2: First SMR CCS then SMR
    if 'SMR_CCS' in hydrogen_output.index.values:
        input_flow_SMR_CCS = res.get_total("flow_conversion_input", scenario=scenario).loc["SMR_CCS"].sum()
        input_flow_SMR_CCS_biomethane = df_gas["biomethane"]
        input_flow_SMR_CCS_biomethane[(input_flow_SMR_CCS_biomethane > input_flow_SMR_CCS)] = input_flow_SMR_CCS
        bio_share_SMR_CCS = (input_flow_SMR_CCS_biomethane / input_flow_SMR_CCS).fillna(0)

        residual = df_gas["biomethane"] - input_flow_SMR_CCS_biomethane
        input_flow_SMR_biomethane = residual
        input_flow_SMR = res.get_total("flow_conversion_input", scenario=scenario).loc["SMR"].sum()
        bio_share_SMR = (input_flow_SMR_biomethane / input_flow_SMR).fillna(0)
    else:
        bio_share_SMR_CCS = 0
        bio_share_SMR = df_gas["biomethane"] / df_gas.sum(axis=1)


    return hydrogen_output, df_gas, bio_share_SMR, bio_share_SMR_CCS


def plot_hydrogen_production(hydrogen_output, df_gas, plot_style=None, scenario=None, save_fig=False, file_type=None):
    if plot_style is not None:
        edgecolor = plot_style["edgecolor"]
        tech_colors = plot_style["tech_colors"]
        tech_hatches = plot_style["tech_hatches"]
        tech_markers = plot_style["tech_markers"]
        bar_width = plot_style["bar_width"]
    else:
        edgecolor = 'black'
        tech_colors = tech_colors_1
        tech_hatches = tech_hatches_1
        tech_markers = tech_markers
        bar_width = 0.3

    hydrogen_output.T.plot.bar(stacked=True, color=[tech_colors[tech] for tech in hydrogen_output.index])
    plt.legend(loc="center right")
    xtick_labels = [year * 2 + 2024 for year in range(len(hydrogen_output.columns))]
    plt.xticks(range(len(hydrogen_output.columns)), labels=xtick_labels)
    plt.title(f"Hydrogen Output of Hydrogen Generating Technologies")
    plt.xlabel("Year")
    plt.ylabel(f"Output Flow [GWh/a]")
    plt.tight_layout()
    if save_fig:
        path = os.path.join(os.getcwd(), "outputs")
        path = os.path.join(path, os.path.basename(res.results[scenario]["analysis"]["dataset"]))
        path = os.path.join(path, "result_plots")
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(os.path.join(path, scenario +"_hydrogen_production" + "." + file_type))
    plt.show()

    df_gas.plot.bar(stacked=True, color=[carrier_colors[carrier] for carrier in df_gas.columns])
    xtick_labels = [year * 2 + 2024 for year in range(len(df_gas.index))]
    plt.xticks(range(len(df_gas.index)), labels=xtick_labels)
    plt.xlabel("Year")
    plt.ylabel("Input Flow [GWh]")
    plt.title("Input Flow of Natural Gas and Biomethane")
    plt.legend(loc="upper right")
    plt.tight_layout()
    if save_fig:
        path = os.path.join(os.getcwd(), "outputs")
        path = os.path.join(path, os.path.basename(res.results[scenario]["analysis"]["dataset"]))
        path = os.path.join(path, "result_plots")
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(os.path.join(path, scenario +  "biomethane_production" + "." + file_type))
    plt.show()

def convert_tech_output_2_pathway_output(hydrogen_output, df_gas):
    production_share = hydrogen_output / hydrogen_output.sum()
    bio_share = df_gas["biomethane"] / df_gas.sum(axis=1)
    production_share.loc["SMR bio"] = bio_share * production_share.loc["SMR"]
    production_share.loc["SMR gas"] = (1 - bio_share) * production_share.loc["SMR"]

    production_share_path = pd.DataFrame()
    production_share_path["CCS"] = production_share.loc["gasification_CCS"] + production_share.loc["SMR_CCS"]
    production_share_path["Bio"] = production_share.loc["SMR bio"] + production_share.loc["gasification"]
    production_share_path["Power"] = production_share.loc["electrolysis"]
    production_share_path["Fossil"] = production_share.loc["SMR gas"]

    return production_share_path


def plot_capacity_multi_bar(res, tech_selection, plot_style, normalize=False, annotation=None, separate_base=False, save_fig=False, file_type=None):
    # Plot style
    bar_width = 0.6 / len(res.scenarios)
    if plot_style is not None:
        edgecolor = plot_style["edgecolor"]
        tech_colors = plot_style["tech_colors"]
        tech_hatches = plot_style["tech_hatches"]

    # Double Bar Plot
    if normalize:
        figsize = (10,10)
    else:
        figsize= (10,8)

    fig, axis = plt.subplots(figsize=figsize)
    for i, scenario in enumerate(res.scenarios):
        # Get the capacity
        cap = res.get_total("capacity", scenario=scenario).groupby(["technology"]).sum().loc[tech_selection]
        # Sort the hydrogen producing technologies
        if tech_selection == h2_tech:
            # Get the hydrogen production
            hydrogen_output, df_gas, bio_share_SMR, bio_share_SMR_CCS = calculate_hydrogen_production(res, scenario=scenario)
            # Update the capacities
            cap = add_biomethane_split(cap, bio_share_SMR, bio_share_SMR_CCS)
            # Sort the index
            cap = cap.reindex(h2_tech_meth)

        if normalize:
            cap = cap.div(cap.sum(axis=0), axis=1)

        # Plotting the stacked bar chart for cap_endog_transposed
        opacity = 1-i*0.05
        bar_center = np.arange(len(cap.columns))
        bottom = np.zeros(len(cap.columns))
        offset = (2 * i - (len(res.scenarios) - 1)) * bar_width / 2
        x = bar_center + offset
        if (i == 0) & (separate_base == True):
            x = x - bar_width/2
        for tech in cap.index:
            # Calculate the x-positions for the bars in the current scenario
            axis.bar(x, cap.loc[tech], bottom=bottom, alpha=opacity, width=bar_width, color=tech_colors[tech],hatch=tech_hatches[tech], edgecolor=edgecolor)
            bottom += cap.loc[tech]
        if annotation is not None:
            anno_index = annotation.loc[i, 0]
            anno_label = annotation.loc[i, "index"]
            y_position = cap[anno_index].sum()
            axis.annotate(anno_label, xy=(x[anno_index], y_position + 0.5),xytext=(x[anno_index], y_position + 2.5 + i),arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=10, ha='center')

    xtick_labels = [year * 2 + 2024 for year in range(len(cap.columns))]
    axis.legend(loc="upper right", labels =cap.index.values, ncols=4)
    axis.set_xlabel("Year")
    axis.set_ylabel("Capacity [GW]")
    axis.set_xlim(-1, len(cap.columns))
    axis.set_xticks(range(len(cap.columns)))
    axis.set_xticklabels(xtick_labels, rotation=90)

    fig.tight_layout()
    if save_fig:
        path = os.path.join(os.getcwd(), "outputs")
        path = os.path.join(path, data_set_name)
        path = os.path.join(path, "result_plots")
        if not os.path.exists(path):
            os.makedirs(path)
        fig.savefig(os.path.join(path, tech_selection [0]+ "_capacity_one_plot" + "." + file_type))
    fig.show()

def add_biomethane_split(df, bio_share_SMR, bio_share_SMR_CCS):
    df.loc["SMR biomethane"] = df.loc["SMR"] * bio_share_SMR
    df.loc["SMR"] = df.loc["SMR"] * (1 - bio_share_SMR)
    if 'SMR_CCS' in df.index:
        df.loc["SMR_CCS biomethane"] = df.loc["SMR_CCS"] * bio_share_SMR_CCS
        df.loc["SMR_CCS"] = df.loc["SMR_CCS"] * (1 - bio_share_SMR_CCS)
    return df

def plot_costs(res, scenario=None,save_fig=False,file_type=None):
    fig, ax = plt.subplots(figsize=(10,8))
    df = pd.DataFrame({'CAPEX': res.get_total("cost_capex_total", scenario=scenario),
                       'OPEX': res.get_total("cost_opex_total", scenario=scenario),
                       'Carrier': res.get_total("cost_carrier_total", scenario=scenario),
                       'Emissions': res.get_total("cost_carbon_emissions_total", scenario=scenario)})

    net_present_cost = res.get_total("net_present_cost", scenario=scenario)
    bottom = np.zeros(len(df.index))
    for cost in df.columns:
        ax.bar(df.index, df[cost], bottom=bottom, label=cost, width=0.5)
        bottom += df[cost]
    ax.set_xlabel('Years')
    ax.set_ylabel('Costs')
    ax.legend(loc='upper right')
    ax.set_title(f'Total costs over the years for {scenario }')

    ax_right = ax.twinx()
    ax_right.plot(net_present_cost, color='black', marker='.', linewidth=0.8, label="Net Present Cost")
    ax_right.set_ylabel('Net present cost [Mio. €]')
    ax_right.legend(loc='upper center')
    fig.show()

    print(f"Net Present cost of {scenario} is {net_present_cost.sum()}")
    if save_fig:
        path = os.path.join(os.getcwd(), "outputs")
        path = os.path.join(path, data_set_name)
        path = os.path.join(path, "result_plots")
        if not os.path.exists(path):
            os.makedirs(path)
        fig.savefig(os.path.join(path, scenario + "cost_split" + "." + file_type))

    return df

def plot_capex_multi_bar(res, plot_style, normalize=False, save_fig=False, file_type=None):
    if plot_style is not None:
        edgecolor = plot_style["edgecolor"]
        tech_colors = plot_style["tech_colors"]
        tech_hatches = plot_style["tech_hatches"]
        tech_markers = plot_style["tech_markers"]
        bar_width = plot_style["bar_width"]
        line_styles = plot_style["line_styles"]
        markers = plot_style["markers"]

    fig, axis = plt.subplots(figsize=(10,10))
    ax_right = axis.twinx()
    for i, scenario in enumerate(res.scenarios):
        costs = res.get_total("cost_capex", scenario=scenario).groupby(["technology"]).sum()
        if normalize:
            costs = costs.div(costs.sum(axis=0), axis=1)
        for tech in costs.index:
            # Plotting the stacked bar chart for cap_endog_transposed
            opacity = 1 - i * 0.3
            bar_center = np.arange(len(costs.columns))
            bottom = np.zeros(len(costs.columns))
            offset = (2 * i - (len(res.scenarios) - 1)) * bar_width / 2
            x = bar_center + offset
            for tech in costs.index:
                # Calculate the x-positions for the bars in the current scenario
                axis.bar(x, costs.loc[tech], bottom=bottom, alpha=opacity, width=bar_width, color=tech_colors[tech],hatch=tech_hatches[tech], edgecolor=edgecolor)
                bottom += costs.loc[tech]
        ax_right.plot(res.get_df("net_present_cost", scenario=scenario), color='darkred', marker=markers[i], linestyle=line_styles[i], linewidth=0.8)
        ax_right.set_ylabel('Net present cost [Mio. €]')

    xtick_labels = [year * 2 + 2024 for year in range(len(costs.columns))]
    axis.legend(loc="upper center", labels=costs.index.values,  bbox_to_anchor=(0.5, 1.16), ncols=4)
    axis.set_xlabel("Year")
    axis.set_ylabel("Costs [Mio. EUR]")
    axis.set_xlim(-1, len(costs.columns))
    axis.set_xticks(range(len(costs.columns)))
    axis.set_xticklabels(xtick_labels, rotation=90)
    fig.tight_layout()
    if save_fig:
        path = os.path.join(os.getcwd(), "outputs")
        path = os.path.join(path, data_set_name)
        path = os.path.join(path, "result_plots")
        if not os.path.exists(path):
            os.makedirs(path)
        fig.savefig(os.path.join(path, "capex_one_plot" + "." + file_type))
    fig.show()


############################################## Result anaylsis ##############################################

# I. Read the results of the models
folder_path = os.path.dirname(__file__)
data_set_name = "20240303_Hydrogen_endog"

res = Results(os.path.join(folder_path, data_set_name))

save_fig = True
file_type = "png"

df_path = os.path.join(os.getcwd(), "outputs", data_set_name)

############################################## PLOTS #########################################################

# Colors
tech_colors_1 = {
    'anaerobic_digestion': (0.24, 0.70, 0.44),
    'biomethane_conversion':(0.5, 0.5, 0),
    'biomethane_transport':  (0.5, 0.5, 0),
    'electrolysis': (0.64, 0.52, 0.26),
    'pv_ground' : (0.92, 0.86, 0.55),
    'pv_rooftop' : (0.87, 0.81, 0.50),
    'wind_onshore' :   (0.53, 0.85, 0.80),
    'wind_offshore' :   (0.00, 0.81, 0.82),
    'SMR_CCS': (0.65, 0.74, 0.87),
    'gasification_CCS': (0.5, 0.56, 0.26),
    'carbon_removal': (0.43, 0.43, 0.43),
    'carbon_pipeline': (0.66, 0.66, 0.66),
    'carbon_storage':  (0.23, 0.23, 0.23),
    'gasification': (0.5, 0.56, 0.26),
    'SMR': (0.65, 0.74, 0.87),
    'dry_biomass_truck':  (0.6, 0.3, 0),
    'hydrogen_pipeline': (0.30, 0.48, 0.73),
    'SMR biomethane': (0.2, 0.58, 0.67),
    'SMR_CCS biomethane':  (0.2, 0.58, 0.67)
}

# Hatches
tech_hatches_1 = {
    'anaerobic_digestion': None,
    'biomethane_conversion':None,
    'biomethane_transport':  None,
    'electrolysis': None,
    'pv_ground' : None,
    'pv_rooftop' : None,
    'wind_onshore' : None,
    'wind_offshore' :  None,
    'SMR_CCS': '//',
    'gasification_CCS': '//',
    'carbon_removal': None,
    'carbon_pipeline': None,
    'carbon_storage':  None,
    'gasification': None,
    'SMR': None,
    'dry_biomass_truck':None,
    'hydrogen_pipeline': None,
    'biomethane_SMR': None,
    'biomethane_SMR_CCS': '//',
    'SMR biomethane':None,
    'SMR_CCS biomethane': '//'
}

# Carrier colors
carrier_colors = {
    "biomethane": (0.7, 0.9, 0),
    "natural_gas": (0.6, 0.3, 0),
}

# Markers
tech_markers = {
    'anaerobic_digestion': '.',
    'biomethane_conversion':'.',
    'biomethane_transport':  '.',
    'electrolysis': '.',
    'pv_ground' : '.',
    'pv_rooftop' : '.',
    'wind_onshore' : '.',
    'wind_offshore' :  '.',
    'SMR_CCS': 'x',
    'gasification_CCS': 'x',
    'carbon_removal': '.',
    'carbon_pipeline': '.',
    'carbon_storage':  '.',
    'gasification': '.',
    'SMR': '.',
    'dry_biomass_truck':'.',
    'hydrogen_pipeline': '.',
    'biomethane_SMR': '.',
    'biomethane_SMR_CCS': 'x',
    'SMR biomethane':'.',
    'SMR_CCS biomethane': 'x'
}

# Colors
tech_colors_2 = {
    'anaerobic_digestion': (1, 0.8, 0),
    'biomethane_conversion':(0.5, 0.5, 0),
    'biomethane_transport':  (0.5, 0.5, 0),
    'electrolysis': (0.64, 0.52, 0.26),
    'pv_ground' : (0.13, 0.55, 0.13),
    'pv_rooftop' : (0.24, 0.70, 0.44),
    'wind_onshore' :   (0.53, 0.85, 0.80),
    'wind_offshore' :   (0.00, 0.81, 0.82),
    'SMR_CCS': (0.3, 0.49, 0.74),
    'gasification_CCS': (0.5, 0.56, 0.26),
    'carbon_removal': (0.0, 0.0, 0.6),
    'carbon_pipeline': (0.0, 0.0, 1.0),
    'carbon_storage':  (0.45, 0.60, 0.75),
    'gasification': (0.75, 0.78, 0.63),
    'SMR':  (0.65, 0.74, 0.87),
    'dry_biomass_truck':  (0.6, 0.3, 0),
    'hydrogen_pipeline': (0.5, 0.5, 0.5),
    'SMR biomethane': (0.2, 0.58, 0.67),
    'SMR_CCS biomethane':  (0.2, 0.58, 0.67),
}

# Hatches
tech_hatches_2 = {
    'anaerobic_digestion': None,
    'biomethane_conversion':None,
    'biomethane_transport':  None,
    'electrolysis': None,
    'pv_ground' : None,
    'pv_rooftop' : None,
    'wind_onshore' : None,
    'wind_offshore' :  None,
    'SMR_CCS': None,
    'gasification_CCS': None,
    'carbon_removal': None,
    'carbon_pipeline': None,
    'carbon_storage':  None,
    'gasification': None,
    'SMR': None,
    'dry_biomass_truck':None,
    'hydrogen_pipeline': None,
    'SMR biomethane': None,
    'SMR_CCS biomethane': None
}

tech_colors_3 = {
    'anaerobic_digestion': (0.24, 0.70, 0.44),
    'biomethane_conversion': (0.5, 0.5, 0),
    'biomethane_transport': (0.5, 0.5, 0),
    'electrolysis': (0.64, 0.52, 0.26),
    'pv_ground': (0.92, 0.86, 0.55),
    'pv_rooftop': (0.87, 0.81, 0.50),
    'wind_onshore': (0.53, 0.85, 0.80),
    'wind_offshore': (0.00, 0.81, 0.82),
    'SMR_CCS': (0.5, 0.6, 0.7),  # Darker color for SMR_CCS
    'gasification_CCS': (0.4, 0.45, 0.2),  # Darker color for gasification_CCS
    'carbon_removal': (0.43, 0.43, 0.43),
    'carbon_pipeline': (0.66, 0.66, 0.66),
    'carbon_storage': (0.23, 0.23, 0.23),
    'gasification': (0.5, 0.56, 0.26),
    'SMR': (0.65, 0.74, 0.87),
    'dry_biomass_truck': (0.6, 0.3, 0),
    'hydrogen_pipeline': (0.30, 0.48, 0.73),
    'SMR biomethane': (0.2, 0.58, 0.67),
    'SMR_CCS biomethane': (0.2, 0.58, 0.67)
}



# Tech groups
h2_tech = [ "SMR","SMR_CCS","gasification", "gasification_CCS", "electrolysis"]
power_tech = ["pv_ground", "pv_rooftop", "wind_onshore", "wind_offshore"]
ccs_tech = ["carbon_removal", "carbon_storage", "carbon_pipeline"]
biometh_tech = ["anaerobic_digestion"]
h2_tech_meth = ["SMR", "SMR_CCS", "SMR biomethane", "SMR_CCS biomethane", "gasification", "gasification_CCS", "electrolysis"]
tech_groups = [h2_tech, power_tech, ccs_tech, biometh_tech]

plot_style = {
    "edgecolor": 'black',
    "tech_colors": tech_colors_1,
    "tech_hatches": tech_hatches_1,
    "tech_markers": tech_markers,
    "bar_width": 0.3,
    "line_styles": ['-', '--', '-.', ':'],
    "markers": ['^','s', '.', 'x', 'o', 's']
}


##### WE MAKE SINGLE PLOTS FOR EACH CASE HERE, NO COMPARISON OF THE CASES YET
for scenario in res.scenarios:
    #### Plot 1: Description of the costs of the technologies
    # Plot 1a: Evolution of the costs over time
    carriers = ["electricity", "hydrogen", "carbon"]
    fig, axis = plt.subplots(1,3, figsize=(10,8))
    for i, carrier in enumerate(carriers):
        unit_cost = calculate_unit_cost_over_time(res, carrier=carrier, scenario=scenario)
        plot_unit_cost_over_time(res, unit_cost, scenario, fig, axis[i], plot_style=plot_style)
    fig.tight_layout()
    fig.show()

    # Plot 1b: Cost reduction rank for technologies
    unit_cost_all = calculate_unit_cost_over_time(res, carrier=None, scenario=scenario)
    cost_reductions_df = plot_cost_reductions(res, unit_cost_all,plot_style=plot_style, scenario=scenario, save_fig=save_fig, file_type=file_type)

    #### Plot 2: Description of the capacities of technologies --> Capacity addition
    component = "capacity"
    plot_component(res, component=component, scenario=scenario, plot_style=plot_style, save_fig=save_fig, file_type=file_type)

    component = "capacity_addition"
    plot_component(res, component=component, scenario=scenario, plot_style=plot_style,save_fig=save_fig, file_type=file_type)


    #### Plot 3: Description of the flows of technologies
    hydrogen_output, df_gas, bio_share_SMR, bio_share_SMR = calculate_hydrogen_production(res, scenario=scenario)
    plot_hydrogen_production(hydrogen_output, df_gas, plot_style=plot_style, scenario=scenario, save_fig=save_fig, file_type=file_type)

    #### Plot 4: Costs
    # Plot the Net present cost split
    plot_costs(res,scenario=scenario,save_fig=save_fig,file_type=file_type)



##### Comparison of Scenarios
# Base Analysis
if data_set_name.endswith('_base'):
    annotation = pd.Series({'EX': 7, 'EN': 7}).reset_index()

    fig, axis = plt.subplots(1, 2, figsize=(10, 8))
    # Cost Evolution
    for i, scenario in enumerate(res.scenarios):
        unit_cost = calculate_unit_cost_over_time(res, carrier ="hydrogen", scenario=scenario)
        plot_unit_cost_over_time(res, unit_cost, scenario, fig, axis[i], plot_style=plot_style, save_fig=save_fig,file_type=file_type)

    fig.tight_layout()
    if save_fig:
        path = os.path.join(os.getcwd(), "outputs")
        path = os.path.join(path, data_set_name)
        path = os.path.join(path, "result_plots")
        if not os.path.exists(path):
            os.makedirs(path)
        fig.savefig(os.path.join(path, "cost_evolution" + "." + file_type))
    fig.show()


    # Capacity
    for tech_selection in tech_groups:
        plot_capacity_multi_bar(res, tech_selection, normalize=False, plot_style=plot_style, annotation=annotation, save_fig=save_fig, file_type=file_type)

    # Costs
    plot_capex_multi_bar(res, plot_style=plot_style, normalize=False, save_fig=save_fig, file_type=file_type)

    # Overestimation of cost reductions
    unit_cost_forecast = calculate_unit_cost_over_time(res, carrier="electricity", scenario="scenario_1", forecast_scenario="scenario_")
    unit_cost_exog = calculate_unit_cost_over_time(res, carrier="electricity", scenario="scenario_")

    fig, ax = plt.subplots()
    tech_colors = tech_colors_1
    for tech in unit_cost_exog.index.levels[0]:
            ax.plot(unit_cost_exog.loc[tech, "cost"].index, unit_cost_exog.loc[tech, "cost"], label=tech, color=tech_colors[tech], linewidth=0.8)
            ax.plot(unit_cost_forecast.loc[tech, "cost"].index, unit_cost_forecast.loc[tech, "cost"],  color=tech_colors[tech], linewidth=0.8, linestyle="--")
            ax.fill_between(unit_cost_exog.loc[tech, "cost"].index, unit_cost_exog.loc[tech, "cost"],unit_cost_forecast.loc[tech, "cost"], color=tech_colors[tech], alpha=0.1)

    xtick_labels = [year * 2 + 2024 for year in range(len(unit_cost_exog.columns))]
    ax.set_xlabel("Years")
    ax.set_ylabel("Unit cost [EUR/kW]")
    ax.set_ylim(0, 3000)
    ax.set_xticks(range(len(unit_cost_exog.columns)))
    ax.set_xticklabels(xtick_labels, rotation=90)
    ax.legend(loc="upper center", ncols=3)
    fig.tight_layout()
    fig.show()





# Myopic foresight Analysis
elif data_set_name.endswith('_myopic'):
    ncols = len(res.scenarios)
    nrows = 1
    if ncols > 4:
        ncols = 4
        nrows = len(res.scenarios)//ncols+1
    fig, axis = plt.subplots(nrows,ncols, figsize=(10,8))
    for i, scenario in enumerate(res.scenarios):
        row = i//ncols
        col = i % ncols
        unit_cost = calculate_unit_cost_over_time(res, carrier="hydrogen", scenario=scenario)
        if nrows == 1:
            plot_unit_cost_over_time(res, unit_cost, scenario, fig, axis[i], plot_style=plot_style, save_fig=save_fig, file_type=file_type)
        else:
            plot_unit_cost_over_time(res, unit_cost, scenario, fig, axis[row, col], plot_style=plot_style, save_fig=save_fig, file_type=file_type)

    fig.tight_layout()
    if save_fig:
        path = os.path.join(os.getcwd(), "outputs")
        path = os.path.join(path, data_set_name)
        path = os.path.join(path, "result_plots")
        if not os.path.exists(path):
            os.makedirs(path)
        fig.savefig(os.path.join(path, "cost_evolution" + "." + file_type))
    fig.show()

    # Capacity
    for tech_selection in tech_groups:
        plot_capacity_multi_bar(res, tech_selection, plot_style=plot_style,save_fig=save_fig, file_type=file_type)


    # Extras
    # Cost Evolution of technologies
    df_path = os.path.join(os.getcwd(), "outputs", data_set_name)

    unit_cost_2y = plot_unit_cost_over_time(res, carrier="hydrogen", scenario="scenario_", save_fig=False,file_type=file_type)
    unit_cost_5y = plot_unit_cost_over_time(res, carrier="hydrogen", scenario="scenario_1", save_fig=False,file_type=file_type)
    unit_cost_10y = plot_unit_cost_over_time(res, carrier="hydrogen", scenario="scenario_2", save_fig=False,file_type=file_type)

    fig, ax = plt.subplots(1,5,figsize=(20, 10))
    i = 0
    for tech in h2_tech:
        color = tech_colors.get(tech)
        ax[i].plot(unit_cost_endog.loc[tech, "cost"], label=tech, color=tech_colors[tech])
        ax[i].plot(unit_cost_2y.loc[tech, "cost"], label="2 year foresight", linestyle="--", color=tech_colors[tech])
        ax[i].plot(unit_cost_5y.loc[tech, "cost"], label="5 year foresight", linestyle=":", color=tech_colors[tech])
        ax[i].plot(unit_cost_10y.loc[tech, "cost"], label="10 year foresight", linestyle="-.", color=tech_colors[tech])
        xtick_labels = [year * 2 + 2024 for year in range(len(unit_cost_endog.columns))]
        for axis in ax:
            axis.set_xlabel("Years")
            axis.set_ylabel("Unit cost [EUR/kW]")
            axis.set_ylim(500, 2250)
            axis.set_xticks(range(len(unit_cost_endog.columns)))
            axis.set_xticklabels(xtick_labels, rotation=90)
            axis.legend(loc="upper right")
        i = i+1
    plt.tight_layout()
    if save_fig:
        path = os.path.join(os.getcwd(), "outputs")
        path = os.path.join(path, data_set_name)
        path = os.path.join(path, "result_plots")
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(os.path.join(path, "MYOPIC_cost_evolution_tech_sep" + "." + file_type))
    plt.show()


    unit_cost_endog_transposed = unit_cost_endog.xs("cost", level=1).T
    unit_cost_2y_transposed = unit_cost_2y.xs("cost", level=1).T
    unit_cost_5y_transposed = unit_cost_5y.xs("cost", level=1).T
    unit_cost_10y_transposed = unit_cost_10y.xs("cost", level=1).T

    fig, ax = plt.subplots(1,4, figsize=(20, 10))
    unit_cost_endog_transposed.plot(ax=ax[0], color=[tech_colors[tech] for tech in unit_cost_endog_transposed.columns], label=tech)
    unit_cost_2y_transposed.plot(ax=ax[1], color=[tech_colors[tech] for tech in unit_cost_2y_transposed.columns], label=tech)
    unit_cost_5y_transposed.plot(ax=ax[2], color=[tech_colors[tech] for tech in unit_cost_5y_transposed.columns], label=tech)
    unit_cost_10y_transposed.plot(ax=ax[3], color=[tech_colors[tech] for tech in unit_cost_10y_transposed.columns], label=tech)
    xtick_labels = [year * 2 + 2024 for year in range(len(unit_cost_endog.columns))]
    for axis in ax:
        axis.set_xlabel("Years")
        axis.set_ylabel("Unit cost [EUR/kW]")
        axis.set_ylim(500, 2250)
        axis.set_xticks(range(len(unit_cost_endog.columns)))
        axis.set_xticklabels(xtick_labels, rotation=90)
        axis.legend(loc="upper right")
    ax[0].set_title("Endogenous Base Case")
    ax[1].set_title("2 year foresight")
    ax[2].set_title("5 year foresight")
    ax[3].set_title("10 year foresight")
    plt.tight_layout()
    if save_fig:
        path = os.path.join(os.getcwd(), "outputs")
        path = os.path.join(path, data_set_name)
        path = os.path.join(path, "result_plots")
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(os.path.join(path, "MYOPIC_cost_evolution_all_tech" + "." + file_type))
    plt.show()
    #
    # # Compare cost reductions
    # diff_unit_cost_2y = (unit_cost_2y - unit_cost_endog) / unit_cost_endog * 100
    # diff_unit_cost_5y = (unit_cost_5y - unit_cost_endog) / unit_cost_endog * 100
    # diff_unit_cost_10y = (unit_cost_10y - unit_cost_endog) / unit_cost_endog * 100
    #
    # fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(18, 5))
    # i = 0
    #
    # # Compare cost reductions over whole horizon
    # cost_reductions = pd.DataFrame()
    # cost_reductions["base"] = pd.read_pickle(os.path.join(df_path, 'cost_reductions_endog.pkl')).loc[h2_tech]
    # cost_reductions["2y"] = plot_cost_reductions(res, unit_cost_2y, scenario="scenario_", save_fig=False, file_type=file_type)
    # cost_reductions["5y"] = plot_cost_reductions(res, unit_cost_5y, scenario="scenario_1", save_fig=False, file_type=file_type)
    # cost_reductions["10y"] = plot_cost_reductions(res, unit_cost_10y, scenario="scenario_2", save_fig=False, file_type=file_type)
    # fig, ax = plt.subplots(figsize=(10, 5))
    # cost_reductions.plot.bar(stacked=False, color=[tech_colors[tech] for tech in cost_reductions.index])
    # plt.show()


    # Capacities
    for tech_selection in tech_groups:
        cap_endog = pd.read_pickle(os.path.join(df_path, 'capacity_endog.pkl')).loc[tech_selection]
        cap_2y = res.get_total("capacity", scenario="scenario_").groupby(["technology"]).sum().loc[tech_selection]
        cap_5y = res.get_total("capacity", scenario="scenario_1").groupby(["technology"]).sum().loc[tech_selection]
        cap_10y = res.get_total("capacity", scenario="scenario_2").groupby(["technology"]).sum().loc[tech_selection]

        cap_endog_transposed = cap_endog.T
        cap_2y_transposed = cap_2y.T
        cap_5y_transposed = cap_5y.T
        cap_10y_transposed = cap_10y.T


        fig, ax = plt.subplots(1, 4, figsize=(10, 5))
        # Plotting the stacked bar chart
        cap_endog_transposed.plot(kind='bar', stacked=True, ax =ax[0], color=[tech_colors[tech] for tech in cap_endog_transposed.columns])
        cap_2y_transposed.plot(kind='bar', stacked=True, ax =ax[1], color=[tech_colors[tech] for tech in cap_endog_transposed.columns])
        cap_5y_transposed.plot(kind='bar', stacked=True, ax =ax[2], color=[tech_colors[tech] for tech in cap_endog_transposed.columns])
        cap_10y_transposed.plot(kind='bar', stacked=True, ax =ax[3], color=[tech_colors[tech] for tech in cap_endog_transposed.columns])


        xtick_labels = [year * 2 + 2024 for year in range(len(cap_endog.columns))]
        for axis in ax:
            axis.set_xlabel("Years")
            axis.set_ylabel("Capacity [GW]")
            axis.set_ylim(0, 65)
            axis.set_xticks(range(len(cap_endog.columns)))
            axis.set_xticklabels(xtick_labels, rotation=90)
            axis.legend(loc="upper right")
        plt.tight_layout()
        if save_fig:
            path = os.path.join(os.getcwd(), "outputs")
            path = os.path.join(path, data_set_name)
            path = os.path.join(path, "result_plots")
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(os.path.join(path, "MYOPIC_capacity_four_plots" + "." + file_type))
        plt.show()


        # Plot with anaerobic digestioon
        cap_anaerobic_endog = pd.read_pickle(os.path.join(df_path, 'capacity_endog.pkl')).loc["anaerobic_digestion"]
        cap_anaerobic_2y = res.get_total("capacity", scenario="scenario_").groupby(["technology"]).sum().loc["anaerobic_digestion"]
        cap_anaerobic_5y = res.get_total("capacity", scenario="scenario_1").groupby(["technology"]).sum().loc["anaerobic_digestion"]
        cap_anaerobic_10y = res.get_total("capacity", scenario="scenario_2").groupby(["technology"]).sum().loc["anaerobic_digestion"]

        hydrogen_output_endog = pd.read_pickle(os.path.join(df_path, 'hydrogen_output_endog.pkl'))
        df_gas_endog = pd.read_pickle(os.path.join(df_path, 'df_gas_endog.pkl'))
        hydrogen_output_2y, df_gas_2y = plot_hydrogen_production(res, scenario="scenario_", save_fig=False,file_type=file_type)
        hydrogen_output_5y, df_gas_5y = plot_hydrogen_production(res, scenario="scenario_1", save_fig=False,file_type=file_type)
        hydrogen_output_10y, df_gas_10y = plot_hydrogen_production(res, scenario="scenario_2", save_fig=False,file_type=file_type)

        production_share_path_endog = convert_tech_output_2_pathway_output(hydrogen_output_endog, df_gas_endog)
        production_share_path_2y = convert_tech_output_2_pathway_output(hydrogen_output_2y, df_gas_2y)
        production_share_path_5y = convert_tech_output_2_pathway_output(hydrogen_output_5y, df_gas_5y)
        production_share_path_10y = convert_tech_output_2_pathway_output(hydrogen_output_10y, df_gas_10y)

        # Plotting the stacked bar chart
        fig, ax = plt.subplots(figsize=(10, 8))
        cap_endog_transposed.plot(kind='bar', stacked=True,ax=ax,position=1.8,alpha=0.6,  width=0.2, color=[tech_colors[tech] for tech in cap_endog_transposed.columns])
        cap_2y_transposed.plot(kind='bar', stacked=True,ax=ax,position=0.8, alpha=0.7, width=0.2,color=[tech_colors[tech] for tech in cap_endog_transposed.columns], label="")
        cap_5y_transposed.plot(kind='bar', stacked=True,ax=ax,position=-0.2,alpha=0.85,  width=0.2, color=[tech_colors[tech] for tech in cap_endog_transposed.columns], label="")
        cap_10y_transposed.plot(kind='bar', stacked=True,ax=ax,position=-1.2, alpha=1.0,width=0.2, color=[tech_colors[tech] for tech in cap_endog_transposed.columns], label="")
        ax.legend(loc="upper right", labels=cap_endog_transposed.columns)
        ax.scatter(cap_anaerobic_endog.index - 0.3, cap_anaerobic_endog, alpha=0.6,color=tech_colors["anaerobic_digestion"], marker='d', label="anaerobic digestion")
        ax.scatter(cap_anaerobic_2y.index - 0.15, cap_anaerobic_2y, color=tech_colors["anaerobic_digestion"],marker='s', label="")
        ax.scatter(cap_anaerobic_5y.index +0.15, cap_anaerobic_5y, color=tech_colors["anaerobic_digestion"],marker='p', label="")
        ax.scatter(cap_anaerobic_10y.index + 0.3, cap_anaerobic_10y, color=tech_colors["anaerobic_digestion"],marker='+', label="")
        ax.set_xlabel("Years")
        ax.set_ylabel("Capacity [GW]")
        ax.set_xlim(-0.5, len(cap_endog.columns))
        ax.set_xticks(range(len(cap_endog.columns)))
        ax.set_xticklabels(xtick_labels, rotation=90)

        ax_right = ax.twinx()
        ax_right.plot(production_share_path_endog["CCS"] * 100, linestyle="-",color="blue", label="CCS share endogenous")
        ax_right.plot(production_share_path_2y["CCS"] * 100, linestyle="--",color="blue", label="CCS share 2y foresight")
        ax_right.plot(production_share_path_5y["CCS"] * 100, linestyle=":",color="blue", label="CCS share 5y foresight")
        ax_right.plot(production_share_path_10y["CCS"] * 100, linestyle="-.",color="blue", label="CCS share 10y foresight")
        ax_right.set_ylabel('Hydrogen production share [%]')
        ax_right.set_ylim(0, 100)
        ax_right.legend(loc="lower right")
        ax.set_xlabel("Years")
        ax.set_ylabel("Capacity [GW]")
        ax.set_xticks(range(len(cap_endog.columns)))
        ax.set_xticklabels(xtick_labels, rotation=90)

        plt.tight_layout()
        if save_fig:
            path = os.path.join(os.getcwd(), "outputs")
            path = os.path.join(path, data_set_name)
            path = os.path.join(path, "result_plots")
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(os.path.join(path, "MYOPIC_capacity_one_plot" + "." + file_type))
        plt.show()


# Non-European Analysis
elif data_set_name.endswith('_non_European'):
    # Cost Evolution
    unit_cost_endog = calculate_unit_cost_over_time(res, carrier="hydrogen", scenario="scenario_")
    unit_cost_non_European = calculate_unit_cost_over_time(res, carrier="hydrogen", scenario="scenario_1")

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    plot_unit_cost_over_time(res, unit_cost_endog, "scenario_", fig, ax[0], plot_style=plot_style, save_fig=False, file_type=None)
    plot_unit_cost_over_time(res, unit_cost_non_European, "scenario_1", fig, ax[1], plot_style=plot_style, save_fig=False, file_type=None)
    # Add non-European vs. European capacities in background
    experience = pd.DataFrame()
    experience["row"] = res.get_df("cum_capacity_row", scenario="scenario_1").unstack().loc[h2_tech].sum()
    experience["eu"] = res.get_total("european_cumulative_capacity", scenario="scenario_1").groupby(["technology"]).sum().loc[h2_tech].sum() - experience["row"]
    ax_right = ax[1].twinx()
    tech_colors = plot_style["tech_colors"]
    experience.plot.bar(stacked=True, ax=ax_right, color=[tech_colors[tech] for tech in tech_colors], alpha=0.1)
    ax_right.set_ylim([0, 10000])
    ax_right.set_ylabel("Cumulative Capacity [GW]")
    ax_right.legend(labels=["ROW","EU"])

    fig.tight_layout()
    if save_fig:
        path = os.path.join(os.getcwd(), "outputs")
        path = os.path.join(path, data_set_name)
        path = os.path.join(path, "result_plots")
        if not os.path.exists(path):
            os.makedirs(path)
        fig.savefig(os.path.join(path, "cost_evolution" + "." + file_type))
    fig.show()

    # Capacity
    for tech_selection in tech_groups:
        plot_style = {"edgecolor" : 'black',
                      "tech_colors": tech_colors_1,
                      "tech_hatches": tech_hatches_1}
        annotation = pd.Series({'Base': 5, 'ROW': 5}).reset_index()
        plot_capacity_multi_bar(res, tech_selection, plot_style=plot_style, annotation=annotation, save_fig=save_fig, file_type=file_type)

    # Extras
    diff_unit_cost = (unit_cost_non_European - unit_cost_endog) / unit_cost_endog * 100

    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(18, 5))
    i = 0
    for tech in diff_unit_cost.index.levels[0]:
        color = tech_colors.get(tech)
        ax[i].bar(diff_unit_cost.columns, diff_unit_cost.loc[tech, "cost"], label=tech, color=tech_colors[tech])
        for axis in ax:
            axis.set_xlabel("Years")
            axis.set_ylabel("Relative cost difference [%]")
            axis.set_ylim(-100, 10)
            axis.set_xticks(range(-1,len(unit_cost_endog.columns)))
            axis.set_xticklabels(xtick_labels, rotation=90)
            axis.legend(loc="lower right")
            axis.axhline(y=0, color='black', linestyle='--', linewidth=0.1)
        i = i + 1
    plt.tight_layout()
    if save_fig:
        path = os.path.join(os.getcwd(), "outputs")
        path = os.path.join(path, data_set_name)
        path = os.path.join(path, "result_plots")
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(os.path.join(path, "NON_EUROPEAN_rel_cost_difference" + "." + file_type))
    plt.show()

    # Get dataframes
    df_non_European = res.get_df("capex_yearly_all_positions", scenario="scenario_1").unstack().groupby(
        ["technology"]).sum()
    df_endogenous = res.get_df("capex_yearly_all_positions", scenario="scenario_").unstack().groupby(
        ["technology"]).sum()

    df_non_European = res.get_total("cost_capex", scenario="scenario_1").groupby(["technology"]).sum()
    df_endogenous = res.get_total("cost_capex", scenario="scenario_").groupby(["technology"]).sum()

    df_non_European_normalized = df_non_European.div(df_non_European.sum(axis=0), axis=1)
    df_endog_normalized = df_endogenous.div(df_endogenous.sum(axis=0), axis=1)

    fig, ax = plt.subplots()
    for tech in df_endogenous.index:
        color = tech_colors.get(tech)
        df_endogenous.loc[tech].plot(kind="bar", stacked=True, color=tech_colors[tech], ax=ax, position=1.1, width=0.3, label=tech)
        df_non_European.loc[tech].plot(kind="bar", stacked=True, color=tech_colors[tech], ax=ax, position=-0.1, width=0.3, label="")
    plt.show()

    fig, ax = plt.subplots()
    for tech in df_endogenous.index:
        color = tech_colors.get(tech)
        df_endog_normalized.loc[tech].plot(kind="bar", stacked=True, alpha=0.7, ax=ax, position=1, width=0.3)
        df_non_European_normalized.loc[tech].plot(kind="bar", stacked=True, alpha=0.7, ax=ax, position=0, width=0.3)
    plt.show()


# Analysis for learning rate variation
elif data_set_name.endswith('_learning_variation'):
    ncols = 3
    nrows = 2
    fig, axis = plt.subplots(nrows, ncols, figsize=(10, 10))
    for i, scenario in enumerate(res.scenarios):
        row = i // ncols
        col = i % ncols
        unit_cost = calculate_unit_cost_over_time(res, carrier="hydrogen", scenario=scenario)
        plot_unit_cost_over_time(res, unit_cost, scenario, fig, axis[row, col], plot_style=plot_style,save_fig=save_fig, file_type=file_type)

    fig.tight_layout()
    if save_fig:
        path = os.path.join(os.getcwd(), "outputs")
        path = os.path.join(path, data_set_name)
        path = os.path.join(path, "result_plots")
        if not os.path.exists(path):
            os.makedirs(path)
        fig.savefig(os.path.join(path, "cost_evolution" + "." + file_type))
    fig.show()

    # Capacity
    for tech_selection in tech_groups:
        annotation = pd.Series({'Base': 5, 'Bio': 5, 'Power': 5, 'CCS': 5, 'SMR':5}).reset_index()
        plot_capacity_multi_bar(res, tech_selection, annotation=annotation, separate_base=True, plot_style=plot_style,save_fig=save_fig, file_type=file_type)

# Analysis for leanring variation extreme
elif data_set_name.endswith('_learning_variation_extreme'):
    ncols = 2
    nrows = 2
    fig, axis = plt.subplots(nrows, ncols, figsize=(10, 10))
    for i, scenario in enumerate(res.scenarios):
        row = i // ncols
        col = i % ncols
        unit_cost = calculate_unit_cost_over_time(res, carrier="hydrogen", scenario=scenario)
        plot_unit_cost_over_time(res, unit_cost, scenario, fig, axis[row, col], plot_style=plot_style,save_fig=save_fig, file_type=file_type)

    fig.tight_layout()
    if save_fig:
        path = os.path.join(os.getcwd(), "outputs")
        path = os.path.join(path, data_set_name)
        path = os.path.join(path, "result_plots")
        if not os.path.exists(path):
            os.makedirs(path)
        fig.savefig(os.path.join(path, "cost_evolution" + "." + file_type))
    fig.show()

    # Capacity
    for tech_selection in tech_groups:
        annotation = pd.Series({'Base': 5, 'Bio': 5, 'Power': 5, 'CCS': 5}).reset_index()
        plot_capacity_multi_bar(res, tech_selection, annotation=annotation, separate_base=True, plot_style=plot_style,
                                save_fig=save_fig, file_type=file_type)

    # Plot costs
    plot_capex_multi_bar(res, plot_style=plot_style, normalize=True, save_fig=save_fig, file_type=file_type)

elif data_set_name.endswith('_single_pathway'):
    # Net present cost of each scenario
    fig, axis = plt.subplots(figsize=(10,8))
    for i,scenario in enumerate(res.scenarios):
        npc = res.get_df("net_present_cost",scenario=scenario).sum()
        axis.bar(i, npc, label=scenario)
    axis.legend()
    fig.show()

    # Capacity of each scenario
    fig, axis = plt.subplots(1,4, figsize=(15,5))
    for i, scenario in enumerate(res.scenarios):
        cap = res.get_total("capacity", scenario=scenario)
        cap_hydrogen = res.extract_reference_carrier(cap, "hydrogen", scenario).groupby(["technology"]).sum()
        cap_hydrogen_transposed = cap_hydrogen.T
        cap_hydrogen_transposed.plot.bar(stacked=True, ax=axis[i])
    fig.tight_layout()
    fig.show()

else:
    # Standard Plots
    ncols = len(res.scenarios)
    nrows = 1
    if ncols > 4:
        ncols = 4
        nrows = len(res.scenarios) // ncols + 1
    fig, axis = plt.subplots(nrows, ncols, figsize=(10, 10))
    for i, scenario in enumerate(res.scenarios):
        unit_cost = calculate_unit_cost_over_time(res, carrier="electricity", scenario=scenario)
        if nrows ==1:
            plot_unit_cost_over_time(res, unit_cost, scenario, fig, axis[i], plot_style=plot_style, save_fig=save_fig,file_type=file_type)
        else:
            row = i // ncols
            col = i % ncols
            plot_unit_cost_over_time(res, unit_cost, scenario, fig, axis[row, col], plot_style=plot_style, save_fig=save_fig,file_type=file_type)

    fig.tight_layout()
    if save_fig:
        path = os.path.join(os.getcwd(), "outputs")
        path = os.path.join(path, data_set_name)
        path = os.path.join(path, "result_plots")
        if not os.path.exists(path):
            os.makedirs(path)
        fig.savefig(os.path.join(path, "cost_evolution" + "." + file_type))
    fig.show()

    # Capacity
    for tech_selection in tech_groups:
        annotation= None
        plot_capacity_multi_bar(res, tech_selection, annotation=annotation ,separate_base=True, plot_style=plot_style, save_fig=save_fig, file_type=file_type)


#######################################################################################################################
# Create custom standard_plots for all scenarios
standard_plots_AX(res, save_fig=save_fig, file_type=file_type)

# Carbon emissions
carbon_out_flow = res.get_total("flow_conversion_output", scenario="scenario_1").groupby(["carrier", "technology"]).sum().loc["carbon"]
import_flow = res.get_total("flow_import", scenario="scenario_1").groupby(["carrier"]).sum()

carbon_intensity = res.get_total("carbon_intensity_carrier", scenario="scenario_1").groupby(["carrier"]).mean()
carbon_emissions = (import_flow*carbon_intensity).sum()


# Plot a map
from geopy.geocoders import Nominatim
# Initialize the geocoder
geolocator = Nominatim(user_agent="myGeocoder")

# Fill dictionary with coordinates
coordinates_nodes = {}

for location in res.results["scenario_"]["system"]["set_nodes"]:
    coordinates_nodes[location] = geolocator.geocode(location)


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



# Plot the Net present cost split
df = pd.DataFrame({'cost_capex_total': res.get_total("cost_capex_total", scenario=scenario),
                   'cost_opex_total': res.get_total("cost_opex_total", scenario=scenario),
                   'cost_carrier_total': res.get_total("cost_carrier_total", scenario=scenario),
                   'cost_carbon_emissions_total': res.get_total("cost_carbon_emissions_total", scenario=scenario)})

df.plot.bar(stacked=True)
plt.xlabel('Years')
plt.ylabel('Costs')
plt.title('Total costs over the years')
plt.show()

scenario = "scenario_1"
save_total(res, scenario=scenario)
year = "13"
target_technologies = ["electrolysis", "SMR", "SMR_CCS", "gasification", "gasification_CCS"]
intermediate_technologies = ["pv_ground", "pv_rooftop","biomethane_conversion", "anaerobic_digestion", "wind_onshore", "wind_offshore"]
title = data_set_name
generate_sankey_diagram(scenario, target_technologies, intermediate_technologies, year, title)


# Variante 1: Nur Hydrogen Tech
df_extact = res.extract_reference_carrier(res.get_df("capacity_addition", scenario=scenario), "hydrogen", scenario).groupby(["technology"]).sum()
# Variante 2: All Tech
df_extact = res.get_df("capacity_addition", scenario=scenario).groupby(["technology"]).sum()
plt.pie(round(df_extact), labels=df_extact.index, autopct='%1.1f%%')
plt.title(f"Shares of Hydrogen Generating Technologies in Capacity Additions over full horizon {scenario}")
plt.show()


df_full = res.get_df("flow_conversion_input", scenario=scenario)
df_extract = res.extract_reference_carrier(df_full, "hydrogen", scenario).groupby(["carrier"]).sum()
try:
    flow_biomethane = df_full.groupby(["carrier"]).sum().loc["biomethane"]
    df_extract["natural_gas"]= df_extract["natural_gas"] - flow_biomethane
    df_extract=pd.concat([df_extract,pd.Series({"biomethane": flow_biomethane})])
except:
    pass
plt.pie(df_extract, labels=df_extract.index, autopct='%1.1f%%')
plt.title("Shares of Hydrogen Generating Carriers over full horizon")
plt.show()

# Hydrogen Production
df_full = res.get_df("flow_conversion_output", scenario=scenario)
df_extract = res.extract_reference_carrier(df_full, "hydrogen", scenario).groupby(["technology", "carrier"]).sum()
filtered_df = df_extract[df_extract.index.get_level_values(1) == 'hydrogen']
plt.pie(filtered_df, labels=filtered_df.index.levels[0], autopct='%1.1f%%')
plt.title("Shares of Production of Hydrogen Generating Technologies over full horizon")
plt.show()

# Electricity Production
df_full = res.get_df("flow_conversion_output", scenario=scenario)
df_extract = res.extract_reference_carrier(df_full, "electricity", scenario).groupby(["technology", "carrier"]).sum()
filtered_df = df_extract[df_extract.index.get_level_values(1) == 'electricity']
plt.pie(filtered_df, labels=filtered_df.index.levels[0], autopct='%1.1f%%')
plt.title("Shares of Production of Electricity Generating Technologies over full horizon")
plt.show()

# Carrier impots
carrier_imports = res.get_df("flow_import", scenario=scenario).groupby(["carrier"]).sum()
plt.pie(carrier_imports, labels=carrier_imports.index, autopct='%1.1f%%')
plt.title("Import of carriers over whole horizon")
plt.show()

# Carbon storage
cap_limit = res.get_df("capacity_limit", scenario=scenario).groupby(["technology", "location"]).max()[res.get_df("capacity_limit", scenario=scenario).groupby(["technology", "location"]).max().index.get_level_values(0) == 'carbon_storage']
cap_access = cap_limit[cap_limit != 0.0]
cap_storage = res.get_df("capacity", scenario=scenario).groupby(["technology", "location"]).max()[res.get_df("capacity", scenario=scenario).groupby(["technology", "location"]).max().index.get_level_values(0) == 'carbon_storage']
cap_storage = cap_storage[cap_storage != 0.0]

storage_load = cap_storage/cap_access* 100

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

# cost of transport technolog
data_total = res.get_total("capex_specific_transport", scenario=scenario).groupby(["technology"]).mean().T
plt.plot(data_total)
plt.title(f"Capex specific of Transport Technologies")
plt.xlabel("Year")
plt.ylabel("Capex")
plt.legend(data_total.columns, loc='center right')
plt.xticks(range(len(data_total.index)))
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
learning_curve_plots(res, carrier="electricity", scenario=scenario, save_fig=True, file_type="svg")
learning_curve_plots(res, carrier="hydrogen", scenario=scenario, save_fig=False, file_type="svg")
learning_curve_plots(res, carrier=None, scenario=scenario)


plt.plot(res.get_total("price_import").groupby(["carrier"]).mean().T)
plt.legend(res.get_total("price_import").groupby(["carrier"]).mean().T.columns, loc='center left')
plt.show()

res.get_total("carbon_emissions_technology").groupby(["technology"]).sum()

plt.plot(res.get_total("carbon_intensity_carrier", scenario="scenario_").groupby(["carrier"]).mean().T)
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

# Capacity Additions ROW
data = res.get_df("cum_capacity_row", scenario="scenario_2").unstack().loc[["SMR","SMR_CCS", "electrolysis", "gasification", "gasification_CCS", "pv_ground", "pv_rooftop", "wind_onshore", "wind_offshore"]]
data_sorted = data.loc[data.sum(axis=1).sort_values(ascending=False).index]
data_sorted.T.plot.bar(stacked=True, width=0.6)
plt.xlabel("Years")
plt.ylabel("Cum Capacity [GW]")
plt.title("Non-European Capacity Additions over the years")
plt.show()

# Net Present Cost
for scenario in res.scenarios:
    res.get_df("net_present_cost", scenario=scenario).sum()
# Check Emissions
res.get_df("net_present_cost").sum()
res.get_df("carbon_emissions_annual").sum()
res.get_df("carbon_emissions_cumulative")


# Demand hydrogen
demand = res.get_total("demand", scenario=scenario).loc["hydrogen"].sum()
plt.bar(demand.index, demand, width=0.7)
plt.show()

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
    TC = alpha / exp * (np.power(u, exp))

    return TC

# total cost for needed capacity for SMR
# get initial cost
tech = "SMR"
capacity_type = "power"
c_initial = res.get_df("total_cost_pwa_initial_unit_cost", scenario=scenario).loc[tech, capacity_type]
# get initial capacity
q_initial = res.get_df("global_initial_capacity", scenario=scenario).loc[tech]
# get learning rate
learning_rate = res.get_df("learning_rate", scenario=scenario).loc[tech]
required_capacity_smr = (demand - res.get_df("capacity_existing").sum()) * res.get_df("conversion_factor", scenario=scenario).loc["SMR"].mean()
inv_cost_smr= fun_total_cost(demand, c_initial, q_initial, learning_rate) - fun_total_cost(q_initial, c_initial, q_initial, learning_rate)

# total cost for needed capacity for electrolysis
tech = "electrolysis"
capacity_type = "power"
c_initial = res.get_df("total_cost_pwa_initial_unit_cost", scenario=scenario).loc[tech, capacity_type]
# get initial capacity
q_initial = res.get_df("global_initial_capacity", scenario=scenario).loc[tech]
# get learning rate
learning_rate = res.get_df("learning_rate", scenario=scenario).loc[tech]
required_capacity = demand * res.get_df("conversion_factor", scenario=scenario).loc["electrolysis"].mean()
inv_cost_ele= fun_total_cost(required_capacity, c_initial, q_initial, learning_rate) - fun_total_cost(q_initial, c_initial, q_initial, learning_rate)

tech = "pv_ground"
capacity_type = "power"
c_initial = res.get_df("total_cost_pwa_initial_unit_cost", scenario=scenario).loc[tech, capacity_type]
# get initial capacity
q_initial = res.get_df("global_initial_capacity", scenario=scenario).loc[tech]
# get learning rate
learning_rate = res.get_df("learning_rate", scenario=scenario).loc[tech]


inv_cost_pv = fun_total_cost(required_capacity, c_initial, q_initial, learning_rate) - fun_total_cost(q_initial, c_initial, q_initial, learning_rate)


# LCOH calculation
year = 13
tech = "electrolysis"
scenario = "scenario_1"
capital_cost = res.get_total("capex_specific_conversion", scenario=scenario).groupby(["technology"]).mean().loc[tech, year]
discount_rate = res.get_df("discount_rate", scenario=scenario)
lifetime = res.get_df("lifetime", scenario=scenario).loc[tech]
f_h = pow((1+discount_rate),lifetime)/(pow((1+discount_rate),lifetime)-1)
demand = res.get_df("demand", scenario=scenario).loc["hydrogen"].sum()


scenario = "scenario_1"
capex = res.get_df("capex_yearly_all_positions", scenario=scenario).unstack().groupby(["technology"]).sum()
opex = res.get_total("opex_yearly", scenario=scenario).groupby(["technology"]).sum()

carrier = "hydrogen"
data_total = res.get_total("flow_conversion_output", scenario=scenario)
data_h2 = res.extract_reference_carrier(data_total, carrier, scenario)
produced_h2 = data_h2.groupby(["technology"]).sum()

lcoh = ((capex + opex)/produced_h2).dropna()

plt.plot(lcoh.T)
plt.xlabel("Years")
plt.ylabel("LCOH [Euro/kW H2]")
plt.legend(lcoh.T.columns, loc='center left')
plt.show()

plt.plot(capex.loc[["SMR", "SMR_CCS", "electrolysis", "gasification", "gasification_CCS"]].T)
plt.xlabel("Years")
plt.ylabel("Capex [Euro/kW H2]")
plt.legend(capex.loc[["SMR", "SMR_CCS", "electrolysis", "gasification", "gasification_CCS"]].T.columns, loc='center left')
plt.show()

plt.plot(opex.loc[["SMR", "SMR_CCS", "electrolysis", "gasification", "gasification_CCS"]].T)
plt.xlabel("Years")
plt.ylabel("Opex [Euro/kW H2]")
plt.legend(opex.loc[["SMR", "SMR_CCS", "electrolysis", "gasification", "gasification_CCS"]].T.columns, loc='center left')
plt.show()

plt.plot(produced_h2.loc[["SMR", "SMR_CCS", "electrolysis", "gasification", "gasification_CCS"]].T)
plt.xlabel("Years")
plt.ylabel("Produced H2 [kW]")
plt.legend(produced_h2.loc[["SMR", "SMR_CCS", "electrolysis", "gasification", "gasification_CCS"]].T.columns, loc='center left')
plt.show()



lcoh.T.mean()

print(f"Cost of electrolysis and pv_ground if all demand supplied by electricity: {inv_cost_pv + inv_cost_ele}")
print(f"Cost of SMR if all demand supplied by natural gas: {inv_cost_smr}")

# Look at investments or look at hydrogen production
# a) hydrogen production
flow_conversion_output = res.extract_reference_carrier(res.get_total("flow_conversion_output", scenario=scenario), "hydrogen", scenario).groupby(["technology","carrier"]).sum()
hydrogen_output = flow_conversion_output[flow_conversion_output.index.get_level_values(1) == "hydrogen"]
hydrogen_output.T.plot.bar(stacked=True, width=0.6)
plt.show()

# b) investments
res.plot("capex_yearly", yearly=True, plot_strings={"title": "Total Capex", "ylabel": "Capex"},
                 save_fig=save_fig, file_type=file_type, scenario=scenario)

# Plot cost capex
res.get_df("capex_yearly_all_positions", scenario="scenario_1").unstack().T.plot.bar(stacked=True, width=0.5)
plt.xlabel('Years')
plt.ylabel('Cost Capex [kEuro]')
plt.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))
plt.tight_layout()
plt.show()

plt.pie(res.get_df("cost_capex", scenario="scenario_").groupby(["technology"]).sum().abs().T,  labels=res.get_df("cost_capex", scenario="scenario_1").groupby(["technology"]).sum().abs().index, autopct='%1.1f%%')
plt.show()

# Compare exog vs. endog
component = "capacity_addition"
exog = res.get_df(component, scenario="scenario_").groupby(["technology"]).sum()
endog = res.get_df(component, scenario="scenario_1").groupby(["technology"]).sum()

categories = exog.index    # Replace 'your_column' with the column containing values for df2

fig, ax = plt.subplots()
# Plot bars to the left
ax.barh(categories, -exog, color='blue', label='exog')
# Plot bars to the right
ax.barh(categories, endog, color='red', label='endog')
# Add a vertical line in the middle
ax.axvline(x=0, color='black', linestyle='--')
# Adding labels and title
ax.set_xlabel('Values')
ax.set_ylabel('Categories')
ax.set_title(f'Comparison of Values for {component}')
ax.legend()
plt.tight_layout()
# Show plot
plt.show()


# Maximaler Ausbau der Technologien
max_cap = (res.get_total("capacity", scenario="scenario_1")/res.get_total("capacity_limit", scenario="scenario_1")).max(axis=1)
not_zero = max_cap[max_cap != 0]

# STATEMENT 1: Early build out of emerging technologies (SMR CCS and gasification CCS), build out of infrastructure
component = "capacity"
cap_exog = res.get_total(component, scenario="scenario_").groupby(["technology"]).sum()
cap_endog = res.get_total(component, scenario="scenario_1").groupby(["technology"]).sum()

# Capacity addition for CCS infrastructure
cap_CCS_exog = cap_exog.loc[["carbon_storage", "carbon_pipeline"]]
cap_CCS_endog = cap_endog.loc[["carbon_storage", "carbon_pipeline"]]

# Plot of CCS capacity
fig, ax = plt.subplots()
ax.plot(cap_CCS_exog.T, label= f"Exog {cap_CCS_exog.index.values}")
ax.plot(cap_CCS_endog.T, label= f"Endog {cap_CCS_exog.index.values}")
ax.legend()
fig.tight_layout()
plt.show()

# Investments
# Option 1: Emerging vs mature
# Exlcuded components: capex_yearly
# Problem with capacity: Not the same unit
# Only include hydrogen poducing technology
component = "capacity_addition"

# Load data
cost_capex_exog = res.get_total(component, scenario="scenario_").groupby(["technology"]).sum()
if component == "capex_yearly":
    cost_capex_endog = res.get_df( "capex_yearly_all_positions", scenario="scenario_1").unstack().groupby(["technology"]).sum()
else:
    cost_capex_endog = res.get_total(component, scenario="scenario_1").groupby(["technology"]).sum()


# Classification 1 a: all tech
emerging_tech = ["SMR_CCS", "gasification_CCS", "carbon_storage", "carbon_removal", "electrolysis", "pv_ground", "pv_rooftop", "wind_offshore"]
mature_tech = ["SMR", "gasification", "wind_onshore", "anaerobic_digestion"]
supporting_tech = ["hydrogen_pipeline", "dry_biomass_truck", "carbon_pipeline"]

# Emerging vs mature
investments_exog = pd.DataFrame()
investments_exog["mature"] = 100* cost_capex_exog.loc[mature_tech].sum() / cost_capex_exog.drop(["biomethane_conversion", "biomethane_transport"]).sum()
investments_exog["emerging"] = 100* cost_capex_exog.loc[emerging_tech].sum() / cost_capex_exog.drop(["biomethane_conversion", "biomethane_transport"]).sum()
investments_exog["supporting"] = 100* cost_capex_exog.loc[supporting_tech].sum() / cost_capex_exog.drop(["biomethane_conversion", "biomethane_transport"]).sum()

investments_endog = pd.DataFrame()
investments_endog["mature"] = 100* cost_capex_endog.loc[mature_tech].sum() / cost_capex_endog.drop(["biomethane_conversion", "biomethane_transport"]).sum()
investments_endog["emerging"] = 100* cost_capex_endog.loc[emerging_tech].sum() / cost_capex_endog.drop(["biomethane_conversion", "biomethane_transport"]).sum()
investments_endog["supporting"] = 100* cost_capex_endog.loc[supporting_tech].sum() / cost_capex_endog.drop(["biomethane_conversion", "biomethane_transport"]).sum()


investments_exog.plot.bar(stacked=True, width=0.7)
plt.title(f"{component} Comparison Exog vs. endog")
plt.show()

investments_endog.plot.bar(stacked=True, width=0.7)
plt.title(f"{component} Comparison Exog vs. endog")
plt.show()

fig, ax = plt.subplots()
ax.plot(investments_exog["emerging"]+investments_exog["supporting"], label="No Learning")
ax.plot(investments_endog["emerging"]+investments_endog["supporting"], label="Technology Learning")
plt.title(f"{component} Comparison Exog vs. endog")
ax.set_xticks(range(len(investments_exog)))
ax.set_xlabel("Year")
ax.set_ylabel("Capacity share of emerging and supporting technologies [%]")
ax.set_ylim([0, 100])
plt.legend()
plt.show()

# Classification 1 b: only h2 tech
emerging_tech = ["SMR_CCS", "gasification_CCS", "electrolysis"]
mature_tech = ["SMR", "gasification"]
h2_tech = ["SMR_CCS", "gasification_CCS", "electrolysis", "SMR", "gasification"]

# Emerging vs mature
investments_exog = pd.DataFrame()
investments_exog["mature"] = 100* cost_capex_exog.loc[mature_tech].sum() / cost_capex_exog.loc[emerging_tech + mature_tech].sum()
investments_exog["emerging"] = 100* cost_capex_exog.loc[emerging_tech].sum() / cost_capex_exog.loc[emerging_tech + mature_tech].sum()

investments_endog = pd.DataFrame()
investments_endog["mature"] = 100* cost_capex_endog.loc[mature_tech].sum() / cost_capex_endog.loc[emerging_tech + mature_tech].sum()
investments_endog["emerging"] = 100* cost_capex_endog.loc[emerging_tech].sum() / cost_capex_endog.loc[emerging_tech + mature_tech].sum()

investments_exog.plot.bar(stacked=True, width=0.7)
plt.title(f"{component} Comparison Exog vs. endog")
plt.show()

investments_endog.plot.bar(stacked=True, width=0.7)
plt.title(f"{component} Comparison Exog vs. endog")
plt.show()

fig, ax = plt.subplots()
ax.plot(investments_exog["emerging"], label="No Learning")
ax.plot(investments_endog["emerging"], label="Technology Learning")
plt.title(f"{component.capitalize()} Comparison Exog vs. endog")
ax.set_xticks(range(len(investments_exog)))
ax.set_xlabel("Year")
ax.set_ylabel("Capacity share of emerging technologies [%]")
ax.set_ylim([0, 100])
plt.legend()
plt.show()

# No Classification
investments_exog = pd.DataFrame()
investments_exog = 100 * cost_capex_exog.loc[mature_tech+emerging_tech] / cost_capex_exog.loc[mature_tech+emerging_tech].sum()

investments_endog = pd.DataFrame()
investments_endog = 100 * cost_capex_endog.loc[mature_tech+emerging_tech] / cost_capex_endog.loc[mature_tech+emerging_tech].sum()

investments_exog.T.plot.bar(stacked=True, width=0.7)
plt.title(f"{component} exogenous")
plt.show()



# Option 2: Decarbonization pathways
biomethane_path = ["anaerobic_digestion"]
CCS_path = ["SMR_CCS", "gasification_CCS", "carbon_storage", "carbon_removal", "carbon_pipeline"]
electrification_path = ["electrolysis", "pv_ground", "pv_rooftop", "wind_offshore", "wind_onshore", "hydrogen_pipeline"]
fossil_path = ["SMR", "gasification", "dry_biomass_truck"]

# Emerging vs mature
investments_exog = pd.DataFrame()
investments_exog["biomethane"] = 100* cost_capex_exog.loc[biomethane_path].sum() / cost_capex_exog.drop(["biomethane_conversion", "biomethane_transport"]).sum()
investments_exog["CCS"] = 100* cost_capex_exog.loc[CCS_path].sum() / cost_capex_exog.drop(["biomethane_conversion", "biomethane_transport"]).sum()
investments_exog["electrification"] = 100* cost_capex_exog.loc[electrification_path].sum() / cost_capex_exog.drop(["biomethane_conversion", "biomethane_transport"]).sum()
investments_exog["fossil"] = 100* cost_capex_exog.loc[fossil_path].sum() / cost_capex_exog.drop(["biomethane_conversion", "biomethane_transport"]).sum()

investments_endog = pd.DataFrame()
investments_endog["biomethane"] = 100* cost_capex_endog.loc[biomethane_path].sum() / cost_capex_endog.drop(["biomethane_conversion", "biomethane_transport"]).sum()
investments_endog["CCS"] = 100* cost_capex_endog.loc[CCS_path].sum() / cost_capex_endog.drop(["biomethane_conversion", "biomethane_transport"]).sum()
investments_endog["electrification"] = 100* cost_capex_endog.loc[electrification_path].sum() / cost_capex_endog.drop(["biomethane_conversion", "biomethane_transport"]).sum()
investments_endog["fossil"] = 100* cost_capex_endog.loc[fossil_path].sum() / cost_capex_endog.drop(["biomethane_conversion", "biomethane_transport"]).sum()

investments_exog.plot.bar(stacked=True, width=0.7)
plt.show()

investments_endog.plot.bar(stacked=True, width=0.7)
plt.show()

fig, ax = plt.subplots()
ax.plot(investments_exog["fossil"], label="Exog Fossil")
ax.plot(investments_endog["fossil"], label="Endog Fossil")
plt.show()

# Compare exog vs. endog


component = "capacity_addition"
year = 1
exog = res.get_total(component, scenario="scenario_").groupby(["technology"]).sum()[year]
endog = res.get_total(component, scenario="scenario_1").groupby(["technology"]).sum()[year]    # Replace 'your_column' with the column containing values for df2

# Define colors for each technology
tech_colors = {
    'SMR': 'darkgrey',
    'gasification': 'brown',
    'electrolysis': 'green',
    'gasification_CCS': 'darkblue',
    'SMR_CCS': 'lightblue',
    'biomethane': 'lightgreen'
}

fig, ax = plt.subplots()
# Plot bars to the left
for tech in mature_tech:
    ax.barh("mature", -exog[tech], color=tech_colors[tech], label=tech)
    ax.barh("mature", endog[tech], color=tech_colors[tech])
for tech in emerging_tech:
    ax.barh("emerging", -exog[tech], color=tech_colors[tech], label=tech)
    ax.barh("emerging", endog[tech], color=tech_colors[tech])

# Add a vertical line in the middle
ax.axvline(x=0, color='black', linestyle='--')
# Adding labels and title
ax.set_xlabel('Capacity Addition [GW]')
ax.set_ylabel('Categories')
ax.set_title(f'Comparison of Values for {component}')
ax.set_xlim([-1.2, 1.2])
ax.legend()
plt.tight_layout()
# Show plot
plt.show()



# STATEMENT 2: Produced hydrogen
flow_conversion_output_exog = res.get_total("flow_conversion_output", scenario="scenario_").groupby(["technology","carrier"]).sum()
flow_conversion_output_endog = res.get_total("flow_conversion_output", scenario="scenario_1").groupby(["technology","carrier"]).sum()

hydrogen_output_exog = flow_conversion_output_exog[flow_conversion_output_exog.index.get_level_values(1) == "hydrogen"]
hydrogen_output_endog = flow_conversion_output_endog[flow_conversion_output_endog.index.get_level_values(1) == "hydrogen"]


# Endogenous case
# Convert to numpy arrays for easier manipulation
y1 = np.array(hydrogen_output_endog.loc["SMR"])
y2 = np.array(hydrogen_output_endog.loc["SMR_CCS"])
y3 = np.array(hydrogen_output_endog.loc["gasification"])
y4 = np.array(hydrogen_output_endog.loc["gasification_CCS"])
y5 = np.array(hydrogen_output_endog.loc["electrolysis"])

# Accumulate the y-values to stack the lines
y_stack = np.vstack([y1, y2, y3, y4, y5])
y_stack = np.cumsum(y_stack, axis=0)

x = hydrogen_output_endog.columns
# Plot the stacked lines
plt.plot(x, y_stack[0], color='darkgrey', label='SMR')
plt.plot(x, y_stack[1], color='lightblue', label='SMR_CCS')
plt.plot(x, y_stack[2], color='brown', label='gasification')
plt.plot(x, y_stack[3], color='darkblue', label='gasification_CCS')
plt.plot(x, y_stack[4], color='green', label='electrolysis')

# Fill the area between lines
plt.fill_between(x, 0, y_stack[0], color='darkgrey', alpha=0.3)
plt.fill_between(x, y_stack[0], y_stack[1], color='lightblue', alpha=0.3)
plt.fill_between(x, y_stack[1], y_stack[2], color='brown', alpha=0.3)
plt.fill_between(x, y_stack[2], y_stack[3], color='darkblue', alpha=0.3)
plt.fill_between(x, y_stack[3], y_stack[4], color='green', alpha=0.3)

# Add labels, title, legend, etc.
plt.xlabel('Year')
plt.ylabel('Hydrogen Production [GW]')
plt.title('Hydrogen Production of Technologies')
plt.xticks(x)
plt.legend()

# Show plot
plt.show()


# Exogenous case
# Convert to numpy arrays for easier manipulation
y1 = np.array(hydrogen_output_exog.loc["SMR"])
y2 = np.array(hydrogen_output_exog.loc["SMR_CCS"])
y3 = np.array(hydrogen_output_exog.loc["gasification"])
y4 = np.array(hydrogen_output_exog.loc["gasification_CCS"])
y5 = np.array(hydrogen_output_exog.loc["electrolysis"])

# Accumulate the y-values to stack the lines
y_stack = np.vstack([y1, y2, y3, y4, y5])
y_stack = np.cumsum(y_stack, axis=0)

x = hydrogen_output_exog.columns
# Plot the stacked lines
plt.plot(x, y_stack[0], color='darkgrey', label='SMR')
plt.plot(x, y_stack[1], color='lightblue', label='SMR_CCS')
plt.plot(x, y_stack[2], color='brown', label='gasification')
plt.plot(x, y_stack[3], color='darkblue', label='gasification_CCS')
plt.plot(x, y_stack[4], color='green', label='electrolysis')

# Fill the area between lines
plt.fill_between(x, 0, y_stack[0], color='darkgrey', alpha=0.3)
plt.fill_between(x, y_stack[0], y_stack[1], color='lightblue', alpha=0.3)
plt.fill_between(x, y_stack[1], y_stack[2], color='brown', alpha=0.3)
plt.fill_between(x, y_stack[2], y_stack[3], color='darkblue', alpha=0.3)
plt.fill_between(x, y_stack[3], y_stack[4], color='green', alpha=0.3)

# Add labels, title, legend, etc.
plt.xlabel('Year')
plt.ylabel('Hydrogen Production [GW]')
plt.title('Hydrogen Production of Technologies')
plt.xticks(x)
plt.legend()

# Show plot
plt.show()


# For plot with myopic foresight
component = "capacity_addition"
for scenario in res.scenarios:
    res.get_total(component, scenario=scenario).groupby(["technology"]).sum().loc[h2_tech].T.plot.bar(stacked=True, width=0.5)
    plt.title(f"Capacity of Hydrogen Technologies {scenario}")
    plt.show()
    res.get_total("carbon_emissions_annual", scenario=scenario).plot.bar(stacked=True, width=0.5)
    plt.title(f"Carbon Emissions {scenario}")
    plt.show()
    res.get_total("cost_carrier", scenario=scenario).groupby(["carrier"]).sum().T.plot.bar(stacked=True,
                                                                                                      width=0.5)
    plt.title(f"Capacity of Hydrogen Technologies {scenario}")
    plt.show()


