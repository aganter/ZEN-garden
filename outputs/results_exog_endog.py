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
def check2(res, scenario=None):
    for tech in res.get_df("capacity", scenario=scenario).index.get_level_values("technology").unique():
        for capacity_type in res.get_df("capacity", scenario=scenario).loc[tech].index.get_level_values("capacity_type").unique():
            # get the global share factor
            gsf = res.get_df("global_share_factor", scenario=scenario).loc[tech]

            interpolated_q = res.get_df("total_cost_pwa_points_lower_bound", scenario=scenario).loc[tech, capacity_type, :].values
            interpolated_q = np.append(interpolated_q, res.get_df("total_cost_pwa_points_upper_bound", scenario=scenario).loc[tech, capacity_type, :].values[-1])

            interpolated_TC = res.get_df("total_cost_pwa_TC_lower_bound", scenario=scenario).loc[tech, capacity_type, :].values
            interpolated_TC = np.append(interpolated_TC, res.get_df("total_cost_pwa_TC_upper_bound", scenario=scenario).loc[tech, capacity_type, :].values[-1])


            res_capacity = (1/gsf)*res.get_df("capacity", scenario=scenario).groupby(level=[0,1,3]).sum().loc[tech, capacity_type, :].values
            res_TC = res.get_df("total_cost_pwa_global_cost", scenario=scenario).loc[tech, capacity_type, :]

            initial_capacity = (1/gsf)*res.get_df("global_initial_capacity", scenario=scenario).loc[tech]

            # plot the total cost function
            plt.plot(interpolated_q, interpolated_TC, label=f'PWA: {tech}', color='red')
            plt.scatter(interpolated_q, interpolated_TC, color='red')
            plt.scatter(res_capacity, res_TC, label=f'Model Results {tech}', color='blue')
            plt.scatter(initial_capacity, res.get_df("total_cost_pwa_initial_global_cost", scenario=scenario).loc[tech], label=f'Initial Capacity {tech}', color='green')
            plt.legend()
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



# CHECK 7: Plot all cost curves of all technologies over the years

# PLOT1: STANDARD PLOTS: Plot all standard plots for all scenarios
def standard_plots_AX(res, scenario=None, save_fig=False, file_type=None):
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

        res.plot("cost_capex_total", yearly=True, plot_strings={"title": "Total Capex", "ylabel": "Capex"},
                 save_fig=save_fig, file_type=file_type, scenario=scenario)
        res.plot("cost_carrier", yearly=True, plot_strings={"title": "Carrier Cost", "ylabel": "Cost"},
                 save_fig=save_fig, file_type=file_type, scenario=scenario)

# PLOT2: LEARNING CURVES: Plot all learning curves
def learning_curve_plots(res, scenario=None):
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

    for tech in res.get_df("capacity", scenario=scenario).index.get_level_values("technology").unique():
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

    for tech in res.get_df("capacity", scenario=scenario).index.get_level_values("technology").unique():
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
            # plt.ylim(0, 0.3)
            # plt.xlim(0,150)

    plt.show()

############################################## Result anaylsis ##############################################

# I. Read the results of the two models
folder_path = os.path.dirname(__file__)
data_set_name = "20240202_Hydrogen_8760_debug_endog"

res = Results(os.path.join(folder_path, data_set_name))
# res_scenario1 = Results(os.path.join(folder_path, data_set_name), scenarios="1")

save_fig = True
file_type = "png"

# Create custom standard_plots for all scenarios
standard_plots_AX(res, save_fig=save_fig, file_type=file_type)

# Create individual plots for scenarios
learning_curve_plots(res, scenario="scenario_2")

# Do indivudal checks for the scenarios
check1(res, scenario="scenario_")
check2(res, scenario="scenario_")
check2(res, scenario="scenario_1")
check2(res, scenario="scenario_2")
check1(res, scenario="scenario_2")

check4(res, scenario="scenario_2")
check5(res, scenario="scenario_2")



print("Done with result analysis")



