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


def exogenous(config, folder_path):
    # run the test
    data_set_name = "20240118_Hydrogen_exog"
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, data_set_name))
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    # compare_variables_results(data_set_name, res, folder_path)
    # test functions get_total() and get_full_ts()
    # check_get_total_get_full_ts(res, discount_to_first_step=False)

    return res

def endogenous(config, folder_path):
    # run the test
    data_set_name = "20240118_Hydrogen_endog"
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, data_set_name))
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    # compare_variables_results(data_set_name, res, folder_path)
    # test functions get_total() and get_full_ts()
    # check_get_total_get_full_ts(res, discount_to_first_step=False)

    return res


if __name__ == "__main__":
    from config import config
    config.solver["keep_files"] = False
    folder_path = os.path.dirname(__file__)
    # I. Run the two models
    res_exog = exogenous(config, folder_path)
    res_endog = endogenous(config, folder_path)


    # II. Plot the results individually
    # II.a Exogenous
    # empty_plot_with_text("Exogenous Model", background_color='lightgray')
    # res_exog.plot(component="capacity", tech_type="conversion")
    # res_exog.plot(component="capacity_addition", tech_type="conversion")
    # res_exog.plot(component="cost_capex")
    # res_exog.plot(component="capex_yearly")
    # res_exog.plot(component="net_present_cost")
    #
    # # PRE-CHECK: Random variabes i like to look at
    # res_exog.get_df("capacity_addition").groupby(['technology', 'capacity_type', 'year']).sum().reset_index()
    #
    # # II.b Endogenous
    # empty_plot_with_text("Endogenous Model", background_color='lightblue')
    #
    # ################################ Plot some variables for visual validation ################################
    # res_endog.plot(component="capacity", tech_type="conversion")
    # res_endog.plot(component="capacity_addition", tech_type="conversion")
    # res_endog.plot(component="capacity_addition")
    # res_endog.plot(component="cost_capex")
    # res_endog.plot(component="capex_yearly_all_positions")
    # res_endog.plot(component="total_cost_pwa_global_cost")
    # res_endog.plot(component="net_present_cost")
    #
    # ################################ Calculations for closer validation ################################
    # # Read Variable results
    # var_segment_position = res_endog.get_df("total_cost_pwa_cum_capacity_segment_position")
    # var_segment_selection = res_endog.get_df("total_cost_pwa_segment_selection")
    # var_total_global_cost = res_endog.get_df("total_cost_pwa_global_cost")
    # var_capacity = res_endog.get_df("capacity")
    # var_capacity_addition = res_endog.get_df("capacity_addition")
    # var_cost_capex = res_endog.get_df("cost_capex")
    # var_global_capacity = res_endog.get_df("global_cumulative_capacity")
    # var_demand = res_endog.get_df("demand")
    #
    # # Read Parameters
    # par_slope = res_endog.get_df("total_cost_pwa_slope")
    # par_intersect = res_endog.get_df("total_cost_pwa_intersect")
    # par_global_share = res_endog.get_df("global_share_factor")
    # par_total_cost_pwa_points_lower_bound = res_endog.get_df("total_cost_pwa_points_lower_bound")
    # par_total_cost_pwa_points_upper_bound = res_endog.get_df("total_cost_pwa_points_upper_bound")
    # par_total_cost_pwa_TC_upper_bound = res_endog.get_df("total_cost_pwa_TC_upper_bound")
    # par_total_cost_pwa_TC_lower_bound = res_endog.get_df("total_cost_pwa_TC_lower_bound")
    # par_total_cost_pwa_initial_global_cost = res_endog.get_df("total_cost_pwa_initial_global_cost")
    # par_capacity_existing = res_endog.get_df("capacity_existing")
    # par_global_initial_capacity = res_endog.get_df("global_initial_capacity")
    #
    # # PRE-CHECK: Random variabes i like to look at
    # var_capacity_addition.groupby(['technology', 'capacity_type', 'year']).sum().reset_index()
    # var_segment_position.groupby(['set_technologies', 'set_capacity_types', 'set_time_steps_yearly']).sum().reset_index()
    #
    # # CHECK 1:  Compare it with the total cost obtained
    # # Calculate the PWA result of the total cost function
    # calc_total_cost = (var_segment_selection*par_intersect + var_segment_position*par_slope).groupby(level=[0,1,3]).sum()
    # diff_total_cost = (var_total_global_cost - calc_total_cost.rename_axis(index={'set_time_steps_yearly': 'year'})).sum()
    # print(f"The difference between the variable total cost and the calculated total cost is {round(diff_total_cost,4)}.")
    #
    #
    # # CHECK 2: See if results on total cost function
    # # for each technology and each capacity type
    # for tech in var_capacity.index.get_level_values("technology").unique():
    #     for capacity_type in var_capacity.loc[tech].index.get_level_values("capacity_type").unique():
    #         # get the global share factor
    #         gsf = par_global_share.loc[tech]
    #
    #         interpolated_q = res_endog.get_df("total_cost_pwa_points_lower_bound").loc[tech, capacity_type, :].values
    #         interpolated_q = np.append(interpolated_q, res_endog.get_df("total_cost_pwa_points_upper_bound").loc[tech, capacity_type, :].values[-1])
    #
    #         interpolated_TC = res_endog.get_df("total_cost_pwa_TC_lower_bound").loc[tech, capacity_type, :].values
    #         interpolated_TC = np.append(interpolated_TC, res_endog.get_df("total_cost_pwa_TC_upper_bound").loc[tech, capacity_type, :].values[-1])
    #
    #
    #         res_capacity = (1/gsf)*var_capacity.groupby(level=[0,1,3]).sum().loc[tech, capacity_type, :].values
    #         res_TC = var_total_global_cost.loc[tech, capacity_type, :]
    #
    #         initial_capacity = (1/gsf)*par_global_initial_capacity.loc[tech]
    #
    #         # plot the total cost function
    #         fig, ax = plt.subplots()
    #         ax.plot(interpolated_q, interpolated_TC, label=f'PWA: {tech}', color='red')
    #         ax.scatter(interpolated_q, interpolated_TC, color='red')
    #         ax.scatter(res_capacity, res_TC, label=f'Model Results {tech}', color='blue')
    #         ax.scatter(initial_capacity, par_total_cost_pwa_initial_global_cost.loc[tech], label=f'Initial Capacity {tech}', color='green')
    #         ax.legend()
    #         plt.show()
    #
    # print("Plot of Total Cost calculations on the curve.")
    #
    #
    # # CHECK 3:  Check if cost capex equal to differenc between total cost in each step
    # # for each technology and each capacity type
    # for tech in var_capacity.index.get_level_values("technology").unique():
    #     for capacity_type in var_capacity.loc[tech].index.get_level_values("capacity_type").unique():
    #         calc_cost_capex = []
    #         for year in var_capacity.index.get_level_values("year").unique():
    #             if year==0:
    #                 calc_cost_capex.append(var_total_global_cost.loc[tech, :, year] - par_total_cost_pwa_initial_global_cost.loc[tech])
    #             else:
    #                 calc_cost_capex.append(var_total_global_cost.loc[tech, :,year] - var_total_global_cost.loc[tech,:,year-1])
    #
    #         diff_calc_cost_capex = (pd.concat(calc_cost_capex).values - var_cost_capex.loc[tech, capacity_type].values).sum().round(4)
    #         print(f"The difference between the calculated cost capex and the variable cost capex for"
    #               f" {tech}-{capacity_type} is {diff_calc_cost_capex}.")
    #
    #
    # # CHECK 4: Check if capacity addition
    # for tech in var_capacity.index.get_level_values("technology").unique():
    #     for capacity_type in var_capacity.loc[tech].index.get_level_values("capacity_type").unique():
    #         calc_capacity_addition = []
    #         for year in var_capacity.index.get_level_values("year").unique():
    #             if year==0:
    #                 calc_capacity_addition.append(1/par_global_share.loc[tech]*(var_total_global_cost.loc[tech, :, year] - par_capacity_existing.loc[tech, capacity_type].sum()))
    #             else:
    #                 calc_capacity_addition.append(1/par_global_share.loc[tech]*(var_global_capacity.loc[tech, :,year] - var_global_capacity.loc[tech,:,year-1]))
    #
    #         diff_calc_capacity_addition = pd.concat(calc_capacity_addition).values - var_capacity_addition.loc[tech, capacity_type].groupby(level=[1]).sum().values.round(4)
    #         print(f"The difference between the calculated capacity addition and the variable capacity addition for"
    #               f" {tech}-{capacity_type} is {diff_calc_cost_capex}.")
    #
    #
    #
    # # CHECK 5: Check if capacity meets demand
    # for carrier in var_demand.index.get_level_values("carrier").unique():
    #     for year in var_demand.index.get_level_values("time_operation").unique(): # will be an issue when intra-year demand
    #         calc_cum_capacity = 1/par_global_share[0]*var_global_capacity.loc[:, :, year].sum() # simplified: assuming all techs supply this one carrier and global share equal for both = 1
    #         diff_demand = (var_demand.loc[carrier, :, year].sum() - calc_cum_capacity).round(4)
    #
    #         print(f"The difference between the demand and installed capacity for year {year} and carrier {carrier} is {diff_demand}.")


    # CHECK 6: Plot all learning curves


    # CHECK 7: Plot all cost curves of all technologies over the years

    print("Done with result analysis")
