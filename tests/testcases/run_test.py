"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      September-2022
Authors:      Janis Fluri (janis.fluri@id.ethz.ch)
              Alissa Ganter (aganter@ethz.ch)
              Davide Tonelli (davidetonelli@outlook.com)
              Jacob Mannhardt (jmannhardt@ethz.ch)
Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:  Compilation  of the optimization problem.
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
            failed_variables[data_row["variable_name"]][data_row["index"]] = {"computedValue": variable_value,
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


def check_get_total_get_full_ts(results: Results):
    """
    Tests the functionality of the Results methods get_total() and get_full_ts()

    :param results: Results instance of testcase function has been called from
    """
    test_variables = ["demand", "capacity"]
    for test_variable in test_variables:
        df = results.get_df(test_variable)
        df_total = results.get_total(test_variable)
        df_full_ts = results.get_full_ts(test_variable)
        scenario = None
        if results.has_scenarios:
            scenario = results.scenarios[0]
            df = df[scenario]
            df_total = df_total.loc[scenario]
            df_full_ts = df_full_ts.loc[scenario]
        component_name, component_data = results._get_component_data(test_variable, scenario=scenario)
        ts_type = results._get_ts_type(component_data, component_name)

        # test get_total()
        # test shape of dataframe
        if ts_type == "operational":
            if results.results["system"]["conduct_time_series_aggregation"]:
                assert df.size == df_total.shape[0] * results.results["system"]["optimized_years"] * results.results["system"][
                    "aggregated_time_steps_per_year"]
            else:
                assert df.size == df_total.shape[0] * results.results["system"]["optimized_years"] * results.results["system"][
                    "unaggregated_time_steps_per_year"]
            assert df_total.columns.size == results.results["system"]["optimized_years"]
        elif ts_type == "yearly":
            assert df.size == df_total.shape[0] * results.results["system"]["optimized_years"]
            assert df_total.columns.size == results.results["system"]["optimized_years"]

        # test summation
        if results.results["system"]["conduct_time_series_aggregation"]:
            number_of_ts = results.results["system"]["aggregated_time_steps_per_year"]
        #    for index in df_total.index:
        #        for year in list(range(0,results.results["system"]["optimized_years"])):
        #            assert df.loc[index][number_of_ts*year:(number_of_ts-1)*(year+1)].sum() == df_total.loc[index]
        else:
            number_of_ts = results.results["system"]["unaggregated_time_steps_per_year"]
            for index in df_total.index:
                for year in list(range(0, results.results["system"]["optimized_years"])):
                    assert df.loc[index][year::results.results["system"]["optimized_years"]].sum() == df_total.loc[index][year]

        # test get_full_ts()
        # test shape of dataframe
        if ts_type == "operational":
            if results.results["system"]["conduct_time_series_aggregation"]:
                assert df.size == df_full_ts.index.size * (
                            df_full_ts.shape[1] - results.results["system"]["optimized_years"] * (
                                results.results["system"]["unaggregated_time_steps_per_year"] - results.results["system"][
                            "aggregated_time_steps_per_year"]))
            else:
                assert df.size == df_full_ts.index.size * df_full_ts.shape[1]
        elif ts_type == "yearly":
            assert df.size == df_full_ts.index.size * results.results["system"]["optimized_years"]

# All the tests
###############

def test_1a(config, folder_path):
    # add duals for this test
    config.solver["add_duals"] = True

    # run the test
    data_set_name = "test_1a"
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, data_set_name))

    # compare the variables of the optimization setup
    compare_variables(data_set_name, optimization_setup, folder_path)
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)
    #test functions get_total() and get_full_ts()
    check_get_total_get_full_ts(res)


def test_1b(config, folder_path):
    # run the test
    data_set_name = "test_1b"
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, data_set_name))

    # compare the variables of the optimization setup
    compare_variables(data_set_name, optimization_setup, folder_path)
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_1c(config, folder_path):
    # run the test
    data_set_name = "test_1c"
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, data_set_name))

    # compare the variables of the optimization setup
    compare_variables(data_set_name, optimization_setup, folder_path)
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_1d(config, folder_path):
    # run the test
    data_set_name = "test_1d"
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, data_set_name))

    # compare the variables of the optimization setup
    compare_variables(data_set_name, optimization_setup, folder_path)
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_2a(config, folder_path):
    # run the test
    data_set_name = "test_2a"
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, data_set_name))

    # compare the variables of the optimization setup
    compare_variables(data_set_name, optimization_setup, folder_path)
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)
    #test functions get_total() and get_full_ts()
    check_get_total_get_full_ts(res)


def test_2b(config, folder_path):
    # run the test
    data_set_name = "test_2b"
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, data_set_name))

    # compare the variables of the optimization setup
    compare_variables(data_set_name, optimization_setup, folder_path)
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_2c(config, folder_path):
    # run the test
    data_set_name = "test_2c"
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, data_set_name))

    # compare the variables of the optimization setup
    compare_variables(data_set_name, optimization_setup, folder_path)
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_2d(config, folder_path):
    # run the test
    data_set_name = "test_2d"
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, data_set_name))

    # compare the variables of the optimization setup
    compare_variables(data_set_name, optimization_setup, folder_path)
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_3a(config, folder_path):
    # run the test
    data_set_name = "test_3a"
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, data_set_name))

    # compare the variables of the optimization setup
    compare_variables(data_set_name, optimization_setup, folder_path)
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)
    #test functions get_total() and get_full_ts()
    check_get_total_get_full_ts(res)


def test_3b(config, folder_path):
    # run the test
    data_set_name = "test_3b"
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, data_set_name))

    # compare the variables of the optimization setup
    compare_variables(data_set_name, optimization_setup, folder_path)
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_4a(config, folder_path):
    # run the test
    data_set_name = "test_4a"
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, data_set_name))

    # compare the variables of the optimization setup
    compare_variables(data_set_name, optimization_setup, folder_path)
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)
    #test functions get_total() and get_full_ts()
    check_get_total_get_full_ts(res)


def test_4b(config, folder_path):
    # run the test
    data_set_name = "test_4b"
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, data_set_name))

    # compare the variables of the optimization setup
    compare_variables(data_set_name, optimization_setup, folder_path)
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)
    #test functions get_total() and get_full_ts()
    check_get_total_get_full_ts(res)


def test_4c(config, folder_path):
    # run the test
    data_set_name = "test_4c"
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, data_set_name))

    # compare the variables of the optimization setup
    compare_variables(data_set_name, optimization_setup, folder_path)
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_4d(config, folder_path):
    # run the test
    data_set_name = "test_4d"
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, data_set_name))

    # compare the variables of the optimization setup
    compare_variables(data_set_name, optimization_setup, folder_path)
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_4e(config, folder_path):
    # run the test
    data_set_name = "test_4e"
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, data_set_name))

    # compare the variables of the optimization setup
    compare_variables(data_set_name, optimization_setup, folder_path)
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_4f(config, folder_path):
    # run the test
    data_set_name = "test_4f"
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, data_set_name))

    # compare the variables of the optimization setup
    compare_variables(data_set_name, optimization_setup, folder_path)
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_4g(config, folder_path):
    # run the test
    data_set_name = "test_4g"
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, data_set_name))

    # compare the variables of the optimization setup
    compare_variables(data_set_name, optimization_setup, folder_path)
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)
    #test functions get_total() and get_full_ts()
    check_get_total_get_full_ts(res)


def test_5a(config, folder_path):
    # run the test
    data_set_name = "test_5a"
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, data_set_name))

    # compare the variables of the optimization setup
    compare_variables(data_set_name, optimization_setup, folder_path)
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)
    #test functions get_total() and get_full_ts()
    check_get_total_get_full_ts(res)


def test_6a(config, folder_path):
    # run the test
    data_set_name = "test_6a"
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, data_set_name))

    # compare the variables of the optimization setup
    compare_variables(data_set_name, optimization_setup, folder_path)
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)
    #test functions get_total() and get_full_ts()
    check_get_total_get_full_ts(res)


def test_6b(config, folder_path):
    # run the test
    data_set_name = "test_6b"
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, data_set_name))

    # compare the variables of the optimization setup
    compare_variables(data_set_name, optimization_setup, folder_path)
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_6c(config, folder_path):
    # run the test
    data_set_name = "test_6c"
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, data_set_name))

    # compare the variables of the optimization setup
    compare_variables(data_set_name, optimization_setup, folder_path)
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_6d(config, folder_path):
    # run the test
    data_set_name = "test_6d"
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, data_set_name))

    # compare the variables of the optimization setup
    compare_variables(data_set_name, optimization_setup, folder_path)
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)


def test_7a(config, folder_path):
    # run the test
    data_set_name = "test_7a"
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, data_set_name))

    # compare the variables of the optimization setup
    compare_variables(data_set_name, optimization_setup, folder_path)
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)
    #test functions get_total() and get_full_ts()
    check_get_total_get_full_ts(res)

def test_7b(config, folder_path):
    # run the test
    data_set_name = "test_7b"
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, data_set_name))

    # compare the variables of the optimization setup
    compare_variables(data_set_name, optimization_setup, folder_path)
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)

def test_7c(config, folder_path):
    # run the test
    data_set_name = "test_7c"
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, data_set_name))

    # compare the variables of the optimization setup
    compare_variables(data_set_name, optimization_setup, folder_path)
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)

def test_8a(config, folder_path):
    # run the test
    data_set_name = "test_8a"
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, data_set_name))

    # compare the variables of the optimization setup
    compare_variables(data_set_name, optimization_setup, folder_path)
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)
    #test functions get_total() and get_full_ts()
    check_get_total_get_full_ts(res)

def test_8b(config, folder_path):
    # run the test
    data_set_name = "test_8b"
    optimization_setup = main(config=config, dataset_path=os.path.join(folder_path, data_set_name))

    # compare the variables of the optimization setup
    compare_variables(data_set_name, optimization_setup, folder_path)
    # read the results and check again
    res = Results(os.path.join("outputs", data_set_name))
    compare_variables_results(data_set_name, res, folder_path)

if __name__ == "__main__":
    from config import config
    config.solver["keep_files"] = False
    folder_path = os.path.dirname(__file__)
    test_6b(config,folder_path)