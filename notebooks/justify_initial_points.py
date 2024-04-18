# First, let's read the content of the file to understand its structure.


file_path20 = r'C:\Users\bekaj\Documents\ETH\Master\Masterarbeit\euler_calcs\check_initial_points_20/protocol_diff_flows_0_20.log'
file_path25 = r'C:\Users\bekaj\Documents\ETH\Master\Masterarbeit\euler_calcs\check_initial_points_25/protocol_diff_flows_0_25.log'
file_path30 = r'C:\Users\bekaj\Documents\ETH\Master\Masterarbeit\euler_calcs\check_initial_points_30/protocol_diff_flows_0_30.log'

file_paths = [file_path20, file_path25, file_path30]

initial_points = dict()


# Reading the first few lines of each file to understand their structure
file_contents = {}
for file_path in file_paths:
    with open(file_path, 'r') as file:
        file_contents[file_path] = [file.readline() for _ in range(5)]

import pandas as pd

def analyze_file(file_path):
    # Read the file into a DataFrame, split by ':'
    df = pd.read_csv(file_path, delimiter=':', engine='python', header=None)
    df.columns = df.iloc[0].str.strip()  # Use the first row as header and strip any extra spaces
    df = df.drop(0).reset_index(drop=True)  # Drop the header row from the data and reset index
    df = df.apply(pd.to_numeric, errors='coerce')  # Convert all columns to numeric, coerce errors to NaN

    results = {}
    for column in df.columns:
        first_zero_index = df[column].eq(0).idxmax() if 0 in df[column].values else None
        zero_count = df[column].eq(0).sum()
        last_is_zero = df[column].iloc[-1] == 0

        results[column] = {
            "First zero iteration": first_zero_index,
            "Total zeros": zero_count,
            "Last iteration is zero": last_is_zero
        }

    total_zeros = sum([res["Total zeros"] for res in results.values()])
    earliest_zero = min([res["First zero iteration"] for res in results.values() if res["First zero iteration"] is not None], default=None)
    no_zeros_count = sum([1 for res in results.values() if res["Total zeros"] == 0])

    return results, total_zeros, earliest_zero, no_zeros_count

# Process each file
analysis_results = {}
comparative_data = {}
for file_path in file_paths:
    result, total_zeros, earliest_zero, no_zeros_count = analyze_file(file_path)
    analysis_results[file_path] = result
    comparative_data[file_path] = {
        "Total zeros": total_zeros,
        "Earliest zero": earliest_zero,
        "Columns with no zeros": no_zeros_count
    }

x = 0
