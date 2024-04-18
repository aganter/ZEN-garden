from zen_garden.postprocess.results import Results

import pandas as pd

# # Example DataFrames
# data1 = {'A': [1, 2, 3], 'B': [4, 5, 6]}
# data2 = {'A': [7, 8, 9], 'B': [10, 11, 12]}
# data3 = {'A': [13, 14, 15], 'B': [16, 17, 18]}
#
# df1 = pd.DataFrame(data1)
# df2 = pd.DataFrame(data2)
# df3 = pd.DataFrame(data3)
#
# # Select specific rows you want to combine
# # For example, to combine the first row from each DataFrame
# row_df1 = df1.iloc[[0]]  # Adjust index for your specific row
# row_df2 = df2.iloc[[0]]  # Adjust index for your specific row
# row_df3 = df3.iloc[[0]]  # Adjust index for your specific row
# # df1.iloc[[1,2]]
#
# # Combine the selected rows into a new DataFrame
# combined_df = pd.concat([row_df1, row_df2, row_df3], ignore_index=True)
#
# print(combined_df)

tt = r'C:\Users\bekaj\Documents\ETH\Master\Masterarbeit\euler_calcs\check_resultsfolder\HSC_solution_algorithm_community_20'
res_tt = Results(tt)

path_to_data_ref = r'C:\Users\bekaj\Documents\ETH\Master\Masterarbeit\euler_calcs\Europe_5years_28032024\HSC_solution_alogrithm_community_nodummy'
path_bayesian = r'C:\Users\bekaj\Documents\ETH\Master\Masterarbeit\euler_calcs\bayesian_run_dynamic_4diff_aggts_07042024'
path_design = r'C:\Users\bekaj\Documents\ETH\Master\Masterarbeit\Model_Code\Software\ZEN-garden\data\outputs\HSC_solution_alogrithm_community'

res_ref = Results(path_to_data_ref)
res_bay = Results(path_bayesian)
res_design = Results(path_design)
# Here we have two different scenarios

df_ref = res_ref.get_total('flow_transport').round(2).loc['biomethane_transport']
df_bay = res_bay.get_total('flow_transport').round(2)
df_design = res_design.get_total('flow_transport').round(2).loc['biomethane_transport']

df_ref_nonzero = df_ref[~(df_ref == 0).all(axis=1)]
df_design_nonzero = df_design[~(df_design == 0).all(axis=1)]

# Convert lists to sets for easier comparison
set1 = set(list(df_ref_nonzero.index))
set2 = set(list(df_design_nonzero.index))

# Elements in list1 but not in list2
not_in_list2 = set1 - set2

# Elements in list2 but not in list1
not_in_list1 = set2 - set1

# Combine elements not in both lists
not_in_both = not_in_list1.union(not_in_list2)


scenarios = [f'scenario_{index}' for index in range(5)]

df_bay_normal = pd.DataFrame()
df_scen_list_normal = []
df_scen_list_dummy = []

for scenario in scenarios:
    df_scen = df_bay.loc[scenario].loc['biomethane_transport']

    normal_edges = [row_name for row_name in df_scen.index if 'dummy' not in row_name]
    dummy_edges = [row_name for row_name in df_scen.index if 'dummy' in row_name]

    df_bay_normal = df_scen.loc[normal_edges]
    df_bay_dummy = df_scen.loc[dummy_edges]

    df_scen_list_normal.append(df_bay_normal)
    df_scen_list_dummy.append(df_bay_dummy)

df_final_normal = pd.concat(df_scen_list_normal)
df_final_dummy = pd.concat(df_scen_list_dummy)

for row_dummy in df_final_dummy.index:
    if row_dummy not in df_final_normal.index:
        df_final_normal = pd.concat([df_final_normal, df_final_dummy.loc[[row_dummy]]])
    else:
        print('already')


# delete duplicates of edges
count_elem = dict()
for row in df_final_normal.index:
    count_elem[row] = list(df_final_normal.index).count(row)

dummy_edges_list = [row_name for row_name in df_final_normal.index if 'dummy' in row_name]
normal_edges_list = [row_name for row_name in df_final_normal.index if 'dummy' not in row_name]
duplicates = []

for edge in dummy_edges_list:
    in_edge = edge.split('-')[0]
    out_edge = edge.split('-')[1]

    if 'dummy' in in_edge:

        in_comp = in_edge[:2]
        out_comp = out_edge + 'dummy'

        edge_comp = f'{in_comp}-{out_comp}'

    else:
        in_comp = in_edge + 'dummy'
        out_comp = out_edge[:2]

        edge_comp = f'{in_comp}-{out_comp}'

    if edge_comp not in duplicates:
        duplicates.append(edge_comp)
    if edge not in duplicates:
        duplicates.append(edge)

new_edges = [duplicates[idx] for idx in range(0, len(duplicates), 2)]
all_edges = new_edges + normal_edges_list

df_end = df_final_normal.loc[[row_name for row_name in df_final_normal.index if row_name in all_edges]]

#Rename rows
for rowname in df_end.index:
    if 'dummy' in rowname:
        node1, node2 = rowname.split('-')

        if 'dummy' in node1:
            node1 = node1[:2]
        else:
            node2 = node2[:2]

        rename_edge = f'{node1}-{node2}'

        df_end = df_end.rename(index={rowname: rename_edge})

diff = df_end - df_ref.loc['biomethane_transport'].round(2)

y = 0