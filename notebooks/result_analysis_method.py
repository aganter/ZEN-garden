from zen_garden.postprocess.results import Results
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

method_path = r'C:\Users\bekaj\Documents\ETH\Master\Masterarbeit\euler_calcs\Europe_20202050_wo_storage_method\output'
method_res = Results(method_path)

normal_path = r'C:\Users\bekaj\Documents\ETH\Master\Masterarbeit\euler_calcs\Europe_ref_2020_2050_400aggts\output'
normal_res = Results(normal_path)

years = [i for i in range(2020, 2040, 1)]
years_normal = [i for i in range(2020, 2052, 2)]
years_axis = [i for i in range(2020,2051,1)]
scenarios = list(method_res.solution_loader.scenarios.keys())

# Net present costs
def net_present_costs(method_res, normal_res, years, years_normal):

    # Net present costs - method
    df_costs = method_res.get_total('net_present_cost').round(1)
    df_costs.columns = years
    costs_sum = df_costs.sum()

    # Net present costs - conventional
    df_costs_norm = normal_res.get_total('net_present_cost').round(1)
    df_costs_norm.index = years_normal

    merged = pd.concat([costs_sum, df_costs_norm], axis=1, keys=['Method', 'Conventional'])
    merged_reset = merged.reset_index()
    merged_reset.rename(columns={'index': 'Year'}, inplace=True)

    # Plotting
    fig, ax = plt.subplots(figsize=(14, 8))
    width = 0.35  # Width of the bars

    # Plotting bars for each dataset
    ax.bar(merged_reset['Year'] - width / 2, merged_reset['Method'], width, label='Method', alpha=0.8)
    ax.bar(merged_reset['Year'] + width / 2, merged_reset['Conventional'], width, label='Conventional', alpha=0.8,
           color='orange')

    # Adding labels and title
    ax.set_xlabel('Year')
    ax.set_ylabel('Net Present Cost')
    ax.set_title('Comparison of Net Present Costs')
    ax.legend()

    # Improve x-axis: show every year and avoid clutter
    plt.xticks(merged_reset['Year'], rotation=90)

    plt.show()

    # plt.figure(figsize=(12, 7))
    # plt.bar(years, costs_sum, label='Algorithm')  # Set width to 1 for clarity
    # plt.bar(years_normal, list(df_costs_norm), label='Conventional')
    # plt.xlabel('Years')
    # plt.ylabel('Net Present Cost [MEuro]')
    # plt.title('Net Present Cost over the Years')
    # plt.xticks(years_axis, rotation=45)  # Rotate x labels for better visibility
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.legend()
    # plt.show()

    x = np.arange(len(years_axis))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, costs_sum, width, label='2021')
    rects2 = ax.bar(x + width / 2, costs_sum, width, label='2022')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Values')
    ax.set_title('Sales by category and year')
    ax.set_xticks(x)
    ax.set_xticklabels(years_axis)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    plt.show()

    return None


def capacities_agg(method_res, normal_res, years, years_normal):

    # Check the capacities, aggregated for the whole energy system, shown per technology and year.
    df_cap = method_res.get_total('capacity').round(2)
    df_cap = df_cap * 8760

    # Capacities - conventional
    df_cap_normal = normal_res.get_total('capacity').round(2)
    df_cap_normal = df_cap_normal * 8760

    techs = method_res.solution_loader.scenarios['scenario_0'].system.set_conversion_technologies

    cap_per_scen = dict()
    for scenario in scenarios:
        cap_per_scen[scenario] = dict()
        df_cap_scen = df_cap.loc[scenario]

        for tech in techs:
            cap_per_scen[scenario][tech] = []
            df_tech = df_cap_scen.loc[tech].loc['power']
            df_tech = df_tech[~df_tech.index.str.contains('dummy')]
            capacity_sum = list(df_tech.sum(axis=0))
            cap_per_scen[scenario][tech].append(capacity_sum)

    cap_tot = dict()
    for tech in techs:
        values = [np.array(cap_per_scen[scenario][tech][0]) for scenario in cap_per_scen]
        summed_values = np.sum(values, axis=0)
        summed_values.tolist()

        cap_tot[tech] = list(summed_values)

    for tech in techs:

        tech_series = pd.Series(cap_tot[tech])
        tech_series.index = years

        tech_series_normal = df_cap_normal.loc[tech].loc['power'].sum(axis=0)
        tech_series_normal.index = years_normal

        merged = pd.concat([tech_series, tech_series_normal], axis=1, keys=['Method', 'Conventional'])
        merged_reset = merged.reset_index()
        merged_reset.rename(columns={'index': 'Year'}, inplace=True)

        # Plotting
        fig, ax = plt.subplots(figsize=(14, 8))
        width = 0.35  # Width of the bars

        # Plotting bars for each dataset
        ax.bar(merged_reset['Year'] - width / 2, merged_reset['Method'], width, label='Method', alpha=0.8)
        ax.bar(merged_reset['Year'] + width / 2, merged_reset['Conventional'], width, label='Conventional', alpha=0.8,
               color='orange')

        # Adding labels and title
        ax.set_xlabel('Year')
        ax.set_ylabel('Capacity [GWh]')
        ax.set_title(tech)
        ax.legend()

        # Improve x-axis: show every year and avoid clutter
        plt.xticks(merged_reset['Year'], rotation=90)

        plt.show()

    # plt.figure(figsize=(12, 7))
    # plt.bar(years, costs_sum, label='Algorithm')  # Set width to 1 for clarity
    # plt.bar(years_normal, list(df_costs_norm), label='Conventional')
    # plt.xlabel('Years')
    # plt.ylabel('Net Present Cost [MEuro]')
    # plt.title('Net Present Cost over the Years')
    # plt.xticks(years_axis, rotation=45)  # Rotate x labels for better visibility
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.legend()
    # plt.show()

    return None


def flow_transport(method_res, normal_res, years, scenarios):

    transports = method_res.solution_loader.scenarios['scenario_0'].system.set_transport_technologies

    # normal_df_help = normal_res.get_total('flow_transport').round(1).loc['biomethane_transport']
    # all_edges = list(normal_df_help.index)
    # all_nodes = list(normal_res.solution_loader.scenarios['none'].system.set_nodes)

    transport_dict = dict()

    for trans in transports:

        # Flow transport
        df_trans_base = method_res.get_total('flow_transport').round(1).loc[scenarios[0]].loc[trans]
        df_trans_base.index = [dynamic_rename(name) for name in df_trans_base.index]


        for scenario in scenarios:

            df_trans_scen = method_res.get_total('flow_transport').round(1).loc[scenario].loc[trans]
            df_trans_scen.index = [dynamic_rename(name) for name in df_trans_scen.index]

            if scenario != scenarios[0]:

                # Edges in method
                edges_base = set(list(df_trans_base.index))
                edges_scen = set(list(df_trans_scen.index))

                # Find elements missing in base
                missing_edges = list(edges_scen.difference(edges_base))
                df_trans_scen_edges = pd.DataFrame(df_trans_scen, index=missing_edges)

                df_trans_base = pd.concat([df_trans_base, df_trans_scen_edges], ignore_index=False)

        df_trans_base_T = df_trans_base.T
        df_trans_base_T.index = years
        transport_dict[trans] = df_trans_base_T


    for trans in transports:
        df_trans_normal = normal_res.get_total('flow_transport').round(1).loc[trans]

        df_trans_normal_T = df_trans_normal.T
        df_trans_normal_T.index = years_normal

        columns_mod = [column + '_conv' for column in df_trans_normal_T.columns]
        df_trans_normal_T.columns = columns_mod

        df_meth = transport_dict[trans]

        merged = pd.concat([df_meth, df_trans_normal_T], axis=1, join='outer')
        merged_reset = merged.reset_index().rename(columns={'index': 'Year'})

        for col in df_meth.columns:

            # Plotting
            fig, ax = plt.subplots(figsize=(14, 8))
            width = 0.35  # Width of the bars

            # Col for conventional
            col_conv = col + '_conv'

            # Plotting bars for each dataset
            ax.bar(merged_reset['Year'] - width / 2, merged_reset[col], width, label='Method', alpha=0.8)
            ax.bar(merged_reset['Year'] + width / 2, merged_reset[col_conv], width, label='Conventional', alpha=0.8,
                   color='orange')

            # Adding labels and title
            ax.set_xlabel('Year')
            ax.set_ylabel('Flow transport [GWh]')
            ax.set_title(trans + ' / ' + col)
            ax.legend()

            # Improve x-axis: show every year and avoid clutter
            plt.xticks(merged_reset['Year'], rotation=90)

            plt.show()




    # zero_columns = df_trans_base.T.columns[(df_trans_base.T == 0).all()]
    # df_trans_base_T = df_trans_base.T.drop(columns=zero_columns)
    # df_trans_base = df_trans_base_T.T
    #
    #
    # # Plotting each of the first 5 columns
    # for column in df_trans_base_T.columns:
    #     plt.figure(figsize=(12, 7))  # Adjust the figure size as needed
    #     plt.bar(years, df_trans_base_T[column], label='Method Mode')  # Create a bar plot
    #     plt.xlabel('Years')  # Set the x-axis label
    #     plt.ylabel('Flow Transport [GW]')  # Set the y-axis label
    #     plt.title('Flow Transport - ' + column)  # Set the title of the plot
    #     plt.xticks(years, rotation=45)
    #     plt.legend()
    #     plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    #     plt.show()

    return None

# def capacities_agg(method_res, years, scenarios):
#     # Check the capacities, aggregated for the whole energy system, shown per technology and year.
#     df_cap = method_res.get_total('capacity').round(2)
#     df_cap = df_cap * 8760
#
#     techs = method_res.solution_loader.scenarios['scenario_0'].system.set_conversion_technologies
#
#     cap_per_scen = dict()
#     for scenario in scenarios:
#         cap_per_scen[scenario] = dict()
#         df_cap_scen = df_cap.loc[scenario]
#
#         for tech in techs:
#             cap_per_scen[scenario][tech] = []
#             df_tech = df_cap_scen.loc[tech].loc['power']
#             df_tech = df_tech[~df_tech.index.str.contains('dummy')]
#             capacity_sum = list(df_tech.sum(axis=0))
#             cap_per_scen[scenario][tech].append(capacity_sum)
#
#     cap_tot = dict()
#     for tech in techs:
#
#         values = [np.array(cap_per_scen[scenario][tech][0]) for scenario in cap_per_scen]
#         summed_values = np.sum(values, axis=0)
#         summed_values.tolist()
#
#         cap_tot[tech] = list(summed_values)
#
#     for tech in techs:
#         y_values = cap_tot[tech]
#         x_values = list(range(1, len(cap_tot[tech]) + 1))
#
#         plt.figure(figsize=(10, 5))
#         plt.plot(x_values, y_values, marker='o', linestyle='-', color='b')
#         plt.title(tech)
#         plt.xlabel('Year')
#         plt.ylabel('Capacity')
#         plt.ylim(bottom=0)
#         plt.grid(True)
#         plt.show()


def dynamic_rename(index_name):
    if 'dummy' in index_name:
        edge_name_1 = index_name.split('-')[0]
        edge_name_2 = index_name.split('-')[1]

        if 'dummy' in edge_name_1:
            index_name_mod = edge_name_1[:2] + '-' + edge_name_2
        else:
            index_name_mod = edge_name_1 + '-' + edge_name_2[:2]

        return index_name_mod
    else:
        return index_name



if __name__ == '__main__':
    # net_present_costs(method_res, normal_res, years, years_normal)
    flow_transport(method_res, normal_res, years, scenarios)
    # capacities_agg(method_res, normal_res, years, years_normal)



    x = 0

