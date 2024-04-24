from zen_garden.postprocess.results import Results
import pandas as pd
import matplotlib.pyplot as plt

ref_case_storage = r'C:\Users\bekaj\Documents\ETH\Master\Masterarbeit\euler_calcs\Europe_storage_20202050_400aggts\europe_20202050_normal_storage'
ref_case = r'C:\Users\bekaj\Documents\ETH\Master\Masterarbeit\euler_calcs\Europe_ref_2020_2050_400aggts\output'
ref_design = r'C:\Users\bekaj\Documents\ETH\Master\Masterarbeit\Model_Code\Software\ZEN-garden\data\outputs\daten_storage_actual'
ref_storage = Results(ref_case_storage)
res_ref_case = Results(ref_case)
res_design = Results(ref_design)
years = [i for i in range(2020, 2052, 2)]

def net_present_costs(res_ref_case, years):

    # Net present costs
    df_costs = res_ref_case.get_total('net_present_cost').round(1)

    plt.figure(figsize=(12, 7))
    plt.bar(years, list(df_costs), width=1.0, label='Normal Mode')  # Set width to 1 for clarity
    plt.xlabel('Years')
    plt.ylabel('Net Present Cost [MEuro]')
    plt.title('Net Present Cost over the Year')
    plt.xticks(years, rotation=45)  # Rotate x labels for better visibility
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    # plt.show()

    return None

def flow_transport(res_ref_case, years):
    # Flow transport
    df_ref = res_ref_case.get_total('flow_transport').round(1)
    zero_columns = df_ref.T.columns[(df_ref.T == 0).all()]
    df_ref_T = df_ref.T.drop(columns=zero_columns)

    #Get the biomethane flow transport
    df_biomethane = df_ref_T['biomethane_transport']

    # Plotting each of the first 5 columns
    for column in df_biomethane.columns[:5]:
        plt.figure(figsize=(12, 7))  # Adjust the figure size as needed
        plt.bar(years, df_biomethane[column], width=1.0, label='Normal Mode')  # Create a bar plot
        plt.xlabel('Years')  # Set the x-axis label
        plt.ylabel('Flow Transport [GW]')  # Set the y-axis label
        plt.title('Flow Transport - ' + column)  # Set the title of the plot
        plt.xticks(years, rotation=45)
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        # plt.show()

    return None


def transport_capacity(res_ref_case, years):
    # Capacity addition transport capacity (multipliziert mit 8760 damit pro Jahr)
    df_cap = res_ref_case.get_total('capacity').round(2)
    df_cap = df_cap * 8760
    zero_columns = df_cap.T.columns[(df_cap.T == 0).all()]
    df_cap_T = df_cap.T.drop(columns=zero_columns)

    df_biomethane = df_cap_T['biomethane_transport']['power']
    # Plotting each of the first 5 columns
    for column in df_biomethane.columns[:5]:
        plt.figure(figsize=(12, 7))  # Adjust the figure size as needed
        plt.bar(years, df_biomethane[column], width=1.0, label='Normal Mode')  # Create a bar plot
        plt.xlabel('Years')  # Set the x-axis label
        plt.ylabel('Transport Capacity [GWh]')  # Set the y-axis label
        plt.title('Transport Capacity / ' + column)  # Set the title of the plot
        plt.xticks(years, rotation=45)
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.show()

    return None



def capacities_agg(res_ref_case, years):
    # Check the capacities, aggregated for the whole energy system, shown per technology and year.
    df_cap = res_ref_case.get_total('capacity').round(2)
    df_cap = df_cap * 8760

    # zero_columns = df_cap.T.columns[(df_cap.T == 0).all()]
    # df_cap_T = df_cap.T.drop(columns=zero_columns)

    # df_cap = df_cap_T.T
    # rows = []
    # for row in df_cap.index:
    #     if row[0] not in rows:
    #         rows.append(row[0])

    techs = res_ref_case.solution_loader.scenarios['none'].system.set_conversion_technologies

    for tech in techs:
        capacity_sum = list(df_cap.loc[tech].loc['power'].sum(axis=0))

        plt.figure(figsize=(12, 7))  # Adjust the figure size as needed
        plt.bar(years, capacity_sum, width=1.0, label='Normal Mode')  # Create a bar plot
        plt.xlabel('Years')  # Set the x-axis label
        plt.ylabel('Capacity [GWh]')  # Set the y-axis label
        plt.title('Capacity for entire Energy System / ' + tech)  # Set the title of the plot
        plt.xticks(years, rotation=45)
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.show()




if __name__ == '__main__':
    flow_transport(res_ref_case, years)

    x = 0