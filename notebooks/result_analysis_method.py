from zen_garden.postprocess.results import Results
import pandas as pd
import matplotlib.pyplot as plt

ref_case = r'C:\Users\bekaj\Documents\ETH\Master\Masterarbeit\euler_calcs\Europe_ref_2020_2050_400aggts\output'
res_ref_case = Results(ref_case)

# Net present costs
df_costs = res_ref_case.get_total('net_present_cost').round(1)
years = [i for i in range(2020, 2052, 2)]
