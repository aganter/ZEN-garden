from zen_garden.postprocess.results import Results

path_to_data = r'C:\Users\bekaj\Documents\ETH\Master\Masterarbeit\Model_Code\Software\ZEN-garden\data\outputs\HSC_solution_alogrithm_community'

res = Results(path_to_data)
# Here we have two different scenarios

df = res.get_total('flow_transport').round(3)

y= 0