"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Model settings. Overwrite default values defined in default_config.py here.
==========================================================================================================================================================================="""

## System - Default dictionary
system = dict()

# Nodes
system["set_nodes"] = ["DE12", "DE21", "DE22", "DED5", "DEE0", "DEF0", "DE80"]
# Paulas nodes: ["AT22", "AT31", "BE21", "BE24", "BE32"]
# system["set_nodes"] = []



# Conversion technologies
system["set_conversion_technologies"] = [
    # hydrogen conversion
    "electrolysis", "SMR", "gasification", "SMR_CCS", "gasification_CCS",
    # carbon conversion
    "carbon_storage", "carbon_removal",
    # hydrogen conditioning
    #"hydrogen_liquefication", "hydrogen_evaporation", "hydrogen_compressor_low",
    # carbon conditioning
    #"carbon_liquefication",
    # renewable electricity generation
    "wind_onshore", "wind_offshore", "pv_ground", "pv_rooftop",
    # other 
    "anaerobic_digestion","biomethane_conversion",
]

# Transport technologies
system["set_transport_technologies"] = [
    # hydrogen transport
    "hydrogen_pipeline", #no conditioning
    # carbon transport
    "carbon_pipeline", #no conditioning
    # biomass transport
    "dry_biomass_truck", "biomethane_transport",
]


# timeseries aggregation settings
system["conduct_time_series_aggregation"] = True #if True, TSA is conducted
system["reference_year"] = 2022
system["unaggregated_time_steps_per_year"] = 1
system["aggregated_time_steps_per_year"] = 1

# time horizon
system["optimized_years"] = 3
system["interval_between_years"] = 2

# rolling horizon
system["use_rolling_horizon"] = False
system["years_in_rolling_horizon"] = 5

# results
system["overwrite_output"] = False

# anyaxie
system["use_endogenous_learning"]      = True
system["global_active_capacity"]   = False