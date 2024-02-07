"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Model settings. Overwrite default values defined in default_config.py here.
==========================================================================================================================================================================="""

## System - Default dictionary
system = dict()

# setNodes
#system["set_nodes"] = ["CH01", "CH02"]

system["set_electricity_generation_technologies"] = ["wind_onshore", "wind_offshore", "pv_ground", "pv_rooftop"]

system["set_biomass_conversion_technologies"] = ["anaerobic_digestion", "gasification_CCS", "gasification", "biomethane_conversion"]

system["set_hydrogen_generation_technologies"] = ["electrolysis_AE", "electrolysis_PEM",
                                                  "SMR", "SMR_CCS", "ATR_CCS"]

system["set_ammonia_conversion_technologies"] = ["haber_bosch", "e_haber_bosch", "ASU"]

system["set_hydrogen_conversion_technologies"] = ["hydrogen_compressor_high", "hydrogen_evaporation", "hydrogen_liquefication"]

system["set_carbon_removal_technologies"] = ["carbon_removal", "carbon_liquefication", "carbon_storage"]

system["set_conversion_technologies"] = system["set_electricity_generation_technologies"] + system["set_biomass_conversion_technologies"] + system["set_hydrogen_generation_technologies"] + system["set_ammonia_conversion_technologies"] + system["set_hydrogen_conversion_technologies"] + system["set_carbon_removal_technologies"]\

# Storage technologies
system["set_storage_technologies"] = ["hydrogen_storage", "battery_storage"]

# Transport technologies
system["set_transport_technologies"] = ["hydrogen_truck_gas", "hydrogen_pipeline", "hydrogen_train", "hydrogen_ship", "hydrogen_truck_liquid",
                                        "carbon_truck", "carbon_pipeline", "carbon_train", "carbon_ship",
                                        "dry_biomass_truck", "biomethane_transport"]

system['set_bidirectional_transport_technologies'] = ['power_line']

# timeSeries settings
system["conduct_time_series_aggregation"] = True #if True, TSA is conducted
system["reference_year"] = 2024
system["unaggregated_time_steps_per_year"] = 8760
system["aggregated_time_steps_per_year"] = 1 # 40
system['multi_grid_time_index'] = False # if True, each element has its own time index; if False, use single time grid approach
system['non_aligned_time_index'] = False # if True, each element runs on a different, not-aligned time grid, by default then also multiGridTimeIndex; if False, use aligned time grid
system['aggregate_by_historic_time_series'] = True # if True, _technologies that do not have time-dependent time series are aggregated by historic time series
system['type_historic_time_series'] = "demand" # choose from "demand", "generation": select type of historic time series used for aggregation,
                                                  # either historic generation of technology or demand of reference carrier

system['conduct_scenario_analysis'] = False

system["optimized_years"] = 13
system["interval_between_years"] = 2
# rolling horizon
system["use_rolling_horizon"] = False
system["years_in_rolling_horizon"] = 1

# results
system["overwrite_output"] = True