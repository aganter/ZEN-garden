"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:  Model settings. Overwrite default values defined in default_config.py here.
==========================================================================================================================================================================="""

## System - Default dictionary
system = dict()

## System - settings update compared to default values
system['set_conversion_technologies']     = ["CHP_plant", "fuel_cell", "photovoltaics"]
system['set_storage_technologies']        = ["hydrogen_storage", "battery_storage"]
system['set_transport_technologies']      = ["hydrogen_pipeline"]

system['set_nodes']                      = ["DE", "CH"]

# time steps
system["reference_year"]                 = 2022
system["unaggregated_time_steps_per_year"]  = 1
system["aggregated_time_steps_per_year"]    = 1
system["conduct_time_series_aggregation"]  = False

system["optimized_years"]                = 10
system["interval_between_years"]          = 2
system["use_rolling_horizon"]             = True
system["years_in_rolling_horizon"]         = 2

system["use_endogenous_learning"]        = False

