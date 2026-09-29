# Taxonomy v2 statistics

Sources: data\engine_bay_reviewed, data\engine_bay_reviewed_hybrid · splits train, val, test · 912 images · 3542 instances · promotion ≥ 60 instances from ≥ 8 vehicles, tier A kept while ≥ 40 instances

## Components

| system | component | instances | vehicles | declared | recommended | trained as |
|---|---|---|---|---|---|---|
| electrical | battery | 255 | 25 | A | A | battery |
| electrical | battery_terminal | 320 | 25 | A | A | battery_terminal |
| electrical | fuse_relay_box | 106 | 25 | A | A | fuse_relay_box |
| electrical | ecu_module | 76 | 19 | A | A | ecu_module |
| electrical | alternator | 51 | 11 | A | A | alternator |
| electrical | starter_motor | 0 | 0 | B | B | other_rotating_device |
| electrical | jump_start_post | 0 | 0 | B | B | other_cap_plug |
| electrical | wiring_harness | 0 | 0 | B | B | — |
| ignition | ignition_coil | 394 | 26 | A | A | ignition_coil |
| ignition | coil_pack | 0 | 0 | B | B | other_module_box |
| ignition | spark_plug_wire | 0 | 0 | B | B | other_hose_line |
| air_intake | air_filter_box | 292 | 28 | A | A | air_filter_box |
| air_intake | air_intake_duct | 384 | 28 | A | A | air_intake_duct |
| air_intake | maf_sensor | 107 | 18 | A | A | maf_sensor |
| air_intake | throttle_body | 159 | 24 | A | A | throttle_body |
| air_intake | intake_manifold | 196 | 24 | A | A | intake_manifold |
| air_intake | map_sensor | 0 | 0 | B | B | other_sensor_actuator |
| air_intake | iat_sensor | 0 | 0 | B | B | other_sensor_actuator |
| air_intake | intake_resonator | 0 | 0 | B | B | other_module_box |
| air_intake | vacuum_hose | 0 | 0 | B | B | other_hose_line |
| forced_induction | turbocharger | 4 | 2 | B | B | other_rotating_device |
| forced_induction | intercooler_piping | 14 | 4 | B | B | other_hose_line |
| forced_induction | wastegate_actuator | 0 | 0 | B | B | other_sensor_actuator |
| fuel_evap | fuel_rail | 0 | 0 | B | B | other_hose_line |
| fuel_evap | fuel_injector | 0 | 0 | B | B | other_sensor_actuator |
| fuel_evap | evap_purge_valve | 0 | 0 | B | B | other_sensor_actuator |
| fuel_evap | fuel_supply_line | 0 | 0 | B | B | other_hose_line |
| fuel_evap | hp_fuel_pump | 0 | 0 | B | B | other_sensor_actuator |
| exhaust_emissions | exhaust_manifold_heat_shield | 68 | 15 | A | A | exhaust_manifold_heat_shield |
| exhaust_emissions | exhaust_manifold | 0 | 0 | B | B | — |
| exhaust_emissions | o2_sensor | 0 | 0 | B | B | other_sensor_actuator |
| exhaust_emissions | egr_valve | 0 | 0 | B | B | other_sensor_actuator |
| cooling | coolant_reservoir | 112 | 23 | A | A | coolant_reservoir |
| cooling | radiator_cap | 73 | 16 | A | A | radiator_cap |
| cooling | radiator_hose | 49 | 18 | A | A | radiator_hose |
| cooling | heater_hose | 0 | 0 | B | B | other_hose_line |
| cooling | thermostat_housing | 0 | 0 | B | B | — |
| cooling | radiator | 1 | 1 | B | B | — |
| cooling | radiator_cooling_fan | 4 | 4 | B | B | — |
| cooling | ect_sensor | 0 | 0 | B | B | other_sensor_actuator |
| lubrication | oil_filler_cap | 204 | 28 | A | A | oil_filler_cap |
| lubrication | oil_dipstick | 192 | 27 | A | A | oil_dipstick |
| lubrication | oil_filter | 15 | 7 | B | B | — |
| lubrication | valve_cover | 0 | 0 | B | B | — |
| lubrication | pcv_valve | 0 | 0 | B | B | other_sensor_actuator |
| lubrication | breather_hose | 0 | 0 | B | B | other_hose_line |
| lubrication | transmission_oil_dipstick | 0 | 0 | B | B | other_cap_plug |
| accessory_drive | serpentine_belt | 9 | 4 | B | B | — |
| accessory_drive | ac_compressor | 1 | 1 | B | B | other_rotating_device |
| accessory_drive | belt_tensioner | 0 | 0 | B | B | other_rotating_device |
| accessory_drive | power_steering_pump | 0 | 0 | B | B | other_rotating_device |
| brakes | brake_fluid_reservoir | 149 | 24 | A | A | brake_fluid_reservoir |
| brakes | brake_booster | 17 | 7 | B | B | — |
| brakes | abs_modulator_unit | 8 | 5 | B | B | other_module_box |
| brakes | brake_master_cylinder | 0 | 0 | B | B | — |
| brakes | clutch_reservoir | 0 | 0 | B | B | other_reservoir |
| steering | power_steering_reservoir | 24 | 3 | B | B | other_reservoir |
| washer | washer_fluid_reservoir | 70 | 21 | A | A | washer_fluid_reservoir |
| washer | windshield_wiper_motor | 3 | 2 | B | B | — |
| body | engine_cover | 85 | 20 | A | A | engine_cover |
| body | strut_tower_brace | 12 | 4 | B | B | — |
| body | hood_latch_mechanism | 24 | 9 | B | B | — |
| tools | multimeter_diagnostic_tool | 64 | 14 | A | A | multimeter_diagnostic_tool |

## Training classes (v2 build)

| id | class | instances | vehicles |
|---|---|---|---|
| 0 | battery | 255 | 25 |
| 1 | battery_terminal | 320 | 25 |
| 2 | fuse_relay_box | 106 | 25 |
| 3 | ecu_module | 76 | 19 |
| 4 | alternator | 51 | 11 |
| 5 | ignition_coil | 394 | 26 |
| 6 | air_filter_box | 292 | 28 |
| 7 | air_intake_duct | 384 | 28 |
| 8 | maf_sensor | 107 | 18 |
| 9 | throttle_body | 159 | 24 |
| 10 | intake_manifold | 196 | 24 |
| 11 | exhaust_manifold_heat_shield | 68 | 15 |
| 12 | coolant_reservoir | 112 | 23 |
| 13 | radiator_cap | 73 | 16 |
| 14 | radiator_hose | 49 | 18 |
| 15 | oil_filler_cap | 204 | 28 |
| 16 | oil_dipstick | 192 | 27 |
| 17 | brake_fluid_reservoir | 149 | 24 |
| 18 | washer_fluid_reservoir | 70 | 21 |
| 19 | engine_cover | 85 | 20 |
| 20 | multimeter_diagnostic_tool | 64 | 14 |
| 21 | other_reservoir | 24 | 3 |
| 22 | other_module_box | 8 | 5 |
| 23 | other_sensor_actuator | 0 | 0 |
| 24 | other_hose_line | 14 | 4 |
| 25 | other_cap_plug | 0 | 0 |
| 26 | other_rotating_device | 5 | 3 |

## Systems

| system | instances |
|---|---|
| electrical | 808 |
| ignition | 394 |
| air_intake | 1138 |
| forced_induction | 18 |
| fuel_evap | 0 |
| exhaust_emissions | 68 |
| cooling | 239 |
| lubrication | 411 |
| accessory_drive | 10 |
| brakes | 174 |
| steering | 24 |
| washer | 73 |
| body | 121 |
| tools | 64 |
