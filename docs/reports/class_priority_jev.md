# Candidate classes ranked by diagnostic value (P5, Jev)

24 DTC groups from configs/diagnosis_knowledge.yaml · 66 components · DTC weights: uniform (no scan-frequency data yet)

**Jev vs the expert-curated DTC→component lists: AUROC 0.969** (61 listed pairs, 1523 unlisted). Values near 0.5 would mean Jev's inspect judgments are noise.

priority = Σ_dtc w·P(inspect) × P(visible with hood open) × P(visual check useful)

| rank | component | state | priority | DTC breadth | visible | visual useful | reviewed instances | strongest DTC links |
|---|---|---|---|---|---|---|---|---|
| 1 | wiring_harness | tier B | 4.51 | 18.43 | 0.34 | 0.72 | 0 | P0335-P0349 (0.91), P0106-P0108 (0.90), P0115-P0119 (0.90) |
| 2 | battery | tier A | 3.25 | 7.41 | 0.51 | 0.86 | 255 | P0562 (0.96), P0620-P0622 (0.86), U0100 (0.79) |
| 3 | vacuum_hose | tier B | 3.23 | 9.38 | 0.40 | 0.86 | 0 | P0171 (0.93), P0505-P0507 (0.90), P0106-P0108 (0.89) |
| 4 | air_intake_duct | tier A | 3.20 | 8.86 | 0.44 | 0.82 | 384 | P0171 (0.96), P0100-P0103 (0.95), P0505-P0507 (0.93) |
| 5 | battery_terminal | tier A | 3.17 | 5.52 | 0.66 | 0.87 | 320 | P0562 (0.96), P0620-P0622 (0.87), U0100 (0.65) |
| 6 | ect_sensor | tier B | 3.17 | 8.07 | 0.51 | 0.77 | 0 | P0115-P0119 (0.95), P0217 (0.88), P0128 (0.84) |
| 7 | breather_hose | tier B | 3.00 | 7.35 | 0.48 | 0.85 | 0 | P0100-P0103 (0.69), P0300 (0.65), P0171 (0.64) |
| 8 | coolant_reservoir | tier A | 2.93 | 4.48 | 0.71 | 0.92 | 112 | P0217 (0.95), P0128 (0.78), P0115-P0119 (0.61) |
| 9 | pcv_valve | tier B | 2.39 | 7.19 | 0.41 | 0.81 | 0 | P0171 (0.90), P0300 (0.80), P0505-P0507 (0.78) |
| 10 | oil_dipstick | tier A | 2.30 | 3.68 | 0.79 | 0.79 | 192 | P0520-P0524 (0.93), P0217 (0.56), P0300 (0.33) |
| 11 | valve_cover | tier B | 2.28 | 3.94 | 0.68 | 0.85 | 0 | P0351-P0358 (0.60), P0301-P0308 (0.58), P0300 (0.57) |
| 12 | air_filter_box | tier A | 2.20 | 5.62 | 0.62 | 0.63 | 292 | P0100-P0103 (0.89), P0171 (0.65), P0172 (0.61) |
| 13 | oil_filler_cap | tier A | 2.15 | 4.74 | 0.62 | 0.73 | 204 | P0300 (0.65), P0171 (0.60), P0172 (0.40) |
| 14 | intake_manifold | tier A | 1.97 | 7.79 | 0.32 | 0.79 | 196 | P0171 (0.95), P0505-P0507 (0.88), P0106-P0108 (0.84) |
| 15 | fuse_relay_box | tier A | 1.90 | 7.90 | 0.33 | 0.73 | 106 | P0562 (0.82), P0217 (0.77), U0100 (0.70) |
| 16 | ecu_module | tier A | 1.89 | 8.88 | 0.28 | 0.76 | 76 | U0100 (0.90), P0600-P0606 (0.89), P0106-P0108 (0.52) |
| 17 | intake_resonator | tier B | 1.86 | 4.54 | 0.54 | 0.76 | 0 | P0100-P0103 (0.60), P0171 (0.59), P0300 (0.45) |
| 18 | spark_plug_wire | tier B | 1.85 | 3.70 | 0.64 | 0.78 | 0 | P0301-P0308 (0.87), P0300 (0.85), P0351-P0358 (0.38) |
| 19 | maf_sensor | tier A | 1.81 | 6.55 | 0.39 | 0.71 | 107 | P0100-P0103 (0.97), P0171 (0.90), P0172 (0.85) |
| 20 | coil_pack | tier B | 1.74 | 4.35 | 0.57 | 0.70 | 0 | P0301-P0308 (0.91), P0300 (0.90), P0351-P0358 (0.89) |
| 21 | radiator_hose | tier A | 1.59 | 2.85 | 0.64 | 0.87 | 49 | P0217 (0.95), P0115-P0119 (0.32), P0128 (0.31) |
| 22 | iat_sensor | tier B | 1.59 | 4.92 | 0.43 | 0.75 | 0 | P0100-P0103 (0.66), P0171 (0.59), P0172 (0.59) |
| 23 | throttle_body | tier A | 1.57 | 5.93 | 0.34 | 0.78 | 159 | P0505-P0507 (0.96), P0121-P0123 (0.89), P0100-P0103 (0.59) |
| 24 | serpentine_belt | tier B | 1.49 | 4.85 | 0.54 | 0.57 | 9 | P0562 (0.87), P0217 (0.86), P0620-P0622 (0.80) |
| 25 | ignition_coil | tier A | 1.47 | 4.51 | 0.46 | 0.71 | 394 | P0351-P0358 (0.96), P0301-P0308 (0.95), P0300 (0.89) |
| 26 | map_sensor | tier B | 1.45 | 5.85 | 0.33 | 0.75 | 0 | P0106-P0108 (0.96), P0172 (0.70), P0171 (0.67) |
| 27 | alternator | tier A | 1.43 | 4.64 | 0.46 | 0.67 | 51 | P0562 (0.94), P0620-P0622 (0.93), P0600-P0606 (0.49) |
| 28 | fuel_rail | tier B | 1.29 | 4.02 | 0.40 | 0.80 | 0 | P0172 (0.68), P0171 (0.61), P0201-P0208 (0.60) |
| 29 | belt_tensioner | tier B | 1.27 | 4.04 | 0.50 | 0.63 | 0 | P0562 (0.74), P0217 (0.71), P0620-P0622 (0.66) |
| 30 | egr_valve | tier B | 1.17 | 4.23 | 0.37 | 0.75 | 0 | P0400-P0404 (0.95), P0300 (0.53), P0505-P0507 (0.45) |
| 31 | brake_booster | tier B | 1.13 | 2.83 | 0.50 | 0.80 | 17 | P0106-P0108 (0.53), P0171 (0.50), P0505-P0507 (0.41) |
| 32 | radiator | tier B | 1.13 | 2.58 | 0.51 | 0.86 | 1 | P0217 (0.95), P0128 (0.49), P0115-P0119 (0.22) |
| 33 | spark_plug | not in taxonomy v2 | 1.01 | 4.75 | 0.30 | 0.71 | 0 | P0300 (0.95), P0301-P0308 (0.95), P0351-P0358 (0.61) |
| 34 | evap_purge_valve | tier B | 1.01 | 4.34 | 0.31 | 0.75 | 0 | P0440-P0446 (0.88), P0455 (0.51), P0172 (0.47) |
| 35 | intercooler_piping | tier B | 0.92 | 3.27 | 0.34 | 0.83 | 14 | P0106-P0108 (0.53), P0100-P0103 (0.46), P0171 (0.41) |
| 36 | thermostat_housing | tier B | 0.89 | 3.97 | 0.31 | 0.72 | 0 | P0128 (0.93), P0217 (0.93), P0115-P0119 (0.51) |
| 37 | fuel_supply_line | tier B | 0.87 | 3.76 | 0.27 | 0.86 | 0 | P0171 (0.61), P0172 (0.48), P0455 (0.48) |
| 38 | exhaust_manifold | tier B | 0.85 | 3.46 | 0.31 | 0.79 | 0 | P0171 (0.45), P0420 (0.45), P0400-P0404 (0.37) |
| 39 | radiator_cap | tier A | 0.83 | 2.16 | 0.63 | 0.61 | 73 | P0217 (0.93), P0128 (0.22), P0115-P0119 (0.18) |
| 40 | fuel_injector | tier B | 0.81 | 5.35 | 0.21 | 0.72 | 0 | P0301-P0308 (0.83), P0300 (0.80), P0172 (0.77) |
| 41 | exhaust_manifold_heat_shield | tier A | 0.75 | 3.46 | 0.34 | 0.64 | 68 | P0130-P0167 (0.43), P0420 (0.35), P0217 (0.32) |
| 42 | engine_cover | tier A | 0.73 | 2.29 | 0.74 | 0.43 | 85 | P0351-P0358 (0.26), P0300 (0.22), P0100-P0103 (0.17) |
| 43 | heater_hose | tier B | 0.71 | 2.81 | 0.31 | 0.82 | 0 | P0217 (0.49), P0128 (0.40), P0115-P0119 (0.27) |
| 44 | o2_sensor | tier B | 0.70 | 4.61 | 0.21 | 0.72 | 0 | P0130-P0167 (0.73), P0172 (0.68), P0420 (0.64) |
| 45 | cam_crank_sensor | not in taxonomy v2 | 0.64 | 3.95 | 0.23 | 0.70 | 0 | P0335-P0349 (0.96), P0300 (0.52), P0600-P0606 (0.25) |
| 46 | radiator_cooling_fan | tier B | 0.63 | 2.29 | 0.46 | 0.60 | 4 | P0217 (0.93), P0115-P0119 (0.15), P0128 (0.11) |
| 47 | power_steering_pump | tier B | 0.53 | 1.20 | 0.50 | 0.88 | 0 | P0505-P0507 (0.17), P0217 (0.14), P0520-P0524 (0.07) |
| 48 | multimeter_diagnostic_tool | tier A | 0.51 | 2.85 | 0.29 | 0.62 | 64 | U0100 (0.25), P0600-P0606 (0.18), P0562 (0.17) |
| 49 | power_steering_reservoir | tier B | 0.50 | 1.07 | 0.54 | 0.87 | 24 | P0505-P0507 (0.16), P0217 (0.10), P0520-P0524 (0.06) |
| 50 | transmission_oil_dipstick | tier B | 0.49 | 0.86 | 0.69 | 0.82 | 0 | P0217 (0.07), U0100 (0.07), P0505-P0507 (0.06) |
| 51 | clutch_reservoir | tier B | 0.45 | 0.79 | 0.65 | 0.88 | 0 | P0505-P0507 (0.06), P0217 (0.05), U0100 (0.05) |
| 52 | brake_fluid_reservoir | tier A | 0.44 | 0.77 | 0.66 | 0.87 | 149 | P0217 (0.05), P0505-P0507 (0.05), P0562 (0.04) |
| 53 | jump_start_post | tier B | 0.44 | 1.43 | 0.46 | 0.67 | 0 | P0562 (0.37), P0620-P0622 (0.23), P0600-P0606 (0.08) |
| 54 | hp_fuel_pump | tier B | 0.43 | 2.40 | 0.23 | 0.78 | 0 | P0172 (0.42), P0300 (0.35), P0171 (0.31) |
| 55 | washer_fluid_reservoir | tier A | 0.42 | 0.76 | 0.63 | 0.88 | 70 | P0217 (0.06), P0440-P0446 (0.06), P0455 (0.05) |
| 56 | oil_filter | tier B | 0.41 | 1.88 | 0.28 | 0.78 | 15 | P0520-P0524 (0.83), P0217 (0.19), P0300 (0.07) |
| 57 | hood_latch_mechanism | tier B | 0.41 | 1.17 | 0.59 | 0.59 | 24 | P0440-P0446 (0.15), P0217 (0.11), P0455 (0.09) |
| 58 | ac_compressor | tier B | 0.41 | 1.32 | 0.39 | 0.79 | 1 | P0217 (0.18), P0505-P0507 (0.18), P0300 (0.10) |
| 59 | wastegate_actuator | tier B | 0.37 | 1.37 | 0.32 | 0.84 | 0 | P0172 (0.13), P0217 (0.12), P0171 (0.11) |
| 60 | turbocharger | tier B | 0.32 | 2.27 | 0.17 | 0.82 | 4 | P0217 (0.24), P0520-P0524 (0.24), P0171 (0.19) |
| 61 | brake_master_cylinder | tier B | 0.20 | 0.57 | 0.42 | 0.84 | 0 | P0505-P0507 (0.04), P0100-P0103 (0.03), P0106-P0108 (0.03) |
| 62 | strut_tower_brace | tier B | 0.18 | 0.91 | 0.44 | 0.45 | 12 | P0300 (0.05), P0505-P0507 (0.05), P0562 (0.05) |
| 63 | abs_modulator_unit | tier B | 0.12 | 0.71 | 0.21 | 0.81 | 8 | U0100 (0.10), P0562 (0.07), P0600-P0606 (0.05) |
| 64 | starter_motor | tier B | 0.09 | 0.93 | 0.16 | 0.62 | 0 | P0562 (0.13), P0335-P0349 (0.06), P0620-P0622 (0.06) |
| 65 | windshield_wiper_motor | tier B | 0.09 | 0.56 | 0.23 | 0.69 | 3 | P0562 (0.04), P0100-P0103 (0.03), P0217 (0.03) |
| 66 | gas_cap | not in taxonomy v2 | 0.04 | 1.01 | 0.06 | 0.59 | 0 | P0440-P0446 (0.19), P0455 (0.14), P0171 (0.05) |

## Suggested first new classes (not yet trained)

1. **wiring_harness** (wiring harness) — priority 4.51, visible 0.34, 0 reviewed instances today
2. **vacuum_hose** (vacuum hose) — priority 3.23, visible 0.40, 0 reviewed instances today
3. **ect_sensor** (engine coolant temperature sensor) — priority 3.17, visible 0.51, 0 reviewed instances today
4. **breather_hose** (crankcase breather hose) — priority 3.00, visible 0.48, 0 reviewed instances today
5. **pcv_valve** (PCV valve) — priority 2.39, visible 0.41, 0 reviewed instances today

Caveats: Jev judges general automotive knowledge from text; check the ranking with the automotive expert; replace uniform DTC weights with Innova scan frequencies before deciding.
