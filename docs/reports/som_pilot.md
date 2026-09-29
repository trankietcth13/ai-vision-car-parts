# Set-of-Mark labelling pilot (W1b-c)

60 bake-off images (val+test vehicles), 221 reviewed objects, box IoU ≥ 0.5, all names mapped to taxonomy-v2 training classes (tier-B names -> generic class).

| system | pred/img | precision | recall | F1 | class-agnostic P / R |
|---|---|---|---|---|---|
| Qwen3-VL (existing) | 3.8 | 41.6% | 42.5% | 0.421 | 45.6% / 46.6% |
| DeepSeek direct boxes (cached bake-off) | 4.0 | 45.4% | 49.3% | 0.473 | 48.8% / 52.9% |
| teacher p5_reg alone (conf 0.25) | 3.7 | 59.1% | 58.8% | 0.590 | 63.6% / 63.3% |
| SoM: teacher+SAM2 regions, deepseek-v4-flash-vision-exp names | 8.5 | 17.2% | 39.8% | 0.240 | 24.8% / 57.5% |

## Per class F1 (≥ 5 reviewed objects)

| class | GT | Qwen3-VL (existing) | DeepSeek direct boxes (cached bake-off) | teacher p5_reg alone (conf 0.25) | SoM: teacher+SAM2 regions, deepseek-v4-flash-vision-exp names |
|---|---|---|---|---|---|
| ignition_coil | 35 | 0.32 | 0.51 | 0.61 | 0.39 |
| air_intake_duct | 25 | 0.50 | 0.45 | 0.57 | 0.33 |
| battery_terminal | 19 | 0.47 | 0.47 | 0.49 | 0.36 |
| air_filter_box | 18 | 0.84 | 0.70 | 0.75 | 0.39 |
| battery | 15 | 0.73 | 0.65 | 0.60 | 0.16 |
| oil_filler_cap | 15 | 0.29 | 0.33 | 0.69 | 0.40 |
| intake_manifold | 13 | 0.67 | 0.82 | 0.44 | 0.11 |
| oil_dipstick | 11 | 0.00 | 0.29 | 0.72 | 0.17 |
| throttle_body | 11 | 0.56 | 0.76 | 0.74 | 0.24 |
| maf_sensor | 9 | 0.35 | 0.35 | 0.67 | 0.33 |
| brake_fluid_reservoir | 8 | 0.24 | 0.35 | 0.67 | 0.46 |
| engine_cover | 7 | 0.43 | 0.50 | 0.55 | 0.14 |
| radiator_cap | 6 | 0.50 | 0.50 | 0.83 | 0.55 |
| radiator_hose | 5 | 0.18 | 0.14 | 0.50 | 0.14 |
| fuse_relay_box | 5 | 0.25 | 0.29 | 0.31 | 0.12 |
| washer_fluid_reservoir | 5 | 0.22 | 0.40 | 0.53 | 0.20 |

## Names outside the reviewed ontology (tier-B / new components the SoM run proposed; not scorable yet)

wiring_harness (183), o2_sensor (19), vacuum_hose (14), exhaust_manifold (13), valve_cover (12), hood_latch_mechanism (4), strut_tower_brace (4), fuel_rail (3), fuel_injector (2), oil_filter (2), ac_compressor (2), power_steering_reservoir (2), windshield_wiper_motor (2), map_sensor (1), spark_plug_wire (1), brake_booster (1), brake_master_cylinder (1), power_steering_pump (1)
