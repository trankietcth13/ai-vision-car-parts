# DINOv2 crop retrieval probe (W1d)

Model `facebook/dinov2-small` · gallery = train split (2788 crops) · queries = val+test (669 crops, other vehicles) · classes = taxonomy-v2 training classes

**Overall top-1:** kNN (k=10) 0.625 · linear probe 0.598

| class | n query | kNN acc | linear acc | most confused with (linear) |
|---|---|---|---|---|
| battery | 53 | 0.66 | 0.58 | air_filter_box (4), multimeter_diagnostic_tool (4) |
| battery_terminal | 67 | 0.57 | 0.76 | battery (5), brake_fluid_reservoir (2) |
| fuse_relay_box | 15 | 0.67 | 0.73 | battery_terminal (2), ecu_module (1) |
| ecu_module | 13 | 0.23 | 0.46 | fuse_relay_box (4), other_hose_line (2) |
| alternator | 14 | 0.71 | 0.79 | radiator_cap (1), engine_cover (1) |
| ignition_coil | 77 | 0.83 | 0.58 | battery_terminal (10), maf_sensor (7) |
| air_filter_box | 73 | 0.41 | 0.34 | air_intake_duct (15), maf_sensor (6) |
| air_intake_duct | 81 | 0.74 | 0.38 | other_hose_line (12), maf_sensor (10) |
| maf_sensor | 31 | 0.35 | 0.48 | air_intake_duct (5), ignition_coil (3) |
| throttle_body | 30 | 0.47 | 0.80 | radiator_hose (1), other_rotating_device (1) |
| intake_manifold | 31 | 0.65 | 0.52 | other_hose_line (8), multimeter_diagnostic_tool (2) |
| exhaust_manifold_heat_shield | 13 | 0.54 | 0.62 | other_hose_line (2), coolant_reservoir (1) |
| coolant_reservoir | 10 | 0.00 | 0.00 | exhaust_manifold_heat_shield (3), brake_fluid_reservoir (2) |
| radiator_cap | 18 | 0.89 | 0.94 | washer_fluid_reservoir (1) |
| radiator_hose | 12 | 0.25 | 0.33 | other_hose_line (3), multimeter_diagnostic_tool (2) |
| oil_filler_cap | 32 | 0.78 | 0.84 | ignition_coil (3), washer_fluid_reservoir (1) |
| oil_dipstick | 30 | 0.80 | 0.80 | oil_filler_cap (3), alternator (1) |
| brake_fluid_reservoir | 25 | 0.88 | 0.92 | exhaust_manifold_heat_shield (1), fuse_relay_box (1) |
| washer_fluid_reservoir | 8 | 0.75 | 0.75 | battery_terminal (1), alternator (1) |
| engine_cover | 18 | 0.44 | 0.61 | air_filter_box (2), intake_manifold (2) |
| multimeter_diagnostic_tool | 13 | 0.92 | 0.92 | other_hose_line (1) |
| other_module_box | 4 | 0.00 | 0.50 | fuse_relay_box (1), other_hose_line (1) |
| other_rotating_device | 1 | 0.00 | 0.00 | exhaust_manifold_heat_shield (1) |

**reservoirs** (n=43): kNN 0.651 · linear 0.674

**caps_small** (n=147): kNN 0.701 · linear 0.810

**boxes** (n=203): kNN 0.522 · linear 0.493

## Against the teacher (class-agnostic IoU ≥ 0.5 matches)

Teacher `runs/segment/p5_reg/weights/avg5.pt` · query instances matched: 429/669

- teacher class correct: 381 · wrong class: 48
- linear probe fixes 21/48 teacher class errors, but would break 115/381 correct ones if it always overrode
- fusion (override if teacher conf < 0.5 and probe p ≥ 0.6): class acc on matched 0.879 vs teacher 0.888 (4 overrides)
- fusion (override if teacher conf < 0.4 and probe p ≥ 0.7): class acc on matched 0.883 vs teacher 0.888 (2 overrides)
- fusion (override if teacher conf < 0.3 and probe p ≥ 0.8): class acc on matched 0.886 vs teacher 0.888 (1 overrides)

Teacher class errors (GT → predicted): fuse_relay_box→air_filter_box (8), exhaust_manifold_heat_shield→coolant_reservoir (6), air_filter_box→fuse_relay_box (4), engine_cover→air_filter_box (4), intake_manifold→air_filter_box (4), air_intake_duct→air_filter_box (2), brake_fluid_reservoir→coolant_reservoir (2), coolant_reservoir→radiator_cap (1), maf_sensor→air_intake_duct (1), engine_cover→air_intake_duct (1)
