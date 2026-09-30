# Model comparison (2026-09-30)

## Held-out test: 125 images of 3 vehicles (Request_ID_13/23/29), box IoU ≥ 0.5 + class, conf 0.25

⚠ kd_n_full and teacher_full were trained on ALL 28 vehicles, including these 3: their numbers here are optimistic.

| model | mode | recall | precision | F1 | GPU s/img |
|---|---|---|---|---|---|
| kd_n_full (deploy, trained on all 28 vehicles) | full image | 0.942 | 0.804 | 0.868 | 0.02 |
| teacher_full (server, trained on all 28 vehicles) | full image | 0.973 | 0.883 | 0.926 | 0.03 |
| p5_reg teacher (held-out test vehicles) | full image | 0.575 | 0.684 | 0.625 | 0.03 |
| p5_reg teacher (held-out test vehicles) | full + 2x2 tiles | 0.606 | 0.611 | 0.609 | 0.09 |
| kd_n_p5t_s0 student (held-out test vehicles) | full image | 0.465 | 0.626 | 0.534 | 0.02 |

Recall by system:

| model · mode | air_intake | electrical | ignition | lubrication | brakes | cooling | body | tools | washer |
|---|---|---|---|---|---|---|---|---|---|
| kd_n_full (deploy, trained on all 28 vehicles) · full image | 0.97 | 0.91 | 0.97 | 0.86 | 0.95 | 0.88 | 1.00 | 1.00 | 1.00 |
| teacher_full (server, trained on all 28 vehicles) · full image | 0.99 | 0.96 | 0.98 | 0.93 | 0.95 | 1.00 | 1.00 | 1.00 | 1.00 |
| p5_reg teacher (held-out test vehicles) · full image | 0.54 | 0.55 | 0.52 | 0.76 | 0.68 | 0.56 | 0.36 | 1.00 | 1.00 |
| p5_reg teacher (held-out test vehicles) · full + 2x2 tiles | 0.58 | 0.57 | 0.53 | 0.83 | 0.68 | 0.62 | 0.36 | 1.00 | 1.00 |
| kd_n_p5t_s0 student (held-out test vehicles) · full image | 0.44 | 0.43 | 0.41 | 0.50 | 0.77 | 0.44 | 0.36 | 1.00 | 0.80 |

## Unseen vehicles (no ground truth): components found (conf 0.25, full + tiles)

| image | kd_n_full | teacher_full | p5_reg | kd_n_p5t_s0 |
|---|---|---|---|---|
| 1971_Triumph_spitfire_2 | 8 | 6 | 14 | 6 |
| 1971_Triumph_spitfire_3 | 9 | 6 | 12 | 7 |
| 1974_unknown-make_unknown-model_4 | 7 | 4 | 6 | 4 |
| 2000_unknown-make_unknown-model_7 | 11 | 7 | 5 | 4 |
| 2004_Acura_TL_11 | 4 | 5 | 6 | 5 |
| 2004_Acura_TL_16 | 5 | 4 | 5 | 5 |
| 2011_Chevrolet_Volt_65 | 10 | 6 | 9 | 7 |
| 2013-2017_Honda_Accord_67 | 4 | 4 | 5 | 7 |
| 2015-2023_Ford_Mustang_EcoBoost_76 | 3 | 3 | 6 | 3 |
| jeep_2.0T | 7 | 4 | 7 | 7 |
| unknown-year_Ford_fossil_fuel_hogg_in_94 | 1 | 2 | 1 | 2 |
| unknown-year_Porsche_911_GT1_turbines_Callas_96 | 3 | 1 | 3 | 2 |
| unknown-year_Porsche_potentialtokinetic_power_powerplant_pow | 0 | 1 | 3 | 6 |
| unknown-year_Porsche_power_powersources_questions_racing_98 | 3 | 3 | 3 | 4 |

Viewers: artifacts/model_test/unseen_results/<model>/<image>.html · overview: artifacts/model_test/index.html

## Visual check on unseen vehicles (4 models side by side)

- Honda Accord battery close-up: p5_reg is the only model that finds the battery body and BOTH terminals; kd_n_full and
  teacher_full miss the negative terminal; the students split the battery into two boxes. All models add a false
  air_intake_duct on a corrugated wire loom and a battery_terminal on a loose cable clamp.
- Chevrolet Volt (plug-in hybrid): every model finds engine cover, brake reservoir, oil cap, dipstick. The **high-voltage
  inverter cover with orange cables is called engine_cover / air_filter_box** and the coolant surge tank cap is called
  oil_filler_cap. There is no HV class: a safety gap for disassembly guidance (automotive-expert rule: HV parts must be
  identified and never guided). Candidate: an `hv_component` class (orange cables / inverter) in taxonomy v2.
- Counts per image are not quality: kd_n_full reports the most parts on old cars (more false positives), teacher_full
  the fewest (most conservative).

## Which model to use for what

| purpose | model |
|---|---|
| demo / deployment on known vehicles | kd_n_full (web app default) or teacher_full (server) |
| honest numbers on new vehicles, labelling new images (teacher_prelabel.py) | p5_reg teacher (+ tiles for recall) |
| edge student on new vehicles | kd_n_p5t_s0 avg5 until E1 / v2 results arrive |

Run on the DGX GPU (model_test/compare_models_dgx.py): whole comparison < 1 min vs > 20 min on the laptop CPU.
