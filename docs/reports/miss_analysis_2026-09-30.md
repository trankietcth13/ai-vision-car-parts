# Miss analysis on the 125 held-out test images (3 vehicles)

Every GT box of a class the model knows, categorised by the best prediction (any score ≥ 0.01, full image, no tiles):
detected = same class, IoU ≥ 0.5, score ≥ 0.25 · low_conf = same class IoU ≥ 0.5 but score < 0.25 · wrong_class = another class IoU ≥ 0.5 score ≥ 0.25 · localization = same class but best IoU 0.1-0.5 · not_found = nothing.

## teacher p5_reg (443 GT objects)

| category | n | share |
|---|---|---|
| detected | 258 | 58% |
| low_conf (0.01-0.25) | 66 | 15% |
| wrong_class | 18 | 4% |
| localization (IoU 0.1-0.5) | 40 | 9% |
| not_found | 61 | 14% |

| size (px at 640) | detected | low_conf (0.01-0.25) | wrong_class | localization (IoU 0.1-0.5) | not_found |
|---|---|---|---|---|---|
| <32 (n=35) | 69% | 6% | 0% | 11% | 14% |
| 32-96 (n=152) | 55% | 14% | 2% | 10% | 19% |
| >=96 (n=256) | 59% | 16% | 6% | 8% | 11% |

| vehicle | detected | low_conf (0.01-0.25) | wrong_class | localization (IoU 0.1-0.5) | not_found |
|---|---|---|---|---|---|
| Request_ID_13 (n=138) | 62% | 17% | 6% | 7% | 9% |
| Request_ID_23 (n=113) | 64% | 16% | 2% | 8% | 11% |
| Request_ID_29 (n=192) | 53% | 13% | 4% | 11% | 19% |

| class | n | detected | low_conf (0.01-0.25) | wrong_class | localization (IoU 0.1-0.5) | not_found |
|---|---|---|---|---|---|---|
| ignition_coil | 58 | 32 | 8 | 0 | 5 | 13 |
| battery_terminal | 53 | 37 | 5 | 0 | 9 | 2 |
| air_intake_duct | 47 | 23 | 8 | 0 | 10 | 6 |
| air_filter_box | 46 | 32 | 4 | 2 | 2 | 6 |
| battery | 37 | 22 | 8 | 1 | 6 | 0 |
| throttle_body | 23 | 14 | 3 | 0 | 0 | 6 |
| maf_sensor | 22 | 10 | 5 | 2 | 0 | 5 |
| oil_filler_cap | 21 | 16 | 3 | 0 | 0 | 2 |
| oil_dipstick | 21 | 16 | 1 | 0 | 1 | 3 |
| brake_fluid_reservoir | 20 | 13 | 3 | 1 | 1 | 2 |
| intake_manifold | 17 | 7 | 4 | 1 | 1 | 4 |
| engine_cover | 14 | 5 | 3 | 2 | 1 | 3 |
| ecu_module | 13 | 1 | 6 | 0 | 1 | 5 |
| alternator | 12 | 7 | 3 | 0 | 1 | 1 |
| fuse_relay_box | 11 | 2 | 1 | 7 | 0 | 1 |
| radiator_cap | 9 | 7 | 1 | 1 | 0 | 0 |
| multimeter_diagnostic_tool | 7 | 7 | 0 | 0 | 0 | 0 |
| washer_fluid_reservoir | 5 | 5 | 0 | 0 | 0 | 0 |
| radiator_hose | 4 | 2 | 0 | 0 | 1 | 1 |
| coolant_reservoir | 3 | 0 | 0 | 1 | 1 | 1 |

Top confusions (GT → predicted): fuse_relay_box→air_filter_box (7), maf_sensor→ignition_coil (2), engine_cover→air_filter_box (1), coolant_reservoir→brake_fluid_reservoir (1), radiator_cap→brake_fluid_reservoir (1), brake_fluid_reservoir→coolant_reservoir (1), battery→battery_terminal (1), intake_manifold→engine_cover (1)

## student kd_n_p5t_s0 (443 GT objects)

| category | n | share |
|---|---|---|
| detected | 211 | 48% |
| low_conf (0.01-0.25) | 80 | 18% |
| wrong_class | 15 | 3% |
| localization (IoU 0.1-0.5) | 51 | 12% |
| not_found | 86 | 19% |

| size (px at 640) | detected | low_conf (0.01-0.25) | wrong_class | localization (IoU 0.1-0.5) | not_found |
|---|---|---|---|---|---|
| <32 (n=35) | 37% | 29% | 0% | 11% | 23% |
| 32-96 (n=152) | 45% | 18% | 1% | 14% | 22% |
| >=96 (n=256) | 51% | 16% | 5% | 10% | 17% |

| vehicle | detected | low_conf (0.01-0.25) | wrong_class | localization (IoU 0.1-0.5) | not_found |
|---|---|---|---|---|---|
| Request_ID_13 (n=138) | 55% | 14% | 4% | 12% | 16% |
| Request_ID_23 (n=113) | 47% | 23% | 0% | 11% | 19% |
| Request_ID_29 (n=192) | 43% | 18% | 5% | 12% | 22% |

| class | n | detected | low_conf (0.01-0.25) | wrong_class | localization (IoU 0.1-0.5) | not_found |
|---|---|---|---|---|---|---|
| ignition_coil | 58 | 23 | 15 | 0 | 4 | 16 |
| battery_terminal | 53 | 29 | 10 | 0 | 13 | 1 |
| air_intake_duct | 47 | 23 | 3 | 1 | 8 | 12 |
| air_filter_box | 46 | 24 | 11 | 3 | 2 | 6 |
| battery | 37 | 17 | 9 | 0 | 7 | 4 |
| throttle_body | 23 | 12 | 2 | 0 | 3 | 6 |
| maf_sensor | 22 | 6 | 4 | 2 | 5 | 5 |
| oil_filler_cap | 21 | 10 | 6 | 0 | 1 | 4 |
| oil_dipstick | 21 | 11 | 6 | 0 | 0 | 4 |
| brake_fluid_reservoir | 20 | 15 | 1 | 0 | 1 | 3 |
| intake_manifold | 17 | 8 | 3 | 1 | 3 | 2 |
| engine_cover | 14 | 5 | 1 | 3 | 1 | 4 |
| ecu_module | 13 | 2 | 4 | 1 | 0 | 6 |
| alternator | 12 | 3 | 3 | 0 | 1 | 5 |
| fuse_relay_box | 11 | 5 | 0 | 3 | 0 | 3 |
| radiator_cap | 9 | 5 | 2 | 0 | 1 | 1 |
| multimeter_diagnostic_tool | 7 | 7 | 0 | 0 | 0 | 0 |
| washer_fluid_reservoir | 5 | 4 | 0 | 0 | 0 | 1 |
| radiator_hose | 4 | 2 | 0 | 0 | 1 | 1 |
| coolant_reservoir | 3 | 0 | 0 | 1 | 0 | 2 |

Top confusions (GT → predicted): fuse_relay_box→air_filter_box (3), engine_cover→air_filter_box (3), maf_sensor→ignition_coil (2), coolant_reservoir→brake_fluid_reservoir (1), air_filter_box→fuse_relay_box (1), intake_manifold→air_filter_box (1), ecu_module→air_intake_duct (1), air_filter_box→engine_cover (1)

