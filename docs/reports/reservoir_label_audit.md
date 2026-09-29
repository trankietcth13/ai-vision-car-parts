# Reservoir / cap / heat-shield label audit (P1, Jev on expert review notes)

888 expert notes on boxes labelled or corrected to brake_fluid_reservoir, coolant_reservoir, exhaust_manifold_heat_shield, power_steering_reservoir, radiator_cap, washer_fluid_reservoir (review rounds: Qwen, DeepSeek hybrid, Phase 2). Jev model: pinned in src/jev/client.py. Jev reads the reviewer's text only; the pixel decision is the reviewer's.

## coolant_reservoir (VLM claimed / reviewer added: 277)

- decisions: wrong_class 78 (28%), correct 61 (22%), not_a_component 59 (21%), bad_geometry 36 (13%), missing_added 33 (12%), duplicate 10 (4%)
- accepted boxes cover: whole_body 82 (63%), other_object 23 (18%), partial 13 (10%), cap_only 8 (6%), not_stated 4 (3%)
- evidence used on accepted boxes: cap_marking 49 (38%), position 33 (25%), shape_material 31 (24%), fluid_colour 16 (12%), none 1 (1%)
- rejected boxes were actually: brake_fluid_reservoir 68 (46%), other_part 43 (29%), nothing 11 (7%), washer_fluid_reservoir 10 (7%), heat_shield_or_cover 8 (5%), power_steering_reservoir 6 (4%), radiator_cap 1 (1%)
- reviewer doubt (Jev noul ≥ 0.5): 15 / 277

Accepted but doubtful (re-check first):
  - `engine_bay_review/ext/ext0064#0` (0.97): translucent tank at firewall, likely coolant expansion tank
  - `engine_bay_review/ext/ext0003#2` (0.96): translucent white reservoir near firewall, likely coolant overflow
  - `engine_bay_review/ext/ext0236#4` (0.96): large translucent tank with blue fluid, likely FC coolant reservoir
  - `engine_bay_review/train/Request_ID_19__img_010__d7712a09#0` (0.95): opaque black pressure bottle with quarter-turn LOCK cap and vent hose at the neck; coolant reservoir most likely, cap icon not readable
  - `engine_bay_review_hybrid/train/Request_ID_07__img_047__dc296f5e#0` (0.94): translucent white plastic tank next to the radiator end tank seen from below; plausibly coolant overflow bottle, no cap/markings visible so 

## brake_fluid_reservoir (VLM claimed / reviewer added: 192)

- decisions: missing_added 74 (39%), not_a_component 35 (18%), bad_geometry 28 (15%), wrong_class 26 (14%), correct 19 (10%), duplicate 10 (5%)
- accepted boxes cover: whole_body 90 (74%), other_object 13 (11%), cap_only 9 (7%), partial 7 (6%), not_stated 2 (2%)
- evidence used on accepted boxes: position 84 (69%), cap_marking 23 (19%), shape_material 11 (9%), none 2 (2%), fluid_colour 1 (1%)
- rejected boxes were actually: other_part 35 (49%), coolant_reservoir 14 (20%), power_steering_reservoir 10 (14%), nothing 5 (7%), brake_fluid_reservoir 3 (4%), washer_fluid_reservoir 3 (4%), heat_shield_or_cover 1 (1%)
- reviewer doubt (Jev noul ≥ 0.5): 3 / 192

## washer_fluid_reservoir (VLM claimed / reviewer added: 94)

- decisions: missing_added 39 (41%), not_a_component 20 (21%), correct 14 (15%), bad_geometry 13 (14%), wrong_class 6 (6%), duplicate 2 (2%)
- accepted boxes cover: cap_only 51 (77%), whole_body 9 (14%), other_object 5 (8%), not_stated 1 (2%)
- evidence used on accepted boxes: cap_marking 36 (55%), position 15 (23%), fluid_colour 12 (18%), shape_material 3 (5%)
- rejected boxes were actually: other_part 18 (64%), nothing 3 (11%), power_steering_reservoir 2 (7%), coolant_reservoir 2 (7%), washer_fluid_reservoir 1 (4%), radiator_cap 1 (4%), brake_fluid_reservoir 1 (4%)
- reviewer doubt (Jev noul ≥ 0.5): 6 / 94

## radiator_cap (VLM claimed / reviewer added: 113)

- decisions: missing_added 39 (35%), bad_geometry 25 (22%), correct 18 (16%), not_a_component 16 (14%), duplicate 9 (8%), wrong_class 6 (5%)
- accepted boxes cover: cap_only 63 (77%), partial 8 (10%), other_object 7 (9%), not_stated 3 (4%), whole_body 1 (1%)
- evidence used on accepted boxes: position 42 (51%), cap_marking 36 (44%), none 2 (2%), fluid_colour 1 (1%), shape_material 1 (1%)
- rejected boxes were actually: coolant_reservoir 18 (58%), other_part 4 (13%), brake_fluid_reservoir 4 (13%), washer_fluid_reservoir 3 (10%), radiator_cap 2 (6%)
- reviewer doubt (Jev noul ≥ 0.5): 0 / 113

## power_steering_reservoir (VLM claimed / reviewer added: 31)

- decisions: not_a_component 17 (55%), missing_added 5 (16%), wrong_class 3 (10%), duplicate 2 (6%), bad_geometry 2 (6%), correct 2 (6%)
- accepted boxes cover: cap_only 3 (33%), whole_body 3 (33%), other_object 2 (22%), partial 1 (11%)
- evidence used on accepted boxes: cap_marking 5 (56%), position 3 (33%), none 1 (11%)
- rejected boxes were actually: other_part 13 (59%), brake_fluid_reservoir 3 (14%), nothing 3 (14%), washer_fluid_reservoir 2 (9%), heat_shield_or_cover 1 (5%)
- reviewer doubt (Jev noul ≥ 0.5): 0 / 31

## exhaust_manifold_heat_shield (VLM claimed / reviewer added: 135)

- decisions: not_a_component 75 (56%), missing_added 39 (29%), bad_geometry 11 (8%), correct 9 (7%), wrong_class 1 (1%)
- accepted boxes cover: whole_body 24 (41%), not_stated 14 (24%), other_object 13 (22%), partial 7 (12%), cap_only 1 (2%)
- evidence used on accepted boxes: shape_material 39 (66%), position 20 (34%)
- rejected boxes were actually: other_part 29 (38%), heat_shield_or_cover 29 (38%), nothing 18 (24%)
- reviewer doubt (Jev noul ≥ 0.5): 1 / 135

