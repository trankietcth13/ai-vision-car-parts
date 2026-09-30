# D2 geometry audit: air intake duct, battery terminal, battery

116 reviewed items (policy: docs/plans/label_policy_reservoirs_heat_shield.md §2a).

## sample (90)

| class | n | ok | needs_fix | split/merge | wrong_class / not_a_component |
|---|---|---|---|---|---|
| air_intake_duct | 30 | 26 | 3 | 0 | 1 |
| battery_terminal | 30 | 24 | 6 | 0 | 0 |
| battery | 30 | 19 | 10 | 0 | 1 |

## test_localisation (26)

| class | n | ok | needs_fix | split/merge | wrong_class / not_a_component |
|---|---|---|---|---|---|
| air_intake_duct | 10 | 4 | 6 | 0 | 0 |
| battery_terminal | 9 | 1 | 8 | 0 | 0 |
| battery | 7 | 1 | 5 | 0 | 1 |

Who is wrong on the test localisation misses: label_wrong 18, model_wrong 6, ambiguous 2

Corrections for the next build: data/engine_bay_review_geometry/CORRECTIONS.jsonl (41 entries).

## Applied and measured

- `apply_geometry_corrections.py` -> `data/engine_bay_reviewed{,_hybrid}_geo/` (39 images touched: 31 SAM2 re-segmentations,
  5 box fallbacks where SAM2 fragments a battery split by its hold-down, 2 drops, 1 reclass; 2 skipped < 0.5 confidence).
- Effect on the held-out test metrics (models unchanged, only the 18 corrected test images differ; `relabel_effect_d2.md`):
  teacher F1 0.612 -> 0.647 (recall on duct/terminal/battery 0.599 -> 0.708), student F1 0.531 -> 0.556.
  => most of the "localisation" misses were label errors; the same errors sit in train (battery 37% of sampled labels),
  so the next build (v10) should train on the corrected labels, and the corrected test labels become "test v2"
  (report both old and new test numbers for every model).
- Bug found by a reviewer and fixed: crops were rendered without EXIF rotation (4 items); open_upright() now used by
  every tool; only item 33's verdict was affected (overridden to ok).
