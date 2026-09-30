# Reservoir re-review, round 2 (pixel check, weak-evidence labels)

100 boxes: every accepted coolant / power-steering label identified by position or shape only (Jev `evidence`
field), plus a 20-box sample of brake labels. 5 expert subagents, originals at full resolution.
Verdicts: data/engine_bay_review_reservoir/verdicts/r2_batch_*.json.

## Real changes (after reading the label each verdict actually produces)

| round | boxes | rename | dropped (not a reservoir / outside ontology / reviewers disagree) | re-box |
|---|---|---|---|---|
| 1 (Jev-flagged queue) | 34 | 3 | 4 | 6 |
| 2 (weak evidence) | 100 | 6 | 10 | 25 |
| **total** | **134** | **9** | **14** | **31** |

- Coolant/PS boxes accepted on position or shape: **16 of 80 had the wrong identity** (brake 5, power steering 2, washer 2,
  hybrid inverter coolant tank 4, strut tower / battery / plenum 3). Brake boxes accepted on position: 19/20 correct
  (position on the master cylinder is a valid cue) but **10/20 need a new box** (shifted, cap cut off, two tanks in one box).
- Geometry is the larger issue: 25% of the round-2 boxes cut off the cap or part of the tank, or spill onto neighbours.

## New failure modes (added to the label policy)

1. **Hybrid inverter coolant tank labelled engine coolant** (Toyota THS, Honda i-MMD, Mirai): a second small tank next to
   the inverter / orange cables. Taxonomy v2 has `inverter_coolant_reservoir` (-> `other_reservoir`); the 36-class
   ontology does not, so these boxes are dropped from the v1 rebuild and must be re-added in the v2 build.
2. **White body parts taken for tanks** on white cars (strut tower with rubber mount cap).
3. **Earlier review inconsistencies**: some notes name another fluid than the final label; the correction tool now reads
   the label each verdict really produces (`new_class` also on `bad_geometry`).

## Not yet captured

Subagents also pointed at real reservoirs that have **no box** (e.g. ext0488 coolant tank at ≈[0.70,0.22,0.87,0.39],
the inverter tank of Request_ID_53 img_009). They are listed in the verdict `cues` text, not as structured `missing`
entries, so the rebuild does not add them yet.

## Apply

`python scripts/data_pipeline/apply_reservoir_corrections.py` wrote corrected verdict copies to
`data/<review root>_rr/` (originals untouched; summary data/engine_bay_review_reservoir/CHANGES.json). Rebuild on the
DGX with `apply_review.py --review data/<root>_rr ...` into a new reviewed dir, then a new dataset build (v10); never
edit an existing build.
