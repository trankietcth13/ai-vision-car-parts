# Reservoir re-review, round 1 (pixel check of the Jev-flagged queue)

34 boxes checked by 2 expert subagents on the images (24 coolant/brake + 10 sampled washer). Verdicts: data/engine_bay_review_reservoir/verdicts/.

- identity changes: **7** (21%); coolant/brake subset: 7/24
- needs re-box: 5 · low confidence (< 0.6, blurry/dark, no readable cap): 6
- box policy: {'whole_body_ok': 16, 'needs_rebox': 5, 'cap_only_ok': 13}

| key | label | → identity | conf | cues |
|---|---|---|---|---|
| engine_bay_review/ext/ext0003#2 | coolant_reservoir | brake_fluid_reservoir | 0.45 | White translucent box cut off by the top edge at the firewall, on the driver side of an RH |
| engine_bay_review/train/Request_ID_03__img_041__641db939#9 | brake_fluid_reservoir | washer_fluid_reservoir | 0.97 | Blue cap embossed 'WASHER ONLY' with a windshield+spray icon, on a long filler neck at the |
| engine_bay_review/train/Request_ID_07__img_013__0ab039bc#2 | coolant_reservoir | power_steering_reservoir | 0.7 | Small round cream reservoir with a black screw cap, a mounting flange with a bolt and a sm |
| engine_bay_review/train/Request_ID_07__img_021__c95cf5ed#2 | coolant_reservoir | power_steering_reservoir | 0.7 | The same Chrysler 3.6 PS reservoir (cylindrical cream body, moulded MAX/MIN, mounting flan |
| engine_bay_review/train/Request_ID_07__img_051__a542de85#2 | coolant_reservoir | radiator_cap | 0.85 | Metal pressure cap embossed 'DO NOT OPEN HOT' on a filler neck at the front radiator suppo |
| engine_bay_review_hybrid/train/Request_ID_07__img_047__dc296f5e#0 | coolant_reservoir | washer_fluid_reservoir | 0.4 | underbody view; large translucent tank low in the front corner next to the radiator lower  |
| engine_bay_review_p2/p2/Request_ID_53__img_009__bd6e0755#2 | brake_fluid_reservoir | coolant_reservoir | 0.85 | metal radiator-type pressure cap with red warning label on a translucent tank with an info |

## Findings

- Errors come from position-only guesses when the cap is not readable; where the cap marking was readable the label was right. This is rule 1 of docs/plans/label_policy_reservoirs_heat_shield.md (≥ 2 independent cues).
- The Jev `cap_only` flag over-flags: the review notes often described the SAM mask, not the box; most flagged coolant/brake boxes already covered the whole body. Jev structured the notes correctly — the notes themselves are ambiguous. Use the Jev fields to prioritise, never to change labels directly.
- Washer: 11/11 sampled cap-only boxes are correct (tank hidden under fender/bumper); the filler neck is included on some cars and not others → policy now: include the visible translucent filler neck with the cap.
- Corrections for the next dataset build: docs/reports/reservoir_corrections_round1.jsonl (10 entries; apply through the review/apply tooling, not by hand-editing a build).
