# Phase 2 label verification — handoff to the verifier

Audience: the agent that verifies the Phase 2 labels, either a Claude verifier subagent or Codex.
Roadmap step: docs/plans/engine_bay_accuracy_roadmap_2026-09-28.md, Phase 2 step 4.

## What happened before you

1. 332 images of `dataset/` had no reviewed label. Teacher v8 pseudo-labelled them.
2. An expert reviewer (the `engine-bay-label-review` skill) corrected those pseudo-labels.
3. The corrected labels were converted to the 21 training classes: `data/engine_bay_p2_view/labels/train`.
4. Teacher v8 and KD student v8 were run on these images. Every place where a model disagrees with a label is listed
   in the queue below. Each disagreement is either a **model mistake** or a **label mistake**. Your verdict decides which.

## Independence rules

- Judge only from the pixels. Do **not** open the expert review verdicts (`data/engine_bay_review_p2/verdicts/`)
  or the review report. The point of this step is a second, independent opinion.
- Be adversarial toward the label. The reviewer may have made systematic mistakes, especially between
  look-alike classes: brake fluid vs coolant reservoir, intake manifold vs engine cover, radiator cap vs
  coolant reservoir cap, alternator vs ignition coil.
- Be equally strict toward the model. A model detection is not evidence by itself.
- If you cannot decide with at least 0.8 confidence, answer `unsure`. An unsure verdict changes nothing.

## Inputs

| File | Content |
|---|---|
| `qa_results/review_queue_p2/train/ERRORS.json` | every image with its numbered errors |
| `qa_results/review_queue_p2/train/overlays/<image>` | overlay with the same numbers |
| `image_path` in each record | the clean image (`data/engine_bay_p2_view/images/train/...`) |
| `label_path` in each record | the label file; line k is `gt_id` k |
| `.claude/skills/engine-bay-label-review/SKILL.md` | visual definition of every class |
| `configs/engine_bay_train_classes.yaml` | the 21 training class names allowed in verdicts |

Overlay legend: green `G<k> class` = label k. Red `E<id> FP class` = model detection with no label.
Orange `E<id> FN|CLS class` = label the model missed or classified differently.

Each error has `error_id`, `type` (`FP`, `FN`, `CLS`, `FN/CLS`) and `box_norm`, plus `gt_id`/`gt_class`
for FN and CLS, `pred_class`/`conf` for FP, and `sources` (`teacher_v8`, `kd_n_v8`).

Always look at the overlay **and** the clean image before deciding. Mask tint can hide detail.

## Decisions

| decision | when | effect |
|---|---|---|
| `model_error` | the label is right, the model is wrong | none |
| `label_missing` | FP only: the detection is a real target component with no label | adds the instance (SAM2 mask from `box_norm`) |
| `label_wrong_class` | the labelled object is a real component, but the class is wrong | changes the class of `gt_id` |
| `label_extra` | the label marks something that is not a target component | removes `gt_id` |
| `label_bad_geometry` | right class, but the mask/box is clearly off | re-segments `gt_id` from `fixed_box_norm` |
| `unsure` | confidence below 0.8 | none |

Rules:

- Boxes are normalized `[x1, y1, x2, y2]` in [0, 1], x to the right, y down.
- Class names must be training classes. `radiator_hose` covers upper and lower hoses. `oil_filter` is not trained.
- For `label_missing`, `class_name` and `box_norm` default to the prediction. Give them only when the prediction's
  class or box is wrong.
- An FP that exists because a nearby label is misplaced: use `label_bad_geometry` on that label's `gt_id`.
- One physical object = one instance. Four coil packs = four instances.

## Output

One JSON file per image in the queue, `qa_results/verify_verdicts_p2/train/<image stem>.json`:

```json
{
  "image": "Request_ID_06__img_012__ab12cd34.jpg",
  "split": "train",
  "reviewer": "claude-verifier",
  "errors": [
    {"error_id": 0, "decision": "model_error"},
    {"error_id": 1, "decision": "label_missing", "class_name": "ignition_coil", "box_norm": [0.41, 0.22, 0.47, 0.31]},
    {"error_id": 2, "decision": "label_wrong_class", "gt_id": 3, "new_class": "brake_fluid_reservoir", "note": "small reservoir on master cylinder, DOT 4 cap"},
    {"error_id": 3, "decision": "label_extra", "gt_id": 5, "note": "harness connector"},
    {"error_id": 4, "decision": "label_bad_geometry", "gt_id": 1, "fixed_box_norm": [0.10, 0.52, 0.33, 0.80]},
    {"error_id": 5, "decision": "unsure", "note": "too dark"}
  ]
}
```

Use `"reviewer": "codex"` if Codex does the work. Give every error of the image a decision. A missing error counts as `unsure`.
Write valid JSON only, and do not modify any other file.

When every image in the queue has a verdict, create the empty file `qa_results/verify_verdicts_p2/DONE`.

## What happens next

`runs/logs/phase3_chain.sh` on the DGX applies the verdicts with `apply_codex_verdicts.py`, which validates every
verdict against the queue and lists rejected ones in `data/engine_bay_p2_verified/REJECTED_VERDICTS.json`.
It then builds v9 and the 28-vehicle dataset and starts the 5-fold cross-validation.
