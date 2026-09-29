# QA error verification — handoff to Codex

Purpose: confirm or reject the disagreements between the trained engine-bay models and the dataset labels.
Your verdicts decide what gets fixed in the dataset before the next re-train. You do not need the DGX or any GPU for this task.

## Inputs

Review **only** the unified queue (it merges the teacher and KD-student reports, so every image has one error-id space):

| file | content |
|---|---|
| `qa_results/review_queue/test/ERRORS.json` + `overlays/` | 94 test images, 439 merged errors (held-out split) |
| `qa_results/review_queue/train/ERRORS.json` + `overlays/` | 200 train images, 368 errors (hard examples) |
| `qa_results/QA_REPORT.md` | test metrics of teacher v4, KD student and baseline student (context only) |

The per-model folders `errors_teacher_v4/` and `errors_kd_n/` are raw inputs of the queue; do not use their error ids.

Overlay legend: **green `G<k> class`** = dataset label number k (line k of the label file, this is the `gt_id`),
**red `E<id> FP class`** = model detection with no label, **orange `E<id> FN|CLS class`** = label the model missed or
classified differently. Each error in `ERRORS.json` has:

- `error_id` (use this in your verdict), `type` (`FP`, `FN`, `CLS`, `FN/CLS`), `box_norm`
- `gt_id` / `gt_class` for FN and CLS, `pred_class` / `conf` for FP
- `sources`: which models produced it (`teacher_v4`, `kd_n`), and `model_view` for FN/CLS

Class definitions and visual cues: `.claude/skills/engine-bay-label-review/SKILL.md` (training uses the 21 classes in
`configs/engine_bay_train_classes.yaml`). Look at the overlay **and** the original image (`image_path`).

## What to decide for every error

Write one JSON file per image to `qa_results/codex_verdicts/<split>/<image stem>.json`:

```json
{
  "image": "Request_ID_23__img_004__ab12cd34.jpg",
  "split": "test",
  "reviewer": "codex",
  "errors": [
    {"error_id": 0, "decision": "model_error"},
    {"error_id": 1, "decision": "label_missing", "class_name": "ignition_coil", "box_norm": [0.41, 0.22, 0.47, 0.31]},
    {"error_id": 2, "decision": "label_wrong_class", "gt_id": 3, "new_class": "brake_fluid_reservoir"},
    {"error_id": 3, "decision": "label_extra", "gt_id": 5},
    {"error_id": 4, "decision": "label_bad_geometry", "gt_id": 1, "fixed_box_norm": [0.10, 0.52, 0.33, 0.80]},
    {"error_id": 5, "decision": "unsure", "note": "too dark to tell"}
  ]
}
```

| decision | when | effect on dataset |
|---|---|---|
| `model_error` | the label is right, the model is wrong | none (train images get oversampled as hard examples) |
| `label_missing` | FP is a real component that has no label (FP errors only; `class_name`/`box_norm` default to the prediction) | add instance (SAM2 mask from `box_norm`) |
| `label_wrong_class` | the labelled object is real but the class is wrong | change class of `gt_id` |
| `label_extra` | the label marks something that is not a target component | remove `gt_id` |
| `label_bad_geometry` | right class, box/mask clearly off | re-segment `gt_id` from `fixed_box_norm` |
| `unsure` | cannot decide | none |

Rules: box coordinates are normalized `[x1, y1, x2, y2]` in [0, 1]; class names must be training classes; only mark
`label_*` when you are confident (≥ 0.8). An error you skip counts as `unsure`.
For FN/CLS errors `gt_id` is taken from the error automatically. For an FP that exists because a nearby label is
misplaced (e.g. the label box sits next to the real part), use `label_bad_geometry` with that label's `gt_id` and a
`fixed_box_norm`. `apply_codex_verdicts.py` validates every verdict against the queue (image, error_id, decision vs
error type, gt_id exists, class names, box format) and lists rejected ones in `REJECTED_VERDICTS.json` of the output dataset.

## Important: test vs train

- **Test split**: label fixes are applied to the test labels only. Test images are **never** moved into training,
  otherwise the test score stops measuring anything.
- **Train split**: label fixes go into the next training dataset (v5) and `model_error` images are oversampled.

## When you are done

Create an empty file `qa_results/codex_verdicts/DONE`. A watcher (`auto_retrain_after_codex.sh`) is running on the
workstation: as soon as `DONE` appears (or every error image has a verdict) it runs all the steps below automatically
and starts the retraining on the DGX. You do not need to run anything yourself.

## After you finish (manual equivalent)

Run (on the machine that has DGX access):

```bash
python scripts/operations/dgx_train.py pull-qa           # if not yet pulled
python scripts/evaluation/apply_codex_verdicts.py        # builds data/engine_bay_train_v5 (+ fixed test labels)
# enlarge val from 3 to 6 vehicles (Request_ID_50/52/36 leave train; only their reviewed images enter val)
python scripts/data_pipeline/resplit_val.py --src data/engine_bay_train_v5 --out data/engine_bay_train_v5s
python scripts/operations/dgx_train.py push --dataset data/engine_bay_train_v5s
python scripts/operations/dgx_train.py teacher --name engine_teacher_v5_dgx --dataset engine_bay_train_v5s --model yolo11l-seg.pt --batch 8 --epochs 60 --patience 20
```

Note: models trained before this re-split (teacher v4, KD/baseline students) saw Request_ID_50/52/36 during
training, so compare them with v5 on the **test** split only, never on the new val split.
