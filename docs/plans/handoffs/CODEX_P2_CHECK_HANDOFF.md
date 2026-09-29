# Check DeepSeek labels on 179 engine-bay images — handoff to Codex

Purpose: DeepSeek V4 Flash Vision drew 462 boxes on 179 photos from `dataset/` that had no label. DeepSeek is known to
**hallucinate** on this kind of image: on earlier reviews it added boxes where nothing was there, and 69 of its boxes
here are `exhaust_manifold_heat_shield`, which often means an underbody photo. Your job is to check every DeepSeek box,
fix or delete the wrong ones, add clearly visible components it missed, and flag photos that are not engine bays.
You do not need the DGX, a GPU or any API for this task.

This is the same kind of work as the 153 images you labelled in `CODEX_P2_LABEL_HANDOFF.md`, but starting from
DeepSeek's boxes instead of an empty image. Your output becomes the draft that SAM2 turns into masks, followed by an
expert review and an independent verification.

## Inputs

| File | Content |
|---|---|
| `data/engine_bay_codex_p2_check/IMAGES.txt` | the 179 clean image paths |
| `data/engine_bay_codex_p2_check/overlays/<stem>.jpg` | the image with every DeepSeek box drawn in red as `#<id> class` |
| `data/engine_bay_codex_p2_check/deepseek/<stem>.json` | the same boxes as data (`id`, `class_name`, `bbox_norm_xyxy`) |
| `data/engine_bay_phase2/upload/<stem>.jpg` | the clean image (1600 px long side) |
| `.claude/skills/engine-bay-label-review/SKILL.md` | the 36-class table: appearance and usual confusions |
| `configs/data_engine_bay.yaml` | the 36 class names; use these names exactly |

Always look at the overlay **and** the clean image. Neighbouring images of the same vehicle (same `Request_ID_xx`)
show the same parts from other angles and help identify them.

## Scene first

Decide the scene before judging boxes:

- `not_engine_bay`: underbody (exhaust pipe, catalytic converter, underbody panels, lift arms), exterior, fuel filler
  door, interior. Set `exclude: true` and return no detections. DeepSeek's boxes on these photos are all wrong.
- `close_up`: part of the engine bay, or a removed engine-bay part on a bench. Keep it (`exclude: false`); label only
  what is a target component.
- `partial_engine_bay` / `full_engine_bay`: normal case.

## Decision per DeepSeek box

For every DeepSeek `id`:

| decision | when |
|---|---|
| `correct` | right class, box fits the visible object reasonably well |
| `wrong_class` | a real target component but another class; give `new_class` |
| `bad_box` | right class but the box is clearly off (cuts off > 30 % or covers a lot of other things) |
| `not_a_component` | not any of the 36 classes (connector, bracket, hose not in the list, hallucination) |
| `duplicate` | the same object is already covered by another box; give `duplicate_of` |

## What goes into `detections`

`detections` is the **final** list for the image, the one that will be used:

- every `correct` box, unchanged;
- every `wrong_class` box with the new class;
- every `bad_box` box with your corrected coordinates;
- plus every clearly visible target component DeepSeek missed.

Rules: one box per physical object (four coil packs = four boxes); boxes tight to the visible extent; add only what
you identify with confidence ≥ 0.8. When unsure, leave it out.
Look-alikes to separate carefully: brake fluid vs coolant reservoir, intake manifold vs engine cover, radiator cap vs
coolant reservoir cap, alternator vs ignition coil, exhaust heat shield vs engine cover.

## Output

One JSON file per image: `data/engine_bay_codex_p2_check/labels/<stem>.json` (UTF-8, valid JSON, nothing else).

```json
{
  "image": "Request_ID_06__img_012__ab12cd34",
  "labeler": "codex",
  "scene": "full_engine_bay | partial_engine_bay | close_up | not_engine_bay",
  "deepseek_review": [
    {"id": 0, "decision": "correct"},
    {"id": 1, "decision": "wrong_class", "new_class": "brake_fluid_reservoir"},
    {"id": 2, "decision": "not_a_component", "note": "harness connector"},
    {"id": 3, "decision": "bad_box"},
    {"id": 4, "decision": "duplicate", "duplicate_of": 0}
  ],
  "detections": [
    {"class_name": "ignition_coil", "bbox_norm_xyxy": [0.41, 0.22, 0.47, 0.31], "confidence": 0.9, "visual_evidence": "DeepSeek #0"},
    {"class_name": "brake_fluid_reservoir", "bbox_norm_xyxy": [0.70, 0.10, 0.80, 0.25], "confidence": 0.9, "visual_evidence": "DeepSeek #1, DOT 4 cap"},
    {"class_name": "throttle_body", "bbox_norm_xyxy": [0.30, 0.52, 0.41, 0.66], "confidence": 0.85, "visual_evidence": "DeepSeek #3 re-boxed"},
    {"class_name": "oil_dipstick", "bbox_norm_xyxy": [0.36, 0.88, 0.40, 0.95], "confidence": 0.85, "visual_evidence": "missed by DeepSeek, yellow ring"}
  ],
  "exclude": false,
  "comment": "one short sentence"
}
```

- `bbox_norm_xyxy` is `[x1, y1, x2, y2]` normalized to [0, 1], x to the right, y down, x1 < x2, y1 < y2.
- `class_name` must be one of the 36 names in `configs/data_engine_bay.yaml`.
- Give a decision for every DeepSeek id. Write one file for every image in `IMAGES.txt`. Do not modify other files.

When all 179 files exist, create the empty file `data/engine_bay_codex_p2_check/DONE`.

## After you finish (Claude runs this)

```bash
python scripts/data_pipeline/prepare_p2_deepseek.py codex --codex data/engine_bay_codex_p2_check
```

It validates your files, replaces DeepSeek's boxes with your `detections`, and reports DeepSeek's precision from your
`deepseek_review`. Invalid entries are listed in `data/engine_bay_codex_p2_check/REJECTED.json`.
