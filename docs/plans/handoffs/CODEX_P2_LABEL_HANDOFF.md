# Label 153 engine-bay images — handoff to Codex

Purpose: these 153 photos from `dataset/` have no label yet. Qwen3-VL and DeepSeek V4 Flash Vision both returned
**no component** on them, but vision models miss about 30 % of real components, so many of these images probably do
contain components. Your job is to draw a box around every clearly visible target component, or to confirm that the
image really has none. You do not need the DGX, a GPU or any API for this task.

What happens after you: Claude turns your boxes into masks (SAM2 on the DGX), an expert reviewer checks every image,
and an independent verifier checks the disagreements. Your labels are a first draft, but a careful draft saves a lot of
correction work downstream.

## Inputs

| File | Content |
|---|---|
| `data/engine_bay_codex_p2/IMAGES.txt` | the 153 image paths, one per line |
| `data/engine_bay_phase2/upload/<stem>.jpg` | the images (resized to 1600 px on the long side) |
| `.claude/skills/engine-bay-label-review/SKILL.md` | the 36-class table: what each component looks like and its usual confusions |
| `configs/data_engine_bay.yaml` | the 36 class names (ids 0–35); use these names exactly |

Most images are from 20 vehicles, several per vehicle (Request_ID_06 has 25, Request_ID_03 has 22). Photos of the same
vehicle show the same parts from other angles, so looking at neighbouring images of that vehicle helps you identify a part.

## What to label

- Every **clearly visible** component of the 36 classes. Hidden or implied parts are not labelled.
- One box per physical object. Four coil packs = four boxes. Two battery terminals = two boxes.
- The box covers the **visible extent** of the object, tight to its outline. A box twice too large is wrong.
- Frequent real misses to look for: ignition coils in a row on the valve cover, oil dipstick handles (yellow/orange ring),
  oil filler cap, air intake duct, throttle body, battery terminals, brake fluid reservoir near the firewall.
- Look-alikes to separate carefully: brake fluid vs coolant reservoir (small, DOT cap, near the firewall), intake manifold
  vs engine cover (runners vs smooth decorative cover), radiator cap vs coolant reservoir cap, alternator vs ignition coil.
- Only label with confidence ≥ 0.8. When unsure, leave it out: a missed part is cheaper to fix than a wrong one.
- If the image is a close-up where none of the 36 classes is visible, return an empty `detections` list. That is a valid answer.

## Output

One JSON file per image: `data/engine_bay_codex_p2/labels/<stem>.json` (UTF-8, valid JSON, nothing else).

```json
{
  "image": "Request_ID_06__img_012__ab12cd34",
  "labeler": "codex",
  "scene": "full_engine_bay | partial_engine_bay | close_up | not_engine_bay",
  "detections": [
    {"class_name": "ignition_coil", "bbox_norm_xyxy": [0.41, 0.22, 0.47, 0.31], "confidence": 0.9,
     "visual_evidence": "coil-on-plug unit with connector, second of four on the valve cover"},
    {"class_name": "oil_dipstick", "bbox_norm_xyxy": [0.36, 0.88, 0.40, 0.95], "confidence": 0.85,
     "visual_evidence": "yellow ring handle"}
  ],
  "exclude": false,
  "comment": "one short sentence"
}
```

- `bbox_norm_xyxy` is `[x1, y1, x2, y2]` normalized to [0, 1] of the image width/height, x to the right, y down,
  with x1 < x2 and y1 < y2. Estimate positions by reading the image as a grid (e.g. left edge at ~35 % of the width).
- `class_name` must be one of the 36 names in `configs/data_engine_bay.yaml`.
- Set `exclude: true` only for images that are not engine bays (underbody, exterior, interior) or unusable (blur, dark).
- Write one file for every image in `IMAGES.txt`, including empty ones. Do not modify any other file.

When all 153 files exist, create the empty file `data/engine_bay_codex_p2/DONE`.

## After you finish (Claude runs this)

```bash
python scripts/data_pipeline/prepare_p2_deepseek.py codex     # validates your files, merges them into the Phase 2 draft
```

Invalid entries (unknown class, bad box) are listed in `data/engine_bay_codex_p2/REJECTED.json` and skipped.
