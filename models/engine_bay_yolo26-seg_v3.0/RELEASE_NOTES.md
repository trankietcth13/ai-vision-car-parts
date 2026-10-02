# engine_bay_yolo26-seg v3.0

Released 2026-10-02 from branch `distill_v3`. The same engine-bay segmentation task as POC v1 (21 classes, 640 px), with
both the teacher and the student moved from YOLO11 to **YOLO26**. YOLO26s-seg is the model Nich uses. This release exists
so the three modes can be compared directly: **teacher only**, **student only** (no distillation), and **student + KD**.

## What is in this folder

| File | Mode | What it is |
|---|---|---|
| `teacher_yolo26l-seg_v3.0.pt` / `.onnx` | Teacher only | YOLO26l-seg, 27.9 M parameters |
| `student_base_yolo26s-seg_v3.0.pt` / `.onnx` | Student only | YOLO26s-seg, 10.4 M parameters, trained on the labels alone (seed 0) |
| `student_kd_yolo26s-seg_v3.0.pt` / `.onnx` | Teacher + student (KD) | Same YOLO26s-seg, distilled from the YOLO26l teacher (seed 1) |

"Teacher + student" is not a model that runs both networks: distillation only uses the teacher during training, and the
result is a student. So the KD mode is a student file, and the fair question is whether it beats the student trained without KD.

## Results in short (118 test photos of 3 vehicles never seen in training, mask mAP50-95)

| Model | mask mAP50-95 | Change vs the YOLO11 model it replaces |
|---|---|---|
| Teacher YOLO26l | **0.356** | +0.6 pt vs YOLO11l (0.350) |
| Student YOLO26s, no KD (seeds 0 / 1) | **0.332 / 0.318** | +2.6 pt on average vs the YOLO11n KD student (0.301 / 0.297) |
| Student YOLO26s + KD (seed 1) | 0.259 | about 6.6 pt **below** the same student without KD |

- **YOLO26s without distillation is the best student so far.** It beats the current production student (YOLO11n + KD) by
  about 2.6 points, but it is 3.7 times larger (10.4 M vs 2.8 M parameters) and slower on a CPU.
- **Distillation hurts YOLO26s with the current recipe.** The KD settings were tuned for YOLO11. With the YOLO26 pair the
  feature-distillation loss is 4-5 times larger and dominates training: the KD student learns more slowly (val mask 0.215
  vs 0.344 at epoch 25) and ends lower. A re-tuned KD (smaller feature weight) is the next experiment; until then the
  "student only" file is the one to use.
- **The teacher gain is within noise.** With 3 test vehicles the 95% interval is about +/-0.05 mAP, so +0.6 point does
  not show that YOLO26l is a better teacher. Its validation score was 3 points higher (0.392 vs 0.360).
- Weak classes are unchanged: coolant reservoir, radiator hose and ECU module stay near zero to 0.2 AP for every model.

## How these models were trained

- Data: `engine_bay_full_v10` (the corrected v10 labels, no relabelling), split by vehicle: test = vehicles 13 / 23 / 29,
  val = 08 / 36 / 49 / 50 / 52 / 58. Unlike POC v1 (`kd_n_full`, all 28 vehicles), these models never saw the test vehicles,
  so they are for comparison and are not yet a production replacement.
- Recipe: the POC p5 recipe unchanged (100 epochs, 640 px, cosine LR, copy-paste 0.3, mixup 0.1, top-5 checkpoint averaging);
  KD = FGD feature + binary-KL logit + IoU box distillation (`configs/kd_hyperparams_p5.yaml`).
- The first KD run (seed 0) diverged: NaN distillation terms under mixed precision from epoch 5, collapse at epochs 20-25
  (test mask 0.171). The KD code now computes the distillation terms in fp32 and drops a non-finite term (commit `e8d639b`).
  Seed 1 trained with that fix shows no NaN; seed 0 is being retrained and its result will be added here.

## Using them

- Web app: `ENGINE_BAY_MODELS_DIR=models/engine_bay_yolo26-seg_v3.0 python apps/engine_bay_web/app.py`, then
  **Benchmark -> So sánh tất cả model** compares all three on the same photos (the app reads `.pt` and `.onnx`).
- The YOLO26 heads are end-to-end: the ONNX output is `[1, 300, 38]` (box, score, class, 32 mask coefficients) and needs
  no NMS. The Android app and the Vercel demo still decode YOLO11 output only.

## Accuracy on the held-out test split (118 photos, vehicles never seen in training)

| Model | Params (M) | mask mAP50-95 | mask mAP50 | box mAP50-95 | Precision | Recall | GPU latency (ms, fp16, network only) |
|---|---|---|---|---|---|---|---|
| teacher (yolo26l-seg) | 27.9 | **0.356** | 0.570 | 0.360 | 0.676 | 0.527 | 9.3 |
| student_base (yolo26s-seg) | 10.4 | **0.332** | 0.533 | 0.338 | 0.712 | 0.488 | 4.6 |
| student_kd (yolo26s-seg) | 10.4 | **0.259** | 0.426 | 0.262 | 0.461 | 0.460 | 4.7 |
| student_base yolo26s seed 1 (reference) | 10.4 | **0.318** | 0.518 | 0.331 | 0.704 | 0.464 | 4.5 |
| teacher yolo11l v10 (reference) | 27.6 | **0.350** | 0.564 | 0.365 | 0.641 | 0.570 | 9.1 |
| student_kd yolo11n v10 seed 0 (reference) | 2.8 | **0.301** | 0.527 | 0.304 | 0.621 | 0.476 | 3.1 |
| student_kd yolo11n v10 seed 1 (reference) | 2.8 | **0.297** | 0.512 | 0.301 | 0.646 | 0.443 | 3.1 |

### Per-class mask AP50-95

| Class | teacher (yolo26l-seg) | student_base (yolo26s-seg) | student_kd (yolo26s-seg) | student_base yolo26s seed 1 (reference) | teacher yolo11l v10 (reference) | student_kd yolo11n v10 seed 0 (reference) | student_kd yolo11n v10 seed 1 (reference) |
|---|---|---|---|---|---|---|---|
| battery | 0.311 | 0.286 | 0.245 | 0.318 | 0.275 | 0.221 | 0.233 |
| battery_terminal | 0.444 | 0.435 | 0.365 | 0.462 | 0.459 | 0.430 | 0.496 |
| fuse_relay_box | 0.388 | 0.313 | 0.134 | 0.273 | 0.230 | 0.162 | 0.218 |
| coolant_reservoir | 0.012 | 0.014 | 0.002 | 0.009 | 0.035 | 0.003 | 0.021 |
| radiator_cap | 0.413 | 0.461 | 0.450 | 0.561 | 0.349 | 0.469 | 0.425 |
| brake_fluid_reservoir | 0.426 | 0.449 | 0.427 | 0.433 | 0.406 | 0.475 | 0.393 |
| washer_fluid_reservoir | 0.672 | 0.624 | 0.611 | 0.535 | 0.576 | 0.569 | 0.607 |
| engine_cover | 0.326 | 0.312 | 0.290 | 0.264 | 0.361 | 0.341 | 0.360 |
| oil_filler_cap | 0.366 | 0.378 | 0.202 | 0.266 | 0.460 | 0.247 | 0.281 |
| oil_dipstick | 0.452 | 0.412 | 0.402 | 0.398 | 0.484 | 0.460 | 0.420 |
| air_filter_box | 0.475 | 0.434 | 0.348 | 0.426 | 0.460 | 0.305 | 0.356 |
| air_intake_duct | 0.428 | 0.355 | 0.282 | 0.329 | 0.383 | 0.356 | 0.326 |
| maf_sensor | 0.314 | 0.290 | 0.191 | 0.240 | 0.314 | 0.168 | 0.179 |
| throttle_body | 0.281 | 0.294 | 0.155 | 0.227 | 0.240 | 0.210 | 0.156 |
| alternator | 0.354 | 0.264 | 0.220 | 0.215 | 0.337 | 0.281 | 0.284 |
| ignition_coil | 0.382 | 0.337 | 0.196 | 0.341 | 0.380 | 0.338 | 0.334 |
| radiator_hose | 0.081 | 0.042 | 0.074 | 0.189 | 0.237 | 0.028 | 0.051 |
| ecu_module | 0.141 | 0.056 | 0.019 | 0.096 | 0.183 | 0.096 | 0.042 |
| multimeter_diagnostic_tool | 0.593 | 0.626 | 0.489 | 0.535 | 0.624 | 0.616 | 0.539 |
| intake_manifold | 0.262 | 0.263 | 0.077 | 0.238 | 0.200 | 0.239 | 0.214 |

## Files

| File | Role | Architecture | Params (M) | NMS | Size (MB) | md5 |
|---|---|---|---|---|---|---|
| `teacher_yolo26l-seg_v3.0.pt` | teacher | yolo26l-seg | 27.9 | none (end-to-end) | 60.6 | `6d348fbfe100f23d976f7e07c50efb1e` |
| `teacher_yolo26l-seg_v3.0.onnx` | teacher | yolo26l-seg | 27.9 | none (end-to-end) | 106.9 | `c20b9614920b130a4be7d313ad94530b` |
| `student_base_yolo26s-seg_v3.0.pt` | student_base | yolo26s-seg | 10.4 | none (end-to-end) | 22.3 | `4e25ff1b169ae797a8a71b5d5e72a22b` |
| `student_base_yolo26s-seg_v3.0.onnx` | student_base | yolo26s-seg | 10.4 | none (end-to-end) | 39.9 | `7ab212e315d0effcccfab4931c1fa868` |
| `student_kd_yolo26s-seg_v3.0.pt` | student_kd | yolo26s-seg | 10.4 | none (end-to-end) | 22.3 | `08eecd35e4fe00b2f96947c419b74a12` |
| `student_kd_yolo26s-seg_v3.0.onnx` | student_kd | yolo26s-seg | 10.4 | none (end-to-end) | 39.9 | `71c5d6760812944b3ec9a037ebe5da83` |

Sources and full metrics: [manifest.json](manifest.json). Weights are not in git (`*.pt`, `*.onnx` are ignored).
