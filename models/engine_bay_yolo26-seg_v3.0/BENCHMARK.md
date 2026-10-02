# Benchmark · engine_bay_yolo26-seg v3.0

## GPU, network only (from the test QA run)

DGX GB10 GPU, batch 1, fp16, 640 px, warm; network forward only, measured by `qa_test.py` right after the test scoring
(GPU idle). Pre- and post-processing are not included.

| Model | Params (M) | Latency (ms) | Test mask mAP50-95 |
|---|---|---|---|
| teacher YOLO26l-seg | 27.9 | 9.3 | 0.356 |
| student YOLO26s-seg, no KD | 10.4 | 4.6 | 0.332 |
| student YOLO26s-seg + KD | 10.4 | 4.7 | 0.259 |
| teacher YOLO11l-seg v10 (reference) | 27.6 | 9.1 | 0.350 |
| student YOLO11n-seg + KD v10 (reference) | 2.8 | 3.1 | 0.301 |

## End-to-end (pending)

End-to-end `model.predict` timings (preprocess / inference / NMS + masks, GPU and CPU, `.pt` and `.onnx`) will be measured
on the DGX with `scripts/deployment/benchmark_models.py` once the GPU is free (seed 0 KD retraining is running). A laptop
CPU run on 2026-10-02 evening was discarded: the same YOLO11n model measured 50 ms and 308 ms minutes apart.
