# Teacher system evaluation (W4)

Weights `runs\segment\p5_reg\weights\avg5.pt` · data\engine_bay_reviewed/test (125 images) · imgsz 640 · conf 0.25 · box IoU ≥ 0.5 with class match · only GT classes the detector knows

| mode | recall | precision | predictions |
|---|---|---|---|
| full image | 0.582 | 0.632 | 408 |
| full + agnostic merge | 0.578 | 0.638 | 401 |
| full + 2x2 tiles | 0.621 | 0.552 | 498 |
| tiles + agnostic merge | 0.616 | 0.575 | 475 |
| tiles + agnostic + count cap | 0.614 | 0.581 | 468 |

Recall by system:

| system | n | full image | full + agnostic merge | full + 2x2 tiles | tiles + agnostic merge | tiles + agnostic + count cap |
|---|---|---|---|---|---|---|
| air_intake | 155 | 0.55 | 0.55 | 0.59 | 0.59 | 0.59 |
| electrical | 126 | 0.55 | 0.54 | 0.60 | 0.59 | 0.58 |
| ignition | 58 | 0.55 | 0.55 | 0.57 | 0.57 | 0.57 |
| lubrication | 42 | 0.76 | 0.76 | 0.83 | 0.83 | 0.83 |
| brakes | 20 | 0.65 | 0.65 | 0.65 | 0.65 | 0.65 |
| cooling | 16 | 0.56 | 0.56 | 0.62 | 0.62 | 0.62 |
| body | 14 | 0.36 | 0.36 | 0.36 | 0.36 | 0.36 |
| tools | 7 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| washer | 5 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
