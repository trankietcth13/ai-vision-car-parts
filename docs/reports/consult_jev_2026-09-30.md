| # | expected gain (0-3) | cost/risk (0-2) | targets top failure | candidate |
|---|---|---|---|---|
| 7 | 2.78 | 1.9 | 0.64 | Collect new vehicles (goal: 60-100 vehicles), prioritised by active learning (teacher/VLM disag |
| 5 | 2.62 | 1.95 | 0.68 | Semi-supervised / noisy-student self-training: teacher system (+ tiles, + SAM2) pseudo-labels a |
| 1 | 1.59 | 0.46 | 0.8 | Per-class thresholds from CV + score calibration (free; targets the 15-18% low-confidence share |
| 2 | 1.96 | 1.39 | 0.5 | Teacher-side TTA/ensemble (flip, 2 scales, 2-3 seeds) + weighted box fusion, for server use and |
| 8 | 2.04 | 1.97 | 0.39 | Foundation-backbone teacher (DINOv2/v3-based segmentation, e.g. RF-DETR-Seg or Mask2Former) dis |
| 4 | 1.52 | 0.95 | 0.32 | Targeted hard-example work on fuse_relay_box vs air_filter_box (more instances, class-balanced  |
| 3 | 1.53 | 1.09 | 0.13 | Label-geometry audit + extent policy for ducts, terminals, battery (targets localisation 9-12%) |
| 9 | 1.55 | 1.42 | 0.33 | Training at higher resolution (960) or on 800-px tiles for the small-object share. |
| 6 | 1.4 | 1.62 | 0.12 | Cross-image copy-paste of components (paste parts from other vehicles) and stronger photometric |
