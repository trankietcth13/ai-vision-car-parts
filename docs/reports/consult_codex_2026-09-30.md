## Ranking for unseen-vehicle accuracy

1. **#7 Collect more vehicles.** The dominant issue is diversity: only 28 vehicles, with training mAP ≈0.72 versus teacher test mAP **0.354**. Images from one vehicle are highly correlated; another view of the same engine is not equivalent to another vehicle.
2. **#5 Semi-supervised/self-training on many new vehicles.** Potentially the fastest way to increase vehicle diversity, but only with conservative pseudo-labels and human auditing. The confidence model’s **AUROC 0.84** is insufficient assurance: ≥0.95 precision covers only **14%** of boxes.
3. **#3 Geometry/extent label audit.** Directly addresses **9% teacher / 12% student** localization failures. Ducts, terminals and batteries account for 25/40 teacher localization errors; reservoir boxes are misplaced/cut in **25%** of audited cases.
4. **#8 Foundation-backbone teacher.** Plausibly improves cross-vehicle representation, but speculative until compared under identical data. Distillation may discard its advantage, and DINO retrieval already failed to improve the teacher’s **88.8%** class correctness after localization.
5. **#9 Train at 960 or on tiles.** Strong student-small-object rationale: <32 px detection is **37% student versus 69% teacher**; tiled inference raised student recall **0.475→0.543**. But only 35/443 objects are <32 px, and prior 1024 training lost to 640.
6. **#2 Teacher TTA/ensemble.** Likely useful for server inference and better pseudo-label proposals, but it treats symptoms and costs several passes. Tiling improved teacher recall only about **0.582→0.621**, with precision falling **0.632→0.552**.
7. **#6 Copy-paste/photometric augmentation.** Cheap, possibly useful for rare/small objects, but stronger augmentation and 175 external images previously delivered little or no unseen-vehicle gain. Unrealistic placement is a real risk.
8. **#4 Fuse-box hard examples.** Fixes a genuine error—7/11 teacher fuse boxes became air-filter boxes—but its maximum aggregate effect is narrow unless this class is operationally critical.
9. **#1 Thresholds/calibration.** Useful deployment tuning, not a substantive AP/generalization improvement. The 15–18% “low-confidence” bucket is not automatically recoverable without false positives; thresholds change the operating point, not ranking quality.

## Missing or incorrect reasoning

The evaluation is too small and clustered: **three vehicles**, 443 objects, with 192 from one vehicle. Report vehicle-macro metrics and bootstrap confidence intervals; ±1–2 mAP points are not persuasive. Several classes have only 3–13 instances.

The miss taxonomy uses best boxes and a fixed 0.25 threshold, so it conflates calibration with detection and says little about mask-boundary quality. The claimed SAM2 improvement (**IoU 0.78→0.83**) cannot repair missing or wrongly classified objects.

Per-class thresholds derived by the current **image-level CV** target known vehicles, not unseen vehicles; tuning should use grouped vehicle folds. Taxonomy v2 also changes the task and may dilute supervision, so it needs old-class and generic-class reporting separately. Finally, compare every intervention with fixed data, initialization, augmentation, checkpoint averaging, and at least two seeds.

## First three GPU experiments

1. **Scale-aware FGD (E1).** Hypothesis: equal-instance weighting closes the student’s small-object KD gap. Single change: enable scale-aware foreground masks. Gate: two-seed mean mask mAP50–95 ≥ **0.296** versus current mean **0.286**, and <32 px recall improves ≥10 points without ≥96 px recall dropping >1 point.

2. **Tile fine-tuning.** Hypothesis: real 800-px crops preserve detail better than inference upscaling. Single change: add 800×800 crops mixed 1:1, evaluating normal full-frame inference. Gate: unseen mask mAP50–95 +≥0.015 and <32 px recall +≥10 points, with precision loss ≤2 points.

3. **New-vehicle pseudo-label pilot.** Hypothesis: breadth beats additional same-vehicle images. Single change: add conservatively filtered pseudo-labels from ≥10 wholly new vehicles; no feature-only KD or other changes. Gate: two-seed unseen-vehicle macro mAP +≥0.02, improvement on at least 2/3 locked test vehicles, and a 200-box audit ≥95% precision.

Sources: [bottleneck brief](</E:/Research/GEN AI/AI Vision/Distillation/docs/reports/bottleneck_brief_2026-09-30.md>), [miss analysis](</E:/Research/GEN AI/AI Vision/Distillation/docs/reports/miss_analysis_2026-09-30.md>), [small-component research](</E:/Research/GEN AI/AI Vision/Distillation/docs/plans/small_component_detection_research_2026-09-29.md>).