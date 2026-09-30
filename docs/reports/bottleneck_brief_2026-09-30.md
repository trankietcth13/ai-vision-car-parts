# Brief: remaining accuracy bottlenecks of the engine-bay segmentation models (for external review)

## Setup
- Task: instance segmentation of engine-bay components in phone photos (6000x4000 originals, stored at 1600, trained at 640).
  21 classes today (battery, battery_terminal, fuse_relay_box, coolant/brake/washer reservoirs, caps, oil dipstick,
  air filter box, air intake duct, MAF, throttle body, alternator, ignition coil, radiator hose, ECU, intake manifold...).
- Data: 1,081 expert-reviewed images from only 28 vehicles (~40 images per vehicle) + 175 external images (train only).
  Labels: VLM boxes -> SAM2 masks -> expert review.
- Models (Ultralytics 8.4): teacher yolo11l-seg (27.6 M), student yolo11n-seg (2.8 M) trained with knowledge distillation
  (FGD feature loss + logit KL + box KD). Deployment: student on edge (3 ms GPU), teacher possible on a server.
- Compute: one NVIDIA GB10 (DGX Spark) shared with a vLLM service; one training job at a time; a teacher run takes ~75 min,
  a student KD run ~50 min.

## Current accuracy on 3 held-out vehicles (125 images, 443 objects)
- Mask mAP50-95: teacher 0.354 (mask mAP50 0.577), student 0.302 / 0.270 (two seeds). Training-set mAP ~0.72 -> large
  generalisation gap to new vehicles.
- What worked so far: averaging the last/top-5 epoch checkpoints (+3..6 pts teacher), fixing labels (+1.5 pts student),
  KD vs plain student (+1..3 pts), 2x2 tiled inference (+3-4 recall teacher, +6.8 student, lower precision),
  SAM2 re-segmentation of predicted boxes (mask IoU 0.78 -> 0.83).
- What did not work: VLM labelling via Set-of-Mark (F1 0.24 vs teacher 0.59), DINOv2 crop retrieval to fix class names
  (teacher is already 89% class-correct when it finds an object), text-only judges on VLM rationales, inference-only upscaling.

## Miss analysis (teacher / student), every GT object categorised by its best prediction (score >= 0.01)
| category | teacher | student |
|---|---|---|
| detected (same class, IoU >= 0.5, score >= 0.25) | 58% | 48% |
| low confidence (right class, IoU >= 0.5, score < 0.25) | 15% | 18% |
| localisation (right class, best IoU 0.1-0.5) | 9% | 12% |
| wrong class (IoU >= 0.5, other class, score >= 0.25) | 4% | 3% |
| not found at all | 14% | 19% |
- Objects < 32 px: teacher detects 69%, student 37% (student: 29% low confidence).
- Worst vehicle (Request_ID_29): teacher 53% detected, 19% not found.
- Not found mostly: ignition coils (13/58), air intake duct, throttle body, ECU, MAF. Low confidence mostly: ECU (6/13), coils, battery.
- Localisation errors mostly: air intake duct (10/47), battery_terminal (9/53), battery (6/37): elongated or ambiguous-extent parts.
- Wrong class dominated by fuse_relay_box -> air_filter_box (7 of 11 fuse boxes).
- Label audit of reservoirs: 16-21% wrong fluid type where reviewers guessed from position, 25% of boxes cut/misplaced.

## Already implemented and queued (not yet measured)
- E1: FGD scale-aware foreground mask (each GT box weighs the same in the feature loss) for the small-object KD gap.
- Taxonomy v2 teacher: 27 classes (+ generic fallback classes) and system grouping.
- Label corrections for 134 reservoir boxes; label policy (>= 2 visual cues, whole visible body).
- Per-class confidence thresholds from 3-fold cross-validation (running now).

## Candidate solutions under consideration
1. Per-class thresholds from CV + score calibration (free; targets the 15-18% low-confidence share).
2. Teacher-side TTA/ensemble (flip, 2 scales, 2-3 seeds) + weighted box fusion, for server use and pseudo-labelling.
3. Label-geometry audit + extent policy for ducts, terminals, battery (targets localisation 9-12%).
4. Targeted hard-example work on fuse_relay_box vs air_filter_box (more instances, class-balanced sampling, cue policy).
5. Semi-supervised / noisy-student self-training: teacher system (+ tiles, + SAM2) pseudo-labels a large unlabeled
   engine-bay pool (crawled + other vehicles), filtered by a learned label-confidence model (AUROC 0.84), student and teacher
   retrained on labeled + pseudo-labeled data; plus feature-only KD on unlabeled images (already implemented, never used).
6. Cross-image copy-paste of components (paste parts from other vehicles) and stronger photometric augmentation.
7. Collect new vehicles (goal: 60-100 vehicles), prioritised by active learning (teacher/VLM disagreement).
8. Foundation-backbone teacher (DINOv2/v3-based segmentation, e.g. RF-DETR-Seg or Mask2Former) distilled into YOLO via
   pseudo-labels, for better generalisation to unseen vehicles.
9. Training at higher resolution (960) or on 800-px tiles for the small-object share.

Questions: Which of these would you prioritise for accuracy on unseen vehicles and why? What is missing or wrong in this
reasoning? What would you measure first, with what gate?
