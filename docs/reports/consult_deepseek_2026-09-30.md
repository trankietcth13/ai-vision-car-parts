**1) Ranking for unseen-vehicle accuracy**

1. **#7 Collect 60–100 vehicles with active learning** – only 28 vehicles is the bottleneck; more vehicle diversity is the most direct fix. Run in parallel with GPU work.
2. **#8 Foundation-backbone teacher + distillation** – pretrained features should transfer better to unseen vehicles and improve pseudo-labels; gate before full commitment.
3. **#5 Semi-supervised/noisy-student on unlabeled engine-bay pool** – cheap diversity, but only after pseudo-label quality is proven.
4. **#9 Higher resolution / 800-px tiles** – directly targets the student small-object gap (37% vs 69% teacher) and localization.
5. **#2 Teacher TTA/ensemble** – improves teacher and pseudo-labels, not the student directly.
6. **#3 Label-geometry audit for ducts/terminals/battery** – worthwhile but expert-time-intensive; moderate accuracy impact.
7. **#6 Copy-paste + stronger augmentation** – useful regularizer, but synthetic; cannot replace vehicle diversity.
8. **#1 Per-class thresholds/calibration** – free but no new generalization; thresholds can overfit the 3-vehicle holdout.
9. **#4 Fuse-relay-box vs air-filter hard mining** – too few errors (7/11 fuse boxes) to prioritize now.

**2) Missing / wrong in reasoning**

- The 3-vehicle holdout is too small; mAP differences of ±0.02–0.03 may be noise. Need more held-out vehicles or nested CV with per-vehicle variance.
- Teacher mAP 0.354 is too weak to trust pseudo-labels; label-confidence AUROC 0.84 does not guarantee precision and can amplify systematic errors (reservoir fluids, fuse/air-filter).
- “Low confidence” is not necessarily calibration; on unseen vehicles it may be true epistemic uncertainty, so thresholds are a weak fix.
- Label errors/ambiguous extent still pollute the ground truth; fixing them is prerequisite for credible metrics.

**3) First 3 GPU experiments**

**Exp1: Teacher TTA/ensemble ceiling.**  
Hypothesis: TTA reduces not-found/localization and yields better pseudo-labels.  
Change: TTA inference (flip + 2 scales + 3 seeds) with WBF vs current teacher.  
Gate: held-out mask mAP50-95 ≥ **0.38** (from 0.354) and not-found ≤ **12%**. If pass, use for pseudo-labeling.

**Exp2: Pseudo-label self-training.**  
Hypothesis: diverse unlabeled images help more than synthetic augmentation.  
Change: add ~2,000 TTA-teacher pseudo-labels filtered by a high-confidence threshold to student KD training.  
Gate: student held-out mAP50-95 ≥ **0.33** (from 0.302/0.270) and not-found ≤ **15%**.

**Exp3: Higher-resolution student training.**  
Hypothesis: the small-object gap is caused by 640-px training resolution.  
Change: train student with KD at `imgsz=960`, keeping data/aug/loss identical.  
Gate: student held-out mAP50-95 ≥ **0.33** and small-object (≤32 px) recall **+10 points**, with precision drop ≤ **2 points**.