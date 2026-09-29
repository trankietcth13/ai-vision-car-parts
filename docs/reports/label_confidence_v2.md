# Label-confidence model from review verdicts (W1e)

4627 reviewed machine boxes from 177 vehicles (2688 class-right, 1939 class-wrong). 5-fold GroupKFold by vehicle.

| model | AUROC | auto-accept coverage @ precision ≥ 0.95 | @ ≥ 0.90 |
|---|---|---|---|
| vlm_conf only | 0.617 | 0.0% | 0.0% |
| logistic (all features) | 0.823 | 12.1% | 29.7% |
| gradient boosting (all features) | 0.844 | 14.4% | 31.6% |
| gradient boosting (no crop evidence) | 0.797 | 8.7% | 19.1% |

## Best model (gradient boosting, all features): out-of-fold detail

- source deepseek: n=1483, base precision 0.610, AUROC 0.880
- source qwen: n=3144, base precision 0.567, AUROC 0.824
- mean score by verdict: bad_geometry 0.67, correct 0.77, duplicate 0.34, not_a_component 0.35, wrong_class 0.40
- score ≥ 0.9: 21.8% of boxes, precision 0.941
- score ≥ 0.8: 36.3% of boxes, precision 0.877
- score ≥ 0.6: 52.6% of boxes, precision 0.826
- score ≥ 0.4: 66.5% of boxes, precision 0.763

Top features (permutation importance, AUROC drop, in-sample): crop_margin 0.193, vlm_conf 0.043, log_size 0.023, prior_density 0.022, cls=engine_cover 0.020, aspect 0.018, max_iou_other 0.016, crop_sim_claimed 0.015, crop_sim_rejected 0.015, n_boxes 0.013
