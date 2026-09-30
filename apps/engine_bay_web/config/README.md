Per-class confidence thresholds, one file per model: `class_thresholds/<model stem>.yaml` (e.g. `kd_n_full.yaml`),
written by `cv_eval.py` in the training project from out-of-fold cross-validation predictions. A shared
`class_thresholds.yaml` here is used only for models without their own file. Format:

```yaml
default: 0.25
thresholds:
  battery: 0.45
  oil_dipstick: 0.15
```

Without any file the app uses one confidence threshold for every class (slider in the UI, `conf` in the API).
