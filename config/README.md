Put `class_thresholds.yaml` here to enable per-class confidence thresholds (written by `cv_eval.py` in the training
project after cross-validation). Format:

```yaml
default: 0.25
thresholds:
  battery: 0.45
  oil_dipstick: 0.15
```

Without the file the app uses one confidence threshold for every class (slider in the UI, `conf` in the API).
