"""
Phase 3 of the engine-bay roadmap: cross-validation report + per-class confidence thresholds (run on the DGX).

For every fold k (model trained on fold{k}_train, never saw fold{k}_val):
    1. Ultralytics val on fold{k}_val  -> box / mask mAP50 and mAP50-95, per-class mask AP50
    2. predictions at conf >= --min-conf on every fold{k}_val image, matched to the labels
       (same class, mask IoU >= --iou, greedy by confidence, as in test_cases.py)
The pooled out-of-fold matches give, per class, the confidence threshold that maximises F1
(classes with fewer than --min-gt labels get the global best threshold).

Outputs (--out):
    CV_REPORT.md / cv_report.json     mean ± std over folds, per-class AP50, threshold table
    class_thresholds.yaml             {class_name: threshold}, also copied to configs/class_thresholds.yaml
                                      (read by test_cases.py --class-conf and web_ui/app.py)

Usage:
    .venv/bin/python scripts/evaluation/cv_eval.py --fold runs/train_kd/kd_n_full_f0/weights/best.pt=data/engine_bay_full/data_fold0.yaml \
        ... --fold ...f4 --name kd_n --out qa_results/cv_kd_n
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def poly_mask(poly_norm, h, w):
    m = np.zeros((h, w), np.uint8)
    pts = (np.asarray(poly_norm, float) * [w, h]).astype(np.int32)
    if len(pts) >= 3:
        cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


def val_images(cfg_path: Path):
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    root = Path(cfg["path"])
    entry = root / cfg["val"]
    if entry.suffix == ".txt":
        imgs = [root / l.strip().lstrip("./") if l.strip().startswith("./") else Path(l.strip())
                for l in entry.read_text().splitlines() if l.strip()]
    else:
        imgs = sorted(entry.glob("*.jpg"))
    return cfg, sorted(set(imgs))


def label_of(img: Path) -> Path:
    parts = list(img.parts)
    i = len(parts) - 1 - parts[::-1].index("images")
    parts[i] = "labels"
    return Path(*parts).with_suffix(".txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", action="append", required=True, help="model.pt=data_foldK.yaml (repeatable)")
    ap.add_argument("--name", default="model")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--min-conf", type=float, default=0.05)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--min-gt", type=int, default=15)
    ap.add_argument("--out", default="qa_results/cv")
    ap.add_argument("--device", default="0")
    ap.add_argument("--no-copy-config", action="store_true", help="do not write configs/class_thresholds.yaml")
    a = ap.parse_args()
    from ultralytics import YOLO

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fold_metrics, per_class_ap = [], defaultdict(list)
    dets = defaultdict(list)  # class -> [(conf, is_tp)]
    n_gt = defaultdict(int)
    names = None
    for k, spec in enumerate(a.fold):
        model_path, data_path = spec.split("=", 1)
        model = YOLO(model_path)
        cfg, imgs = val_images(Path(data_path))
        names = {int(i): n for i, n in cfg["names"].items()}
        m = model.val(data=data_path, split="val", imgsz=a.imgsz, batch=8, device=a.device, plots=False,
                      verbose=False, project=str(out.resolve() / "val_runs"), name=f"fold{k}", exist_ok=True)
        fm = {"fold": k, "model": model_path, "images": len(imgs),
              "box_map50": float(m.box.map50), "box_map": float(m.box.map),
              "mask_map50": float(m.seg.map50), "mask_map": float(m.seg.map)}
        fold_metrics.append(fm)
        for ci, c in enumerate(m.seg.ap_class_index):
            per_class_ap[names[int(c)]].append(float(m.seg.ap50[ci]))
        print(json.dumps(fm), flush=True)

        for img_path in imgs:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            gt = []
            lp = label_of(img_path)
            if lp.exists():
                for line in lp.read_text().splitlines():
                    s = line.split()
                    if len(s) >= 7:
                        gt.append((int(s[0]), poly_mask(np.array(s[1:], float).reshape(-1, 2), h, w)))
            for c, _ in gt:
                n_gt[names[c]] += 1
            r = model.predict(img, imgsz=a.imgsz, conf=a.min_conf, device=a.device, verbose=False, retina_masks=True)[0]
            if r.masks is None or not len(r.masks):
                continue
            preds = []
            for mk, c, cf in zip(r.masks.data.cpu().numpy(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                mk = mk.astype(bool) if mk.shape == (h, w) else \
                    cv2.resize(mk.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                preds.append((int(c), float(cf), mk))
            used = set()
            for c, cf, mk in sorted(preds, key=lambda x: -x[1]):
                best, bj = 0.0, None
                for j, (gc, gm) in enumerate(gt):
                    if j in used or gc != c:
                        continue
                    union = np.logical_or(mk, gm).sum()
                    iou = np.logical_and(mk, gm).sum() / union if union else 0.0
                    if iou > best:
                        best, bj = iou, j
                tp = bj is not None and best >= a.iou
                if tp:
                    used.add(bj)
                dets[names[c]].append((cf, tp))

    grid = [round(x, 2) for x in np.arange(max(0.05, a.min_conf), 0.851, 0.05)]

    def f1_at(pairs, gt_count, t):
        tp = sum(1 for cf, ok in pairs if cf >= t and ok)
        fp = sum(1 for cf, ok in pairs if cf >= t and not ok)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / gt_count if gt_count else 0.0
        return (2 * p * r / (p + r) if p + r else 0.0), p, r

    all_pairs = [x for v in dets.values() for x in v]
    total_gt = sum(n_gt.values())
    global_t = max(grid, key=lambda t: f1_at(all_pairs, total_gt, t)[0])
    thresholds, rows = {}, []
    for c in sorted(set(n_gt) | set(dets)):
        pairs, g = dets.get(c, []), n_gt.get(c, 0)
        if g >= a.min_gt and pairs:
            t = max(grid, key=lambda t: (f1_at(pairs, g, t)[0], -abs(t - global_t)))
            how = "tuned"
        else:
            t, how = global_t, f"global (<{a.min_gt} labels)"
        thresholds[c] = float(t)
        f_def, p_def, r_def = f1_at(pairs, g, 0.25)
        f_t, p_t, r_t = f1_at(pairs, g, t)
        rows.append({"class": c, "labels": g, "threshold": t, "how": how,
                     "f1_at_025": round(f_def, 3), "p_at_025": round(p_def, 3), "r_at_025": round(r_def, 3),
                     "f1_tuned": round(f_t, 3), "p_tuned": round(p_t, 3), "r_tuned": round(r_t, 3),
                     "mask_ap50_mean": round(statistics.mean(per_class_ap[c]), 3) if per_class_ap.get(c) else None})

    def ms(key):
        v = [f[key] for f in fold_metrics]
        return statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0.0)

    summary = {k: {"mean": round(ms(k)[0], 4), "std": round(ms(k)[1], 4)}
               for k in ("mask_map50", "mask_map", "box_map50", "box_map")}
    ov_def = f1_at(all_pairs, total_gt, 0.25)
    tuned_pairs_tp = sum(1 for c, v in dets.items() for cf, ok in v if cf >= thresholds[c] and ok)
    tuned_pairs_fp = sum(1 for c, v in dets.items() for cf, ok in v if cf >= thresholds[c] and not ok)
    tp_p = tuned_pairs_tp / max(1, tuned_pairs_tp + tuned_pairs_fp)
    tp_r = tuned_pairs_tp / max(1, total_gt)
    report = {"name": a.name, "folds": fold_metrics, "summary": summary, "global_threshold": global_t,
              "overall_at_025": {"f1": round(ov_def[0], 3), "precision": round(ov_def[1], 3), "recall": round(ov_def[2], 3)},
              "overall_tuned": {"f1": round(2 * tp_p * tp_r / max(1e-9, tp_p + tp_r), 3),
                                "precision": round(tp_p, 3), "recall": round(tp_r, 3)},
              "classes": rows}
    (out / "cv_report.json").write_text(json.dumps(report, indent=2))
    thr_yaml = yaml.safe_dump({"model": a.name, "source": "cv_eval.py out-of-fold F1", "default": float(global_t),
                               "thresholds": thresholds}, sort_keys=False)
    (out / "class_thresholds.yaml").write_text(thr_yaml)
    if not a.no_copy_config:
        (ROOT / "configs" / "class_thresholds.yaml").write_text(thr_yaml)

    md = [f"# Cross-validation report — {a.name}", "",
          f"{len(fold_metrics)} folds, image-level split of `dataset/` (goal A: the same vehicles, not new ones).", "",
          "| metric | mean | std |", "|---|---|---|"]
    md += [f"| {k} | {v['mean']:.3f} | {v['std']:.3f} |" for k, v in summary.items()]
    md += ["", "| fold | images | mask mAP50 | mask mAP50-95 | box mAP50 |", "|---|---|---|---|---|"]
    md += [f"| {f['fold']} | {f['images']} | {f['mask_map50']:.3f} | {f['mask_map']:.3f} | {f['box_map50']:.3f} |"
           for f in fold_metrics]
    md += ["", f"Pooled out-of-fold detections, mask IoU >= {a.iou}:", "",
           f"- conf 0.25 for every class: F1 {report['overall_at_025']['f1']}, precision "
           f"{report['overall_at_025']['precision']}, recall {report['overall_at_025']['recall']}",
           f"- per-class thresholds: F1 {report['overall_tuned']['f1']}, precision "
           f"{report['overall_tuned']['precision']}, recall {report['overall_tuned']['recall']}", "",
           "| class | labels | mask AP50 | threshold | how | F1 @0.25 | F1 tuned | P tuned | R tuned |",
           "|---|---|---|---|---|---|---|---|---|"]
    md += [f"| {r['class']} | {r['labels']} | {r['mask_ap50_mean']} | {r['threshold']} | {r['how']} | "
           f"{r['f1_at_025']} | {r['f1_tuned']} | {r['p_tuned']} | {r['r_tuned']} |"
           for r in sorted(rows, key=lambda r: -r["labels"])]
    (out / "CV_REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "overall_at_025": report["overall_at_025"],
                      "overall_tuned": report["overall_tuned"]}, indent=2))


if __name__ == "__main__":
    main()
