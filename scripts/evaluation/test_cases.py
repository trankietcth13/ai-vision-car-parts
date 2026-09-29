"""
Per-image test cases for an engine-bay segmentation model (run on the DGX).

Picks N images from a labelled split (default: the held-out, expert-reviewed test split), runs the model
and scores every image against its ground truth:
    match       prediction <-> label, same class, mask IoU >= --iou (greedy, highest confidence first)
    TP / FP / FN per image, precision, recall, mean mask IoU of matches
    PASS        precision >= --min-precision and recall >= --min-recall (images with no labels and no
                predictions pass; no labels but predictions -> FAIL on precision)
Outputs (in --out):
    test_cases.csv      one row per image
    TEST_CASES.md       summary + per-class hit rate + worst images
    gallery.html        side-by-side label vs prediction for every image (open locally)
    images/<stem>.jpg   the side-by-side panels

Usage:
    .venv/bin/python scripts/evaluation/test_cases.py --model runs/train_kd/kd_n_v6_s0/weights/best.pt --imgsz 640 \
        --data data/engine_bay_train_v6/data_engine_bay_train.yaml --split test --n 100 --out test_cases_kd_v6
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=100, help="images to sample; 0 = every image of the split")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--class-conf", default=None,
                    help="YAML with per-class thresholds (cv_eval.py output, e.g. configs/class_thresholds.yaml); "
                         "overrides --conf per class")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--min-precision", type=float, default=0.5)
    ap.add_argument("--min-recall", type=float, default=0.5)
    ap.add_argument("--out", default="test_cases")
    ap.add_argument("--device", default="0")
    return ap.parse_args()


def poly_mask(poly_norm, h, w):
    m = np.zeros((h, w), np.uint8)
    pts = (np.asarray(poly_norm, float) * [w, h]).astype(np.int32)
    if len(pts) >= 3:
        cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


def draw(img, items, color_fn, label_fn):
    over = img.copy()
    for it in items:
        cv2.fillPoly(over, [it["pts"]], color_fn(it))
    out = cv2.addWeighted(over, 0.35, img, 0.65, 0)
    for it in items:
        c = color_fn(it)
        cv2.polylines(out, [it["pts"]], True, c, 2)
        x, y = it["pts"].min(0)
        t = label_fn(it)
        cv2.putText(out, t, (int(x) + 2, int(y) + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(out, t, (int(x) + 2, int(y) + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
    return out


def main():
    a = parse_args()
    import yaml
    from ultralytics import YOLO

    cfg = yaml.safe_load(Path(a.data).read_text(encoding="utf-8"))
    names = {int(k): v for k, v in cfg["names"].items()}
    root = Path(cfg["path"])
    img_dir = root / cfg[a.split]
    lbl_dir = root / cfg[a.split].replace("images", "labels")
    imgs = sorted(img_dir.glob("*.jpg"))
    random.Random(a.seed).shuffle(imgs)
    imgs = sorted(imgs[: a.n] if a.n > 0 else imgs)
    class_conf, fallback_conf = {}, a.conf
    if a.class_conf:
        cc = yaml.safe_load(Path(a.class_conf).read_text(encoding="utf-8"))
        by_name = {v: k for k, v in names.items()}
        class_conf = {by_name[n]: float(t) for n, t in cc.get("thresholds", {}).items() if n in by_name}
        fallback_conf = float(cc.get("default", a.conf))
    pred_conf = min([fallback_conf, *class_conf.values()])
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    model = YOLO(a.model)
    model.predict(np.zeros((a.imgsz, a.imgsz, 3), np.uint8), imgsz=a.imgsz, device=a.device, verbose=False)  # warm-up

    rows, per_class = [], defaultdict(Counter)
    for p in imgs:
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        gt = []
        lp = lbl_dir / f"{p.stem}.txt"
        if lp.exists():
            for line in lp.read_text().splitlines():
                s = line.split()
                if len(s) >= 7:
                    poly = np.array(s[1:], float).reshape(-1, 2)
                    gt.append({"cls": int(s[0]), "mask": poly_mask(poly, h, w), "pts": (poly * [w, h]).astype(np.int32)})
        t0 = time.perf_counter()
        r = model.predict(str(p), imgsz=a.imgsz, conf=pred_conf, device=a.device, verbose=False, retina_masks=True)[0]
        ms = (time.perf_counter() - t0) * 1000
        preds = []
        if r.masks is not None and len(r.masks):
            for m, poly, c, cf in zip(r.masks.data.cpu().numpy(), r.masks.xy, r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                if cf < class_conf.get(int(c), fallback_conf):
                    continue
                mk = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool) if m.shape != (h, w) else m.astype(bool)
                preds.append({"cls": int(c), "conf": float(cf), "mask": mk, "pts": np.asarray(poly, np.int32)})

        used, tp_ious = set(), []
        for pr in sorted(preds, key=lambda x: -x["conf"]):
            best, bj = 0.0, None
            for j, g in enumerate(gt):
                if j in used or g["cls"] != pr["cls"]:
                    continue
                inter = np.logical_and(pr["mask"], g["mask"]).sum()
                union = np.logical_or(pr["mask"], g["mask"]).sum()
                iou = inter / union if union else 0.0
                if iou > best:
                    best, bj = iou, j
            if bj is not None and best >= a.iou:
                used.add(bj)
                pr["tp"] = True
                tp_ious.append(best)
                per_class[names[pr["cls"]]]["tp"] += 1
            else:
                pr["tp"] = False
                per_class[names[pr["cls"]]]["fp"] += 1
        for j, g in enumerate(gt):
            g["hit"] = j in used
            per_class[names[g["cls"]]]["gt"] += 1
            if not g["hit"]:
                per_class[names[g["cls"]]]["fn"] += 1

        tp, fp, fn = len(used), len(preds) - len(used), len(gt) - len(used)
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / (tp + fn) if (tp + fn) else 1.0
        passed = precision >= a.min_precision and recall >= a.min_recall
        rows.append({"image": p.name, "gt": len(gt), "pred": len(preds), "tp": tp, "fp": fp, "fn": fn,
                     "precision": round(precision, 3), "recall": round(recall, 3),
                     "mean_mask_iou": round(float(np.mean(tp_ious)), 3) if tp_ious else "",
                     "latency_ms": round(ms, 1), "result": "PASS" if passed else "FAIL",
                     "missed": "; ".join(names[g["cls"]] for g in gt if not g["hit"]),
                     "false_detections": "; ".join(f"{names[x['cls']]} {x['conf']:.2f}" for x in preds if not x["tp"])})

        left = draw(img, gt, lambda g: (0, 200, 0) if g["hit"] else (0, 140, 255),
                    lambda g: f"{names[g['cls']]}{'' if g['hit'] else ' MISSED'}")
        right = draw(img, preds, lambda x: (0, 200, 0) if x["tp"] else (0, 0, 255),
                     lambda x: f"{names[x['cls']]} {x['conf']:.2f}{'' if x['tp'] else ' FP'}")
        for im, t in ((left, "LABEL (green=found, orange=missed)"), (right, f"PREDICTION (green=correct, red=false)  {rows[-1]['result']}")):
            cv2.rectangle(im, (0, 0), (w, 34), (0, 0, 0), -1)
            cv2.putText(im, t, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        panel = np.hstack([left, right])
        s = min(1.0, 1800 / panel.shape[1])
        panel = cv2.resize(panel, (int(panel.shape[1] * s), int(panel.shape[0] * s)))
        cv2.imwrite(str(out / "images" / f"{p.stem}.jpg"), panel, [cv2.IMWRITE_JPEG_QUALITY, 85])

    with open(out / "test_cases.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    n_pass = sum(r["result"] == "PASS" for r in rows)
    T = Counter()
    for r in rows:
        T.update({"gt": r["gt"], "tp": r["tp"], "fp": r["fp"], "fn": r["fn"]})
    lat = sorted(r["latency_ms"] for r in rows)
    summary = {"model": a.model, "split": a.split, "images": len(rows), "pass": n_pass, "fail": len(rows) - n_pass,
               "pass_rate": round(n_pass / len(rows), 3), "instances": T["gt"], "tp": T["tp"], "fp": T["fp"], "fn": T["fn"],
               "precision": round(T["tp"] / max(1, T["tp"] + T["fp"]), 3), "recall": round(T["tp"] / max(1, T["gt"]), 3),
               "latency_ms_median": lat[len(lat) // 2], "latency_ms_p95": lat[int(len(lat) * 0.95) - 1],
               "criteria": f"mask IoU>={a.iou}, conf>={'per-class (' + a.class_conf + ')' if class_conf else a.conf}, PASS if precision>={a.min_precision} and recall>={a.min_recall}"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    md = [f"# Test cases - {len(rows)} images ({a.split} split)", "", f"Model: `{a.model}` @ {a.imgsz}", "",
          f"Criteria: {summary['criteria']}", "",
          f"- **PASS {n_pass} / {len(rows)}** ({summary['pass_rate']:.0%})",
          f"- Instances: {T['gt']} labelled, {T['tp']} found, {T['fn']} missed, {T['fp']} false detections",
          f"- Instance precision {summary['precision']}, recall {summary['recall']}",
          f"- Latency (end-to-end predict incl. pre/post-processing): median {summary['latency_ms_median']} ms, p95 {summary['latency_ms_p95']} ms",
          "", "## Per class", "", "| class | labelled | found | missed | false det. | hit rate |", "|---|---|---|---|---|---|"]
    for c, v in sorted(per_class.items(), key=lambda kv: -(kv[1]["gt"])):
        hr = f"{v['tp'] / v['gt']:.0%}" if v["gt"] else "-"
        md.append(f"| {c} | {v['gt']} | {v['tp']} | {v['fn']} | {v['fp']} | {hr} |")
    worst = sorted(rows, key=lambda r: (r["result"] == "PASS", r["recall"], r["precision"]))[:15]
    md += ["", "## 15 weakest images", "", "| image | labelled | found | missed | false det. | result | missed classes |",
           "|---|---|---|---|---|---|---|"]
    md += [f"| {r['image']} | {r['gt']} | {r['tp']} | {r['fn']} | {r['fp']} | {r['result']} | {r['missed']} |" for r in worst]
    (out / "TEST_CASES.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    cards = []
    for r in sorted(rows, key=lambda r: (r["result"] == "PASS", r["recall"])):
        stem = Path(r["image"]).stem
        cards.append(f'<div class="card {r["result"].lower()}"><div class="hd"><b>{html.escape(r["image"])}</b> '
                     f'<span class="tag">{r["result"]}</span> P {r["precision"]} · R {r["recall"]} · '
                     f'{r["tp"]}/{r["gt"]} found · {r["fp"]} false</div>'
                     f'<img loading="lazy" src="images/{stem}.jpg" alt="{html.escape(r["image"])}">'
                     + (f'<div class="miss">Missed: {html.escape(r["missed"])}</div>' if r["missed"] else "")
                     + (f'<div class="miss">False: {html.escape(r["false_detections"])}</div>' if r["false_detections"] else "")
                     + "</div>")
    page = f"""<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Engine Bay Test Cases</title><style>
:root{{--bg:#f6f7f9;--fg:#1d2330;--card:#fff;--pass:#1f8a4c;--fail:#c0392b;--muted:#5b6474}}
@media (prefers-color-scheme:dark){{:root{{--bg:#12151b;--fg:#e6e9ef;--card:#1b2029;--muted:#9aa3b2}}}}
body{{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif}}
h1{{font-size:20px;margin:0 0 4px}} .sum{{color:var(--muted);margin-bottom:16px}}
.card{{background:var(--card);border-radius:10px;padding:10px;margin:0 0 14px;border-left:5px solid var(--pass)}}
.card.fail{{border-left-color:var(--fail)}} .card img{{width:100%;height:auto;border-radius:6px;margin-top:6px}}
.tag{{font-weight:700;color:var(--pass)}} .fail .tag{{color:var(--fail)}} .hd{{word-break:break-all}}
.miss{{color:var(--muted);font-size:13px;margin-top:4px}}</style></head><body>
<h1>Engine bay test cases</h1><div class="sum">PASS {n_pass}/{len(rows)} · precision {summary['precision']} · recall {summary['recall']} ·
model {html.escape(Path(a.model).parent.parent.name)} @ {a.imgsz} · {html.escape(summary['criteria'])}. Failed images first.</div>
{''.join(cards)}</body></html>"""
    (out / "gallery.html").write_text(page, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
