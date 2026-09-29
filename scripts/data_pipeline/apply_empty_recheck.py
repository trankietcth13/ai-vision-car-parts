"""
Apply the recheck of label-less images (2026-09-29, after the user saw unlabelled images in the label-QA page).

Input : data/engine_bay_empty_check/verdicts/<stem>.json  ({scene, exclude, labels:[{class_name, box_norm, confidence}]})
        images from data/engine_bay_full/images/dataset/<stem>.jpg (the Phase 3 pool)
Output: data/engine_bay_empty_fix/images|labels/train/<stem>   training-id YOLO-seg labels, SAM2 masks from the boxes
        data/engine_bay_empty_fix/EXCLUDED.txt                 stems the recheck excluded (underbody etc.)
        data/engine_bay_empty_fix/EMPTY_FIX_SUMMARY.json
Use it as the highest-precedence source of build_full_dataset.py:
    --reviewed-train data/engine_bay_empty_fix:train data/engine_bay_p2_verified:train
Run on the DGX (SAM2 on GPU), when nothing else is training.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdicts", type=Path, default=ROOT / "data" / "engine_bay_empty_check" / "verdicts")
    ap.add_argument("--images", type=Path, default=ROOT / "data" / "engine_bay_full" / "images" / "dataset")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "engine_bay_empty_fix")
    ap.add_argument("--train-classes", type=Path, default=ROOT / "configs" / "engine_bay_train_classes.yaml")
    ap.add_argument("--sam", default="sam2.1_b.pt")
    ap.add_argument("--min-conf", type=float, default=0.8)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()

    names = {int(k): v for k, v in yaml.safe_load(a.train_classes.read_text(encoding="utf-8"))["names"].items()}
    cid = {v: k for k, v in names.items() if v != "oil_filter"}
    alias = {"radiator_hose_upper": "radiator_hose", "radiator_hose_lower": "radiator_hose"}
    for sub in ("images", "labels"):
        (a.out / sub / "train").mkdir(parents=True, exist_ok=True)
    from ultralytics import SAM

    sam = SAM(a.sam)
    stats, cls, excluded, rejected = Counter(), Counter(), [], []
    for vp in sorted(a.verdicts.glob("*.json")):
        v = json.loads(vp.read_text(encoding="utf-8"))
        stem = vp.stem
        stats["verdicts"] += 1
        if v.get("exclude") or v.get("scene") == "not_engine_bay":
            excluded.append(stem)
            continue
        labels = []
        for d in v.get("labels", []):
            c = alias.get(d.get("class_name"), d.get("class_name"))
            b = d.get("box_norm")
            if c not in cid or not (isinstance(b, list) and len(b) == 4 and all(0 <= x <= 1 for x in b)
                                    and b[0] < b[2] and b[1] < b[3]) or float(d.get("confidence", 1)) < a.min_conf:
                rejected.append({"image": stem, "label": d})
                continue
            labels.append((cid[c], b))
        if not labels:
            stats["kept_empty"] += 1
            continue  # stays an empty negative in the pool, nothing to override
        src = a.images / f"{stem}.jpg"
        img = cv2.imread(str(src))
        h, w = img.shape[:2]
        boxes = [[b[0] * w, b[1] * h, b[2] * w, b[3] * h] for _, b in labels]
        res = sam.predict(img, bboxes=boxes, device=a.device, verbose=False)[0]
        masks = (res.masks.data.cpu().numpy() > 0.5) if res.masks is not None else []
        lines = []
        for k, (c, b) in enumerate(labels):
            poly = None
            if k < len(masks):
                m = masks[k].astype(np.uint8)
                x1, y1, x2, y2 = [int(round(v)) for v in boxes[k]]
                clip = np.zeros_like(m)
                clip[max(0, y1):y2, max(0, x1):x2] = m[max(0, y1):y2, max(0, x1):x2]
                cs, _ = cv2.findContours(clip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cs:
                    cnt = cv2.approxPolyDP(max(cs, key=cv2.contourArea), 0.002 * np.hypot(w, h), True).reshape(-1, 2)
                    if len(cnt) >= 3 and cv2.contourArea(cnt) >= 0.15 * (x2 - x1) * (y2 - y1):
                        poly = (cnt / [w, h]).tolist()
            if poly is None:  # SAM failed or mask too small: box polygon
                poly = [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]
                stats["box_fallback"] += 1
            lines.append(f"{c} " + " ".join(f"{min(max(x, 0), 1):.6f}" for pt in poly for x in pt))
            cls[names[c]] += 1
        shutil.copy2(src, a.out / "images" / "train" / f"{stem}.jpg")
        (a.out / "labels" / "train" / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        stats["images_labelled"] += 1
        stats["instances"] += len(lines)
    (a.out / "EXCLUDED.txt").write_text("".join(f"train/{s}\n" for s in excluded))
    summary = {**stats, "excluded": len(excluded), "rejected_labels": len(rejected),
               "instances_per_class": dict(cls.most_common()), "rejected": rejected[:20]}
    (a.out / "EMPTY_FIX_SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "rejected"}, indent=2))


if __name__ == "__main__":
    main()
