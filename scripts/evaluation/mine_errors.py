"""
Mine disagreements between a model and the dataset labels, for external verification (Codex).

For every image of a split it matches predictions (conf >= --conf) to ground-truth instances by
box IoU (>= --iou) and records:
    FP   prediction with no matching GT            -> model hallucination OR missing label
    FN   GT with no matching prediction            -> model miss OR wrong/extra label
    CLS  matched box, different class              -> model confusion OR wrong label class
Images are ranked by error count; the top --max-images get an overlay
(GT = green "G<id> class", prediction = red "P<id> class conf") and an entry in
<out>/<split>/ERRORS.json with numbered errors that a reviewer can accept or reject.

Usage (on the DGX):
    .venv/bin/python scripts/evaluation/mine_errors.py --model runs/segment/engine_teacher_v4_dgx/weights/best.pt --imgsz 1024 \
        --data data/engine_bay_train_v4/data_engine_bay_train.yaml --split train --out qa_results/errors
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="qa_results/errors")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--max-images", type=int, default=200)
    ap.add_argument("--device", default="0")
    return ap.parse_args()


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def read_gt(label_path: Path):
    out = []
    if label_path.exists():
        for k, line in enumerate(l for l in label_path.read_text().splitlines() if l.strip()):
            p = line.split()
            pts = np.array(p[1:], dtype=float).reshape(-1, 2)
            out.append({"id": k, "cls": int(p[0]), "box": [pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()],
                        "poly": pts})
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
    out = Path(a.out) / a.split
    (out / "overlays").mkdir(parents=True, exist_ok=True)

    model = YOLO(a.model)
    records = []
    imgs = sorted(img_dir.glob("*.jpg"))
    for r in model.predict([str(p) for p in imgs], imgsz=a.imgsz, conf=a.conf, device=a.device, stream=True, verbose=False):
        p = Path(r.path)
        gt = read_gt(lbl_dir / f"{p.stem}.txt")
        preds = []
        if r.boxes is not None and len(r.boxes):
            for k, (b, c, cf) in enumerate(zip(r.boxes.xyxyn.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())):
                preds.append({"id": k, "cls": int(c), "box": b, "conf": round(float(cf), 3)})
        used_gt, errors = set(), []
        for pr in sorted(preds, key=lambda x: -x["conf"]):
            best, bi = 0.0, None
            for g in gt:
                if g["id"] in used_gt:
                    continue
                v = iou(pr["box"], g["box"])
                if v > best:
                    best, bi = v, g
            if bi is not None and best >= a.iou:
                used_gt.add(bi["id"])
                if bi["cls"] != pr["cls"]:
                    errors.append({"type": "CLS", "gt_id": bi["id"], "gt_class": names[bi["cls"]], "pred_id": pr["id"],
                                   "pred_class": names[pr["cls"]], "conf": pr["conf"], "box_norm": [round(x, 4) for x in pr["box"]]})
            else:
                errors.append({"type": "FP", "pred_id": pr["id"], "pred_class": names[pr["cls"]], "conf": pr["conf"],
                               "box_norm": [round(x, 4) for x in pr["box"]]})
        for g in gt:
            if g["id"] not in used_gt:
                errors.append({"type": "FN", "gt_id": g["id"], "gt_class": names[g["cls"]],
                               "box_norm": [round(float(x), 4) for x in g["box"]]})
        for i, e in enumerate(errors):
            e["error_id"] = i
        if errors:
            records.append({"image": p.name, "image_path": str(p), "label_path": str(lbl_dir / f"{p.stem}.txt"),
                            "n_gt": len(gt), "n_pred": len(preds), "errors": errors, "_r": r, "_gt": gt})

    records.sort(key=lambda x: -len(x["errors"]))
    keep = records[: a.max_images]
    for rec in keep:
        im = cv2.imread(rec["image_path"])
        h, w = im.shape[:2]
        for g in rec["_gt"]:
            cv2.polylines(im, [(g["poly"] * [w, h]).astype(np.int32)], True, (0, 200, 0), 2)
            x1, y1 = int(g["box"][0] * w), int(g["box"][1] * h)
            cv2.putText(im, f"G{g['id']} {names[g['cls']]}", (x1 + 2, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(im, f"G{g['id']} {names[g['cls']]}", (x1 + 2, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 0), 1)
        r = rec["_r"]
        if r.boxes is not None:
            for k, (b, c, cf) in enumerate(zip(r.boxes.xyxyn.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())):
                x1, y1, x2, y2 = int(b[0] * w), int(b[1] * h), int(b[2] * w), int(b[3] * h)
                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
                t = f"P{k} {names[int(c)]} {cf:.2f}"
                cv2.putText(im, t, (x1 + 2, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
                cv2.putText(im, t, (x1 + 2, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 255), 1)
        ov = out / "overlays" / rec["image"]
        cv2.imwrite(str(ov), im, [cv2.IMWRITE_JPEG_QUALITY, 88])
        rec["overlay"] = str(ov)
        rec.pop("_r"); rec.pop("_gt")
    summary = {"model": a.model, "split": a.split, "images_scanned": len(imgs), "images_with_errors": len(records),
               "errors_by_type": {t: sum(1 for r in records for e in r["errors"] if e["type"] == t) for t in ("FP", "FN", "CLS")},
               "exported_images": len(keep), "conf": a.conf, "iou": a.iou, "class_names": names}
    (out / "ERRORS.json").write_text(json.dumps({"summary": summary, "images": keep}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
