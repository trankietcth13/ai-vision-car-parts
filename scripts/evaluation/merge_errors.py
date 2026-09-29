"""
Merge per-model error reports (mine_errors.py) into ONE review queue per split with a single error-id space.

Why: the same test image can appear in several model reports, each numbering its own errors, so a
single verdict file per image would be ambiguous. This builds, for every image, a de-duplicated list:
    FN / CLS errors  -> keyed by the ground-truth instance (gt_id): one entry even if several models missed it
    FP errors        -> merged when same predicted class and box IoU >= --fp-iou
Every merged error keeps `sources` (which models produced it) and gets a stable `error_id` (0..n-1 per image).
Overlays are redrawn with the merged ids: green G<gt_id> = label, red E<error_id> = model-only detection (FP),
orange E<error_id> on the label box = FN / CLS.

Output: qa_results/review_queue/<split>/ERRORS.json + overlays/   (this is what Codex reviews)

Usage:
    python scripts/evaluation/merge_errors.py --report teacher_v4=qa_results/errors_teacher_v4 --report kd_n=qa_results/errors_kd_n
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def local_path(remote: str, dataset: Path, split: str, sub: str) -> Path:
    return dataset / sub / split / Path(remote).name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="append", required=True, help="name=folder containing <split>/ERRORS.json")
    ap.add_argument("--dataset", type=Path, default=ROOT / "data" / "engine_bay_train_v4",
                    help="Local copy of the dataset the reports were mined on")
    ap.add_argument("--out", type=Path, default=ROOT / "qa_results" / "review_queue")
    ap.add_argument("--splits", nargs="+", default=["test", "train"])
    ap.add_argument("--fp-iou", type=float, default=0.5)
    a = ap.parse_args()

    import yaml

    names = {int(k): v for k, v in yaml.safe_load((a.dataset / "data_engine_bay_train.yaml").read_text(encoding="utf-8"))["names"].items()}
    for split in a.splits:
        per_image: dict[str, dict] = {}
        used_reports = []
        for spec in a.report:
            src, folder = spec.split("=", 1)
            f = Path(folder) / split / "ERRORS.json"
            if not f.exists():
                continue
            used_reports.append(src)
            for rec in json.loads(f.read_text(encoding="utf-8"))["images"]:
                item = per_image.setdefault(rec["image"], {"image": rec["image"], "split": split, "gt": {}, "fp": []})
                for e in rec["errors"]:
                    if e["type"] in ("FN", "CLS"):
                        key = e["gt_id"]
                        g = item["gt"].setdefault(key, {"gt_id": key, "gt_class": e["gt_class"], "box_norm": e["box_norm"],
                                                        "types": {}, "sources": []})
                        g["types"][src] = e["type"] if e["type"] == "FN" else f"CLS->{e['pred_class']}"
                        g["sources"].append(src)
                    else:
                        for fp in item["fp"]:
                            if fp["pred_class"] == e["pred_class"] and iou(fp["box_norm"], e["box_norm"]) >= a.fp_iou:
                                fp["sources"].append(src)
                                fp["conf"][src] = e["conf"]
                                break
                        else:
                            item["fp"].append({"pred_class": e["pred_class"], "box_norm": e["box_norm"],
                                               "conf": {src: e["conf"]}, "sources": [src]})
        if not per_image:
            print(f"[merge] {split}: no reports")
            continue

        out_dir = a.out / split
        (out_dir / "overlays").mkdir(parents=True, exist_ok=True)
        images = []
        for name, item in sorted(per_image.items()):
            errors = []
            for g in sorted(item["gt"].values(), key=lambda x: x["gt_id"]):
                kinds = set(g["types"].values())
                etype = "FN" if kinds == {"FN"} else "CLS" if all(k.startswith("CLS") for k in kinds) else "FN/CLS"
                errors.append({"type": etype, "gt_id": g["gt_id"], "gt_class": g["gt_class"], "box_norm": g["box_norm"],
                               "model_view": g["types"], "sources": sorted(set(g["sources"]))})
            for fp in item["fp"]:
                errors.append({"type": "FP", "pred_class": fp["pred_class"], "box_norm": fp["box_norm"],
                               "conf": fp["conf"], "sources": sorted(set(fp["sources"]))})
            for i, e in enumerate(errors):
                e["error_id"] = i
            img_path = local_path(name, a.dataset, split, "images")
            lbl_path = local_path(Path(name).with_suffix(".txt").name, a.dataset, split, "labels")

            im = cv2.imread(str(img_path))
            if im is not None:
                h, w = im.shape[:2]
                if lbl_path.exists():
                    for k, line in enumerate(l for l in lbl_path.read_text().splitlines() if l.strip()):
                        p = line.split()
                        pts = (np.array(p[1:], float).reshape(-1, 2) * [w, h]).astype(np.int32)
                        cv2.polylines(im, [pts], True, (0, 200, 0), 2)
                        x, y = pts.min(0)
                        t = f"G{k} {names[int(p[0])]}"
                        cv2.putText(im, t, (int(x) + 2, int(y) + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                        cv2.putText(im, t, (int(x) + 2, int(y) + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 0), 1)
                for e in errors:
                    b = e["box_norm"]
                    x1, y1, x2, y2 = int(b[0] * w), int(b[1] * h), int(b[2] * w), int(b[3] * h)
                    col = (0, 0, 255) if e["type"] == "FP" else (0, 140, 255)
                    cv2.rectangle(im, (x1, y1), (x2, y2), col, 3 if e["type"] == "FP" else 2)
                    t = f"E{e['error_id']} {e['type']} {e.get('pred_class', e.get('gt_class'))}"
                    cv2.putText(im, t, (x1 + 2, max(14, y2 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
                    cv2.putText(im, t, (x1 + 2, max(14, y2 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)
                cv2.imwrite(str(out_dir / "overlays" / name), im, [cv2.IMWRITE_JPEG_QUALITY, 88])
            images.append({"image": name, "split": split, "image_path": str(img_path), "label_path": str(lbl_path),
                           "overlay": str(out_dir / "overlays" / name), "errors": errors})

        summary = {"split": split, "reports": used_reports, "images": len(images),
                   "errors": sum(len(i["errors"]) for i in images),
                   "by_type": {t: sum(1 for i in images for e in i["errors"] if e["type"] == t) for t in ("FP", "FN", "CLS", "FN/CLS")},
                   "class_names": names}
        (out_dir / "ERRORS.json").write_text(json.dumps({"summary": summary, "images": images}, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in summary.items() if k != "class_names"}))


if __name__ == "__main__":
    main()
