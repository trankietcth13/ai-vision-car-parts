"""
Pseudo-label round for the engine-bay dataset: teacher predictions + trusted machine labels.

For every train image that has NOT been expert-reviewed:
  1. run the teacher (Ultralytics *-seg) and keep predictions with conf >= --conf
     (per-class override via --class-conf name=value)
  2. keep an existing machine label (Qwen3-VL box + SAM2 mask) only if
       - its class is "trusted" (review precision >= --trust-precision, from REVIEW_REPORT.json), and
       - no teacher prediction of the same class overlaps it (box IoU > --dup-iou)
  3. machine labels of untrusted classes are dropped (the teacher is the only source for them)
Reviewed train images, val and test are copied unchanged (hard links) from the source dataset.

Output: a new dataset folder with the same layout + PSEUDO_SUMMARY.json.

Usage:
    python scripts/data_pipeline/pseudo_label_merge.py --teacher runs/segment/engine_teacher_v1/weights/best.pt \
        --src data/engine_bay_train_v2 --out data/engine_bay_train_v3
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from data_pipeline.refine_masks_sam2 import atomic_write, box_iou  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--src", type=Path, default=PROJECT_ROOT / "data" / "engine_bay_train_v2")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "engine_bay_train_v3")
    ap.add_argument("--reviewed", type=Path, default=PROJECT_ROOT / "data" / "engine_bay_reviewed")
    ap.add_argument("--report", type=Path, default=PROJECT_ROOT / "data" / "engine_bay_review" / "REVIEW_REPORT.json")
    ap.add_argument("--conf", type=float, default=0.40)
    ap.add_argument("--class-conf", nargs="*", default=[], help="Per-class thresholds, e.g. ignition_coil=0.3")
    ap.add_argument("--trust-precision", type=float, default=0.50)
    ap.add_argument("--trust-min-labeled", type=int, default=10)
    ap.add_argument("--dup-iou", type=float, default=0.50)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--device", default="0")
    return ap.parse_args()


def link(src: Path, dst: Path):
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def poly_box(coords: np.ndarray):
    return [float(coords[:, 0].min()), float(coords[:, 1].min()), float(coords[:, 0].max()), float(coords[:, 1].max())]


def read_labels(path: Path):
    out = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) >= 7:
                out.append((int(p[0]), np.array(p[1:], dtype=np.float64).reshape(-1, 2)))
    return out


def main():
    args = parse_args()
    src, out = args.src.resolve(), args.out.resolve()
    cfg = yaml.safe_load((src / "data_engine_bay_train.yaml").read_text(encoding="utf-8"))
    names = {int(k): v for k, v in cfg["names"].items()}

    report = json.loads(args.report.read_text(encoding="utf-8"))["per_class"]
    trusted = {n for n, s in report.items()
               if s["labeled"] >= args.trust_min_labeled and (s["precision"] or 0) >= args.trust_precision}
    trusted_ids = {k for k, v in names.items() if v in trusted}
    class_conf = {k: args.conf for k in names}
    for item in args.class_conf:
        n, v = item.split("=")
        class_conf[{v2: k2 for k2, v2 in names.items()}[n]] = float(v)
    print(f"[pseudo] trusted machine-label classes: {sorted(trusted)}")

    reviewed_train = {p.stem for p in (args.reviewed / "labels" / "train").glob("*.txt")}

    # copy val/test and reviewed train unchanged
    for split in ("val", "test", "train"):
        for lp in (src / "labels" / split).glob("*.txt"):
            if split == "train" and lp.stem not in reviewed_train:
                continue
            link(lp, out / "labels" / split / lp.name)
            for ip in (src / "images" / split).glob(f"{lp.stem}.*"):
                link(ip, out / "images" / split / ip.name)

    todo = sorted(p for p in (src / "images" / "train").iterdir() if p.stem not in reviewed_train)
    print(f"[pseudo] {len(reviewed_train)} reviewed train images kept, {len(todo)} to pseudo-label")

    from ultralytics import YOLO

    model = YOLO(args.teacher)
    stats = Counter()
    per_class = Counter()
    for r in model.predict([str(p) for p in todo], imgsz=args.imgsz, conf=min(class_conf.values()),
                           device=args.device, stream=True, verbose=False, retina_masks=False):
        ip = Path(r.path)
        final = []
        if r.masks is not None and len(r.masks):
            for poly, c, cf in zip(r.masks.xyn, r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                c = int(c)
                if cf < class_conf[c] or poly is None or len(poly) < 3:
                    continue
                final.append((c, np.asarray(poly, dtype=np.float64), "teacher"))
        for c, coords in read_labels(src / "labels" / "train" / f"{ip.stem}.txt"):
            if c not in trusted_ids:
                stats["machine_dropped_untrusted"] += 1
                continue
            b = poly_box(coords)
            if any(fc == c and box_iou(b, poly_box(fp)) > args.dup_iou for fc, fp, _ in final):
                stats["machine_dup_of_teacher"] += 1
                continue
            final.append((c, coords, "machine"))
        for c, _, s in final:
            stats[f"kept_{s}"] += 1
            per_class[names[c]] += 1
        lines = [f"{c} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in np.clip(p, 0, 1)) for c, p, _ in final]
        atomic_write(out / "labels" / "train" / f"{ip.stem}.txt", "\n".join(lines) + ("\n" if lines else ""))
        link(ip, out / "images" / "train" / ip.name)
        stats["images"] += 1
        stats["empty_images"] += not lines

    cfg["path"] = str(out).replace("\\", "/")
    atomic_write(out / "data_engine_bay_train.yaml", yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    summary = {"teacher": args.teacher, "conf": args.conf, "class_conf": {names[k]: v for k, v in class_conf.items()},
               "trusted_classes": sorted(trusted), "stats": dict(stats), "pseudo_instances_per_class": dict(per_class.most_common())}
    atomic_write(out / "PSEUDO_SUMMARY.json", json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary["stats"], indent=2))


if __name__ == "__main__":
    main()
