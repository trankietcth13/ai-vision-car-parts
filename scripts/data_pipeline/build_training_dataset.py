"""
Assemble the engine-bay training dataset from the available label sources.

Priority per image:
    1. data/engine_bay_reviewed/labels/<split>/<stem>.txt  (expert-reviewed)
    2. data/engine_bay_seg/labels/<split>/<stem>.txt       (Qwen3-VL boxes + SAM2 masks)

Class ids are remapped from the 36-class annotation ontology (configs/data_engine_bay.yaml)
to the training ontology (configs/engine_bay_train_classes.yaml); classes mapped to null are dropped.

For val/test only reviewed images are used by default (--allow-unreviewed-eval to override),
so evaluation numbers are never computed against unreviewed machine labels.

Usage:
    python scripts/data_pipeline/build_training_dataset.py --out data/engine_bay_train
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from data_pipeline.refine_masks_sam2 import atomic_write, materialize  # noqa: E402


def write_resized(src: Path, dst: Path, max_side: int):
    import cv2

    img = cv2.imread(str(src))  # applies EXIF orientation, same as annotation/SAM stages
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 92])


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", type=Path, default=PROJECT_ROOT / "data" / "engine_bay_seg")
    ap.add_argument("--reviewed", type=Path, default=PROJECT_ROOT / "data" / "engine_bay_reviewed")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "engine_bay_train")
    ap.add_argument("--annotation-config", type=Path, default=PROJECT_ROOT / "configs" / "data_engine_bay.yaml")
    ap.add_argument("--train-classes", type=Path, default=PROJECT_ROOT / "configs" / "engine_bay_train_classes.yaml")
    ap.add_argument("--max-side", type=int, default=1600,
                    help="Write images resized to this max side (0 = hard-link originals). Labels are normalized.")
    ap.add_argument("--allow-unreviewed-eval", action="store_true")
    ap.add_argument("--drop-empty-train", type=float, default=0.0,
                    help="Fraction of label-free train images to drop (0 keeps all as negatives)")
    return ap.parse_args()


def main():
    args = parse_args()
    ann_names = yaml.safe_load(args.annotation_config.read_text(encoding="utf-8"))["names"]
    tcfg = yaml.safe_load(args.train_classes.read_text(encoding="utf-8"))
    t_names = {int(k): v for k, v in tcfg["names"].items()}
    t_id = {v: k for k, v in t_names.items()}
    remap = {}
    for aid, aname in ann_names.items():
        target = tcfg["map"].get(aname, None)
        remap[int(aid)] = t_id[target] if target is not None else None

    out = args.out.resolve()
    stats = {}
    import random

    rng = random.Random(0)
    excluded_file = args.reviewed / "EXCLUDED.txt"
    excluded = set(excluded_file.read_text(encoding="utf-8").split()) if excluded_file.exists() else set()
    for split in ("train", "val", "test"):
        seg_lbl = args.seg / "labels" / split
        rev_lbl = args.reviewed / "labels" / split
        stems = {p.stem for p in seg_lbl.glob("*.txt")} | {p.stem for p in rev_lbl.glob("*.txt")}
        c = Counter()
        cls = Counter()
        for stem in sorted(stems):
            if f"{split}/{stem}" in excluded:
                c["excluded_by_review"] += 1
                continue
            if (rev_lbl / f"{stem}.txt").exists():
                lp, img_dir, source = rev_lbl / f"{stem}.txt", args.reviewed / "images" / split, "reviewed"
            elif split == "train" or args.allow_unreviewed_eval:
                lp, img_dir, source = seg_lbl / f"{stem}.txt", args.seg / "images" / split, "auto"
            else:
                c["skipped_unreviewed"] += 1
                continue
            imgs = list(img_dir.glob(f"{stem}.*"))
            if not imgs:
                c["missing_image"] += 1
                continue
            lines = []
            for line in lp.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if not parts:
                    continue
                new = remap.get(int(parts[0]))
                if new is None:
                    c["instances_dropped_class"] += 1
                    continue
                lines.append(" ".join([str(new), *parts[1:]]))
                cls[t_names[new]] += 1
            if split == "train" and not lines and args.drop_empty_train > 0 and rng.random() < args.drop_empty_train:
                c["empty_dropped"] += 1
                continue
            dst = out / "images" / split / (imgs[0].stem + ".jpg")
            if args.max_side <= 0:
                materialize(imgs[0], out / "images" / split / imgs[0].name, copy=False)
            elif not dst.exists():
                write_resized(imgs[0], dst, args.max_side)
            atomic_write(out / "labels" / split / f"{stem}.txt", "\n".join(lines) + ("\n" if lines else ""))
            c[f"images_{source}"] += 1
            c["empty_images"] += not lines
        stats[split] = {"counts": dict(c), "instances_per_class": dict(cls.most_common())}

    cfg = {"path": str(out).replace("\\", "/"), "train": "images/train", "val": "images/val", "test": "images/test",
           "nc": len(t_names), "names": t_names}
    atomic_write(out / "data_engine_bay_train.yaml", yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    atomic_write(out / "BUILD_SUMMARY.json", json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps({s: v["counts"] for s, v in stats.items()}, indent=2))
    print(f"[build] dataset yaml: {out / 'data_engine_bay_train.yaml'}")


if __name__ == "__main__":
    main()
