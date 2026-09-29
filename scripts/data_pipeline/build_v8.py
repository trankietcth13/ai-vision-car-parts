"""
Build dataset v8 = v7 with the DeepSeek-relabelled, expert-reviewed train images merged in.

Source: apply_review.py output of the hybrid round (data/engine_bay_reviewed_hybrid, split "train",
36-class annotation ids). Every image there is a formerly unreviewed train image whose Qwen labels were
replaced by DeepSeek V4 Flash Vision boxes -> SAM2 masks -> expert review.

Per reviewed image:
  * already in base train  -> its label file is REPLACED by the reviewed one (the base usually has a
                              teacher pseudo-label; --keep-codex keeps Codex-corrected labels instead)
  * not in base train      -> image (resized to --max-side) + label ADDED, one train-list entry
  * vehicle (Request_ID) in base val/test -> SKIPPED, so no val/test vehicle leaks into train
  * excluded by the reviewer (EXCLUDED.txt) -> skipped; an existing base label is left as is
Class ids are remapped to the training ontology (configs/engine_bay_train_classes.yaml); oil_filter and
non-training classes are dropped, as in v6/v7. val/test and the train list order/repeats are unchanged,
so v8 results stay directly comparable with v7.

Base files are hard links: labels are unlinked before being rewritten so the base dataset is never modified.

Usage (DGX):
    python scripts/data_pipeline/build_v8.py --base data/engine_bay_train_v7 \
        --hybrid data/engine_bay_reviewed_hybrid --out data/engine_bay_train_v8
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def link(src: Path, dst: Path):
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_resized(src: Path, dst: Path, max_side: int):
    import cv2

    img = cv2.imread(str(src))  # applies EXIF orientation, same as annotation/SAM stages
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 95])


def vehicle(stem: str) -> str:
    return stem.split("__")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True, help="dataset to extend (v7)")
    ap.add_argument("--hybrid", type=Path, default=ROOT / "data" / "engine_bay_reviewed_hybrid")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ann-config", type=Path, default=ROOT / "configs" / "data_engine_bay.yaml")
    ap.add_argument("--train-classes", type=Path, default=ROOT / "configs" / "engine_bay_train_classes.yaml")
    ap.add_argument("--keep", type=Path, default=ROOT / "keep_labels_v6.json",
                    help="v6 protected stems; used with --keep-codex")
    ap.add_argument("--keep-codex", action="store_true",
                    help="keep Codex-corrected base labels instead of replacing them")
    ap.add_argument("--max-side", type=int, default=1600, help="resize added images (base v5s/v6 use 1600)")
    ap.add_argument("--hybrid-split", default="train",
                    help="split folder inside --hybrid holding the reviewed labels (train for v8, p2 for the Phase 2 round)")
    ap.add_argument("--label-ids", choices=("annotation", "train"), default="annotation",
                    help="class-id space of the --hybrid labels: 36-class annotation ids (apply_review output, remapped) "
                         "or training ids already (e.g. the verified Phase 2 view, used as is)")
    ap.add_argument("--tag", default="v8", help="name used for the train list / summary files (v8, v9, ...)")
    a = ap.parse_args()

    base, hyb, out = a.base.resolve(), a.hybrid.resolve(), a.out.resolve()
    if out in (base, hyb):
        raise SystemExit("--out must differ from --base and --hybrid")
    cfg = yaml.safe_load((base / "data_engine_bay_train.yaml").read_text(encoding="utf-8"))
    ann = {int(k): v for k, v in yaml.safe_load(a.ann_config.read_text(encoding="utf-8"))["names"].items()}
    tc = yaml.safe_load(a.train_classes.read_text(encoding="utf-8"))
    t_id = {v: int(k) for k, v in tc["names"].items()}
    remap = {k: (t_id[tc["map"][n]] if tc["map"].get(n) else None) for k, n in ann.items()}
    remap = {k: (None if v == t_id.get("oil_filter") else v) for k, v in remap.items()}
    codex = set(json.loads(a.keep.read_text(encoding="utf-8")).get("codex", [])) if a.keep_codex else set()

    # copy base as-is (hard links)
    for split in ("train", "val", "test"):
        for p in (base / "images" / split).iterdir():
            link(p, out / "images" / split / p.name)
        for p in (base / "labels" / split).glob("*.txt"):
            link(p, out / "labels" / split / p.name)
    lst = (base / cfg["train"]).read_text().split()

    base_train = {p.stem for p in (base / "images" / "train").iterdir() if p.suffix.lower() in IMG_SUFFIXES}
    held_out_vehicles = {vehicle(p.stem) for split in ("val", "test") for p in (base / "images" / split).iterdir()}
    leak = sorted(s for s in base_train if vehicle(s) in held_out_vehicles)
    if leak:
        print(f"WARNING: base train already holds {len(leak)} images of val/test vehicles, e.g. {leak[:3]}")

    excluded = set((hyb / "EXCLUDED.txt").read_text().split()) if (hyb / "EXCLUDED.txt").exists() else set()
    stats, cls_before, cls_after = Counter(), Counter(), Counter()
    changes = []
    for lp in sorted((hyb / "labels" / a.hybrid_split).glob("*.txt")):
        stem = lp.stem
        if f"{a.hybrid_split}/{stem}" in excluded or stem in excluded:
            stats["skipped_excluded"] += 1
            continue
        if vehicle(stem) in held_out_vehicles:
            stats["skipped_val_test_vehicle"] += 1
            continue
        if stem in codex:
            stats["kept_codex_label"] += 1
            continue
        lines = []
        for line in lp.read_text().splitlines():
            s = line.split()
            if len(s) < 7:
                continue
            new = int(s[0]) if a.label_ids == "train" else remap.get(int(s[0]))
            if new is None or (a.label_ids == "train" and new == t_id.get("oil_filter")):
                stats["instances_dropped_non_training_class"] += 1
                continue
            lines.append(" ".join([str(new), *s[1:]]))
            cls_after[tc["names"][new]] += 1

        dst = out / "labels" / "train" / f"{stem}.txt"
        if stem in base_train:
            old = dst.read_text().splitlines() if dst.exists() else []
            for line in old:
                if line.split():
                    cls_before[tc["names"][int(line.split()[0])]] += 1
            if dst.exists():
                dst.unlink()  # break the hard link to the base label before writing
            dst.write_text("\n".join(lines) + ("\n" if lines else ""))
            stats["labels_replaced"] += 1
            stats["instances_before"] += len(old)
            stats["instances_after"] += len(lines)
            changes.append({"stem": stem, "action": "replaced", "before": len(old), "after": len(lines)})
        else:
            if not lines:
                stats["skipped_new_empty"] += 1
                continue
            img = next(p for p in (hyb / "images" / a.hybrid_split).glob(f"{stem}.*") if p.suffix.lower() in IMG_SUFFIXES)
            write_resized(img, out / "images" / "train" / f"{stem}.jpg", a.max_side)
            dst.write_text("\n".join(lines) + "\n")
            lst.append(f"./images/train/{stem}.jpg")
            stats["images_added"] += 1
            stats["instances_after"] += len(lines)
            changes.append({"stem": stem, "action": "added", "after": len(lines)})

    (out / f"train_{a.tag}.txt").write_text("\n".join(lst) + "\n")
    cfg["path"] = str(out)
    cfg["train"] = f"train_{a.tag}.txt"
    (out / "data_engine_bay_train.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    summary = {
        **stats,
        "base": str(base),
        "keep_codex": a.keep_codex,
        "base_train_images_of_val_test_vehicles": len(leak),
        "train_list_entries": len(lst),
        "instances_per_class_before": dict(cls_before.most_common()),
        "instances_per_class_after": dict(cls_after.most_common()),
    }
    (out / f"{a.tag.upper()}_SUMMARY.json").write_text(json.dumps(summary, indent=2))
    (out / f"{a.tag.upper()}_CHANGES.jsonl").write_text("".join(json.dumps(c) + "\n" for c in changes))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
