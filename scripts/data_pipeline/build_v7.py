"""
Build dataset v7 = v6 + expert-reviewed external engine-bay / component images (train split only).

External labels come from apply_review.py run on split "ext" (36-class annotation ids); they are remapped
to the 21 training classes (configs/engine_bay_train_classes.yaml), oil_filter and non-training classes dropped.
val / test are untouched, so v7 results stay directly comparable with v6.

Usage (DGX or local):
    python scripts/data_pipeline/build_v7.py --v6 data/engine_bay_train_v6 --ext data/engine_bay_reviewed_ext --out data/engine_bay_train_v7
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


def link(src: Path, dst: Path):
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v6", type=Path, required=True)
    ap.add_argument("--ext", type=Path, required=True, help="apply_review output with images/ext, labels/ext, EXCLUDED.txt")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ann-config", type=Path, default=ROOT / "configs" / "data_engine_bay.yaml")
    ap.add_argument("--train-classes", type=Path, default=ROOT / "configs" / "engine_bay_train_classes.yaml")
    ap.add_argument("--ext-repeat", type=int, default=1)
    a = ap.parse_args()

    v6, ext, out = a.v6.resolve(), a.ext.resolve(), a.out.resolve()
    cfg = yaml.safe_load((v6 / "data_engine_bay_train.yaml").read_text(encoding="utf-8"))
    ann = {int(k): v for k, v in yaml.safe_load(a.ann_config.read_text(encoding="utf-8"))["names"].items()}
    tc = yaml.safe_load(a.train_classes.read_text(encoding="utf-8"))
    t_id = {v: int(k) for k, v in tc["names"].items()}
    remap = {k: (t_id[tc["map"][n]] if tc["map"].get(n) else None) for k, n in ann.items()}
    remap = {k: (None if v == t_id.get("oil_filter") else v) for k, v in remap.items()}

    # copy v6 as-is
    for split in ("train", "val", "test"):
        for p in (v6 / "images" / split).iterdir():
            link(p, out / "images" / split / p.name)
        for p in (v6 / "labels" / split).glob("*.txt"):
            link(p, out / "labels" / split / p.name)
    lst = (v6 / cfg["train"]).read_text().split()

    excluded = set((ext / "EXCLUDED.txt").read_text().split()) if (ext / "EXCLUDED.txt").exists() else set()
    stats, cls = Counter(), Counter()
    for lp in sorted((ext / "labels" / "ext").glob("*.txt")):
        if f"ext/{lp.stem}" in excluded:
            stats["ext_excluded"] += 1
            continue
        lines = []
        for line in lp.read_text().splitlines():
            s = line.split()
            if len(s) < 7:
                continue
            new = remap.get(int(s[0]))
            if new is None:
                stats["ext_instances_dropped"] += 1
                continue
            lines.append(" ".join([str(new), *s[1:]]))
            cls[tc["names"][new]] += 1
        if not lines:
            stats["ext_empty_skipped"] += 1
            continue
        img = next((ext / "images" / "ext").glob(f"{lp.stem}.*"))
        name = f"EXT__{img.name}"
        link(img, out / "images" / "train" / name)
        (out / "labels" / "train" / f"EXT__{lp.stem}.txt").write_text("\n".join(lines) + "\n")
        lst += [f"./images/train/{name}"] * a.ext_repeat
        stats["ext_images_added"] += 1

    (out / "train_v7.txt").write_text("\n".join(lst) + "\n")
    cfg["path"] = str(out)
    cfg["train"] = "train_v7.txt"
    (out / "data_engine_bay_train.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    summary = {**stats, "train_list_entries": len(lst), "ext_instances_per_class": dict(cls.most_common())}
    (out / "V7_SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
