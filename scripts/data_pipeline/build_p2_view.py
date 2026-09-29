"""
Phase 2, step 4 (label verification): turn the reviewed Phase 2 labels into a small dataset in the TRAINING
ontology, so the existing error-mining / verification tools (mine_errors.py -> merge_errors.py -> verifier
verdicts -> apply_codex_verdicts.py) work on it unchanged.

Input : data/engine_bay_reviewed_p2 (apply_review.py output, 36-class ids, split p2, EXCLUDED.txt)
Output: data/engine_bay_p2_view/images|labels/train (+ empty val/test), data_engine_bay_train.yaml

Usage (DGX):
    .venv/bin/python scripts/data_pipeline/build_p2_view.py
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviewed", type=Path, default=ROOT / "data" / "engine_bay_reviewed_p2")
    ap.add_argument("--split", default="p2")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "engine_bay_p2_view")
    ap.add_argument("--ann-config", type=Path, default=ROOT / "configs" / "data_engine_bay.yaml")
    ap.add_argument("--train-classes", type=Path, default=ROOT / "configs" / "engine_bay_train_classes.yaml")
    a = ap.parse_args()

    ann = {int(k): v for k, v in yaml.safe_load(a.ann_config.read_text(encoding="utf-8"))["names"].items()}
    tc = yaml.safe_load(a.train_classes.read_text(encoding="utf-8"))
    names = {int(k): v for k, v in tc["names"].items()}
    t_id = {v: k for k, v in names.items()}
    remap = {k: (t_id[tc["map"][n]] if tc["map"].get(n) else None) for k, n in ann.items()}
    remap = {k: (None if v == t_id.get("oil_filter") else v) for k, v in remap.items()}
    ex = a.reviewed / "EXCLUDED.txt"
    excluded = {s.split("/")[-1] for s in ex.read_text().split()} if ex.exists() else set()

    out = a.out.resolve()
    for sub in ("images", "labels"):
        for sp in ("train", "val", "test"):
            (out / sub / sp).mkdir(parents=True, exist_ok=True)
    stats, cls = Counter(), Counter()
    for lp in sorted((a.reviewed / "labels" / a.split).glob("*.txt")):
        if lp.stem in excluded:
            stats["excluded"] += 1
            continue
        img = next((p for p in (a.reviewed / "images" / a.split).glob(f"{lp.stem}.*")), None)
        if img is None:
            stats["missing_image"] += 1
            continue
        lines = []
        for line in lp.read_text().splitlines():
            s = line.split()
            if len(s) < 7:
                continue
            new = remap.get(int(s[0]))
            if new is None:
                stats["dropped_non_training_class"] += 1
                continue
            lines.append(" ".join([str(new), *s[1:]]))
            cls[names[new]] += 1
        dst = out / "images" / "train" / f"{lp.stem}.jpg"
        if not dst.exists():
            try:
                os.link(img, dst)
            except OSError:
                shutil.copy2(img, dst)
        (out / "labels" / "train" / f"{lp.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        stats["images"] += 1
        stats["instances"] += len(lines)
    (out / "data_engine_bay_train.yaml").write_text(yaml.safe_dump(
        {"path": str(out), "train": "images/train", "val": "images/val", "test": "images/test",
         "nc": len(names), "names": names}, sort_keys=False, allow_unicode=True))
    summary = {**stats, "instances_per_class": dict(cls.most_common())}
    (out / "P2_VIEW_SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
