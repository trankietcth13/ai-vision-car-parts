"""
Enlarge the validation split by moving whole Request_IDs (vehicles) out of train.

Rules (keeps evaluation honest):
  * a moved request leaves train completely - no image of that vehicle stays in training
  * only expert-reviewed images of a moved request enter val (their labels are the reviewed ones);
    its unreviewed / pseudo-labelled images are dropped
  * test is never touched
  * a train list file (e.g. train_v5.txt with oversampled hard examples) is filtered the same way

Usage:
    python scripts/data_pipeline/resplit_val.py --src data/engine_bay_train_v5 --out data/engine_bay_train_v5s \
        --move Request_ID_50 Request_ID_52 Request_ID_36
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


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
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--move", nargs="+", default=["Request_ID_50", "Request_ID_52", "Request_ID_36"])
    ap.add_argument("--reviewed", type=Path, default=ROOT / "data" / "engine_bay_reviewed" / "labels" / "train")
    a = ap.parse_args()

    src, out = a.src.resolve(), a.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; choose a new --out")
    cfg = yaml.safe_load((src / "data_engine_bay_train.yaml").read_text(encoding="utf-8"))
    reviewed = {p.stem for p in a.reviewed.glob("*.txt")}
    moved = set(a.move)
    stats = Counter()

    for split in ("val", "test"):
        for lp in (src / "labels" / split).glob("*.txt"):
            link(lp, out / "labels" / split / lp.name)
            for ip in (src / "images" / split).glob(f"{lp.stem}.*"):
                link(ip, out / "images" / split / ip.name)
            stats[f"{split}_kept"] += 1

    for lp in (src / "labels" / "train").glob("*.txt"):
        req = lp.stem.split("__")[0]
        imgs = list((src / "images" / "train").glob(f"{lp.stem}.*"))
        if req in moved:
            if lp.stem in reviewed:
                link(lp, out / "labels" / "val" / lp.name)
                for ip in imgs:
                    link(ip, out / "images" / "val" / ip.name)
                stats["train_to_val_reviewed"] += 1
            else:
                stats["train_dropped_unreviewed"] += 1
            continue
        link(lp, out / "labels" / "train" / lp.name)
        for ip in imgs:
            link(ip, out / "images" / "train" / ip.name)
        stats["train_kept"] += 1

    cfg["path"] = str(out).replace("\\", "/")
    train_entry = cfg.get("train", "images/train")
    if str(train_entry).endswith(".txt"):
        lines = (src / train_entry).read_text().splitlines()
        kept = [l for l in lines if l.strip() and Path(l).stem.split("__")[0] not in moved]
        (out / train_entry).write_text("\n".join(kept) + "\n")
        stats["train_list_entries"] = len(kept)
    for extra in ("V5_SUMMARY.json", "PSEUDO_SUMMARY.json"):
        if (src / extra).exists():
            shutil.copy2(src / extra, out / extra)
    (out / "data_engine_bay_train.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    summary = {"moved_requests": sorted(moved), **stats}
    (out / "RESPLIT_SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
