"""
Build dataset v6 (run on the DGX: the re-pseudo-labelling step needs the GPU).

Changes vs v5s:
  1. drop class `oil_filter` everywhere (9 train instances, AP 0 for every model); class ids stay stable
  2. re-pseudo-label train images that are neither expert-reviewed nor touched by Codex, using teacher v4
     (better than the v2 teacher that produced the old pseudo labels)
  3. cut label-free train images down to --empty-frac of train (kept ones are chosen at random)
  4. repeat-factor sampling (LVIS style): image repeat = max over its classes of max(1, sqrt(t / f_c)),
     f_c = fraction of train images containing class c; combined with Codex hard-example oversampling
     (max of both), capped at --max-repeat. Written to train_v6.txt.
val / test: copied, only the oil_filter instances are removed.

Usage (DGX):
    .venv/bin/python scripts/data_pipeline/build_v6.py --src data/engine_bay_train_v5s --out data/engine_bay_train_v6 \
        --teacher runs/segment/engine_teacher_v4_dgx/weights/best.pt --keep keep_labels_v6.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from collections import Counter
from pathlib import Path

import yaml


def link(src: Path, dst: Path):
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def read(lp: Path):
    return [l.split() for l in lp.read_text().splitlines() if l.strip()] if lp.exists() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--keep", type=Path, required=True, help="JSON {reviewed: [...stems], codex: [...stems], hard: [...stems]}")
    ap.add_argument("--drop-class", default="oil_filter")
    ap.add_argument("--conf", type=float, default=0.40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--empty-frac", type=float, default=0.15)
    ap.add_argument("--rfs-t", type=float, default=0.10)
    ap.add_argument("--hard-repeat", type=int, default=2)
    ap.add_argument("--max-repeat", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    src, out = a.src.resolve(), a.out.resolve()
    cfg = yaml.safe_load((src / "data_engine_bay_train.yaml").read_text(encoding="utf-8"))
    names = {int(k): v for k, v in cfg["names"].items()}
    drop_id = {v: k for k, v in names.items()}[a.drop_class]
    keep = json.loads(a.keep.read_text())
    protected = set(keep["reviewed"]) | set(keep["codex"])
    hard = set(keep.get("hard", []))
    rng = random.Random(a.seed)
    stats = Counter()

    def write(split, stem, lines):
        lines = [l for l in lines if int(l[0]) != drop_id]
        p = out / "labels" / split / f"{stem}.txt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(" ".join(l) for l in lines) + ("\n" if lines else ""))
        return lines

    for split in ("val", "test"):
        for lp in (src / "labels" / split).glob("*.txt"):
            before = read(lp)
            after = write(split, lp.stem, before)
            stats[f"{split}_{a.drop_class}_removed"] += len(before) - len(after)
            for ip in (src / "images" / split).glob(f"{lp.stem}.*"):
                link(ip, out / "images" / split / ip.name)

    # ---- train: protected labels kept, the rest re-pseudo-labelled with the teacher
    train_imgs = sorted(p for p in (src / "images" / "train").iterdir())
    todo = [p for p in train_imgs if p.stem not in protected]
    new_labels = {}
    from ultralytics import YOLO

    model = YOLO(a.teacher)
    for r in model.predict([str(p) for p in todo], imgsz=a.imgsz, conf=a.conf, device=0, stream=True, verbose=False):
        lines = []
        if r.masks is not None and len(r.masks):
            for poly, c in zip(r.masks.xyn, r.boxes.cls.tolist()):
                if poly is not None and len(poly) >= 3:
                    lines.append([str(int(c))] + [f"{min(max(v, 0.0), 1.0):.6f}" for pt in poly for v in pt])
        new_labels[Path(r.path).stem] = lines
    stats["train_repseudo"] = len(todo)
    stats["train_protected"] = len(train_imgs) - len(todo)

    labels = {}
    for p in train_imgs:
        lines = new_labels[p.stem] if p.stem in new_labels else read(src / "labels" / "train" / f"{p.stem}.txt")
        labels[p.stem] = [l for l in lines if int(l[0]) != drop_id]

    # ---- limit label-free images (drop non-protected empties first)
    empties = [s for s, l in labels.items() if not l]
    target = int(a.empty_frac * (len(labels) - len(empties)) / (1 - a.empty_frac))
    rng.shuffle(empties)
    empties.sort(key=lambda s: s in protected)  # unprotected first -> dropped first
    drop = set(empties[: max(0, len(empties) - target)])
    stats["train_empty_before"] = len(empties)
    stats["train_empty_dropped"] = len(drop)

    kept = {s: l for s, l in labels.items() if s not in drop}
    for p in train_imgs:
        if p.stem in kept:
            write("train", p.stem, kept[p.stem])
            link(p, out / "images" / "train" / p.name)

    # ---- repeat-factor sampling
    n = len(kept)
    img_classes = {s: {int(l[0]) for l in ls} for s, ls in kept.items()}
    freq = Counter(c for cs in img_classes.values() for c in cs)
    rc = {c: max(1.0, math.sqrt(a.rfs_t / (freq[c] / n))) for c in freq}
    entries, rep_hist = [], Counter()
    for p in train_imgs:
        if p.stem not in kept:
            continue
        r = max([rc[c] for c in img_classes[p.stem]] + [1.0])
        if p.stem in hard:
            r = max(r, a.hard_repeat)
        k = min(a.max_repeat, int(math.floor(r)) + (1 if rng.random() < r - math.floor(r) else 0))
        rep_hist[k] += 1
        entries += [f"./images/train/{p.name}"] * k
    (out / "train_v6.txt").write_text("\n".join(entries) + "\n")

    cls_count = Counter(int(l[0]) for ls in kept.values() for l in ls)
    cfg["path"] = str(out)
    cfg["train"] = "train_v6.txt"
    (out / "data_engine_bay_train.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    summary = {**stats, "train_images": n, "train_list_entries": len(entries),
               "repeat_histogram": dict(sorted(rep_hist.items())),
               "class_repeat_factor": {names[c]: round(v, 2) for c, v in sorted(rc.items(), key=lambda x: -x[1])},
               "train_instances_per_class": {names[c]: cls_count[c] for c in sorted(names) if c != drop_id}}
    (out / "V6_SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
