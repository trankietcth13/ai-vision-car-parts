"""Apply geometry-matched label patches (build_label_patches.py) to the YOLO-seg labels of a NEW build.

Each patch finds, in labels/<stem>.txt, the line whose class (mapped to the build's ontology) matches and whose box
has the highest IoU >= --min-iou with the patch box, and replaces it with the patch polygons (none = removal).
Patch class names are annotation (36-class) names; they are mapped to the build's names with the training map
(configs/engine_bay_train_classes.yaml, e.g. radiator_hose_upper -> radiator_hose; unmapped -> instance removed)
or, with --taxonomy, through taxonomy v2. Lines are rewritten through a temp file so hard links elsewhere are not
modified. Use it only on a build directory created for this purpose (never on an older build).

    python scripts/data_pipeline/patch_labels.py --labels data/engine_bay_full_v10/labels/all \
        --data-yaml data/engine_bay_full_v10/data_full.yaml --patches data/label_patches_v10.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


def box_of(parts):
    xy = np.asarray(parts, float).reshape(-1, 2)
    return [float(xy[:, 0].min()), float(xy[:, 1].min()), float(xy[:, 0].max()), float(xy[:, 1].max())]


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def class_mapper(train_classes: Path | None, taxonomy: Path | None):
    if taxonomy:
        from data_pipeline.taxonomy import load_taxonomy
        tax = load_taxonomy(taxonomy)
        return tax.train_class_for
    tc = yaml.safe_load((train_classes or ROOT / "configs" / "engine_bay_train_classes.yaml").read_text(encoding="utf-8"))
    m = tc["map"]
    return lambda name: m.get(name, name if name in tc["names"].values() else None)


def apply(lines, patch, name_to_id, mapper, min_iou):
    """Return (new_lines, status) for one patch on one file's lines."""
    want = mapper(patch["match"]["class"])
    if want is None or want not in name_to_id:
        return lines, "class_not_in_build"
    cid = name_to_id[want]
    best, bi = min_iou, None
    for i, l in enumerate(lines):
        p = l.split()
        if len(p) >= 7 and int(p[0]) == cid:
            v = iou(box_of(p[1:]), patch["match"]["box"])
            if v >= best:
                best, bi = v, i
    if bi is None:
        return lines, "unmatched"
    repl = []
    for r in patch["replace"]:
        c = mapper(r["class"])
        if c is None or c not in name_to_id:
            continue
        repl.append(f"{name_to_id[c]} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in r["polygon"]))
    return lines[:bi] + repl + lines[bi + 1:], ("removed" if not repl else "replaced")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, required=True, help="labels folder of the new build (e.g. labels/all)")
    ap.add_argument("--data-yaml", type=Path, required=True, help="yaml with the build's names")
    ap.add_argument("--patches", type=Path, default=ROOT / "data" / "label_patches_v10.jsonl")
    ap.add_argument("--train-classes", type=Path, default=None)
    ap.add_argument("--taxonomy", type=Path, default=None)
    ap.add_argument("--min-iou", type=float, default=0.8)
    a = ap.parse_args()

    names = yaml.safe_load(a.data_yaml.read_text(encoding="utf-8"))["names"]
    names = dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}
    name_to_id = {v: k for k, v in names.items()}
    mapper = class_mapper(a.train_classes, a.taxonomy)
    patches = [json.loads(l) for l in a.patches.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_stem = collections.defaultdict(list)
    for p in patches:
        by_stem[p["stem"]].append(p)
    status, details = collections.Counter(), []
    for stem, ps in by_stem.items():
        lp = a.labels / f"{stem}.txt"
        if not lp.exists() and (a.labels / f"EXT__{stem}.txt").exists():  # external images carry an EXT__ prefix in builds
            lp = a.labels / f"EXT__{stem}.txt"
        if not lp.exists():
            status["image_not_in_build"] += len(ps)
            continue
        lines = [l for l in lp.read_text().splitlines() if l.strip()]
        for p in ps:
            lines, st = apply(lines, p, name_to_id, mapper, a.min_iou)
            status[st] += 1
            details.append({"stem": stem, "source": p.get("source"), "status": st})
        tmp = lp.with_suffix(".tmp")
        tmp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        os.replace(tmp, lp)  # new inode: hard-linked copies in other folders keep the old content
    out = a.labels.parent / f"PATCH_REPORT_{a.labels.name}.json"
    out.write_text(json.dumps({"status": dict(status), "details": details}, indent=1), encoding="utf-8")
    print(json.dumps(dict(status), indent=1))


if __name__ == "__main__":
    main()
