"""Turn label corrections into geometry-matched PATCHES that can be applied to any later build (v10+).

Why patches: corrections are made against reviewed labels, but training builds (v8/v9/full) layer several sources
on top (Codex fixes, hybrid replacements, Phase 2), so line indices are not stable. A patch says "the instance of
class C whose box is ~B in image S becomes <these polygons>" (empty = remove) and is matched by class + box IoU.

Sources:
  reservoir re-review  data/engine_bay_review_reservoir/CHANGES.json (+ verdicts for the fixed boxes); the label line
                       of a verdict instance comes from the reviewed dir's provenance file (instance order = line order)
  geometry audit (D2)  data/engine_bay_review_geometry/PATCHES.jsonl (written by apply_geometry_corrections.py)
Output: data/label_patches_v10.jsonl  (+ LABEL_PATCHES_SUMMARY.json)

    python scripts/data_pipeline/build_label_patches.py --device cpu
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from data_pipeline.apply_geometry_corrections import BoxSegmenter, polygon_fill  # noqa: E402
from data_pipeline.taxonomy_stats import read_names  # noqa: E402

REVIEWED = {"engine_bay_review": "engine_bay_reviewed", "engine_bay_review_hybrid": "engine_bay_reviewed_hybrid",
            "engine_bay_review_p2": "engine_bay_reviewed_p2"}


def box_of(line):
    xy = np.asarray(line.split()[1:], float).reshape(-1, 2)
    return [float(xy[:, 0].min()), float(xy[:, 1].min()), float(xy[:, 0].max()), float(xy[:, 1].max())]


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def reservoir_patches(seg_factory, report):
    changes = json.loads((ROOT / "data/engine_bay_review_reservoir/CHANGES.json").read_text(encoding="utf-8"))["changes"]
    fixed = {}
    for f in (ROOT / "data/engine_bay_review_reservoir/verdicts").glob("*.json"):
        for r in json.loads(f.read_text(encoding="utf-8")):
            fixed[r["key"]] = r
    out = []
    for ch in changes:
        key = ch["key"]
        path, inst = key.split("#")
        root, split, stem = path.split("/")
        rdir = ROOT / "data" / ("engine_bay_reviewed_ext" if split == "ext" else REVIEWED[root])
        lp, pp = rdir / "labels" / split / f"{stem}.txt", rdir / "provenance" / split / f"{stem}.json"
        if not lp.exists() or not pp.exists():
            report["reservoir_missing_reviewed_dir"] += 1
            print(f"[warn] {key}: missing {lp if not lp.exists() else pp} (patch skipped)", file=sys.stderr)
            continue
        names = read_names(rdir)
        lines = [l for l in lp.read_text().splitlines() if l.strip()]
        prov = json.loads(pp.read_text(encoding="utf-8"))["instances"]
        k = None
        if not inst.startswith("missing"):
            k = next((i for i, p in enumerate(prov) if p.get("id") == int(inst) and p.get("from") != "missing_added"), None)
        else:  # reviewer-added box: match the verdict's missing box against the label lines of that class
            v = json.loads((ROOT / "data" / root / "verdicts" / split / f"{stem}.json").read_text(encoding="utf-8"))
            mb = v["missing"][int(inst[len("missing"):])]["box_norm"]
            cands = [(iou(box_of(l), mb), i) for i, l in enumerate(lines) if names[int(l.split()[0])] == ch["from"]]
            k = max(cands)[1] if cands and max(cands)[0] >= 0.3 else None
        if k is None or k >= len(lines):
            report["reservoir_unmatched_line"] += 1
            continue
        line = lines[k]
        cls_name = names[int(line.split()[0])]
        patch = {"stem": stem, "source": f"reservoir:{key}", "match": {"class": cls_name, "box": box_of(line)}}
        act = ch["action"]
        if act in ("dropped", "removed_missing"):
            patch["replace"] = []
        elif act == "rename":
            pts = np.asarray(line.split()[1:], float).reshape(-1, 2).round(6).tolist()
            patch["replace"] = [{"class": ch["to"], "polygon": pts}]
        elif act in ("rebox", "rebox+rename"):
            b = fixed[key]["fixed_box_norm"]
            img = next(p for e in (".jpg", ".jpeg", ".png") if (p := rdir / "images" / split / (stem + e)).exists())
            pts = seg_factory()(img, b)
            min_fill = 0.5 if cls_name == "battery" else 0.2
            if not pts or polygon_fill(pts, b) < min_fill:
                pts = [(b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3])]
                report["reservoir_box_fallback"] += 1
            new_cls = ch["to"] if act == "rebox+rename" else cls_name
            patch["replace"] = [{"class": new_cls, "polygon": [[round(x, 6), round(y, 6)] for x, y in pts]}]
        else:
            continue
        report[f"reservoir_{act}"] += 1
        out.append(patch)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "label_patches_v10.jsonl")
    a = ap.parse_args()
    seg = {}

    def seg_factory():
        if "s" not in seg:
            seg["s"] = BoxSegmenter(str(ROOT / "sam2.1_b.pt"), a.device)
        return seg["s"]

    report = collections.Counter()
    patches = reservoir_patches(seg_factory, report)
    geo = ROOT / "data/engine_bay_review_geometry/PATCHES.jsonl"
    g = [json.loads(l) for l in geo.read_text(encoding="utf-8").splitlines() if l.strip()]
    report["geometry"] = len(g)
    patches += g
    a.out.write_text("".join(json.dumps(p) + "\n" for p in patches), encoding="utf-8")
    summary = {"patches": len(patches), "images": len({p["stem"] for p in patches}), **report}
    (a.out.parent / "LABEL_PATCHES_SUMMARY.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
