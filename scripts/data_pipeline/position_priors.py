"""Per-class position priors (where a component usually sits in a full engine-bay photo).

For every taxonomy-v2 training class, builds a smoothed 2-D histogram of normalized box centres plus
size statistics from reviewed labels. Used as one signal of the label-confidence model (W1e) and by the
context rules of the teacher system (W4). Only TRAIN-split images are used, so val/test stay clean.

Caveat: photos are not registered (different vehicles, viewpoints, close-ups), so the prior is weak
by nature; treat it as a feature, never as a hard filter.

    python scripts/data_pipeline/position_priors.py --src data/engine_bay_reviewed data/engine_bay_reviewed_hybrid \
        --out artifacts/priors/position_priors_v2.json
    # lookup
    from data_pipeline.position_priors import PositionPriors
    pp = PositionPriors.load("artifacts/priors/position_priors_v2.json")
    pp.score("battery", cx=0.2, cy=0.4)     # density relative to a uniform prior (1.0 = uninformative)
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_pipeline.taxonomy import load_taxonomy  # noqa: E402
from data_pipeline.taxonomy_stats import read_names  # noqa: E402

GRID = 10


def box_of(parts):
    xy = np.asarray(parts, float).reshape(-1, 2)
    return xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()


def build(srcs, split="train", grid=GRID, alpha=1.0):
    tax = load_taxonomy()
    centers = collections.defaultdict(list)
    sizes = collections.defaultdict(list)
    per_image = collections.defaultdict(list)  # instances of a class per image where it occurs
    seen = set()
    for src in srcs:
        names = read_names(src)
        for lp in sorted((src / "labels" / split).glob("*.txt")):
            if lp.stem in seen:
                continue
            seen.add(lp.stem)
            img_count = collections.Counter()
            for line in lp.read_text().splitlines():
                p = line.split()
                if len(p) < 5:
                    continue
                tc = tax.train_class_for(names[int(p[0])])
                if tc is None:
                    continue
                x1, y1, x2, y2 = box_of(p[1:])
                centers[tc].append(((x1 + x2) / 2, (y1 + y2) / 2))
                sizes[tc].append(np.sqrt(max(x2 - x1, 1e-6) * max(y2 - y1, 1e-6)))
                img_count[tc] += 1
            for tc, k in img_count.items():
                per_image[tc].append(k)
    out = {"grid": grid, "split": split, "images": len(seen), "classes": {}}
    for cls in tax.training_names:
        pts = np.asarray(centers.get(cls, []), float).reshape(-1, 2)
        hist = np.full((grid, grid), alpha)  # Laplace smoothing keeps unseen cells > 0
        for cx, cy in pts:
            hist[min(int(cy * grid), grid - 1), min(int(cx * grid), grid - 1)] += 1
        dens = hist / hist.sum() * grid * grid  # 1.0 == uniform
        s = np.asarray(sizes.get(cls, []), float)
        out["classes"][cls] = {
            "n": int(len(pts)),
            "density": np.round(dens, 3).tolist(),
            "zone_p10_p90": ([float(v) for v in np.percentile(pts, 10, axis=0)] + [float(v) for v in np.percentile(pts, 90, axis=0)])
            if len(pts) >= 5 else None,
            "size_sqrt_area": {"p10": float(np.percentile(s, 10)), "p50": float(np.median(s)), "p90": float(np.percentile(s, 90))}
            if len(s) >= 5 else None,
            # max instances per image (used by the teacher system to cap duplicates); None = too little data
            "count_max": int(max(per_image[cls])) if len(per_image.get(cls, [])) >= 5 else None,
        }
    return out


class PositionPriors:
    def __init__(self, data: dict):
        self.data = data
        self.grid = data["grid"]

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def score(self, cls: str, cx: float, cy: float) -> float:
        c = self.data["classes"].get(cls)
        if not c or c["n"] < 5:
            return 1.0
        g = self.grid
        return float(c["density"][min(max(int(cy * g), 0), g - 1)][min(max(int(cx * g), 0), g - 1)])

    def size_z(self, cls: str, sqrt_area: float) -> float:
        """0 inside the p10-p90 size band, otherwise log-distance to the band (larger = more unusual)."""
        c = self.data["classes"].get(cls)
        if not c or not c["size_sqrt_area"]:
            return 0.0
        lo, hi = c["size_sqrt_area"]["p10"], c["size_sqrt_area"]["p90"]
        if lo <= sqrt_area <= hi:
            return 0.0
        return float(abs(np.log(sqrt_area / (lo if sqrt_area < lo else hi))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", nargs="+", required=True, type=Path)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", type=Path, default=Path("artifacts/priors/position_priors_v2.json"))
    args = ap.parse_args()
    data = build(args.src, args.split)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data), encoding="utf-8")
    print(f"{data['images']} {args.split} images -> {args.out}")
    for cls, c in data["classes"].items():
        peak = max(max(r) for r in c["density"])
        print(f"  {cls:30s} n={c['n']:4d} peak density {peak:5.2f}  zone {c['zone_p10_p90'] and [round(v, 2) for v in c['zone_p10_p90']]}")


if __name__ == "__main__":
    main()
