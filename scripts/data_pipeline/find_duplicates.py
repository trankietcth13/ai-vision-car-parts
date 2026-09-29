"""
Find exact and near-duplicate photos in dataset/ (roadmap phase 3: image-level CV folds must keep duplicates together,
otherwise the same photo sits in train and val and inflates the score).

exact : identical file bytes (sha1)
near  : 16x16 difference-hash (dHash, 256 bits) Hamming distance <= --max-dist on the grayscale thumbnail

Output data/engine_bay_phase2/DUPLICATES.json: {"groups": [[stem, ...], ...]} (stems use the annotator convention) and a
summary. build_full_dataset.py --dup-groups puts every group in one fold.
Usage: python scripts/data_pipeline/find_duplicates.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "data_pipeline"))
from phase2_candidates import stem_for  # noqa: E402

IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def dhash(path: Path, n: int = 16):
    im = cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_8)
    if im is None:
        return None
    small = cv2.resize(im, (n + 1, n), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).flatten()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=ROOT / "dataset")
    ap.add_argument("--max-dist", type=int, default=10, help="max differing bits of 256 for a near duplicate")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "engine_bay_phase2" / "DUPLICATES.json")
    a = ap.parse_args()
    imgs = sorted(p for p in a.dataset.rglob("*") if p.suffix.lower() in IMG)
    stems, sha, hashes = [], {}, []
    for p in imgs:
        s = stem_for(a.dataset, p)
        stems.append(s)
        sha[s] = hashlib.sha1(p.read_bytes()).hexdigest()
        hashes.append(dhash(p))
    parent = list(range(len(stems)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    by_sha = defaultdict(list)
    for i, s in enumerate(stems):
        by_sha[sha[s]].append(i)
    exact_pairs = 0
    for idx in by_sha.values():
        for j in idx[1:]:
            union(idx[0], j)
            exact_pairs += 1
    valid = [i for i, h in enumerate(hashes) if h is not None]
    H = np.array([hashes[i] for i in valid], dtype=bool)
    near_pairs = []
    for k, i in enumerate(valid):
        d = (H[k + 1:] != H[k]).sum(1)
        for m in np.nonzero(d <= a.max_dist)[0]:
            j = valid[k + 1 + m]
            near_pairs.append((stems[i], stems[j], int(d[m])))
            union(i, j)
    groups = defaultdict(list)
    for i, s in enumerate(stems):
        groups[find(i)].append(s)
    groups = sorted((sorted(g) for g in groups.values() if len(g) > 1), key=lambda g: g[0])
    cross_vehicle = [g for g in groups if len({s.split("__")[0] for s in g}) > 1]
    out = {"images": len(stems), "exact_duplicate_pairs": exact_pairs, "near_pairs": len(near_pairs),
           "groups": groups, "images_in_groups": sum(len(g) for g in groups), "cross_vehicle_groups": cross_vehicle,
           "near_pair_list": near_pairs, "max_dist": a.max_dist}
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("groups", "near_pair_list", "cross_vehicle_groups")}))
    print("cross-vehicle groups:", len(cross_vehicle), cross_vehicle[:3])
    print("distance histogram:", np.bincount([d for _, _, d in near_pairs], minlength=a.max_dist + 1).tolist())


if __name__ == "__main__":
    main()
