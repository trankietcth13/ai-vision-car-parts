"""
Curate external engine-bay / under-hood component images from H:\\AI_Datasets (no exterior or interior views).

Sources used
  * Wikimedia Commons (vehicle_components/wikimedia_commons): only under-hood component classes
    engine, battery, alternator, fuse_box, ignition_coil, throttle_body, air_filter, brake_master_cylinder
  * Toyota Corolla engine bay (engine_bay_sources/arabeitak_toyota_corolla): ONE vehicle, so only a capped,
    folder-stratified sample is used
Excluded on purpose: carparts-seg / drbimmer (exterior body), Wikimedia exterior & interior classes,
Engine Assembly (industrial bench assembly, not a vehicle engine bay), oil_filter (class dropped in v6).

Outputs (data/external_candidates/):
    candidates.csv          id, source, source_class, path, license, author, source_page
    sheets/sheet_XX.jpg     numbered 5x5 contact sheets for the quick relevance screen
    sheets/sheet_XX.json    id list per sheet in grid order
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WM_CLASSES = ["engine", "battery", "alternator", "fuse_box", "ignition_coil", "throttle_body", "air_filter",
              "brake_master_cylinder"]


def imread(p: Path):
    """cv2.imread fails on non-ASCII Windows paths; decode from bytes instead."""
    data = np.fromfile(str(p), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("H:/AI_Datasets"))
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "external_candidates")
    ap.add_argument("--corolla-cap", type=int, default=150)
    ap.add_argument("--cell", type=int, default=360)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rows = []

    wm = a.src / "vehicle_components" / "wikimedia_commons"
    meta = {}
    with open(wm / "metadata.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            meta[Path(r["file_path"]).name] = r
    for cls in WM_CLASSES:
        for p in sorted((wm / "images" / cls).glob("*")):
            m = meta.get(p.name, {})
            rows.append({"source": "wikimedia", "source_class": cls, "path": str(p), "license": m.get("license", ""),
                         "author": m.get("author", ""), "source_page": m.get("source_page_url", "")})

    cor = a.src / "engine_bay_sources" / "arabeitak_toyota_corolla"
    folders = [d for d in sorted(cor.iterdir()) if d.is_dir()]
    files = {d.name: sorted(p for p in d.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}) for d in folders}
    per = max(1, a.corolla_cap // max(1, len(files)))
    for name, fs in files.items():
        rng.shuffle(fs)
        for p in fs[:per]:
            rows.append({"source": "toyota_corolla", "source_class": name, "path": str(p),
                         "license": "Apache-2.0", "author": "ARabeitak (Wagdy, Hany, Raafat, Albert et al.)",
                         "source_page": "https://huggingface.co/datasets/stevenalbert10/Toyota-Corolla-Car-Parts"})

    # drop unreadable files, assign ids
    ok = []
    for r in rows:
        im = imread(Path(r["path"]))
        if im is None or min(im.shape[:2]) < 200:
            continue
        r["id"] = f"ext{len(ok):04d}"
        r["width"], r["height"] = im.shape[1], im.shape[0]
        ok.append(r)
    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "candidates.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "source", "source_class", "path", "width", "height", "license", "author", "source_page"])
        w.writeheader()
        w.writerows(ok)

    # numbered contact sheets, 25 per sheet
    sd = a.out / "sheets"
    sd.mkdir(exist_ok=True)
    c = a.cell
    for s in range(0, len(ok), 25):
        batch = ok[s:s + 25]
        cells = []
        for k, r in enumerate(batch):
            im = imread(Path(r["path"]))
            h, w = im.shape[:2]
            f = (c - 28) / max(h, w)
            im = cv2.resize(im, (max(1, int(w * f)), max(1, int(h * f))))
            cell = np.full((c, c, 3), 35, np.uint8)
            cell[28:28 + im.shape[0], :im.shape[1]] = im
            cv2.putText(cell, f"{k + 1}  {r['id']}", (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cells.append(cell)
        while len(cells) % 5:
            cells.append(np.full((c, c, 3), 35, np.uint8))
        grid = np.vstack([np.hstack(cells[i:i + 5]) for i in range(0, len(cells), 5)])
        n = s // 25
        cv2.imwrite(str(sd / f"sheet_{n:02d}.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 82])
        (sd / f"sheet_{n:02d}.json").write_text(json.dumps([{"cell": k + 1, "id": r["id"], "source": r["source"],
                                                              "source_class": r["source_class"]} for k, r in enumerate(batch)], indent=1))
    by = {}
    for r in ok:
        by[(r["source"], r["source_class"])] = by.get((r["source"], r["source_class"]), 0) + 1
    print(f"{len(ok)} readable candidates (of {len(rows)}), {(len(ok) + 24) // 25} sheets")
    for k, v in sorted(by.items()):
        print(" ", k, v)


if __name__ == "__main__":
    main()
