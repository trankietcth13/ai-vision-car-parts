"""Apply D2 geometry-audit corrections to reviewed YOLO-seg labels -> NEW label folders (originals untouched).

Input : data/engine_bay_review_geometry/CORRECTIONS.jsonl  (key = "<reviewed dir>/<split>/<stem>#<line index>")
Output: data/<reviewed dir>_geo/labels/<split>/<stem>.txt for every touched image (other images: use the original)
        data/engine_bay_review_geometry/APPLIED.json

    needs_fix        re-segment with SAM2 from fixed_box_norm (EXIF-upright image, box padded 5 %), same class
    split_instances  one SAM2 instance per box in `boxes`
    wrong_class      keep the polygon, new class if it is in the 36-class ontology, else drop the line
    not_a_component  drop the line
Corrections below --min-conf are skipped (listed in APPLIED.json).

    python scripts/data_pipeline/apply_geometry_corrections.py --device cpu     # ~40 boxes, a few minutes on CPU
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from data_pipeline.refine_masks_sam2 import mask_to_polygon  # noqa: E402
from data_pipeline.taxonomy_stats import read_names  # noqa: E402
from inference.teacher_system import open_upright  # noqa: E402

GEO = ROOT / "data" / "engine_bay_review_geometry"


class BoxSegmenter:
    def __init__(self, ckpt="sam2.1_b.pt", device="cpu", max_side=1536):
        from ultralytics import SAM
        self.model, self.device, self.max_side = SAM(ckpt), device, max_side

    def __call__(self, image_path, box_norm, pad=0.05):
        im = open_upright(image_path, draft_side=self.max_side)
        im.thumbnail((self.max_side, self.max_side))
        W, H = im.size
        x1, y1, x2, y2 = box_norm
        pw, ph = (x2 - x1) * pad, (y2 - y1) * pad
        b = [max(0, (x1 - pw) * W), max(0, (y1 - ph) * H), min(W, (x2 + pw) * W), min(H, (y2 + ph) * H)]
        r = self.model.predict(np.array(im)[:, :, ::-1], bboxes=[b], device=self.device, verbose=False)[0]
        if r.masks is None or not len(r.masks):
            return None
        m = r.masks.data[0].cpu().numpy() > 0.5
        # keep the mask inside the (padded) box: the reviewer's box is the extent decision
        clip = np.zeros_like(m)
        clip[int(b[1]):int(b[3]) + 1, int(b[0]):int(b[2]) + 1] = True
        poly = mask_to_polygon(m & clip, 0.002 * float(np.hypot(W, H)), 64)
        if poly is None:
            return None
        return [(float(x) / W, float(y) / H) for x, y in poly]


def polygon_fill(pts, box):
    """Polygon area / box area (normalised coordinates)."""
    x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
    area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return area / max((box[2] - box[0]) * (box[3] - box[1]), 1e-9)


def fmt(cls_id, pts):
    return f"{cls_id} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in pts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corrections", type=Path, default=GEO / "CORRECTIONS.jsonl")
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sam", default=str(ROOT / "sam2.1_b.pt"))
    a = ap.parse_args()

    corr = [json.loads(l) for l in a.corrections.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_file = collections.defaultdict(list)
    for c in corr:
        path, line = c["key"].split("#")
        by_file[path].append((int(line), c))
    seg = None
    applied, skipped = collections.Counter(), []
    for path, items in by_file.items():
        src_dir, split, stem = path.split("/")
        src = ROOT / "data" / src_dir
        names = read_names(src)
        ids = {v: k for k, v in names.items()}
        lines = (src / "labels" / split / f"{stem}.txt").read_text().splitlines()
        img = next(p for e in (".jpg", ".jpeg", ".png") if (p := src / "images" / split / (stem + e)).exists())
        new_lines = {i: [l] for i, l in enumerate(lines)}
        for li, c in items:
            if float(c.get("confidence") or 0) < a.min_conf:
                skipped.append({"key": c["key"], "reason": f"confidence {c.get('confidence')}"})
                continue
            cls_id = int(lines[li].split()[0])
            v = c["verdict"]
            if v == "not_a_component" or (v == "wrong_class" and c.get("new_class") not in ids):
                new_lines[li] = []
                applied["dropped"] += 1
            elif v == "wrong_class":
                new_lines[li] = [" ".join([str(ids[c["new_class"]]), *lines[li].split()[1:]])]
                applied["reclassed"] += 1
            elif v in ("needs_fix", "split_instances"):
                boxes = [c["fixed_box_norm"]] if v == "needs_fix" else (c.get("boxes") or [])
                seg = seg or BoxSegmenter(a.sam, a.device)
                out = []
                for b in boxes:
                    if not b:
                        continue
                    pts = seg(img, b)
                    # SAM2 fragments big boxy parts split by hold-downs/cables: fall back to the reviewer's box when the
                    # mask fills too little of it (battery case is ~rectangular from above; ducts/terminals are not)
                    min_fill = 0.5 if names[cls_id] == "battery" else 0.2
                    if pts and polygon_fill(pts, b) < min_fill:
                        pts = None
                    out.append(fmt(cls_id, pts if pts else [(b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3])]))
                    applied["resegmented" if pts else "box_fallback"] += 1
                if out:
                    new_lines[li] = out
        dst = ROOT / "data" / f"{src_dir}_geo" / "labels" / split / f"{stem}.txt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text("\n".join(l for i in sorted(new_lines) for l in new_lines[i]) + "\n", encoding="utf-8")
        yml = ROOT / "data" / f"{src_dir}_geo" / "data_engine_bay_reviewed.yaml"
        if not yml.exists():
            yml.write_text(yaml.safe_dump({"names": names, "nc": len(names),
                                           "note": f"only images touched by the D2 geometry audit; others: data/{src_dir}"},
                                          sort_keys=False, allow_unicode=True), encoding="utf-8")
    (GEO / "APPLIED.json").write_text(json.dumps({"applied": dict(applied), "skipped": skipped,
                                                  "files": sorted(by_file)}, indent=1), encoding="utf-8")
    print(json.dumps({"applied": dict(applied), "skipped": len(skipped), "files": len(by_file)}, indent=1))


if __name__ == "__main__":
    main()
