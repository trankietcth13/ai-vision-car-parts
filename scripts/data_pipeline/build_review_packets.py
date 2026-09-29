"""
Build per-image review packets for the engine-bay label audit.

For every image in the SAM2-refined dataset (`data/engine_bay_seg`) this writes to
`data/engine_bay_review/packets/<split>/`:
    <stem>.jpg        overlay: numbered instances (#id class), mask tint, dropped candidates as D<id>
    <stem>_clean.jpg  the same image without annotations (resized)
    <stem>.json       instance list consumed by the reviewer (see .claude/skills/engine-bay-label-review)

Already-reviewed images (verdict file exists) are skipped unless --force.

Usage:
    python scripts/data_pipeline/build_review_packets.py --splits val test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", type=Path, default=PROJECT_ROOT / "data" / "engine_bay_seg")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "engine_bay_review")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stems-file", type=Path, default=None, help="Only build packets for stems listed in this file")
    return ap.parse_args()


def color_for(i: int):
    rng = np.random.default_rng(1000 + i)
    return tuple(int(c) for c in rng.integers(40, 255, size=3))


def draw_tag(img, text, x, y, color):
    font, scale, th = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    (tw, tht), _ = cv2.getTextSize(text, font, scale, th)
    x = int(np.clip(x, 0, img.shape[1] - tw - 4))
    y = int(np.clip(y, tht + 4, img.shape[0] - 2))
    cv2.rectangle(img, (x, y - tht - 4), (x + tw + 4, y + 2), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 2, y - 2), font, scale, color, th, cv2.LINE_AA)


def dashed_rect(img, p1, p2, color, dash=10):
    x1, y1 = p1
    x2, y2 = p2
    for x in range(x1, x2, dash * 2):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), color, 2)
        cv2.line(img, (x, y2), (min(x + dash, x2), y2), color, 2)
    for y in range(y1, y2, dash * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash, y2)), color, 2)
        cv2.line(img, (x2, y), (x2, min(y + dash, y2)), color, 2)


def main():
    args = parse_args()
    seg, out = args.seg.resolve(), args.out.resolve()
    n_written = 0
    for split in args.splits:
        meta_dir = seg / "refine_meta" / split
        if not meta_dir.exists():
            print(f"[packets] split '{split}': no refined data yet")
            continue
        metas = sorted(meta_dir.glob("*.json"))
        if args.stems_file:
            wanted = {l.strip() for l in args.stems_file.read_text(encoding="utf-8").splitlines() if l.strip()}
            metas = [m for m in metas if m.stem in wanted]
        if args.limit:
            metas = metas[: args.limit]
        pdir = out / "packets" / split
        pdir.mkdir(parents=True, exist_ok=True)
        for mp in metas:
            stem = mp.stem
            if not args.force and (out / "verdicts" / split / f"{stem}.json").exists():
                continue
            meta = json.loads(mp.read_text(encoding="utf-8"))
            img = cv2.imread(str(seg / meta["image"]))
            if img is None:
                continue
            h0, w0 = img.shape[:2]
            s = min(1.0, args.max_side / max(h0, w0))
            if s < 1.0:
                img = cv2.resize(img, (round(w0 * s), round(h0 * s)), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]

            lines = []
            lp = seg / "labels" / split / f"{stem}.txt"
            if lp.exists():
                lines = [l.split() for l in lp.read_text(encoding="utf-8").splitlines() if l.strip()]

            instances, li = [], 0
            tint = img.copy()
            draw_ops = []
            for k, inst in enumerate(meta["instances"]):
                b = inst["source_box_norm"]
                entry = {"id": k, "class_name": inst["class_name"], "class_id": inst["class_id"],
                         "confidence": inst.get("confidence"), "box_norm": [round(v, 4) for v in b]}
                px = (int(b[0] * w), int(b[1] * h), int(b[2] * w), int(b[3] * h))
                if inst["method"] == "dropped_for_review":
                    entry["status"] = "dropped_candidate"
                    draw_ops.append(("drop", k, inst["class_name"], px, None))
                else:
                    entry["status"] = "labeled"
                    poly = None
                    if li < len(lines):
                        coords = np.array(lines[li][1:], dtype=np.float32).reshape(-1, 2) * [w, h]
                        poly = coords.astype(np.int32)
                        entry["polygon_points"] = len(coords)
                    li += 1
                    draw_ops.append(("lab", k, inst["class_name"], px, poly))
                instances.append(entry)

            for kind, k, name, px, poly in draw_ops:
                if kind == "lab" and poly is not None:
                    cv2.fillPoly(tint, [poly], color_for(k))
            vis = cv2.addWeighted(tint, 0.35, img, 0.65, 0)
            for kind, k, name, px, poly in draw_ops:
                if kind == "lab":
                    c = color_for(k)
                    if poly is not None:
                        cv2.polylines(vis, [poly], True, c, 2)
                    cv2.rectangle(vis, px[:2], px[2:], c, 1)
                    draw_tag(vis, f"#{k} {name}", px[0], px[1], c)
                else:
                    dashed_rect(vis, px[:2], px[2:], (0, 0, 255))
                    draw_tag(vis, f"D{k} {name}?", px[0], px[1], (80, 80, 255))
            if not instances:
                draw_tag(vis, "NO LABELS - check for missing components", 10, 30, (0, 255, 255))

            cv2.imwrite(str(pdir / f"{stem}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
            cv2.imwrite(str(pdir / f"{stem}_clean.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            packet = {
                "image": stem, "split": split,
                "overlay": f"data/engine_bay_review/packets/{split}/{stem}.jpg",
                "clean_image": f"data/engine_bay_review/packets/{split}/{stem}_clean.jpg",
                "source_image": str((seg / meta["image"]).relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "width": w, "height": h, "instances": instances,
                "verdict_path": f"data/engine_bay_review/verdicts/{split}/{stem}.json",
            }
            (pdir / f"{stem}.json").write_text(json.dumps(packet, indent=2, ensure_ascii=False), encoding="utf-8")
            n_written += 1
        print(f"[packets] split '{split}': {len(list(pdir.glob('*.json')))} packets in {pdir}")
    print(f"[packets] wrote {n_written} packets")


if __name__ == "__main__":
    main()
