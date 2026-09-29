"""
Pseudo-label screened external images with a teacher and lay them out like data/engine_bay_seg, so the
existing review tools (build_review_packets.py -> expert-review skill -> apply_review.py) work unchanged
with split name "ext" (or any name given with --split, e.g. "p2" for the Phase 2 relabel round).

Run on the DGX (GPU). Input: a folder of images named <id>.jpg (uploaded kept candidates).
Output: <out>/images/ext, labels/ext (36-class annotation ids), refine_meta/ext.

Usage (DGX):
    .venv/bin/python scripts/data_pipeline/external_to_seg.py --images ext_upload --teacher runs/segment/engine_teacher_v6/weights/best.pt \
        --out data/engine_bay_ext_seg
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
TRAIN_TO_ANN = {"radiator_hose": "radiator_hose_upper"}  # merged training class -> an annotation class the reviewer can fix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ann-config", type=Path, default=ROOT / "configs" / "data_engine_bay.yaml")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max-side", type=int, default=1600)
    ap.add_argument("--split", default="ext", help="split name in the output layout (ext, p2, ...)")
    a = ap.parse_args()

    ann = {v: int(k) for k, v in yaml.safe_load(a.ann_config.read_text(encoding="utf-8"))["names"].items()}
    for sub in ("images", "labels", "refine_meta"):
        (a.out / sub / a.split).mkdir(parents=True, exist_ok=True)

    imgs = sorted(p for p in a.images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    # normalise images (size, jpg) first so labels and packets refer to the stored copy
    stored = []
    for p in imgs:
        im = cv2.imread(str(p))
        if im is None:
            continue
        h, w = im.shape[:2]
        s = min(1.0, a.max_side / max(h, w))
        if s < 1.0:
            im = cv2.resize(im, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        dst = a.out / "images" / a.split / f"{p.stem}.jpg"
        cv2.imwrite(str(dst), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
        stored.append(dst)

    from ultralytics import YOLO

    model = YOLO(a.teacher)
    n_inst = 0
    for r in model.predict([str(p) for p in stored], imgsz=a.imgsz, conf=a.conf, device=0, stream=True, verbose=False):
        p = Path(r.path)
        lines, meta = [], []
        if r.masks is not None and len(r.masks):
            for poly, box, c, cf in zip(r.masks.xyn, r.boxes.xyxyn.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                name = TRAIN_TO_ANN.get(r.names[int(c)], r.names[int(c)])
                if name not in ann or poly is None or len(poly) < 3:
                    continue
                lines.append(f"{ann[name]} " + " ".join(f"{min(max(v, 0), 1):.6f}" for pt in poly for v in pt))
                meta.append({"class_id": ann[name], "class_name": name, "confidence": round(float(cf), 3),
                             "source_box_norm": [round(v, 4) for v in box], "method": "sam2"})
        (a.out / "labels" / a.split / f"{p.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        (a.out / "refine_meta" / a.split / f"{p.stem}.json").write_text(json.dumps(
            {"source_annotation": "teacher_pseudo_label", "image": f"images/{a.split}/{p.name}", "teacher": a.teacher,
             "instances": meta}, indent=2))
        n_inst += len(lines)
    print(json.dumps({"images": len(stored), "pseudo_instances": n_inst}))


if __name__ == "__main__":
    main()
