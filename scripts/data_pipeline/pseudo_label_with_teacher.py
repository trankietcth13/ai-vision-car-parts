"""
Generate YOLO-segmentation pseudo-labels for unlabeled images with a trained teacher.

The output folder follows the Ultralytics layout and can be merged with the
labeled set (or listed as an extra train source in the data yaml).

Usage:
    python scripts/data_pipeline/pseudo_label_with_teacher.py \
        --teacher runs/segment/teacher_yolo11m_seg/weights/best.pt \
        --images  data/unlabeled_exterior \
        --output  data/pseudo_exterior --conf 0.5 --imgsz 640
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--images", required=True, help="Folder with raw images (searched recursively)")
    ap.add_argument("--output", required=True, help="Output dataset root (images/train, labels/train)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--conf", type=float, default=0.5, help="Keep predictions with confidence >= conf")
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--min-points", type=int, default=6, help="Discard polygons with fewer vertices")
    ap.add_argument("--keep-empty", action="store_true", help="Also copy images that get no prediction")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.teacher)
    files = sorted(p for p in Path(args.images).rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not files:
        raise SystemExit(f"No images found under {args.images}")

    out_img = Path(args.output) / "images" / args.split
    out_lbl = Path(args.output) / "labels" / args.split
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    n_img, n_inst, n_empty = 0, 0, 0
    per_class = {}
    for r in model.predict(
        source=[str(f) for f in files],
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        retina_masks=True,
        stream=True,
        verbose=False,
    ):
        src = Path(r.path)
        lines = []
        if r.masks is not None and len(r.masks):
            for poly, cls in zip(r.masks.xyn, r.boxes.cls.tolist()):
                if poly is None or len(poly) < args.min_points:
                    continue
                coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in poly.tolist())
                lines.append(f"{int(cls)} {coords}")
                per_class[int(cls)] = per_class.get(int(cls), 0) + 1
        if not lines and not args.keep_empty:
            n_empty += 1
            continue
        stem = f"{src.parent.name}_{src.stem}" if src.parent != Path(args.images) else src.stem
        shutil.copy2(src, out_img / f"{stem}{src.suffix.lower()}")
        (out_lbl / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        n_img += 1
        n_inst += len(lines)

    names = model.names
    summary = {
        "teacher": str(args.teacher),
        "source": str(args.images),
        "conf": args.conf,
        "images_written": n_img,
        "images_without_prediction": n_empty,
        "instances": n_inst,
        "instances_per_class": {names.get(k, k): v for k, v in sorted(per_class.items())},
    }
    (Path(args.output) / "PSEUDO_LABEL_SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    data_yaml = Path(args.output) / "data_pseudo.yaml"
    data_yaml.write_text(
        "path: " + str(Path(args.output).resolve()).replace("\\", "/") + "\n"
        f"train: images/{args.split}\nval: images/{args.split}\n"
        f"nc: {len(names)}\nnames:\n" + "".join(f"  {k}: {v}\n" for k, v in names.items()),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[Pseudo-label] dataset yaml: {data_yaml}")


if __name__ == "__main__":
    main()
