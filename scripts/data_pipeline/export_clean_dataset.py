"""
Export a clean YOLO bounding-box detection dataset from auto-labeled annotations.

Filters raw annotations to include ONLY objects that are:
- `training_eligible: true`
- `review_required: false`

This isolates the verified, safe annotations (e.g. 3,000 components) for initial
baseline object detection training. Ultralytics YOLO dataset conventions are strictly
followed: `images/<split>/*.jpg` and `labels/<split>/*.txt`.

Usage:
    python scripts/data_pipeline/export_clean_dataset.py
    python scripts/data_pipeline/export_clean_dataset.py --source data/engine_bay_labeled_v2 --output data/engine_bay_clean
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export clean YOLO detection dataset.")
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / "data" / "engine_bay_labeled_v2",
        help="Path to labeled dataset containing raw_annotations and images",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "engine_bay_clean",
        help="Output directory for clean dataset",
    )
    parser.add_argument(
        "--config-in",
        type=Path,
        default=PROJECT_ROOT / "configs" / "data_engine_bay.yaml",
        help="Input dataset config containing class ontology",
    )
    parser.add_argument(
        "--config-out",
        type=Path,
        default=PROJECT_ROOT / "configs" / "data_engine_bay_clean.yaml",
        help="Output dataset YAML config path",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Always copy images instead of creating hardlinks",
    )
    parser.add_argument(
        "--exclude-empty",
        action="store_true",
        help="Exclude background images that contain 0 clean components",
    )
    return parser.parse_args()


def materialize_image(source: Path, destination: Path, copy_images: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    if copy_images:
        shutil.copy2(source, destination)
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def export_clean_dataset(
    source_dir: Path,
    output_dir: Path,
    config_in: Path,
    config_out: Path,
    copy_images: bool = False,
    exclude_empty: bool = False,
) -> dict[str, Any]:
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()

    raw_dir = source_dir / "raw_annotations"
    images_dir = source_dir / "images"

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw annotations directory not found: {raw_dir}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    # Load ontology from config_in
    with open(config_in, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names", {})
    nc = cfg.get("nc", len(names))

    splits = ["train", "val", "test"]
    overall_class_counts: Counter[str] = Counter()
    split_stats: dict[str, dict[str, int]] = {}
    total_images_copied = 0
    total_clean_objects = 0
    total_empty_images = 0

    print(f"[Export Clean Dataset] Source: {source_dir}")
    print(f"[Export Clean Dataset] Output: {output_dir}")

    for split in splits:
        split_raw = raw_dir / split
        split_img_src = images_dir / split
        if not split_raw.exists():
            continue

        split_img_dst = output_dir / "images" / split
        split_lbl_dst = output_dir / "labels" / split
        split_img_dst.mkdir(parents=True, exist_ok=True)
        split_lbl_dst.mkdir(parents=True, exist_ok=True)

        json_files = sorted(split_raw.glob("*.json"))
        split_clean_objs = 0
        split_empty_imgs = 0
        split_processed = 0

        for jf in json_files:
            stem = jf.stem
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)

            detections = data.get("detections", [])
            clean_dets = [
                d
                for d in detections
                if d.get("training_eligible") and not d.get("review_required")
            ]

            if not clean_dets and exclude_empty:
                continue

            # Find matching image file
            img_candidates = list(split_img_src.glob(f"{stem}.*"))
            if not img_candidates:
                # Try finding based on output_image or source_image in JSON
                out_img_rel = data.get("output_image")
                if out_img_rel and (source_dir / out_img_rel).exists():
                    img_path = source_dir / out_img_rel
                else:
                    print(f"Warning: Image file not found for {jf.name}, skipping.")
                    continue
            else:
                img_path = img_candidates[0]

            # Copy or link image
            dst_img_path = split_img_dst / img_path.name
            materialize_image(img_path, dst_img_path, copy_images)
            split_processed += 1
            total_images_copied += 1

            # Format YOLO bounding box lines: class_id xc yc w h
            bbox_lines = []
            for d in clean_dets:
                cid = d["class_id"]
                cname = d.get("class_name", str(cid))
                overall_class_counts[cname] += 1
                x1, y1, x2, y2 = d["bbox_norm_xyxy"]

                # Clamp values to [0, 1]
                x1 = max(0.0, min(1.0, float(x1)))
                y1 = max(0.0, min(1.0, float(y1)))
                x2 = max(0.0, min(1.0, float(x2)))
                y2 = max(0.0, min(1.0, float(y2)))

                xc = (x1 + x2) / 2.0
                yc = (y1 + y2) / 2.0
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)

                bbox_lines.append(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

            # Write label file (empty string if 0 objects, acts as negative background sample)
            dst_lbl_path = split_lbl_dst / f"{stem}.txt"
            dst_lbl_path.write_text("\n".join(bbox_lines) + ("\n" if bbox_lines else ""), encoding="utf-8")

            split_clean_objs += len(clean_dets)
            if not clean_dets:
                split_empty_imgs += 1

        split_stats[split] = {
            "images": split_processed,
            "clean_objects": split_clean_objs,
            "empty_images": split_empty_imgs,
        }
        total_clean_objects += split_clean_objs
        total_empty_images += split_empty_imgs

        print(
            f"  - Split '{split}': {split_processed} images | "
            f"{split_clean_objs} clean objects | {split_empty_imgs} background images"
        )

    # Write summary
    summary_data = {
        "source": str(source_dir),
        "total_images": total_images_copied,
        "total_clean_objects": total_clean_objects,
        "total_empty_images": total_empty_images,
        "split_stats": split_stats,
        "class_counts": dict(sorted(overall_class_counts.items(), key=lambda x: x[1], reverse=True)),
    }
    summary_file = output_dir / "clean_summary.json"
    summary_file.write_text(json.dumps(summary_data, indent=2, ensure_ascii=False), encoding="utf-8")

    # Generate YOLO configuration file
    # We use a relative path from the config location to output_dir
    try:
        rel_dataset_path = os.path.relpath(output_dir, config_out.parent).replace("\\", "/")
    except ValueError:
        rel_dataset_path = str(output_dir).replace("\\", "/")

    clean_cfg = {
        "path": rel_dataset_path,
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": nc,
        "names": names,
    }

    config_out.parent.mkdir(parents=True, exist_ok=True)
    with open(config_out, "w", encoding="utf-8") as f:
        yaml.dump(clean_cfg, f, sort_keys=False, allow_unicode=True)

    print(f"\n[Export Clean Dataset] Successfully exported {total_clean_objects} clean objects across {total_images_copied} images.")
    print(f"[Export Clean Dataset] Summary written to: {summary_file}")
    print(f"[Export Clean Dataset] YOLO Config written to: {config_out}")

    return summary_data


if __name__ == "__main__":
    args = parse_args()
    export_clean_dataset(
        source_dir=args.source,
        output_dir=args.output,
        config_in=args.config_in,
        config_out=args.config_out,
        copy_images=args.copy_images,
        exclude_empty=args.exclude_empty,
    )
