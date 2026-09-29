"""
Converter from Supervisely JSON polygon annotations (drbimmer Car damages / parts)
to standard YOLO Segmentation format:
  <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn> (normalized [0, 1])
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image


# Standard 21 Exterior Classes mapping from drbimmer
CLASS_MAPPING = {
    "front-bumper": 0,
    "back-bumper": 1,
    "hood": 2,
    "headlight": 3,
    "tail-light": 4,
    "front-door": 5,
    "back-door": 6,
    "fender": 7,
    "quarter-panel": 8,
    "rocker-panel": 9,
    "mirror": 10,
    "grille": 11,
    "windshield": 12,
    "back-windshield": 13,
    "front-window": 14,
    "back-window": 15,
    "roof": 16,
    "front-wheel": 17,
    "back-wheel": 18,
    "trunk": 19,
    "license-plate": 20
}


def convert_supervisely_to_yolo(
    ann_dir: str,
    img_dir: str,
    output_img_dir: str,
    output_lbl_dir: str
):
    ann_path = Path(ann_dir)
    img_path = Path(img_dir)
    out_img = Path(output_img_dir)
    out_lbl = Path(output_lbl_dir)

    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    json_files = list(ann_path.glob("*.json"))
    print(f"[Converter] Found {len(json_files)} annotation files in {ann_dir}")

    converted_count = 0
    total_objects = 0

    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as fp:
                data = json.load(fp)

            img_size = data.get("size", {})
            img_w = img_size.get("width")
            img_h = img_size.get("height")

            # Determine corresponding image file name
            # e.g. "Car damages 11.png.json" -> "Car damages 11.png"
            img_name = jf.stem  # strips .json, leaving "Car damages 11.png" or "Car damages 11"
            if not (img_name.lower().endswith((".png", ".jpg", ".jpeg"))):
                img_name = jf.name[:-5]  # remove ".json"

            actual_img_file = img_path / img_name
            if not actual_img_file.exists():
                # try alternative extensions
                stem_base = actual_img_file.stem
                for ext in [".png", ".jpg", ".jpeg"]:
                    cand = img_path / f"{stem_base}{ext}"
                    if cand.exists():
                        actual_img_file = cand
                        break

            if not actual_img_file.exists():
                continue

            if not img_w or not img_h:
                with Image.open(actual_img_file) as im:
                    img_w, img_h = im.size

            yolo_lines = []
            for obj in data.get("objects", []):
                c_title = obj.get("classTitle", "").lower().strip()
                g_type = obj.get("geometryType", "").lower()
                
                if g_type != "polygon":
                    continue

                if c_title not in CLASS_MAPPING:
                    # check fuzzy match
                    for k, cid in CLASS_MAPPING.items():
                        if k in c_title or c_title in k:
                            c_title = k
                            break

                if c_title not in CLASS_MAPPING:
                    continue

                class_id = CLASS_MAPPING[c_title]
                points = obj.get("points", {}).get("exterior", [])
                if len(points) < 3:
                    continue

                coords = []
                for pt in points:
                    x, y = pt[0], pt[1]
                    norm_x = max(0.0, min(1.0, float(x) / img_w))
                    norm_y = max(0.0, min(1.0, float(y) / img_h))
                    coords.extend([f"{norm_x:.6f}", f"{norm_y:.6f}"])

                line = f"{class_id} " + " ".join(coords)
                yolo_lines.append(line)
                total_objects += 1

            if yolo_lines:
                # Target clean filename
                safe_stem = actual_img_file.stem.replace(" ", "_")
                dst_img = out_img / f"{safe_stem}{actual_img_file.suffix}"
                dst_lbl = out_lbl / f"{safe_stem}.txt"

                shutil.copy2(actual_img_file, dst_img)
                with open(dst_lbl, "w", encoding="utf-8") as fp:
                    fp.write("\n".join(yolo_lines) + "\n")
                converted_count += 1

        except Exception as e:
            print(f"Error converting {jf.name}: {e}")

    print(f"[Converter Done] Converted {converted_count} images with {total_objects} polygon objects!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Supervisely to YOLO Segmentation")
    parser.add_argument("--ann_dir", type=str, 
                        default=r"./datasets/vehicle_components/drbimmer_parts_damage/Car damages dataset/File1/ann")
    parser.add_argument("--img_dir", type=str, 
                        default=r"./datasets/vehicle_components/drbimmer_parts_damage/Car damages dataset/File1/img")
    parser.add_argument("--output_img", type=str, default=r"./data/exterior_parts_dataset/images")
    parser.add_argument("--output_lbl", type=str, default=r"./data/exterior_parts_dataset/labels")
    args = parser.parse_args()

    convert_supervisely_to_yolo(args.ann_dir, args.img_dir, args.output_img, args.output_lbl)
