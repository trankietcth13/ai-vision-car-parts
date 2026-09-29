"""
Converter for DrBimmer Vehicle Damage Dataset into YOLO Segmentation Format:
  <class_id> <x1> <y1> <x2> <y2> ... (normalized [0, 1])
Classes:
  0: scratch
  1: dent
  2: broken_part
  3: paint_chip
  4: missing_part
  5: flaking
  6: corrosion
  7: cracked
"""

import os
import json
import shutil
from pathlib import Path
from PIL import Image


DAMAGE_MAPPING = {
    "scratch": 0,
    "dent": 1,
    "broken part": 2,
    "broken_part": 2,
    "paint chip": 3,
    "paint_chip": 3,
    "missing part": 4,
    "missing_part": 4,
    "flaking": 5,
    "corrosion": 6,
    "cracked": 7
}


def convert_damage_data(
    ann_dir: str,
    img_dir: str,
    output_dir: str
):
    ann_p = Path(ann_dir)
    img_p = Path(img_dir)
    out_img = Path(output_dir) / "images"
    out_lbl = Path(output_dir) / "labels"

    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    json_files = list(ann_p.glob("*.json"))
    print(f"[Damage Converter] Processing {len(json_files)} annotation files...")

    converted = 0
    total_objs = 0

    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as fp:
                data = json.load(fp)

            img_name = jf.stem
            if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                img_name = jf.name[:-5]

            actual_img = img_p / img_name
            if not actual_img.exists():
                for ext in [".png", ".jpg", ".jpeg"]:
                    cand = img_p / f"{actual_img.stem}{ext}"
                    if cand.exists():
                        actual_img = cand
                        break

            if not actual_img.exists():
                continue

            with Image.open(actual_img) as im:
                img_w, img_h = im.size

            lines = []
            for obj in data.get("objects", []):
                c_title = obj.get("classTitle", "").lower().strip()
                g_type = obj.get("geometryType", "").lower()

                if g_type != "polygon" or c_title not in DAMAGE_MAPPING:
                    continue

                cid = DAMAGE_MAPPING[c_title]
                points = obj.get("points", {}).get("exterior", [])
                if len(points) < 3:
                    continue

                coords = []
                for pt in points:
                    x, y = pt[0], pt[1]
                    norm_x = max(0.0, min(1.0, float(x) / img_w))
                    norm_y = max(0.0, min(1.0, float(y) / img_h))
                    coords.extend([f"{norm_x:.6f}", f"{norm_y:.6f}"])

                lines.append(f"{cid} " + " ".join(coords))
                total_objs += 1

            if lines:
                safe_stem = actual_img.stem.replace(" ", "_")
                dst_img = out_img / f"{safe_stem}{actual_img.suffix}"
                dst_lbl = out_lbl / f"{safe_stem}.txt"

                shutil.copy2(actual_img, dst_img)
                with open(dst_lbl, "w", encoding="utf-8") as fp:
                    fp.write("\n".join(lines) + "\n")
                converted += 1

        except Exception as e:
            print(f"Error {jf.name}: {e}")

    print(f"[Damage Converter Done] Converted {converted} images with {total_objs} damage polygons!")


if __name__ == "__main__":
    convert_damage_data(
        ann_dir=r"./datasets/vehicle_components/drbimmer_parts_damage/Car parts dataset/File1/ann",
        img_dir=r"./datasets/vehicle_components/drbimmer_parts_damage/Car damages dataset/File1/img",
        output_dir=r"./data/damage_dataset"
    )
