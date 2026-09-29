"""
Dataset Splitter for Vehicle Inspection & Distillation
Supports:
1. Standard YOLO dataset splitting (images/ and labels/)
2. Request-Group splitting: Splits by vehicle inspection folders (Request_ID_XX)
   to ensure zero data leakage between Train, Validation, and Test sets.
"""

import os
import shutil
import random
import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Optional


def split_grouped_dataset(
    dataset_root: str,
    output_dir: str,
    ratios: Tuple[float, float, float] = (0.75, 0.125, 0.125),
    seed: int = 42,
    copy_files: bool = True,
    resize_max: Optional[int] = 1280
):
    """
    Splits by Request_ID folders to keep all angles/shots of the same car in one split.
    """
    random.seed(seed)
    root = Path(dataset_root)
    request_dirs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.lower().startswith("request_")])

    if not request_dirs:
        print(f"[Warning] No Request_ID_* folders found in {dataset_root}. Checking for standard images/...")
        split_yolo_dataset(dataset_root, output_dir, ratios=ratios, seed=seed)
        return

    random.shuffle(request_dirs)
    n_req = len(request_dirs)
    n_train = max(1, int(n_req * ratios[0]))
    n_val = max(1, int(n_req * ratios[1]))

    splits = {
        "train": request_dirs[:n_train],
        "val": request_dirs[n_train:n_train + n_val],
        "test": request_dirs[n_train + n_val:]
    }

    print(f"\n[Group Splitter] Found {n_req} Request Folders.")
    print(f"  - Train Requests ({len(splits['train'])}): {[d.name for d in splits['train']]}")
    print(f"  - Val Requests   ({len(splits['val'])}): {[d.name for d in splits['val']]}")
    print(f"  - Test Requests  ({len(splits['test'])}): {[d.name for d in splits['test']]}")

    out_p = Path(output_dir)
    total_imgs = 0
    split_counts = {"train": 0, "val": 0, "test": 0}

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for split_name, folders in splits.items():
        dst_img_dir = out_p / "images" / split_name
        dst_lbl_dir = out_p / "labels" / split_name
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        for folder in folders:
            images = [f for f in folder.iterdir() if f.suffix.lower() in valid_exts]
            for img_file in images:
                total_imgs += 1
                split_counts[split_name] += 1
                # Format name: Request_ID_01_img_001.jpg
                new_stem = f"{folder.name}_{img_file.stem}"
                dst_img = dst_img_dir / f"{new_stem}{img_file.suffix.lower()}"
                
                if copy_files:
                    if resize_max and resize_max > 0:
                        from PIL import Image
                        try:
                            with Image.open(img_file) as im:
                                im.thumbnail((resize_max, resize_max))
                                im.save(dst_img, quality=90)
                        except Exception:
                            shutil.copy2(img_file, dst_img)
                    else:
                        shutil.copy2(img_file, dst_img)

                # Check if label exists in same folder or parallel label folder
                lbl_candidate = folder / f"{img_file.stem}.txt"
                dst_lbl = dst_lbl_dir / f"{new_stem}.txt"
                if lbl_candidate.exists():
                    shutil.copy2(lbl_candidate, dst_lbl)
                else:
                    dst_lbl.touch(exist_ok=True)

    print(f"\n[Group Splitter Done]: Total {total_imgs} images:")
    for s_name, count in split_counts.items():
        pct = (count / total_imgs * 100) if total_imgs > 0 else 0
        print(f"  -> {s_name}: {count} images ({pct:.1f}%)")


def split_yolo_dataset(
    source_dir: str, 
    dest_dir: str, 
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42
):
    """
    Standard flat YOLO split.
    """
    random.seed(seed)
    train_r, val_r, test_r = ratios
    src_img_dir = os.path.join(source_dir, "images")
    src_lbl_dir = os.path.join(source_dir, "labels")

    if not os.path.exists(src_img_dir):
        print(f"[Error] Image directory does not exist: {src_img_dir}")
        return

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [f for f in os.listdir(src_img_dir) if Path(f).suffix.lower() in valid_exts]
    random.shuffle(image_files)

    n_total = len(image_files)
    n_train = int(n_total * train_r)
    n_val = int(n_total * val_r)

    splits = {
        "train": image_files[:n_train],
        "val": image_files[n_train:n_train + n_val],
        "test": image_files[n_train + n_val:]
    }

    for split_name, files in splits.items():
        dst_img_split = os.path.join(dest_dir, "images", split_name)
        dst_lbl_split = os.path.join(dest_dir, "labels", split_name)
        os.makedirs(dst_img_split, exist_ok=True)
        os.makedirs(dst_lbl_split, exist_ok=True)

        for img_file in files:
            stem = Path(img_file).stem
            shutil.copy2(os.path.join(src_img_dir, img_file), os.path.join(dst_img_split, img_file))
            lbl_file = stem + ".txt"
            src_lbl_path = os.path.join(src_lbl_dir, lbl_file)
            if os.path.exists(src_lbl_path):
                shutil.copy2(src_lbl_path, os.path.join(dst_lbl_split, lbl_file))
            else:
                open(os.path.join(dst_lbl_split, lbl_file), 'w').close()

    print(f"[Dataset Splitter] Done! Total: {n_total} -> Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split Vehicle Inspection Dataset")
    parser.add_argument("--source", type=str, default="./dataset", help="Source dataset path")
    parser.add_argument("--output", type=str, default="./data/car_parts_dataset", help="Target output directory")
    parser.add_argument("--group-by-request", action="store_true", default=True, help="Group split by Request_ID folders")
    parser.add_argument("--resize-max", type=int, default=1280, help="Resize max dimension (0 to keep original 6000x4000)")
    args = parser.parse_args()

    resize_val = args.resize_max if args.resize_max > 0 else None

    if os.path.exists(args.source):
        if args.group_by_request:
            split_grouped_dataset(args.source, args.output, resize_max=resize_val)
        else:
            split_yolo_dataset(args.source, args.output)
    else:
        print(f"[Dataset Splitter Ready] Waiting for dataset in: {args.source}")
