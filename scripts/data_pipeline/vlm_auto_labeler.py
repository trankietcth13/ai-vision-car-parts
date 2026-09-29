"""
High-Throughput Auto-Labeler using DGX Qwen3-VL-30B Visual Grounding
Automatically detects vehicle engine bay components and generates standard YOLO segmentation polygon labels:
  <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn> (normalized [0, 1])
"""

import os
import re
import io
import json
import base64
import argparse
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import yaml


DGX_API_URL = os.environ.get("DGX_VLM_API_URL", "http://dgx-host:8000/v1/chat/completions")
MODEL_ID = "qwen3-vl-30b"


ALIAS_MAP = {
    "oil filler cap": "oil_filler_cap",
    "oil cap": "oil_filler_cap",
    "oil dipstick": "oil_dipstick",
    "dipstick": "oil_dipstick",
    "battery": "battery",
    "car battery": "battery",
    "battery terminal": "battery_terminal",
    "terminal": "battery_terminal",
    "fuse box": "fuse_relay_box",
    "fuse and relay box": "fuse_relay_box",
    "relay box": "fuse_relay_box",
    "coolant reservoir": "coolant_reservoir",
    "coolant tank": "coolant_reservoir",
    "expansion tank": "coolant_reservoir",
    "radiator cap": "radiator_cap",
    "brake fluid reservoir": "brake_fluid_reservoir",
    "washer fluid reservoir": "washer_fluid_reservoir",
    "washer bottle": "washer_fluid_reservoir",
    "engine cover": "engine_cover",
    "valve cover": "engine_cover",
    "air filter box": "air_filter_box",
    "air box": "air_filter_box",
    "air cleaner": "air_filter_box",
    "air intake duct": "air_intake_duct",
    "intake hose": "air_intake_duct",
    "intake manifold": "air_intake_duct",
    "maf sensor": "maf_sensor",
    "throttle body": "throttle_body",
    "alternator": "alternator",
    "ignition coil": "ignition_coil",
    "coil pack": "ignition_coil",
    "radiator hose": "radiator_hose_upper",
    "serpentine belt": "serpentine_belt",
    "ecu": "ecu_module",
    "multimeter": "multimeter_diagnostic_tool",
    "voltmeter": "multimeter_diagnostic_tool"
}


class VLMAutoLabeler:
    def __init__(self, config_path: str, api_url: str = DGX_API_URL, model_id: str = MODEL_ID):
        self.api_url = api_url
        self.model_id = model_id
        
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            
        self.names = cfg.get("names", {})
        self.name_to_id = {v.lower().replace("-", "_"): int(k) for k, v in self.names.items()}
        print(f"[VLMAutoLabeler] Initialized with {len(self.name_to_id)} classes from {config_path}")

    def _parse_grounding_text(self, text: str) -> List[Tuple[int, List[float]]]:
        results = []
        pattern = r'([a-zA-Z\s_]+):\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
        matches = re.findall(pattern, text)
        
        for label_str, y1_s, x1_s, y2_s, x2_s in matches:
            label_norm = label_str.strip().lower()
            matched_class = None
            for alias, target in ALIAS_MAP.items():
                if alias in label_norm:
                    matched_class = target
                    break
            if not matched_class:
                for cname in self.name_to_id:
                    if cname in label_norm or label_norm in cname:
                        matched_class = cname
                        break
                        
            if matched_class and matched_class in self.name_to_id:
                cid = self.name_to_id[matched_class]
                y1, x1, y2, x2 = float(y1_s)/1000.0, float(x1_s)/1000.0, float(y2_s)/1000.0, float(x2_s)/1000.0
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                # Minimum area check to avoid single-pixel noise
                if (x2 - x1) < 0.01 or (y2 - y1) < 0.01:
                    continue

                # 4-point polygon box
                poly = [
                    round(x1, 6), round(y1, 6),
                    round(x2, 6), round(y1, 6),
                    round(x2, 6), round(y2, 6),
                    round(x1, 6), round(y2, 6)
                ]
                results.append((cid, poly))

        return results

    def label_single_image(self, img_path: Path, output_lbl_path: Path, overwrite: bool = False) -> int:
        if output_lbl_path.exists() and output_lbl_path.stat().st_size > 0 and not overwrite:
            return 0  # Skip already labeled

        try:
            with Image.open(img_path) as im:
                im.thumbnail((800, 800))
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=80)
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                data_uri = f"data:image/jpeg;base64,{b64}"

            prompt = (
                "Locate all visible vehicle engine bay components in this photograph: "
                "battery, battery terminal, fuse box, coolant reservoir, radiator cap, "
                "brake fluid reservoir, engine cover, oil filler cap, oil dipstick, air filter box, "
                "air intake duct, throttle body, alternator, ignition coil, multimeter tool. "
                "For each part detected, output on a separate line exactly: label: [ymin, xmin, ymax, xmax] (normalized 0 to 1000)."
            )

            payload = {
                "model": self.model_id,
                "messages": [
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": prompt}
                    ]}
                ],
                "max_tokens": 512,
                "temperature": 0.1
            }

            req = urllib.request.Request(
                self.api_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=40) as resp:
                data = json.loads(resp.read().decode())
                out_text = data["choices"][0]["message"]["content"]

            parsed = self._parse_grounding_text(out_text)

            output_lbl_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_lbl_path, "w", encoding="utf-8") as fp:
                for cid, poly in parsed:
                    fp.write(f"{cid} " + " ".join(map(str, poly)) + "\n")

            return len(parsed)

        except Exception as e:
            # write empty file on failure to mark processed
            output_lbl_path.parent.mkdir(parents=True, exist_ok=True)
            output_lbl_path.touch(exist_ok=True)
            return 0

    def label_dataset(self, dataset_root: str, max_workers: int = 4, max_images: Optional[int] = None):
        root = Path(dataset_root)
        splits = ["train", "val", "test"]
        
        all_tasks = []
        for s in splits:
            img_dir = root / "images" / s
            lbl_dir = root / "labels" / s
            if not img_dir.exists():
                continue
            for img_p in sorted(img_dir.glob("*.[jJ][pP]*[gG]")):
                lbl_p = lbl_dir / f"{img_p.stem}.txt"
                all_tasks.append((img_p, lbl_p, s))

        if max_images:
            all_tasks = all_tasks[:max_images]

        total = len(all_tasks)
        print(f"\n[VLMAutoLabeler] Starting auto-labeling for {total} images using {max_workers} worker threads...")

        completed = 0
        total_objects = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_img = {
                executor.submit(self.label_single_image, img_p, lbl_p): (img_p, s)
                for img_p, lbl_p, s in all_tasks
            }

            for future in as_completed(future_to_img):
                img_p, split_name = future_to_img[future]
                completed += 1
                try:
                    num_objs = future.result()
                    total_objects += num_objs
                    if completed % 20 == 0 or completed == total:
                        print(f"  [{completed}/{total}] ({split_name}) Labeled: {img_p.name} -> {num_objs} objects (Total objects so far: {total_objects})")
                except Exception as e:
                    print(f"  [{completed}/{total}] Error processing {img_p.name}: {e}")

        print(f"\n[VLMAutoLabeler Complete] Successfully processed {completed} images with {total_objects} total objects generated!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DGX VLM Auto-Labeler for Engine Bay")
    parser.add_argument("--dataset_root", type=str, default="./data/car_parts_dataset", help="Dataset directory")
    parser.add_argument("--config", type=str, default="./configs/data_engine_bay.yaml", help="Ontology YAML config")
    parser.add_argument("--workers", type=int, default=4, help="Parallel worker threads")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images for testing")
    args = parser.parse_args()

    labeler = VLMAutoLabeler(config_path=args.config)
    labeler.label_dataset(args.dataset_root, max_workers=args.workers, max_images=args.limit)
