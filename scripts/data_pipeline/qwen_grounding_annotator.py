"""Conservative Qwen-VL auto-annotation for the engine-bay inspection dataset.

Outputs three synchronized annotation forms:

* ``labels/<split>/*.txt``: YOLO segmentation labels using a rectangular polygon.
  This is directly consumable by the segmentation models already used in this repo.
* ``labels_bbox/<split>/*.txt``: standard YOLO detection labels.
* ``raw_annotations/<split>/*.json``: class names, normalized/pixel boxes,
  confidence, visual evidence, provenance, and review flags.

The input is split at Request_ID level so near-duplicate photos from the same
inspection never leak across train/validation/test.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import random
import re
import shutil
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageOps


DEFAULT_API_URL = os.environ.get("DGX_VLM_API_URL", "http://dgx-host:8000/v1/chat/completions")
DEFAULT_MODEL = "qwen3-vl-30b"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


PROMPT_TEMPLATE = """You are a conservative senior automotive technician and object-detection annotator.
Inspect the vehicle engine-bay photograph carefully and find every CLEARLY VISIBLE object from this closed vocabulary only:

{class_definitions}

Rules:
1. Label only visible pixels. Never infer a hidden component.
2. Omit uncertain guesses, generic hoses/wires/connectors, body panels, tools other than a diagnostic multimeter, and any object outside the vocabulary.
3. Inspect the entire image including its edges. Multiple instances of a class are allowed.
4. Boxes must tightly enclose the visible part. A partly cropped object is allowed only when its identity is still unmistakable.
5. Do not merge nearby objects. Each ignition coil or battery terminal is a separate instance.
6. Use the exact class names below. Coordinates are XYXY integers normalized from 0 to 1000 relative to the FULL image.

Return a JSON object only, with this exact shape:
{{"detections":[{{"class":"exact_class_name","bbox_2d":[xmin,ymin,xmax,ymax],"confidence":0.0,"visual_evidence":"brief concrete visual evidence"}}]}}
Return {{"detections":[]}} when none of the allowed classes is clearly visible.
"""


CLASS_DEFINITIONS = {
    "battery": "the main rectangular 12V automotive battery body",
    "battery_terminal": "one terminal post/clamp or its red positive protective cover; box only the terminal assembly, never the whole battery",
    "fuse_relay_box": "the electrical fuse/relay enclosure with a removable plastic lid",
    "coolant_reservoir": "the coolant expansion/overflow tank connected to the cooling system; not a brake-fluid tank",
    "radiator_cap": "the pressure cap on the radiator or coolant filler neck",
    "brake_fluid_reservoir": "the small translucent tank mounted on the brake master cylinder, usually near the firewall",
    "washer_fluid_reservoir": "the windshield-washer bottle or its clearly visible filler neck and cap",
    "engine_cover": "a decorative removable engine cover or exposed valve/cam cover; NEVER curved intake-manifold runners",
    "oil_filler_cap": "the round engine-oil fill cap, normally carrying an oil-can symbol",
    "oil_dipstick": "the colored engine-oil dipstick pull handle",
    "air_filter_box": "the large air-cleaner/filter housing",
    "air_intake_duct": "the pre-throttle air hose or tube; NEVER the air-filter box or intake manifold",
    "maf_sensor": "the mass-air-flow sensor module mounted directly on the air-intake duct",
    "throttle_body": "the metal throttle housing between the intake duct and intake manifold",
    "alternator": "the vented belt-driven electrical generator",
    "ignition_coil": "an individual coil-on-plug unit mounted over a spark plug",
    "radiator_hose_upper": "the large upper coolant hose running from radiator to engine",
    "serpentine_belt": "the visible black accessory drive belt",
    "ecu_module": "an engine-control computer module with one or more large multi-pin harness connectors; not an ordinary connector",
    "multimeter_diagnostic_tool": "an external handheld digital meter and its attached probes/leads/clamps used for diagnosis",
    "radiator": "the main cooling radiator core assembly positioned at the front of the vehicle behind the grille",
    "radiator_cooling_fan": "the electric cooling fan and shroud assembly mounted directly behind or on the radiator",
    "radiator_hose_lower": "the lower radiator coolant return hose running from radiator bottom to engine",
    "intake_manifold": "the engine intake manifold runner assembly and plenum distributing air to cylinder ports",
    "abs_modulator_unit": "the ABS hydraulic control unit pump with metal valve body block and multiple rigid steel brake lines",
    "brake_booster": "the large black circular vacuum booster drum mounted on the firewall behind the brake master cylinder",
    "power_steering_reservoir": "the hydraulic power steering fluid reservoir tank, often with cap marked with steering icon",
    "transmission_oil_dipstick": "the transmission fluid dipstick pull handle (typically red or marked ATF/TRANS, distinct from engine oil dipstick)",
    "ac_compressor": "the belt-driven air conditioning AC compressor pump mounted low on the accessory drive",
    "exhaust_manifold_heat_shield": "the metal heat shield covering the exhaust manifold or turbo manifold",
    "turbocharger": "the exhaust gas turbocharger turbine/compressor unit or boost assembly",
    "intercooler_piping": "the boost charge-air piping and silicone couplers connecting turbo, intercooler, and intake",
    "strut_tower_brace": "the front suspension strut tower mount or structural brace bar visible at the sides of the engine bay",
    "hood_latch_mechanism": "the front center hood latch/lock mechanism and safety catch at the radiator support frame",
    "windshield_wiper_motor": "the electric wiper motor and linkage assembly mounted at the cowl/firewall base",
    "oil_filter": "the engine oil filter spin-on canister or top-mount cartridge housing",
}


ALIASES = {
    "car battery": "battery",
    "12v battery": "battery",
    "battery positive terminal": "battery_terminal",
    "positive battery terminal": "battery_terminal",
    "battery negative terminal": "battery_terminal",
    "negative battery terminal": "battery_terminal",
    "fuse box": "fuse_relay_box",
    "relay box": "fuse_relay_box",
    "coolant tank": "coolant_reservoir",
    "expansion tank": "coolant_reservoir",
    "brake reservoir": "brake_fluid_reservoir",
    "washer bottle": "washer_fluid_reservoir",
    "windshield washer": "washer_fluid_reservoir",
    "oil cap": "oil_filler_cap",
    "dipstick": "oil_dipstick",
    "air box": "air_filter_box",
    "air filter": "air_filter_box",
    "intake hose": "air_intake_duct",
    "mass air flow sensor": "maf_sensor",
    "maf": "maf_sensor",
    "coil pack": "ignition_coil",
    "radiator hose": "radiator_hose_upper",
    "upper radiator hose": "radiator_hose_upper",
    "lower radiator hose": "radiator_hose_lower",
    "drive belt": "serpentine_belt",
    "fan belt": "serpentine_belt",
    "accessory belt": "serpentine_belt",
    "ecu": "ecu_module",
    "pcm": "ecu_module",
    "multimeter": "multimeter_diagnostic_tool",
    "voltmeter": "multimeter_diagnostic_tool",
    "cooling fan": "radiator_cooling_fan",
    "radiator fan": "radiator_cooling_fan",
    "intake plenum": "intake_manifold",
    "manifold": "intake_manifold",
    "abs unit": "abs_modulator_unit",
    "abs module": "abs_modulator_unit",
    "abs pump": "abs_modulator_unit",
    "vacuum booster": "brake_booster",
    "power steering fluid": "power_steering_reservoir",
    "power steering tank": "power_steering_reservoir",
    "trans dipstick": "transmission_oil_dipstick",
    "atf dipstick": "transmission_oil_dipstick",
    "a/c compressor": "ac_compressor",
    "heat shield": "exhaust_manifold_heat_shield",
    "turbo": "turbocharger",
    "charge pipe": "intercooler_piping",
    "strut bar": "strut_tower_brace",
    "strut tower": "strut_tower_brace",
    "hood latch": "hood_latch_mechanism",
    "wiper motor": "windshield_wiper_motor",
    "oil filter housing": "oil_filter",
    "oil filter cap": "oil_filter",
}


@dataclass(frozen=True)
class ImageTask:
    source: Path
    request_id: str
    split: str
    output_stem: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-label engine-bay images with Qwen3-VL")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("data/engine_bay_labeled"))
    parser.add_argument("--config", type=Path, default=Path("configs/data_engine_bay.yaml"))
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-side", type=int, default=1400)
    parser.add_argument("--jpeg-quality", type=int, default=88)
    parser.add_argument("--min-confidence", type=float, default=0.80)
    parser.add_argument("--review-confidence", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--copy-images", action="store_true", help="Copy instead of hard-linking images")
    # A timed-out HTTP connection does not necessarily cancel work already queued
    # inside vLLM. Conservative defaults prevent duplicate requests from retries.
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=600)
    return parser.parse_args()


def load_ontology(config_path: Path) -> tuple[dict[int, str], dict[str, int]]:
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    names = {int(k): str(v) for k, v in cfg.get("names", {}).items()}
    if not names:
        raise ValueError(f"No class names found in {config_path}")
    expected = set(CLASS_DEFINITIONS)
    actual = set(names.values())
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"Ontology mismatch. Missing={missing}; extra={extra}")
    return names, {name: class_id for class_id, name in names.items()}


def discover_images(dataset_root: Path) -> list[Path]:
    return sorted(
        path
        for path in dataset_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def split_requests(
    request_ids: list[str], seed: int, val_fraction: float, test_fraction: float
) -> dict[str, str]:
    groups = sorted(set(request_ids))
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_test = max(1, round(len(groups) * test_fraction)) if test_fraction else 0
    n_val = max(1, round(len(groups) * val_fraction)) if val_fraction else 0
    if n_test + n_val >= len(groups):
        raise ValueError("Validation/test fractions leave no request groups for training")
    result: dict[str, str] = {}
    for index, request_id in enumerate(groups):
        if index < n_test:
            split = "test"
        elif index < n_test + n_val:
            split = "val"
        else:
            split = "train"
        result[request_id] = split
    return result


def make_tasks(images: list[Path], dataset_root: Path, split_map: dict[str, str]) -> list[ImageTask]:
    tasks: list[ImageTask] = []
    for source in images:
        relative = source.relative_to(dataset_root)
        request_id = relative.parts[0] if len(relative.parts) > 1 else "ungrouped"
        safe_request = re.sub(r"[^A-Za-z0-9_-]+", "_", request_id)
        digest = hashlib.sha1(relative.as_posix().encode("utf-8")).hexdigest()[:8]
        stem = f"{safe_request}__{source.stem}__{digest}"
        tasks.append(ImageTask(source, request_id, split_map[request_id], stem))
    return tasks


def build_prompt(names: dict[int, str]) -> str:
    definitions = "\n".join(
        f"- {name}: {CLASS_DEFINITIONS[name]}" for _, name in sorted(names.items())
    )
    return PROMPT_TEMPLATE.format(class_definitions=definitions)


def encode_image(path: Path, max_side: int, jpeg_quality: int) -> tuple[str, int, int]:
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        width, height = image.size
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}", width, height


def call_vlm(
    image_uri: str,
    prompt: str,
    api_url: str,
    model: str,
    timeout: int,
    retries: int,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_uri}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": 2500,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    body = json.dumps(payload).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                api_url, data=body, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                response_data = json.loads(response.read().decode("utf-8"))
            return str(response_data["choices"][0]["message"]["content"])
        except (OSError, KeyError, ValueError, urllib.error.URLError) as error:
            last_error = error
            if attempt < retries:
                time.sleep((1, 3, 8, 15)[min(attempt, 3)])
    raise RuntimeError(f"VLM request failed after {retries + 1} attempts: {last_error}")


def extract_json(text: str) -> Any:
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Attempt repair by closing truncated JSON object/array
    last_brace = stripped.rfind("}")
    if last_brace > 0:
        candidates = [
            stripped[: last_brace + 1] + "]}",
            stripped[: last_brace + 1] + "}",
            stripped[: last_brace + 1],
        ]
        for candidate in candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    starts = [pos for pos in (stripped.find("{"), stripped.find("[")) if pos >= 0]
    if not starts:
        raise json.JSONDecodeError("No JSON structure found", stripped, 0)
    start = min(starts)
    end = max(stripped.rfind("}"), stripped.rfind("]"))
    if end <= start:
        raise json.JSONDecodeError("Incomplete JSON boundaries", stripped, start)
    return json.loads(stripped[start : end + 1])


def normalize_class(value: Any, class_to_id: dict[str, int]) -> str | None:
    label = str(value or "").strip().lower().replace("-", "_")
    label = re.sub(r"\s+", "_", label)
    if label in class_to_id:
        return label
    spaced = label.replace("_", " ")
    return ALIASES.get(spaced)


def parse_detections(
    response_text: str,
    class_to_id: dict[str, int],
    width: int,
    height: int,
    min_confidence: float,
    review_confidence: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    parsed = extract_json(response_text)
    if isinstance(parsed, dict):
        raw_detections = parsed.get("detections", [])
    elif isinstance(parsed, list):
        raw_detections = parsed
    else:
        raw_detections = []

    detections: list[dict[str, Any]] = []
    warnings: list[str] = []
    for index, raw in enumerate(raw_detections):
        if not isinstance(raw, dict):
            warnings.append(f"detection_{index}:not_an_object")
            continue
        class_name = normalize_class(raw.get("class", raw.get("label")), class_to_id)
        if class_name is None:
            warnings.append(f"detection_{index}:unknown_class={raw.get('class', raw.get('label'))}")
            continue
        box = raw.get("bbox_2d", raw.get("bbox"))
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            warnings.append(f"detection_{index}:invalid_bbox")
            continue
        try:
            x1, y1, x2, y2 = (float(value) for value in box)
        except (TypeError, ValueError):
            warnings.append(f"detection_{index}:non_numeric_bbox")
            continue
        x1, x2 = sorted((max(0.0, min(1000.0, x1)), max(0.0, min(1000.0, x2))))
        y1, y2 = sorted((max(0.0, min(1000.0, y1)), max(0.0, min(1000.0, y2))))
        if x2 - x1 < 3 or y2 - y1 < 3:
            warnings.append(f"detection_{index}:degenerate_bbox")
            continue
        confidence_missing = "confidence" not in raw
        try:
            confidence = float(raw.get("confidence", 0.50))
        except (TypeError, ValueError):
            confidence = 0.50
            confidence_missing = True
        confidence = max(0.0, min(1.0, confidence))
        nx1, ny1, nx2, ny2 = x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0
        area = (nx2 - nx1) * (ny2 - ny1)
        reasons: list[str] = []
        if confidence_missing:
            reasons.append("confidence_missing")
        if confidence < review_confidence:
            reasons.append("confidence_below_review_threshold")
        if area > 0.45:
            reasons.append("unusually_large_box")
        if sum((nx1 <= 0.002, ny1 <= 0.002, nx2 >= 0.998, ny2 >= 0.998)) >= 2:
            reasons.append("box_touches_multiple_borders")
        evidence = str(raw.get("visual_evidence", raw.get("description", ""))).strip()
        evidence_lower = evidence.lower()
        if class_name == "engine_cover" and "intake manifold" in evidence_lower:
            reasons.append("possible_intake_manifold_confusion")
        if class_name == "air_intake_duct" and area > 0.30:
            reasons.append("intake_duct_box_too_large")
        training_eligible = confidence >= min_confidence and area <= 0.95
        detections.append(
            {
                "class_id": class_to_id[class_name],
                "class_name": class_name,
                "bbox_norm_xyxy": [round(nx1, 6), round(ny1, 6), round(nx2, 6), round(ny2, 6)],
                "bbox_pixel_xyxy": [
                    round(nx1 * width),
                    round(ny1 * height),
                    round(nx2 * width),
                    round(ny2 * height),
                ],
                "confidence": round(confidence, 4),
                "visual_evidence": evidence,
                "training_eligible": training_eligible,
                "review_required": bool(reasons),
                "review_reasons": reasons,
            }
        )
    return detections, warnings


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.{threading.get_ident()}.tmp")
    temp.write_text(content, encoding="utf-8")
    temp.replace(path)


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


def export_labels(task: ImageTask, output: Path, detections: list[dict[str, Any]]) -> None:
    eligible = [item for item in detections if item["training_eligible"]]
    bbox_lines: list[str] = []
    segment_lines: list[str] = []
    for item in eligible:
        x1, y1, x2, y2 = item["bbox_norm_xyxy"]
        xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
        box_w, box_h = x2 - x1, y2 - y1
        class_id = item["class_id"]
        bbox_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {box_w:.6f} {box_h:.6f}")
        segment_lines.append(
            f"{class_id} {x1:.6f} {y1:.6f} {x2:.6f} {y1:.6f} "
            f"{x2:.6f} {y2:.6f} {x1:.6f} {y2:.6f}"
        )
    atomic_write_text(
        output / "labels_bbox" / task.split / f"{task.output_stem}.txt",
        "\n".join(bbox_lines) + ("\n" if bbox_lines else ""),
    )
    atomic_write_text(
        output / "labels" / task.split / f"{task.output_stem}.txt",
        "\n".join(segment_lines) + ("\n" if segment_lines else ""),
    )


def annotation_path(output: Path, task: ImageTask) -> Path:
    return output / "raw_annotations" / task.split / f"{task.output_stem}.json"


def process_task(
    task: ImageTask,
    dataset_root: Path,
    output: Path,
    class_to_id: dict[str, int],
    prompt: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    ann_path = annotation_path(output, task)
    if ann_path.exists() and not args.overwrite:
        try:
            existing = json.loads(ann_path.read_text(encoding="utf-8"))
            if existing.get("status") == "complete":
                detections = existing.get("detections", [])
                export_labels(task, output, detections)
                destination = output / "images" / task.split / f"{task.output_stem}{task.source.suffix.lower()}"
                materialize_image(task.source, destination, args.copy_images)
                return {"status": "cached", "detections": detections, "task": task}
        except (OSError, ValueError, KeyError):
            pass

    started = time.time()
    image_uri, width, height = encode_image(task.source, args.max_side, args.jpeg_quality)
    response_text = call_vlm(
        image_uri, prompt, args.api_url, args.model, args.timeout, args.retries
    )
    detections, parse_warnings = parse_detections(
        response_text,
        class_to_id,
        width,
        height,
        args.min_confidence,
        args.review_confidence,
    )
    record = {
        "schema_version": "1.0",
        "status": "complete",
        "source_image": task.source.relative_to(dataset_root).as_posix(),
        "output_image": f"images/{task.split}/{task.output_stem}{task.source.suffix.lower()}",
        "request_id": task.request_id,
        "split": task.split,
        "image_width": width,
        "image_height": height,
        "model": args.model,
        "api_url": args.api_url,
        "coordinate_convention": "xyxy; normalized values in bbox_norm_xyxy and pixels in bbox_pixel_xyxy",
        "annotation_policy": "auto_qwen_vl; human review required before production training",
        "min_training_confidence": args.min_confidence,
        "detections": detections,
        "parse_warnings": parse_warnings,
        "raw_model_response": response_text,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    atomic_write_text(ann_path, json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    export_labels(task, output, detections)
    destination = output / "images" / task.split / f"{task.output_stem}{task.source.suffix.lower()}"
    materialize_image(task.source, destination, args.copy_images)
    return {"status": "complete", "detections": detections, "task": task}


def write_dataset_files(
    output: Path,
    names: dict[int, str],
    split_map: dict[str, str],
    args: argparse.Namespace,
) -> None:
    manifest = {
        "schema_version": "1.0",
        "source_dataset": str(args.dataset.resolve()),
        "output_dataset": str(output.resolve()),
        "seed": args.seed,
        "split_strategy": "grouped_by_request_id",
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "request_splits": split_map,
        "classes": names,
        "model": args.model,
        "api_url": args.api_url,
        "min_training_confidence": args.min_confidence,
    }
    atomic_write_text(output / "split_manifest.json", json.dumps(manifest, indent=2) + "\n")

    dataset_yaml = {
        "path": str(output.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(names),
        "names": names,
    }
    atomic_write_text(
        output / "data_engine_bay_labeled.yaml",
        "# Auto-generated dataset config. labels/ contains rectangular YOLO segmentation polygons.\n"
        + yaml.safe_dump(dataset_yaml, sort_keys=False, allow_unicode=True),
    )

    readme = """# Auto-labeled engine-bay dataset

This directory was generated by `scripts/data_pipeline/qwen_grounding_annotator.py`.

- `images/`: hard links to source images by default (copies when hard links are unavailable).
- `labels/`: YOLO segmentation syntax using each bounding box as a four-corner polygon; use with the repository's `*-seg` models.
- `labels_bbox/`: standard YOLO detection labels (`class x_center y_center width height`).
- `raw_annotations/`: auditable JSON with class names, pixel/normalized boxes, confidence, descriptions, and review flags.
- `split_manifest.json`: reproducible Request_ID-level split assignment.

These are machine-generated labels. Review the generated QC report before production training, especially detections carrying `review_required: true`.
"""
    atomic_write_text(output / "README.md", readme)


def write_summary(output: Path, results: list[dict[str, Any]], failures: list[dict[str, str]]) -> dict[str, Any]:
    class_counts: Counter[str] = Counter()
    split_images: Counter[str] = Counter()
    review_objects = 0
    eligible_objects = 0
    empty_images = 0
    for result in results:
        task: ImageTask = result["task"]
        split_images[task.split] += 1
        detections = result["detections"]
        if not detections:
            empty_images += 1
        for item in detections:
            class_counts[item["class_name"]] += 1
            review_objects += int(item.get("review_required", False))
            eligible_objects += int(item.get("training_eligible", False))
    summary = {
        "processed_images": len(results),
        "failed_images": len(failures),
        "empty_images": empty_images,
        "split_images": dict(sorted(split_images.items())),
        "total_objects": sum(class_counts.values()),
        "training_eligible_objects": eligible_objects,
        "review_required_objects": review_objects,
        "class_counts": dict(sorted(class_counts.items())),
        "failures": failures,
    }
    atomic_write_text(output / "annotation_summary.json", json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> int:
    args = parse_args()
    args.dataset = args.dataset.resolve()
    args.output = args.output.resolve()
    args.config = args.config.resolve()
    names, class_to_id = load_ontology(args.config)
    images = discover_images(args.dataset)
    if not images:
        raise FileNotFoundError(f"No images found under {args.dataset}")

    request_ids = [path.relative_to(args.dataset).parts[0] for path in images]
    split_map = split_requests(request_ids, args.seed, args.val_fraction, args.test_fraction)
    tasks = make_tasks(images, args.dataset, split_map)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    write_dataset_files(args.output, names, split_map, args)
    prompt = build_prompt(names)

    print(f"[QwenAnnotator] Images: {len(tasks)} | workers: {args.workers}")
    print(f"[QwenAnnotator] Output: {args.output}")
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_map = {
            executor.submit(
                process_task,
                task,
                args.dataset,
                args.output,
                class_to_id,
                prompt,
                args,
            ): task
            for task in tasks
        }
        for completed, future in enumerate(as_completed(future_map), start=1):
            task = future_map[future]
            try:
                result = future.result()
                results.append(result)
                object_count = len(result["detections"])
                status = result["status"]
            except Exception as error:  # keep the batch running and make failures auditable
                failures.append({"image": str(task.source), "error": str(error)})
                object_count = 0
                status = "FAILED"
            if completed % 10 == 0 or completed == len(tasks):
                print(
                    f"[{completed}/{len(tasks)}] {status}: {task.request_id}/{task.source.name} "
                    f"-> {object_count} objects | failures={len(failures)}",
                    flush=True,
                )

    summary = write_summary(args.output, results, failures)
    print(json.dumps(summary, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
