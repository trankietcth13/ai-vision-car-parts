"""Create JPEG previews and a static HTML report for YOLO engine-bay labels."""

from __future__ import annotations

import argparse
import html
import json
import random
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps


PALETTE = [
    "#ef4444", "#f97316", "#eab308", "#22c55e", "#14b8a6",
    "#06b6d4", "#3b82f6", "#6366f1", "#a855f7", "#ec4899",
    "#84cc16", "#10b981", "#0ea5e9", "#8b5cf6", "#d946ef",
    "#f43f5e", "#fb7185", "#f59e0b", "#2dd4bf", "#60a5fa",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an HTML QC report for engine-bay labels")
    parser.add_argument("--dataset", type=Path, default=Path("data/engine_bay_labeled"))
    parser.add_argument("--num-samples", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-preview-side", type=int, default=1200)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_names(dataset_root: Path) -> dict[int, str]:
    config_path = dataset_root / "data_engine_bay_labeled.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    return {int(key): str(value) for key, value in cfg["names"].items()}


def collect_records(dataset_root: Path) -> list[tuple[Path, Path]]:
    records: list[tuple[Path, Path]] = []
    for annotation in sorted((dataset_root / "raw_annotations").rglob("*.json")):
        data = json.loads(annotation.read_text(encoding="utf-8"))
        image_path = dataset_root / data["output_image"]
        if image_path.exists():
            records.append((annotation, image_path))
    return records


def choose_records(records: list[tuple[Path, Path]], count: int, seed: int) -> list[tuple[Path, Path]]:
    rng = random.Random(seed)
    flagged: list[tuple[Path, Path]] = []
    clean: list[tuple[Path, Path]] = []
    for record in records:
        data = json.loads(record[0].read_text(encoding="utf-8"))
        if any(item.get("review_required") for item in data.get("detections", [])):
            flagged.append(record)
        else:
            clean.append(record)
    rng.shuffle(flagged)
    rng.shuffle(clean)
    flagged_quota = min(len(flagged), max(1, round(count * 0.7)))
    selection = flagged[:flagged_quota]
    selection.extend(clean[: max(0, count - len(selection))])
    if len(selection) < count:
        selection.extend(flagged[flagged_quota : flagged_quota + count - len(selection)])
    rng.shuffle(selection)
    return selection[:count]


def render_preview(
    image_path: Path,
    annotation: dict,
    names: dict[int, str],
    output_path: Path,
    max_side: int,
) -> None:
    with Image.open(image_path) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
    original_width, original_height = image.size
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    scale_x = image.width / original_width
    scale_y = image.height / original_height
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for detection in annotation.get("detections", []):
        x1, y1, x2, y2 = detection["bbox_pixel_xyxy"]
        coords = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
        class_id = int(detection["class_id"])
        color = PALETTE[class_id % len(PALETTE)]
        width = 5 if detection.get("review_required") else 3
        draw.rectangle(coords, outline=color, width=width)
        suffix = " REVIEW" if detection.get("review_required") else ""
        label = f"{names[class_id]} {detection.get('confidence', 0):.2f}{suffix}"
        box = draw.textbbox((0, 0), label, font=font)
        text_w, text_h = box[2] - box[0], box[3] - box[1]
        text_x, text_y = coords[0], max(0, coords[1] - text_h - 7)
        draw.rectangle([text_x, text_y, text_x + text_w + 8, text_y + text_h + 6], fill=color)
        draw.text((text_x + 4, text_y + 3), label, fill="white", font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, "JPEG", quality=88, optimize=True)


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset.resolve()
    report_dir = (args.output or dataset_root / "qc_report").resolve()
    preview_dir = report_dir / "previews"
    names = load_names(dataset_root)
    all_records = collect_records(dataset_root)
    selected = choose_records(all_records, min(args.num_samples, len(all_records)), args.seed)
    rows: list[str] = []
    class_counts: Counter[str] = Counter()
    for index, (annotation_path, image_path) in enumerate(selected, start=1):
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        preview_name = f"{index:04d}_{image_path.stem}.jpg"
        render_preview(
            image_path, annotation, names, preview_dir / preview_name, args.max_preview_side
        )
        detections = annotation.get("detections", [])
        for item in detections:
            class_counts[item["class_name"]] += 1
        object_lines = "".join(
            "<li>"
            + html.escape(item["class_name"])
            + f" — conf {item.get('confidence', 0):.2f}"
            + (" — <b>REVIEW</b>: " + html.escape(", ".join(item.get("review_reasons", []))) if item.get("review_required") else "")
            + (" — " + html.escape(item.get("visual_evidence", "")) if item.get("visual_evidence") else "")
            + "</li>"
            for item in detections
        ) or "<li>No target-class object detected</li>"
        rows.append(
            f"<article><h2>{html.escape(annotation['source_image'])}</h2>"
            f"<img loading='lazy' src='previews/{html.escape(preview_name)}'>"
            f"<ul>{object_lines}</ul></article>"
        )

    legend = "".join(
        f"<span style='border-color:{PALETTE[class_id % len(PALETTE)]}'>{class_id}: {html.escape(name)}</span>"
        for class_id, name in sorted(names.items())
    )
    report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Engine-bay annotation QC</title><style>
body{{font-family:system-ui,sans-serif;margin:24px;background:#111827;color:#e5e7eb}}
h1{{margin-bottom:8px}} .meta{{color:#9ca3af;margin-bottom:20px}}
.legend{{display:flex;flex-wrap:wrap;gap:8px;margin:16px 0 24px}}
.legend span{{border-left:8px solid;padding:5px 9px;background:#1f2937;border-radius:4px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:18px}}
article{{background:#1f2937;border-radius:10px;padding:14px;box-shadow:0 2px 10px #0008}}
article h2{{font-size:14px;word-break:break-all}} img{{width:100%;height:auto;border-radius:6px}}
li{{margin:5px 0;line-height:1.35}} b{{color:#fbbf24}}
</style></head><body><h1>Engine-bay annotation QC</h1>
<div class="meta">{len(selected)} sampled images from {len(all_records)} annotated images. Thick boxes and REVIEW text require human attention.</div>
<div class="legend">{legend}</div><div class="grid">{''.join(rows)}</div></body></html>
"""
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "index.html").write_text(report, encoding="utf-8")
    stats = {"sampled_images": len(selected), "available_images": len(all_records), "sample_class_counts": dict(sorted(class_counts.items()))}
    (report_dir / "summary.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"QC report: {report_dir / 'index.html'}")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
