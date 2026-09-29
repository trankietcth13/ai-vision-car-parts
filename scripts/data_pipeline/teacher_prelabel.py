"""P4: teacher-first pre-labels for new images, in the layout build_review_packets.py already consumes.

The SoM pilot showed the teacher labels unseen vehicles better than the VLMs (bake-off F1 0.590 vs 0.473 / 0.421),
so new images start from the teacher system (full image + 2x2 tiles = recall mode) instead of Qwen/DeepSeek:

    <out>/images/<split>/<stem>.jpg        links / copies of the source images
    <out>/labels/<split>/<stem>.txt        YOLO-seg polygons (36-class annotation ids) of the confident boxes
    <out>/refine_meta/<split>/<stem>.json  instances: confident = "teacher", low score = "dropped_for_review"
                                           (shown to the reviewer as D<id> candidates, so recall is not lost)
    <out>/data_engine_bay_seg.yaml

then:  build_review_packets.py --seg <out> --out <review dir> --splits <split>  -> expert review skill -> apply_review.py

    python scripts/data_pipeline/teacher_prelabel.py --weights runs/segment/p5_reg/weights/avg5.pt \
        --images <folder or files> --out data/engine_bay_prelabel_new --split new
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.teacher_system import TeacherSystem, load_image  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
# v1 training names that are not literally in the 36-class annotation ontology
TO_ANNOTATION = {"radiator_hose": "radiator_hose_upper"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, default=ROOT / "runs" / "segment" / "p5_reg" / "weights" / "avg5.pt")
    ap.add_argument("--images", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", default="new")
    ap.add_argument("--candidate-conf", type=float, default=0.15, help="lowest score shown to the reviewer")
    ap.add_argument("--label-conf", type=float, default=0.35, help="score from which a box becomes a label")
    ap.add_argument("--no-tiles", action="store_true")
    ap.add_argument("--ann-config", type=Path, default=ROOT / "configs" / "data_engine_bay.yaml")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    ann = {int(k): v for k, v in yaml.safe_load(args.ann_config.read_text(encoding="utf-8"))["names"].items()}
    ann_id = {v: k for k, v in ann.items()}
    files = []
    for p in args.images:
        files += sorted(q for q in p.iterdir() if q.suffix.lower() in IMG_EXT) if p.is_dir() else [p]
    ts = TeacherSystem(args.weights, conf=args.candidate_conf, device=args.device)
    out = args.out
    for sub in ("images", "labels", "refine_meta"):
        (out / sub / args.split).mkdir(parents=True, exist_ok=True)
    summary = {"images": 0, "labels": 0, "candidates": 0}
    for k, path in enumerate(files):
        stem = path.stem
        dst = out / "images" / args.split / f"{stem}{path.suffix.lower()}"
        link_or_copy(path, dst)
        dets = ts.detect(load_image(path), use_tiles=not args.no_tiles)
        dets.sort(key=lambda d: -d["score"])
        instances, lines = [], []
        for d in dets:
            name = TO_ANNOTATION.get(d["cls"], d["cls"])
            if name not in ann_id:
                continue
            confident = d["score"] >= args.label_conf and d["poly"] is not None and len(d["poly"]) >= 3
            instances.append({"class_id": ann_id[name], "class_name": name, "confidence": round(d["score"], 3),
                              "source_box_norm": [round(v, 4) for v in d["box"]],
                              "method": "teacher" if confident else "dropped_for_review"})
            if confident:
                pts = " ".join(f"{x:.5f} {y:.5f}" for x, y in d["poly"][:: max(1, len(d["poly"]) // 80)])
                lines.append(f"{ann_id[name]} {pts}")
        (out / "labels" / args.split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        meta = {"source": str(path), "image": f"images/{args.split}/{dst.name}", "teacher": str(args.weights),
                "tiles": not args.no_tiles, "instances": instances}
        (out / "refine_meta" / args.split / f"{stem}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        summary["images"] += 1
        summary["labels"] += len(lines)
        summary["candidates"] += len(instances) - len(lines)
        print(f"[{k + 1}/{len(files)}] {stem}: {len(lines)} labels, {len(instances) - len(lines)} low-score candidates", flush=True)
    (out / "data_engine_bay_seg.yaml").write_text(yaml.safe_dump(
        {"path": str(out.resolve()), "names": ann, "nc": len(ann), args.split: f"images/{args.split}"},
        sort_keys=False, allow_unicode=True), encoding="utf-8")
    (out / "PRELABEL_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
