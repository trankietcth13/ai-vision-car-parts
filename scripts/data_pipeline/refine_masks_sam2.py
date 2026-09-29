"""
Box -> mask refinement for the engine bay dataset with SAM 2.1 (runs locally on GPU, no DGX calls).

Reads the Qwen3-VL box annotations produced by `qwen_grounding_annotator.py`
(`<src>/raw_annotations/<split>/*.json`, READ-ONLY) and writes a separate
YOLO-segmentation dataset whose polygons follow the real component outline:

    <out>/images/<split>/*          hard links (or copies) of the source images
    <out>/labels/<split>/*.txt      YOLO-seg polygons from SAM masks
    <out>/refine_meta/<split>/*.json per-instance quality info (mask/box IoU, fallback flags)
    <out>/qa/<split>/*.jpg          optional overlays for visual review (--qa)
    <out>/data_engine_bay_seg.yaml
    <out>/REFINE_SUMMARY.json

The script is incremental: an image is (re)processed only when its source JSON is
newer than the refined label, so it can be re-run while the box annotator is still
producing annotations.

Usage:
    python scripts/data_pipeline/refine_masks_sam2.py --src data/engine_bay_labeled --out data/engine_bay_seg --qa
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def parse_args():
    ap = argparse.ArgumentParser(description="Refine box annotations into SAM2 instance masks")
    ap.add_argument("--src", type=Path, default=PROJECT_ROOT / "data" / "engine_bay_labeled")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "engine_bay_seg")
    ap.add_argument("--sam", default="sam2.1_b.pt", help="Ultralytics SAM checkpoint (sam2.1_t/s/b/l.pt)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--max-side", type=int, default=1536, help="Resize images before SAM (speed/memory)")
    ap.add_argument("--only-eligible", action="store_true", help="Keep only detections with training_eligible=true")
    ap.add_argument("--min-confidence", type=float, default=0.0, help="Extra confidence filter on top of the source")
    ap.add_argument("--min-fill", type=float, default=0.15, help="Mask area / box area below this -> box fallback")
    ap.add_argument("--min-box-iou", type=float, default=0.50, help="IoU(mask bbox, prompt box) below this -> box fallback")
    ap.add_argument("--box-pad", type=float, default=0.10, help="Clip masks to the prompt box enlarged by this fraction")
    ap.add_argument("--fallback", choices=["drop", "box"], default="drop",
                    help="When SAM disagrees with a box: 'drop' it (listed in REVIEW_QUEUE) or keep the rectangle")
    ap.add_argument("--epsilon", type=float, default=0.002, help="Polygon simplification, fraction of image diagonal")
    ap.add_argument("--min-area-px", type=int, default=64, help="Ignore mask fragments smaller than this (resized px)")
    ap.add_argument("--copy-images", action="store_true", help="Copy images instead of hard-linking")
    ap.add_argument("--qa", action="store_true", help="Write overlay images for visual review")
    ap.add_argument("--force", action="store_true", help="Reprocess every image")
    ap.add_argument("--limit", type=int, default=None)
    return ap.parse_args()


# ----------------------------------------------------------------------------- geometry helpers
def box_iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def mask_to_polygon(mask: np.ndarray, epsilon_px: float, min_area_px: int):
    """Largest external contour of a binary mask as an (N, 2) pixel array, or None."""
    mask_u8 = mask.astype(np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area_px]
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, epsilon_px, True).reshape(-1, 2)
    return c if len(c) >= 3 else None


def box_polygon(b):
    x1, y1, x2, y2 = b
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def materialize(src: Path, dst: Path, copy: bool):
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not copy:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def atomic_write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- main
def main():
    args = parse_args()
    src, out = args.src.resolve(), args.out.resolve()
    ann_root = src / "raw_annotations"
    if not ann_root.exists():
        raise SystemExit(f"No annotations found under {ann_root}")
    if out == src:
        raise SystemExit("--out must differ from --src (the source dataset is read-only)")

    jsons = sorted(ann_root.glob("*/*.json"))
    todo = []
    for jp in jsons:
        split = jp.parent.name
        lbl = out / "labels" / split / f"{jp.stem}.txt"
        if args.force or not lbl.exists() or lbl.stat().st_mtime < jp.stat().st_mtime:
            todo.append(jp)
    if args.limit:
        todo = todo[: args.limit]
    print(f"[SAM2] {len(jsons)} source annotations, {len(todo)} to (re)process -> {out}")
    if not todo:
        write_dataset_yaml(src, out)
        return

    from ultralytics import SAM

    sam = SAM(args.sam)
    colors = np.random.default_rng(0).integers(60, 255, size=(64, 3))

    review = []
    stats = {"images": 0, "skipped": 0, "instances": 0, "sam": 0, "fallback_box": 0, "dropped": 0}
    t0 = time.time()
    for i, jp in enumerate(todo, 1):
        try:
            ann = json.loads(jp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            stats["skipped"] += 1
            continue  # file may be mid-write by the annotator; picked up on the next run
        if ann.get("status") != "complete":
            stats["skipped"] += 1
            continue
        split = ann.get("split", jp.parent.name)
        img_src = src / ann["output_image"]
        if not img_src.exists():
            stats["skipped"] += 1
            continue

        dets = [
            d for d in ann.get("detections", [])
            if (not args.only_eligible or d.get("training_eligible", False))
            and float(d.get("confidence", 1.0)) >= args.min_confidence
        ]

        img = cv2.imread(str(img_src))
        if img is None:
            stats["skipped"] += 1
            continue
        h0, w0 = img.shape[:2]
        scale = min(1.0, args.max_side / max(h0, w0))
        if scale < 1.0:
            img = cv2.resize(img, (round(w0 * scale), round(h0 * scale)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        eps = args.epsilon * float(np.hypot(w, h))

        boxes = [[d["bbox_norm_xyxy"][0] * w, d["bbox_norm_xyxy"][1] * h,
                  d["bbox_norm_xyxy"][2] * w, d["bbox_norm_xyxy"][3] * h] for d in dets]
        masks = []
        if boxes:
            res = sam.predict(img, bboxes=boxes, device=args.device, verbose=False)[0]
            masks = res.masks.data.cpu().numpy() > 0.5 if res.masks is not None else []

        lines, meta = [], []
        overlay = img.copy() if args.qa else None
        for k, d in enumerate(dets):
            b = boxes[k]
            m = masks[k] if k < len(masks) else None
            info = {"class_id": d["class_id"], "class_name": d["class_name"], "confidence": d.get("confidence"),
                    "source_box_norm": d["bbox_norm_xyxy"], "method": "sam2"}
            poly = None
            if m is not None and m.any():
                bw, bh = b[2] - b[0], b[3] - b[1]
                cx1, cy1 = max(0, int(b[0] - args.box_pad * bw)), max(0, int(b[1] - args.box_pad * bh))
                cx2, cy2 = min(w, int(np.ceil(b[2] + args.box_pad * bw))), min(h, int(np.ceil(b[3] + args.box_pad * bh)))
                clipped = np.zeros_like(m)
                clipped[cy1:cy2, cx1:cx2] = m[cy1:cy2, cx1:cx2]
                m = clipped
            if m is not None and m.any():
                ys, xs = np.nonzero(m)
                mbox = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
                box_area = max((b[2] - b[0]) * (b[3] - b[1]), 1.0)
                # only the part of the mask inside a slightly padded prompt box counts toward fill
                fill = float(m.sum()) / box_area
                iou = box_iou(mbox, b)
                info.update(mask_fill=round(fill, 3), mask_box_iou=round(iou, 3))
                if fill >= args.min_fill and iou >= args.min_box_iou:
                    poly = mask_to_polygon(m, eps, args.min_area_px)
            if poly is None and args.fallback == "drop":
                info["method"] = "dropped_for_review"
                meta.append(info)
                stats["dropped"] += 1
                review.append({"image": f"images/{split}/{img_src.name}", "class_name": d["class_name"],
                               "box_norm": d["bbox_norm_xyxy"], "confidence": d.get("confidence"),
                               "mask_fill": info.get("mask_fill"), "mask_box_iou": info.get("mask_box_iou")})
                if overlay is not None:
                    cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 2)
                    cv2.putText(img, f"DROP {d['class_name']}", (int(b[0]) + 3, int(b[1]) + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                continue
            if poly is None:
                info["method"] = "box_fallback"
                poly = box_polygon(b)
                stats["fallback_box"] += 1
            else:
                stats["sam"] += 1
            pn = poly.astype(np.float64) / np.array([w, h])
            pn = np.clip(pn, 0.0, 1.0)
            lines.append(f"{d['class_id']} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in pn))
            info["n_points"] = int(len(poly))
            meta.append(info)
            if overlay is not None:
                col = tuple(int(c) for c in colors[d["class_id"] % 64])
                cv2.fillPoly(overlay, [poly.astype(np.int32)], col)
                cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, 2)
                tag = f"{d['class_name']}{'*' if info['method'] == 'box_fallback' else ''}"
                cv2.putText(img, tag, (int(b[0]) + 3, int(b[1]) + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                cv2.putText(img, tag, (int(b[0]) + 3, int(b[1]) + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 1)

        stem = jp.stem
        dst_img = out / "images" / split / img_src.name
        materialize(img_src, dst_img, args.copy_images)
        atomic_write(out / "refine_meta" / split / f"{stem}.json", json.dumps(
            {"source_annotation": str(jp.relative_to(src)).replace("\\", "/"), "image": f"images/{split}/{img_src.name}",
             "sam_checkpoint": args.sam, "instances": meta}, indent=2, ensure_ascii=False))
        # label written last: its mtime marks the image as done
        atomic_write(out / "labels" / split / f"{stem}.txt", "\n".join(lines) + ("\n" if lines else ""))
        if overlay is not None:
            qa = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)
            (out / "qa" / split).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / "qa" / split / f"{stem}.jpg"), qa, [cv2.IMWRITE_JPEG_QUALITY, 85])

        stats["images"] += 1
        stats["instances"] += len(lines)
        if i % 25 == 0 or i == len(todo):
            print(f"[SAM2] {i}/{len(todo)} images | {stats['sam']} sam masks, {stats['fallback_box']} box fallbacks "
                  f"| {time.time() - t0:.0f}s")

    write_dataset_yaml(src, out)
    rq = out / "REVIEW_QUEUE.jsonl"
    old = {}
    if rq.exists() and not args.force:
        for line in rq.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            old[(r["image"], r["class_name"], tuple(r["box_norm"]))] = r
    for r in review:
        old[(r["image"], r["class_name"], tuple(r["box_norm"]))] = r
    atomic_write(rq, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in old.values()))
    summary = summarize(out)
    summary["last_run"] = stats
    atomic_write(out / "REFINE_SUMMARY.json", json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def write_dataset_yaml(src: Path, out: Path):
    import yaml

    src_yaml = next(iter(sorted(src.glob("*.yaml"))), None)
    names = {}
    if src_yaml:
        names = (yaml.safe_load(src_yaml.read_text(encoding="utf-8")) or {}).get("names", {})
    if not names:
        names = (yaml.safe_load((PROJECT_ROOT / "configs" / "data_engine_bay.yaml").read_text(encoding="utf-8")) or {})["names"]
    cfg = {"path": str(out).replace("\\", "/"), "train": "images/train", "val": "images/val", "test": "images/test",
           "nc": len(names), "names": names}
    atomic_write(out / "data_engine_bay_seg.yaml", yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))


def summarize(out: Path) -> dict:
    per_split, per_class, methods = {}, {}, {"sam2": 0, "box_fallback": 0}
    for mp in (out / "refine_meta").glob("*/*.json"):
        meta = json.loads(mp.read_text(encoding="utf-8"))
        s = mp.parent.name
        per_split.setdefault(s, {"images": 0, "instances": 0})
        per_split[s]["images"] += 1
        per_split[s]["instances"] += sum(1 for x in meta["instances"] if x["method"] != "dropped_for_review")
        for inst in meta["instances"]:
            methods[inst["method"]] = methods.get(inst["method"], 0) + 1
            if inst["method"] == "dropped_for_review":
                continue
            per_class[inst["class_name"]] = per_class.get(inst["class_name"], 0) + 1
    return {"per_split": per_split, "per_class": dict(sorted(per_class.items(), key=lambda kv: -kv[1])),
            "methods": methods}


if __name__ == "__main__":
    main()
