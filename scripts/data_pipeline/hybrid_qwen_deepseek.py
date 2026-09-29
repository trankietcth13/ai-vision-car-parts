"""
Route low-confidence Qwen images to DeepSeek V4 Flash Vision and merge the two label sets.

An image is routed when any Qwen detection has confidence < --threshold (images where Qwen found
nothing are kept as negatives unless --route-empty).
Merge strategies for a routed image:
    replace   use DeepSeek's detections only
    fill      keep Qwen detections with confidence >= threshold, drop the rest, add DeepSeek
              detections that do not overlap (IoU >= 0.5, any class) a kept Qwen box
Images that are not routed keep Qwen's labels unchanged.

    --evaluate   score strategies x thresholds on the reviewed images already cached by
                 vlm_bakeoff.py (no API calls unless --fetch-missing)
    --apply      call DeepSeek for every routed image in data/engine_bay_labeled and write a merged
                 copy of raw_annotations to data/engine_bay_labeled_hybrid (the Qwen source stays
                 read-only). Feed it to SAM2 with:
                     python scripts/data_pipeline/refine_masks_sam2.py --src data/engine_bay_labeled_hybrid \
                         --out data/engine_bay_seg_hybrid

Images are sent to the external DeepSeek API; the key is read from .env (see vlm_bakeoff.py).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data_pipeline"))
from qwen_grounding_annotator import atomic_write_text, build_prompt, parse_detections  # noqa: E402
from vlm_bakeoff import grounding_to_json, iou, load_gt, pick_images, prf, run_candidate, score  # noqa: E402

THRESHOLDS = (0.85, 0.9, 0.95, 1.0)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--evaluate", action="store_true")
    mode.add_argument("--apply", action="store_true")
    ap.add_argument("--strategy", choices=["replace", "fill"], default="fill")
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--route-empty", action="store_true", help="also send images where Qwen found nothing")
    ap.add_argument("--labeled", type=Path, default=PROJECT_ROOT / "data/engine_bay_labeled")
    ap.add_argument("--reviewed", type=Path, default=PROJECT_ROOT / "data/engine_bay_reviewed")
    ap.add_argument("--hybrid-out", type=Path, default=PROJECT_ROOT / "data/engine_bay_labeled_hybrid")
    ap.add_argument("--splits", nargs="+", default=None, help="default: val test (evaluate) / all (apply)")
    ap.add_argument("--limit", type=int, default=60, help="evaluate: number of reviewed images")
    ap.add_argument("--fetch-missing", action="store_true", help="evaluate: call the API for uncached images")
    ap.add_argument("--max-images", type=int, default=None, help="apply: cap routed images (smoke test)")
    ap.add_argument("--force-stems", type=Path, default=None,
                    help="apply: file of image stems routed to DeepSeek regardless of Qwen confidence")
    # passed through to vlm_bakeoff.run_candidate
    ap.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/vlm_bakeoff", help="DeepSeek response cache root")
    ap.add_argument("--model", default="deepseek-v4-flash-vision-exp")
    ap.add_argument("--thinking", choices=["disabled", "enabled"], default="disabled")
    ap.add_argument("--base-url", default="https://api.deepseek.com/v1")
    ap.add_argument("--key-env", default="DEEPSEEK_API_KEY")
    ap.add_argument("--base-url-env", default="DEEPSEEK_BASE_URL")
    ap.add_argument("--max-side", type=int, default=1400)
    ap.add_argument("--jpeg-quality", type=int, default=88)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--no-json-mode", action="store_true")
    return ap.parse_args()


def cache_dir(args) -> Path:
    slug = re.sub(r"[^\w.-]", "_", args.model)
    return args.output / "raw" / f"{slug}__thinking_{args.thinking}"


def load_deepseek_full(args, stem: str, class_to_id: dict[str, int]) -> list[dict] | None:
    """Full parse_detections records (class_id, pixel boxes, ...) from the cached response."""
    path = cache_dir(args) / f"{stem}.json"
    if not path.exists():
        return None
    rec = json.loads(path.read_text(encoding="utf-8"))
    text = rec["response"]
    if "<|det|>" in text:
        text = grounding_to_json(text)
    dets, _ = parse_detections(text, class_to_id, rec["width"], rec["height"], 0.80, 0.90)
    return dets


def is_routed(qwen_dets: list[dict], threshold: float, route_empty: bool = False) -> bool:
    # Qwen-empty images are mostly true negatives (11/13 on reviewed images) where DeepSeek adds false
    # positives, so they are not routed unless --route-empty
    if not qwen_dets:
        return route_empty
    return min(float(d["confidence"]) for d in qwen_dets) < threshold


def merge(qwen_dets: list[dict], ds_dets: list[dict], threshold: float, strategy: str) -> list[dict]:
    """Works on both full records (bbox_norm_xyxy) and scoring records (box)."""
    def box(d):
        return d.get("bbox_norm_xyxy", d.get("box"))

    if strategy == "replace":
        return list(ds_dets)
    kept = [d for d in qwen_dets if float(d["confidence"]) >= threshold]
    added = [d for d in ds_dets if all(iou(box(d), box(k)) < 0.5 for k in kept)]
    return kept + added


# ---------------------------------------------------------------- evaluate

def evaluate(args, names, class_to_id) -> None:
    splits = args.splits or ["val", "test"]
    images = pick_images(args.reviewed, args.labeled, splits, args.limit)
    if args.fetch_missing:
        run_candidate(args, images, class_to_id, build_prompt(names))

    gt, qwen, ds = {}, {}, {}
    for split, stem, _ in images:
        full = load_deepseek_full(args, stem, class_to_id)
        if full is None:
            continue
        ann = json.loads((args.labeled / "raw_annotations" / split / f"{stem}.json").read_text(encoding="utf-8"))
        qwen[stem] = [{"class_name": d["class_name"], "box": d["bbox_norm_xyxy"], "confidence": d["confidence"]}
                      for d in ann["detections"]]
        ds[stem] = [{"class_name": d["class_name"], "box": d["bbox_norm_xyxy"], "confidence": d["confidence"]}
                    for d in full]
        gt[stem] = load_gt(args.reviewed, split, stem, names)
    stems = list(gt)
    if not stems:
        sys.exit("no cached DeepSeek responses; run vlm_bakeoff.py first or pass --fetch-missing")

    def row(label, preds, routed=None):
        t = score(preds, gt, stems, 0.5)["tot"]
        p, r, f = prf(t["tp"], t["fp"], t["fn"])
        cost = f"{routed}/{len(stems)} ({routed / len(stems):.0%})" if routed is not None else "—"
        return f"| {label} | {cost} | {p:.1%} | {r:.1%} | {f:.3f} |"

    lines = [
        "# Hybrid Qwen -> DeepSeek routing (reviewed images)",
        "",
        f"{len(stems)} images, {sum(len(g) for g in gt.values())} reviewed objects, IoU 0.5. "
        f"An image is routed to DeepSeek when min(Qwen confidence) < threshold"
        f"{' or Qwen found nothing' if args.route_empty else ' (Qwen-empty images kept)'}.",
        "",
        "| system | images sent to DeepSeek | precision | recall | F1 |",
        "|---|---|---|---|---|",
        row("Qwen only", qwen, 0),
        row("DeepSeek only", ds, len(stems)),
    ]
    for strategy in ("fill", "replace"):
        for thr in THRESHOLDS:
            routed = [s for s in stems if is_routed(qwen[s], thr, args.route_empty)]
            hybrid = {s: (merge(qwen[s], ds[s], thr, strategy) if s in routed else qwen[s]) for s in stems}
            lines.append(row(f"{strategy} @ conf < {thr}", hybrid, len(routed)))

    # does Qwen's own low confidence mark the images where DeepSeek does better?
    lines += ["", "## Where does DeepSeek win? (per-image F1 difference)", "",
              "| image group | n | mean F1 Qwen | mean F1 DeepSeek |", "|---|---|---|---|"]
    groups = {"Qwen empty": [], "Qwen min conf < 0.9": [], "Qwen min conf >= 0.9": []}
    for s in stems:
        key = ("Qwen empty" if not qwen[s] else
               "Qwen min conf < 0.9" if min(d["confidence"] for d in qwen[s]) < 0.9 else "Qwen min conf >= 0.9")
        groups[key].append(s)
    def image_f1(preds, s):
        tp, fp, fn = (score(preds, gt, [s], 0.5)["tot"][k] for k in ("tp", "fp", "fn"))
        return 1.0 if tp + fp + fn == 0 else prf(tp, fp, fn)[2]  # empty GT and no predictions = correct

    for key, ss in groups.items():
        if not ss:
            continue
        fq = [image_f1(qwen, s) for s in ss]
        fd = [image_f1(ds, s) for s in ss]
        lines.append(f"| {key} | {len(ss)} | {sum(fq) / len(ss):.3f} | {sum(fd) / len(ss):.3f} |")

    report = "\n".join(lines) + "\n"
    out = args.output / "REPORT_hybrid_routing.md"
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"wrote {out}")


# ---------------------------------------------------------------- apply

def apply(args, names, class_to_id) -> None:
    splits = args.splits or sorted(p.name for p in (args.labeled / "raw_annotations").iterdir() if p.is_dir())
    anns = []
    for split in splits:
        for jp in sorted((args.labeled / "raw_annotations" / split).glob("*.json")):
            ann = json.loads(jp.read_text(encoding="utf-8"))
            if ann.get("status") == "complete":
                anns.append((split, jp.stem, ann))
    forced = set()
    if args.force_stems:
        forced = {l.strip() for l in args.force_stems.read_text(encoding="utf-8").splitlines() if l.strip()}
    # images routed in an earlier run stay routed, so re-running with --force-stems only adds images
    previous = set()
    for jp in (args.hybrid_out / "raw_annotations").glob("*/*.json"):
        if json.loads(jp.read_text(encoding="utf-8")).get("hybrid", {}).get("routed"):
            previous.add(jp.stem)
    routed = [(split, stem, ann) for split, stem, ann in anns
              if stem in forced or stem in previous or is_routed(ann["detections"], args.threshold, args.route_empty)]
    if args.max_images:
        routed = routed[: args.max_images]
    print(f"{len(anns)} Qwen images; routing {len(routed)} to {args.model} "
          f"(min confidence < {args.threshold}, route_empty={args.route_empty}, forced={len(forced)}, "
          f"previously routed={len(previous)}, strategy={args.strategy})", flush=True)

    images = [(split, stem, (args.labeled / ann["output_image"]).resolve()) for split, stem, ann in routed]
    run_candidate(args, images, class_to_id, build_prompt(names))

    routed_stems = {stem for _, stem, _ in routed}
    stats = Counter()
    for split, stem, ann in anns:
        out = dict(ann)
        # absolute path so refine_masks_sam2 (src / output_image) finds the original image
        out["output_image"] = str((args.labeled / ann["output_image"]).resolve()).replace("\\", "/")
        for d in ann["detections"]:
            d.setdefault("source", "qwen")
        if stem in routed_stems:
            ds = load_deepseek_full(args, stem, class_to_id)
            if ds is None:  # API failure: keep Qwen labels, mark for retry
                out["hybrid"] = {"routed": True, "deepseek": "failed"}
                stats["failed"] += 1
            else:
                for d in ds:
                    d["source"] = "deepseek"
                out["detections"] = merge(ann["detections"], ds, args.threshold, args.strategy)
                out["hybrid"] = {"routed": True, "model": args.model, "strategy": args.strategy,
                                 "threshold": args.threshold, "forced": stem in forced,
                                 "qwen_before": len(ann["detections"]), "after": len(out["detections"])}
                stats["routed"] += 1
        else:
            out["hybrid"] = {"routed": False}
            stats["kept_qwen"] += 1
        stats.update(d["source"] for d in out["detections"])
        # write only on change: refine_masks_sam2 reprocesses any annotation newer than its label
        dest = args.hybrid_out / "raw_annotations" / split / f"{stem}.json"
        text = json.dumps(out, ensure_ascii=False, indent=2)
        if not dest.exists() or dest.read_text(encoding="utf-8") != text:
            atomic_write_text(dest, text)
            stats["files_written"] += 1

    yaml_src = args.labeled / "data_engine_bay_labeled.yaml"
    if yaml_src.exists():
        (args.hybrid_out / "data_engine_bay_labeled.yaml").write_text(yaml_src.read_text(encoding="utf-8"),
                                                                     encoding="utf-8")
    summary = {"model": args.model, "strategy": args.strategy, "threshold": args.threshold, **stats}
    (args.hybrid_out / "hybrid_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    names = yaml.safe_load((args.reviewed / "data_engine_bay_reviewed.yaml").read_text(encoding="utf-8"))["names"]
    names = {int(k): v for k, v in names.items()}
    class_to_id = {v: k for k, v in names.items()}
    evaluate(args, names, class_to_id) if args.evaluate else apply(args, names, class_to_id)


if __name__ == "__main__":
    main()
