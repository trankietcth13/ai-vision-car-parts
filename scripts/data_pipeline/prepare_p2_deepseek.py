"""
Phase 2 relabel round with DeepSeek (user request 2026-09-28): stage the 332 images of dataset/ that have no reviewed
label so scripts/data_pipeline/hybrid_qwen_deepseek.py --apply can relabel them, then (step `remote`) rewrite the result for SAM2
on the DGX.

  stage   -> <staging>/raw_annotations/p2/<stem>.json for every candidate in CANDIDATES.txt
             * Qwen record copied when it exists (output_image made absolute)
             * a minimal empty record for the 18 images whose Qwen JSON failed
  remote  -> copy <hybrid-out>/raw_annotations/p2 to <remote-out>, pointing output_image to the resized upload copy
             on the DGX (phase2_upload/<stem>.jpg; boxes are normalized so the resize does not matter)

Usage:
    python scripts/data_pipeline/prepare_p2_deepseek.py stage
    python scripts/data_pipeline/hybrid_qwen_deepseek.py --apply --labeled data/engine_bay_labeled_p2src \
        --hybrid-out data/engine_bay_labeled_p2ds --strategy replace --force-stems data/engine_bay_phase2/CANDIDATES.txt
    python scripts/data_pipeline/prepare_p2_deepseek.py remote
    # the images DeepSeek left empty are labelled by Codex (docs/plans/handoffs/CODEX_P2_LABEL_HANDOFF.md), then:
    python scripts/data_pipeline/prepare_p2_deepseek.py codex     # merge + re-run `remote`; push and re-run SAM2 on the DGX
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
REMOTE_ROOT = os.environ.get("DGX_REMOTE_DIR", "/srv/distillation_workspace")


def cmd_stage(a):
    recs = {json.loads(l)["stem"]: json.loads(l) for l in a.candidates_jsonl.read_text(encoding="utf-8").splitlines()}
    stems = a.candidates.read_text().split()
    out = a.staging / "raw_annotations" / "p2"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    made = {"copied": 0, "synthesized": 0}
    for stem in stems:
        js = list((a.labeled / "raw_annotations").glob(f"*/{stem}.json"))
        if js:
            ann = json.loads(js[0].read_text(encoding="utf-8"))
            ann["output_image"] = str((a.labeled / ann["output_image"]).resolve()).replace("\\", "/")
            ann["split"] = "p2"
            made["copied"] += 1
        else:
            src = ROOT / recs[stem]["path"]
            im = cv2.imread(str(src))
            h, w = im.shape[:2]
            ann = {"schema_version": "1.0", "status": "complete", "source_image": recs[stem]["path"].split("dataset/")[-1],
                   "output_image": str(src.resolve()).replace("\\", "/"), "request_id": stem.split("__")[0], "split": "p2",
                   "image_width": w, "image_height": h, "model": "none (qwen json failed)", "detections": [],
                   "parse_warnings": ["qwen_json_failed_placeholder"]}
            made["synthesized"] += 1
        (out / f"{stem}.json").write_text(json.dumps(ann, ensure_ascii=False, indent=2), encoding="utf-8")
    for f in ("data_engine_bay_labeled.yaml",):
        if (a.labeled / f).exists():
            shutil.copy2(a.labeled / f, a.staging / f)
    print(json.dumps({**made, "staging": str(a.staging)}))


def cmd_remote(a):
    src = a.hybrid_out / "raw_annotations" / "p2"
    out = a.remote_out / "raw_annotations" / "p2"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    n, dets, routed_fail = 0, 0, 0
    for jp in sorted(src.glob("*.json")):
        ann = json.loads(jp.read_text(encoding="utf-8"))
        if ann.get("hybrid", {}).get("deepseek") == "failed":
            routed_fail += 1
        ann["output_image"] = f"{REMOTE_ROOT}/phase2_upload/{jp.stem}.jpg"
        (out / jp.name).write_text(json.dumps(ann, ensure_ascii=False, indent=2), encoding="utf-8")
        n += 1
        dets += len(ann["detections"])
    for f in ("data_engine_bay_labeled.yaml", "hybrid_summary.json"):
        if (a.hybrid_out / f).exists():
            shutil.copy2(a.hybrid_out / f, a.remote_out / f)
    print(json.dumps({"images": n, "detections": dets, "deepseek_failed": routed_fail, "out": str(a.remote_out)}))


def cmd_codex(a):
    """Merge Codex labels (data/engine_bay_codex_p2/labels/<stem>.json) into the DeepSeek draft, then rebuild the DGX copy."""
    import yaml

    names = yaml.safe_load((ROOT / "configs" / "data_engine_bay.yaml").read_text(encoding="utf-8"))["names"]
    cid = {v: int(k) for k, v in names.items()}
    todo = [Path(l.strip()).stem for l in (a.codex / "IMAGES.txt").read_text().splitlines() if l.strip()]
    rejected, stats, excluded = [], {"merged_images": 0, "detections": 0, "missing_files": 0, "excluded": 0}, []
    for stem in todo:
        f = a.codex / "labels" / f"{stem}.json"
        if not f.exists():
            stats["missing_files"] += 1
            continue
        try:
            v = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            rejected.append({"image": stem, "reason": f"invalid JSON: {e}"})
            continue
        dp = a.hybrid_out / "raw_annotations" / "p2" / f"{stem}.json"
        ann = json.loads(dp.read_text(encoding="utf-8"))
        dets = []
        for i, d in enumerate(v.get("detections", [])):
            b, c = d.get("bbox_norm_xyxy"), d.get("class_name")
            if c not in cid:
                rejected.append({"image": stem, "index": i, "reason": f"unknown class {c}"})
                continue
            if not (isinstance(b, list) and len(b) == 4 and all(isinstance(x, (int, float)) and 0 <= x <= 1 for x in b)
                    and b[0] < b[2] and b[1] < b[3]):
                rejected.append({"image": stem, "index": i, "reason": f"bad box {b}"})
                continue
            w, h = ann.get("image_width", 1), ann.get("image_height", 1)
            dets.append({"class_id": cid[c], "class_name": c, "bbox_norm_xyxy": [round(float(x), 4) for x in b],
                         "bbox_pixel_xyxy": [round(b[0] * w), round(b[1] * h), round(b[2] * w), round(b[3] * h)],
                         "confidence": float(d.get("confidence", 0.8)), "visual_evidence": d.get("visual_evidence", ""),
                         "source": "codex", "training_eligible": True, "review_required": True})
        ann["detections"] = dets
        ann["codex"] = {"labeler": v.get("labeler", "codex"), "scene": v.get("scene"), "exclude": bool(v.get("exclude")),
                        "comment": v.get("comment", "")}
        if v.get("exclude"):
            excluded.append(stem)
            stats["excluded"] += 1
        dp.write_text(json.dumps(ann, ensure_ascii=False, indent=2), encoding="utf-8")
        stats["merged_images"] += 1
        stats["detections"] += len(dets)
    # DeepSeek precision from Codex's per-box decisions (check round only)
    ds_review = {}
    for stem in todo:
        f = a.codex / "labels" / f"{stem}.json"
        if f.exists():
            try:
                for r in json.loads(f.read_text(encoding="utf-8")).get("deepseek_review", []):
                    ds_review[r.get("decision", "missing")] = ds_review.get(r.get("decision", "missing"), 0) + 1
            except json.JSONDecodeError:
                pass
    if ds_review:
        n = sum(ds_review.values())
        stats["deepseek_review"] = ds_review
        stats["deepseek_precision"] = round(ds_review.get("correct", 0) / n, 3) if n else None
    (a.codex / "REJECTED.json").write_text(json.dumps(rejected, indent=2))
    (a.codex / "EXCLUDE_SUGGESTED.txt").write_text("".join(s + "\n" for s in excluded))
    print(json.dumps({**stats, "rejected": len(rejected), "images_in_list": len(todo)}))
    cmd_remote(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stage", "remote", "codex"])
    ap.add_argument("--labeled", type=Path, default=ROOT / "data" / "engine_bay_labeled")
    ap.add_argument("--candidates", type=Path, default=ROOT / "data" / "engine_bay_phase2" / "CANDIDATES.txt")
    ap.add_argument("--candidates-jsonl", type=Path, default=ROOT / "data" / "engine_bay_phase2" / "CANDIDATES.jsonl")
    ap.add_argument("--staging", type=Path, default=ROOT / "data" / "engine_bay_labeled_p2src")
    ap.add_argument("--hybrid-out", type=Path, default=ROOT / "data" / "engine_bay_labeled_p2ds")
    ap.add_argument("--remote-out", type=Path, default=ROOT / "data" / "engine_bay_labeled_p2ds_dgx")
    ap.add_argument("--codex", type=Path, default=ROOT / "data" / "engine_bay_codex_p2")
    a = ap.parse_args()
    {"stage": cmd_stage, "remote": cmd_remote, "codex": cmd_codex}[a.cmd](a)


if __name__ == "__main__":
    main()
