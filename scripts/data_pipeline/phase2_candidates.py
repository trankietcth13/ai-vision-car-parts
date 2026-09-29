"""
Phase 2 of the engine-bay roadmap (docs/plans/engine_bay_accuracy_roadmap_2026-09-28.md): find every image of
`dataset/` that still lacks an expert-reviewed label, and stage it for teacher pseudo-labelling + review.

Provenance sources counted as "reviewed" (same precedence as build_v8.py):
    data/engine_bay_reviewed/labels/<split>        expert-review skill (train/val/test), minus EXCLUDED.txt
    data/engine_bay_reviewed_hybrid/labels/train   DeepSeek->SAM2->expert review round (v8), minus EXCLUDED.txt
    keep_labels_v6.json["codex"]                   Codex-corrected train labels (v5)

Every `dataset/<Request_ID>/<img>.jpg` gets one status:
    reviewed                  nothing to do
    excluded                  a reviewer excluded it (not an engine bay etc.) -> stays out
    unreadable                cv2 cannot decode the file -> drop from the dataset
    unreviewed_in_train       in the current train list with a pseudo/Qwen label -> relabel
    unreviewed_not_in_train   never entered training (empty Qwen label, failed Qwen JSON, capped) -> label

Outputs (data/engine_bay_phase2/):
    CANDIDATES.jsonl   one record per dataset image (stem, vehicle, status, held-out flag, source path)
    CANDIDATES.txt     stems to relabel (unreviewed_*), grouped by vehicle
    batches/batch_NN.txt   review batches of --batch stems
    upload/<stem>.jpg  resized copies (max side --max-side) to send to the DGX for external_to_seg.py
    SUMMARY.md / SUMMARY.json

Usage:
    python scripts/data_pipeline/phase2_candidates.py --train-stems data/engine_bay_phase2/v8_train_stems.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# vehicles in the v6/v7/v8 val + test splits (resplit_val.py); their images may be labelled for goal A
# (all-28-vehicle model) but must never join the goal-B train split
HELDOUT_DEFAULT = ["Request_ID_08", "Request_ID_13", "Request_ID_23", "Request_ID_29", "Request_ID_36",
                   "Request_ID_49", "Request_ID_50", "Request_ID_52", "Request_ID_58"]


def stem_for(dataset_root: Path, img: Path) -> str:
    """Same convention as qwen_grounding_annotator.make_tasks."""
    rel = img.relative_to(dataset_root)
    request_id = rel.parts[0] if len(rel.parts) > 1 else "ungrouped"
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", request_id)
    return f"{safe}__{img.stem}__{hashlib.sha1(rel.as_posix().encode('utf-8')).hexdigest()[:8]}"


def stems_of(label_root: Path, excluded_file: Path | None) -> tuple[set[str], set[str]]:
    excluded = set()
    if excluded_file and excluded_file.exists():
        excluded = {line.split("/")[-1].strip() for line in excluded_file.read_text().split() if line.strip()}
    stems = set()
    if label_root.exists():
        for split_dir in label_root.iterdir():
            if split_dir.is_dir():
                stems |= {p.stem for p in split_dir.glob("*.txt")}
    return stems - excluded, excluded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=ROOT / "dataset")
    ap.add_argument("--reviewed", type=Path, default=ROOT / "data" / "engine_bay_reviewed")
    ap.add_argument("--hybrid", type=Path, default=ROOT / "data" / "engine_bay_reviewed_hybrid")
    ap.add_argument("--keep", type=Path, default=ROOT / "keep_labels_v6.json")
    ap.add_argument("--train-stems", type=Path, default=None,
                    help="text file with the stems of the current train split (ls images/train on the DGX)")
    ap.add_argument("--heldout-vehicles", nargs="*", default=HELDOUT_DEFAULT)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "engine_bay_phase2")
    ap.add_argument("--batch", type=int, default=30)
    ap.add_argument("--max-side", type=int, default=1600)
    ap.add_argument("--no-stage", action="store_true", help="only write the lists, do not resize/copy images")
    a = ap.parse_args()

    reviewed, excl_r = stems_of(a.reviewed / "labels", a.reviewed / "EXCLUDED.txt")
    hybrid, excl_h = stems_of(a.hybrid / "labels", a.hybrid / "EXCLUDED.txt")
    codex = set(json.loads(a.keep.read_text(encoding="utf-8")).get("codex", [])) if a.keep.exists() else set()
    excluded = excl_r | excl_h
    train_stems = set(a.train_stems.read_text().split()) if a.train_stems else set()
    heldout = set(a.heldout_vehicles)

    import cv2

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "batches").mkdir(exist_ok=True)
    if not a.no_stage:
        (a.out / "upload").mkdir(exist_ok=True)

    records, by_status, per_vehicle = [], Counter(), defaultdict(Counter)
    images = sorted(p for p in a.dataset.rglob("*") if p.suffix.lower() in IMG_SUFFIXES)
    for img in images:
        stem = stem_for(a.dataset, img)
        vehicle = stem.split("__")[0]
        source = None
        if stem in excluded:
            status = "excluded"
        elif stem in hybrid:
            status, source = "reviewed", "hybrid_review"
        elif stem in reviewed:
            status, source = "reviewed", "expert_review"
        elif stem in codex:
            status, source = "reviewed", "codex_fix"
        else:
            im = cv2.imread(str(img))
            if im is None:
                status = "unreadable"
            else:
                status = "unreviewed_in_train" if stem in train_stems else "unreviewed_not_in_train"
                if not a.no_stage:
                    dst = a.out / "upload" / f"{stem}.jpg"
                    if not dst.exists():
                        h, w = im.shape[:2]
                        s = min(1.0, a.max_side / max(h, w))
                        if s < 1.0:
                            im = cv2.resize(im, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
                        cv2.imwrite(str(dst), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
        records.append({"stem": stem, "vehicle": vehicle, "status": status, "source": source,
                        "heldout_vehicle": vehicle in heldout, "in_train": stem in train_stems,
                        "path": str(img.relative_to(ROOT)).replace("\\", "/")})
        by_status[status] += 1
        per_vehicle[vehicle][status] += 1

    cands = [r for r in records if r["status"].startswith("unreviewed")]
    (a.out / "CANDIDATES.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    (a.out / "CANDIDATES.txt").write_text("\n".join(r["stem"] for r in cands) + "\n", encoding="utf-8")
    for old in (a.out / "batches").glob("batch_*.txt"):
        old.unlink()
    for i in range(0, len(cands), a.batch):
        (a.out / "batches" / f"batch_{i // a.batch + 1:02d}.txt").write_text(
            "\n".join(r["stem"] for r in cands[i:i + a.batch]) + "\n", encoding="utf-8")
    n_batches = (len(cands) + a.batch - 1) // a.batch

    summary = {
        "dataset_images": len(images), "vehicles": len(per_vehicle), **by_status,
        "candidates": len(cands),
        "candidates_heldout_vehicles": sum(r["heldout_vehicle"] for r in cands),
        "candidates_train_vehicles": sum(not r["heldout_vehicle"] for r in cands),
        "unreadable_files": [r["path"] for r in records if r["status"] == "unreadable"],
        "review_batches": n_batches, "batch_size": a.batch,
        "train_stems_given": bool(a.train_stems), "train_stems": len(train_stems),
    }
    (a.out / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = ["# Phase 2 candidates — images of `dataset/` without a reviewed label", "",
             f"Dataset: {len(images)} images, {len(per_vehicle)} vehicles.", "",
             "| Status | Images |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in sorted(by_status.items())]
    lines += ["", f"**Candidates to relabel: {len(cands)}** "
                  f"({summary['candidates_train_vehicles']} on train vehicles, "
                  f"{summary['candidates_heldout_vehicles']} on val/test vehicles) in {n_batches} batches of {a.batch}.",
              "", "| Vehicle | reviewed | unreviewed_in_train | unreviewed_not_in_train | excluded | unreadable | held-out |",
              "|---|---|---|---|---|---|---|"]
    for v in sorted(per_vehicle):
        c = per_vehicle[v]
        lines.append(f"| {v} | {c['reviewed']} | {c['unreviewed_in_train']} | {c['unreviewed_not_in_train']} | "
                     f"{c['excluded']} | {c['unreadable']} | {'yes' if v in heldout else ''} |")
    if summary["unreadable_files"]:
        lines += ["", "Unreadable files (drop):", *[f"- `{p}`" for p in summary["unreadable_files"]]]
    lines += ["", "Next: upload `upload/` to the DGX, run `scripts/data_pipeline/external_to_seg.py --split p2` with teacher v8, "
                  "pull the seg layout back, then `build_review_packets.py --seg data/engine_bay_p2_seg --splits p2`."]
    (a.out / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "unreadable_files"}, indent=2))
    print("unreadable:", summary["unreadable_files"])


if __name__ == "__main__":
    main()
