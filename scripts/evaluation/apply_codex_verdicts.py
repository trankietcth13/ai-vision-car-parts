"""
Apply Codex's verification verdicts (see docs/plans/handoffs/CODEX_QA_HANDOFF.md) to build dataset v5.

    * train split: label fixes applied; images with confirmed `model_error` are oversampled
      (listed --oversample times in train_v5.txt) as hard examples
    * test split : label fixes applied to the test labels only; test images never enter training
    * val split  : copied unchanged

New / re-boxed instances get a SAM2 mask. SAM runs on the DGX (training-only-on-DGX rule): this script
writes a job file, uploads it with the images it needs, runs SAM there and pulls the polygons back.
Use --local-sam only if you explicitly want to run SAM on this machine.

Usage:
    python scripts/evaluation/apply_codex_verdicts.py --src data/engine_bay_train_v4 --out data/engine_bay_train_v5
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DGX = os.environ.get("DGX_SSH_HOST", "admin@dgx-host")
REMOTE = os.environ.get("DGX_REMOTE_DIR", "/srv/distillation_workspace")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT / "data" / "engine_bay_train_v4")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "engine_bay_train_v5")
    ap.add_argument("--verdicts", type=Path, default=ROOT / "qa_results" / "codex_verdicts")
    ap.add_argument("--queue", type=Path, default=ROOT / "qa_results" / "review_queue",
                    help="Unified review queue (merge_errors.py) the verdicts must refer to")
    ap.add_argument("--oversample", type=int, default=3, help="Times a confirmed hard example appears in train")
    ap.add_argument("--local-sam", action="store_true")
    return ap.parse_args()


def link(src: Path, dst: Path):
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def box_poly(b):
    x1, y1, x2, y2 = b
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def run_sam(jobs: list[dict], local: bool) -> dict:
    """jobs: [{"key", "image", "box_norm"}] -> {key: polygon (normalized)}."""
    if not jobs:
        return {}
    script = r'''
import json, sys, cv2, numpy as np
from ultralytics import SAM
jobs = json.load(open(sys.argv[1])); out = {}
sam = SAM("sam2.1_b.pt")
by_img = {}
for j in jobs: by_img.setdefault(j["image"], []).append(j)
for path, js in by_img.items():
    img = cv2.imread(path); h, w = img.shape[:2]
    boxes = [[j["box_norm"][0]*w, j["box_norm"][1]*h, j["box_norm"][2]*w, j["box_norm"][3]*h] for j in js]
    res = sam.predict(img, bboxes=boxes, verbose=False)[0]
    masks = res.masks.data.cpu().numpy() > 0.5 if res.masks is not None else []
    for k, j in enumerate(js):
        poly = None
        if k < len(masks):
            m = masks[k].astype(np.uint8)
            x1, y1, x2, y2 = [int(v) for v in boxes[k]]
            clip = np.zeros_like(m); clip[max(0,y1):y2, max(0,x1):x2] = m[max(0,y1):y2, max(0,x1):x2]
            cs, _ = cv2.findContours(clip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cs:
                c = cv2.approxPolyDP(max(cs, key=cv2.contourArea), 0.002*np.hypot(w, h), True).reshape(-1, 2)
                if len(c) >= 3: poly = (c / [w, h]).tolist()
        out[j["key"]] = poly
json.dump(out, open(sys.argv[2], "w"))
'''
    work = ROOT / "qa_results" / "_sam_job"
    work.mkdir(parents=True, exist_ok=True)
    (work / "sam_job.py").write_text(script, encoding="utf-8")
    if local:
        (work / "jobs.json").write_text(json.dumps(jobs))
        subprocess.run([sys.executable, str(work / "sam_job.py"), str(work / "jobs.json"), str(work / "polys.json")], check=True)
        return json.loads((work / "polys.json").read_text())
    # remote: images already live on the DGX under the same dataset layout
    remote_jobs = [dict(j, image=f"{REMOTE}/data/{Path(j['image']).parents[2].name}/{Path(j['image']).relative_to(Path(j['image']).parents[2]).as_posix()}")
                   for j in jobs]
    (work / "jobs.json").write_text(json.dumps(remote_jobs))
    subprocess.run(["scp", "-o", "BatchMode=yes", str(work / "sam_job.py"), str(work / "jobs.json"), f"{DGX}:{REMOTE}/qa_results/"], check=True)
    subprocess.run(["ssh", "-o", "BatchMode=yes", DGX, f"cd {REMOTE} && .venv/bin/python qa_results/sam_job.py qa_results/jobs.json qa_results/polys.json"], check=True)
    subprocess.run(["scp", "-o", "BatchMode=yes", f"{DGX}:{REMOTE}/qa_results/polys.json", str(work / "polys.json")], check=True)
    polys = json.loads((work / "polys.json").read_text())
    keymap = {rj["key"]: rj for rj in remote_jobs}
    return {k: v for k, v in polys.items() if k in keymap}


def main():
    a = parse_args()
    src, out = a.src.resolve(), a.out.resolve()
    cfg = yaml.safe_load((src / "data_engine_bay_train.yaml").read_text(encoding="utf-8"))
    names = {int(k): v for k, v in cfg["names"].items()}
    cid = {v: k for k, v in names.items()}

    # copy dataset
    for split in ("train", "val", "test"):
        for p in (src / "images" / split).iterdir():
            link(p, out / "images" / split / p.name)
        for p in (src / "labels" / split).glob("*.txt"):
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, out / "labels" / split / p.name)

    # ---- load the unified review queue: verdicts are validated against it
    queue = {}
    for split in ("test", "train", "val"):
        qf = a.queue / split / "ERRORS.json"
        if qf.exists():
            for rec in json.loads(qf.read_text(encoding="utf-8"))["images"]:
                queue[(split, rec["image"])] = {e["error_id"]: e for e in rec["errors"]}
    if not queue:
        raise SystemExit(f"no review queue under {a.queue}; run merge_errors.py first")
    rejected = []

    def n_labels(split, stem):
        lp = src / "labels" / split / f"{stem}.txt"
        return sum(1 for l in lp.read_text().splitlines() if l.strip()) if lp.exists() else 0

    def validate(split, image, e):
        q = queue.get((split, image))
        if q is None:
            return "image not in review queue"
        err = q.get(e.get("error_id"))
        if err is None:
            return f"error_id {e.get('error_id')} not in queue for this image"
        d = e.get("decision", "unsure")
        if d not in ("model_error", "label_missing", "label_wrong_class", "label_extra", "label_bad_geometry", "unsure"):
            return f"unknown decision {d}"
        if d == "label_missing" and err["type"] != "FP":
            return "label_missing is only valid for FP errors"
        if d in ("label_wrong_class", "label_extra", "label_bad_geometry"):
            gid = e.get("gt_id", err.get("gt_id"))
            if gid is None or not (0 <= int(gid) < n_labels(split, Path(image).stem)):
                return f"gt_id {gid} does not exist in the label file"
            e["gt_id"] = int(gid)
        if d == "label_wrong_class" and e.get("new_class") not in cid:
            return f"new_class {e.get('new_class')} is not a training class"
        if d == "label_missing" and e.get("class_name", err.get("pred_class")) not in cid:
            return "class_name is not a training class"
        if d == "label_missing":
            e.setdefault("class_name", err.get("pred_class"))
            e.setdefault("box_norm", err.get("box_norm"))
        for key in ("box_norm", "fixed_box_norm"):
            b = e.get(key)
            if b is not None and not (len(b) == 4 and all(0 <= x <= 1 for x in b) and b[0] < b[2] and b[1] < b[3]):
                return f"{key} is not a valid normalized xyxy box"
        if d == "label_bad_geometry" and not e.get("fixed_box_norm"):
            return "label_bad_geometry needs fixed_box_norm"
        return None

    stats, hard = Counter(), []
    edits = []  # (split, stem, list of ops)
    sam_jobs = []
    for vp in sorted(a.verdicts.glob("*/*.json")):
        split = vp.parent.name
        v = json.loads(vp.read_text(encoding="utf-8"))
        stem = Path(v["image"]).stem
        ops = []
        for e in v.get("errors", []):
            problem = validate(split, v["image"], e)
            if problem:
                rejected.append({"file": str(vp), "error_id": e.get("error_id"), "reason": problem})
                stats["rejected"] += 1
                continue
            d = e.get("decision", "unsure")
            stats[f"{split}:{d}"] += 1
            if d == "model_error" and split == "train":
                hard.append(stem)
            elif d == "label_missing" and e.get("class_name") in cid:
                key = f"{split}/{stem}/{e['error_id']}"
                sam_jobs.append({"key": key, "image": str(src / "images" / split / v["image"]), "box_norm": e["box_norm"]})
                ops.append(("add", cid[e["class_name"]], key, e["box_norm"]))
            elif d == "label_wrong_class" and e.get("new_class") in cid:
                ops.append(("reclass", int(e["gt_id"]), cid[e["new_class"]]))
            elif d == "label_extra":
                ops.append(("remove", int(e["gt_id"])))
            elif d == "label_bad_geometry" and e.get("fixed_box_norm"):
                key = f"{split}/{stem}/{e['error_id']}"
                sam_jobs.append({"key": key, "image": str(src / "images" / split / v["image"]), "box_norm": e["fixed_box_norm"]})
                ops.append(("rebox", int(e["gt_id"]), key, e["fixed_box_norm"]))
        if ops:
            edits.append((split, stem, ops))

    polys = run_sam(sam_jobs, a.local_sam)

    for split, stem, ops in edits:
        lp = out / "labels" / split / f"{stem}.txt"
        lines = [l.split() for l in lp.read_text().splitlines() if l.strip()] if lp.exists() else []
        remove = set()
        for op in ops:
            if op[0] == "reclass" and op[1] < len(lines):
                lines[op[1]][0] = str(op[2])
            elif op[0] == "remove":
                remove.add(op[1])
            elif op[0] == "rebox" and op[1] < len(lines):
                poly = polys.get(op[2]) or box_poly(op[3])
                lines[op[1]] = [lines[op[1]][0]] + [f"{c:.6f}" for pt in poly for c in pt]
            elif op[0] == "add":
                poly = polys.get(op[2]) or box_poly(op[3])
                lines.append([str(op[1])] + [f"{c:.6f}" for pt in poly for c in pt])
        kept = [l for i, l in enumerate(lines) if i not in remove]
        lp.write_text("\n".join(" ".join(l) for l in kept) + ("\n" if kept else ""))

    # train list with hard-example oversampling (paths relative to the dataset root)
    train_imgs = sorted((out / "images" / "train").iterdir())
    hard_set = set(hard)
    lst = []
    for p in train_imgs:
        lst += [f"./images/train/{p.name}"] * (a.oversample if p.stem in hard_set else 1)
    (out / "train_v5.txt").write_text("\n".join(lst) + "\n")
    cfg["path"] = str(out).replace("\\", "/")
    cfg["train"] = "train_v5.txt"
    (out / "data_engine_bay_train.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (out / "REJECTED_VERDICTS.json").write_text(json.dumps(rejected, indent=2))
    summary = {"decisions": dict(stats), "rejected": len(rejected), "hard_examples": len(hard_set), "train_entries": len(lst),
               "label_edits": sum(len(o) for _, _, o in edits), "sam_masks": sum(1 for p in polys.values() if p)}
    (out / "V5_SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
