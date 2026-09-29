"""
Score expert-review verdicts and apply them to build a reviewed engine-bay dataset.

Inputs
    data/engine_bay_review/packets/<split>/<stem>.json   (build_review_packets.py)
    data/engine_bay_review/verdicts/<split>/<stem>.json  (reviewer, see .claude/skills/engine-bay-label-review)
    data/engine_bay_seg/labels|images/<split>/...          (refine_masks_sam2.py)

Outputs
    data/engine_bay_reviewed/images|labels/<split>/...     corrected YOLO-seg dataset (only reviewed images)
    data/engine_bay_reviewed/provenance/<split>/<stem>.json  where every final instance came from
    data/engine_bay_reviewed/data_engine_bay_reviewed.yaml
    data/engine_bay_review/REVIEW_REPORT.json / .md          scores per image / class / decision
    data/engine_bay_review/SPOT_CHECK.txt                    images scoring below --spot-threshold

Decisions -> action
    correct          keep polygon (dropped_candidate: re-segment from its box with SAM2)
    wrong_class      keep polygon, change class
    bad_geometry     re-segment from fixed_box_norm (dropped if no box given)
    not_a_component  drop
    duplicate        drop
    missing[]        segment from box with SAM2 (confidence >= --min-missing-conf)

Usage:
    python scripts/data_pipeline/apply_review.py --splits val
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from data_pipeline.refine_masks_sam2 import atomic_write, box_polygon, materialize, mask_to_polygon  # noqa: E402

DECISIONS = ("correct", "wrong_class", "bad_geometry", "not_a_component", "duplicate")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", type=Path, default=PROJECT_ROOT / "data" / "engine_bay_review")
    ap.add_argument("--seg", type=Path, default=PROJECT_ROOT / "data" / "engine_bay_seg")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "engine_bay_reviewed")
    ap.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "data_engine_bay.yaml")
    ap.add_argument("--splits", nargs="+", default=["val", "test", "train"])
    ap.add_argument("--sam", default="sam2.1_b.pt")
    ap.add_argument("--device", default="0")
    ap.add_argument("--max-side", type=int, default=1536)
    ap.add_argument("--box-pad", type=float, default=0.10)
    ap.add_argument("--min-missing-conf", type=float, default=0.8)
    ap.add_argument("--spot-threshold", type=float, default=60.0)
    ap.add_argument("--exclude-scenes", nargs="*", default=["not_engine_bay"],
                    help="Exclude images whose reviewer scene is in this list (overrides the per-image flag, "
                         "so all reviewers follow one policy). Use --honor-exclude-flag to trust the flag instead.")
    ap.add_argument("--honor-exclude-flag", action="store_true")
    ap.add_argument("--no-sam", action="store_true", help="Use rectangles for new boxes instead of SAM2 masks")
    return ap.parse_args()


class Segmenter:
    """Lazy SAM2 box->polygon helper working in normalized coordinates."""

    def __init__(self, ckpt, device, max_side, pad, disabled=False):
        self.ckpt, self.device, self.max_side, self.pad, self.disabled = ckpt, device, max_side, pad, disabled
        self.model = None

    def polygons(self, image_path: Path, boxes_norm):
        if not boxes_norm:
            return []
        if self.disabled:
            return [box_polygon(b).astype(np.float64) for b in boxes_norm]
        if self.model is None:
            from ultralytics import SAM

            self.model = SAM(self.ckpt)
        img = cv2.imread(str(image_path))
        h0, w0 = img.shape[:2]
        s = min(1.0, self.max_side / max(h0, w0))
        if s < 1.0:
            img = cv2.resize(img, (round(w0 * s), round(h0 * s)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        px = [[b[0] * w, b[1] * h, b[2] * w, b[3] * h] for b in boxes_norm]
        res = self.model.predict(img, bboxes=px, device=self.device, verbose=False)[0]
        masks = res.masks.data.cpu().numpy() > 0.5 if res.masks is not None else []
        out = []
        for k, b in enumerate(px):
            poly = None
            if k < len(masks):
                m = masks[k]
                bw, bh = b[2] - b[0], b[3] - b[1]
                x1, y1 = max(0, int(b[0] - self.pad * bw)), max(0, int(b[1] - self.pad * bh))
                x2, y2 = min(w, int(np.ceil(b[2] + self.pad * bw))), min(h, int(np.ceil(b[3] + self.pad * bh)))
                clipped = np.zeros_like(m)
                clipped[y1:y2, x1:x2] = m[y1:y2, x1:x2]
                if clipped.sum() >= 0.15 * max(bw * bh, 1.0):
                    poly = mask_to_polygon(clipped, 0.002 * float(np.hypot(w, h)), 64)
            if poly is None:
                poly = box_polygon(b)
            out.append(np.clip(poly.astype(np.float64) / [w, h], 0, 1))
        return out


def score_image(verdict, packet):
    labeled_ids = {i["id"] for i in packet["instances"] if i["status"] == "labeled"}
    dec = {int(v["id"]): v for v in verdict.get("instances", [])}
    c = Counter(dec[i]["decision"] for i in labeled_ids if i in dec)
    reviewed = sum(c.values())
    missing = sum(1 for m in verdict.get("missing", []) if float(m.get("confidence", 1)) >= 0.0)
    rescued = sum(1 for i in packet["instances"]
                  if i["status"] == "dropped_candidate" and dec.get(i["id"], {}).get("decision") == "correct")
    true_found = c["correct"] + c["wrong_class"] + c["bad_geometry"]
    precision = c["correct"] / reviewed if reviewed else (1.0 if not missing else 0.0)
    recall = true_found / (true_found + missing + rescued) if (true_found + missing + rescued) else 1.0
    return {
        "labeled": len(labeled_ids), "reviewed": reviewed, "decisions": dict(c), "missing": missing,
        "rescued": rescued, "precision": round(precision, 3), "recall": round(recall, 3),
        "score": round(100 * (0.5 * precision + 0.5 * recall), 1),
    }


def main():
    args = parse_args()
    names = yaml.safe_load(args.config.read_text(encoding="utf-8"))["names"]
    name_to_id = {v: int(k) for k, v in names.items()}
    seg = Segmenter(args.sam, args.device, args.max_side, args.box_pad, args.no_sam)

    images_report, class_stats = [], defaultdict(Counter)
    confusions, problems = Counter(), []
    for split in args.splits:
        vdir = args.review / "verdicts" / split
        if not vdir.exists():
            continue
        for vp in sorted(vdir.glob("*.json")):
            stem = vp.stem
            pp = args.review / "packets" / split / f"{stem}.json"
            if not pp.exists():
                problems.append(f"{split}/{stem}: packet missing")
                continue
            try:
                verdict = json.loads(vp.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                problems.append(f"{split}/{stem}: invalid verdict JSON ({e})")
                continue
            packet = json.loads(pp.read_text(encoding="utf-8"))
            dec = {int(v["id"]): v for v in verdict.get("instances", [])}
            for v in dec.values():
                if v.get("decision") not in DECISIONS:
                    problems.append(f"{split}/{stem}: id {v.get('id')} unknown decision {v.get('decision')}")

            sc = score_image(verdict, packet)
            excluded = (bool(verdict.get("exclude_from_training")) if args.honor_exclude_flag
                        else verdict.get("scene") in set(args.exclude_scenes or []))
            sc.update(image=stem, split=split, scene=verdict.get("scene"),
                      excluded=excluded, comment=verdict.get("comment", ""))
            images_report.append(sc)

            # per-class accuracy of the machine labels
            for inst in packet["instances"]:
                if inst["status"] != "labeled":
                    continue
                d = dec.get(inst["id"], {}).get("decision", "unreviewed")
                class_stats[inst["class_name"]][d] += 1
                if d == "wrong_class":
                    confusions[(inst["class_name"], dec[inst["id"]].get("new_class"))] += 1
            for m in verdict.get("missing", []):
                class_stats[m["class_name"]]["missing"] += 1

            if excluded:
                continue

            # ---- build corrected label
            lines = []
            lp = args.seg / "labels" / split / f"{stem}.txt"
            if lp.exists():
                lines = [l.split() for l in lp.read_text(encoding="utf-8").splitlines() if l.strip()]
            final, provenance, to_segment = [], [], []
            li = 0
            for inst in packet["instances"]:
                v = dec.get(inst["id"], {"decision": "correct" if inst["status"] == "labeled" else "not_a_component"})
                poly = None
                if inst["status"] == "labeled":
                    if li < len(lines):
                        poly = np.array(lines[li][1:], dtype=np.float64).reshape(-1, 2)
                    li += 1
                d = v["decision"]
                if d == "correct" and inst["status"] == "labeled" and poly is not None:
                    final.append((inst["class_name"], poly)); provenance.append({"from": "auto", "id": inst["id"]})
                elif d == "correct" and inst["status"] == "dropped_candidate":
                    to_segment.append((inst["class_name"], inst["box_norm"], {"from": "rescued", "id": inst["id"]}))
                elif d == "wrong_class" and poly is not None and v.get("new_class") in name_to_id:
                    final.append((v["new_class"], poly))
                    provenance.append({"from": "relabeled", "id": inst["id"], "was": inst["class_name"]})
                elif d == "bad_geometry" and v.get("fixed_box_norm"):
                    to_segment.append((v.get("new_class", inst["class_name"]), v["fixed_box_norm"],
                                       {"from": "reboxed", "id": inst["id"]}))
            for m in verdict.get("missing", []):
                if m.get("class_name") in name_to_id and float(m.get("confidence", 1)) >= args.min_missing_conf:
                    to_segment.append((m["class_name"], m["box_norm"], {"from": "missing_added"}))

            # de-duplicate new boxes: same class and IoU > 0.6 with an earlier new box -> keep the first
            from data_pipeline.refine_masks_sam2 import box_iou
            dedup = []
            for item in to_segment:
                if any(item[0] == d[0] and box_iou(item[1], d[1]) > 0.6 for d in dedup):
                    continue
                dedup.append(item)
            to_segment = dedup

            src_img = PROJECT_ROOT / packet["source_image"]
            new_polys = seg.polygons(src_img, [b for _, b, _ in to_segment])
            for (cname, _, prov), poly in zip(to_segment, new_polys):
                final.append((cname, poly)); provenance.append(prov)

            out_lines = [f"{name_to_id[c]} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in p) for c, p in final
                         if c in name_to_id and len(p) >= 3]
            materialize(src_img, args.out / "images" / split / src_img.name, copy=False)
            atomic_write(args.out / "labels" / split / f"{stem}.txt", "\n".join(out_lines) + ("\n" if out_lines else ""))
            atomic_write(args.out / "provenance" / split / f"{stem}.json", json.dumps(
                {"verdict": str(vp.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                 "instances": [{"class_name": c, **p} for (c, _), p in zip(final, provenance)]}, indent=2))

    # ---- excluded images (read by build_training_dataset.py so auto labels never sneak back in)
    atomic_write(args.out / "EXCLUDED.txt",
                 "".join(f"{r['split']}/{r['image']}\n" for r in images_report if r["excluded"]))

    # ---- dataset yaml
    cfg = {"path": str(args.out.resolve()).replace("\\", "/"), "train": "images/train", "val": "images/val",
           "test": "images/test", "nc": len(names), "names": names}
    atomic_write(args.out / "data_engine_bay_reviewed.yaml", yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))

    # ---- report
    tot = Counter()
    for r in images_report:
        tot.update(r["decisions"])
        tot["missing"] += r["missing"]
    labeled = sum(r["reviewed"] for r in images_report)
    found = tot["correct"] + tot["wrong_class"] + tot["bad_geometry"]
    overall = {
        "images_reviewed": len(images_report),
        "images_excluded": sum(r["excluded"] for r in images_report),
        "instances_reviewed": labeled,
        "decisions": dict(tot),
        "machine_precision": round(tot["correct"] / labeled, 3) if labeled else None,
        "machine_recall_est": round(found / (found + tot["missing"]), 3) if (found + tot["missing"]) else None,
        "mean_image_score": round(float(np.mean([r["score"] for r in images_report])), 1) if images_report else None,
    }
    per_class = {}
    for cname in names.values():
        c = class_stats.get(cname, Counter())
        lab = sum(v for k, v in c.items() if k in DECISIONS)
        per_class[cname] = {"labeled": lab, "correct": c["correct"], "wrong_class": c["wrong_class"],
                            "bad_geometry": c["bad_geometry"], "not_a_component": c["not_a_component"],
                            "duplicate": c["duplicate"], "missing": c["missing"],
                            "precision": round(c["correct"] / lab, 3) if lab else None}
    report = {"overall": overall, "per_class": per_class,
              "confusions": [{"labeled_as": a, "actually": b, "count": n} for (a, b), n in confusions.most_common()],
              "images": sorted(images_report, key=lambda r: r["score"]), "problems": problems}
    atomic_write(args.review / "REVIEW_REPORT.json", json.dumps(report, indent=2, ensure_ascii=False))

    spot = [f"{r['split']}/{r['image']}  score={r['score']}  {r['comment']}" for r in report["images"]
            if r["score"] < args.spot_threshold and not r["excluded"]]
    atomic_write(args.review / "SPOT_CHECK.txt", "\n".join(spot) + ("\n" if spot else ""))

    md = ["# Engine bay label review report", "",
          f"- Images reviewed: {overall['images_reviewed']} (excluded {overall['images_excluded']})",
          f"- Machine label precision: {overall['machine_precision']}",
          f"- Machine label recall (estimate): {overall['machine_recall_est']}",
          f"- Mean image score: {overall['mean_image_score']}",
          f"- Images below {args.spot_threshold:.0f} (human spot-check): {len(spot)}", "",
          "## Per class", "", "| class | labeled | correct | wrong class | bad geometry | not a component | missing | precision |",
          "|---|---|---|---|---|---|---|---|"]
    for cname, s in per_class.items():
        if s["labeled"] or s["missing"]:
            md.append(f"| {cname} | {s['labeled']} | {s['correct']} | {s['wrong_class']} | {s['bad_geometry']} | "
                      f"{s['not_a_component']} | {s['missing']} | {s['precision']} |")
    if report["confusions"]:
        md += ["", "## Most common class confusions", "", "| labeled as | actually | count |", "|---|---|---|"]
        md += [f"| {c['labeled_as']} | {c['actually']} | {c['count']} |" for c in report["confusions"][:15]]
    if problems:
        md += ["", "## Problems", ""] + [f"- {p}" for p in problems]
    atomic_write(args.review / "REVIEW_REPORT.md", "\n".join(md) + "\n")
    print(json.dumps(overall, indent=2))
    if problems:
        print("[apply_review] problems:\n  " + "\n  ".join(problems))


if __name__ == "__main__":
    main()
