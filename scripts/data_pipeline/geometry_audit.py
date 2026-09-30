"""D2: geometry (extent) audit of reviewed labels for elongated / ambiguous-extent classes.

    sample  - N reviewed instances per class from train/val (stratified by vehicle) + every test instance of the class
              the teacher localises badly (best same-class IoU 0.1-0.5, the "localisation" misses of the miss analysis);
              renders one review crop per item from the original photo: label polygon (green), teacher box (orange,
              test items), grid labelled in ORIGINAL-image normalised coordinates -> fixed boxes can be read off directly
    summary - aggregates reviewer verdicts (data/engine_bay_review_geometry/verdicts/*.json)

Policy: docs/plans/label_policy_reservoirs_heat_shield.md section 2a.

    python scripts/data_pipeline/geometry_audit.py sample --per-class 30
    python scripts/data_pipeline/geometry_audit.py summary --out docs/reports/geometry_audit_d2.md
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from data_pipeline.taxonomy_stats import read_names  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
OUT = ROOT / "data" / "engine_bay_review_geometry"
CLASSES = ["air_intake_duct", "battery_terminal", "battery"]
SOURCES = [(ROOT / "data" / "engine_bay_reviewed", ("train", "val", "test")),
           (ROOT / "data" / "engine_bay_reviewed_hybrid", ("train",))]


def _grid_fn():
    spec = importlib.util.spec_from_file_location(
        "inspect_image", ROOT / ".claude" / "skills" / "automotive-expert" / "scripts" / "inspect_image.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.draw_grid


def instances():
    seen, out = set(), []
    for src, splits in SOURCES:
        names = read_names(src)
        for split in splits:
            for lp in sorted((src / "labels" / split).glob("*.txt")):
                if lp.stem in seen:
                    continue
                seen.add(lp.stem)
                img = next((src / "images" / split / (lp.stem + e) for e in (".jpg", ".jpeg", ".png")
                            if (src / "images" / split / (lp.stem + e)).exists()), None)
                if img is None:
                    continue
                for li, line in enumerate(lp.read_text().splitlines()):
                    p = line.split()
                    if len(p) < 7 or names[int(p[0])] not in CLASSES:
                        continue
                    xy = np.asarray(p[1:], float).reshape(-1, 2)
                    out.append(dict(key=f"{src.name}/{split}/{lp.stem}#{li}", cls=names[int(p[0])], split=split,
                                    vehicle=lp.stem.split("__")[0], image=str(img), poly=xy.tolist(),
                                    box=[float(xy[:, 0].min()), float(xy[:, 1].min()), float(xy[:, 0].max()), float(xy[:, 1].max())]))
    return out


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def test_localisation_cases(items, weights):
    """Test instances whose best same-class teacher box has IoU in [0.1, 0.5) -> attach the teacher box."""
    from inference.teacher_system import TeacherSystem, load_image
    ts = TeacherSystem(weights, conf=0.01)
    by_img = collections.defaultdict(list)
    for it in items:
        if it["split"] == "test":
            by_img[it["image"]].append(it)
    picked = []
    for path, its in by_img.items():
        dets = ts.postprocess(ts.raw(load_image(path), use_tiles=False)[0], use_cap=False, use_agnostic=False)
        for it in its:
            same = [d for d in dets if d["cls"] == it["cls"]]
            if not same:
                continue
            best = max(same, key=lambda d: iou(d["box"], it["box"]))
            v = iou(best["box"], it["box"])
            if 0.1 <= v < 0.5:
                picked.append({**it, "teacher_box": [round(x, 4) for x in best["box"]], "teacher_iou": round(v, 3),
                               "teacher_score": round(best["score"], 3)})
    return picked


def render(it, out_path, draw_grid):
    from inference.teacher_system import open_upright
    im = open_upright(it["image"], draft_side=3000)  # labels are in EXIF-upright coordinates
    W, H = im.size
    boxes = [it["box"]] + ([it["teacher_box"]] if it.get("teacher_box") else [])
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    w, h = max(x1 - x0, 0.12), max(y1 - y0, 0.12)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    c = [max(cx - w * 0.85, 0), max(cy - h * 0.85, 0), min(cx + w * 0.85, 1), min(cy + h * 0.85, 1)]
    view = im.crop((int(c[0] * W), int(c[1] * H), int(c[2] * W), int(c[3] * H)))
    s = min(1.0, 1400 / max(view.size))
    view = view.resize((max(1, int(view.width * s)), max(1, int(view.height * s))))
    vw, vh = view.size

    def to_view(x, y):
        return ((x - c[0]) / (c[2] - c[0]) * vw, (y - c[1]) / (c[3] - c[1]) * vh)

    overlay = Image.new("RGBA", view.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    pts = [to_view(x, y) for x, y in it["poly"]]
    d.polygon(pts, fill=(0, 255, 0, 50), outline=(0, 255, 0, 255), width=3)
    if it.get("teacher_box"):
        b = it["teacher_box"]
        d.rectangle([*to_view(b[0], b[1]), *to_view(b[2], b[3])], outline=(255, 140, 0, 255), width=3)
    view = Image.alpha_composite(view.convert("RGBA"), overlay).convert("RGB")
    step = 0.02 if (c[2] - c[0]) < 0.25 else 0.05
    view = draw_grid(view, c[0], c[1], c[2], c[3], step)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    view.save(out_path, quality=88)
    return c


def cmd_sample(a):
    rng = random.Random(a.seed)
    items = instances()
    draw_grid = _grid_fn()
    picked = []
    for cls in CLASSES:
        pool = [it for it in items if it["cls"] == cls and it["split"] != "test"]
        by_v = collections.defaultdict(list)
        for it in pool:
            by_v[it["vehicle"]].append(it)
        for v in by_v.values():
            rng.shuffle(v)
        queues = [by_v[k] for k in sorted(by_v)]
        rng.shuffle(queues)
        n = 0
        while n < a.per_class and any(queues):
            for q in queues:
                if q and n < a.per_class:
                    picked.append({**q.pop(), "origin": "sample"})
                    n += 1
    loc = test_localisation_cases(items, a.teacher)
    picked += [{**it, "origin": "test_localisation"} for it in loc]
    for k, it in enumerate(picked):
        crop = OUT / "items" / f"{k:03d}_{it['cls']}.jpg"
        it["crop_box_norm"] = [round(v, 4) for v in render(it, crop, draw_grid)]
        it["crop"] = str(crop.relative_to(ROOT)).replace("\\", "/")
        it["id"] = k
        it.pop("poly")
    batches = [picked[i:i + a.batch] for i in range(0, len(picked), a.batch)]
    (OUT / "batches").mkdir(parents=True, exist_ok=True)
    for i, b in enumerate(batches):
        (OUT / "batches" / f"geo_batch_{i}.json").write_text(json.dumps(b, indent=1), encoding="utf-8")
    print(json.dumps({"items": len(picked), "by_class": collections.Counter(p["cls"] for p in picked),
                      "by_origin": collections.Counter(p["origin"] for p in picked), "batches": len(batches)}, indent=1))


def cmd_summary(a):
    items = {}
    for f in sorted((OUT / "batches").glob("geo_batch_*.json")):
        for it in json.loads(f.read_text(encoding="utf-8")):
            items[it["key"]] = it
    by_key = {}
    for f in sorted((OUT / "verdicts").glob("*.json")):  # later files (e.g. zz_*_overrides.json) win per key
        for v in json.loads(f.read_text(encoding="utf-8")):
            by_key[v["key"]] = v
    ver = list(by_key.values())
    L = ["# D2 geometry audit: air intake duct, battery terminal, battery", "",
         f"{len(ver)} reviewed items (policy: docs/plans/label_policy_reservoirs_heat_shield.md §2a).", ""]
    for origin in ("sample", "test_localisation"):
        vs = [v for v in ver if items.get(v["key"], {}).get("origin") == origin]
        if not vs:
            continue
        L += [f"## {origin} ({len(vs)})", "", "| class | n | ok | needs_fix | split/merge | wrong_class / not_a_component |",
              "|---|---|---|---|---|---|"]
        for cls in CLASSES:
            cv = [v for v in vs if items[v["key"]]["cls"] == cls]
            c = collections.Counter(v["verdict"] for v in cv)
            L.append(f"| {cls} | {len(cv)} | {c['ok']} | {c['needs_fix']} | {c['split_instances']} | "
                     f"{c['wrong_class'] + c['not_a_component']} |")
        if origin == "test_localisation":
            src = collections.Counter(v.get("error_source", "?") for v in vs)
            L += ["", "Who is wrong on the test localisation misses: " + ", ".join(f"{k} {n}" for k, n in src.most_common())]
        L.append("")
    fixes = [v for v in ver if v["verdict"] != "ok"]
    (OUT / "CORRECTIONS.jsonl").write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in fixes), encoding="utf-8")
    L += [f"Corrections for the next build: data/engine_bay_review_geometry/CORRECTIONS.jsonl ({len(fixes)} entries)."]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["sample", "summary"])
    ap.add_argument("--per-class", type=int, default=30)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--teacher", type=Path, default=ROOT / "runs" / "segment" / "p5_reg" / "weights" / "avg5.pt")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "reports" / "geometry_audit_d2.md")
    a = ap.parse_args()
    {"sample": cmd_sample, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()
