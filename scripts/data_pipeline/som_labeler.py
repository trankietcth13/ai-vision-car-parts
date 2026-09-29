"""W1b-c pilot: Set-of-Mark labelling. Geometry proposes regions, the VLM only names numbered regions.

    propose  teacher detections (p5_reg, conf >= 0.15) + SAM2 "segment everything" (16x16 point grid, CPU ok)
             -> filtered regions (area, duplicates, cap) -> image with numbered outlines
    label    VLM (DeepSeek, thinking OFF) names each numbered region from the taxonomy-v2 component list,
             with a visible cue and the look-alikes it ruled out; may add clearly visible unmarked components
    score    same 60 bake-off images and IoU >= 0.5 matching as vlm_bakeoff.py, every name mapped to the
             taxonomy-v2 training class: Qwen (existing), DeepSeek direct boxes (cached), teacher alone, SoM

    python scripts/data_pipeline/som_labeler.py propose --limit 60
    python scripts/data_pipeline/som_labeler.py label             # sends marked images to the VLM API
    python scripts/data_pipeline/som_labeler.py score --out docs/reports/som_pilot.md
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_pipeline.taxonomy import load_taxonomy  # noqa: E402
from data_pipeline.vlm_bakeoff import load_gt, load_qwen, match, pick_images, prf  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "vlm_bakeoff" / "som"
Image.MAX_IMAGE_PIXELS = None
COLORS = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
          (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255)]


def box_iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def open_view(path, max_side):
    im = Image.open(path)
    if im.format == "JPEG":
        im.draft("RGB", (max_side, max_side))
    im = im.convert("RGB")
    im.thumbnail((max_side, max_side))
    return im


# ---------------------------------------------------------------------------- propose
class SamEverything:
    def __init__(self, ckpt="sam2.1_b.pt", imgsz=1024, stride=16):
        from ultralytics.models.sam import SAM2Predictor
        self.p = SAM2Predictor(overrides=dict(model=ckpt, device="cpu", verbose=False, imgsz=imgsz, save=False))
        self.p.setup_model(verbose=False)
        self.imgsz, self.stride = imgsz, stride

    def __call__(self, im: Image.Image):
        import torch
        bgr = np.ascontiguousarray(np.array(im)[:, :, ::-1])
        self.p.setup_source(bgr)
        batch = next(iter(self.p.dataset))
        self.p.batch = batch
        x = self.p.preprocess(batch[1])
        with torch.no_grad():
            masks, scores, boxes = self.p.generate(x, points_stride=self.stride, points_batch_size=64,
                                                   conf_thres=0.8, stability_score_thresh=0.9)
        h, w = bgr.shape[:2]
        r = self.imgsz / max(h, w)  # letterbox is top-left aligned
        out = []
        for m, s, b in zip(masks.numpy(), scores.numpy(), boxes.numpy()):
            m = m[: round(h * r), : round(w * r)]
            box = [float(b[0] / r / w), float(b[1] / r / h), float(b[2] / r / w), float(b[3] / r / h)]
            out.append(dict(source="sam", score=float(s), box=box, mask=Image.fromarray(m.astype(np.uint8) * 255).resize((w, h))))
        return out


def teacher_regions(model, im, conf=0.15):
    r = model.predict(im, imgsz=640, conf=conf, device="cpu", verbose=False, retina_masks=True)[0]
    out = []
    if r.boxes is None:
        return out
    ms = r.masks.data.numpy() if r.masks is not None else [None] * len(r.boxes)
    for b, s, c, m in zip(r.boxes.xyxyn.numpy(), r.boxes.conf.numpy(), r.boxes.cls.numpy().astype(int), ms):
        mask = Image.fromarray((m > 0.5).astype(np.uint8) * 255).resize(im.size) if m is not None else None
        out.append(dict(source="teacher", teacher_class=model.names[int(c)], score=float(s),
                        box=[float(v) for v in b], mask=mask))
    return out


def select_regions(teacher, sam, max_regions=45, min_area=0.0015, max_area=0.30, dup_iou=0.7):
    keep = sorted(teacher, key=lambda d: -d["score"])
    for d in sorted(sam, key=lambda d: -d["score"]):
        b = d["box"]
        a = (b[2] - b[0]) * (b[3] - b[1])
        if not (min_area <= a <= max_area):
            continue
        if any(box_iou(b, k["box"]) > dup_iou for k in keep):
            continue
        keep.append(d)
    return keep[:max_regions]


def label_point(mask: Image.Image | None, box, size):
    """A point well inside the region (distance-transform peak), falls back to the box centre."""
    W, H = size
    if mask is not None:
        import cv2
        m = (np.array(mask) > 127).astype(np.uint8)
        if m.sum() > 0:
            d = cv2.distanceTransform(m, cv2.DIST_L2, 5)
            y, x = np.unravel_index(int(d.argmax()), d.shape)
            return float(x), float(y)
    return (box[0] + box[2]) / 2 * W, (box[1] + box[3]) / 2 * H


def draw_marks(im: Image.Image, regions):
    import cv2
    canvas = im.copy()
    W, H = im.size
    arr = np.array(canvas)
    for i, rg in enumerate(regions):
        col = COLORS[i % len(COLORS)]
        if rg["mask"] is not None:
            m = (np.array(rg["mask"]) > 127).astype(np.uint8)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(arr, cnts, -1, col, 2)
        else:
            b = rg["box"]
            cv2.rectangle(arr, (int(b[0] * W), int(b[1] * H)), (int(b[2] * W), int(b[3] * H)), col, 2)
    canvas = Image.fromarray(arr)
    dr = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    for i, rg in enumerate(regions):
        x, y = rg["label_xy"]
        t = str(i + 1)
        r = 13
        dr.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255), outline=COLORS[i % len(COLORS)], width=3)
        tw = dr.textlength(t, font=font)
        dr.text((x - tw / 2, y - 10), t, fill=(0, 0, 0), font=font)
    return canvas


def cmd_propose(args):
    from ultralytics import YOLO
    names = yaml.safe_load((args.reviewed / "data_engine_bay_reviewed.yaml").read_text(encoding="utf-8"))["names"]
    images = pick_images(args.reviewed, args.labeled, ["val", "test"], args.limit)
    teacher = YOLO(str(args.teacher))
    sam = SamEverything(stride=args.stride)
    (OUT / "regions").mkdir(parents=True, exist_ok=True)
    (OUT / "marked").mkdir(parents=True, exist_ok=True)
    for k, (split, stem, path) in enumerate(images):
        jp = OUT / "regions" / f"{stem}.json"
        if jp.exists():
            continue
        im = open_view(path, args.max_side)
        regs = select_regions(teacher_regions(teacher, im), sam(im))
        for rg in regs:
            rg["label_xy"] = label_point(rg["mask"], rg["box"], im.size)
        draw_marks(im, regs).save(OUT / "marked" / f"{stem}.jpg", quality=90)
        jp.write_text(json.dumps({"split": split, "stem": stem, "size": im.size, "regions": [
            {"id": i + 1, **{k: v for k, v in rg.items() if k != "mask"}} for i, rg in enumerate(regs)]}), encoding="utf-8")
        print(f"[{k + 1}/{len(images)}] {stem}: {len(regs)} regions "
              f"({sum(r['source'] == 'teacher' for r in regs)} teacher)", flush=True)


# ---------------------------------------------------------------------------- label
def build_prompt(tax):
    lines = []
    for sysname, comps in tax.by_system().items():
        lines.append(f"[{tax.systems[sysname]['en']}]")
        for n in comps:
            c = tax.components[n]
            cue = "; ".join(c.cues[:2])
            look = ", ".join(c.confusers) if c.confusers else ""
            lines.append(f"- {n}: {cue}" + (f" | look-alikes: {look}" if look else ""))
    comp_list = "\n".join(lines)
    return f"""You are a senior automotive technician. The photo shows a vehicle engine bay. Coloured outlines with
white numbered circles mark candidate regions. Decide which numbered regions are one of the components listed below.

Rules:
- Name a region only when its outline covers mostly ONE listed component (the whole part, or its clearly visible portion).
- Give at least one visible cue (shape, colour, text/icon, what it connects to).
- For reservoirs, caps, boxes and sensors check the look-alikes and say what rules them out.
- Skip regions that are background, body panels, shadows, unnamed hoses/wires, or parts not in the list.
- If a listed component is clearly visible but has NO number, add it to "unmarked" with bbox_2d [x1,y1,x2,y2]
  in 0-1000 image coordinates.
- Never guess hidden parts.

Components (name: cues | look-alikes):
{comp_list}

Return JSON only:
{{"marks":[{{"id":<int>,"class":"<component name>","cue":"<visible cue>","ruled_out":"<look-alikes ruled out or empty>"}}],
 "unmarked":[{{"class":"<component name>","bbox_2d":[x1,y1,x2,y2],"cue":"<visible cue>"}}]}}"""


def get_client(args):
    from dotenv import dotenv_values, load_dotenv
    from openai import OpenAI
    load_dotenv(ROOT / ".env")
    key_env = args.key_env
    if not os.environ.get(key_env):
        found = [k for k in dotenv_values(ROOT / ".env") if re.search(r"deepseek.*key|key.*deepseek", k, re.I)]
        if len(found) != 1:
            sys.exit(f"{args.key_env} not set and {len(found)} *DEEPSEEK*KEY* variables in .env")
        key_env = found[0]
    return OpenAI(api_key=os.environ[key_env], base_url=os.environ.get(args.base_url_env) or args.base_url,
                  timeout=args.timeout, max_retries=2)


def cmd_label(args):
    tax = load_taxonomy()
    prompt = build_prompt(tax)
    client = get_client(args)
    raw = OUT / "raw" / re.sub(r"[^\w.-]", "_", args.model)
    raw.mkdir(parents=True, exist_ok=True)
    stems = sorted(p.stem for p in (OUT / "regions").glob("*.json"))

    def one(stem):
        cache = raw / f"{stem}.json"
        if cache.exists():
            return stem, "cached"
        buf = io.BytesIO()
        Image.open(OUT / "marked" / f"{stem}.jpg").save(buf, "JPEG", quality=88)
        uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        try:
            resp = client.chat.completions.create(
                model=args.model, temperature=0.0, max_tokens=3000,
                response_format={"type": "json_object"}, extra_body={"thinking": {"type": "disabled"}},
                messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": uri}},
                                                       {"type": "text", "text": prompt}]}])
            text = resp.choices[0].message.content or ""
            json.loads(text)
        except Exception as e:  # noqa: BLE001 - recorded, retried on the next run
            return stem, f"ERR {type(e).__name__}: {str(e)[:120]}"
        cache.write_text(json.dumps({"response": text}, ensure_ascii=False), encoding="utf-8")
        return stem, "ok"

    with ThreadPoolExecutor(args.workers) as pool:
        for i, (stem, status) in enumerate(pool.map(one, stems), 1):
            print(f"[{i}/{len(stems)}] {stem}: {status}", flush=True)


# ---------------------------------------------------------------------------- score
def som_predictions(stem, tax, model):
    reg = json.loads((OUT / "regions" / f"{stem}.json").read_text(encoding="utf-8"))
    rp = OUT / "raw" / re.sub(r"[^\w.-]", "_", model) / f"{stem}.json"
    if not rp.exists():
        return None, Counter()
    ans = json.loads(json.loads(rp.read_text(encoding="utf-8"))["response"])
    by_id = {r["id"]: r for r in reg["regions"]}
    preds, extra = [], Counter()
    for m in ans.get("marks", []):
        r = by_id.get(int(m.get("id", -1))) if str(m.get("id", "")).lstrip("-").isdigit() else None
        cls = str(m.get("class", "")).strip()
        if r is None or tax.component_for(cls) is None:
            continue
        preds.append({"class_name": cls, "box": r["box"], "source": r["source"]})
    for u in ans.get("unmarked", []):
        cls = str(u.get("class", "")).strip()
        b = u.get("bbox_2d")
        if tax.component_for(cls) is None or not (isinstance(b, list) and len(b) == 4):
            continue
        preds.append({"class_name": cls, "box": [float(v) / 1000 for v in b], "source": "unmarked"})
    for p in preds:
        comp = tax.component_for(p["class_name"])
        if tax.train_class_for(p["class_name"]) is None or comp.tier == "B":
            extra[p["class_name"]] += 1
    return preds, extra


def to_v2(dets, tax):
    out = []
    for d in dets:
        tc = tax.train_class_for(d["class_name"])
        if tc is not None:
            out.append({**d, "class_name": tc})
    return out


def cmd_score(args):
    from data_pipeline.qwen_grounding_annotator import parse_detections
    from ultralytics import YOLO
    tax = load_taxonomy()
    names = {int(k): v for k, v in yaml.safe_load((args.reviewed / "data_engine_bay_reviewed.yaml").read_text(encoding="utf-8"))["names"].items()}
    class_to_id = {v: k for k, v in names.items()}
    images = pick_images(args.reviewed, args.labeled, ["val", "test"], args.limit)
    teacher = YOLO(str(args.teacher))
    systems = {"Qwen3-VL (existing)": {}, "DeepSeek direct boxes (cached bake-off)": {},
               "teacher p5_reg alone (conf 0.25)": {}, f"SoM: teacher+SAM2 regions, {args.model} names": {}}
    gt, extra_all, src_tp = {}, Counter(), Counter()
    ds_raw = ROOT / "data" / "vlm_bakeoff" / "raw" / "deepseek-v4-flash-vision-exp__thinking_disabled"
    stems = []
    for split, stem, path in images:
        som, extra = som_predictions(stem, tax, args.model)
        dsp = ds_raw / f"{stem}.json"
        if som is None or not dsp.exists():
            continue
        stems.append(stem)
        extra_all.update(extra)
        gt[stem] = to_v2(load_gt(args.reviewed, split, stem, names), tax)
        systems["Qwen3-VL (existing)"][stem] = to_v2(load_qwen(args.labeled, split, stem), tax)
        rec = json.loads(dsp.read_text(encoding="utf-8"))
        try:
            dd, _ = parse_detections(rec["response"], class_to_id, rec["width"], rec["height"], 0.0, 0.0)
        except Exception:  # noqa: BLE001
            dd = []
        systems["DeepSeek direct boxes (cached bake-off)"][stem] = to_v2(
            [{"class_name": d["class_name"], "box": d["bbox_norm_xyxy"]} for d in dd], tax)
        r = teacher.predict(open_view(path, 1600), imgsz=640, conf=0.25, device="cpu", verbose=False)[0]
        systems["teacher p5_reg alone (conf 0.25)"][stem] = to_v2(
            [{"class_name": teacher.names[int(c)], "box": [float(v) for v in b]}
             for b, c in zip(r.boxes.xyxyn.numpy(), r.boxes.cls.numpy())], tax)
        systems[f"SoM: teacher+SAM2 regions, {args.model} names"][stem] = to_v2(som, tax)
    n_gt = sum(len(gt[s]) for s in stems)
    lines = ["# Set-of-Mark labelling pilot (W1b-c)", "",
             f"{len(stems)} bake-off images (val+test vehicles), {n_gt} reviewed objects, box IoU ≥ 0.5, "
             "all names mapped to taxonomy-v2 training classes (tier-B names -> generic class).", "",
             "| system | pred/img | precision | recall | F1 | class-agnostic P / R |", "|---|---|---|---|---|---|"]
    per_sys = {}
    for name, preds in systems.items():
        tot, per = Counter(), Counter()
        for s in stems:
            tp, fp, fn, pc = match(preds.get(s, []), gt[s], 0.5, True)
            atp, afp, afn, _ = match(preds.get(s, []), gt[s], 0.5, False)
            tot.update(tp=tp, fp=fp, fn=fn, atp=atp, afp=afp, afn=afn)
            per.update(pc)
        per_sys[name] = per
        p, r, f = prf(tot["tp"], tot["fp"], tot["fn"])
        ap_, ar_, _ = prf(tot["atp"], tot["afp"], tot["afn"])
        lines.append(f"| {name} | {(tot['tp'] + tot['fp']) / max(len(stems), 1):.1f} | {p:.1%} | {r:.1%} | {f:.3f} | {ap_:.1%} / {ar_:.1%} |")
    gcls = Counter(g["class_name"] for s in stems for g in gt[s])
    lines += ["", "## Per class F1 (≥ 5 reviewed objects)", "", "| class | GT | " + " | ".join(systems) + " |",
              "|---|---|" + "---|" * len(systems)]
    for c, n in gcls.most_common():
        if n >= 5:
            lines.append(f"| {c} | {n} | " + " | ".join(
                f"{prf(per_sys[m][(c, 'tp')], per_sys[m][(c, 'fp')], per_sys[m][(c, 'fn')])[2]:.2f}" for m in systems) + " |")
    lines += ["", "## Names outside the reviewed ontology (tier-B / new components the SoM run proposed; not scorable yet)", "",
              ", ".join(f"{k} ({v})" for k, v in extra_all.most_common()) or "none"]
    report = "\n".join(lines) + "\n"
    print(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["propose", "label", "score"])
    ap.add_argument("--reviewed", type=Path, default=ROOT / "data" / "engine_bay_reviewed")
    ap.add_argument("--labeled", type=Path, default=ROOT / "data" / "engine_bay_labeled")
    ap.add_argument("--teacher", type=Path, default=ROOT / "runs" / "segment" / "p5_reg" / "weights" / "avg5.pt")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--max-side", type=int, default=1400, help="same view size as the bake-off")
    ap.add_argument("--stride", type=int, default=16, help="SAM2 point grid per side")
    ap.add_argument("--model", default="deepseek-v4-flash-vision-exp")
    ap.add_argument("--base-url", default="https://api.deepseek.com/v1")
    ap.add_argument("--key-env", default="DEEPSEEK_API_KEY")
    ap.add_argument("--base-url-env", default="DEEPSEEK_BASE_URL")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "reports" / "som_pilot.md")
    args = ap.parse_args()
    {"propose": cmd_propose, "label": cmd_label, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
