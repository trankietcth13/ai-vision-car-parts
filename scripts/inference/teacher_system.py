"""W4 teacher system: whole-bay analysis grouped by vehicle system (server / offline, not the edge student).

    image ─► detector on the full image + an overlapping 2x2 tile grid ─► drop tile boxes cut by inner tile
          borders ─► class-wise NMS ─► per-class count cap (priors count_max + 1) ─► optional per-class thresholds
          ─► JSON grouped by system (taxonomy v2 names, VI/EN) ─► optional self-contained HTML viewer

Design: docs/plans/teacher_all_components_workflow_2026-09-29.md (W4). SAM2 mask refinement and retrieval
naming of other_* classes plug in later; they are not needed to produce the grouped output.

    # analyse images, write JSON + HTML viewers
    python scripts/inference/teacher_system.py analyze --weights runs/segment/p5_reg/weights/avg5.pt \
        --images data/ref/jeep.jpg --out artifacts/teacher_system/jeep
    # measure what tiling / caps change on a labelled split (box IoU >= 0.5, class match)
    python scripts/inference/teacher_system.py eval --weights runs/segment/p5_reg/weights/avg5.pt \
        --data data/engine_bay_reviewed --split test --out docs/reports/teacher_system_eval.md
"""

from __future__ import annotations

import argparse
import base64
import collections
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_pipeline.position_priors import PositionPriors  # noqa: E402
from data_pipeline.taxonomy import load_taxonomy  # noqa: E402
from data_pipeline.taxonomy_stats import read_names  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRIORS = ROOT / "artifacts" / "priors" / "position_priors_v2.json"


_EXIF_TRANSPOSE = {2: Image.FLIP_LEFT_RIGHT, 3: Image.ROTATE_180, 4: Image.FLIP_TOP_BOTTOM, 5: Image.TRANSPOSE,
                   6: Image.ROTATE_270, 7: Image.TRANSVERSE, 8: Image.ROTATE_90}


def open_upright(path, draft_side: int | None = None) -> Image.Image:
    """RGB image in its EXIF-upright orientation, as cv2.imread / the label pipeline sees it.
    (PIL and Ultralytics' PIL input path ignore the EXIF orientation tag; ~2% of the photos carry one.)
    `draft_side` lets the JPEG decoder downscale on load (fast for 6000 px originals)."""
    im = Image.open(path)
    orientation = im.getexif().get(274, 1)
    if draft_side and im.format == "JPEG":
        im.draft("RGB", (draft_side, draft_side))
    im = im.convert("RGB")
    if orientation in _EXIF_TRANSPOSE:
        im = im.transpose(_EXIF_TRANSPOSE[orientation])
    return im


def load_image(path, max_side=1600) -> Image.Image:
    """Open like the training data (stored at 1600 px long side, EXIF-upright)."""
    im = open_upright(path, draft_side=max_side)
    f = max_side / max(im.size)
    if f < 1:
        im = im.resize((round(im.size[0] * f), round(im.size[1] * f)), Image.BILINEAR)
    return im


class TeacherSystem:
    def __init__(self, weights, imgsz=640, conf=0.25, iou=0.5, tiles=(2, 2), overlap=0.25, cap_extra=1,
                 priors: Path | None = DEFAULT_PRIORS, class_conf: dict | None = None, device="cpu", band=(0.6, 0.35),
                 agnostic_iou: float | None = 0.85):
        from ultralytics import YOLO
        self.model = YOLO(str(weights))
        self.names = self.model.names
        self.tax = load_taxonomy()
        self.imgsz, self.conf, self.iou, self.device = imgsz, conf, iou, device
        self.tiles, self.overlap, self.cap_extra = tiles, overlap, cap_extra
        self.priors = PositionPriors.load(priors) if priors and Path(priors).exists() else None
        self.class_conf = class_conf or {}
        self.band = band
        self.agnostic_iou = agnostic_iou

    # ------------------------------------------------------------------ detection
    def _run(self, im: Image.Image, x0=0.0, y0=0.0, W=None, H=None, inner=None):
        """Detections of one pass in global normalised coordinates. `inner` = (left, top, right, bottom) flags of
        tile edges that lie inside the image: boxes touching them are dropped (likely truncated objects)."""
        W, H = W or im.size[0], H or im.size[1]
        r = self.model.predict(im, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)[0]
        out = []
        if r.boxes is None or not len(r.boxes):
            return out
        tw, th = im.size
        polys = r.masks.xyn if r.masks is not None else [None] * len(r.boxes)
        for b, s, c, poly in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int), polys):
            if inner is not None:
                l, t, rr, bb = inner
                if (l and b[0] < 3) or (t and b[1] < 3) or (rr and b[2] > tw - 3) or (bb and b[3] > th - 3):
                    continue
            box = [float((b[0] + x0) / W), float((b[1] + y0) / H), float((b[2] + x0) / W), float((b[3] + y0) / H)]
            gp = None
            if poly is not None and len(poly):
                gp = np.stack([(poly[:, 0] * tw + x0) / W, (poly[:, 1] * th + y0) / H], 1)
            out.append(dict(cls=self.names[int(c)], score=float(s), box=box, poly=gp))
        return out

    def raw(self, im: Image.Image, use_tiles=True):
        """Model passes only: (full-image detections, tile detections)."""
        W, H = im.size
        full, tiled = self._run(im), []
        if use_tiles and self.tiles:
            gx, gy = self.tiles
            tw, th = W / (gx - (gx - 1) * self.overlap), H / (gy - (gy - 1) * self.overlap)
            for i in range(gx):
                for j in range(gy):
                    x0, y0 = i * tw * (1 - self.overlap), j * th * (1 - self.overlap)
                    tile = im.crop((int(x0), int(y0), int(min(x0 + tw, W)), int(min(y0 + th, H))))
                    tiled += self._run(tile, int(x0), int(y0), W, H, inner=(i > 0, j > 0, i < gx - 1, j < gy - 1))
        return full, tiled

    def detect(self, im: Image.Image, use_tiles=True, use_cap=True, use_agnostic=True, raw=None):
        full, tiled = raw if raw is not None else self.raw(im, use_tiles)
        return self.postprocess(full + (tiled if use_tiles else []), use_cap, use_agnostic)

    def postprocess(self, dets, use_cap=True, use_agnostic=True):
        dets = [d for d in dets if d["score"] >= self.class_conf.get(d["cls"], self.conf)]
        if not dets:
            return []
        boxes = torch.tensor([d["box"] for d in dets], dtype=torch.float32)
        scores = torch.tensor([d["score"] for d in dets])
        # small contiguous ids: batched_nms offsets boxes by id * max_coord, large ids destroy float32 precision
        order = {c: i for i, c in enumerate(sorted({d["cls"] for d in dets}))}
        cls_ids = torch.tensor([order[d["cls"]] for d in dets])
        keep = torchvision.ops.batched_nms(boxes, scores, cls_ids, self.iou).tolist()
        dets = [dets[k] for k in keep]
        if use_agnostic and self.agnostic_iou and dets:
            # one physical object, two class names (e.g. brake vs washer reservoir on the same box): keep the stronger
            b = torch.tensor([d["box"] for d in dets], dtype=torch.float32)
            s = torch.tensor([d["score"] for d in dets])
            dets = [dets[k] for k in torchvision.ops.nms(b, s, self.agnostic_iou).tolist()]
        if use_cap and self.priors:
            seen = collections.Counter()
            capped = []
            for d in sorted(dets, key=lambda d: -d["score"]):
                c = self.priors.data["classes"].get(d["cls"], {}).get("count_max")
                seen[d["cls"]] += 1
                if c is None or seen[d["cls"]] <= c + self.cap_extra:
                    capped.append(d)
            dets = capped
        return dets

    # ------------------------------------------------------------------ grouped output
    def analyze(self, path, use_tiles=True, use_cap=True):
        im = load_image(path)
        dets = self.detect(im, use_tiles, use_cap)
        systems = {}
        hi, mid = self.band
        for n, d in enumerate(sorted(dets, key=lambda d: -d["score"])):
            comp = self.tax.component_for(d["cls"])
            sysname = comp.system if comp else ("generic" if d["cls"] in self.tax.generic else "unknown")
            card = self.tax.generic.get(d["cls"], {}) if comp is None else {"vi": comp.vi, "en": comp.en}
            entry = {
                "id": n, "class": d["cls"], "name_vi": card.get("vi", d["cls"]), "name_en": card.get("en", d["cls"]),
                "score": round(d["score"], 3),
                "band": "high" if d["score"] >= hi else ("medium" if d["score"] >= mid else "low"),
                "box_norm": [round(v, 4) for v in d["box"]],
                "polygon_norm": None if d["poly"] is None
                else np.round(d["poly"][:: max(1, len(d["poly"]) // 60)].astype(float), 4).tolist(),
            }
            safety = self.tax.safety_of(d["cls"])
            if safety:
                entry["safety"] = safety
            if self.priors:
                cx, cy = (d["box"][0] + d["box"][2]) / 2, (d["box"][1] + d["box"][3]) / 2
                entry["position_prior"] = round(self.priors.score(d["cls"], cx, cy), 2)
            meta = self.tax.systems.get(sysname, {"vi": sysname, "en": sysname})
            systems.setdefault(sysname, {"vi": meta["vi"], "en": meta["en"], "components": []})["components"].append(entry)
        return im, {
            "image": str(path), "width": im.size[0], "height": im.size[1],
            "model": str(self.model.ckpt_path if hasattr(self.model, "ckpt_path") else ""),
            "settings": {"imgsz": self.imgsz, "conf": self.conf, "tiles": list(self.tiles) if use_tiles else None,
                         "count_cap": use_cap},
            "n_components": len(dets),
            "warnings": sorted({f"{e['safety']}: {e['class']} detected - identify only, do not guide disassembly; refer to a qualified technician"
                                for g in systems.values() for e in g["components"] if e.get("safety")}),
            "systems": systems,
        }


# ---------------------------------------------------------------------- HTML viewer
PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#ca8a04", "#db2777",
           "#4f46e5", "#059669", "#b91c1c", "#7c3aed", "#0d9488", "#64748b"]


def render_html(im: Image.Image, result: dict) -> str:
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    W, H = im.size
    colors = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(result["systems"])}
    # "</" escaped so no string in the data can close the <script> element
    data = json.dumps({"W": W, "H": H, "systems": result["systems"], "colors": colors}, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Engine bay by system</title>
<style>
:root{{--bg:#f8fafc;--fg:#0f172a;--muted:#64748b;--card:#fff;--line:#e2e8f0}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0b1120;--fg:#e2e8f0;--muted:#94a3b8;--card:#111827;--line:#1f2937}}}}
body{{margin:0;font:14px/1.45 system-ui,sans-serif;background:var(--bg);color:var(--fg)}}
header{{padding:12px 16px;border-bottom:1px solid var(--line)}} header small{{color:var(--muted)}}
main{{display:grid;grid-template-columns:minmax(0,3fr) minmax(260px,1fr);gap:16px;padding:16px}}
@media (max-width:900px){{main{{grid-template-columns:1fr}}}}
.stage{{position:relative}} .stage img{{width:100%;display:block;border-radius:8px}}
.stage svg{{position:absolute;inset:0;width:100%;height:100%}}
polygon,rect{{fill-opacity:.18;stroke-width:3;vector-effect:non-scaling-stroke;cursor:pointer}}
.dim{{opacity:.08}} .hot{{fill-opacity:.45}}
.chips{{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px}}
.chip{{border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:999px;padding:3px 10px;cursor:pointer}}
.chip.on{{border-color:currentColor;font-weight:600}}
.sys{{background:var(--card);border:1px solid var(--line);border-radius:8px;margin-bottom:10px;padding:8px 10px}}
.sys h3{{margin:0 0 6px;font-size:14px}} .item{{display:flex;justify-content:space-between;gap:8px;padding:2px 4px;border-radius:4px;cursor:pointer}}
.item:hover{{background:var(--line)}} .sc{{color:var(--muted);font-variant-numeric:tabular-nums}}
.low{{opacity:.6}}
</style></head><body>
<header><b>Phân rã khoang máy theo hệ thống</b> <small>— {result['n_components']} linh kiện · teacher system (full + tiles) · điểm thấp = cần kiểm tra</small></header>
<main><div><div class="chips" id="chips"></div><div class="stage"><img src="data:image/jpeg;base64,{b64}" alt="engine bay">
<svg id="svg" viewBox="0 0 {W} {H}" preserveAspectRatio="none"></svg></div></div><aside id="list"></aside></main>
<script>
const D={data};let active=null;
const svg=document.getElementById('svg'),list=document.getElementById('list'),chips=document.getElementById('chips');
const ns='http://www.w3.org/2000/svg';const shapes={{}};
function draw(){{svg.innerHTML='';for(const[s,g]of Object.entries(D.systems)){{for(const c of g.components){{let el;
 if(c.polygon_norm){{el=document.createElementNS(ns,'polygon');el.setAttribute('points',c.polygon_norm.map(p=>(p[0]*D.W)+','+(p[1]*D.H)).join(' '));}}
 else{{const b=c.box_norm;el=document.createElementNS(ns,'rect');el.setAttribute('x',b[0]*D.W);el.setAttribute('y',b[1]*D.H);el.setAttribute('width',(b[2]-b[0])*D.W);el.setAttribute('height',(b[3]-b[1])*D.H);}}
 el.setAttribute('fill',D.colors[s]);el.setAttribute('stroke',D.colors[s]);el.dataset.sys=s;el.dataset.id=c.id;
 const t=document.createElementNS(ns,'title');t.textContent=c.name_vi+' ('+c.class+') '+c.score;el.appendChild(t);svg.appendChild(el);shapes[c.id]=el;}}}}}}
function apply(){{for(const el of svg.children){{el.classList.toggle('dim',!!active&&el.dataset.sys!==active);}}
 for(const ch of chips.children)ch.classList.toggle('on',ch.dataset.sys===active);}}
function hot(id,on){{const el=shapes[id];if(el)el.classList.toggle('hot',on);}}
for(const[s,g]of Object.entries(D.systems)){{const ch=document.createElement('button');ch.className='chip';ch.dataset.sys=s;ch.style.color=D.colors[s];
 ch.textContent=g.vi+' ('+g.components.length+')';ch.onclick=()=>{{active=active===s?null:s;apply();}};chips.appendChild(ch);
 const box=document.createElement('div');box.className='sys';
 const h=document.createElement('h3');h.style.color=D.colors[s];h.textContent=g.vi+' ';const sm=document.createElement('small');sm.className='sc';sm.textContent=g.en;h.appendChild(sm);box.appendChild(h);
 for(const c of g.components){{const it=document.createElement('div');it.className='item'+(c.band==='low'?' low':'');
  const a=document.createElement('span');a.textContent=c.name_vi;const b=document.createElement('span');b.className='sc';b.textContent=c.score.toFixed(2);
  it.append(a,b);it.onmouseenter=()=>hot(c.id,true);it.onmouseleave=()=>hot(c.id,false);box.appendChild(it);}}
 list.appendChild(box);}}
draw();apply();
</script></body></html>"""


# ---------------------------------------------------------------------- evaluation
def _iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def evaluate(ts: TeacherSystem, data: Path, split: str, out: Path):
    tax = ts.tax
    names = read_names(data)
    known = set(ts.names.values())
    modes = {  # (tiles, count cap, class-agnostic merge)
        "full image": (False, False, False),
        "full + agnostic merge": (False, False, True),
        "full + 2x2 tiles": (True, False, False),
        "tiles + agnostic merge": (True, False, True),
        "tiles + agnostic + count cap": (True, True, True),
    }
    res = {m: dict(tp=0, n=0, npred=0, sys_tp=collections.Counter(), sys_n=collections.Counter()) for m in modes}
    imgs = sorted((data / "images" / split).glob("*.jp*g"))
    for k, ip in enumerate(imgs):
        lp = data / "labels" / split / (ip.stem + ".txt")
        gts = []
        for line in (lp.read_text().splitlines() if lp.exists() else []):
            p = line.split()
            if len(p) < 5:
                continue
            tc = tax.train_class_for(names[int(p[0])])
            if tc is None or tc not in known:  # only classes this detector can output
                continue
            xy = np.asarray(p[1:], float).reshape(-1, 2)
            gts.append((tc, [xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()]))
        im = load_image(ip)
        raw = ts.raw(im, use_tiles=True)  # one set of model passes shared by every mode
        for m, (tiles, cap, agn) in modes.items():
            dets = ts.detect(im, tiles, cap, agn, raw=raw)
            r = res[m]
            r["npred"] += len(dets)
            used = set()
            for d in sorted(dets, key=lambda d: -d["score"]):
                best, bj = 0.5, -1
                for j, (c, g) in enumerate(gts):
                    if j not in used and c == d["cls"]:
                        v = _iou(d["box"], g)
                        if v >= best:
                            best, bj = v, j
                if bj >= 0:
                    used.add(bj)
            for j, (c, _) in enumerate(gts):
                s = tax.system_of(c) or "generic"
                r["n"] += 1
                r["sys_n"][s] += 1
                if j in used:
                    r["tp"] += 1
                    r["sys_tp"][s] += 1
        if k % 25 == 0:
            print(f"  {k}/{len(imgs)}", flush=True)
    lines = ["# Teacher system evaluation (W4)", "",
             f"Weights `{ts.model.ckpt_path}` · {data}/{split} ({len(imgs)} images) · imgsz {ts.imgsz} · conf {ts.conf} · "
             f"box IoU ≥ 0.5 with class match · only GT classes the detector knows", "",
             "| mode | recall | precision | predictions |", "|---|---|---|---|"]
    for m, r in res.items():
        lines.append(f"| {m} | {r['tp'] / max(r['n'], 1):.3f} | {r['tp'] / max(r['npred'], 1):.3f} | {r['npred']} |")
    systems = sorted(res["full image"]["sys_n"], key=lambda s: -res["full image"]["sys_n"][s])
    lines += ["", "Recall by system:", "", "| system | n | " + " | ".join(modes) + " |", "|---|---|" + "---|" * len(modes)]
    for s in systems:
        n = res["full image"]["sys_n"][s]
        lines.append(f"| {s} | {n} | " + " | ".join(f"{res[m]['sys_tp'][s] / max(n, 1):.2f}" for m in modes) + " |")
    report = "\n".join(lines) + "\n"
    print(report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("analyze", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--weights", required=True)
        p.add_argument("--imgsz", type=int, default=640)
        p.add_argument("--conf", type=float, default=0.25)
        p.add_argument("--device", default="cpu")
        p.add_argument("--class-conf", type=Path, default=None, help="yaml {class: threshold} from cv_eval.py")
    a_ = sub.choices["analyze"]
    a_.add_argument("--images", nargs="+", required=True, type=Path)
    a_.add_argument("--out", type=Path, required=True)
    a_.add_argument("--no-tiles", action="store_true")
    a_.add_argument("--no-html", action="store_true")
    e_ = sub.choices["eval"]
    e_.add_argument("--data", type=Path, required=True)
    e_.add_argument("--split", default="test")
    e_.add_argument("--out", type=Path, default=Path("docs/reports/teacher_system_eval.md"))
    args = ap.parse_args()

    class_conf = None
    if args.class_conf:
        import yaml
        cc = yaml.safe_load(args.class_conf.read_text(encoding="utf-8"))
        class_conf = cc.get("thresholds", cc)
    ts = TeacherSystem(args.weights, imgsz=args.imgsz, conf=args.conf, device=args.device, class_conf=class_conf)
    if args.cmd == "eval":
        evaluate(ts, args.data, args.split, args.out)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    for ip in args.images:
        im, result = ts.analyze(ip, use_tiles=not args.no_tiles)
        (args.out / f"{ip.stem}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        if not args.no_html:
            (args.out / f"{ip.stem}.html").write_text(render_html(im, result), encoding="utf-8")
        counts = {s: len(g["components"]) for s, g in result["systems"].items()}
        print(f"{ip.name}: {result['n_components']} components {counts}")


if __name__ == "__main__":
    main()
