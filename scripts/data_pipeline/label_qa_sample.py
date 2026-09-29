"""
Roadmap phase 2, step 5: human spot check of the final labels (target: label precision and recall >= 0.9).

make   picks --n images (default 50, spread over vehicles, optionally only some sources) from a YOLO-seg
       folder, draws every instance with a number, and writes a self-contained review page:
           <out>/index.html   open locally; per instance tick "sai" (wrong class / not a component / bad mask),
                              per image enter how many clearly visible target components have NO label,
                              then press "Tải kết quả (CSV)" -> label_qa_results.csv
           <out>/images/*.jpg, <out>/sample.json
score  reads the CSV and reports label precision = correct / labelled and
       recall = correct / (correct + missing), overall and per class, against --target.

Usage:
    python scripts/data_pipeline/label_qa_sample.py make --data data/engine_bay_full --split dataset --out qa_results/label_qa_p2
    python scripts/data_pipeline/label_qa_sample.py score --out qa_results/label_qa_p2 --csv ~/Downloads/label_qa_results.csv
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
PALETTE = [(56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207), (10, 249, 72), (23, 204, 146),
           (134, 219, 61), (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0), (147, 69, 52), (255, 115, 100),
           (236, 24, 0), (255, 56, 132), (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255), (80, 80, 80)]


def cmd_make(a):
    data = Path(a.data)
    cfg = yaml.safe_load(Path(a.names).read_text(encoding="utf-8"))
    names = {int(k): v for k, v in cfg["names"].items()}
    lbl_dir, img_dir = data / "labels" / a.split, data / "images" / a.split
    stems = sorted(p.stem for p in lbl_dir.glob("*.txt"))
    if a.only_stems:
        keep = set(Path(a.only_stems).read_text().split())
        stems = [s for s in stems if s in keep]
    rng = random.Random(a.seed)
    by_v = defaultdict(list)
    for s in stems:
        by_v[s.split("__")[0]].append(s)
    for v in by_v.values():
        rng.shuffle(v)
    pick, vs = [], sorted(by_v)
    while len(pick) < min(a.n, len(stems)):  # round robin over vehicles
        for v in vs:
            if by_v[v] and len(pick) < a.n:
                pick.append(by_v[v].pop())
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    cards, sample = [], []
    for stem in sorted(pick):
        img = cv2.imread(str(next(img_dir.glob(f"{stem}.*"))))
        h, w = img.shape[:2]
        f = min(1.0, 1400 / max(h, w))
        img = cv2.resize(img, (round(w * f), round(h * f))) if f < 1 else img
        h, w = img.shape[:2]
        over, inst = img.copy(), []
        for i, line in enumerate((lbl_dir / f"{stem}.txt").read_text().splitlines()):
            s = line.split()
            if len(s) < 7:
                continue
            c = int(s[0])
            pts = (np.array(s[1:], float).reshape(-1, 2) * [w, h]).astype(np.int32)
            col = PALETTE[c % len(PALETTE)]
            cv2.fillPoly(over, [pts], col)
            inst.append({"id": len(inst), "cls": names[c], "pts": pts, "col": col})
        out_img = cv2.addWeighted(over, 0.3, img, 0.7, 0)
        for it in inst:
            cv2.polylines(out_img, [it["pts"]], True, it["col"], 2)
            x, y = it["pts"].min(0)
            t = f"#{it['id']} {it['cls']}"
            cv2.putText(out_img, t, (int(x) + 2, int(y) + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(out_img, t, (int(x) + 2, int(y) + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imwrite(str(out / "images" / f"{stem}.jpg"), out_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        cv2.imwrite(str(out / "images" / f"{stem}_clean.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        sample.append({"stem": stem, "instances": [{"id": it["id"], "class": it["cls"]} for it in inst]})
        rows = "".join(f'<label><input type="checkbox" data-stem="{stem}" data-id="{it["id"]}" data-cls="{it["cls"]}"> '
                       f'sai &nbsp;<b>#{it["id"]}</b> {html.escape(it["cls"])}</label>' for it in inst) or "<i>không có nhãn</i>"
        cards.append(f'<section class="card" data-stem="{stem}"><h2>{html.escape(stem)}</h2>'
                     f'<div class="pics"><img loading="lazy" src="images/{stem}.jpg" alt="nhãn">'
                     f'<img loading="lazy" src="images/{stem}_clean.jpg" alt="ảnh gốc"></div>'
                     f'<div class="inst">{rows}</div><label class="miss">Số linh kiện rõ ràng bị thiếu nhãn: '
                     f'<input type="number" min="0" value="0" data-miss="{stem}"></label>'
                     f'<input class="note" placeholder="ghi chú (tuỳ chọn)" data-note="{stem}"></section>')
    (out / "sample.json").write_text(json.dumps(sample, indent=1))
    page = f"""<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Label QA Sample</title><style>
:root{{--bg:#f5f6f8;--fg:#1c2230;--card:#fff;--muted:#5d6676;--acc:#2458d6}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#12151b;--fg:#e5e8ee;--card:#1b2029;--muted:#9aa3b2;--acc:#7ea2ff}}}}
body{{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif}}
header{{position:sticky;top:0;background:var(--bg);padding:8px 0 12px;z-index:2}}
h1{{font-size:20px;margin:0}} p{{color:var(--muted);margin:4px 0}} h2{{font-size:13px;margin:0 0 6px;word-break:break-all}}
.card{{background:var(--card);border-radius:10px;padding:12px;margin:0 0 14px}}
.pics{{display:grid;grid-template-columns:1fr 1fr;gap:8px}} .pics img{{width:100%;border-radius:6px}}
@media (max-width:700px){{.pics{{grid-template-columns:1fr}}}}
.inst{{display:flex;flex-wrap:wrap;gap:6px 18px;margin:8px 0}} .miss{{display:block;margin:6px 0}}
.miss input{{width:60px}} .note{{width:100%;box-sizing:border-box;padding:4px}}
button{{background:var(--acc);color:#fff;border:0;border-radius:6px;padding:8px 14px;font-size:14px;cursor:pointer}}
</style></head><body><header><h1>Kiểm tra chất lượng nhãn — {len(pick)} ảnh</h1>
<p>Mỗi nhãn: đánh dấu <b>sai</b> nếu sai loại, không phải linh kiện, hoặc mask lệch rõ. Mỗi ảnh: nhập số linh kiện nhìn rõ mà chưa có nhãn.
Ảnh trái có nhãn, ảnh phải là ảnh gốc. Xong thì bấm nút và gửi file CSV.</p>
<button id="dl">Tải kết quả (CSV)</button> <span id="st"></span></header>
{''.join(cards)}
<script>
document.getElementById('dl').onclick=()=>{{
  const rows=[['stem','instance_id','class','wrong','missing_in_image','note']];
  document.querySelectorAll('section.card').forEach(c=>{{
    const s=c.dataset.stem, m=c.querySelector('[data-miss]').value||'0', n=c.querySelector('[data-note]').value.replace(/[\\n,]/g,' ');
    const boxes=c.querySelectorAll('input[type=checkbox]');
    if(!boxes.length) rows.push([s,'','',0,m,n]);
    boxes.forEach((b,i)=>rows.push([s,b.dataset.id,b.dataset.cls,b.checked?1:0,i===0?m:'',i===0?n:'']));
  }});
  const blob=new Blob([rows.map(r=>r.join(',')).join('\\n')],{{type:'text/csv'}});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='label_qa_results.csv';a.click();
  document.getElementById('st').textContent='Đã tải label_qa_results.csv';
}};
</script></body></html>"""
    (out / "index.html").write_text(page, encoding="utf-8")
    print(json.dumps({"images": len(pick), "instances": sum(len(s["instances"]) for s in sample),
                      "vehicles": len({s.split('__')[0] for s in pick}), "page": str(out / "index.html")}))


def cmd_score(a):
    rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
    labelled, wrong, missing = Counter(), Counter(), 0
    for r in rows:
        if r["class"]:
            labelled[r["class"]] += 1
            wrong[r["class"]] += int(r["wrong"] or 0)
        if r["missing_in_image"]:
            missing += int(r["missing_in_image"])
    n, nw = sum(labelled.values()), sum(wrong.values())
    correct = n - nw
    prec = correct / n if n else 1.0
    rec = correct / (correct + missing) if correct + missing else 1.0
    res = {"images": len({r["stem"] for r in rows}), "instances": n, "wrong": nw, "missing": missing,
           "label_precision": round(prec, 3), "label_recall": round(rec, 3), "target": a.target,
           "pass": prec >= a.target and rec >= a.target,
           "per_class_precision": {c: round((labelled[c] - wrong[c]) / labelled[c], 3) for c in labelled}}
    out = Path(a.out)
    (out / "LABEL_QA.json").write_text(json.dumps(res, indent=2))
    md = ["# Label QA (human spot check)", "", f"- Images {res['images']}, labelled instances {n}",
          f"- Label precision **{prec:.3f}**, recall **{rec:.3f}** (target >= {a.target}): "
          f"**{'PASS' if res['pass'] else 'FAIL'}**", "", "| class | labelled | wrong | precision |", "|---|---|---|---|"]
    md += [f"| {c} | {labelled[c]} | {wrong[c]} | {res['per_class_precision'][c]} |" for c in sorted(labelled, key=lambda c: -labelled[c])]
    (out / "LABEL_QA.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(res, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--data", required=True, help="YOLO-seg root with images/<split>, labels/<split>")
    m.add_argument("--split", default="dataset")
    m.add_argument("--names", default=str(ROOT / "configs" / "engine_bay_train_classes.yaml"))
    m.add_argument("--only-stems", default=None, help="text file of stems to sample from (e.g. the Phase 2 candidates)")
    m.add_argument("--n", type=int, default=50)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--csv", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--target", type=float, default=0.9)
    a = ap.parse_args()
    {"make": cmd_make, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
