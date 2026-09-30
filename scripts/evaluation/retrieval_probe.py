"""W1d probe: can DINOv2 crop embeddings name engine-bay components (and fix the teacher's class errors)?

1. Crops every reviewed instance (box + padding) from the original-resolution images and embeds it
   with DINOv2 (CLS token, L2-normalised). Cached in --cache.
2. Gallery = train split, queries = val + test split (different vehicles by construction).
   Classifiers: cosine kNN (k, similarity-weighted) and a logistic-regression linear probe.
3. Optional --teacher: runs the detector on the query images, matches predictions to GT boxes
   class-agnostically (IoU >= 0.5) and reports how often retrieval corrects the teacher's class.

Classes are taxonomy-v2 training classes. CPU is fine (~3.5k crops).

    python scripts/evaluation/retrieval_probe.py --src data/engine_bay_reviewed data/engine_bay_reviewed_hybrid \
        --teacher runs/segment/p5_reg/weights/avg5.pt --out docs/reports/retrieval_probe_v2.md
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_pipeline.taxonomy import load_taxonomy  # noqa: E402
from data_pipeline.taxonomy_stats import read_names  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
GROUPS = {
    "reservoirs": ["coolant_reservoir", "brake_fluid_reservoir", "washer_fluid_reservoir", "other_reservoir"],
    "caps_small": ["oil_filler_cap", "radiator_cap", "battery_terminal", "oil_dipstick"],
    "boxes": ["battery", "fuse_relay_box", "ecu_module", "air_filter_box", "engine_cover", "intake_manifold"],
}


def find_image(src: Path, split: str, stem: str):
    for ext in (".jpg", ".jpeg", ".png", ".JPG"):
        p = src / "images" / split / (stem + ext)
        if p.exists():
            return p
    return None


def collect_instances(srcs, tax):
    """-> list of dict(stem, split, path, cls, box_norm) for every mapped instance."""
    out, seen = [], set()
    for src in srcs:
        names = read_names(src)
        for split in ("train", "val", "test"):
            for lp in sorted((src / "labels" / split).glob("*.txt")):
                if lp.stem in seen:
                    continue
                seen.add(lp.stem)
                ip = find_image(src, split, lp.stem)
                if ip is None:
                    continue
                for line in lp.read_text().splitlines():
                    p = line.split()
                    if len(p) < 5:
                        continue
                    tc = tax.train_class_for(names[int(p[0])])
                    if tc is None:
                        continue
                    xy = np.asarray(p[1:], float).reshape(-1, 2)
                    out.append(dict(stem=lp.stem, split=split, path=str(ip), cls=tc,
                                    box=[xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()]))
    return out


def crop(im: Image.Image, box_norm, pad=0.15, size=224):
    W, H = im.size
    x1, y1, x2, y2 = box_norm[0] * W, box_norm[1] * H, box_norm[2] * W, box_norm[3] * H
    w, h = x2 - x1, y2 - y1
    side = max(w, h) * (1 + 2 * pad)  # square crop keeps aspect for the ViT
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    box = (int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0)), int(min(cx + side / 2, W)), int(min(cy + side / 2, H)))
    return im.crop(box).convert("RGB").resize((size, size), Image.BICUBIC)


class Embedder:
    def __init__(self, model_id="facebook/dinov2-small", device="cpu"):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.torch = torch
        self.proc = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).eval().to(device)
        self.device = device

    def __call__(self, crops):
        with self.torch.no_grad():
            x = self.proc(images=crops, return_tensors="pt", do_center_crop=False,
                          size={"height": 224, "width": 224}).to(self.device)
            f = self.model(**x).last_hidden_state[:, 0]
            return self.torch.nn.functional.normalize(f, dim=-1).cpu().numpy()


def embed_all(insts, embedder, batch=32):
    by_img = collections.defaultdict(list)
    for i, d in enumerate(insts):
        by_img[d["path"]].append(i)
    feats = np.zeros((len(insts), 0), np.float32)
    buf, idx, chunks = [], [], []
    for n, (path, ids) in enumerate(by_img.items()):
        from inference.teacher_system import open_upright
        im = open_upright(path, draft_side=3000)  # EXIF-upright like the labels; fast 1/2-res JPEG decode
        for i in ids:
            buf.append(crop(im, insts[i]["box"]))
            idx.append(i)
        if len(buf) >= batch or n == len(by_img) - 1:
            chunks.append((list(idx), embedder(buf)))
            buf, idx = [], []
        if n % 100 == 0:
            print(f"  embedded images {n}/{len(by_img)}", flush=True)
    dim = chunks[0][1].shape[1]
    feats = np.zeros((len(insts), dim), np.float32)
    for ids, f in chunks:
        feats[ids] = f
    return feats


def knn_predict(gal_f, gal_y, q_f, k=10):
    sims = q_f @ gal_f.T
    top = np.argsort(-sims, axis=1)[:, :k]
    preds, conf = [], []
    for r, row in enumerate(top):
        votes = collections.Counter()
        for j in row:
            votes[gal_y[j]] += float(np.exp(sims[r, j] / 0.07))
        lab, v = votes.most_common(1)[0]
        preds.append(lab)
        conf.append(v / sum(votes.values()))
    return np.array(preds), np.array(conf)


def teacher_matches(weights, query_insts, imgsz=640, conf=0.1):
    """For each query instance: class predicted by the teacher on a class-agnostic IoU>=0.5 match (or None)."""
    from ultralytics import YOLO
    model = YOLO(weights)
    names = model.names
    by_img = collections.defaultdict(list)
    for qi, d in enumerate(query_insts):
        by_img[d["path"]].append(qi)
    out = [None] * len(query_insts)
    for n, (path, ids) in enumerate(by_img.items()):
        r = model.predict(path, imgsz=imgsz, conf=conf, device="cpu", verbose=False)[0]
        pb = r.boxes.xyxyn.cpu().numpy()
        pc = r.boxes.cls.cpu().numpy().astype(int)
        ps = r.boxes.conf.cpu().numpy()
        used = set()
        for qi in sorted(ids, key=lambda i: -(query_insts[i]["box"][2] - query_insts[i]["box"][0])):
            g = query_insts[qi]["box"]
            best, bj = 0.5, -1
            for j, b in enumerate(pb):
                if j in used:
                    continue
                x1, y1, x2, y2 = max(g[0], b[0]), max(g[1], b[1]), min(g[2], b[2]), min(g[3], b[3])
                inter = max(0, x2 - x1) * max(0, y2 - y1)
                iou = inter / ((g[2] - g[0]) * (g[3] - g[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)
                if iou >= best:
                    best, bj = iou, j
            if bj >= 0:
                used.add(bj)
                out[qi] = (names[int(pc[bj])], float(ps[bj]))
        if n % 25 == 0:
            print(f"  teacher images {n}/{len(by_img)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", nargs="+", required=True, type=Path)
    ap.add_argument("--model", default="facebook/dinov2-small")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--teacher", default=None, help="detector weights to compare against (optional)")
    ap.add_argument("--cache", type=Path, default=Path("artifacts/retrieval"))
    ap.add_argument("--out", type=Path, default=Path("docs/reports/retrieval_probe_v2.md"))
    args = ap.parse_args()

    tax = load_taxonomy()
    insts = collect_instances(args.src, tax)
    args.cache.mkdir(parents=True, exist_ok=True)
    tag = args.model.split("/")[-1]
    fpath, mpath = args.cache / f"feats_{tag}.npy", args.cache / f"insts_{tag}.json"
    if fpath.exists() and mpath.exists() and json.loads(mpath.read_text()) == insts:
        feats = np.load(fpath)
        print(f"loaded cached embeddings {feats.shape}")
    else:
        print(f"embedding {len(insts)} crops with {args.model} (CPU)")
        feats = embed_all(insts, Embedder(args.model))
        np.save(fpath, feats)
        mpath.write_text(json.dumps(insts))

    y = np.array([d["cls"] for d in insts])
    split = np.array([d["split"] for d in insts])
    g, q = split == "train", split != "train"
    knn_y, knn_c = knn_predict(feats[g], y[g], feats[q], k=args.k)

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=3000, C=2.0, class_weight="balanced").fit(feats[g], y[g])
    lin_y = clf.predict(feats[q])
    yq = y[q]

    lines = ["# DINOv2 crop retrieval probe (W1d)", "",
             f"Model `{args.model}` · gallery = train split ({int(g.sum())} crops) · queries = val+test "
             f"({int(q.sum())} crops, other vehicles) · classes = taxonomy-v2 training classes", "",
             f"**Overall top-1:** kNN (k={args.k}) {np.mean(knn_y == yq):.3f} · linear probe {np.mean(lin_y == yq):.3f}", "",
             "| class | n query | kNN acc | linear acc | most confused with (linear) |", "|---|---|---|---|---|"]
    for cls in tax.training_names:
        m = yq == cls
        if not m.any():
            continue
        wrong = collections.Counter(lin_y[m & (lin_y != yq)])
        conf_with = ", ".join(f"{k} ({v})" for k, v in wrong.most_common(2)) or "—"
        lines.append(f"| {cls} | {int(m.sum())} | {np.mean(knn_y[m] == cls):.2f} | {np.mean(lin_y[m] == cls):.2f} | {conf_with} |")
    for gname, members in GROUPS.items():
        m = np.isin(yq, members)
        if m.any():
            lines.append("")
            lines.append(f"**{gname}** (n={int(m.sum())}): kNN {np.mean(knn_y[m] == yq[m]):.3f} · linear {np.mean(lin_y[m] == yq[m]):.3f}")

    if args.teacher:
        qinsts = [d for d, s in zip(insts, q) if s]
        tm = teacher_matches(args.teacher, qinsts)
        matched = np.array([t is not None for t in tm])
        t_cls = np.array([t[0] if t else "" for t in tm])
        t_conf = np.array([t[1] if t else 0.0 for t in tm])
        t_ok = matched & (t_cls == yq)
        t_wrong = matched & (t_cls != yq)
        fix = t_wrong & (lin_y == yq)
        break_ = t_ok & (lin_y != yq)
        lines += ["", "## Against the teacher (class-agnostic IoU ≥ 0.5 matches)", "",
                  f"Teacher `{args.teacher}` · query instances matched: {int(matched.sum())}/{len(qinsts)}", "",
                  f"- teacher class correct: {int(t_ok.sum())} · wrong class: {int(t_wrong.sum())}",
                  f"- linear probe fixes {int(fix.sum())}/{int(t_wrong.sum())} teacher class errors, "
                  f"but would break {int(break_.sum())}/{int(t_ok.sum())} correct ones if it always overrode",
                  ]
        # simple fusion: override only when the teacher is unsure and the probe is confident
        proba = clf.predict_proba(feats[q])
        p_max = proba.max(1)
        for t_thr, p_thr in [(0.5, 0.6), (0.4, 0.7), (0.3, 0.8)]:
            override = matched & (t_conf < t_thr) & (p_max >= p_thr) & (lin_y != t_cls)
            fused = np.where(override, lin_y, t_cls)
            lines.append(f"- fusion (override if teacher conf < {t_thr} and probe p ≥ {p_thr}): "
                         f"class acc on matched {np.mean(fused[matched] == yq[matched]):.3f} "
                         f"vs teacher {np.mean(t_cls[matched] == yq[matched]):.3f} ({int(override.sum())} overrides)")
        tw = collections.Counter(zip(yq[t_wrong], t_cls[t_wrong]))
        lines += ["", "Teacher class errors (GT → predicted): " + ", ".join(f"{a}→{b} ({n})" for (a, b), n in tw.most_common(10))]

    report = "\n".join(lines) + "\n"
    print(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
