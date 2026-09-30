"""W1e: learn a label-confidence model from past review verdicts (instead of hand-set weights).

Every machine-proposed box that an expert reviewed becomes one training row:
    y = 1  class right    (verdict correct | bad_geometry)
    y = 0  class wrong    (wrong_class | not_a_component | duplicate)
Features are pixel/geometry signals only (text rationales carry no signal, see memory
text-only-label-audit-fails): VLM confidence and source, claimed class, box size/shape/border,
position-prior density and size outlier score, duplicate/overlap statistics, and DINOv2 crop
evidence (similarity to reviewed-correct crops of the claimed class vs to rejected crops, computed
leave-vehicle-out).  Teacher agreement is intentionally NOT used yet: the available teachers were
trained on these images (leak); add it with the phase-3 CV fold teachers.

Evaluation: GroupKFold by vehicle; AUROC and "auto-accept coverage at >= 95 % precision" versus the
VLM confidence alone.

    python scripts/data_pipeline/label_confidence.py \
        --review data/engine_bay_review data/engine_bay_review_hybrid data/engine_bay_review_p2 \
        --priors artifacts/priors/position_priors_v2.json --out docs/reports/label_confidence_v2.md
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
from data_pipeline.position_priors import PositionPriors  # noqa: E402
from data_pipeline.taxonomy import load_taxonomy  # noqa: E402
from evaluation.retrieval_probe import Embedder, crop  # noqa: E402

POS = {"correct", "bad_geometry"}
NEG = {"wrong_class", "not_a_component", "duplicate"}


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def load_rows(review_roots, tax):
    rows = []
    for root in review_roots:
        source = "deepseek" if ("hybrid" in root.name or "p2" in root.name) else "qwen"
        for vp in sorted(root.glob("verdicts/*/*.json")):
            split, stem = vp.parent.name, vp.stem
            pp = root / "packets" / split / f"{stem}.json"
            img = root / "packets" / split / f"{stem}_clean.jpg"
            if not pp.exists() or not img.exists():
                continue
            v = json.loads(vp.read_text(encoding="utf-8"))
            p = json.loads(pp.read_text(encoding="utf-8"))
            if v.get("exclude_from_training"):
                continue
            insts = {i["id"]: i for i in p["instances"]}
            boxes = [(i["class_name"], i["box_norm"]) for i in p["instances"]]
            for it in v.get("instances", []):
                d = it.get("decision")
                if d not in POS | NEG or it["id"] not in insts:
                    continue
                inst = insts[it["id"]]
                cls = inst["class_name"]
                b = [float(x) for x in inst["box_norm"]]
                if b[2] <= b[0] or b[3] <= b[1]:
                    continue
                same = [bb for c, bb in boxes if c == cls and bb is not inst["box_norm"]]
                other = [bb for c, bb in boxes if c != cls]
                rows.append(dict(
                    key=f"{root.name}/{split}/{stem}#{it['id']}", img=str(img), vehicle=stem.split("__")[0],
                    source=source, cls=cls, train_cls=tax.train_class_for(cls) or "ignored", box=b,
                    conf=float(inst.get("confidence") or 0.5), y=int(d in POS), decision=d,
                    n_same=len(same), max_iou_same=max([iou(b, s) for s in same], default=0.0),
                    max_iou_other=max([iou(b, o) for o in other], default=0.0), n_boxes=len(boxes)))
    return rows


def embed_rows(rows, cache: Path, model_id):
    cache.parent.mkdir(parents=True, exist_ok=True)
    keys = [r["key"] for r in rows]
    fpath, kpath = cache.with_suffix(".npy"), cache.with_suffix(".keys.json")
    if fpath.exists() and kpath.exists() and json.loads(kpath.read_text(encoding="utf-8")) == keys:
        return np.load(fpath)  # plain float array, no pickle
    emb = Embedder(model_id)
    by_img = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_img[r["img"]].append(i)
    feats = None
    for n, (path, ids) in enumerate(by_img.items()):
        from inference.teacher_system import open_upright
        im = open_upright(path)
        f = emb([crop(im, rows[i]["box"]) for i in ids])
        if feats is None:
            feats = np.zeros((len(rows), f.shape[1]), np.float32)
        feats[ids] = f
        if n % 200 == 0:
            print(f"  embedded {n}/{len(by_img)} images", flush=True)
    np.save(fpath, feats)
    kpath.write_text(json.dumps(keys), encoding="utf-8")
    return feats


def crop_evidence(rows, feats, k=5):
    """Leave-vehicle-out: mean top-k cosine to accepted crops of the same claimed class and to rejected crops."""
    veh = np.array([r["vehicle"] for r in rows])
    cls = np.array([r["cls"] for r in rows])
    y = np.array([r["y"] for r in rows])
    s_pos, s_neg, s_any = np.zeros(len(rows)), np.zeros(len(rows)), np.zeros(len(rows))
    sims = feats @ feats.T
    for i in range(len(rows)):
        other = veh != veh[i]
        pos = other & (cls == cls[i]) & (y == 1)
        neg = other & (y == 0)
        acc = other & (y == 1)
        def topk(m):
            s = sims[i, m]
            return float(np.sort(s)[-k:].mean()) if s.size else 0.0
        s_pos[i], s_neg[i], s_any[i] = topk(pos), topk(neg), topk(acc)
    return s_pos, s_neg, s_any


def build_matrix(rows, feats, priors, tax):
    s_pos, s_neg, s_any = crop_evidence(rows, feats)
    classes = sorted({r["cls"] for r in rows})
    X, names = [], []
    for i, r in enumerate(rows):
        b = r["box"]
        w, h = b[2] - b[0], b[3] - b[1]
        sq = float(np.sqrt(w * h))
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        tc = r["train_cls"]
        f = {
            "vlm_conf": r["conf"], "src_deepseek": float(r["source"] == "deepseek"),
            "log_size": np.log(sq + 1e-6), "aspect": np.log((w + 1e-6) / (h + 1e-6)),
            "border": float(min(b[0], b[1], 1 - b[2], 1 - b[3]) < 0.01),
            "prior_density": np.log(priors.score(tc, cx, cy)) if tc != "ignored" else 0.0,
            "size_z": priors.size_z(tc, sq) if tc != "ignored" else 0.0,
            "n_same": r["n_same"], "max_iou_same": r["max_iou_same"], "max_iou_other": r["max_iou_other"],
            "n_boxes": r["n_boxes"],
            "crop_sim_claimed": s_pos[i], "crop_sim_rejected": s_neg[i], "crop_margin": s_pos[i] - s_neg[i],
            "crop_sim_any_accepted": s_any[i],
        }
        for c in classes:
            f[f"cls={c}"] = float(r["cls"] == c)
        if not names:
            names = list(f)
        X.append([f[n] for n in names])
    return np.asarray(X, float), names


def coverage_at_precision(y, p, target=0.95):
    order = np.argsort(-p)
    tp = np.cumsum(y[order])
    prec = tp / np.arange(1, len(y) + 1)
    ok = np.where(prec >= target)[0]
    if not len(ok):
        return 0.0, None
    k = ok.max()
    return (k + 1) / len(y), float(p[order][k])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--review", nargs="+", type=Path, required=True)
    ap.add_argument("--priors", type=Path, default=Path("artifacts/priors/position_priors_v2.json"))
    ap.add_argument("--model", default="facebook/dinov2-small")
    ap.add_argument("--cache", type=Path, default=Path("artifacts/retrieval/verdict_crops"))
    ap.add_argument("--out", type=Path, default=Path("docs/reports/label_confidence_v2.md"))
    ap.add_argument("--save-model", type=Path, default=Path("artifacts/label_confidence/model_v2.joblib"))
    args = ap.parse_args()

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    tax = load_taxonomy()
    rows = load_rows(args.review, tax)
    print(f"{len(rows)} reviewed machine boxes, positives {sum(r['y'] for r in rows)}")
    feats = embed_rows(rows, args.cache, args.model)
    X, names = build_matrix(rows, feats, PositionPriors.load(args.priors), tax)
    y = np.array([r["y"] for r in rows])
    groups = np.array([r["vehicle"] for r in rows])

    models = {
        "vlm_conf only": None,
        "logistic (all features)": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5)),
        "gradient boosting (all features)": lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0),
    }
    no_crop = [i for i, n in enumerate(names) if not n.startswith("crop_")]
    models["gradient boosting (no crop evidence)"] = ("nocrop", lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0))

    gkf = GroupKFold(n_splits=5)
    lines = ["# Label-confidence model from review verdicts (W1e)", "",
             f"{len(rows)} reviewed machine boxes from {len(set(groups))} vehicles "
             f"({int(y.sum())} class-right, {int((1 - y).sum())} class-wrong). 5-fold GroupKFold by vehicle.", "",
             "| model | AUROC | auto-accept coverage @ precision ≥ 0.95 | @ ≥ 0.90 |", "|---|---|---|---|"]
    oof_best = None
    for name, spec in models.items():
        oof = np.zeros(len(y))
        if spec is None:
            oof = X[:, names.index("vlm_conf")]
        else:
            cols = slice(None)
            if isinstance(spec, tuple):
                cols, spec = no_crop, spec[1]
            for tr, te in gkf.split(X, y, groups):
                m = spec().fit(X[tr][:, cols], y[tr])
                oof[te] = m.predict_proba(X[te][:, cols])[:, 1]
        auc = roc_auc_score(y, oof)
        c95, _ = coverage_at_precision(y, oof, 0.95)
        c90, _ = coverage_at_precision(y, oof, 0.90)
        lines.append(f"| {name} | {auc:.3f} | {c95:.1%} | {c90:.1%} |")
        if name == "gradient boosting (all features)":
            oof_best = oof

    # per-source and per-decision view of the best model
    src = np.array([r["source"] for r in rows])
    dec = np.array([r["decision"] for r in rows])
    lines += ["", "## Best model (gradient boosting, all features): out-of-fold detail", ""]
    for s in sorted(set(src)):
        m = src == s
        lines.append(f"- source {s}: n={int(m.sum())}, base precision {y[m].mean():.3f}, AUROC {roc_auc_score(y[m], oof_best[m]):.3f}")
    lines.append("- mean score by verdict: " + ", ".join(f"{d} {oof_best[dec == d].mean():.2f}" for d in sorted(set(dec))))
    for thr in (0.9, 0.8, 0.6, 0.4):
        m = oof_best >= thr
        if m.any():
            lines.append(f"- score ≥ {thr}: {m.mean():.1%} of boxes, precision {y[m].mean():.3f}")

    final = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0).fit(X, y)
    from sklearn.inspection import permutation_importance
    imp = permutation_importance(final, X, y, n_repeats=5, random_state=0, scoring="roc_auc")
    top = np.argsort(-imp.importances_mean)[:10]
    lines += ["", "Top features (permutation importance, AUROC drop, in-sample): " +
              ", ".join(f"{names[i]} {imp.importances_mean[i]:.3f}" for i in top)]
    report = "\n".join(lines) + "\n"
    print(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    # joblib (pickle) is only written here, for our own later use on this machine; never load a model file
    # from an untrusted source.
    try:
        import joblib
        args.save_model.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": final, "features": names}, args.save_model)
    except Exception as e:  # noqa: BLE001
        print(f"model not saved: {e}")


if __name__ == "__main__":
    main()
