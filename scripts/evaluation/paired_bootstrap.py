"""Paired, image-level bootstrap comparison of segmentation models (mask mAP50-95 with a confidence interval).

Why: the QA gates compare 2-seed means against +0.01 thresholds, but the KD seed spread is 0.02-0.03 and the test set
has only 3 vehicles / ~125 images, so a +0.01 difference is noise-level. This tool resamples test IMAGES (same resample
for every model = paired) and reports the difference with a 95% interval. Seed noise is NOT in the interval: pass the
seeds as a group (their mean is compared) and read the per-seed values next to it.

  collect  (DGX, GPU)  run the Ultralytics validator once per model and save the per-image matching statistics
                       (tp_m for masks, tp for boxes, conf, pred_cls, target_cls) -> <out>/<name>.npz
  compare  (anywhere)  load the .npz files, keep the images and the class NAMES shared by all models, bootstrap.
                       Models may come from different datasets/taxonomies (e.g. v2 vs p5): classes are matched by name,
                       and the report says whether the ground truth of the shared images/classes is identical.

    python scripts/evaluation/paired_bootstrap.py collect --data data/engine_bay_full_v10/data_vehicle.yaml \
        --model p5_reg=runs/segment/p5_reg/weights/avg5.pt@640 --model v10=... --out eval_stats/test_v2
    python scripts/evaluation/paired_bootstrap.py compare --stats eval_stats/test_v2 \
        --group p5_kd=kd_n_p5t_s0,kd_n_p5t_s1 --group v10_kd=kd_n_v10_s0,kd_n_v10_s1 --pair v10_kd:p5_kd

Removing predictions/targets of a class never changes the matches of another class (Ultralytics only matches equal
classes), so restricting to shared classes from the saved statistics is exact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

KEYS = ("tp_m", "tp", "conf", "pred_cls", "target_cls")


# ----------------------------------------------------------------------------------------------------------- collect
def collect(a):
    from ultralytics import YOLO
    from ultralytics.models.yolo.segment import SegmentationValidator

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for spec in a.model:
        name, rest = spec.split("=", 1)
        path, imgsz = (rest.rsplit("@", 1) + ["640"])[:2]
        if not Path(path).exists():
            print(f"[collect] {name}: missing {path}")
            continue
        grabbed = {}

        class StatsValidator(SegmentationValidator):
            def update_metrics(self, preds, batch):
                super().update_metrics(preds, batch)
                self._names = getattr(self, "_names", []) + [Path(f).name for f in batch["im_file"][: len(preds)]]

            def get_stats(self):
                grabbed["stats"] = {k: list(v) for k, v in self.metrics.stats.items()}  # copied before clear_stats()
                grabbed["im_names"], grabbed["names"] = list(self._names), dict(self.names)
                return super().get_stats()

        r = YOLO(path).val(validator=StatsValidator, data=a.data, split=a.split, imgsz=int(imgsz), batch=a.batch,
                           device=a.device, plots=False, verbose=False, project=str(out / "val"), name=name, exist_ok=True)
        st = grabbed["stats"]
        assert len(grabbed["im_names"]) == len(st["conf"]), "per-image stats and image names are out of step"
        arrays = {"im_names": np.array(grabbed["im_names"]),
                  "names": np.array([grabbed["names"][i] for i in range(len(grabbed["names"]))]),
                  "qa_mask_map50_95": np.array(float(r.seg.map))}
        for k in KEYS:
            arrays[f"{k}_len"] = np.array([len(x) for x in st[k]])
            arrays[k] = np.concatenate(st[k], 0) if st[k] else np.zeros(0)
        np.savez_compressed(out / f"{name}.npz", **arrays)
        print(f"[collect] {name}: {len(grabbed['im_names'])} images, mask mAP50-95 {float(r.seg.map):.3f} -> {out / name}.npz")


# ----------------------------------------------------------------------------------------------------------- compare
class ModelStats:
    """Per-image statistics of one model, re-indexed to a common image order and class-name space."""

    def __init__(self, path: Path):
        z = np.load(path, allow_pickle=False)
        self.name, self.names = path.stem, [str(x) for x in z["names"]]
        self.qa_map = float(z["qa_mask_map50_95"])
        self.images = {}
        offs = {k: np.concatenate([[0], np.cumsum(z[f"{k}_len"])]) for k in KEYS}
        for i, im in enumerate(z["im_names"]):
            self.images[str(im)] = {k: z[k][offs[k][i]:offs[k][i + 1]] for k in KEYS}

    def restricted(self, images, classes):
        """Per-image stats limited to `classes` (names), class ids replaced by their index in `classes`."""
        lut = np.full(len(self.names), -1)
        for j, c in enumerate(classes):
            if c in self.names:
                lut[self.names.index(c)] = j
        out = []
        for im in images:
            s = self.images[im]
            pc, tc = lut[s["pred_cls"].astype(int)], lut[s["target_cls"].astype(int)]
            kp, kt = pc >= 0, tc >= 0
            out.append({"tp_m": s["tp_m"][kp], "conf": s["conf"][kp], "pred_cls": pc[kp], "target_cls": tc[kt]})
        return out


def mask_map(per_image, idx):
    from ultralytics.utils.metrics import ap_per_class

    tp = np.concatenate([per_image[i]["tp_m"] for i in idx], 0)
    tc = np.concatenate([per_image[i]["target_cls"] for i in idx], 0)
    if len(tc) == 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0
    conf = np.concatenate([per_image[i]["conf"] for i in idx], 0)
    pc = np.concatenate([per_image[i]["pred_cls"] for i in idx], 0)
    ap = ap_per_class(tp, conf, pc, tc)[5]  # (classes present in targets, 10 IoU thresholds)
    return float(ap.mean()) if len(ap) else 0.0


def compare(a):
    files = sorted(Path(a.stats).glob("*.npz")) if a.stats else []
    files += [Path(f) for f in a.file or []]
    models = {m.name: m for m in (ModelStats(f) for f in files)}
    groups = {}
    for g in a.group or []:
        gname, members = g.split("=", 1)
        groups[gname] = members.split(",")
    for m in models:
        groups.setdefault(m, [m])
    missing = sorted({m for ms in groups.values() for m in ms} - set(models))
    groups = {g: ms for g, ms in groups.items() if not set(ms) & set(missing)}
    pair_groups = {g for p in a.pair for g in p.split(":")}
    used = sorted({m for g in pair_groups if g in groups for m in groups[g]}) or sorted(models)

    images = sorted(set.intersection(*(set(models[m].images) for m in used)))
    classes = sorted(set.intersection(*(set(models[m].names) for m in used)))
    if a.classes:
        classes = [c for c in classes if c in a.classes.split(",")]
    per = {m: models[m].restricted(images, classes) for m in used}

    # ground truth agreement on the shared images/classes (same labels -> exact paired comparison)
    ref = used[0]
    gt_diff = {m: sum(1 for i in range(len(images))
                      if sorted(per[m][i]["target_cls"].tolist()) != sorted(per[ref][i]["target_cls"].tolist()))
               for m in used[1:]}

    rng = np.random.default_rng(a.seed)
    n = len(images)
    boots = [rng.integers(0, n, n) for _ in range(a.n_boot)]
    full = np.arange(n)
    point = {m: mask_map(per[m], full) for m in used}
    samples = {m: np.array([mask_map(per[m], b) for b in boots]) for m in used}

    def gmean(g, d):
        return np.mean([d[m] for m in groups[g]], axis=0)

    L = [f"# Paired image-bootstrap comparison ({a.n_boot} resamples)", "",
         f"- images shared by all compared models: **{n}**; classes compared ({len(classes)}): {', '.join(classes)}",
         f"- ground truth differs from `{ref}` on: " + (", ".join(f"{m} {k} images" for m, k in gt_diff.items()) or "-")
         + " (0 = identical labels, the comparison is exactly paired)",
         "- interval = 2.5-97.5 percentile over resampled test images; seed noise is NOT included (see per-seed values)",
         "", "| model | mask mAP50-95 (these images/classes) | 95% interval | QA report value (all classes) |", "|---|---|---|---|"]
    for m in used:
        lo, hi = np.percentile(samples[m], [2.5, 97.5])
        L.append(f"| {m} | {point[m]:.3f} | {lo:.3f} - {hi:.3f} | {models[m].qa_map:.3f} |")
    L += ["", "| comparison | difference | 95% interval | P(difference > 0) | per-member values | verdict |", "|---|---|---|---|---|---|"]
    result = []
    for p in a.pair:
        ga, gb = p.split(":")
        if ga not in groups or gb not in groups:
            L.append(f"| {ga} - {gb} | missing models ({', '.join(missing)}) | | | | |")
            continue
        d = gmean(ga, samples) - gmean(gb, samples)
        dp = gmean(ga, point) - gmean(gb, point)
        lo, hi = np.percentile(d, [2.5, 97.5])
        pgt = float((d > 0).mean())
        verdict = "better" if lo > a.margin else ("worse" if hi < -a.margin else "no detectable difference")
        members = "; ".join(f"{g}: " + ", ".join(f"{point[m]:.3f}" for m in groups[g]) for g in (ga, gb))
        L.append(f"| {ga} - {gb} | {dp:+.3f} | {lo:+.3f} .. {hi:+.3f} | {pgt:.2f} | {members} | {verdict} |")
        result.append({"a": ga, "b": gb, "diff": dp, "ci": [lo, hi], "p_gt0": pgt, "verdict": verdict})
    L += ["", f"Verdict rule: 'better' when the whole interval is above +{a.margin}, 'worse' when below -{a.margin}."]
    text = "\n".join(L) + "\n"
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
        Path(a.out).with_suffix(".json").write_text(json.dumps(
            {"images": n, "classes": classes, "gt_diff": gt_diff, "point": point, "pairs": result}, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--model", action="append", required=True, help="name=path.pt@imgsz (repeatable)")
    c.add_argument("--data", required=True)
    c.add_argument("--split", default="test")
    c.add_argument("--batch", type=int, default=8)
    c.add_argument("--device", default="0")
    c.add_argument("--out", required=True)
    m = sub.add_parser("compare")
    m.add_argument("--stats", help="folder of .npz files from collect")
    m.add_argument("--file", action="append", help="extra .npz files (e.g. a model collected on another dataset)")
    m.add_argument("--group", action="append", help="name=model1,model2 (seeds averaged)")
    m.add_argument("--pair", action="append", required=True, help="groupA:groupB (reports A - B)")
    m.add_argument("--classes", default=None, help="comma list to restrict the classes further")
    m.add_argument("--n-boot", type=int, default=1000)
    m.add_argument("--margin", type=float, default=0.0)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--out", default=None, help="markdown report path (.json written next to it)")
    a = ap.parse_args()
    collect(a) if a.cmd == "collect" else compare(a)


if __name__ == "__main__":
    main()
