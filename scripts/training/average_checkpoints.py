"""
Roadmap phase 5: average the weights of the best epoch checkpoints of one Ultralytics run
(val mAP swings about ±4 points between epochs, so a single "best.pt" is a noisy pick).

Needs a run trained with save_period (weights/epoch{N}.pt). The top-k epochs by --metric in results.csv
that have a checkpoint are averaged (EMA weights, float32, BatchNorm running stats averaged too) and saved
as a normal Ultralytics checkpoint that YOLO() / val / export accept.

Usage (DGX):
    .venv/bin/python scripts/training/average_checkpoints.py --run runs/segment/p5_sgd --top-k 5 --out runs/segment/p5_sgd/weights/avg5.pt
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--metric", default="metrics/mAP50-95(M)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--last-k", action="store_true",
                    help="average the LAST top-k saved epochs instead of the best by --metric (no use of the val split: "
                         "needed for cross-validation, where val is the fold being evaluated)")
    ap.add_argument("--strip-kd", action="store_true",
                    help="KD student runs: epoch checkpoints still hold KD adapters; strip them from the averaged model")
    a = ap.parse_args()

    rows = list(csv.DictReader((a.run / "results.csv").open()))
    rows = [{k.strip(): v for k, v in r.items()} for r in rows]
    wdir = a.run / "weights"
    cands = []
    for r in rows:
        e = int(float(r["epoch"]))
        # Ultralytics names periodic checkpoints by the 0-based epoch index (epoch0, epoch5, ...);
        # results.csv is 1-based, so csv epoch e <-> epoch{e-1}.pt. No fallback: it would pair two rows with one file.
        f = wdir / f"epoch{e - 1}.pt"
        if f.exists():
            cands.append((float(r[a.metric]), e, f))
    if not cands:
        raise SystemExit(f"no epoch checkpoints in {wdir} (train with save_period>0)")
    top = sorted(cands, key=lambda x: -x[1])[: a.top_k] if a.last_k else sorted(cands, key=lambda x: -x[0])[: a.top_k]
    print("averaging:", [(e, round(m, 4), f.name) for m, e, f in top])

    base = torch.load(top[0][2], map_location="cpu", weights_only=False)
    key = "ema" if base.get("ema") is not None else "model"
    model = base[key].float()
    sd = {k: v.clone().float() if v.is_floating_point() else v.clone() for k, v in model.state_dict().items()}
    for _, _, f in top[1:]:
        other = torch.load(f, map_location="cpu", weights_only=False)
        osd = (other.get("ema") or other["model"]).float().state_dict()
        for k, v in sd.items():
            if v.is_floating_point():
                v += osd[k].float()
    for k, v in sd.items():
        if v.is_floating_point():
            v /= len(top)
    model.load_state_dict(sd)

    ckpt = copy.copy(base)
    ckpt[key] = model.half()
    if key == "ema":
        ckpt["model"] = None
    ckpt["optimizer"] = None
    ckpt["epoch"] = -1
    ckpt["averaged_from"] = [str(f) for _, _, f in top]
    out = a.out or wdir / f"avg{len(top)}.pt"
    torch.save(ckpt, out)
    if a.strip_kd:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
        from distillation.ultralytics_kd import strip_kd_checkpoint

        print("stripped KD modules:", strip_kd_checkpoint(out))
    print(json.dumps({"out": str(out), "epochs": [e for _, e, _ in top],
                      "metric_values": [round(m, 4) for m, _, _ in top]}))


if __name__ == "__main__":
    main()
