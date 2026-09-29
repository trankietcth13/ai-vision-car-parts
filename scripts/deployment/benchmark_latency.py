"""
Roadmap phase 6: standard latency benchmark for the engine-bay student.

For each format (PyTorch .pt, ONNX, TensorRT .engine when TensorRT is installed):
    * exports the model if needed (ONNX opset 18 / TensorRT FP16, imgsz fixed)
    * warm-up (--warmup calls), then --iters timed calls on real images, cycling through --images
    * reports model-only time (Ultralytics speed['inference']) and the full pipeline
      (wall clock of predict(): decode already done, letterbox + inference + NMS + mask upsampling)
      as median / p90 / p99 / mean in ms

Output: <out>/latency.json and LATENCY.md

Usage (DGX, nothing else on the GPU):
    .venv/bin/python scripts/deployment/benchmark_latency.py --weights runs/train_kd/kd_n_v8_s0/weights/best.pt \
        --images data/engine_bay_train_v8/images/test --formats pt onnx engine --out qa_results/latency_kd_n_v8
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import statistics
import time
from pathlib import Path

import cv2
import numpy as np


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]


def stats(v):
    return {"median": round(statistics.median(v), 2), "p90": round(pct(v, 0.9), 2), "p99": round(pct(v, 0.99), 2),
            "mean": round(statistics.mean(v), 2), "n": len(v)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--formats", nargs="+", default=["pt", "onnx", "engine"])
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    ap.add_argument("--half", action="store_true", default=True)
    ap.add_argument("--out", default="qa_results/latency")
    a = ap.parse_args()
    import torch
    from ultralytics import YOLO

    imgs = [cv2.imread(str(p)) for p in sorted(Path(a.images).glob("*.jpg"))[:50]]
    imgs = [im for im in imgs if im is not None]
    if not imgs:
        raise SystemExit(f"no images in {a.images}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gpu = torch.cuda.is_available() and a.device != "cpu"
    results = {"weights": a.weights, "imgsz": a.imgsz, "iters": a.iters, "warmup": a.warmup,
               "input_images": len(imgs), "input_size_example": list(imgs[0].shape[:2]),
               "device": torch.cuda.get_device_name(0) if gpu else platform.processor() or "cpu",
               "torch": torch.__version__, "formats": {}}

    for fmt in a.formats:
        try:
            if fmt == "pt":
                path = a.weights
            elif fmt == "onnx":
                path = YOLO(a.weights).export(format="onnx", imgsz=a.imgsz, opset=18, simplify=True, dynamic=False)
            elif fmt == "engine":
                if not importlib.util.find_spec("tensorrt"):
                    results["formats"][fmt] = {"skipped": "tensorrt not installed on this machine"}
                    continue
                path = YOLO(a.weights).export(format="engine", imgsz=a.imgsz, half=a.half, device=a.device)
            else:
                results["formats"][fmt] = {"skipped": "unknown format"}
                continue
            model = YOLO(str(path), task="segment")
            dev = a.device if (gpu and fmt != "onnx") or fmt == "engine" else ("0" if gpu else "cpu")
            kw = dict(imgsz=a.imgsz, conf=a.conf, device=dev, verbose=False, half=(a.half and gpu and fmt == "pt"))
            for i in range(a.warmup):
                model.predict(imgs[i % len(imgs)], **kw)
            full, infer, pre, post = [], [], [], []
            for i in range(a.iters):
                im = imgs[i % len(imgs)]
                if gpu:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                r = model.predict(im, **kw)[0]
                if gpu:
                    torch.cuda.synchronize()
                full.append((time.perf_counter() - t0) * 1000)
                infer.append(r.speed["inference"])
                pre.append(r.speed["preprocess"])
                post.append(r.speed["postprocess"])
            size_mb = Path(path).stat().st_size / 2**20 if Path(path).is_file() else None
            results["formats"][fmt] = {"path": str(path), "size_mb": round(size_mb, 1) if size_mb else None,
                                       "device": dev, "half": kw["half"] or (fmt == "engine" and a.half),
                                       "model_only_ms": stats(infer), "preprocess_ms": stats(pre),
                                       "postprocess_ms": stats(post), "full_pipeline_ms": stats(full),
                                       "fps_full_pipeline": round(1000 / statistics.median(full), 1)}
            print(fmt, json.dumps(results["formats"][fmt]), flush=True)
        except Exception as exc:  # keep benchmarking the other formats
            results["formats"][fmt] = {"error": f"{type(exc).__name__}: {exc}"[:400]}
            print(fmt, "ERROR", exc, flush=True)

    (out / "latency.json").write_text(json.dumps(results, indent=2))
    md = [f"# Latency benchmark — {Path(a.weights).parent.parent.name}", "",
          f"Device {results['device']}, imgsz {a.imgsz}, {a.warmup} warm-up calls, {a.iters} timed calls on "
          f"{len(imgs)} real images (example input {imgs[0].shape[1]}x{imgs[0].shape[0]}).", "",
          "| format | size MB | model only (median / p90) | full pipeline (median / p90 / p99) | FPS (full) |",
          "|---|---|---|---|---|"]
    for fmt, r in results["formats"].items():
        if "model_only_ms" in r:
            mo, fp = r["model_only_ms"], r["full_pipeline_ms"]
            md.append(f"| {fmt} ({r['device']}{', fp16' if r['half'] else ''}) | {r['size_mb']} | {mo['median']} / {mo['p90']} ms | "
                      f"{fp['median']} / {fp['p90']} / {fp['p99']} ms | {r['fps_full_pipeline']} |")
        else:
            md.append(f"| {fmt} | - | {r.get('skipped') or r.get('error')} | | |")
    md += ["", "Model only = network forward (Ultralytics `speed['inference']`). Full pipeline = `predict()` wall clock: "
               "letterbox, forward, NMS, mask upsampling (JPEG decode excluded). TensorRT engines must be built on the target "
               "device itself; numbers from the DGX do not transfer to a Jetson/tablet."]
    (out / "LATENCY.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
