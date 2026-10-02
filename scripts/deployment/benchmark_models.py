"""End-to-end latency of several models on the same photos, measured like the web app benchmark.

Per image: downscale to 1600 px (as the app), 2 warm-up runs at that letterbox shape, then --iters timed runs of
model.predict (batch 1, fp32, conf 0.35). Reports wall-clock median / mean / p90 and Ultralytics' preprocess /
inference / postprocess split (postprocess = NMS + masks; YOLO26 has no NMS).

    python scripts/deployment/benchmark_models.py --model kd=runs/.../avg5.pt --model teacher=... \
        --images apps/engine_bay_web/examples --device 0 --out bench_gpu.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

MAX_SIDE = 1600


def downscale(img: np.ndarray) -> np.ndarray:
    s = MAX_SIDE / max(img.shape[:2])
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img


def load(path: str) -> YOLO:
    if path.endswith(".onnx"):  # an export's file name rarely says -seg; all engine-bay models segment
        return YOLO(path, task="segment")
    return YOLO(path)


def bench(path: str, images: list[tuple[str, np.ndarray]], device: str, iters: int, conf: float, imgsz: int) -> dict:
    model = load(path)
    kw = dict(imgsz=imgsz, conf=conf, device=device, verbose=False)
    lat, parts, dets = [], {"preprocess": [], "inference": [], "postprocess": []}, 0
    for _, img in images:
        for _ in range(2):
            model.predict(img, **kw)
        for _ in range(iters):
            t0 = time.perf_counter()
            r = model.predict(img, **kw)[0]
            lat.append((time.perf_counter() - t0) * 1000)
            for k in parts:
                parts[k].append((r.speed or {}).get(k) or 0.0)
        dets += len(r.boxes) if r.boxes is not None else 0
    lat_sorted = sorted(lat)
    params = None
    if not path.endswith(".onnx"):
        params = round(sum(p.numel() for p in model.model.parameters()) / 1e6, 2)
    return {"weights": path, "format": Path(path).suffix[1:], "params_M": params,
            "median_ms": round(statistics.median(lat), 1), "mean_ms": round(statistics.fmean(lat), 1),
            "p90_ms": round(lat_sorted[int(0.9 * (len(lat_sorted) - 1))], 1),
            **{f"{k}_ms": round(statistics.median(v), 2) for k, v in parts.items()},
            "detections": dets, "runs": len(lat)}


def cpu_name() -> str:
    try:
        if platform.system() == "Windows":
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def environment(device: str) -> dict:
    env = {"torch": torch.__version__, "python": platform.python_version(), "platform": platform.platform()}
    import ultralytics

    env["ultralytics"] = ultralytics.__version__
    if device != "cpu" and torch.cuda.is_available():
        env["device"] = torch.cuda.get_device_name(int(device))
    else:
        env["device"] = f"CPU {cpu_name()} ({torch.get_num_threads()} threads)"
    return env


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, help="name=path (.pt or .onnx), repeatable")
    ap.add_argument("--images", default="apps/engine_bay_web/examples")
    ap.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    images = [(p.name, downscale(cv2.imread(str(p)))) for p in sorted(Path(a.images).glob("*.jpg"))]
    if not images:
        raise SystemExit(f"no .jpg in {a.images}")
    out = {"device_arg": a.device, "environment": environment(a.device), "images": [n for n, _ in images],
           "iters_per_image": a.iters, "conf": a.conf, "imgsz": a.imgsz, "models": {}}
    for spec in a.model:
        name, path = spec.split("=", 1)
        out["models"][name] = bench(path, images, a.device, a.iters, a.conf, a.imgsz)
        m = out["models"][name]
        print(f"[bench] {name}: median {m['median_ms']} ms (pre {m['preprocess_ms']} / inf {m['inference_ms']} / "
              f"post {m['postprocess_ms']}), {m['detections']} dets", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[bench] wrote {a.out}")


if __name__ == "__main__":
    main()
