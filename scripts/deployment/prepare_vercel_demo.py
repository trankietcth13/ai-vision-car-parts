"""Build the static Vercel demo (apps/engine_bay_vercel): the POC student runs IN THE BROWSER with onnxruntime-web.

Why static + in-browser: the Python app (torch + ultralytics + gradio) does not fit a Vercel Function, while the
yolo11n-seg student is 11 MB as ONNX and runs in the browser (WASM, multi-threaded when cross-origin isolated).
Photos never leave the device. The ONNX file is downloadable by anyone who can open the site.

Generated (not in git, like every other weight file): model/<name>.onnx, vendor/ort/*, config.json, examples/.

    python scripts/deployment/prepare_vercel_demo.py            # then: cd apps/engine_bay_vercel && vercel deploy
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "apps" / "engine_bay_web"
OUT = ROOT / "apps" / "engine_bay_vercel"
ORT_FILES = ("ort.wasm.min.mjs", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm")


def vi_names() -> dict[str, str]:
    """VI_NAMES from the web app, read without importing it (it pulls in torch and gradio)."""
    for n in ast.parse((WEB / "app.py").read_text(encoding="utf-8")).body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "VI_NAMES":
            return ast.literal_eval(n.value)
    raise RuntimeError("VI_NAMES not found in apps/engine_bay_web/app.py")


def export_model(pt: Path, model_dir: Path) -> dict:
    """Export the .pt to <model_dir>/<stem>.onnx (static 640, no NMS) and return the inference config shared by the
    browser demo and the Android app: class names (EN/VI), colours, per-class thresholds of this model, NMS settings."""
    from ultralytics import YOLO
    from ultralytics.utils.plotting import colors

    model_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:  # export next to a copy so nothing is written into apps/engine_bay_web
        src = Path(tmp) / pt.name
        shutil.copy2(pt, src)
        model = YOLO(str(src))
        onnx = Path(model.export(format="onnx", imgsz=640, opset=17, simplify=True, dynamic=False, nms=False))
        shutil.copy2(onnx, model_dir / f"{pt.stem}.onnx")
    names = [model.names[i] for i in range(len(model.names))]
    thr_file = WEB / "config" / "class_thresholds" / f"{pt.stem}.yaml"
    thr = yaml.safe_load(thr_file.read_text(encoding="utf-8")).get("thresholds", {}) if thr_file.is_file() else {}
    vi = vi_names()
    return {
        "model": f"model/{pt.stem}.onnx",
        "model_name": pt.stem,
        "model_md5": hashlib.md5(pt.read_bytes()).hexdigest(),
        "release": "poc-v1",
        "imgsz": 640,
        "iou": 0.7,  # ultralytics default NMS IoU
        "max_det": 300,
        "default_conf": 0.35,
        "max_side": 1600,  # photos are downscaled first, as in the Python app
        "names": names,
        "names_vi": [vi.get(n, n) for n in names],
        "colors": ["#%02x%02x%02x" % colors(i, False) for i in range(len(names))],
        "thresholds": {k: float(v) for k, v in thr.items()},
    }


def vendor_ort(version: str) -> None:
    dst = OUT / "vendor" / "ort"
    if all((dst / f).exists() for f in ORT_FILES) and (dst / "VERSION").read_text().strip() == version:
        return
    dst.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        npm = shutil.which("npm") or "npm"  # npm.cmd on Windows
        subprocess.run([npm, "pack", f"onnxruntime-web@{version}", "--silent"], cwd=tmp, check=True)
        tgz = next(Path(tmp).glob("onnxruntime-web-*.tgz"))
        with tarfile.open(tgz) as t:
            for f in ORT_FILES:
                (dst / f).write_bytes(t.extractfile(f"package/dist/{f}").read())
    (dst / "VERSION").write_text(version + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(WEB / "models" / "kd_n_full.pt"))
    ap.add_argument("--ort-version", default="1.30.0")
    a = ap.parse_args()

    pt = Path(a.model)
    config = export_model(pt, OUT / "model")
    (OUT / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8")

    ex = OUT / "examples"
    ex.mkdir(exist_ok=True)
    examples = sorted((WEB / "examples").glob("*.jpg"))
    for p in examples:
        shutil.copy2(p, ex / p.name)
    (OUT / "examples" / "index.json").write_text(json.dumps([p.name for p in examples]), encoding="utf-8")
    vendor_ort(a.ort_version)
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 2**20
    print(f"ok: {OUT} ({size:.1f} MB), model {config['model']} md5(pt) {config['model_md5']}, "
          f"{len(config['names'])} classes, {len(config['thresholds'])} thresholds, {len(examples)} examples")


if __name__ == "__main__":
    main()
