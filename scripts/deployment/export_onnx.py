"""
Export an engine-bay segmentation checkpoint (KD student or teacher) to ONNX.

The checkpoint is a plain Ultralytics model: train_kd.py and average_checkpoints.py --strip-kd remove the KD adapters /
GcBlocks, so the standard Ultralytics exporter is used (opset 18, optional simplify).

Usage:
    python scripts/deployment/export_onnx.py --checkpoint runs/train_kd/kd_n_full/weights/best.pt \
        --output artifacts/deployment/kd_n_full_640.onnx --imgsz 640
"""

import argparse
import shutil
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def export_onnx(checkpoint: str, output: str, imgsz: int = 640, simplify: bool = True, dynamic: bool = False,
                half: bool = False) -> Path:
    from ultralytics import YOLO

    if not Path(checkpoint).is_file():
        raise FileNotFoundError(checkpoint)
    model = YOLO(checkpoint)
    print(f"[Export ONNX] {checkpoint}: task={model.task}, classes={len(model.names)}")
    exported = Path(model.export(format="onnx", imgsz=imgsz, opset=18, simplify=simplify, dynamic=dynamic, half=half))
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if exported.resolve() != out.resolve():
        shutil.move(str(exported), out)
    print(f"[Export ONNX] Saved: {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Export an Ultralytics segmentation checkpoint to ONNX")
    ap.add_argument("--checkpoint", default="runs/train_kd/kd_n_full/weights/best.pt")
    ap.add_argument("--output", default="artifacts/deployment/kd_n_full_640.onnx")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--dynamic", action="store_true", help="dynamic batch/shape axes")
    ap.add_argument("--half", action="store_true", help="FP16 export (needs a GPU)")
    ap.add_argument("--no-simplify", action="store_true")
    a = ap.parse_args()
    export_onnx(a.checkpoint, a.output, a.imgsz, not a.no_simplify, a.dynamic, a.half)
