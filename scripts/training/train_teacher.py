"""
Train the Teacher model (large YOLO-seg) with plain supervised learning.

The teacher is later frozen and used by `train_kd.py` to distill a small student.
A same-family YOLO teacher is preferred over D-FINE for this project because
the target task is instance segmentation (D-FINE has no mask head) and the
head layouts (DFL boxes, sigmoid class logits, mask coefficients) match the
student one-to-one, which makes logit and localization distillation direct.

Usage:
    python scripts/training/train_teacher.py --model yolo11m-seg.pt --data configs/data_exterior_parts.yaml --epochs 50 --batch 8
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Train teacher YOLO-seg model")
    parser.add_argument("--model", default="yolo11m-seg.pt", help="Pretrained Ultralytics seg checkpoint")
    parser.add_argument("--data", default=str(PROJECT_ROOT / "configs" / "data_exterior_parts.yaml"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0, help="0 recommended on Windows")
    parser.add_argument("--name", default="teacher_yolo11m_seg")
    parser.add_argument("--project", default=str(PROJECT_ROOT / "runs" / "segment"))
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=True,
        patience=args.patience,
        seed=args.seed,
        plots=True,
    )
    best = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\n[Teacher] Training finished. Best checkpoint: {best}")
    print("[Teacher] Next: python scripts/training/train_kd.py --teacher", best)


if __name__ == "__main__":
    main()
