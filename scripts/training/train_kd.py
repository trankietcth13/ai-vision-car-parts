"""
Knowledge Distillation training: large YOLO-seg teacher -> small YOLO-seg student.

Runs the real Ultralytics segmentation trainer (supervised task loss on labeled data)
with three extra distillation terms:
  * feature KD (FGD, GT-box foreground/background masks, GcBlock global term)
  * logit KD (binary KL on sigmoid class logits, weighted inside GT boxes)
  * localization KD (LD on DFL distributions, or IoU distillation for DFL-free heads)

Examples
--------
# 1) Train the teacher once
python scripts/training/train_teacher.py --model yolo11m-seg.pt --data configs/data_exterior_parts.yaml --epochs 50 --batch 8

# 2) Distill into a small student
python scripts/training/train_kd.py --teacher runs/segment/teacher_yolo11m_seg/weights/best.pt --student yolov8n-seg.pt \
                   --data configs/data_exterior_parts.yaml --epochs 50 --batch 16 --name kd_yolov8n_seg

# 3) Same schedule without KD (fair baseline)
python scripts/training/train_kd.py --no-kd --student yolov8n-seg.pt --data configs/data_exterior_parts.yaml --epochs 50 --name baseline_yolov8n_seg

# 4) Add unlabeled images (feature-only KD)
python scripts/training/train_kd.py ... --unlabeled data/unlabeled_exterior
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser(description="YOLO-seg knowledge distillation (Ultralytics)")
    p.add_argument("--kd", default=str(PROJECT_ROOT / "configs" / "kd_hyperparams.yaml"), help="KD config yaml")
    p.add_argument("--data", default=str(PROJECT_ROOT / "configs" / "data_exterior_parts.yaml"))
    p.add_argument("--teacher", default=None, help="Teacher checkpoint (overrides kd yaml)")
    p.add_argument("--student", default="yolov8n-seg.pt", help="Student model: .pt (pretrained) or .yaml (scratch)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", "--batch-size", dest="batch", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--project", default=str(PROJECT_ROOT / "runs" / "train_kd"))
    p.add_argument("--name", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--unlabeled", default=None, help="Directory of unlabeled images for feature-only KD")
    p.add_argument("--no-kd", action="store_true", help="Run the identical schedule without a teacher (baseline)")
    p.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=None,
        help="Resume an interrupted run. Give the run's weights/last.pt (recommended) or omit the value to "
        "resume the most recent run under --project/--name.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.kd, "r", encoding="utf-8") as f:
        kd_yaml = yaml.safe_load(f) or {}
    kd_cfg = dict(kd_yaml.get("distillation", {}))
    train_cfg = dict(kd_yaml.get("training", {}))

    if args.teacher:
        kd_cfg["teacher"] = args.teacher
    if kd_cfg.get("teacher"):
        kd_cfg["teacher"] = str((PROJECT_ROOT / kd_cfg["teacher"]).resolve()) if not Path(kd_cfg["teacher"]).is_absolute() else kd_cfg["teacher"]
    if args.unlabeled:
        kd_cfg.setdefault("unlabeled", {})
        kd_cfg["unlabeled"]["dir"] = args.unlabeled

    student_stem = Path(args.student).stem
    default_name = f"{'baseline' if args.no_kd else 'kd'}_{student_stem}"
    overrides = {
        **train_cfg,
        "model": args.student,
        "data": args.data,
        "device": args.device,
        "project": args.project,
        "name": args.name or default_name,
        "exist_ok": True,
        "task": "segment",
        "plots": True,
    }
    for k in ("epochs", "batch", "imgsz", "workers", "seed"):
        v = getattr(args, k)
        if v is not None:
            overrides[k] = v
    if args.resume:
        last = args.resume
        if last is True:
            last = Path(overrides["project"]) / overrides["name"] / "weights" / "last.pt"
        last = Path(last)
        if not last.exists():
            raise SystemExit(f"[resume] checkpoint not found: {last}")
        overrides["resume"] = str(last)
        overrides["model"] = str(last)
        print(f"[resume] continuing from {last}")

    print("=" * 72)
    print(f" Knowledge Distillation ({'DISABLED - baseline run' if args.no_kd else 'ENABLED'})")
    print(f"  student : {args.student}")
    print(f"  teacher : {None if args.no_kd else kd_cfg.get('teacher')}")
    print(f"  data    : {args.data}")
    print(f"  run     : {overrides['project']}/{overrides['name']}")
    if not args.no_kd:
        print(f"  weights : alpha_feature={kd_cfg.get('alpha_feature')} beta_cls={kd_cfg.get('beta_cls')} "
              f"gamma_loc={kd_cfg.get('gamma_loc')} feature_loss={kd_cfg.get('feature_loss')}")
        if kd_cfg.get("unlabeled", {}).get("dir"):
            print(f"  unlabeled: {kd_cfg['unlabeled']['dir']}")
    print("=" * 72)

    if args.no_kd:
        from ultralytics.models.yolo.segment import SegmentationTrainer

        trainer = SegmentationTrainer(overrides=overrides)
    else:
        from distillation.ultralytics_kd import KDSegmentationTrainer

        trainer = KDSegmentationTrainer(overrides=overrides, kd_cfg=kd_cfg)

    trainer.train()

    metrics = getattr(trainer, "metrics", {}) or {}
    keys = ["metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/mAP50(M)", "metrics/mAP50-95(M)"]
    print("\n[Result] best.pt validation:")
    for k in keys:
        if k in metrics:
            print(f"  {k:<24s} {metrics[k]:.4f}")
    print(f"[Result] weights: {trainer.best}")
    print("[Result] the checkpoint is a plain Ultralytics model: YOLO(str(path)).predict(...) works directly.")


if __name__ == "__main__":
    main()
