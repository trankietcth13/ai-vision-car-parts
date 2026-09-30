"""
QA test for the engine-bay segmentation models (run on the DGX).

For every model it:
  1. evaluates on the held-out, expert-reviewed TEST split (never used for training or model selection)
     -> box / mask mAP50 and mAP50-95, precision, recall, per-class mask AP
  2. measures inference latency (batch 1, warm, fp16 when CUDA is available)
  3. draws predictions on a fixed set of QA images (test samples + any extra images)
and writes qa_results/QA_REPORT.json + QA_REPORT.md + overlays/.

Pass/fail gates (edit with flags):
  * student mask mAP50-95 >= --min-student-map
  * KD student >= baseline student (KD must not hurt)
  * per-class: flags every class with mask AP50 < --weak-ap50

Usage (on the DGX):
    .venv/bin/python scripts/evaluation/qa_test.py \
        --model teacher_v4=runs/segment/engine_teacher_v4_dgx/weights/best.pt@1024 \
        --model kd_n=runs/train_kd/kd_yolo11n_seg_640/weights/best.pt@640 \
        --model baseline_n=runs/train_kd/baseline_yolo11n_seg_640/weights/best.pt@640 \
        --data data/engine_bay_train_v4/data_engine_bay_train.yaml --extra-images qa_images
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="name=path.pt@imgsz (repeatable)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="qa_results")
    ap.add_argument("--extra-images", default=None, help="Folder with extra QA images (e.g. user screenshots)")
    ap.add_argument("--n-samples", type=int, default=12)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--device", default="0")
    ap.add_argument("--min-student-map", type=float, default=0.30)
    ap.add_argument("--weak-ap50", type=float, default=0.30)
    ap.add_argument("--batch", type=int, default=8, help="Validation batch size (lower for 960/1280 evaluation)")
    ap.add_argument(
        "--native-masks",
        action="store_true",
        help="Evaluate masks at input resolution instead of the default prototype resolution (imgsz/4).",
    )
    return ap.parse_args()


def native_segmentation_validator():
    """Return a validator that upsamples predicted masks before metric computation.

    Ultralytics normally selects this path only for save_json/save_txt. Keeping it in
    a validator avoids those output side effects and makes the QA report explicit.
    """
    from ultralytics.models.yolo.segment import SegmentationValidator
    from ultralytics.utils import ops

    class NativeMaskValidator(SegmentationValidator):
        def init_metrics(self, model):
            super().init_metrics(model)
            self.process = ops.process_mask_native

    return NativeMaskValidator


def latency_ms(model, imgsz, device, n=50):
    import torch

    x = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    half = torch.cuda.is_available()
    for _ in range(5):
        model.predict(x, imgsz=imgsz, device=device, half=half, verbose=False)
    t = time.perf_counter()
    for _ in range(n):
        model.predict(x, imgsz=imgsz, device=device, half=half, verbose=False)
    return (time.perf_counter() - t) / n * 1000


def main():
    a = parse_args()
    import yaml
    from ultralytics import YOLO

    out = Path(a.out)
    (out / "overlays").mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(a.data).read_text(encoding="utf-8"))
    root = Path(cfg["path"])
    split = root / cfg[a.split]
    if split.suffix == ".txt":  # vehicle splits are image lists, not folders
        test_imgs = sorted((root / l.strip()).resolve() for l in split.read_text().splitlines() if l.strip())
    else:
        test_imgs = sorted(split.glob("*.jpg"))
    random.Random(0).shuffle(test_imgs)
    samples = test_imgs[: a.n_samples]
    if a.extra_images and Path(a.extra_images).exists():
        samples += sorted(p for p in Path(a.extra_images).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    report = {
        "data": a.data,
        "split": a.split,
        "n_images": len(test_imgs),
        "mask_evaluation": "native_input_resolution" if a.native_masks else "prototype_resolution_imgsz_div_4",
        "models": {},
    }
    validator = native_segmentation_validator() if a.native_masks else None
    for spec in a.model:
        name, rest = spec.split("=", 1)
        path, imgsz = (rest.rsplit("@", 1) + ["640"])[:2]
        imgsz = int(imgsz)
        if not Path(path).exists():
            report["models"][name] = {"error": f"missing {path}"}
            print(f"[QA] {name}: missing {path}")
            continue
        model = YOLO(path)
        r = model.val(
            validator=validator,
            data=a.data,
            split=a.split,
            imgsz=imgsz,
            batch=a.batch,
            device=a.device,
            plots=False,
            verbose=False,
            project=str(out / "val"),
            name=name,
            exist_ok=True,
        )
        per_class = {}
        for i, c in enumerate(r.seg.ap_class_index):
            per_class[r.names[c]] = {"mask_ap50": round(float(r.seg.ap50[i]), 3), "mask_ap50_95": round(float(r.seg.ap[i]), 3),
                                     "box_ap50_95": round(float(r.box.maps[c]), 3)}
        model = YOLO(path)  # fresh instance: val() leaves the model fused/half, which breaks predict warmup
        lat = latency_ms(model, imgsz, a.device)
        m = {
            "weights": path, "imgsz": imgsz,
            "mask_evaluation": report["mask_evaluation"],
            "params_M": round(sum(p.numel() for p in model.model.parameters()) / 1e6, 2),
            "box_map50": round(float(r.box.map50), 3), "box_map50_95": round(float(r.box.map), 3),
            "mask_map50": round(float(r.seg.map50), 3), "mask_map50_95": round(float(r.seg.map), 3),
            "mask_precision": round(float(r.seg.mp), 3), "mask_recall": round(float(r.seg.mr), 3),
            "latency_ms_bs1": round(lat, 1), "per_class": per_class,
            "weak_classes": sorted(k for k, v in per_class.items() if v["mask_ap50"] < a.weak_ap50),
        }
        report["models"][name] = m
        print(f"[QA] {name}: mask mAP50-95 {m['mask_map50_95']} | mAP50 {m['mask_map50']} | {m['latency_ms_bs1']} ms")
        for p in samples:
            res = model.predict(str(p), imgsz=imgsz, conf=a.conf, device=a.device, verbose=False)[0]
            import cv2

            cv2.imwrite(str(out / "overlays" / f"{name}__{p.stem}.jpg"), res.plot(line_width=2, font_size=12))

    # gates
    ms = report["models"]
    gates = []
    students = {k: v for k, v in ms.items() if "error" not in v and not k.startswith("teacher")}
    for k, v in students.items():
        gates.append({"gate": f"{k} mask mAP50-95 >= {a.min_student_map}", "value": v["mask_map50_95"],
                      "pass": v["mask_map50_95"] >= a.min_student_map})
    kd = next((v for k, v in students.items() if k.startswith("kd")), None)
    base = next((v for k, v in students.items() if k.startswith("baseline")), None)
    if kd and base:
        gates.append({"gate": "KD student >= baseline (mask mAP50-95)", "value": round(kd["mask_map50_95"] - base["mask_map50_95"], 3),
                      "pass": kd["mask_map50_95"] >= base["mask_map50_95"]})
    report["gates"] = gates
    (out / "QA_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    names = [k for k in ms if "error" not in ms[k]]
    md = [f"# QA report - {a.split} split ({report['n_images']} expert-reviewed images)", "",
          f"Mask evaluation: `{report['mask_evaluation']}`", "",
          "| model | imgsz | params (M) | mask mAP50 | mask mAP50-95 | box mAP50-95 | P | R | latency ms (bs1) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k in names:
        v = ms[k]
        md.append(f"| {k} | {v['imgsz']} | {v['params_M']} | {v['mask_map50']} | {v['mask_map50_95']} | {v['box_map50_95']} | "
                  f"{v['mask_precision']} | {v['mask_recall']} | {v['latency_ms_bs1']} |")
    md += ["", "## Gates", ""] + [f"- {'PASS' if g['pass'] else 'FAIL'}: {g['gate']} (value {g['value']})" for g in gates]
    classes = sorted({c for k in names for c in ms[k]["per_class"]})
    md += ["", "## Per-class mask AP50-95", "", "| class | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for c in classes:
        md.append(f"| {c} | " + " | ".join(str(ms[k]["per_class"].get(c, {}).get("mask_ap50_95", "-")) for k in names) + " |")
    md += ["", f"Overlays: {out / 'overlays'} ({len(samples)} images per model, conf {a.conf})"]
    (out / "QA_REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
