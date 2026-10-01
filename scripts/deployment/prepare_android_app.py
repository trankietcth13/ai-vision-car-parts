"""Prepare apps/engine_bay_android: model + config in assets, component-info template, unit-test fixtures.

Generated (not in git, like the other weight files):
  app/src/main/assets/model/<name>.onnx, config.json, examples/*.jpg     same export/config as the browser demo
  app/src/test/resources/fixtures/*                                       parity fixtures produced by Ultralytics
Written ONCE and then left alone (it is edited by hand / by Gemini):
  app/src/main/assets/components.json                                     component information shown in the app

Fixtures (one example photo): the decoded RGB pixels, Ultralytics' LetterBox output for them, the raw ONNX outputs
for that letterboxed tensor, and Ultralytics' NMS + process_mask result. The Kotlin unit tests check that the app's
letterbox matches cv2 and that its post-processing reproduces Ultralytics' detections and masks.

    python scripts/deployment/prepare_android_app.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from deployment.prepare_vercel_demo import WEB, export_model  # noqa: E402

APP = ROOT / "apps" / "engine_bay_android" / "app"
ASSETS = APP / "src" / "main" / "assets"
FIXTURES = APP / "src" / "test" / "resources" / "fixtures"
EXAMPLES = ("Request_ID_13_img_002.jpg", "Request_ID_23_img_004.jpg", "Request_ID_29_img_011.jpg")
FIXTURE_IMAGE = "Request_ID_23_img_004.jpg"


def components_template(config: dict) -> dict:
    """Starter content from configs/diagnosis_knowledge.yaml (English, marked draft) - to be replaced by reviewed text."""
    k = yaml.safe_load((ROOT / "configs" / "diagnosis_knowledge.yaml").read_text(encoding="utf-8"))
    comps = k.get("components", {})
    dtcs = k.get("dtc", [])
    out = {}
    for name, vi in zip(config["names"], config["names_vi"]):
        c = comps.get(name, {})
        related = [{"code": "/".join(d["codes"]), "meaning": d["meaning"]} for d in dtcs if name in d.get("components", [])]
        out[name] = {
            "name_vi": vi,
            "name_en": c.get("name", name.replace("_", " ")),
            "draft": True,
            "summary": c.get("description", ""),
            "function": "",
            "location_hint": "",
            "inspection_checks": [],
            "common_symptoms": [],
            "related_dtcs": related,
            "safety_notes": [],
        }
    return {"schema_version": 1, "language": "vi", "source": "draft from configs/diagnosis_knowledge.yaml", "components": out}


def fixtures(onnx_path: Path, config: dict) -> None:
    import cv2
    import onnxruntime as ort
    import torch
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils import ops
    from ultralytics.utils.nms import non_max_suppression

    FIXTURES.mkdir(parents=True, exist_ok=True)
    bgr = cv2.imread(str(WEB / "examples" / FIXTURE_IMAGE))
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    (FIXTURES / "image_rgb.bin").write_bytes(rgb.tobytes())
    lb = LetterBox(new_shape=(640, 640), auto=False)(image=bgr)  # what Ultralytics feeds an exported (static) model
    lb_rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
    (FIXTURES / "letterbox_rgb.bin").write_bytes(lb_rgb.tobytes())
    x = (lb_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out0, out1 = sess.run(None, {sess.get_inputs()[0].name: x})
    (FIXTURES / "output0.bin").write_bytes(out0.astype("<f4").tobytes())
    (FIXTURES / "output1.bin").write_bytes(out1.astype("<f4").tobytes())
    conf = config["default_conf"]
    det = non_max_suppression(torch.from_numpy(out0), conf_thres=conf, iou_thres=config["iou"], nc=len(config["names"]),
                              max_det=config["max_det"])[0]
    masks = ops.process_mask(torch.from_numpy(out1)[0], det[:, 6:], det[:, :4], (640, 640), upsample=True).numpy() > 0
    (FIXTURES / "masks.bin").write_bytes(np.packbits(masks.reshape(len(det), -1), axis=1).tobytes())
    meta = {"image": FIXTURE_IMAGE, "width": w, "height": h, "conf": conf, "iou": config["iou"], "nc": len(config["names"]),
            "out0_shape": list(out0.shape), "out1_shape": list(out1.shape),
            "detections": [{"cls": int(d[5]), "score": float(d[4]), "box": [float(v) for v in d[:4]],
                            "mask_area": int(m.sum())} for d, m in zip(det.numpy(), masks)]}
    (FIXTURES / "reference.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"fixtures: {len(det)} reference detections for {FIXTURE_IMAGE} ({w}x{h})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(WEB / "models" / "kd_n_full.pt"))
    a = ap.parse_args()
    config = export_model(Path(a.model), ASSETS / "model")
    (ASSETS / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8")
    ex = ASSETS / "examples"
    ex.mkdir(exist_ok=True)
    for name in EXAMPLES:
        shutil.copy2(WEB / "examples" / name, ex / name)
    comp = ASSETS / "components.json"
    if not comp.exists():
        comp.write_text(json.dumps(components_template(config), ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote template {comp}")
    else:
        print(f"kept existing {comp}")
    fixtures(ASSETS / config["model"], config)
    print(f"ok: {ASSETS} (model {config['model']}, {len(config['names'])} classes, {len(EXAMPLES)} examples)")


if __name__ == "__main__":
    main()
