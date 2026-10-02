"""Engine Bay Vision: web app + JSON API for the engine-bay component segmentation models.

Self-contained: models are read from ./models/*.pt, optional per-class thresholds from
./config/class_thresholds/<model>.yaml (fallback ./config/class_thresholds.yaml),
sample photos from ./examples. Configuration by CLI flag or environment variable (see README.md).

    python app.py                               # http://127.0.0.1:7860
    ENGINE_BAY_HOST=0.0.0.0 python app.py       # serve on the network

Endpoints: /           web UI (upload a photo, components are segmented automatically)
           /healthz    liveness + loaded model
           /api/detect POST multipart `file` (+ optional form fields `conf`, `model`, `per_class`) -> JSON detections
"""

from __future__ import annotations

import argparse
import base64
import html
import os
import threading
import time
from collections import defaultdict
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from ultralytics import YOLO

APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = Path(os.environ.get("ENGINE_BAY_MODELS_DIR", APP_DIR / "models"))
DEFAULT_MODEL = os.environ.get("ENGINE_BAY_MODEL", "kd_n_full.pt")  # file name in MODELS_DIR or a path
DEVICE = os.environ.get("ENGINE_BAY_DEVICE") or ("0" if torch.cuda.is_available() else "cpu")
DEFAULT_CONF = float(os.environ.get("ENGINE_BAY_CONF", "0.35"))
IMAGE_SIZE = 640  # all current engine-bay models are trained and evaluated at 640
MAX_SIDE = 1600  # inputs are downscaled first (training images are 1600 px); keeps labels readable and plotting fast
EXAMPLE_DIR = APP_DIR / "examples"
CLASS_THRESHOLDS_DIR = APP_DIR / "config" / "class_thresholds"  # <model stem>.yaml, calibrated per model by cv_eval.py
CLASS_THRESHOLDS_FILE = Path(os.environ.get("ENGINE_BAY_THRESHOLDS", APP_DIR / "config" / "class_thresholds.yaml"))
_THRESHOLDS: dict[str, dict[str, float]] = {}


def class_thresholds(model_path: str | Path) -> dict[str, float]:
    """Per-class confidence thresholds of one model ({} = none): config/class_thresholds/<stem>.yaml, else the shared
    file. Thresholds are model specific (the student and the teacher are calibrated separately)."""
    stem = Path(model_path).stem
    if stem not in _THRESHOLDS:
        f = CLASS_THRESHOLDS_DIR / f"{stem}.yaml"
        f = f if f.is_file() else CLASS_THRESHOLDS_FILE
        thr = {}
        if f.is_file():
            import yaml

            thr = {k: float(v) for k, v in (yaml.safe_load(f.read_text(encoding="utf-8")) or {}).get("thresholds", {}).items()}
        _THRESHOLDS[stem] = thr
    return _THRESHOLDS[stem]

VI_NAMES = {
    "battery": "Ắc quy",
    "battery_terminal": "Cọc bình ắc quy",
    "fuse_relay_box": "Hộp cầu chì / rơ-le",
    "coolant_reservoir": "Bình nước làm mát",
    "radiator_cap": "Nắp két nước",
    "brake_fluid_reservoir": "Bình dầu phanh",
    "washer_fluid_reservoir": "Bình nước rửa kính",
    "engine_cover": "Nắp che động cơ",
    "oil_filler_cap": "Nắp châm dầu",
    "oil_dipstick": "Que thăm dầu",
    "air_filter_box": "Hộp lọc gió",
    "air_intake_duct": "Ống hút gió",
    "maf_sensor": "Cảm biến lưu lượng khí (MAF)",
    "throttle_body": "Cổ họng ga",
    "alternator": "Máy phát điện",
    "ignition_coil": "Bô-bin đánh lửa",
    "radiator_hose": "Ống két nước",
    "ecu_module": "Hộp ECU",
    "multimeter_diagnostic_tool": "Đồng hồ đo / thiết bị chẩn đoán",
    "intake_manifold": "Cổ hút",
    "oil_filter": "Lọc dầu",
}


MODEL_SUFFIXES = (".pt", ".onnx")


def discover_models() -> list[Path]:
    """Checkpoints (.pt) and exports (.onnx) in MODELS_DIR: students first, then teachers, each alphabetical."""
    found = sorted(p.resolve() for p in MODELS_DIR.iterdir() if p.is_file() and p.suffix in MODEL_SUFFIXES) \
        if MODELS_DIR.is_dir() else []
    return sorted(found, key=lambda p: ("teacher" in p.stem, p.stem, p.suffix))


def resolve_model(name: str | None) -> Path | None:
    if not name:
        return None
    p = Path(name)
    for cand in (p, MODELS_DIR / p, APP_DIR / p):
        if cand.is_file():
            return cand.resolve()
    return None


def served_model(name: str) -> Path | None:
    """API lookup: only models in MODELS_DIR, by stem ("kd_n_full", as /healthz lists them) or file name."""
    return next((p for p in discover_models() if name in (p.stem, p.name)), None)


def served_names() -> list[str]:
    """/healthz model names: the stem, or the file name when a .pt and its .onnx export share the stem."""
    models = discover_models()
    stems = [p.stem for p in models]
    return [p.stem if stems.count(p.stem) == 1 else p.name for p in models]


_INFO: dict[tuple[str, float], dict] = {}


def model_info(path: str | Path) -> dict:
    """Architecture of a checkpoint or ONNX export, read from its metadata (cached by path + mtime):
    arch "YOLO26s-seg", params in millions (None for ONNX), nms_free (YOLO26 end-to-end head: no NMS step)."""
    path = Path(path)
    key = (str(path), path.stat().st_mtime)
    if key in _INFO:
        return _INFO[key]
    arch, params, nms_free, task = None, None, False, "segment"
    try:
        if path.suffix == ".onnx":
            import onnx

            meta = {p.key: p.value for p in onnx.load(str(path), load_external_data=False).metadata_props}
            arch = next((w for w in meta.get("description", "").split() if w.lower().startswith("yolo")), None)
            nms_free = meta.get("end2end", "").lower() == "true" or bool(arch and arch.lower().startswith("yolo26"))
            task = meta.get("task", task)
        else:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            m = ckpt.get("ema") or ckpt.get("model")
            arch = Path(m.yaml.get("yaml_file", "")).stem or None
            params = sum(p.numel() for p in m.parameters()) / 1e6
            nms_free = bool(getattr(m.model[-1], "end2end", False))
    except Exception:  # unknown file layout: the model still runs, only the label is shorter
        pass
    if arch:
        arch = arch.replace("yolov", "YOLOv").replace("yolo", "YOLO")
    _INFO[key] = {"arch": arch, "params": params, "nms_free": nms_free, "task": task,
                  "role": "teacher" if "teacher" in path.stem else "student", "format": path.suffix[1:].upper()}
    return _INFO[key]


def model_label(path: Path) -> str:
    info = model_info(path)
    kind = "teacher · chính xác hơn, chậm hơn" if info["role"] == "teacher" else "student · nhanh"
    arch = " · ".join(x for x in (info["arch"], "không NMS" if info["nms_free"] else None, info["format"]) if x)
    size = path.stat().st_size / 2**20
    return f"{path.stem} · {kind} · {arch} · {size:.0f} MB"


class ModelCache:
    """Keeps the most recently used model in memory."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._path: str | None = None
        self._model: YOLO | None = None

    def get(self, path: str) -> YOLO:
        with self._lock:
            if self._model is None or self._path != path:
                if not Path(path).is_file():
                    raise FileNotFoundError(f"Không tìm thấy model: {path}")
                # an export's file name rarely says "-seg", so Ultralytics would guess task=detect and drop the masks
                task = model_info(path)["task"] if Path(path).suffix == ".onnx" else None
                self._model, self._path = YOLO(path, task=task), path
            return self._model


CACHE = ModelCache()


def _data_url(image_rgb: np.ndarray) -> str:
    ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode("ascii")


def viewer_html(result_rgb: np.ndarray | None = None, original_rgb: np.ndarray | None = None) -> str:
    """Zoomable viewer: result and original layers share one zoom/pan state.

    Behaviour (wheel zoom, drag, double-click reset, original toggle, fullscreen) lives in VIEWER_JS."""
    if result_rgb is None:
        return ('<div class="viewer-wrap"><div class="viewer empty">'
                "Kết quả hiện ở đây · cuộn chuột để phóng to</div></div>")
    res = _data_url(result_rgb)
    orig = _data_url(original_rgb) if original_rgb is not None else res
    return (
        '<div class="viewer-wrap"><div class="viewer">'
        f'<img class="layer orig" draggable="false" alt="Ảnh gốc" src="{orig}">'
        f'<img class="layer res" draggable="false" alt="Kết quả" src="{res}">'
        '<span class="badge">Ảnh gốc</span></div>'
        '<div class="viewer-bar"><span>Cuộn để phóng to · kéo để di chuyển · nhấp đúp để đặt lại</span>'
        '<button type="button" class="vw-orig" title="Chuyển giữa kết quả và ảnh gốc (phím O)">Ảnh gốc</button>'
        '<button type="button" class="vw-full">Toàn màn hình</button>'
        f'<a class="vw-dl" download="ket_qua.jpg" href="{res}">Tải ảnh</a></div></div>'
    )


def downscale(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    s = MAX_SIDE / max(h, w)
    return cv2.resize(image, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else image


def summary_html(text: str, tone: str = "") -> str:
    return f'<div class="summary {tone}">{text}</div>'


def parts_html(parts: dict[str, list[float]]) -> str:
    if not parts:
        return ""
    rows = []
    for name, scores in sorted(parts.items(), key=lambda kv: -max(kv[1])):
        count = f'<span class="count">×{len(scores)}</span>' if len(scores) > 1 else ""
        pct = round(100 * max(scores))
        rows.append(
            f'<li><span class="name">{html.escape(VI_NAMES.get(name, name))}{count}</span>'
            f'<span class="bar"><i style="width:{pct}%"></i></span><span class="pct">{pct}%</span></li>'
        )
    return f'<ul class="parts">{"".join(rows)}</ul>'


def analyze(image: np.ndarray | None, model_path: str, confidence: float, per_class: bool = False):
    if image is None:
        return viewer_html(), summary_html("Tải ảnh khoang máy lên để bắt đầu.", "muted"), ""
    try:
        model = CACHE.get(model_path)
    except FileNotFoundError as e:
        raise gr.Error(str(e))
    image = downscale(image)
    thresholds = class_thresholds(model_path)
    use_thr = bool(per_class and thresholds)
    pred_conf = min([float(confidence), *thresholds.values()]) if use_thr else float(confidence)
    started = time.perf_counter()
    result = model.predict(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), imgsz=IMAGE_SIZE, conf=pred_conf,
                           device=DEVICE, verbose=False)[0]
    if use_thr and result.boxes is not None and len(result.boxes):
        keep = [i for i, (c, sc) in enumerate(zip(result.boxes.cls.tolist(), result.boxes.conf.tolist()))
                if sc >= thresholds.get(result.names[int(c)], float(confidence))]
        result = result[keep]
    ms = (time.perf_counter() - started) * 1000
    plotted = cv2.cvtColor(result.plot(conf=False), cv2.COLOR_BGR2RGB)  # line/font size scale with the image

    parts: dict[str, list[float]] = defaultdict(list)
    if result.boxes is not None:
        for cls_id, score in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist()):
            parts[result.names[int(cls_id)]].append(float(score))
    total = sum(len(v) for v in parts.values())
    if not total:
        text = f"Không tìm thấy linh kiện ở ngưỡng {confidence:.0%}. Thử giảm ngưỡng trong <b>Tùy chọn</b>."
        return viewer_html(plotted, image), summary_html(text, "muted"), ""
    text = f"<b>{total}</b> linh kiện · <b>{len(parts)}</b> loại · {ms:.0f} ms"
    return viewer_html(plotted, image), summary_html(text), parts_html(parts)


CSS = """
.gradio-container { max-width: 1180px !important; margin: 0 auto !important; }
#title h1 { font-size: 22px; font-weight: 650; margin: 12px 0 0; }
#title p { color: var(--body-text-color-subdued); margin: 2px 0 8px; font-size: 14px; }
.summary { font-size: 15px; padding: 6px 2px; }
.summary.muted { color: var(--body-text-color-subdued); }
.parts { list-style: none; padding: 0; margin: 4px 0 0; display: grid; gap: 6px 28px;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); }
.parts li { display: grid; grid-template-columns: 1fr 90px 40px; align-items: center; gap: 10px; font-size: 14px; }
.parts .count { color: var(--body-text-color-subdued); margin-left: 6px; font-size: 12px; }
.parts .bar { height: 6px; border-radius: 3px; background: var(--border-color-primary); overflow: hidden; }
.parts .bar i { display: block; height: 100%; background: var(--color-accent); }
.parts .pct { text-align: right; font-variant-numeric: tabular-nums; color: var(--body-text-color-subdued); }
footer { display: none !important; }
.viewer-wrap { display: flex; flex-direction: column; background: var(--block-background-fill);
  border: 1px solid var(--block-border-color); border-radius: var(--block-radius); overflow: hidden; }
.viewer { position: relative; height: 560px; overflow: hidden; cursor: grab; touch-action: none; }
.viewer.dragging { cursor: grabbing; }
.viewer.empty { display: flex; align-items: center; justify-content: center; cursor: default;
  color: var(--body-text-color-subdued); font-size: 14px; }
.viewer .layer { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain;
  transform-origin: 0 0; user-select: none; }
.viewer.show-orig .res { visibility: hidden; }
.viewer .badge { position: absolute; top: 8px; left: 8px; display: none; font-size: 12px; padding: 2px 8px;
  border-radius: 10px; background: rgba(0,0,0,.6); color: #fff; }
.viewer.show-orig .badge { display: block; }
.viewer-bar { display: flex; align-items: center; gap: 12px; padding: 6px 10px; font-size: 12px;
  color: var(--body-text-color-subdued); border-top: 1px solid var(--block-border-color); }
.viewer-bar span { flex: 1; }
.viewer-bar button, .viewer-bar a { font-size: 12px; color: var(--body-text-color); background: none; border: none;
  cursor: pointer; text-decoration: none; padding: 2px 4px; }
.viewer-bar button:hover, .viewer-bar a:hover, .viewer-bar button.on { color: var(--color-accent); }
.viewer-wrap:fullscreen { border-radius: 0; background: #000; }
.viewer-wrap:fullscreen .viewer { height: auto; flex: 1; }
"""

VIEWER_JS = """
<script>
(() => {
  const st = new WeakMap();
  const viewerOf = (t) => (t && t.closest ? t.closest('.viewer:not(.empty)') : null);
  const get = (box) => st.get(box) || {s: 1, x: 0, y: 0};
  const apply = (box, v) => {
    if (!(v.s > 1)) { v = {s: 1, x: 0, y: 0}; }
    st.set(box, v);
    box.querySelectorAll('.layer').forEach((img) => {
      img.style.transform = 'translate(' + v.x + 'px, ' + v.y + 'px) scale(' + v.s + ')';
    });
  };
  document.addEventListener('wheel', (e) => {
    const box = viewerOf(e.target);
    if (!box) return;
    e.preventDefault();
    const r = box.getBoundingClientRect(), v = get(box);
    const px = e.clientX - r.left, py = e.clientY - r.top;
    const ns = Math.min(10, Math.max(1, v.s * (e.deltaY < 0 ? 1.2 : 1 / 1.2)));
    apply(box, {s: ns, x: px - (px - v.x) * ns / v.s, y: py - (py - v.y) * ns / v.s});
  }, {passive: false});
  let drag = null;
  document.addEventListener('pointerdown', (e) => {
    const box = viewerOf(e.target);
    if (!box || e.button !== 0) return;
    const v = get(box);
    drag = {box: box, sx: e.clientX - v.x, sy: e.clientY - v.y};
    box.classList.add('dragging');
  });
  window.addEventListener('pointermove', (e) => {
    if (!drag) return;
    const v = get(drag.box);
    if (v.s > 1) apply(drag.box, {s: v.s, x: e.clientX - drag.sx, y: e.clientY - drag.sy});
  });
  window.addEventListener('pointerup', () => { if (drag) drag.box.classList.remove('dragging'); drag = null; });
  document.addEventListener('dblclick', (e) => { const box = viewerOf(e.target); if (box) apply(box, {s: 1}); });
  const toggleOrig = (wrap) => {
    const box = wrap && wrap.querySelector('.viewer:not(.empty)');
    if (!box) return;
    const on = box.classList.toggle('show-orig');
    const btn = wrap.querySelector('.vw-orig');
    if (btn) btn.classList.toggle('on', on);
  };
  document.addEventListener('click', (e) => {
    const t = e.target;
    if (!t || !t.closest) return;
    if (t.closest('.vw-orig')) toggleOrig(t.closest('.viewer-wrap'));
    if (t.closest('.vw-full')) {
      const wrap = t.closest('.viewer-wrap');
      if (document.fullscreenElement) document.exitFullscreen(); else wrap.requestFullscreen();
    }
  });
  document.addEventListener('keydown', (e) => {
    if ((e.key === 'o' || e.key === 'O') && !/input|textarea|select/i.test(e.target.tagName)) {
      toggleOrig(document.fullscreenElement || document.querySelector('.viewer-wrap'));
    }
  });
})();
</script>
"""


def on_upload_model(file_obj):
    """Handle user uploading a .pt or .onnx model file."""
    if file_obj is None:
        models = discover_models()
        return gr.update(choices=[(model_label(p), str(p)) for p in models]), "Chưa chọn file model"
    src = Path(file_obj.name)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dst = MODELS_DIR / src.name
    import shutil
    shutil.copyfile(src, dst)
    models = discover_models()
    choices = [(model_label(p), str(p)) for p in models]
    threading.Thread(target=lambda: CACHE.get(str(dst)).predict(
        np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8), imgsz=IMAGE_SIZE, device=DEVICE, verbose=False),
        daemon=True).start()
    return gr.update(choices=choices, value=str(dst)), f"Đã nạp model: {dst.name} ({dst.stat().st_size / 2**20:.1f} MB)"


BENCH_CURRENT = "Ảnh hiện tại"
BENCH_ONE, BENCH_ALL = "Model đang chọn", "So sánh tất cả model"
_CARD = "background:var(--background-fill-secondary);padding:10px;border-radius:8px;text-align:center;"
_SUB = "font-size:11px;color:var(--body-text-color-subdued);"
_TH = "padding:6px;white-space:nowrap;"


def _bench_images(scope: str, current_image: np.ndarray | None) -> list[tuple[str, np.ndarray]]:
    if scope == BENCH_CURRENT and current_image is not None:
        return [(BENCH_CURRENT, downscale(current_image))]
    images = []
    for p in (sorted(EXAMPLE_DIR.glob("*.jpg"))[:6] if EXAMPLE_DIR.is_dir() else []):
        im = cv2.imread(str(p))
        if im is not None:
            images.append((p.name, downscale(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))))
    return images


def _bench_one(model_path: str, images: list[tuple[str, np.ndarray]], iters: int, conf: float) -> dict:
    """Wall-clock latency of model.predict per image, split into Ultralytics' preprocess / inference / postprocess
    times (postprocess = NMS + masks; YOLO26 has no NMS). Every image is warmed up first: photos of different aspect
    ratios give different letterbox shapes, and the first run at a new shape builds new CPU kernels."""
    model = CACHE.get(model_path)
    kw = dict(imgsz=IMAGE_SIZE, conf=conf, device=DEVICE, verbose=False)
    rows = []
    for name, img in images:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for _ in range(2):
            model.predict(bgr, **kw)
        lat, parts, r = [], defaultdict(float), None
        for _ in range(iters):
            t0 = time.perf_counter()
            r = model.predict(bgr, **kw)[0]
            lat.append((time.perf_counter() - t0) * 1000)
            for k in ("preprocess", "inference", "postprocess"):
                parts[k] += (r.speed or {}).get(k) or 0.0
        rows.append({"name": name, "w": img.shape[1], "h": img.shape[0], "avg": sum(lat) / len(lat), "min": min(lat),
                     "max": max(lat), **{k: v / iters for k, v in parts.items()},
                     "dets": len(r.boxes) if r is not None and r.boxes is not None else 0})
    mean = lambda k: sum(x[k] for x in rows) / len(rows)  # noqa: E731
    return {"path": model_path, "images": rows, "avg": mean("avg"), "min": min(x["min"] for x in rows),
            "max": max(x["max"] for x in rows), "preprocess": mean("preprocess"), "inference": mean("inference"),
            "postprocess": mean("postprocess"), "dets": sum(x["dets"] for x in rows)}


def _arch_text(path: str | Path) -> str:
    info = model_info(path)
    params = f"{info['params']:.1f}M tham số" if info["params"] else None
    nms = "không NMS (end-to-end)" if info["nms_free"] else "có NMS"
    return " · ".join(html.escape(x) for x in (info["arch"] or "?", params, nms, info["format"]) if x)


def _render_one(s: dict, iters: int) -> str:
    fps = 1000.0 / s["avg"] if s["avg"] > 0 else 0
    cards = [("Độ trễ trung bình", f"{s['avg']:.1f} ms", f"min {s['min']:.1f} | max {s['max']:.1f}"),
             ("Tốc độ (FPS)", f"{fps:.1f} FPS", f"{DEVICE} · batch 1"),
             ("Suy luận / hậu xử lý", f"{s['inference']:.1f} / {s['postprocess']:.1f} ms",
              f"tiền xử lý {s['preprocess']:.1f} ms"),
             ("Tổng phát hiện", str(s["dets"]), f"trên {len(s['images'])} ảnh")]
    cards_html = "".join(
        f"<div style='{_CARD}'><div style='{_SUB}text-transform:uppercase;'>{t}</div>"
        f"<div style='font-size:20px;font-weight:700;color:var(--color-accent);'>{v}</div><div style='{_SUB}'>{sub}</div></div>"
        for t, v, sub in cards)
    rows = "".join(
        f"<tr style='border-bottom:1px solid var(--block-border-color);'><td style='padding:6px;'><b>{html.escape(x['name'])}</b></td>"
        f"<td>{x['w']}×{x['h']}</td><td><b>{x['avg']:.1f} ms</b></td>"
        f"<td>{x['preprocess']:.1f} / {x['inference']:.1f} / {x['postprocess']:.1f}</td><td>{x['dets']}</td></tr>"
        for x in s["images"])
    return f"""
    <div style="background:var(--block-background-fill);border:1px solid var(--block-border-color);border-radius:10px;padding:14px;margin-top:10px;">
      <h3 style="margin:0 0 4px;font-size:15px;">⚡ Benchmark: {html.escape(Path(s['path']).name)} ({iters} lần lặp)</h3>
      <div style="{_SUB}margin-bottom:10px;">{_arch_text(s['path'])}</div>
      <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(130px, 1fr));gap:10px;margin-bottom:12px;">{cards_html}</div>
      <div style="overflow-x:auto;"><table style="width:100%;border-collapse:collapse;font-size:13px;text-align:left;">
        <thead><tr style="border-bottom:1px solid var(--block-border-color);color:var(--body-text-color-subdued);">
          <th style="{_TH}">Ảnh</th><th style="{_TH}">Kích thước</th><th style="{_TH}">Độ trễ</th>
          <th style="{_TH}">Tiền xử lý / suy luận / hậu xử lý (ms)</th><th style="{_TH}">Linh kiện</th></tr></thead>
        <tbody>{rows}</tbody>
      </table></div>
    </div>
    """


def _render_all(stats: list[dict], errors: list[tuple[str, str]], iters: int, n_images: int) -> str:
    fastest = min((s["avg"] for s in stats), default=0)
    rows = "".join(
        f"<tr style='border-bottom:1px solid var(--block-border-color);'>"
        f"<td style='padding:6px;'><b>{html.escape(Path(s['path']).stem)}</b><div style='{_SUB}'>{_arch_text(s['path'])}</div></td>"
        f"<td>{model_info(s['path'])['role']}</td><td><b>{s['avg']:.1f}</b></td><td>{1000.0 / s['avg']:.1f}</td>"
        f"<td>{s['preprocess']:.1f} / {s['inference']:.1f} / {s['postprocess']:.1f}</td>"
        f"<td>{s['dets']}</td><td>×{s['avg'] / fastest:.1f}</td></tr>"
        for s in stats)
    rows += "".join(f"<tr><td style='padding:6px;'><b>{html.escape(Path(p).stem)}</b></td>"
                    f"<td colspan='6' style='color:var(--body-text-color-subdued);'>lỗi: {html.escape(e)}</td></tr>"
                    for p, e in errors)
    return f"""
    <div style="background:var(--block-background-fill);border:1px solid var(--block-border-color);border-radius:10px;padding:14px;margin-top:10px;">
      <h3 style="margin:0 0 4px;font-size:15px;">⚡ So sánh {len(stats)} model · {n_images} ảnh × {iters} lần lặp · {DEVICE} · batch 1</h3>
      <div style="{_SUB}margin-bottom:10px;">Số linh kiện chỉ để đối chiếu; độ chính xác (mAP) phải đo trên tập test có nhãn.</div>
      <div style="overflow-x:auto;"><table style="width:100%;border-collapse:collapse;font-size:13px;text-align:left;">
        <thead><tr style="border-bottom:1px solid var(--block-border-color);color:var(--body-text-color-subdued);">
          <th style="{_TH}">Model</th><th style="{_TH}">Vai trò</th><th style="{_TH}">Độ trễ TB (ms)</th><th style="{_TH}">FPS</th>
          <th style="{_TH}">Tiền xử lý / suy luận / hậu xử lý (ms)</th><th style="{_TH}">Linh kiện</th>
          <th style="{_TH}">So với nhanh nhất</th></tr></thead>
        <tbody>{rows}</tbody>
      </table></div>
    </div>
    """


def benchmark_model(model_path: str, iterations: float, scope: str, current_image: np.ndarray | None,
                    confidence: float = DEFAULT_CONF, target: str = BENCH_ONE, progress=gr.Progress()) -> str:
    """Latency benchmark of the selected model, or of every model in MODELS_DIR side by side."""
    paths = [str(p) for p in discover_models()] if target == BENCH_ALL else ([model_path] if model_path else [])
    if not paths:
        return "<div class='summary muted'>Vui lòng chọn model trước khi benchmark.</div>"
    images = _bench_images(scope, current_image)
    if not images:
        return "<div class='summary muted'>Không có ảnh để benchmark (hãy tải ảnh lên hoặc đặt ảnh vào examples/).</div>"
    iters = max(1, int(iterations))
    stats, errors = [], []
    for i, p in enumerate(paths):
        progress(i / len(paths), desc=f"Benchmark {Path(p).name}")
        try:
            stats.append(_bench_one(p, images, iters, float(confidence)))
        except Exception as e:  # one broken model must not hide the others
            errors.append((p, str(e)))
    if target != BENCH_ALL:
        if errors:
            return f"<div class='summary muted'>Lỗi nạp model: {html.escape(errors[0][1])}</div>"
        return _render_one(stats[0], iters)
    return _render_all(stats, errors, iters, len(images))


def build_app(initial_model: str | None = None) -> gr.Blocks:
    models = discover_models()
    choices = [(model_label(p), str(p)) for p in models]
    chosen = resolve_model(initial_model) or resolve_model(DEFAULT_MODEL)
    if chosen and str(chosen) not in {v for _, v in choices}:
        choices.insert(0, (model_label(chosen), str(chosen)))
    default = str(chosen) if chosen else (choices[0][1] if choices else None)
    examples = sorted(EXAMPLE_DIR.glob("*.jpg"))[:6] if EXAMPLE_DIR.is_dir() else []
    if default:  # load + warm up in the background so the first upload is fast
        threading.Thread(target=lambda: CACHE.get(default).predict(
            np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8), imgsz=IMAGE_SIZE, device=DEVICE, verbose=False),
            daemon=True).start()

    with gr.Blocks(title="Engine Bay Vision") as demo:
        gr.HTML('<div id="title"><h1>Nhận diện &amp; Benchmark linh kiện khoang máy</h1>'
                "<p>Tải ảnh hoặc upload model tùy chỉnh (.pt / .onnx) để tự đánh giá benchmark ngay trên máy này.</p></div>")
        with gr.Row():
            with gr.Column(scale=2, min_width=280):
                image_in = gr.Image(label="Ảnh", type="numpy", sources=["upload", "clipboard", "webcam"],
                                    height=260, buttons=[])
                if examples:
                    gr.Examples([[str(p)] for p in examples], inputs=image_in,
                                label="Ảnh mẫu", examples_per_page=6)
                with gr.Accordion("Quản lý Model & Tùy chọn", open=True):
                    model = gr.Dropdown(choices, value=default, label="Model đang chạy")
                    model_file = gr.File(label="Tải lên model tùy chỉnh (.pt / .onnx)", file_types=[".pt", ".onnx"])
                    upload_status = gr.HTML("")
                    confidence = gr.Slider(0.05, 0.9, value=DEFAULT_CONF, step=0.05, label="Ngưỡng tin cậy")
                    any_thr = any(class_thresholds(p) for p in discover_models())
                    per_class = gr.Checkbox(value=any_thr, interactive=any_thr,
                                            label="Ngưỡng riêng từng loại linh kiện (hiệu chỉnh bằng cross-validation; "
                                                  "khi bật, thanh ngưỡng chỉ áp cho loại chưa hiệu chỉnh)"
                                            if any_thr else "Ngưỡng riêng từng loại (chưa có config/class_thresholds/)")

                with gr.Accordion("⚡ Đánh giá Benchmark Model", open=False):
                    bench_target = gr.Radio([BENCH_ONE, BENCH_ALL], value=BENCH_ONE, label="Model")
                    bench_scope = gr.Radio(["Bộ ảnh mẫu (Suite)", BENCH_CURRENT], value="Bộ ảnh mẫu (Suite)", label="Phạm vi")
                    bench_iters = gr.Slider(1, 50, value=10, step=1, label="Số lần lặp (iterations)")
                    bench_btn = gr.Button("⚡ Chạy Benchmark Model", variant="primary")

            with gr.Column(scale=5):
                image_out = gr.HTML(viewer_html(), padding=False)
                summary = gr.HTML(summary_html("Tải ảnh khoang máy lên để bắt đầu.", "muted"))
                parts = gr.HTML("")
                bench_out = gr.HTML("")

        inputs, outputs = [image_in, model, confidence, per_class], [image_out, summary, parts]
        image_in.change(analyze, inputs, outputs, api_name="analyze")
        model.change(analyze, inputs, outputs, api_visibility="private")
        confidence.release(analyze, inputs, outputs, api_visibility="private")
        per_class.change(analyze, inputs, outputs, api_visibility="private")

        model_file.upload(on_upload_model, inputs=[model_file], outputs=[model, upload_status])
        bench_btn.click(benchmark_model, inputs=[model, bench_iters, bench_scope, image_in, confidence, bench_target],
                        outputs=[bench_out])
    return demo


def detect_json(image_bgr: np.ndarray, model_path: str, conf: float, per_class: bool) -> dict:
    """Detections as plain data for the REST API (coordinates in pixels of the submitted image)."""
    model = CACHE.get(model_path)
    thresholds = class_thresholds(model_path)
    use_thr = bool(per_class and thresholds)
    pred_conf = min([conf, *thresholds.values()]) if use_thr else conf
    started = time.perf_counter()
    r = model.predict(image_bgr, imgsz=IMAGE_SIZE, conf=pred_conf, device=DEVICE, verbose=False)[0]
    ms = (time.perf_counter() - started) * 1000
    dets = []
    if r.boxes is not None and len(r.boxes):
        polys = r.masks.xy if r.masks is not None else [None] * len(r.boxes)
        for c, sc, box, poly in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist(), polys):
            name = r.names[int(c)]
            if use_thr and sc < thresholds.get(name, conf):
                continue
            dets.append({"class": name, "name_vi": VI_NAMES.get(name, name), "confidence": round(float(sc), 4),
                         "box_xyxy": [round(float(v), 1) for v in box],
                         "polygon": [[round(float(x), 1), round(float(y), 1)] for x, y in poly] if poly is not None else None})
    h, w = image_bgr.shape[:2]
    return {"model": Path(model_path).stem, "image_size": [w, h], "inference_ms": round(ms, 1),
            "per_class_thresholds": use_thr, "count": len(dets), "detections": dets}


def create_server(initial_model: str | None = None):
    """FastAPI app: /healthz, /api/detect and the Gradio UI mounted at /."""
    demo = build_app(initial_model).queue(default_concurrency_limit=1)
    default = resolve_model(initial_model) or resolve_model(DEFAULT_MODEL) or next(iter(discover_models()), None)
    api = FastAPI(title="Engine Bay Vision API")

    @api.get("/healthz")
    def healthz():
        return {"status": "ok", "device": DEVICE, "default_model": default.stem if default else None,
                "models": served_names(),
                "per_class_thresholds": {p.stem: bool(class_thresholds(p)) for p in discover_models()}}

    @api.post("/api/detect")
    async def detect(file: UploadFile = File(...), conf: float = Form(DEFAULT_CONF), model: str | None = Form(None),
                     per_class: bool = Form(True)):
        path = served_model(model) if model else default
        if path is None:
            raise HTTPException(404, f"model not found: {model}")
        img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "file is not a readable image")
        return detect_json(img, str(path), float(conf), bool(per_class))

    return gr.mount_gradio_app(api, demo, path="/", css=CSS, head=VIEWER_JS, show_error=True,
                               theme=gr.themes.Base(primary_hue="orange", neutral_hue="stone"))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Engine-bay component segmentation web app + API")
    ap.add_argument("--model", default=None,
                    help="model selected at start: file name in models/ or a path (default: $ENGINE_BAY_MODEL or kd_n_full.pt)")
    ap.add_argument("--host", default=os.environ.get("ENGINE_BAY_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("ENGINE_BAY_PORT", "7860")))
    return ap.parse_args()


if __name__ == "__main__":
    import uvicorn

    args = parse_args()
    uvicorn.run(create_server(args.model), host=args.host, port=args.port, log_level="info")
