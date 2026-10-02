"""Model discovery and architecture labels of apps/engine_bay_web (YOLO11 + YOLO26, .pt + .onnx)."""

import importlib
import sys
from pathlib import Path

import pytest

pytest.importorskip("gradio")
pytest.importorskip("fastapi")
APP_DIR = Path(__file__).resolve().parents[1] / "apps" / "engine_bay_web"


@pytest.fixture
def web(tmp_path, monkeypatch):
    monkeypatch.setenv("ENGINE_BAY_MODELS_DIR", str(tmp_path))
    monkeypatch.setenv("ENGINE_BAY_DEVICE", "cpu")
    monkeypatch.syspath_prepend(str(APP_DIR))
    sys.modules.pop("app", None)
    return importlib.import_module("app")


def _save_yolo(cfg: str, path: Path) -> None:
    import torch
    from ultralytics.nn.tasks import SegmentationModel

    torch.save({"model": SegmentationModel(cfg, nc=21, verbose=False).half()}, path)


def test_discover_lists_pt_and_onnx_students_first(web, tmp_path):
    for name in ("teacher_a.pt", "kd_b.onnx", "kd_b.pt", "notes.txt"):
        (tmp_path / name).write_bytes(b"x")
    assert [p.name for p in web.discover_models()] == ["kd_b.onnx", "kd_b.pt", "teacher_a.pt"]
    assert web.served_names() == ["kd_b.onnx", "kd_b.pt", "teacher_a"]
    assert web.served_model("kd_b.onnx").suffix == ".onnx"


def test_model_info_reads_yolo26_and_yolo11(web, tmp_path):
    _save_yolo("yolo26s-seg.yaml", tmp_path / "kd_26s.pt")
    _save_yolo("yolo11n-seg.yaml", tmp_path / "teacher_11n.pt")
    v26, v11 = web.model_info(tmp_path / "kd_26s.pt"), web.model_info(tmp_path / "teacher_11n.pt")
    assert (v26["arch"], v26["nms_free"], v26["role"]) == ("YOLO26s-seg", True, "student")
    assert (v11["arch"], v11["nms_free"], v11["role"]) == ("YOLO11n-seg", False, "teacher")
    assert "không NMS" in web.model_label(tmp_path / "kd_26s.pt")


def test_model_info_unreadable_file_still_labels(web, tmp_path):
    (tmp_path / "broken.onnx").write_bytes(b"not a model")
    info = web.model_info(tmp_path / "broken.onnx")
    assert info["arch"] is None and info["task"] == "segment" and info["format"] == "ONNX"
    assert web.model_label(tmp_path / "broken.onnx").startswith("broken · student")
