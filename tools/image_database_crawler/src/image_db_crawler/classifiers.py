from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Protocol

import httpx
from PIL import Image

from .models import ClassificationResult


class ImageClassifier(Protocol):
    def classify(self, content: bytes) -> ClassificationResult: ...


def _category(labels: list[str]) -> str:
    joined = " ".join(labels).lower()
    if "sensor" in joined or "ecu" in joined:
        return "sensor"
    if any(token in joined for token in ("engine", "radiator", "alternator", "intake", "battery")):
        return "engine_bay"
    return "vehicle_component"


class UltralyticsJevClassifier:
    """Adapter for a local Jev checkpoint exported in Ultralytics format.

    Detection/segmentation checkpoints accept an image when at least one object is
    found. Classification checkpoints accept only the configured target classes.
    """

    DEFAULT_TARGETS = ("vehicle_component", "engine_bay", "sensor")

    def __init__(
        self,
        model_path: str | Path,
        confidence: float = 0.35,
        target_classes: tuple[str, ...] = DEFAULT_TARGETS,
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(model_path))
        self.confidence = confidence
        self.targets = {value.lower().replace(" ", "_") for value in target_classes}

    def classify(self, content: bytes) -> ClassificationResult:
        image = Image.open(io.BytesIO(content)).convert("RGB")
        result = self.model.predict(image, conf=self.confidence, verbose=False)[0]
        names = result.names

        if getattr(result, "boxes", None) is not None and len(result.boxes):
            pairs = sorted(
                ((float(conf), str(names[int(cls)])) for conf, cls in zip(result.boxes.conf, result.boxes.cls)),
                reverse=True,
            )
            labels = list(dict.fromkeys(label for _, label in pairs))
            return ClassificationResult(
                accepted=True,
                category=_category(labels),
                component_name=", ".join(labels),
                confidence=pairs[0][0],
                labels=tuple(labels),
            )

        probs = getattr(result, "probs", None)
        if probs is None:
            return ClassificationResult(False)
        top_index = int(probs.top1)
        label = str(names[top_index])
        score = float(probs.top1conf)
        normalized = label.lower().replace(" ", "_")
        accepted = normalized in self.targets and score >= self.confidence
        return ClassificationResult(accepted, normalized if accepted else "", label, score, (label,))


class JevApiClassifier:
    """Adapter for a Jev HTTP service.

    POST body is JSON with ``image_base64``. Expected response fields are:
    ``accepted``, ``category``, ``component_name``, ``confidence``, ``labels``.
    The aliases ``accept`` and ``component`` are also supported.
    """

    def __init__(self, endpoint: str, api_key: str | None = None, timeout: float = 60.0) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.Client(timeout=timeout, headers=headers)
        self.endpoint = endpoint

    def classify(self, content: bytes) -> ClassificationResult:
        response = self.client.post(
            self.endpoint,
            json={"image_base64": base64.b64encode(content).decode("ascii")},
        )
        response.raise_for_status()
        data = response.json()
        labels = tuple(str(value) for value in data.get("labels", []))
        component = str(data.get("component_name") or data.get("component") or ", ".join(labels))
        accepted = bool(data.get("accepted", data.get("accept", False)))
        category = str(data.get("category") or (_category(list(labels)) if accepted else ""))
        return ClassificationResult(
            accepted=accepted,
            category=category,
            component_name=component,
            confidence=float(data.get("confidence", 0.0)),
            labels=labels,
        )

