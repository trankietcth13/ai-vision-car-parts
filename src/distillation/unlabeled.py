"""
Feature-only distillation on unlabeled images.

Feature KD does not need ground truth, so any pool of raw images from the target
domain can be used to pull the student's neck features toward the teacher's.
This module plugs into the Ultralytics training loop through callbacks: before
every labeled batch it runs one extra student/teacher forward on an unlabeled
batch, computes the FGD loss (teacher-attention weighted, no GT masks) and
accumulates its gradients so the trainer's regular optimizer step applies them.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator, List

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import autocast, unwrap_model

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(root: str | Path) -> List[Path]:
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)


def letterbox(img: np.ndarray, size: int, color=(114, 114, 114)) -> np.ndarray:
    h, w = img.shape[:2]
    r = size / max(h, w)
    nh, nw = max(1, round(h * r)), max(1, round(w * r))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), color, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top : top + nh, left : left + nw] = img
    return canvas


class UnlabeledImageDataset(Dataset):
    def __init__(self, image_dir: str | Path, imgsz: int = 640, hflip: float = 0.5):
        self.files = list_images(image_dir)
        if not self.files:
            raise FileNotFoundError(f"No images found under {image_dir}")
        self.imgsz = imgsz
        self.hflip = hflip

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.files[idx]))
        if img is None:
            img = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        img = letterbox(img, self.imgsz)
        if random.random() < self.hflip:
            img = img[:, ::-1]
        img = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))  # BGR->RGB, HWC->CHW
        return torch.from_numpy(img)


class UnlabeledFeatureKD:
    """Holds the unlabeled loader and performs the extra feature-KD backward pass."""

    def __init__(self, image_dir: str | Path, imgsz: int, batch_size: int = 8, weight: float = 0.5, every: int = 1):
        self.dataset = UnlabeledImageDataset(image_dir, imgsz)
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
        self.weight = weight
        self.every = max(1, every)
        self._iter: Iterator | None = None
        self._step = 0
        self._sum = 0.0
        self._n = 0
        LOGGER.info(f"[KD] unlabeled feature-KD enabled: {len(self.dataset)} images from {image_dir} "
                    f"(batch={batch_size}, weight={weight}, every={self.every})")

    def _next(self) -> torch.Tensor:
        if self._iter is None:
            self._iter = iter(self.loader)
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self.loader)
            return next(self._iter)

    def step(self, trainer) -> None:
        self._step += 1
        if self._step % self.every:
            return
        model = unwrap_model(trainer.model)
        criterion = getattr(model, "criterion", None)
        if criterion is None or not hasattr(criterion, "feature_kd_on_images"):
            return  # criterion is created lazily on the first labeled batch
        from distillation.ultralytics_kd import get_teacher

        teacher = get_teacher(criterion.teacher_key)
        if teacher is None:
            return
        img = self._next().to(trainer.device, non_blocking=True).float() / 255
        with autocast(trainer.amp):
            loss = criterion.feature_kd_on_images(trainer.model, teacher, img) * self.weight * img.shape[0]
        trainer.scaler.scale(loss).backward()
        self._sum += float(loss.detach()) / img.shape[0]
        self._n += 1

    def log_epoch(self, trainer) -> None:
        if self._n:
            LOGGER.info(f"[KD] epoch {trainer.epoch + 1}: unlabeled kd_feat (weighted) = {self._sum / self._n:.4f}")
        self._sum, self._n = 0.0, 0
