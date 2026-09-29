"""
Classification-logit distillation losses.

Two variants are provided:

* `SoftLogitKLDivergenceLoss` : classic Hinton KD with softmax over the CLASS
  dimension. Suitable for single-label heads (softmax classifiers, DETR-style).
  The class dimension is detected from the tensor layout so that dense
  [B, C, H, W] or [B, C, A] maps are handled correctly (softmax is never taken
  over the spatial axis).

* `BinaryKLLogitLoss` : per-class binary KL for sigmoid (multi-label) heads such
  as YOLOv8/YOLO11/YOLO26, optionally weighted per anchor (e.g. foreground = 1,
  background = small). This is the correct choice for YOLO students.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_rows(logits: torch.Tensor, class_dim: int) -> torch.Tensor:
    """Move the class dimension last and flatten to [N, C]."""
    if logits.dim() == 2:
        return logits
    c = logits.shape[class_dim]
    logits = logits.movedim(class_dim, -1)
    return logits.reshape(-1, c)


class SoftLogitKLDivergenceLoss(nn.Module):
    """tau^2 * KL( softmax(t / tau) || softmax(s / tau) ), softmax over the class dimension."""

    def __init__(self, temperature: float = 3.0, class_dim: int = 1):
        super().__init__()
        self.temperature = temperature
        self.class_dim = class_dim

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        if student_logits.shape != teacher_logits.shape:
            raise ValueError(
                f"student/teacher logits must have identical shapes, got {tuple(student_logits.shape)} "
                f"vs {tuple(teacher_logits.shape)}"
            )
        s = _to_rows(student_logits.float(), self.class_dim)
        t = _to_rows(teacher_logits.float(), self.class_dim)
        tau = self.temperature
        log_p_s = F.log_softmax(s / tau, dim=-1)
        p_t = F.softmax(t / tau, dim=-1)
        return F.kl_div(log_p_s, p_t, reduction="batchmean") * (tau**2)


class BinaryKLLogitLoss(nn.Module):
    """Per-class binary KL divergence for sigmoid heads, weighted per anchor.

    For every (anchor, class):  KL(Bern(sigmoid(t/tau)) || Bern(sigmoid(s/tau)))
    which equals BCEWithLogits(s/tau, sigmoid(t/tau)) minus the entropy of the
    teacher distribution (so the loss is exactly 0 when student == teacher).
    """

    def __init__(self, temperature: float = 2.0, class_dim: int = 1):
        super().__init__()
        self.temperature = temperature
        self.class_dim = class_dim

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            student_logits: [B, C, A] (Ultralytics layout) or [N, C].
            teacher_logits: same shape as student.
            weights: optional [B, A] (or [N]) non-negative per-anchor weights.
        """
        if student_logits.shape != teacher_logits.shape:
            raise ValueError(
                f"student/teacher logits must have identical shapes, got {tuple(student_logits.shape)} "
                f"vs {tuple(teacher_logits.shape)}"
            )
        tau = self.temperature
        s = _to_rows(student_logits.float(), self.class_dim) / tau
        t = _to_rows(teacher_logits.float(), self.class_dim) / tau
        p_t = torch.sigmoid(t)
        ce = F.binary_cross_entropy_with_logits(s, p_t, reduction="none")
        ent = F.binary_cross_entropy_with_logits(t, p_t, reduction="none")  # teacher entropy
        kl = (ce - ent).sum(dim=-1)  # [N]

        if weights is None:
            loss = kl.mean()
        else:
            w = weights.float().reshape(-1)
            if w.numel() != kl.numel():
                raise ValueError(f"weights has {w.numel()} entries but there are {kl.numel()} anchors")
            loss = (kl * w).sum() / (w.sum() + 1e-6)
        return loss * (tau**2)
