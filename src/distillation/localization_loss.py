"""
Localization distillation losses.

* `DFLDistillationLoss` : Localization Distillation (LD, Zheng et al., CVPR 2022).
  Treats each box side as a discrete distribution over `reg_max` bins (the DFL
  representation used by YOLOv8 / YOLO11) and applies temperature-scaled KL
  between teacher and student distributions on positive anchors.
  Requires teacher and student to share the same `reg_max` (> 1).

* `IoUBoxDistillationLoss` : fallback for heads without DFL (YOLO26, reg_max = 1)
  or mismatched reg_max: 1 - CIoU between decoded teacher and student boxes on
  positive anchors.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import bbox_iou


def _select_weights(weights: Optional[torch.Tensor], fg_mask: torch.Tensor) -> torch.Tensor:
    if weights is None:
        return torch.ones(int(fg_mask.sum()), device=fg_mask.device)
    return weights.float()[fg_mask]


class DFLDistillationLoss(nn.Module):
    """LD: KL between DFL side distributions of teacher and student on foreground anchors."""

    def __init__(self, temperature: float = 10.0):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        student_dist: torch.Tensor,
        teacher_dist: torch.Tensor,
        fg_mask: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            student_dist: [B, 4*reg_max, A] raw DFL logits (Ultralytics `preds["boxes"]`).
            teacher_dist: [B, 4*reg_max, A] raw DFL logits of the teacher.
            fg_mask: [B, A] bool mask of positive anchors.
            weights: optional [B, A] per-anchor weights (e.g. target scores).
        """
        if student_dist.shape != teacher_dist.shape:
            raise ValueError(
                f"DFL shapes differ: {tuple(student_dist.shape)} vs {tuple(teacher_dist.shape)}"
            )
        b, ch, a = student_dist.shape
        reg_max = ch // 4
        if reg_max <= 1:
            return student_dist.sum() * 0.0
        fg_mask = fg_mask.bool()
        if not fg_mask.any():
            return student_dist.sum() * 0.0

        s = student_dist.float().view(b, 4, reg_max, a).permute(0, 3, 1, 2)[fg_mask]  # [N, 4, reg_max]
        t = teacher_dist.float().view(b, 4, reg_max, a).permute(0, 3, 1, 2)[fg_mask]
        tau = self.temperature
        log_p_s = F.log_softmax(s / tau, dim=-1)
        p_t = F.softmax(t / tau, dim=-1)
        kl = F.kl_div(log_p_s, p_t, reduction="none").sum(-1).mean(-1)  # [N]
        w = _select_weights(weights, fg_mask)
        return (kl * w).sum() / (w.sum() + 1e-6) * (tau**2)


class IoUBoxDistillationLoss(nn.Module):
    """1 - CIoU between decoded teacher and student boxes on foreground anchors."""

    def forward(
        self,
        student_xyxy: torch.Tensor,
        teacher_xyxy: torch.Tensor,
        fg_mask: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            student_xyxy: [B, A, 4] decoded student boxes (any consistent unit).
            teacher_xyxy: [B, A, 4] decoded teacher boxes (same unit).
            fg_mask: [B, A] bool mask of positive anchors.
            weights: optional [B, A] per-anchor weights.
        """
        fg_mask = fg_mask.bool()
        if not fg_mask.any():
            return student_xyxy.sum() * 0.0
        s = student_xyxy.float()[fg_mask]
        t = teacher_xyxy.float()[fg_mask]
        iou = bbox_iou(s, t, xywh=False, CIoU=True).squeeze(-1)
        w = _select_weights(weights, fg_mask)
        return ((1.0 - iou) * w).sum() / (w.sum() + 1e-6)
