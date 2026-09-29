"""
Feature-level distillation losses for multi-scale detector features (P3/P4/P5).

Implements Focal and Global Distillation (FGD, Yang et al., CVPR 2022):
  * Focal part   : MSE between teacher and projected student features, split into
                   foreground / background regions using GT box masks and weighted
                   by the teacher's spatial and channel attention.
  * Attention    : L1 between student and teacher spatial / channel attention maps.
  * Global part  : MSE between GcBlock (global context) outputs of both features.

All inputs are dicts keyed by pyramid level ("P3", "P4", "P5").
Student features are expected to be already projected to the teacher's channel
count (see the 1x1 conv adapters built in `src/distillation/ultralytics_kd.py`).
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

MaskType = Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]]


def spatial_attention(x: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """FGD spatial attention A^S: [B, C, H, W] -> [B, 1, H, W], sums to H*W per image."""
    b, _, h, w = x.shape
    attn = x.abs().mean(dim=1).view(b, -1)
    attn = F.softmax(attn / temperature, dim=-1) * (h * w)
    return attn.view(b, 1, h, w)


def channel_attention(x: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """FGD channel attention A^C: [B, C, H, W] -> [B, C, 1, 1], sums to C per image."""
    b, c = x.shape[:2]
    attn = x.abs().mean(dim=(2, 3))
    attn = F.softmax(attn / temperature, dim=-1) * c
    return attn.view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    """Module wrapper kept for backward compatibility with older code/tests."""

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return spatial_attention(x, self.temperature)


class ChannelAttention(nn.Module):
    """Module wrapper kept for backward compatibility with older code/tests."""

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return channel_attention(x, self.temperature)


class GcBlock(nn.Module):
    """Global Context block (GCNet) used by FGD for the global relation loss.

    out = x + W_v2( ReLU( LN( W_v1( sum_j softmax(W_k x)_j * x_j ) ) ) )
    """

    def __init__(self, channels: int, ratio: float = 0.5):
        super().__init__()
        hidden = max(int(channels * ratio), 8)
        self.mask_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.transform = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.LayerNorm([hidden, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )
        nn.init.zeros_(self.transform[-1].weight)
        nn.init.zeros_(self.transform[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        context_mask = self.mask_conv(x).view(b, 1, h * w).softmax(dim=-1)  # [B, 1, HW]
        # weighted sum over positions; elementwise form instead of torch.bmm, whose K=1 backward
        # dispatches to a Triton kernel that needs a C toolchain + Python headers (missing on the DGX)
        context = (x.view(b, c, h * w) * context_mask).sum(dim=-1).view(b, c, 1, 1)
        return x + self.transform(context)


class FocalAndGlobalDistillationLoss(nn.Module):
    """Focal and Global Distillation loss over a feature pyramid.

    Args:
        alpha_fg: weight of the foreground focal term.
        alpha_bg: weight of the background focal term.
        lambda_attn: weight of the attention (spatial + channel) L1 term.
        lambda_global: weight of the GcBlock global relation term.
        temperature: softmax temperature for attention maps (FGD uses 0.5).
        channels: optional {level: channels}. When given, trainable GcBlocks are
            created for each level and the global loss is enabled.
    """

    def __init__(
        self,
        alpha_fg: float = 1.0,
        alpha_bg: float = 0.5,
        lambda_attn: float = 0.5,
        lambda_global: float = 0.5,
        temperature: float = 0.5,
        channels: Optional[Dict[str, int]] = None,
        # legacy keyword names (older configs / tests)
        alpha_spatial: Optional[float] = None,
        alpha_channel: Optional[float] = None,
    ):
        super().__init__()
        if alpha_spatial is not None or alpha_channel is not None:
            lambda_attn = float(alpha_spatial or 0.0) + float(alpha_channel or 0.0)
            lambda_attn = lambda_attn / 2 if (alpha_spatial and alpha_channel) else lambda_attn
        self.alpha_fg = alpha_fg
        self.alpha_bg = alpha_bg
        self.lambda_attn = lambda_attn
        self.lambda_global = lambda_global
        self.temperature = temperature

        self.gc_student = nn.ModuleDict()
        self.gc_teacher = nn.ModuleDict()
        if channels:
            for lvl, c in channels.items():
                self.gc_student[lvl] = GcBlock(c)
                self.gc_teacher[lvl] = GcBlock(c)

        # Populated after each forward for logging/debugging (python floats).
        self.last_components: Dict[str, float] = {}

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _mask_for(masks: MaskType, lvl: str, shape_hw) -> Optional[torch.Tensor]:
        if masks is None:
            return None
        m = masks[lvl] if isinstance(masks, dict) else masks
        if m is None:
            return None
        m = m.float()
        if m.dim() == 3:
            m = m.unsqueeze(1)
        if m.shape[-2:] != tuple(shape_hw):
            m = F.interpolate(m, size=tuple(shape_hw), mode="nearest")
        return m

    def _single_level(self, lvl: str, feat_s: torch.Tensor, feat_t: torch.Tensor, mask: Optional[torch.Tensor]):
        feat_s = feat_s.float()
        feat_t = feat_t.float()

        as_s, as_t = spatial_attention(feat_s, self.temperature), spatial_attention(feat_t, self.temperature)
        ac_s, ac_t = channel_attention(feat_s, self.temperature), channel_attention(feat_t, self.temperature)
        loss_attn = F.l1_loss(as_s, as_t) + F.l1_loss(ac_s, ac_t)

        # Focal term: teacher-attention-weighted squared error, fg / bg separated.
        weight = (as_t * ac_t).detach()
        diff = (feat_s - feat_t).pow(2) * weight
        c = feat_s.shape[1]
        if mask is None:
            loss_fg = diff.mean()
            loss_bg = diff.new_zeros(())
            feat_loss = self.alpha_fg * loss_fg
        else:
            # mask is binary (area-normalised fg) or FGD scale weights 1/area per box (box-normalised fg,
            # build_box_masks(scale_aware=True)); background is every cell outside all boxes either way
            fg = mask
            bg = 1.0 - (mask > 0).float()
            loss_fg = (diff * fg).sum() / (fg.sum() * c + 1e-6)
            loss_bg = (diff * bg).sum() / (bg.sum() * c + 1e-6)
            feat_loss = self.alpha_fg * loss_fg + self.alpha_bg * loss_bg

        # Global term via GcBlocks (only when built for this level).
        if lvl in self.gc_student:
            loss_global = F.mse_loss(self.gc_student[lvl](feat_s), self.gc_teacher[lvl](feat_t))
        else:
            loss_global = diff.new_zeros(())

        total = feat_loss + self.lambda_attn * loss_attn + self.lambda_global * loss_global
        return total, loss_fg.detach(), loss_bg.detach(), loss_attn.detach(), loss_global.detach()

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        projected_student: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor],
        masks: MaskType = None,
    ) -> torch.Tensor:
        levels = [k for k in projected_student if k in teacher_features]
        if not levels:
            any_feat = next(iter(projected_student.values()))
            return any_feat.sum() * 0.0

        total = 0.0
        comp = {"fg": 0.0, "bg": 0.0, "attn": 0.0, "global": 0.0}
        for lvl in levels:
            fs, ft = projected_student[lvl], teacher_features[lvl]
            if fs.shape[-2:] != ft.shape[-2:]:
                fs = F.interpolate(fs, size=ft.shape[-2:], mode="bilinear", align_corners=False)
            mask = self._mask_for(masks, lvl, ft.shape[-2:])
            loss_lvl, fg, bg, attn, glob = self._single_level(lvl, fs, ft, mask)
            total = total + loss_lvl
            comp["fg"] += float(fg)
            comp["bg"] += float(bg)
            comp["attn"] += float(attn)
            comp["global"] += float(glob)

        n = len(levels)
        self.last_components = {k: v / n for k, v in comp.items()}
        return total / n


class SimpleMSEFeatureLoss(nn.Module):
    """Plain MSE between feature pyramids (optionally restricted to foreground masks)."""

    def forward(
        self,
        projected_student: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor],
        masks: MaskType = None,
    ) -> torch.Tensor:
        loss = 0.0
        count = 0
        for lvl in projected_student:
            if lvl not in teacher_features:
                continue
            fs, ft = projected_student[lvl].float(), teacher_features[lvl].float()
            if fs.shape[-2:] != ft.shape[-2:]:
                fs = F.interpolate(fs, size=ft.shape[-2:], mode="bilinear", align_corners=False)
            m = FocalAndGlobalDistillationLoss._mask_for(masks, lvl, ft.shape[-2:])
            if m is None:
                loss = loss + F.mse_loss(fs, ft)
            else:
                diff = (fs - ft).pow(2)
                loss = loss + (diff * m).sum() / (m.sum() * fs.shape[1] + 1e-6)
            count += 1
        return loss / max(count, 1)


def build_box_masks(
    bboxes_xywh_norm: torch.Tensor,
    batch_idx: torch.Tensor,
    batch_size: int,
    sizes_hw,
    device: torch.device,
    scale_aware: bool = False,
) -> Dict[str, torch.Tensor]:
    """Rasterize normalized GT boxes into foreground masks, one per pyramid level.

    Args:
        bboxes_xywh_norm: [N, 4] boxes in normalized xywh (Ultralytics batch format).
        batch_idx: [N] image index of every box.
        batch_size: number of images in the batch.
        sizes_hw: iterable of (h, w) feature sizes, ordered P3, P4, P5, ...
        device: target device.
        scale_aware: FGD scale mask. Every cell of a box weighs 1 / (number of cells of that box), so each
            box contributes equally to the foreground loss whatever its size (the binary mask lets large
            parts dominate); overlapping boxes keep the larger weight (the smaller box). A box that covers
            no cell centre at a level still gets the cell containing its centre.

    Returns:
        {"P3": [B,1,h3,w3], "P4": ..., ...} float masks: 1 inside any GT box, or the scale weights.
    """
    out: Dict[str, torch.Tensor] = {}
    n = bboxes_xywh_norm.shape[0]
    if n:
        b = bboxes_xywh_norm.to(device).float()
        x1 = (b[:, 0] - b[:, 2] / 2).clamp(0, 1)
        y1 = (b[:, 1] - b[:, 3] / 2).clamp(0, 1)
        x2 = (b[:, 0] + b[:, 2] / 2).clamp(0, 1)
        y2 = (b[:, 1] + b[:, 3] / 2).clamp(0, 1)
        bidx = batch_idx.to(device).long().view(-1)
    for i, (h, w) in enumerate(sizes_hw):
        lvl = f"P{3 + i}"
        mask = torch.zeros(batch_size, h, w, device=device)
        if n:
            ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / h
            xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) / w
            in_y = (ys[None, :] >= y1[:, None]) & (ys[None, :] <= y2[:, None])  # [N, h]
            in_x = (xs[None, :] >= x1[:, None]) & (xs[None, :] <= x2[:, None])  # [N, w]
            m = (in_y[:, :, None] & in_x[:, None, :]).float()  # [N, h, w]
            if scale_aware:
                empty = m.flatten(1).sum(1) == 0
                if empty.any():
                    cy = (((y1 + y2) / 2) * h).long().clamp(0, h - 1)
                    cx = (((x1 + x2) / 2) * w).long().clamp(0, w - 1)
                    idx = empty.nonzero(as_tuple=True)[0]
                    m[idx, cy[idx], cx[idx]] = 1.0
                m = m / m.flatten(1).sum(1).clamp(min=1.0)[:, None, None]
                mask.index_reduce_(0, bidx, m, "amax", include_self=True)
            else:
                mask.index_add_(0, bidx, m)
                mask.clamp_(max=1.0)
        out[lvl] = mask.unsqueeze(1)
    return out
