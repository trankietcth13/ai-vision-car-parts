"""
Knowledge distillation on top of the real Ultralytics YOLO segmentation trainer.

Teacher : any Ultralytics *-seg checkpoint (e.g. yolo11m-seg trained on the same data), frozen.
Student : any Ultralytics *-seg model (yolov8n/s-seg, yolo11n/s-seg, yolo26n/s-seg).

Total loss = L_task (box + seg + cls + dfl, unchanged Ultralytics loss)
           + alpha * L_feat  (FGD on neck P3/P4/P5 through 1x1 adapters, GT-box fg/bg masks)
           + beta  * L_cls   (binary KL on sigmoid class logits, weighted inside GT boxes)
           + gamma * L_loc   (LD on DFL distributions, or 1-CIoU for DFL-free heads)

Design notes
------------
* The Ultralytics head already returns `feats` (neck outputs), `boxes` (DFL logits)
  and `scores` (class logits) in its prediction dict, so no forward hooks are needed.
* Adapters and FGD GcBlocks are registered as sub-modules of the student
  (`kd_adapters`, `kd_fgd`) so the stock optimizer / EMA / AMP machinery trains them.
  They are stripped from `best.pt` / `last.pt` when training ends, leaving a plain
  `SegmentationModel` that loads with `YOLO(...)` anywhere.
* The teacher is kept out of the student module tree (a process-level registry) so
  EMA deep-copies and checkpoints never include it.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.models import yolo
from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.loss import E2ELoss, v8SegmentationLoss
from ultralytics.utils.tal import dist2bbox, make_anchors
from ultralytics.utils.torch_utils import unwrap_model

from distillation.feature_loss import FocalAndGlobalDistillationLoss, SimpleMSEFeatureLoss, build_box_masks
from distillation.localization_loss import DFLDistillationLoss, IoUBoxDistillationLoss
from distillation.logit_loss import BinaryKLLogitLoss

KD_LOSS_NAMES: Tuple[str, str, str] = ("kd_feat", "kd_cls", "kd_loc")

DEFAULT_KD_CFG: Dict[str, Any] = {
    "teacher": None,  # path to teacher checkpoint
    "feature_loss": "fgd",  # "fgd" | "mse"
    "alpha_feature": 1.0,
    "beta_cls": 1.0,
    "gamma_loc": 1.0,
    "temperature_cls": 2.0,
    "temperature_ld": 10.0,
    "cls_bg_weight": 0.05,  # weight of anchors outside GT boxes in the logit KD
    "fgd": {
        "alpha_fg": 1.0,
        "alpha_bg": 0.5,
        "lambda_attn": 0.5,
        "lambda_global": 0.5,
        "temperature": 0.5,
        "use_global": True,
        "scale_aware": False,  # FGD scale mask (1/area per box); experiment E1, small-object gap
    },
    "warmup_adapter_epochs": 1,  # epochs where only adapters/GcBlocks train (student frozen)
    "unlabeled": {"dir": None, "weight": 0.5, "batch": 8, "every": 1},
}


# ----------------------------------------------------------------------------- teacher registry
_TEACHERS: Dict[str, nn.Module] = {}


def register_teacher(key: str, teacher: nn.Module) -> None:
    _TEACHERS[key] = teacher


def get_teacher(key: Optional[str]) -> Optional[nn.Module]:
    return _TEACHERS.get(key) if key else None


def load_teacher(path: str | Path) -> nn.Module:
    """Load a frozen fp32 Ultralytics segmentation model from a checkpoint."""
    from ultralytics import YOLO

    wrapper = YOLO(str(path))
    model = wrapper.model.float().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if not hasattr(model.model[-1], "reg_max"):
        raise ValueError(f"Teacher {path} does not look like an Ultralytics detection/segmentation model")
    return model


def merge_kd_cfg(user_cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_KD_CFG)
    for k, v in (user_cfg or {}).items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg


# ----------------------------------------------------------------------------- helpers
def preds_dict(out: Any) -> Dict[str, torch.Tensor]:
    """Normalize a model output (train or eval, e2e or not) to the one-to-many prediction dict."""
    if isinstance(out, tuple):
        out = out[1]
    if isinstance(out, dict) and "one2many" in out:
        out = out["one2many"]
    return out


def _decode_boxes(dist: torch.Tensor, anchor_points: torch.Tensor, reg_max: int) -> torch.Tensor:
    """dist: [B, A, 4*reg_max] raw -> xyxy in grid units [B, A, 4]."""
    if reg_max > 1:
        b, a, c = dist.shape
        proj = torch.arange(reg_max, dtype=torch.float32, device=dist.device)
        dist = dist.float().view(b, a, 4, reg_max).softmax(3).matmul(proj)
    return dist2bbox(dist.float(), anchor_points, xywh=False)


def build_kd_modules(
    student: nn.Module, teacher: nn.Module, kd_cfg: Dict[str, Any], imgsz: int = 256
) -> Tuple[List[str], nn.ModuleDict, nn.Module]:
    """Create 1x1 adapters (student ch -> teacher ch) and the feature criterion for each neck level."""
    was_training = student.training
    student.eval()
    teacher.eval()
    device = next(student.parameters()).device
    with torch.no_grad():
        x = torch.zeros(1, 3, imgsz, imgsz, device=device)
        s_feats = preds_dict(student(x))["feats"]
        t_feats = preds_dict(teacher.to(device)(x))["feats"]
    if was_training:
        student.train()

    n_levels = min(len(s_feats), len(t_feats))
    levels = [f"P{3 + i}" for i in range(n_levels)]
    adapters = nn.ModuleDict()
    t_channels: Dict[str, int] = {}
    for i, lvl in enumerate(levels):
        cs, ct = s_feats[i].shape[1], t_feats[i].shape[1]
        conv = nn.Conv2d(cs, ct, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
        adapters[lvl] = nn.Sequential(conv, nn.BatchNorm2d(ct))
        t_channels[lvl] = ct
        LOGGER.info(f"[KD] adapter {lvl}: student {cs}ch {tuple(s_feats[i].shape[-2:])} -> teacher {ct}ch {tuple(t_feats[i].shape[-2:])}")

    if kd_cfg.get("feature_loss", "fgd") == "fgd":
        f = kd_cfg["fgd"]
        criterion = FocalAndGlobalDistillationLoss(
            alpha_fg=f["alpha_fg"],
            alpha_bg=f["alpha_bg"],
            lambda_attn=f["lambda_attn"],
            lambda_global=f["lambda_global"],
            temperature=f["temperature"],
            channels=t_channels if f.get("use_global", True) else None,
        )
    else:
        criterion = SimpleMSEFeatureLoss()
    return levels, adapters, criterion


# ----------------------------------------------------------------------------- losses
class CapturingSegLoss(v8SegmentationLoss):
    """v8SegmentationLoss that remembers the last TaskAligned assignment (fg_mask etc.)."""

    def get_assigned_targets_and_loss(self, preds, batch):
        out = super().get_assigned_targets_and_loss(preds, batch)
        self.last_assigned = out[0]  # (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor)
        return out


class KDCriterion:
    """Wraps the stock segmentation loss and appends feature / logit / localization KD terms."""

    def __init__(self, model: "KDSegmentationModel", base):
        self.base = base
        self.o2m: CapturingSegLoss = base.one2many if isinstance(base, E2ELoss) else base
        self.device = self.o2m.device
        self.stride = self.o2m.stride
        self.nc = self.o2m.nc
        self.reg_max = model.model[-1].reg_max
        self.cfg = model.kd_cfg
        self.levels = model.kd_levels
        self.adapters = model.kd_adapters
        self.feat_criterion = model.kd_fgd
        self.teacher_key = model.kd_teacher_key

        self.cls_kd = BinaryKLLogitLoss(temperature=float(self.cfg["temperature_cls"]))
        self.ld = DFLDistillationLoss(temperature=float(self.cfg["temperature_ld"]))
        self.iou_kd = IoUBoxDistillationLoss()
        self.weights = torch.tensor(
            [float(self.cfg["alpha_feature"]), float(self.cfg["beta_cls"]), float(self.cfg["gamma_loc"])],
            device=self.device,
        )
        self._warned: set = set()
        self.nonfinite = [0, 0, 0]  # batches whose feature / logit / box KD term was dropped

    # -- delegation needed by the Ultralytics trainer for E2E losses
    def update(self):
        if hasattr(self.base, "update"):
            self.base.update()

    @property
    def updates(self):
        return getattr(self.base, "updates", 0)

    @updates.setter
    def updates(self, v):
        if hasattr(self.base, "updates"):
            self.base.updates = v

    def _warn_once(self, key: str, msg: str):
        if key not in self._warned:
            self._warned.add(key)
            LOGGER.warning(msg)

    # -- feature projection shared by the labeled and unlabeled paths
    def project_student(self, s_feats: List[torch.Tensor], t_feats: List[torch.Tensor]):
        proj, tdict = {}, {}
        for i, lvl in enumerate(self.levels):
            sf = self.adapters[lvl](s_feats[i].float())
            if sf.shape[-2:] != t_feats[i].shape[-2:]:
                sf = F.interpolate(sf, size=t_feats[i].shape[-2:], mode="bilinear", align_corners=False)
            proj[lvl] = sf
            tdict[lvl] = t_feats[i]
        return proj, tdict

    @torch.no_grad()
    def teacher_forward(self, teacher: nn.Module, img: torch.Tensor) -> Dict[str, torch.Tensor]:
        # fp32: under AMP the yolo26l teacher produced NaN at some locations of some augmented batches
        with torch.autocast(device_type=img.device.type, enabled=False):
            return preds_dict(teacher(img.float()))

    def feature_kd_on_images(self, model: nn.Module, teacher: nn.Module, img: torch.Tensor) -> torch.Tensor:
        """Feature-only KD for unlabeled images (teacher attention weighting, no GT masks)."""
        s_preds = preds_dict(model(img))
        t_preds = self.teacher_forward(teacher, img)
        proj, tdict = self.project_student(s_preds["feats"], t_preds["feats"])
        return self.feat_criterion(proj, tdict, None)

    # -- main entry
    def __call__(self, preds, batch):
        base_loss, base_items = self.base(preds, batch)
        p = preds_dict(preds)
        bs = p["boxes"].shape[0]
        kd = torch.zeros(3, device=self.device)

        teacher = get_teacher(self.teacher_key)
        if teacher is not None and self.adapters.training:
            with torch.autocast(device_type=self.device.type, enabled=False):  # KD terms in fp32
                kd = self._finite(self._kd_terms(p, batch, teacher))

        weighted = kd * self.weights
        loss = torch.cat([base_loss, weighted * bs])
        items = torch.cat([base_items, weighted.detach()])
        return loss, items

    def _finite(self, terms: List[torch.Tensor]) -> torch.Tensor:
        """Stack the KD terms, dropping any non-finite one for this batch. A dropped term leaves the graph entirely:
        masking it in place would still backpropagate 0 * NaN = NaN, and one NaN step corrupts the EMA (the trainer
        then zeroes its NaN weights), which is how kd_26s_v26_s0 collapsed."""
        out = []
        for i, t in enumerate(terms):
            if not bool(torch.isfinite(t)):
                self.nonfinite[i] += 1
                if self.nonfinite[i] <= 5 or self.nonfinite[i] % 100 == 0:
                    LOGGER.warning(f"[KD] non-finite {('feature', 'logit', 'box')[i]} KD term dropped "
                                   f"({self.nonfinite[i]} batches so far)")
                t = torch.zeros((), device=self.device)
            out.append(t.float().reshape(()))
        return torch.stack(out)

    def _kd_terms(self, p: Dict[str, torch.Tensor], batch: Dict[str, Any], teacher: nn.Module) -> List[torch.Tensor]:
        img = batch["img"]
        bs = img.shape[0]
        tp = self.teacher_forward(teacher, img)
        s_feats, t_feats = p["feats"], tp["feats"]
        zero = torch.zeros((), device=self.device)
        kd = [zero, zero, zero]

        # 1) Feature distillation with GT-box foreground / background masks.
        masks = build_box_masks(
            batch["bboxes"], batch["batch_idx"], bs, [f.shape[-2:] for f in t_feats[: len(self.levels)]], self.device,
            scale_aware=bool((self.cfg.get("fgd") or {}).get("scale_aware", False)),
        )
        proj, tdict = self.project_student(s_feats, t_feats)
        kd[0] = self.feat_criterion(proj, tdict, masks)

        same_anchors = tp["scores"].shape[-1] == p["scores"].shape[-1]
        if not same_anchors:
            self._warn_once("anchors", "[KD] teacher/student anchor counts differ; logit & box KD disabled")
            return kd

        # 2) Logit distillation (binary KL, sigmoid heads) weighted inside GT boxes.
        t_nc = tp["scores"].shape[1]
        if t_nc == self.nc and self.weights[1] > 0:
            fg_anchor = torch.cat([(m > 0).float().flatten(1) for m in masks.values()], dim=1)  # [B, A], binary
            w = fg_anchor + float(self.cfg["cls_bg_weight"]) * (1.0 - fg_anchor)
            kd[1] = self.cls_kd(p["scores"], tp["scores"], w)
        elif t_nc != self.nc:
            self._warn_once("nc", f"[KD] teacher nc={t_nc} != student nc={self.nc}; logit KD disabled")

        # 3) Localization distillation on positive anchors from the TaskAligned assigner.
        assigned = getattr(self.o2m, "last_assigned", None)
        if assigned is not None and self.weights[2] > 0:
            fg_mask, _, _, anchor_points, stride_tensor = assigned
            t_reg_max = teacher.model[-1].reg_max
            if t_reg_max == self.reg_max and self.reg_max > 1:
                kd[2] = self.ld(p["boxes"], tp["boxes"], fg_mask)
            else:
                self._warn_once(
                    "ld",
                    f"[KD] reg_max student={self.reg_max} teacher={t_reg_max}; using IoU box distillation instead of LD",
                )
                s_xyxy = _decode_boxes(p["boxes"].permute(0, 2, 1), anchor_points, self.reg_max) * stride_tensor
                t_xyxy = _decode_boxes(tp["boxes"].permute(0, 2, 1), anchor_points, t_reg_max) * stride_tensor
                kd[2] = self.iou_kd(s_xyxy, t_xyxy.detach(), fg_mask)
        return kd


# ----------------------------------------------------------------------------- model
class KDSegmentationModel(SegmentationModel):
    """SegmentationModel carrying trainable KD modules and a KD-aware criterion."""

    def attach_kd(self, teacher: nn.Module, kd_cfg: Dict[str, Any], teacher_key: str) -> None:
        self.kd_cfg = kd_cfg
        self.kd_teacher_key = teacher_key
        levels, adapters, fgd = build_kd_modules(self, teacher, kd_cfg)
        self.kd_levels = levels
        self.kd_adapters = adapters
        self.kd_fgd = fgd

    def init_criterion(self):
        base = E2ELoss(self, CapturingSegLoss) if getattr(self, "end2end", False) else CapturingSegLoss(self)
        if not hasattr(self, "kd_adapters"):
            return base
        return KDCriterion(self, base)


def strip_kd_modules(model: nn.Module) -> SegmentationModel:
    """Return a plain SegmentationModel with identical weights and no KD sub-modules."""
    clean = SegmentationModel(copy.deepcopy(model.yaml), ch=model.yaml.get("channels", 3), nc=model.yaml["nc"], verbose=False)
    sd = {k: v for k, v in model.state_dict().items() if not k.startswith(("kd_adapters", "kd_fgd"))}
    clean.load_state_dict(sd, strict=True)
    for attr in ("names", "args", "nc", "stride", "task", "end2end"):
        if hasattr(model, attr):
            setattr(clean, attr, getattr(model, attr))
    clean.criterion = None
    dtype = next(model.parameters()).dtype
    return clean.to(dtype).eval()


def strip_kd_checkpoint(path: str | Path) -> bool:
    """Rewrite an Ultralytics checkpoint so `model` is a plain SegmentationModel (loadable via YOLO())."""
    path = Path(path)
    if not path.exists():
        return False
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    changed = False
    for key in ("model", "ema"):
        m = ckpt.get(key)
        if isinstance(m, KDSegmentationModel):
            ckpt[key] = strip_kd_modules(m)
            changed = True
    if changed:
        torch.save(ckpt, path)
        LOGGER.info(f"[KD] stripped adapters/GcBlocks from {path}")
    return changed


# ----------------------------------------------------------------------------- callbacks
def _cb_adapter_warmup(trainer: "KDSegmentationTrainer") -> None:
    warm = int(trainer.kd_cfg.get("warmup_adapter_epochs", 0) or 0)
    if warm <= 0:
        return
    model = unwrap_model(trainer.model)
    freeze = trainer.epoch < warm
    if freeze == getattr(trainer, "_kd_student_frozen", None):
        return
    for name, prm in model.named_parameters():
        if not name.startswith(("kd_adapters", "kd_fgd")):
            prm.requires_grad_(not freeze)
    trainer._kd_student_frozen = freeze
    LOGGER.info(
        f"[KD] epoch {trainer.epoch + 1}: student {'FROZEN (adapter warm-up)' if freeze else 'unfrozen (joint KD)'}"
    )


def _cb_unlabeled_step(trainer: "KDSegmentationTrainer") -> None:
    ul = getattr(trainer, "unlabeled", None)
    if ul is None:
        return
    ul.step(trainer)


def _cb_unlabeled_epoch_end(trainer: "KDSegmentationTrainer") -> None:
    ul = getattr(trainer, "unlabeled", None)
    if ul is not None:
        ul.log_epoch(trainer)


def _cb_strip(trainer: "KDSegmentationTrainer") -> None:
    if RANK not in {-1, 0}:
        return
    for f in (trainer.best, trainer.last):
        try:
            strip_kd_checkpoint(f)
        except Exception as exc:  # pragma: no cover - best effort
            LOGGER.warning(f"[KD] could not strip {f}: {exc}")


# ----------------------------------------------------------------------------- trainer
class KDSegmentationTrainer(SegmentationTrainer):
    """SegmentationTrainer with a frozen teacher and KD losses."""

    def __init__(self, cfg=DEFAULT_CFG, overrides: Optional[dict] = None, _callbacks=None, kd_cfg: Optional[dict] = None):
        self.kd_cfg = merge_kd_cfg(kd_cfg)
        teacher_path = self.kd_cfg.get("teacher")
        if not teacher_path:
            raise ValueError("kd_cfg['teacher'] must point to a trained Ultralytics *-seg checkpoint")
        self.teacher = load_teacher(teacher_path)
        self.teacher_key = f"kd_teacher_{id(self.teacher)}"
        register_teacher(self.teacher_key, self.teacher)
        LOGGER.info(f"[KD] teacher loaded from {teacher_path} "
                    f"(nc={self.teacher.model[-1].nc}, reg_max={self.teacher.model[-1].reg_max})")

        super().__init__(cfg, overrides, _callbacks)

        self.unlabeled = None
        ul_cfg = self.kd_cfg.get("unlabeled") or {}
        if ul_cfg.get("dir"):
            from distillation.unlabeled import UnlabeledFeatureKD

            self.unlabeled = UnlabeledFeatureKD(
                image_dir=ul_cfg["dir"],
                imgsz=self.args.imgsz,
                batch_size=int(ul_cfg.get("batch", 8)),
                weight=float(ul_cfg.get("weight", 0.5)),
                every=int(ul_cfg.get("every", 1)),
            )
            self.add_callback("on_train_batch_start", _cb_unlabeled_step)
            self.add_callback("on_train_epoch_end", _cb_unlabeled_epoch_end)

        self.add_callback("on_train_epoch_start", _cb_adapter_warmup)
        self.add_callback("on_train_end", _cb_strip)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = KDSegmentationModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        model.attach_kd(teacher=self.teacher, kd_cfg=self.kd_cfg, teacher_key=self.teacher_key)
        return model

    def _setup_train(self):
        super()._setup_train()
        self.teacher.to(self.device).eval()

    def get_validator(self):
        self.loss_names = ("box_loss", "seg_loss", "cls_loss", "dfl_loss", "sem_loss", *KD_LOSS_NAMES)
        return yolo.segment.SegmentationValidator(
            self.test_loader, save_dir=self.save_dir, args=copy.copy(self.args), _callbacks=self.callbacks
        )
