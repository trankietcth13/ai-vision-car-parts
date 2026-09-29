"""CPU tests for the Ultralytics-integrated KD criterion (no dataset, no GPU needed)."""

import unittest

import torch

from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG

from distillation.ultralytics_kd import (
    KD_LOSS_NAMES,
    KDCriterion,
    KDSegmentationModel,
    merge_kd_cfg,
    register_teacher,
    strip_kd_modules,
)

NC = 3
IMGSZ = 128


def _fake_batch(bs=2, imgsz=IMGSZ):
    torch.manual_seed(0)
    img = torch.rand(bs, 3, imgsz, imgsz)
    # 3 instances: two in image 0, one in image 1 (normalized xywh)
    batch_idx = torch.tensor([0.0, 0.0, 1.0])
    cls = torch.tensor([[0.0], [1.0], [2.0]])
    bboxes = torch.tensor([[0.3, 0.3, 0.4, 0.4], [0.7, 0.7, 0.2, 0.2], [0.5, 0.5, 0.6, 0.6]])
    m = imgsz // 4
    masks = torch.zeros(bs, m, m)  # overlap_mask=True layout: instance index per pixel
    masks[0, int(0.1 * m) : int(0.5 * m), int(0.1 * m) : int(0.5 * m)] = 1
    masks[0, int(0.6 * m) : int(0.8 * m), int(0.6 * m) : int(0.8 * m)] = 2
    masks[1, int(0.2 * m) : int(0.8 * m), int(0.2 * m) : int(0.8 * m)] = 1
    # semantic masks (class index per pixel) required by heads with a semantic branch (YOLO26-seg)
    sem = torch.zeros(bs, m, m, dtype=torch.long)
    sem[0][masks[0] == 1] = 0
    sem[0][masks[0] == 2] = 1
    sem[1][masks[1] == 1] = 2
    return {"img": img, "batch_idx": batch_idx, "cls": cls, "bboxes": bboxes, "masks": masks, "sem_masks": sem}


def _make_models(student_cfg="yolov8n-seg.yaml", teacher_cfg="yolov8s-seg.yaml"):
    args = get_cfg(DEFAULT_CFG)
    teacher = SegmentationModel(teacher_cfg, nc=NC, verbose=False).eval()
    teacher.args = args
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = KDSegmentationModel(student_cfg, nc=NC, verbose=False)
    student.args = args
    key = f"test_teacher_{id(teacher)}"
    register_teacher(key, teacher)
    student.attach_kd(teacher=teacher, kd_cfg=merge_kd_cfg({"teacher": "unused"}), teacher_key=key)
    return student, teacher


class TestKDCriterion(unittest.TestCase):
    def test_full_kd_step_and_strip(self):
        student, teacher = _make_models()
        student.train()
        batch = _fake_batch()

        preds = student(batch["img"])
        loss, items = student.loss(batch, preds)
        self.assertIsInstance(student.criterion, KDCriterion)
        self.assertEqual(loss.numel(), 5 + len(KD_LOSS_NAMES))
        self.assertEqual(items.numel(), 5 + len(KD_LOSS_NAMES))
        self.assertTrue(torch.isfinite(loss).all(), loss)
        kd_feat, kd_cls, kd_loc = items[5:].tolist()
        self.assertGreater(kd_feat, 0.0)
        self.assertGreater(kd_cls, 0.0)  # same nc -> logit KD active
        self.assertGreater(kd_loc, 0.0)  # same reg_max -> LD active

        loss.sum().backward()
        self.assertTrue(any(p.grad is not None for p in student.kd_adapters.parameters()))
        self.assertTrue(any(p.grad is not None for p in student.kd_fgd.parameters()))
        self.assertTrue(any(p.grad is not None for p in student.model[0].parameters()))

        # In eval mode (validation) KD terms are skipped but the item vector keeps its length
        student.eval()
        with torch.no_grad():
            _, items_eval = student.loss(batch, student(batch["img"]))
        self.assertEqual(items_eval.numel(), 5 + len(KD_LOSS_NAMES))
        self.assertEqual(items_eval[5:].abs().sum().item(), 0.0)

        # Stripping yields a plain SegmentationModel with identical predictions
        clean = strip_kd_modules(student)
        self.assertIs(type(clean), SegmentationModel)
        self.assertFalse(any(k.startswith("kd_") for k in clean.state_dict()))
        with torch.no_grad():
            (y_kd, _), _ = student(batch["img"])
            (y_clean, _), _ = clean(batch["img"])
        self.assertTrue(torch.allclose(y_kd, y_clean, atol=1e-5))

    def test_iou_fallback_when_reg_max_differs(self):
        # yolo26 student (reg_max=1, end2end) with a DFL teacher -> IoU box distillation path
        student, teacher = _make_models(student_cfg="yolo26n-seg.yaml", teacher_cfg="yolov8n-seg.yaml")
        student.train()
        batch = _fake_batch()
        loss, items = student.loss(batch, student(batch["img"]))
        self.assertTrue(torch.isfinite(loss).all(), loss)
        self.assertEqual(items.numel(), 5 + len(KD_LOSS_NAMES))
        self.assertGreater(items[5].item(), 0.0)
        self.assertGreater(items[7].item(), 0.0)
        loss.sum().backward()


if __name__ == "__main__":
    unittest.main()
