import unittest

import torch

from distillation.feature_loss import FocalAndGlobalDistillationLoss, SimpleMSEFeatureLoss, build_box_masks
from distillation.localization_loss import DFLDistillationLoss, IoUBoxDistillationLoss
from distillation.logit_loss import BinaryKLLogitLoss, SoftLogitKLDivergenceLoss


def _pyramid(requires_grad=False):
    return {
        "P3": torch.randn(2, 256, 40, 40, requires_grad=requires_grad),
        "P4": torch.randn(2, 256, 20, 20, requires_grad=requires_grad),
        "P5": torch.randn(2, 512, 10, 10, requires_grad=requires_grad),
    }


class TestFeatureLosses(unittest.TestCase):
    def test_fgd_without_mask(self):
        fgd = FocalAndGlobalDistillationLoss()
        s, t = _pyramid(True), _pyramid()
        loss = fgd(s, t)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        for lvl in s:
            self.assertIsNotNone(s[lvl].grad)

    def test_fgd_with_masks_and_global(self):
        channels = {"P3": 256, "P4": 256, "P5": 512}
        fgd = FocalAndGlobalDistillationLoss(channels=channels)
        s, t = _pyramid(True), _pyramid()
        masks = {k: (torch.rand(2, 1, *v.shape[-2:]) > 0.5).float() for k, v in t.items()}
        loss = fgd(s, t, masks)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(p.grad is not None for p in fgd.gc_student.parameters()))
        self.assertIn("global", fgd.last_components)
        self.assertGreater(fgd.last_components["fg"], 0.0)

    def test_fgd_zero_when_identical(self):
        fgd = FocalAndGlobalDistillationLoss(lambda_global=0.0)
        t = _pyramid()
        loss = fgd({k: v.clone() for k, v in t.items()}, t)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_mse_feature_loss(self):
        s, t = _pyramid(True), _pyramid()
        loss = SimpleMSEFeatureLoss()(s, t)
        self.assertTrue(torch.isfinite(loss))

    def test_build_box_masks(self):
        bboxes = torch.tensor([[0.5, 0.5, 0.5, 0.5], [0.1, 0.1, 0.2, 0.2]])  # xywh normalized
        batch_idx = torch.tensor([0, 1])
        masks = build_box_masks(bboxes, batch_idx, 2, [(8, 8), (4, 4)], torch.device("cpu"))
        self.assertEqual(set(masks), {"P3", "P4"})
        m3 = masks["P3"]
        self.assertEqual(tuple(m3.shape), (2, 1, 8, 8))
        self.assertEqual(int(m3[0, 0, 2:6, 2:6].sum()), 16)  # centre box of image 0 covers 4x4 cells
        self.assertEqual(int(m3[0].sum()), 16)
        self.assertEqual(int(m3[1, 0, 0, 0]), 1)  # top-left corner box of image 1
        self.assertEqual(int(m3[1, 0, 7, 7]), 0)


class TestLogitLosses(unittest.TestCase):
    def test_soft_kl_2d(self):
        kl = SoftLogitKLDivergenceLoss(temperature=3.0)
        s = torch.randn(8, 24, requires_grad=True)
        t = torch.randn(8, 24)
        loss = kl(s, t)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(s.grad)

    def test_soft_kl_uses_class_dim_on_dense_maps(self):
        kl = SoftLogitKLDivergenceLoss(temperature=1.0, class_dim=1)
        t = torch.randn(2, 24, 10, 10)
        # identical distributions -> 0 regardless of layout (would be huge if softmax ran over W)
        self.assertAlmostEqual(kl(t.clone(), t).item(), 0.0, places=6)

    def test_binary_kl_zero_when_identical_and_positive_otherwise(self):
        bkl = BinaryKLLogitLoss(temperature=2.0)
        t = torch.randn(2, 21, 100)
        self.assertAlmostEqual(bkl(t.clone(), t).item(), 0.0, places=5)
        s = torch.randn(2, 21, 100, requires_grad=True)
        w = torch.rand(2, 100)
        loss = bkl(s, t, w)
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(s.grad)


class TestLocalizationLosses(unittest.TestCase):
    def test_ld_zero_when_identical(self):
        ld = DFLDistillationLoss(temperature=10.0)
        t = torch.randn(2, 64, 50)
        fg = torch.zeros(2, 50, dtype=torch.bool)
        fg[0, :5] = True
        self.assertAlmostEqual(ld(t.clone(), t, fg).item(), 0.0, places=5)
        s = torch.randn(2, 64, 50, requires_grad=True)
        loss = ld(s, t, fg)
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(s.grad)

    def test_ld_no_positives(self):
        ld = DFLDistillationLoss()
        s = torch.randn(1, 64, 10, requires_grad=True)
        loss = ld(s, torch.randn(1, 64, 10), torch.zeros(1, 10, dtype=torch.bool))
        self.assertEqual(loss.item(), 0.0)

    def test_iou_distillation(self):
        crit = IoUBoxDistillationLoss()
        t = torch.tensor([[[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 20.0, 20.0]]])
        fg = torch.tensor([[True, True]])
        self.assertAlmostEqual(crit(t.clone(), t, fg).item(), 0.0, places=5)
        s = torch.tensor([[[0.0, 0.0, 5.0, 5.0], [5.0, 5.0, 20.0, 20.0]]], requires_grad=True)
        loss = crit(s, t, fg)
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(s.grad)


if __name__ == "__main__":
    unittest.main()
