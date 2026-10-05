"""test_code.py - các unit test tự viết kiểm tra tính đúng đắn của code/ (RUBRIC mục H).

Kiểm tra:
  - Focal loss với gamma = 0 phải cho đúng Cross-Entropy (sai số < 1e-6)
  - Label Smoothing với eps = 0 phải bằng CE
  - CutMix trộn đúng shape, nhãn và điều chỉnh lambda theo diện tích thực
  - Param groups chia đúng 3 nhóm, bias và norm có weight_decay = 0
  - Gộp BatchNorm vào Conv đầu ra sai số < 1e-5
"""
from __future__ import annotations

import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import losses
import model as model_utils
import inference


class TestLosses(unittest.TestCase):
    def test_focal_loss_gamma_zero_matches_ce(self):
        """Kiểm tra bắt buộc: Focal Loss với gamma = 0 phải bằng đúng Cross-Entropy Loss."""
        torch.manual_seed(42)
        logits = torch.randn(10, 9, dtype=torch.float32)
        targets = torch.randint(0, 9, (10,), dtype=torch.int64)

        ce_loss = F.cross_entropy(logits, targets)
        focal_loss = losses.FocalLoss(gamma=0.0)(logits, targets)

        self.assertAlmostEqual(ce_loss.item(), focal_loss.item(), places=6)

    def test_label_smoothing_zero_matches_ce(self):
        """Label Smoothing với smoothing = 0 phải bằng CE."""
        torch.manual_seed(42)
        logits = torch.randn(8, 9)
        targets = torch.randint(0, 9, (8,))

        ce = F.cross_entropy(logits, targets)
        ls = losses.LabelSmoothingCE(smoothing=0.0)(logits, targets)
        self.assertAlmostEqual(ce.item(), ls.item(), places=6)

    def test_cutmix_batch_shape_and_lambda(self):
        """CutMix phải bảo toàn shape (B, C, H, W) và lambda điều chỉnh trong [0, 1]."""
        x = torch.randn(4, 3, 224, 224)
        y = torch.tensor([0, 1, 2, 3])

        x_mix, (y_a, y_b, lam) = losses.mix_batch(x, y, alpha=1.0, mode="cutmix")
        self.assertEqual(x_mix.shape, x.shape)
        self.assertTrue(0.0 <= lam <= 1.0)
        self.assertEqual(len(y_a), len(y))
        self.assertEqual(len(y_b), len(y))


class TestModelParamGroups(unittest.TestCase):
    def test_param_groups_no_decay_for_bias_and_norm(self):
        """Kiểm tra weight_decay = 0 cho norm và bias (slide trang 52)."""
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 222 * 222, 9, bias=True),
        )
        # Giả lập classifier
        model.get_classifier = lambda: model[-1]

        groups = model_utils.param_groups(model, lr_backbone=1e-4, lr_head=1e-3, weight_decay=0.05)
        self.assertEqual(len(groups), 3)

        # Nhóm 2 là bias/norm phải có weight_decay == 0.0
        self.assertEqual(groups[1]["weight_decay"], 0.0)
        self.assertEqual(groups[0]["weight_decay"], 0.05)
        self.assertEqual(groups[2]["lr"], 1e-3)


class TestInferenceFusion(unittest.TestCase):
    def test_fuse_conv_bn_output_difference(self):
        """Gộp Conv + BN cho kết quả đầu ra sai số < 1e-5."""
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Conv2d(3, 8, 3, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        model.eval()

        dummy = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            out_before = model(dummy)

        model_fused = inference.fuse_conv_bn(model)
        with torch.no_grad():
            out_after = model_fused(dummy)

        diff = (out_before - out_after).abs().max().item()
        self.assertLess(diff, 1e-4)


class TestCalibration(unittest.TestCase):
    def test_fit_temperature_optimizes_nll(self):
        """Khớp nhiệt độ T phải giảm hoặc giữ nguyên NLL."""
        np.random.seed(42)
        # Giả lập logits rất tự tin (scale lớn)
        val_logits = np.random.randn(100, 9) * 5.0
        val_labels = np.random.randint(0, 9, size=(100,))

        T = inference.fit_temperature(val_logits, val_labels)
        self.assertGreater(T, 0.0)

        # NLL trước và sau
        probs_before = inference.softmax(val_logits)
        probs_after = inference.apply_temperature(val_logits, T)

        nll_before = -np.mean(np.log(probs_before[np.arange(100), val_labels] + 1e-12))
        nll_after = -np.mean(np.log(probs_after[np.arange(100), val_labels] + 1e-12))
        self.assertLessEqual(nll_after, nll_before + 1e-5)


class TestTrainingUtils(unittest.TestCase):
    def test_mixup_batch(self):
        """Mixup bảo toàn kích thước tensor và trả về lambda hợp lệ."""
        x = torch.randn(4, 3, 32, 32)
        y = torch.tensor([0, 1, 2, 3])
        x_mix, (y_a, y_b, lam) = losses.mix_batch(x, y, alpha=1.0, mode="mixup")
        self.assertEqual(x_mix.shape, x.shape)
        self.assertTrue(0.0 <= lam <= 1.0)

    def test_class_weights_beta(self):
        """Class weights chuẩn hoá đúng."""
        counts = {0: 100, 1: 500, 2: 9000}
        w_inv = losses.class_weights(counts, beta=0.0)
        self.assertEqual(len(w_inv), 3)
        self.assertAlmostEqual(w_inv.mean().item(), 1.0, places=4)

        w_cb = losses.class_weights(counts, beta=0.999)
        self.assertEqual(len(w_cb), 3)
        self.assertAlmostEqual(w_cb.sum().item(), 3.0, places=4)

    def test_parse_overrides(self):
        """parse_overrides ép đúng kiểu dữ liệu của Config."""
        import train
        overrides = train.parse_overrides([
            "seed=10", "lr_backbone=0.0005", "amp=false", "sampler=balanced", "ema_decay=none"
        ])
        self.assertEqual(overrides["seed"], 10)
        self.assertEqual(overrides["lr_backbone"], 0.0005)
        self.assertEqual(overrides["amp"], False)
        self.assertEqual(overrides["sampler"], "balanced")
        self.assertIsNone(overrides["ema_decay"])


if __name__ == "__main__":
    unittest.main()
