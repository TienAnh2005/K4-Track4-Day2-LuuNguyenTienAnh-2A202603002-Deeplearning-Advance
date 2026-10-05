"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

Cài đặt Label Smoothing, Focal Loss, Class-balanced Weights, Mixup và CutMix.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_criterion(kind: str = "ce", **kw):
    """Trả về hàm loss theo `kind`: 'ce', 'ls', 'focal', 'ce_weighted'."""
    weight = kw.get("weight", None)
    if isinstance(weight, (list, np.ndarray)):
        weight = torch.as_tensor(weight, dtype=torch.float32)

    if kind == "ce":
        return nn.CrossEntropyLoss(weight=weight)
    elif kind == "ls":
        smoothing = kw.get("smoothing", kw.get("label_smoothing", 0.1))
        return LabelSmoothingCE(smoothing=smoothing, weight=weight)
    elif kind == "focal":
        gamma = kw.get("gamma", kw.get("focal_gamma", 2.0))
        alpha = kw.get("alpha", weight)
        return FocalLoss(gamma=gamma, alpha=alpha)
    elif kind == "ce_weighted":
        if weight is None:
            raise ValueError("ce_weighted cần truyền `weight` tensor")
        return nn.CrossEntropyLoss(weight=weight)
    else:
        raise ValueError(f"Không hỗ trợ loss loại '{kind}'")


class LabelSmoothingCE(nn.Module):
    """Cross-entropy với label smoothing (slide Day 2, trang 56)."""

    def __init__(self, smoothing: float = 0.1, weight: torch.Tensor | None = None):
        super().__init__()
        self.smoothing = float(smoothing)
        self.register_buffer("weight", weight if weight is None else torch.as_tensor(weight, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, target, weight=self.weight, label_smoothing=self.smoothing)


class FocalLoss(nn.Module):
    """Focal Loss đa lớp (slide Day 2, trang 57; Lin et al. arXiv:1708.02002):

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Khi gamma = 0: tương đương hoàn toàn với Cross-Entropy Loss chuẩn.
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | list | None = None):
        super().__init__()
        self.gamma = float(gamma)
        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.as_tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (B, C), target: (B,)
        log_p = F.log_softmax(logits, dim=-1)  # (B, C)
        p = torch.exp(log_p)                  # (B, C)

        # Lấy log_p và p của nhãn đúng
        target_view = target.view(-1, 1)
        log_pt = log_p.gather(1, target_view).squeeze(1)  # (B,)
        pt = p.gather(1, target_view).squeeze(1)          # (B,)

        focal_weight = torch.pow(1.0 - pt, self.gamma)   # (B,)
        loss = -focal_weight * log_pt

        if self.alpha is not None:
            at = self.alpha.to(logits.device).gather(0, target)  # (B,)
            loss = at * loss

        return loss.mean()


def class_weights(counts: dict | list | np.ndarray, beta: float = 0.0) -> torch.Tensor:
    """Trọng số theo lớp từ số ảnh mỗi lớp trong tập TRAIN (slide trang 57).

    - beta = 0: w_c tỉ lệ nghịch với số ảnh (1 / n_c), chuẩn hoá trung bình = 1.0.
    - beta > 0: class-balanced theo số mẫu hiệu dụng (1 - beta) / (1 - beta^n_c),
      chuẩn hoá tổng bằng C (số lớp).
    """
    if isinstance(counts, dict):
        # Sắp xếp theo thứ tự nhãn 0..8
        n_list = [counts[i] for i in range(len(counts))]
    else:
        n_list = list(counts)

    n_arr = np.array(n_list, dtype=np.float64)
    C = len(n_arr)

    if beta <= 0.0:
        weights = 1.0 / np.maximum(n_arr, 1.0)
        weights = weights / weights.mean()  # Chuẩn hoá trung bình bằng 1
    else:
        # Effective number of samples
        effective_num = 1.0 - np.power(beta, n_arr)
        weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
        weights = weights / weights.sum() * C  # Tổng bằng C

    return torch.as_tensor(weights, dtype=torch.float32)


def mix_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0, mode: str = "cutmix") -> tuple:
    """Trộn một batch ảnh và nhãn theo Mixup hoặc CutMix."""
    if alpha <= 0.0:
        return x, (y, y, 1.0)

    lam = float(np.random.beta(alpha, alpha))
    batch_size = x.size(0)
    perm = torch.randperm(batch_size, device=x.device)
    y_a = y
    y_b = y[perm]

    if mode == "mixup":
        x_mixed = lam * x + (1.0 - lam) * x[perm]
        return x_mixed, (y_a, y_b, lam)
    elif mode == "cutmix":
        W = x.size(3)
        H = x.size(2)

        # Cắt hộp chữ nhật tỉ lệ diện tích = 1 - lam
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # Tọa độ tâm
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        x_mixed = x.clone()
        x_mixed[:, :, bby1:bby2, bbx1:bbx2] = x[perm, :, bby1:bby2, bbx1:bbx2]

        # Điều chỉnh lam theo diện tích thực tế
        actual_area = (bbx2 - bbx1) * (bby2 - bby1)
        lam_adj = 1.0 - (actual_area / float(W * H))
        return x_mixed, (y_a, y_b, lam_adj)
    else:
        return x, (y, y, 1.0)


def mixed_loss(criterion: nn.Module, logits: torch.Tensor, targets: tuple) -> torch.Tensor:
    """Loss cho batch đã trộn: lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)."""
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
