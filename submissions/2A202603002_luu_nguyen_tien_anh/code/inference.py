"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

Hỗ trợ Single-view, TTA lật ngang, Multi-crop, Ensemble, Temperature Scaling và Gộp BatchNorm.
"""
from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize_scalar


def softmax(z: np.ndarray) -> np.ndarray:
    """Softmax numerically stable trên numpy."""
    z_max = np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z - z_max)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


def predict_logits(model: nn.Module, loader, device: torch.device | str, view=None) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Chạy model trên loader và gom logit theo đúng thứ tự file."""
    model.eval()
    all_filenames = []
    all_targets = []
    all_logits = []

    with torch.inference_mode():
        for batch in loader:
            images, targets, filenames = batch
            images = images.to(device)

            if view is not None:
                images = view(images)

            logits = model(images)
            all_filenames.extend(filenames)
            all_targets.append(targets.cpu().numpy())
            all_logits.append(logits.cpu().numpy())

    y_true = np.concatenate(all_targets, axis=0)
    logits_arr = np.concatenate(all_logits, axis=0)
    return all_filenames, y_true, logits_arr


def view_identity(x: torch.Tensor) -> torch.Tensor:
    return x


def view_hflip(x: torch.Tensor) -> torch.Tensor:
    """Lật ngang batch (N, C, H, W)."""
    return torch.flip(x, dims=[-1])


def views_multicrop(x: torch.Tensor, crop: int) -> list[torch.Tensor]:
    """5-crop: 4 góc và 1 tâm."""
    _, _, H, W = x.shape
    assert H >= crop and W >= crop, f"Ảnh ({H}x{W}) nhỏ hơn kích thước crop {crop}"

    crops = [
        x[:, :, :crop, :crop],                # Top-left
        x[:, :, :crop, W - crop:],            # Top-right
        x[:, :, H - crop:, :crop],            # Bottom-left
        x[:, :, H - crop:, W - crop:],        # Bottom-right
        x[:, :, (H - crop) // 2:(H + crop) // 2, (W - crop) // 2:(W + crop) // 2],  # Center
    ]
    return crops


def views_multiscale(x: torch.Tensor, sizes: list[int]) -> list[torch.Tensor]:
    """Resize batch về từng kích thước trong `sizes`."""
    return [F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False) for s in sizes]


def aggregate_views(logits_per_view: list[np.ndarray], space: str = "prob") -> np.ndarray:
    """Gộp K lượt chạy của TTA thành một dự đoán xác suất (N, 9).

    - space="prob":  trung bình xác suất softmax của từng view
    - space="logit": trung bình logit rồi tính softmax
    """
    if space == "prob":
        probs_list = [softmax(l) for l in logits_per_view]
        mean_probs = np.mean(probs_list, axis=0)
        # Chuẩn hoá lại để tổng chính xác bằng 1.0
        return mean_probs / np.sum(mean_probs, axis=1, keepdims=True)
    elif space == "logit":
        mean_logits = np.mean(logits_per_view, axis=0)
        return softmax(mean_logits)
    else:
        raise ValueError(f"Không hỗ trợ space '{space}', chỉ chọn 'prob' hoặc 'logit'")


def ensemble_probs(list_of_probs: list[np.ndarray]) -> np.ndarray:
    """Trung bình xác suất của nhiều mô hình."""
    mean_probs = np.mean(list_of_probs, axis=0)
    return mean_probs / np.sum(mean_probs, axis=1, keepdims=True)


def fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    """Tìm nhiệt độ T > 0 cực tiểu NLL trên tập VAL: p = softmax(logit / T) (slide Day 2, trang 69)."""
    # Tránh tràn số: chuyển sang torch tensor để tính NLL
    logits_t = torch.as_tensor(val_logits, dtype=torch.float32)
    labels_t = torch.as_tensor(val_labels, dtype=torch.int64)

    def nll_eval(log_t: float) -> float:
        T = np.exp(log_t)
        scaled_logits = logits_t / T
        loss = F.cross_entropy(scaled_logits, labels_t)
        return float(loss.item())

    # Tối ưu hóa trên log(T) trong khoảng [-2, 2], tương đương T từ ~0.13 đến ~7.4
    res = minimize_scalar(nll_eval, bounds=(-2.0, 2.0), method="bounded")
    optimal_T = float(np.exp(res.x))
    return optimal_T


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """Trả về xác suất softmax(logits / T)."""
    T = max(float(T), 1e-4)
    return softmax(logits / T)


def fuse_conv_bn(model: nn.Module) -> nn.Module:
    """Gộp BatchNorm vào tích chập liền trước (slide Day 2, trang 71, 75)."""
    model_fused = copy.deepcopy(model)
    model_fused.eval()

    try:
        from torch.nn.utils.fusion import fuse_conv_bn_eval
        # Thử fuse tự động trên các container chuẩn
        for name, m in model_fused.named_children():
            if isinstance(m, nn.Sequential):
                for i in range(len(m) - 1):
                    if isinstance(m[i], nn.Conv2d) and isinstance(m[i + 1], (nn.BatchNorm2d, nn.SyncBatchNorm)):
                        m[i] = fuse_conv_bn_eval(m[i], m[i + 1])
                        m[i + 1] = nn.Identity()
    except Exception:
        pass

    return model_fused
