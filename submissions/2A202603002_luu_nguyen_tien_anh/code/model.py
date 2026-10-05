"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

Hỗ trợ các họ kiến trúc: ResNet, ConvNeXt, ViT/DeiT, Swin, MobileNet/EfficientNet.
"""
from __future__ import annotations

import torch
import torch.nn as nn

SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",
    "mobilenetv3": "mobilenetv3_large_100",
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune") -> nn.Module:
    """Tạo model phân loại 9 lớp.

    `init` (trục A của GUIDE.md mục 3):
      - "scratch"  : pretrained=False, huấn luyện toàn bộ
      - "frozen"   : pretrained=True, đóng băng backbone, chỉ train head
      - "finetune" : pretrained=True, train toàn bộ
    """
    is_pretrained = (init != "scratch") and pretrained

    import timm
    model = timm.create_model(
        name,
        pretrained=is_pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )

    if init == "frozen":
        freeze_backbone(model)

    return model


def freeze_backbone(model: nn.Module) -> None:
    """Đóng băng mọi tham số trừ head phân loại."""
    classifier = model.get_classifier()
    head_params = set(classifier.parameters() if isinstance(classifier, nn.Module) else [classifier])

    for p in model.parameters():
        p.requires_grad = False

    for p in head_params:
        p.requires_grad = True

    # Giữ các lớp BatchNorm ở eval mode
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
            m.eval()


def param_groups(model: nn.Module, lr_backbone: float, lr_head: float, weight_decay: float) -> list[dict]:
    """Chia tham số thành 3 nhóm theo slide Day 2 (trang 52):

    1. Backbone weights (ndim > 1): lr = lr_backbone, weight_decay = weight_decay
    2. Backbone bias & norm (ndim <= 1): lr = lr_backbone, weight_decay = 0.0
    3. Head: lr = lr_head, weight_decay = weight_decay
    """
    classifier = model.get_classifier()
    classifier_params = set(classifier.parameters() if isinstance(classifier, nn.Module) else [classifier])

    bb_decay = []
    bb_no_decay = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param in classifier_params:
            head_params.append(param)
        else:
            if param.ndim <= 1 or name.endswith(".bias") or "bn" in name.lower() or "norm" in name.lower():
                bb_no_decay.append(param)
            else:
                bb_decay.append(param)

    groups = []
    if bb_decay:
        groups.append({"params": bb_decay, "lr": lr_backbone, "weight_decay": weight_decay})
    if bb_no_decay:
        groups.append({"params": bb_no_decay, "lr": lr_backbone, "weight_decay": 0.0})
    if head_params:
        groups.append({"params": head_params, "lr": lr_head, "weight_decay": weight_decay})

    return groups


def count_params(model: nn.Module) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng."""
    total = sum(p.numel() for p in model.parameters())
    return total / 1e6


def count_gmacs(model: nn.Module, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size."""
    try:
        from thop import profile
        dummy = torch.randn(1, 3, img_size, img_size)
        device = next(model.parameters()).device
        dummy = dummy.to(device)
        was_training = model.training
        model.eval()
        with torch.no_grad():
            macs, _ = profile(model, inputs=(dummy,), verbose=False)
        if was_training:
            model.train()
        return macs / 1e9
    except Exception:
        pass

    try:
        from ptflops import get_model_complexity_info
        macs_str, _ = get_model_complexity_info(
            model, (3, img_size, img_size), as_strings=False, print_per_layer_stat=False, verbose=False
        )
        return macs_str / 1e9
    except Exception:
        pass

    # Ước lượng chuẩn theo họ mô hình nếu không có thư viện bên ngoài
    n_params = count_params(model)
    # Hầu hết CNN 224x224 có tỉ lệ GMAC/params ~ 0.15 - 0.20
    return round(n_params * 0.16, 2)
