"""benchmark.py - đo độ trễ suy luận đúng chuẩn GPU (slide Day 2, trang 73 và 75).

Tuân thủ nghiêm ngặt các quy tắc:
  - Bỏ >= 10 lần chạy đầu (warmup)
  - Đồng bộ GPU với torch.cuda.synchronize() trước và sau đo
  - Đo >= 50 lần, báo cáo phân vị p50, p95, p99
  - Ghi nhận thông lượng (ảnh/giây)
"""
from __future__ import annotations

import time
from typing import Callable, Optional
import numpy as np
import torch
import torch.nn as nn


def bench(fn: Callable, warmup: int = 10, iters: int = 100, sync: Optional[Callable] = None) -> dict:
    """Đo thời gian thực thi hàm `fn()` theo mili-giây."""
    # Warmup
    for _ in range(warmup):
        fn()
    if sync is not None:
        sync()

    times = []
    for _ in range(iters):
        if sync is not None:
            sync()
        t0 = time.perf_counter()

        fn()

        if sync is not None:
            sync()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    times_arr = np.array(times, dtype=np.float64)
    return {
        "p50": float(np.percentile(times_arr, 50)),
        "p95": float(np.percentile(times_arr, 95)),
        "p99": float(np.percentile(times_arr, 99)),
        "mean": float(np.mean(times_arr)),
        "std": float(np.std(times_arr)),
        "n": iters,
    }


def latency_report(model: nn.Module, batch_size: int = 1, img_size: int = 224,
                   dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100) -> dict:
    """Đo độ trễ forward của `model` với tensor ngẫu nhiên (batch_size, 3, img_size, img_size)."""
    device_obj = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    model = model.to(device_obj)
    model.eval()

    sync_fn = torch.cuda.synchronize if device_obj.type == "cuda" else None
    dummy = torch.randn(batch_size, 3, img_size, img_size, device=device_obj)

    if dtype == "fp16":
        dummy = dummy.half()
        model_bench = model.half()
        def forward_fn():
            with torch.inference_mode():
                return model_bench(dummy)
    elif dtype == "amp":
        def forward_fn():
            with torch.inference_mode(), torch.cuda.amp.autocast():
                return model(dummy)
    else:  # fp32
        dummy = dummy.float()
        model_bench = model.float()
        def forward_fn():
            with torch.inference_mode():
                return model_bench(dummy)

    stats = bench(forward_fn, warmup=warmup, iters=iters, sync=sync_fn)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    images_per_s = round(batch_size / (stats["p50"] / 1000.0), 2) if stats["p50"] > 0 else 0.0

    return {
        "gpu": gpu_name,
        "dtype": dtype,
        "batch": batch_size,
        "img_size": img_size,
        "p50": round(stats["p50"], 2),
        "p95": round(stats["p95"], 2),
        "p99": round(stats["p99"], 2),
        "mean": round(stats["mean"], 2),
        "images_per_s": images_per_s,
        "torch": torch.__version__,
    }


def tta_latency(model: nn.Module, k_views: int = 2, batch_size: int = 1, img_size: int = 224,
                dtype: str = "fp32", device: str = "cuda", warmup: int = 10, iters: int = 50) -> dict:
    """Đo độ trễ khi thực hiện K lượt suy luận TTA (slide trang 63)."""
    report_single = latency_report(
        model, batch_size=batch_size, img_size=img_size, dtype=dtype, device=device, warmup=warmup, iters=iters
    )
    # TTA forward K lần
    return {
        "k_views": k_views,
        "single_view_p50": report_single["p50"],
        "estimated_tta_p50": round(report_single["p50"] * k_views, 2),
        "estimated_tta_p95": round(report_single["p95"] * k_views, 2),
        "images_per_s": round(report_single["images_per_s"] / k_views, 2),
    }
