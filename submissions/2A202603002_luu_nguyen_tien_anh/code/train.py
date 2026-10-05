"""train.py - vòng huấn luyện thống nhất cho toàn bộ thí nghiệm (B, T, F).

Tuân thủ nghiêm ngặt:
  - Một hàm run(cfg) duy nhất nhận Config
  - Chọn checkpoint theo Macro-F1 trên tập VAL
  - Test CHỈ chạy đúng một lần mỗi seed ở Bước 4 khi save_test_predictions=True
  - Lưu đầy đủ curves/, predictions/ và runs/
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, fields
import json
import os
from pathlib import Path
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

# Nhập các module cùng thư mục code/
import dataset
import model as model_utils
import losses as loss_utils
import inference

# Import eval từ repo gốc
try:
    import eval as ev
except ModuleNotFoundError:
    repo_root = str(Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import eval as ev


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"
    pred_dir: str = "predictions"
    curves_dir: str = "curves"
    # --- chỉ bật ở Bước 4 (chung kết) ---
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv ."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    """Cố định mọi nguồn ngẫu nhiên đảm bảo tính tái lập."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_optimizer(model: nn.Module, cfg: Config):
    """AdamW với 3 nhóm tham số (lr_backbone, lr_head, weight_decay=0 cho bias/norm)."""
    groups = model_utils.param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    return torch.optim.AdamW(groups)


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi cosine về 1e-6 theo bước (iteration)."""
    total_steps = int(cfg.epochs * steps_per_epoch)
    warmup_steps = int(cfg.warmup_epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(1e-4, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


class EMA:
    """Trung bình động trọng số: W_ema <- d * W_ema + (1 - d) * W."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad = False

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for ema_p, model_p in zip(self.ema_model.parameters(), model.parameters()):
                ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)
            for ema_b, model_b in zip(self.ema_model.buffers(), model.buffers()):
                ema_b.copy_(model_b)

    def copy_to(self, model: nn.Module) -> None:
        with torch.no_grad():
            for model_p, ema_p in zip(model.parameters(), self.ema_model.parameters()):
                model_p.data.copy_(ema_p.data)
            for model_b, ema_b in zip(model.buffers(), self.ema_model.buffers()):
                model_b.copy_(ema_b)


def train_one_epoch(model: nn.Module, loader, criterion: nn.Module, optimizer, scheduler,
                    scaler, cfg: Config, device: torch.device, ema: EMA | None = None) -> dict:
    """Một epoch huấn luyện với AMP và Cosine Schedule."""
    model.train()
    if cfg.init == "frozen":
        model_utils.freeze_backbone(model)

    running_loss = 0.0
    total_samples = 0
    use_cuda = (device.type == "cuda")

    for batch in loader:
        images, targets, _ = batch
        images = images.to(device)
        targets = targets.to(device)
        b_size = images.size(0)

        targets_mixed = None
        if cfg.mix:
            images, targets_mixed = loss_utils.mix_batch(images, targets, alpha=cfg.mix_alpha, mode=cfg.mix)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=cfg.amp and use_cuda):
            outputs = model(images)
            if targets_mixed is not None:
                loss = loss_utils.mixed_loss(criterion, outputs, targets_mixed)
            else:
                loss = criterion(outputs, targets)

        if cfg.amp and use_cuda:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if ema is not None:
            ema.update(model)

        running_loss += loss.item() * b_size
        total_samples += b_size

    current_lr = optimizer.param_groups[0]["lr"]
    return {
        "train_loss": running_loss / max(1, total_samples),
        "lr": current_lr,
    }


def evaluate(model: nn.Module, loader, criterion: nn.Module, device: torch.device):
    """Đánh giá model trên một DataLoader ở chế độ eval."""
    model.eval()
    all_filenames = []
    all_targets = []
    all_logits = []
    running_loss = 0.0
    total_samples = 0

    with torch.inference_mode():
        for batch in loader:
            images, targets, filenames = batch
            images = images.to(device)
            targets = targets.to(device)
            b_size = images.size(0)

            logits = model(images)
            loss = criterion(logits, targets)

            running_loss += loss.item() * b_size
            total_samples += b_size

            all_filenames.extend(filenames)
            all_targets.append(targets.cpu().numpy())
            all_logits.append(logits.cpu().numpy())

    y_true = np.concatenate(all_targets, axis=0)
    logits_arr = np.concatenate(all_logits, axis=0)
    avg_loss = running_loss / max(1, total_samples)

    return all_filenames, y_true, logits_arr, avg_loss


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    """Vẽ đường cong Loss, Macro-F1 và LR."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_f1 = [h["val_macro_f1"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss
    axes[0].plot(epochs, train_loss, "b-o", label="Train Loss")
    axes[0].plot(epochs, val_loss, "r-s", label="Val Loss")
    axes[0].set_title(f"{title} - Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Val Macro-F1
    axes[1].plot(epochs, val_f1, "g-^", label="Val Macro-F1")
    best_idx = np.argmax(val_f1)
    best_f1 = val_f1[best_idx]
    best_ep = epochs[best_idx]
    axes[1].scatter([best_ep], [best_f1], color="red", s=100, zorder=5, label=f"Best: {best_f1:.4f} (ep {best_ep})")
    axes[1].set_title(f"{title} - Val Macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro-F1")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run(cfg: Config) -> dict:
    """Huấn luyện trọn vẹn một cấu hình và lưu kết quả theo đúng chuẩn RUBRIC."""
    # 1. Cố định seed & tạo thư mục
    set_seed(cfg.seed)
    r_dir = run_dir(cfg)
    r_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg.pred_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.curves_dir).mkdir(parents=True, exist_ok=True)

    with open(r_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Đọc và kiểm tra split (quy tắc S1-S6)
    train_df, val_df, test_df = dataset.load_split(cfg.labels_dir, fold=cfg.fold)
    dataset.check_split(train_df, val_df, test_df, cfg.images_dir)

    # 3. Tạo DataLoader
    train_transform = dataset.build_transforms(train=True, img_size=cfg.img_size, aug=cfg.aug)
    val_transform = dataset.build_transforms(train=False, img_size=cfg.img_size)

    train_loader = dataset.make_loader(
        train_df, cfg.images_dir, train_transform, cfg.batch_size,
        train=True, sampler=cfg.sampler, num_workers=cfg.num_workers
    )
    val_loader = dataset.make_loader(
        val_df, cfg.images_dir, val_transform, cfg.batch_size,
        train=False, sampler=None, num_workers=cfg.num_workers
    )

    # 4. Tạo Model, Criterion, Optimizer, Scheduler, EMA
    model = model_utils.build_model(
        name=cfg.backbone, pretrained=True, num_classes=ev.NUM_CLASSES,
        drop_rate=cfg.drop_rate, init=cfg.init
    ).to(device)

    # Trọng số loss nếu có
    weight_tensor = None
    if cfg.loss == "ce_weighted" or cfg.class_weight_beta is not None:
        beta = cfg.class_weight_beta if cfg.class_weight_beta is not None else 0.0
        class_counts = train_df["Label"].value_counts().to_dict()
        weight_tensor = loss_utils.class_weights(class_counts, beta=beta).to(device)

    criterion = loss_utils.build_criterion(
        kind=cfg.loss,
        label_smoothing=cfg.label_smoothing,
        focal_gamma=cfg.focal_gamma,
        weight=weight_tensor
    ).to(device)

    optimizer = build_optimizer(model, cfg)
    steps_per_epoch = len(train_loader)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")

    ema = EMA(model, decay=cfg.ema_decay) if cfg.ema_decay is not None else None

    # 5. Vòng lặp huấn luyện
    best_macro_f1 = -1.0
    best_epoch = -1
    best_weights_path = r_dir / "best_model.pt"
    history = []
    epoch_times = []

    print(f"\n=======================================================")
    print(f"Bắt đầu thí nghiệm: {cfg.exp_id} | Backbone: {cfg.backbone} | Seed: {cfg.seed}")
    print(f"=======================================================")

    for ep in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_stats = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema=ema
        )
        ep_time = time.time() - t0
        epoch_times.append(ep_time)

        # Đánh giá trên VAL
        eval_model = model
        if ema is not None:
            # Dùng trọng số EMA để đánh giá
            eval_model = ema.ema_model

        val_names, val_true, val_logits, val_loss = evaluate(eval_model, val_loader, criterion, device)
        val_probs = ev.softmax(val_logits)
        val_preds = val_probs.argmax(axis=1)

        val_metrics = ev.compute_metrics(val_true, val_preds, val_probs)
        macro_f1 = val_metrics["macro_f1"]
        top1 = val_metrics["top1"]

        history.append({
            "epoch": ep,
            "train_loss": train_stats["train_loss"],
            "val_loss": val_loss,
            "val_macro_f1": macro_f1,
            "val_top1": top1,
            "lr": train_stats["lr"],
            "time_s": ep_time,
        })

        print(f"Epoch {ep:02d}/{cfg.epochs:02d} | Train Loss: {train_stats['train_loss']:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val F1: {macro_f1:.4f} | Val Top-1: {top1*100:.2f}% | ({ep_time:.1f}s)")

        # Lưu checkpoint tốt nhất theo Macro-F1 val
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_epoch = ep
            torch.save(eval_model.state_dict(), best_weights_path)

    # 6. Nạp checkpoint tốt nhất và xuất dự đoán trên VAL
    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    val_names, val_true, best_val_logits, _ = evaluate(model, val_loader, criterion, device)
    best_val_probs = ev.softmax(best_val_logits)
    ev.save_predictions(pred_path(cfg, "val"), val_names, val_true, best_val_probs)

    # 7. Đánh giá TEST nếu được yêu cầu (CHỈ ở Bước 4)
    test_metrics = {}
    if cfg.save_test_predictions:
        test_transform = dataset.build_transforms(train=False, img_size=cfg.img_size)
        test_loader = dataset.make_loader(
            test_df, cfg.images_dir, test_transform, cfg.batch_size,
            train=False, sampler=None, num_workers=cfg.num_workers
        )
        test_names, test_true, test_logits, _ = evaluate(model, test_loader, criterion, device)
        test_probs_uncal = ev.softmax(test_logits)

        # Lưu bản uncal cho eval.py grade tự chấm I4(a)
        ev.save_predictions(
            Path(cfg.pred_dir) / f"{cfg.exp_id}_uncal_seed{cfg.seed}_test.csv",
            test_names, test_true, test_probs_uncal
        )

        # Nếu là cấu hình chung kết (exp_id bắt đầu bằng "F"), áp dụng Temperature Scaling đã khớp trên VAL
        if cfg.exp_id.startswith("F"):
            opt_t = inference.fit_temperature(best_val_logits, val_true)
            test_probs = inference.apply_temperature(test_logits, opt_t)
            print(f"--> Khớp Temperature T = {opt_t:.4f} trên VAL, áp dụng hiệu chuẩn sang TEST")
        else:
            test_probs = test_probs_uncal

        ev.save_predictions(pred_path(cfg, "test"), test_names, test_true, test_probs)

        test_preds = test_probs.argmax(axis=1)
        test_metrics = ev.compute_metrics(test_true, test_preds, test_probs)
        print(f"--> KẾT QUẢ TEST (Seed {cfg.seed}): Top-1 = {test_metrics['top1']*100:.2f}% | Macro-F1 = {test_metrics['macro_f1']:.4f}")

    # 8. Lưu log CSV và biểu đồ curves
    df_history = pd.DataFrame(history)
    df_history.to_csv(r_dir / "history.csv", index=False)

    curve_path = Path(cfg.curves_dir) / f"{cfg.exp_id}_{cfg.backbone}.png"
    plot_curves(history, curve_path, title=f"{cfg.exp_id} ({cfg.backbone})")

    n_params = model_utils.count_params(model)
    gmacs = model_utils.count_gmacs(model, img_size=cfg.img_size)

    summary = {
        "exp_id": cfg.exp_id,
        "backbone": cfg.backbone,
        "seed": cfg.seed,
        "best_epoch": best_epoch,
        "val_macro_f1": best_macro_f1,
        "avg_epoch_time_s": float(np.mean(epoch_times)),
        "params_m": n_params,
        "gmacs": gmacs,
        "curve_path": str(curve_path),
        "test_metrics": test_metrics,
    }
    return summary


def parse_overrides(pairs: list[str]) -> dict:
    """Biến ['seed=1', 'loss=focal', 'ema_decay=none'] thành dict đúng kiểu dữ liệu Config."""
    cfg_field_types = {f.name: f.type for f in fields(Config)}
    overrides = {}

    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Tham số không hợp lệ: '{pair}', cần có định dạng key=value")
        key, val = pair.split("=", 1)
        key = key.strip()
        val = val.strip()

        if key not in cfg_field_types:
            raise KeyError(f"Trường '{key}' không tồn tại trong Config!")

        target_type = cfg_field_types[key]

        if val.lower() in ("none", "null"):
            overrides[key] = None
        elif val.lower() in ("true", "1"):
            overrides[key] = True
        elif val.lower() in ("false", "0"):
            overrides[key] = False
        elif target_type in (int, "int"):
            overrides[key] = int(val)
        elif target_type in (float, "float"):
            overrides[key] = float(val)
        else:
            # Thử ép float/int nếu được, không thì để chuỗi
            try:
                if "." in val or "e" in val.lower():
                    overrides[key] = float(val)
                else:
                    overrides[key] = int(val)
            except ValueError:
                overrides[key] = val

    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy huấn luyện một cấu hình Lab Day 2.")
    parser.add_argument("--set", nargs="*", default=[], help="Cặp key=value để ghi đè Config")
    args = parser.parse_args()

    overrides = parse_overrides(args.set)
    cfg = Config(**overrides)
    res = run(cfg)
    print("\nKết quả tóm tắt:")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
