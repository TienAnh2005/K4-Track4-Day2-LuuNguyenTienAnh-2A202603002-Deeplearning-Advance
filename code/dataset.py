"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Tuân thủ nghiêm ngặt quy tắc chia dữ liệu S1-S6 (README.md, mục 2.1).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
try:
    import torchvision.transforms as T
except ImportError:
    T = None

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EXPECTED_TOTAL_IMAGES = 17509


def load_split(labels_dir: str | Path, fold: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1).

    Mỗi file có cột `Filename, Label, Species`. Trả về ba DataFrame.
    KHÔNG sửa, lọc hay chia lại dữ liệu.
    """
    labels_path = Path(labels_dir)
    train_file = labels_path / f"train_subset{fold}.csv"
    val_file = labels_path / f"val_subset{fold}.csv"
    test_file = labels_path / f"test_subset{fold}.csv"

    if not train_file.exists() or not val_file.exists() or not test_file.exists():
        raise FileNotFoundError(
            f"Không tìm thấy đủ file split fold {fold} tại '{labels_path}'. "
            f"Cần: {train_file.name}, {val_file.name}, {test_file.name}"
        )

    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        for col in ["Filename", "Label", "Species"]:
            if col not in df.columns:
                raise ValueError(f"File {name} thiếu cột bắt buộc '{col}'")

    return train_df, val_df, test_df


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). In ra và trả về dict số liệu.

    1. Số ảnh mỗi tập và số ảnh mỗi lớp trong từng tập (kỳ vọng xấp xỉ 60/20/20)
    2. Giao của từng cặp tập theo Filename phải RỖNG (train∩val, train∩test, val∩test)
    3. Hợp ba tập phải bằng đúng 17.509 ảnh
    4. Mọi Filename đều tồn tại trong `images_dir` (nếu thư mục tồn tại)
    """
    img_path = Path(images_dir)
    set_train = set(train_df["Filename"])
    set_val = set(val_df["Filename"])
    set_test = set(test_df["Filename"])

    # 1. Kiểm tra giao giữa các tập (phải rỗng)
    inter_train_val = set_train.intersection(set_val)
    inter_train_test = set_train.intersection(set_test)
    inter_val_test = set_val.intersection(set_test)

    assert len(inter_train_val) == 0, f"RÒ RỈ: train và val giao nhau {len(inter_train_val)} ảnh!"
    assert len(inter_train_test) == 0, f"RÒ RỈ: train và test giao nhau {len(inter_train_test)} ảnh!"
    assert len(inter_val_test) == 0, f"RÒ RỈ: val và test giao nhau {len(inter_val_test)} ảnh!"

    # 2. Kiểm tra tổng số ảnh
    union_all = set_train.union(set_val).union(set_test)
    assert len(union_all) == EXPECTED_TOTAL_IMAGES, (
        f"Tổng số ảnh 3 tập là {len(union_all)}, khác kỳ vọng {EXPECTED_TOTAL_IMAGES} ảnh!"
    )

    # 3. Kiểm tra file trên đĩa
    if img_path.exists():
        missing_files = [f for f in union_all if not (img_path / f).exists()]
        assert len(missing_files) == 0, (
            f"Có {len(missing_files)} file ảnh trong CSV không tìm thấy tại '{img_path}'. Ví dụ: {missing_files[:5]}"
        )

    # 4. Thống kê theo lớp
    stats = {
        "n": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
            "total": len(union_all),
        },
        "ratio": {
            "train": len(train_df) / len(union_all),
            "val": len(val_df) / len(union_all),
            "test": len(test_df) / len(union_all),
        },
        "per_class": {
            "train": train_df["Label"].value_counts().sort_index().to_dict(),
            "val": val_df["Label"].value_counts().sort_index().to_dict(),
            "test": test_df["Label"].value_counts().sort_index().to_dict(),
        },
        "overlap": {
            "train_val": len(inter_train_val),
            "train_test": len(inter_train_test),
            "val_test": len(inter_val_test),
        },
    }
    return stats


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """Tạo transform theo đúng quy chuẩn.

    Train:
      - 'basic': RandomResizedCrop + RandomHorizontalFlip + ToTensor + Normalize
      - 'color': basic + ColorJitter
      - 'randaug': RandomResizedCrop + RandomHorizontalFlip + RandAugment + ToTensor + Normalize
      - 'trivial': RandomResizedCrop + RandomHorizontalFlip + TrivialAugmentWide + ToTensor + Normalize
    Val/test:
      - Resize 256 -> CenterCrop(img_size) (nếu img_size==256 thì Resize 256) + ToTensor + Normalize
    """
    if T is None:
        raise ImportError("Cần cài đặt torchvision: pip install torchvision")

    if train:
        t_list = []
        t_list.append(T.RandomResizedCrop(img_size, scale=(0.7, 1.0)))
        t_list.append(T.RandomHorizontalFlip(p=0.5))

        if aug == "color":
            t_list.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05))
        elif aug == "randaug":
            t_list.append(T.RandAugment(num_ops=2, magnitude=9))
        elif aug == "trivial":
            t_list.append(T.TrivialAugmentWide())
        elif aug != "basic":
            # Mặc định về basic nếu chuỗi lạ
            pass

        t_list.extend([
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        return T.Compose(t_list)
    else:
        if img_size == 256:
            return T.Compose([
                T.Resize((256, 256)),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def _safe_build_val_transforms(img_size: int = 224) -> T.Compose:
    return build_transforms(train=False, img_size=img_size)


class DeepWeedsDataset(Dataset):
    """Dataset đọc ảnh từ `images_dir` theo DataFrame (Filename, Label, Species).

    __getitem__(i) trả về (image_tensor, int(label), filename_str).
    """

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform: Optional[Callable] = None):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.filenames = self.df["Filename"].tolist()
        self.labels = self.df["Label"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int, str]:
        fname = self.filenames[i]
        label = self.labels[i]
        img_path = self.images_dir / fname
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, label, fname


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2) -> DataLoader:
    """Tạo DataLoader tương thích cả Windows và Linux/Colab."""
    dataset = DeepWeedsDataset(df=df, images_dir=images_dir, transform=transform)

    batch_sampler = None
    shuffle = False

    if train:
        if sampler == "balanced":
            class_counts = df["Label"].value_counts().to_dict()
            sample_weights = [1.0 / class_counts[y] for y in dataset.labels]
            sampler_obj = WeightedRandomSampler(
                weights=torch.as_tensor(sample_weights, dtype=torch.double),
                num_samples=len(sample_weights),
                replacement=True,
            )
            shuffle = False
            batch_sampler = sampler_obj
        else:
            shuffle = True
    else:
        shuffle = False

    # Tránh drop_last nếu số mẫu ít hơn batch_size
    drop_last = train and (len(dataset) > batch_size)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if batch_sampler is None else False,
        sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
    )
    return loader
