"""Image augmentation utility — analog landmark_aug.py untuk Unified Protocol.

Pakai protokol ini untuk image-based unimodal (CNN scratch, CNN_TL) supaya
B3 fair dibandingkan dengan landmark-based unified sweep.

Unified scenario definition:
  - B1: no aug, no class weight, no weighted sampler (baseline)
  - B2: WeightedRandomSampler (class-balanced sampling) + no augmentation
  - B3: WeightedRandomSampler + true per-batch random augmentation

Aug pipeline (per __getitem__ random):
  - hflip horizontal (p=0.5)
  - rotate ±10° around image center, reflect-pad boundary
  - brightness shift ±10% (additive in [0,1] space, clipped)
  - contrast scaling ×0.9–1.1 (multiplicative around 0.5 midpoint)
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


def augment_image_hwc(
    img: np.ndarray,
    rng: np.random.Generator,
    *,
    p_hflip: float = 0.5,
    rotation_deg: float = 10.0,
    brightness: float = 0.10,
    contrast: float = 0.10,
) -> np.ndarray:
    """Apply random augmentation to HWC float32 image in [0, 1].

    Returns augmented HWC float32, same shape.
    """
    h, w = img.shape[:2]
    out = img

    if rng.random() < p_hflip:
        out = out[:, ::-1, :]

    angle = float(rng.uniform(-rotation_deg, rotation_deg))
    if abs(angle) > 0.1:
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        out = cv2.warpAffine(
            np.ascontiguousarray(out), M, (w, h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
        )

    b = float(rng.uniform(-brightness, brightness))
    c = float(rng.uniform(1.0 - contrast, 1.0 + contrast))
    out = np.clip((out - 0.5) * c + 0.5 + b, 0.0, 1.0).astype(np.float32, copy=False)
    return np.ascontiguousarray(out)


class AugmentingImageDataset(Dataset):
    """Dataset yang return (image_chw_float, label_long).

    `X` expected shape (N, H, W, C) float32 in [0, 1]. Augmentasi diterapkan
    per `__getitem__` call kalau `augment=True` → setiap epoch model lihat
    variant berbeda (true on-the-fly).
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        augment: bool = False,
        seed: int = 42,
    ):
        self.X = X
        self.y = np.asarray(y, dtype=np.int64)
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        img = np.asarray(self.X[idx], dtype=np.float32)
        if self.augment:
            img = augment_image_hwc(img, self.rng)
        img_chw = np.transpose(img, (2, 0, 1)).copy()
        return torch.from_numpy(img_chw), int(self.y[idx])


def make_balanced_sampler(y: np.ndarray, num_classes: int) -> WeightedRandomSampler:
    """WeightedRandomSampler dengan probability ∝ 1/class_count.

    Identik dengan landmark_aug.make_balanced_sampler — di-duplicate di sini
    supaya image script tidak perlu import dari landmark module.
    """
    counts = np.bincount(y.astype(np.int64), minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv = 1.0 / counts
    sample_weights = inv[y.astype(np.int64)]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(y),
        replacement=True,
    )
