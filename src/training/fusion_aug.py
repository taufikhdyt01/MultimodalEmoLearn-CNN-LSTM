"""Synced image+landmark/heatmap augmentation untuk Unified Protocol fusion.

Geometric transformations (hflip, rotation) **harus** diaplikasikan sinkron ke
semua modalitas supaya pasangan (image, landmark) atau (image, heatmap) tetap
semantically konsisten. Photometric (brightness, contrast) hanya untuk image
karena landmark coords & heatmap tidak punya domain photometric.

Augmentation pipeline (B3):
  Synced (image + landmark/heatmap):
    - hflip (p=0.5) + HFLIP_PERM swap untuk landmark
    - rotate ±10° around image center (warpAffine untuk image/heatmap, matrix rot untuk landmark)
  Image-only:
    - brightness ±10%
    - contrast ×0.9-1.1

Pakai bersama `IntermediateFusionDataset` (image + landmark coords) atau
`EarlyFusionDataset` (image + heatmap stacked sebagai channel 4).
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from .landmark_aug import HFLIP_PERM


def _augment_image_photometric(
    img_hwc: np.ndarray,
    rng: np.random.Generator,
    *,
    brightness: float = 0.10,
    contrast: float = 0.10,
) -> np.ndarray:
    """Apply brightness + contrast (image-only)."""
    b = float(rng.uniform(-brightness, brightness))
    c = float(rng.uniform(1.0 - contrast, 1.0 + contrast))
    out = np.clip((img_hwc - 0.5) * c + 0.5 + b, 0.0, 1.0)
    return out.astype(np.float32, copy=False)


def _hflip_image(img_hwc: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(img_hwc[:, ::-1, :])


def _hflip_heatmap(hm_hw: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(hm_hw[:, ::-1])


def _hflip_landmark_136(lm_flat: np.ndarray) -> np.ndarray:
    """Flip x coord ke 1-x, lalu swap left/right indices."""
    pts = lm_flat.reshape(68, 2).copy()
    pts[:, 0] = 1.0 - pts[:, 0]
    pts = pts[HFLIP_PERM]
    return pts.flatten().astype(np.float32)


def _rotate_image(img_hwc: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img_hwc.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(
        np.ascontiguousarray(img_hwc), M, (w, h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
    )


def _rotate_heatmap(hm_hw: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = hm_hw.shape
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(
        np.ascontiguousarray(hm_hw), M, (w, h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
    )


def _rotate_landmark_136(lm_flat: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate normalized landmark coords around (0.5, 0.5).

    Note: rotation matrix di image space (cv2) rotates CCW for positive angle dengan
    convention y-axis-pointing-down. Kita pakai konvensi sama: positive angle = CCW
    di display (y-down). Math: x' = x*cos + y*sin, y' = -x*sin + y*cos (untuk
    convention y-down dan positive=CCW di display).
    cv2.getRotationMatrix2D dengan angle positif rotates CCW di y-down image.
    Untuk sync, landmark juga rotate dengan rule yang sama.
    """
    pts = lm_flat.reshape(68, 2).copy()
    a = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(a), np.sin(a)
    centered = pts - 0.5
    # y-down convention (sesuai image coordinate): positive angle = CCW di display
    # x' = x cos a + y sin a
    # y' = -x sin a + y cos a
    rotated = np.stack([
        centered[:, 0] * cos_a + centered[:, 1] * sin_a,
        -centered[:, 0] * sin_a + centered[:, 1] * cos_a,
    ], axis=1)
    return (rotated + 0.5).flatten().astype(np.float32)


class IntermediateFusionDataset(Dataset):
    """Returns (image_chw, landmark_feature, label).

    Aug behavior tergantung `landmark_is_coord`:
    - landmark_is_coord=True (default, raw 2D coords 136-dim): full synced aug —
      hflip + rotate apply ke image + landmark, brightness/contrast image-only.
    - landmark_is_coord=False (FACS_28 / Blendshape_52 / FB80): landmark adalah
      hand-crafted/learned scalar features, **TIDAK bisa di-aug spasial**.
      Geometric aug image (hflip/rotate) di-skip supaya tidak ada mismatch antara
      image (transformed) dan landmark feature (original). Hanya brightness/contrast
      image-only di-apply saat augment=True. Trade-off konsisten dengan unimodal L2:
      blendshape/FACS part tidak ter-augment.
    """

    def __init__(
        self,
        X_img: np.ndarray,     # (N, H, W, C) float32 in [0,1]
        X_lm: np.ndarray,      # (N, landmark_dim) float32
        y: np.ndarray,         # (N,) int
        *,
        augment: bool = False,
        seed: int = 42,
        p_hflip: float = 0.5,
        rotation_deg: float = 10.0,
        brightness: float = 0.10,
        contrast: float = 0.10,
        landmark_is_coord: bool = True,
    ):
        self.X_img = X_img
        self.X_lm = X_lm
        self.y = y.astype(np.int64)
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.p_hflip = p_hflip
        self.rotation_deg = rotation_deg
        self.brightness = brightness
        self.contrast = contrast
        self.landmark_is_coord = landmark_is_coord

    def __len__(self) -> int:
        return len(self.X_img)

    def __getitem__(self, idx: int):
        img = np.asarray(self.X_img[idx], dtype=np.float32)
        lm = np.asarray(self.X_lm[idx], dtype=np.float32)

        if self.augment:
            if self.landmark_is_coord:
                # Synced geometric aug (coords-only)
                if self.rng.random() < self.p_hflip:
                    img = _hflip_image(img)
                    lm = _hflip_landmark_136(lm)
                angle = float(self.rng.uniform(-self.rotation_deg, self.rotation_deg))
                if abs(angle) > 0.1:
                    img = _rotate_image(img, angle)
                    lm = _rotate_landmark_136(lm, angle)
            # Photometric aug (image-only, always safe)
            img = _augment_image_photometric(
                img, self.rng, brightness=self.brightness, contrast=self.contrast
            )

        img_chw = np.transpose(img, (2, 0, 1)).copy()
        return (
            torch.from_numpy(img_chw),
            torch.from_numpy(lm.copy()),
            int(self.y[idx]),
        )


class EarlyFusionDataset(Dataset):
    """Returns (combined_4chw, label).

    Combined = stack(RGB image + 1-channel heatmap) along channel → 4×H×W.
    Synced augmentation: hflip + rotate apply ke kedua kanal; brightness/contrast
    hanya ke RGB (heatmap dibiarkan).
    """

    def __init__(
        self,
        X_img: np.ndarray,     # (N, H, W, 3) float32 in [0,1]
        X_hm: np.ndarray,      # (N, H, W) float32 in [0,1]
        y: np.ndarray,
        *,
        augment: bool = False,
        seed: int = 42,
        p_hflip: float = 0.5,
        rotation_deg: float = 10.0,
        brightness: float = 0.10,
        contrast: float = 0.10,
    ):
        self.X_img = X_img
        self.X_hm = X_hm
        self.y = y.astype(np.int64)
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.p_hflip = p_hflip
        self.rotation_deg = rotation_deg
        self.brightness = brightness
        self.contrast = contrast

    def __len__(self) -> int:
        return len(self.X_img)

    def __getitem__(self, idx: int):
        img = np.asarray(self.X_img[idx], dtype=np.float32)
        hm = np.asarray(self.X_hm[idx], dtype=np.float32)

        if self.augment:
            if self.rng.random() < self.p_hflip:
                img = _hflip_image(img)
                hm = _hflip_heatmap(hm)
            angle = float(self.rng.uniform(-self.rotation_deg, self.rotation_deg))
            if abs(angle) > 0.1:
                img = _rotate_image(img, angle)
                hm = _rotate_heatmap(hm, angle)
            img = _augment_image_photometric(
                img, self.rng, brightness=self.brightness, contrast=self.contrast
            )

        img_chw = np.transpose(img, (2, 0, 1))           # (3, H, W)
        hm_1hw = hm[np.newaxis, :, :]                     # (1, H, W)
        combined = np.concatenate([img_chw, hm_1hw], axis=0).copy()  # (4, H, W)
        return torch.from_numpy(combined), int(self.y[idx])


def make_balanced_sampler(y: np.ndarray, num_classes: int) -> WeightedRandomSampler:
    """WeightedRandomSampler dengan probability ∝ 1/class_count.

    Identik dengan implementasi di landmark_aug / image_aug.
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
