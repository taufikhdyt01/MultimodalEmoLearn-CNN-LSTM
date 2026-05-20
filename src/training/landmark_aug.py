"""
Unified landmark augmentation + class balancing protocol.

Best practices:
1. **True per-batch random aug** — augmentasi diapplikasikan di `__getitem__`,
   parameter random per call → setiap epoch model lihat variant berbeda.
2. **Semantically-correct hflip** — saat horizontal flip, swap left↔right
   landmark index supaya geometri tetap valid (mata kanan → posisi mata kiri).
3. **Separated concern** — balancing pakai WeightedRandomSampler (oversample
   minoritas secara probabilistik), bukan duplikasi explicit.

Unified scenario definition:
  - B1: no aug, no class weight, no weighted sampler (baseline)
  - B2: WeightedRandomSampler (class-balanced sampling) + no augmentation
  - B3: WeightedRandomSampler + true per-batch random augmentation

Pakai protokol ini untuk semua arch × feature × source supaya fair comparison.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


# dlib 68-point left/right index mapping (untuk hflip yang semantically correct).
# Setelah hflip horizontal, landmark posisi kiri-kanan tertukar — perlu swap
# index supaya tetap mata kanan di idx 36-41, mata kiri di 42-47, dst.
HFLIP_INDEX_MAP = {
    # Jaw 0-16: 0↔16, 1↔15, ..., 7↔9, 8↔8 (center)
    0: 16, 1: 15, 2: 14, 3: 13, 4: 12, 5: 11, 6: 10, 7: 9, 8: 8,
    9: 7, 10: 6, 11: 5, 12: 4, 13: 3, 14: 2, 15: 1, 16: 0,
    # Right brow 17-21 ↔ Left brow 22-26 (reversed pair)
    17: 26, 18: 25, 19: 24, 20: 23, 21: 22,
    22: 21, 23: 20, 24: 19, 25: 18, 26: 17,
    # Nose vertical 27-30 stay (center column)
    27: 27, 28: 28, 29: 29, 30: 30,
    # Nose horizontal 31-35: 31↔35, 32↔34, 33↔33
    31: 35, 32: 34, 33: 33, 34: 32, 35: 31,
    # Right eye 36-41 ↔ Left eye 42-47 (with reversed inner order)
    36: 45, 37: 44, 38: 43, 39: 42, 40: 47, 41: 46,
    42: 39, 43: 38, 44: 37, 45: 36, 46: 41, 47: 40,
    # Outer lip 48-59: 48↔54, 49↔53, 50↔52, 51↔51, 55↔59, 56↔58, 57↔57
    48: 54, 49: 53, 50: 52, 51: 51, 52: 50, 53: 49, 54: 48,
    55: 59, 56: 58, 57: 57, 58: 56, 59: 55,
    # Inner lip 60-67: 60↔64, 61↔63, 62↔62, 65↔67, 66↔66
    60: 64, 61: 63, 62: 62, 63: 61, 64: 60,
    65: 67, 66: 66, 67: 65,
}
HFLIP_PERM = np.array([HFLIP_INDEX_MAP[i] for i in range(68)], dtype=np.int64)
assert len(set(HFLIP_PERM)) == 68, "permutation not bijective"


def augment_landmark_136(
    lm_flat: np.ndarray,
    rng: np.random.Generator,
    *,
    p_hflip: float = 0.5,
    rotation_deg: float = 10.0,
    scale_range: tuple = (0.95, 1.05),
    translation: float = 0.02,
    noise_sigma: float = 0.005,
) -> np.ndarray:
    """Apply random augmentation to a single 136-dim landmark vector.

    Augmentations applied (each independently random per call):
      - hflip with probability p_hflip + left/right index swap
      - Random rotation [-rotation_deg, +rotation_deg] around face center
      - Random scale [scale_range[0], scale_range[1]]
      - Random translation [-translation, +translation]
      - Gaussian noise per coordinate sigma=noise_sigma

    Args:
      lm_flat: (136,) normalized landmark coords (x0, y0, x1, y1, ..., x67, y67)
      rng: numpy random generator (for reproducibility)
    Returns:
      (136,) augmented landmark.
    """
    pts = lm_flat.reshape(68, 2).copy()

    # 1. Hflip + index swap (semantically correct)
    if rng.random() < p_hflip:
        pts[:, 0] = 1.0 - pts[:, 0]
        pts = pts[HFLIP_PERM]

    # 2. Rotation around (0.5, 0.5)
    angle = np.deg2rad(rng.uniform(-rotation_deg, rotation_deg))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    centered = pts - 0.5
    rotated = np.stack([
        centered[:, 0] * cos_a - centered[:, 1] * sin_a,
        centered[:, 0] * sin_a + centered[:, 1] * cos_a,
    ], axis=1)
    pts = rotated + 0.5

    # 3. Scale around center
    scale = rng.uniform(scale_range[0], scale_range[1])
    pts = (pts - 0.5) * scale + 0.5

    # 4. Translation
    tx = rng.uniform(-translation, translation)
    ty = rng.uniform(-translation, translation)
    pts[:, 0] += tx
    pts[:, 1] += ty

    # 5. Per-coordinate Gaussian noise
    if noise_sigma > 0:
        pts += rng.normal(0, noise_sigma, size=pts.shape)

    return pts.flatten().astype(np.float32)


class AugmentingLandmarkDataset(Dataset):
    """Wrap a landmark dataset with TRUE per-call random augmentation.

    Setiap call ke __getitem__ apply transformasi random baru. Setup ini:
    - landmark_aug=True → augment 136-dim landmark
    - feature_fn=None → return raw 136-dim landmark
    - feature_fn=facs_fn → apply feature_fn AFTER augmentation (FACS distance dll)

    Untuk feature seperti blendshape yang TIDAK dari landmark (di-extract terpisah),
    pakai `BlendshapeAugmentingDataset` atau setup berbeda — augmentasi tidak
    bisa diaplikasikan ke blendshape karena tidak ada transformasi yang
    semantically valid di space ARKit coefficient.
    """

    def __init__(
        self,
        X_landmarks: np.ndarray,
        y: np.ndarray,
        *,
        augment: bool = True,
        feature_fn=None,
        seed: int = 42,
    ):
        """Args:
          X_landmarks: (N, 136) landmark coords
          y: (N,) labels
          augment: kalau True, apply random aug per call
          feature_fn: optional callable(lm_flat) → feature vector (e.g., FACS dist)
        """
        self.X = X_landmarks.astype(np.float32)
        self.y = y.astype(np.int64)
        self.augment = augment
        self.feature_fn = feature_fn
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        lm = self.X[i]
        if self.augment:
            lm = augment_landmark_136(lm, self.rng)
        if self.feature_fn is not None:
            feat = self.feature_fn(lm[None, :])[0]
        else:
            feat = lm
        return torch.from_numpy(feat), torch.tensor(self.y[i])


class FeatureConcatDataset(Dataset):
    """Concat landmark-derived feature + extra static feature (e.g., blendshape).

    Augmentation hanya di-apply ke landmark part. Extra feature tetap sama
    (mis. blendshape per-sample dari MediaPipe extractor, tidak di-augment).
    """

    def __init__(
        self,
        X_landmarks: np.ndarray,
        X_extra: np.ndarray,
        y: np.ndarray,
        *,
        augment: bool = True,
        feature_fn=None,
        seed: int = 42,
    ):
        self.X_lm = X_landmarks.astype(np.float32)
        self.X_ex = X_extra.astype(np.float32)
        self.y = y.astype(np.int64)
        self.augment = augment
        self.feature_fn = feature_fn
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        lm = self.X_lm[i]
        if self.augment:
            lm = augment_landmark_136(lm, self.rng)
        if self.feature_fn is not None:
            feat_lm = self.feature_fn(lm[None, :])[0]
        else:
            feat_lm = lm
        feat = np.concatenate([feat_lm, self.X_ex[i]])
        return torch.from_numpy(feat.astype(np.float32)), torch.tensor(self.y[i])


def make_balanced_sampler(y: np.ndarray, num_classes: int) -> WeightedRandomSampler:
    """WeightedRandomSampler: probabilitas inverse-frequency per class.

    Setiap epoch model lihat batch dengan distribusi kelas yang ~balanced,
    tanpa duplikasi explicit data. Setiap sample bisa muncul 0×, 1×, atau 2×
    per epoch tergantung kelas-nya.
    """
    counts = np.bincount(y, minlength=num_classes)
    class_weights = 1.0 / np.maximum(counts, 1)
    sample_weights = class_weights[y]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(y),  # one epoch = N draws
        replacement=True,
    )
