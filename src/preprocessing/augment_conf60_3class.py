"""
Generate 3-class augmented dataset dari conf60 untuk scenario B3.

REMAP_3 valence mapping (Russell 1980 circumplex simplification):
  positive = happy + surprised              (label 0)
  neutral  = neutral                        (label 1)
  negative = sad + angry + fearful + disgusted  (label 2)

Minority classes (positive ~690, negative ~414) di-augment sampai TARGET_MIN.
Majority neutral (~5691) tidak di-augment.

Augmentation: hflip, rotation, brightness adjust (match src/preprocessing/augment_minority.py).

Usage:
    python src/preprocessing/augment_conf60_3class.py
    python src/preprocessing/augment_conf60_3class.py --target-min 1500

Output:
    data/dataset_frontonly_conf60_3class_augmented/
      ├── X_train_{images,landmarks,heatmaps}.npy   (augmented)
      ├── X_{val,test}_*.npy                        (copy dari source, no aug)
      ├── y_train.npy (3-class hard labels)
      ├── y_{val,test}.npy (3-class hard labels)
      ├── class_weights.json
      └── dataset_info.json
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / 'data' / 'dataset_frontonly_conf60'
OUT_DIR = PROJECT_ROOT / 'data' / 'dataset_frontonly_conf60_3class_augmented'

# 7-class → 3-class valence mapping (per user spec)
# 7-class order: [neutral, happy, sad, angry, fearful, disgusted, surprised]
REMAP_3 = np.array([1, 0, 2, 2, 2, 2, 0], dtype=np.int64)
EMOTIONS_3 = ['positive', 'neutral', 'negative']

TECHNIQUES = ['hflip', 'rotate_pos', 'rotate_neg', 'bright_up', 'bright_down', 'flip_rot']


def augment_image(img, technique, rng):
    h, w = img.shape[:2]
    if technique == 'hflip':
        return np.fliplr(img).copy()
    if technique == 'rotate_pos':
        angle = rng.uniform(5, 15)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    if technique == 'rotate_neg':
        angle = rng.uniform(-15, -5)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    if technique == 'bright_up':
        factor = rng.uniform(1.05, 1.20)
        return np.clip(img * factor, 0, 1).astype(np.float32)
    if technique == 'bright_down':
        factor = rng.uniform(0.80, 0.95)
        return np.clip(img * factor, 0, 1).astype(np.float32)
    if technique == 'flip_rot':
        flipped = np.fliplr(img).copy()
        angle = rng.uniform(-10, 10)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(flipped, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    raise ValueError(f'Unknown technique: {technique}')


def augment_landmark(lm_136, technique):
    """Landmark 136-d (normalized [0,1]): match image augmentation geometrically."""
    pts = lm_136.reshape(-1, 2).copy()  # (68, 2)
    if technique == 'hflip':
        pts[:, 0] = 1.0 - pts[:, 0]
        return pts.flatten().astype(np.float32)
    if technique in ('bright_up', 'bright_down'):
        return lm_136.astype(np.float32)
    if technique in ('rotate_pos', 'rotate_neg', 'flip_rot'):
        # Rotation around center (0.5, 0.5)
        if technique == 'rotate_pos':
            angle = np.deg2rad(10)
        elif technique == 'rotate_neg':
            angle = np.deg2rad(-10)
        else:  # flip_rot
            pts[:, 0] = 1.0 - pts[:, 0]  # flip first
            angle = np.deg2rad(5)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        centered = pts - 0.5
        rotated = np.stack([
            centered[:, 0] * cos_a - centered[:, 1] * sin_a,
            centered[:, 0] * sin_a + centered[:, 1] * cos_a,
        ], axis=1)
        return (rotated + 0.5).flatten().astype(np.float32)
    raise ValueError(technique)


def augment_heatmap(hm, technique, rng):
    """Heatmap 224×224 (single channel): geometric augmentation only."""
    if technique == 'hflip':
        return np.fliplr(hm).copy()
    if technique in ('bright_up', 'bright_down'):
        return hm.astype(np.float32)
    if technique in ('rotate_pos', 'rotate_neg', 'flip_rot'):
        h, w = hm.shape
        if technique == 'rotate_pos':
            angle = 10
        elif technique == 'rotate_neg':
            angle = -10
        else:
            hm = np.fliplr(hm).copy()
            angle = 5
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(hm, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    raise ValueError(technique)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target-min', type=int, default=1500,
                    help='Minimum sample count per kelas minoritas setelah augmentasi')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    print(f'Source: {SRC_DIR}')
    print(f'Output: {OUT_DIR}')
    print(f'Target min per class: {args.target_min}')

    # ── Load train data ──
    X_img = np.load(SRC_DIR / 'X_train_images.npy')
    X_lm  = np.load(SRC_DIR / 'X_train_landmarks.npy')
    X_hm  = np.load(SRC_DIR / 'X_train_heatmaps.npy')
    y7    = np.load(SRC_DIR / 'y_train.npy')
    y3    = REMAP_3[y7]
    N0 = len(y3)
    counter = Counter(y3.tolist())
    print(f'\nBefore augmentation ({N0} train samples):')
    for c in range(3):
        print(f'  {EMOTIONS_3[c]:>9} ({c}): {counter[c]}')

    # ── Augment minority ──
    aug_imgs, aug_lms, aug_hms, aug_y = [X_img], [X_lm], [X_hm], [y3]

    for cls in range(3):
        n_cur = counter[cls]
        if n_cur >= args.target_min:
            continue
        need = args.target_min - n_cur
        idx_cls = np.where(y3 == cls)[0]
        print(f'\n  Augmenting {EMOTIONS_3[cls]}: {n_cur} → {args.target_min} (+{need})')

        for i in range(need):
            src_idx = idx_cls[rng.randint(len(idx_cls))]
            tech = TECHNIQUES[rng.randint(len(TECHNIQUES))]
            aug_imgs.append(augment_image(X_img[src_idx], tech, rng)[None, ...])
            aug_lms.append(augment_landmark(X_lm[src_idx], tech)[None, ...])
            aug_hms.append(augment_heatmap(X_hm[src_idx], tech, rng)[None, ...])
            aug_y.append(np.array([cls], dtype=np.int64))

    X_img_aug = np.concatenate(aug_imgs, axis=0)
    X_lm_aug  = np.concatenate(aug_lms, axis=0)
    X_hm_aug  = np.concatenate(aug_hms, axis=0)
    y_aug     = np.concatenate(aug_y, axis=0)

    # Shuffle
    perm = rng.permutation(len(y_aug))
    X_img_aug = X_img_aug[perm]
    X_lm_aug  = X_lm_aug[perm]
    X_hm_aug  = X_hm_aug[perm]
    y_aug     = y_aug[perm]

    post_counter = Counter(y_aug.tolist())
    print(f'\nAfter augmentation ({len(y_aug)} train samples):')
    for c in range(3):
        print(f'  {EMOTIONS_3[c]:>9} ({c}): {post_counter[c]}')

    # ── Save train ──
    np.save(OUT_DIR / 'X_train_images.npy', X_img_aug)
    np.save(OUT_DIR / 'X_train_landmarks.npy', X_lm_aug)
    np.save(OUT_DIR / 'X_train_heatmaps.npy', X_hm_aug)
    np.save(OUT_DIR / 'y_train.npy', y_aug)

    # ── Copy val/test (apply REMAP_3, no augmentation) ──
    for split in ['val', 'test']:
        X_img_s = np.load(SRC_DIR / f'X_{split}_images.npy')
        X_lm_s  = np.load(SRC_DIR / f'X_{split}_landmarks.npy')
        X_hm_s  = np.load(SRC_DIR / f'X_{split}_heatmaps.npy')
        y_s     = REMAP_3[np.load(SRC_DIR / f'y_{split}.npy')]
        np.save(OUT_DIR / f'X_{split}_images.npy', X_img_s)
        np.save(OUT_DIR / f'X_{split}_landmarks.npy', X_lm_s)
        np.save(OUT_DIR / f'X_{split}_heatmaps.npy', X_hm_s)
        np.save(OUT_DIR / f'y_{split}.npy', y_s)
        print(f'  {split}: {len(y_s)} samples, class dist: {Counter(y_s.tolist())}')

    # ── Class weights (from augmented train) ──
    counts = np.array([post_counter[c] for c in range(3)], dtype=np.float32)
    total = counts.sum()
    weights = total / (3 * counts)
    weights_norm = weights / weights.sum() * 3
    with open(OUT_DIR / 'class_weights.json', 'w') as f:
        json.dump({
            'emotions': EMOTIONS_3,
            'counts': counts.tolist(),
            'weights_array': weights_norm.tolist(),
        }, f, indent=2)
    print(f'\nClass weights (normalized): {weights_norm.tolist()}')

    # ── Dataset info ──
    with open(OUT_DIR / 'dataset_info.json', 'w') as f:
        json.dump({
            'source': 'dataset_frontonly_conf60',
            'remap': '3-class valence (positive=happy+surprised, neutral, negative=sad+angry+fearful+disgusted)',
            'target_min': args.target_min,
            'train_total': int(len(y_aug)),
            'train_distribution': {EMOTIONS_3[c]: int(post_counter[c]) for c in range(3)},
            'emotions': EMOTIONS_3,
            'remap_indices': REMAP_3.tolist(),
        }, f, indent=2)

    print(f'\nSaved to: {OUT_DIR}')


if __name__ == '__main__':
    main()
