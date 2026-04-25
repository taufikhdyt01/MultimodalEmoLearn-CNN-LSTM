"""
Compute Liliana 2019 style 20-dim geometric features dari 68-point landmarks.

10 facial components × 2 metrik (eccentricity + distance ratio) = 20-dim GF.

Reference: Liliana et al. (2019) — "Fuzzy emotion: a natural approach to automatic
facial expression recognition from psychological perspective using fuzzy system",
Cognitive Processing (Springer), Table 3.

Input:  {data_dir}/X_{split}_landmarks.npy  (N, 136) = 68×2 normalized
Output: {data_dir}/X_{split}_geometric.npy  (N, 20)

Scale-invariant via face height (chin ↔ brow) + face width (ear ↔ ear).

Usage:
    python src/preprocessing/compute_geometric_features.py
    python src/preprocessing/compute_geometric_features.py --data-dir data/dataset_frontonly_conf60_3class_augmented
"""
import argparse
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 68-point iBug/dlib mapping → 10 facial components (Liliana 2019 Table 3)
COMPONENTS = {
    'gf1_left_eyebrow':  (17, 21, 19, 27),
    'gf2_right_eyebrow': (22, 26, 24, 27),
    'gf3_inner_eyebrow': (21, 22, 21, 22),
    'gf4_left_eye':      (36, 39, 37, 41),
    'gf5_right_eye':     (42, 45, 43, 47),
    'gf6_nose':          (31, 35, 30, 33),
    'gf7_upper_lip':     (48, 54, 50, 62),
    'gf8_lower_lip':     (48, 54, 66, 57),
    'gf9_inner_mouth':   (60, 64, 62, 66),
    'gf10_outer_mouth':  (48, 54, 51, 57),
}


def face_scale(pts, img_size=224):
    """Return (face_width_px, face_height_px). pts: (68, 2) scaled to img_size."""
    fw = abs(pts[16, 0] - pts[0, 0])
    fh = abs(pts[8, 1] - pts[27, 1])
    return max(fw, 1e-6), max(fh, 1e-6)


def compute_gf(coords_136, img_size=224):
    """Compute 20-dim GF vector dari 136-d landmark (normalized 0-1)."""
    pts = coords_136.reshape(-1, 2) * img_size

    fw, fh = face_scale(pts, img_size)

    gf = np.zeros(20, dtype=np.float32)
    for i, (p1, p2, p3, p4) in enumerate(COMPONENTS.values()):
        x1, y1 = pts[p1]; x2, y2 = pts[p2]
        x3, y3 = pts[p3]; x4, y4 = pts[p4]

        # Eccentricity (elliptic shape of facial component)
        a = abs(x2 - x1) / 2 / fw
        b = abs(y3 - y4) / 2 / fh
        if a > 1e-6 and a >= b:
            e = np.sqrt(max(a ** 2 - b ** 2, 0.0)) / a
        else:
            e = 0.0
        gf[i] = float(e)

        # Distance ratio (vertical / horizontal span, scale-invariant)
        num = abs(y3 - y4) / fh
        den = abs(x2 - x1) / fw
        gf[i + 10] = float(num / den) if den > 1e-6 else 0.0
    return gf


def compute_batch(coords_batch, img_size=224):
    return np.stack([compute_gf(c, img_size) for c in coords_batch], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=str,
                    default='data/dataset_frontonly_conf60',
                    help='Directory containing X_{split}_landmarks.npy')
    args = ap.parse_args()

    data_dir = PROJECT_ROOT / args.data_dir
    print(f'Data dir: {data_dir}')

    for split in ['train', 'val', 'test']:
        lm_path = data_dir / f'X_{split}_landmarks.npy'
        if not lm_path.exists():
            print(f'  [SKIP] {lm_path.name} missing')
            continue
        coords = np.load(lm_path)
        gf = compute_batch(coords)
        out = data_dir / f'X_{split}_geometric.npy'
        np.save(out, gf)
        print(f'  {split}: {coords.shape} → {gf.shape} saved (mean={gf.mean():.3f}, '
              f'std={gf.std():.3f}, min={gf.min():.3f}, max={gf.max():.3f})')


if __name__ == '__main__':
    main()
