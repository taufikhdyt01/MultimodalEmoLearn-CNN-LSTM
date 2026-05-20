#!/usr/bin/env python3
"""Derive FACS Euclidean distance (28-dim) dari raw landmark 136-dim.

One-shot util untuk extend secondary datasets dengan FACS distance feature.
Reuse `compute_facs_distances` logic dari `run_unified_landmark.py` (FACS_PAIRS
+ interocular normalization).

Output: `X_{train,val,test}_facs.npy` di setiap target data dir.

Usage:
  # Derive untuk semua benchmark datasets yang lengkap
  python scripts/derive_facs_distance.py --data-dirs data/benchmark/kdef_7class data/benchmark/kdef_4class data/benchmark/rafdb_7class data/benchmark/rafdb_4class

  # Atau primer (cek konsistensi vs unified results existing)
  python scripts/derive_facs_distance.py --data-dirs data/dataset_frontonly_conf60
"""
import argparse
from pathlib import Path

import numpy as np

# FACS pair definition (identik dengan run_unified_landmark.py)
FACS_PAIRS = [
    ("AU1_inner_brow_R", 21, 39), ("AU1_inner_brow_L", 22, 42),
    ("AU2_outer_brow_R", 17, 36), ("AU2_outer_brow_L", 26, 45),
    ("AU4_frown_inner_brows", 21, 22), ("AU4_brow_to_eye_R", 19, 37),
    ("AU4_brow_to_eye_L", 24, 44),
    ("AU5_eye_open_R_1", 37, 41), ("AU5_eye_open_R_2", 38, 40),
    ("AU5_eye_open_L_1", 43, 47), ("AU5_eye_open_L_2", 44, 46),
    ("AU6_cheek_R", 36, 31), ("AU6_cheek_L", 45, 35),
    ("AU9_nose_width", 31, 35), ("AU9_nose_bridge", 27, 30),
    ("AU10_nose_to_upperlip", 33, 51), ("AU10_lip_top_to_nose_R", 50, 33),
    ("AU10_lip_top_to_nose_L", 52, 33),
    ("AU12_corner_R_to_eye", 48, 36), ("AU12_corner_L_to_eye", 54, 45),
    ("AU15_corner_R_to_chin", 48, 8), ("AU15_corner_L_to_chin", 54, 8),
    ("AU20_mouth_width_outer", 48, 54), ("AU23_mouth_width_inner", 60, 64),
    ("AU25_lip_open_outer", 51, 57), ("AU25_lip_open_inner", 62, 66),
    ("AU26_nose_to_chin", 33, 8), ("face_height", 8, 27),
]
INTEROCULAR_PAIR = (36, 45)


def compute_facs_distances(landmarks_flat: np.ndarray) -> np.ndarray:
    """(N, 136) → (N, 28) FACS Euclidean distances normalized by interocular."""
    pts = landmarks_flat.reshape(-1, 68, 2)
    a, b = INTEROCULAR_PAIR
    iod = np.maximum(np.linalg.norm(pts[:, a] - pts[:, b], axis=1), 1e-6)
    out = np.zeros((len(pts), len(FACS_PAIRS)), dtype=np.float32)
    for i, (_, p1, p2) in enumerate(FACS_PAIRS):
        d = np.linalg.norm(pts[:, p1] - pts[:, p2], axis=1)
        out[:, i] = (d / iod).astype(np.float32)
    return out


def derive_for_data_dir(data_dir: Path, splits=("train", "val", "test"),
                       landmark_file_pattern: str = "X_{split}_landmarks.npy",
                       output_pattern: str = "X_{split}_facs.npy",
                       force: bool = False):
    """Read raw landmarks per split, derive FACS distance, save."""
    derived = []
    for split in splits:
        lm_fp = data_dir / landmark_file_pattern.format(split=split)
        if not lm_fp.exists():
            print(f"  [skip] {split}: {lm_fp.name} not found")
            continue
        out_fp = data_dir / output_pattern.format(split=split)
        if out_fp.exists() and not force:
            existing = np.load(out_fp)
            print(f"  [exists] {split}: {out_fp.name} shape={existing.shape} (use --force to overwrite)")
            derived.append(split)
            continue

        lm = np.load(lm_fp).astype(np.float32)
        if lm.ndim != 2 or lm.shape[1] != 136:
            print(f"  [skip] {split}: {lm_fp.name} shape {lm.shape} != (N, 136)")
            continue
        facs = compute_facs_distances(lm)
        np.save(out_fp, facs)
        print(f"  [saved] {split}: {lm_fp.name} {lm.shape} → {out_fp.name} {facs.shape}")
        derived.append(split)
    return derived


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dirs", nargs="+", required=True,
                    help="Daftar data dir yang punya X_{split}_landmarks.npy")
    ap.add_argument("--landmark-source", default="mediapipe",
                    choices=["mediapipe", "faceapi"],
                    help="Mana yang dipakai: mediapipe = X_*_landmarks.npy, faceapi = X_*_faceapi_landmarks.npy")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--force", action="store_true", help="Overwrite existing output")
    args = ap.parse_args()

    if args.landmark_source == "mediapipe":
        in_pattern = "X_{split}_landmarks.npy"
        out_pattern = "X_{split}_facs.npy"
    else:
        in_pattern = "X_{split}_faceapi_landmarks.npy"
        out_pattern = "X_{split}_faceapi_facs.npy"

    for d in args.data_dirs:
        data_dir = Path(d)
        print(f"\n=== {data_dir} ({args.landmark_source}) ===")
        if not data_dir.exists():
            print(f"  [skip] dir not found")
            continue
        derive_for_data_dir(data_dir, splits=args.splits,
                            landmark_file_pattern=in_pattern,
                            output_pattern=out_pattern,
                            force=args.force)

    print("\nDone.")


if __name__ == "__main__":
    main()
