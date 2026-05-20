#!/usr/bin/env python3
"""
Re-extract MediaPipe features (3D landmark + blendshapes + head pose) untuk
seluruh face-crop di dataset_frontonly_conf60. Replicate gaya feature set
Bachtiar 2024 dengan pure MediaPipe source.

Output per split:
  X_{split}_mp3d_landmarks.npy  — (N, 68, 3) 3D landmark (68 points, x/y/z),
                                  pakai LANDMARK_68_INDICES mapping dari
                                  face_crop_landmark.py (dlib 68 convention)
  X_{split}_mp_blendshapes.npy  — (N, 52) blendshape coefficients
  X_{split}_mp_headpose.npy     — (N, 3) pitch, yaw, roll (radians) dari
                                  facial_transformation_matrix (rotasi 3x3)

NaN untuk frame dengan deteksi gagal (akan di-impute saat training).

Usage:
  python scripts/extract_mp_features.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "dataset_frontonly_conf60"
MODEL_PATH = str(PROJECT_ROOT / "tools" / "face_landmarker_v2_with_blendshapes.task")

# 478 -> 68 mapping (sama dengan face_crop_landmark.py — dlib convention)
LANDMARK_68_INDICES = [
    162, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365,
    70, 63, 105, 66, 107,
    336, 296, 334, 293, 300,
    168, 6, 197, 195, 5, 4, 1, 275, 281,
    33, 160, 158, 133, 153, 144,
    362, 385, 387, 263, 373, 380,
    61, 39, 37, 0, 267, 269, 291, 321, 314, 17, 84, 91,
    78, 82, 13, 312, 308, 317, 14, 87,
]
assert len(LANDMARK_68_INDICES) == 68

N_BLENDSHAPES = 52


def rotation_matrix_to_euler(R: np.ndarray) -> tuple:
    """3x3 rotation matrix → (pitch, yaw, roll) radians, intrinsic XYZ."""
    # Standard ZYX decomposition
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        pitch = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(-R[2, 0], sy)
        roll = np.arctan2(R[1, 0], R[0, 0])
    else:
        pitch = np.arctan2(-R[1, 2], R[1, 1])
        yaw = np.arctan2(-R[2, 0], sy)
        roll = 0.0
    return float(pitch), float(yaw), float(roll)


def create_landmarker():
    opts = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    return vision.FaceLandmarker.create_from_options(opts)


def process_split(split: str):
    print(f"\n=== {split} ===")
    img_path = DATA_DIR / f"X_{split}_images.npy"
    imgs = np.load(img_path, mmap_mode="r")
    N = len(imgs)
    print(f"  {N} images, shape {imgs.shape[1:]}")

    lm3d = np.full((N, 68, 3), np.nan, dtype=np.float32)
    bs = np.full((N, N_BLENDSHAPES), np.nan, dtype=np.float32)
    hp = np.full((N, 3), np.nan, dtype=np.float32)
    mask = np.zeros(N, dtype=bool)

    landmarker = create_landmarker()
    t0 = time.time()
    fail = 0
    for i in range(N):
        if i % 500 == 0 and i > 0:
            elapsed = time.time() - t0
            eta = elapsed / i * (N - i)
            print(f"  {i}/{N}  elapsed={elapsed:.0f}s  eta={eta:.0f}s  fail={fail}")
        img_u8 = (np.asarray(imgs[i]) * 255).astype(np.uint8)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_u8)
        res = landmarker.detect(mp_img)
        if not res.face_landmarks:
            fail += 1
            continue
        # 478 landmarks → 68
        pts = res.face_landmarks[0]
        # Normalize coords: pts have .x .y in [0,1] normalized, .z is relative depth
        sub = np.array(
            [[pts[idx].x, pts[idx].y, pts[idx].z] for idx in LANDMARK_68_INDICES],
            dtype=np.float32,
        )
        lm3d[i] = sub
        if res.face_blendshapes:
            scores = [b.score for b in res.face_blendshapes[0]]
            bs[i] = np.array(scores[:N_BLENDSHAPES], dtype=np.float32)
        if res.facial_transformation_matrixes:
            M = res.facial_transformation_matrixes[0]  # 4x4
            R = M[:3, :3]
            hp[i] = rotation_matrix_to_euler(R)
        mask[i] = True

    elapsed = time.time() - t0
    print(f"  done: {N - fail}/{N} success ({fail} failed), {elapsed:.0f}s")

    np.save(DATA_DIR / f"X_{split}_mp3d_landmarks.npy", lm3d)
    np.save(DATA_DIR / f"X_{split}_mp_blendshapes.npy", bs)
    np.save(DATA_DIR / f"X_{split}_mp_headpose.npy", hp)
    np.save(DATA_DIR / f"mask_{split}_mp_features.npy", mask)
    print(f"  saved: X_{split}_mp3d_landmarks.npy  X_{split}_mp_blendshapes.npy  X_{split}_mp_headpose.npy  mask_{split}_mp_features.npy")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None,
                    help="Override DATA_DIR (default: data/dataset_frontonly_conf60)")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip splits yang sudah punya X_{split}_mp_blendshapes.npy")
    args = ap.parse_args()

    global DATA_DIR
    if args.data_dir is not None:
        DATA_DIR = Path(args.data_dir).resolve()
    print(f"DATA_DIR: {DATA_DIR}")

    for split in args.splits:
        bs_path = DATA_DIR / f"X_{split}_mp_blendshapes.npy"
        if args.skip_existing and bs_path.exists():
            print(f"\n=== {split} === [skip: {bs_path.name} exists]")
            continue
        img_path = DATA_DIR / f"X_{split}_images.npy"
        if not img_path.exists():
            print(f"\n=== {split} === [skip: {img_path.name} not found]")
            continue
        process_split(split)
    print("\nDone.")


if __name__ == "__main__":
    main()
