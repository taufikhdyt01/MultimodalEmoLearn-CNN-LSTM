#!/usr/bin/env python3
"""Prepare CK+ Kaggle original dataset (shawon10/ckplus, 981 gambar, 48×48).

Dataset: data/benchmark/ckplus_kaggle_raw/CK+48/
Kelas: anger, contempt, disgust, fear, happy, sadness, surprise (7 kelas, tanpa neutral)
Ini adalah versi yang digunakan oleh Grover & Bansal (2024) dan Singh et al. (2025).

Gambar asli 48×48 grayscale → resize ke 224×224 → convert ke RGB → ekstrak landmark.

Split: subject-wise, seed=42, rasio 80:10:10.
Subject ID diparse dari filename: S010_004_00000017.png → S010.
Jika filename tidak mengandung subject ID (non-standard), fallback ke sample-level split.

Label scheme (Kaggle 7c):
  0=anger, 1=contempt, 2=disgust, 3=fear, 4=happy, 5=sadness, 6=surprise

Output: data/benchmark/ckplus_kaggle_orig/

Usage:
    python scripts/prepare_ckplus_kaggle_orig.py
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

RAW_DIR  = PROJECT_ROOT / "data" / "benchmark" / "ckplus_kaggle_raw" / "CK+48"
OUT_DIR  = PROJECT_ROOT / "data" / "benchmark" / "ckplus_kaggle_orig"
IMG_SIZE = 224
SEED     = 42
HEATMAP_SIGMA = 3.0

EMOTIONS = ["anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]

FOLDER_TO_LABEL = {e: i for i, e in enumerate(EMOTIONS)}

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

LANDMARKS_68_MAP = [
    162, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365,
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
    168, 6, 197, 195, 5, 4, 1, 275, 281,
    33, 160, 158, 133, 153, 144,
    362, 385, 387, 263, 373, 380,
    61, 39, 37, 0, 267, 269, 291, 321, 314, 17, 84, 91, 78,
    82, 13, 312, 308, 317, 14, 87,
]


def init_landmarker():
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    model_path = str(PROJECT_ROOT / "tools" / "face_landmarker_v2_with_blendshapes.task")
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.2,
        min_face_presence_confidence=0.2,
    )
    return vision.FaceLandmarker.create_from_options(options)


def extract_landmarks(landmarker, image_rgb):
    import mediapipe as mp
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None
    face_lm = result.face_landmarks[0]
    coords = []
    for idx in LANDMARKS_68_MAP:
        coords.extend([face_lm[idx].x, face_lm[idx].y] if idx < len(face_lm) else [0.0, 0.0])
    return np.array(coords, dtype=np.float32)


def load_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    # Grayscale → RGB
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE))


def parse_subject(filename):
    m = re.match(r"(S\d+)_", filename)
    return m.group(1) if m else None


def compute_facs(lms):
    pts = lms.reshape(-1, 68, 2)
    a, b = INTEROCULAR_PAIR
    iod = np.maximum(np.linalg.norm(pts[:, a] - pts[:, b], axis=1), 1e-6)
    out = np.zeros((len(pts), len(FACS_PAIRS)), dtype=np.float32)
    for i, (_, p1, p2) in enumerate(FACS_PAIRS):
        out[:, i] = (np.linalg.norm(pts[:, p1] - pts[:, p2], axis=1) / iod).astype(np.float32)
    return out


def gen_heatmaps(lms):
    N = len(lms)
    y_grid, x_grid = np.ogrid[:IMG_SIZE, :IMG_SIZE]
    denom = 2.0 * HEATMAP_SIGMA ** 2
    heatmaps = np.zeros((N, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    for i in range(N):
        hm = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        for x_n, y_n in lms[i].reshape(-1, 2):
            g = np.exp(-((x_grid - x_n * IMG_SIZE) ** 2 + (y_grid - y_n * IMG_SIZE) ** 2) / denom)
            hm = np.maximum(hm, g.astype(np.float32))
        heatmaps[i] = hm
        if (i + 1) % 100 == 0 or (i + 1) == N:
            print(f"    heatmap {i+1}/{N}", flush=True)
    return heatmaps


def collect_samples():
    samples = []
    no_subject = []
    for folder in sorted(RAW_DIR.iterdir()):
        if not folder.is_dir():
            continue
        label = FOLDER_TO_LABEL.get(folder.name.lower())
        if label is None:
            print(f"  WARN: unknown folder {folder.name}")
            continue
        for f in sorted(folder.iterdir()):
            sub = parse_subject(f.name)
            if sub:
                samples.append((f, label, sub))
            else:
                no_subject.append((f, label, f"NOSUB_{folder.name}_{len(no_subject)}"))
    if no_subject:
        print(f"  WARN: {len(no_subject)} files without subject ID — treated as unique subjects")
        samples.extend(no_subject)
    return samples


def subject_split(samples):
    unique_subs = sorted({s[2] for s in samples})
    rng = np.random.RandomState(SEED)
    rng.shuffle(unique_subs)
    n = len(unique_subs)
    n_tr = max(1, int(n * 0.8))
    n_va = max(1, int(n * 0.1))
    if n_tr + n_va >= n:
        n_tr, n_va = max(1, n - 2), 1
    tr_s = set(unique_subs[:n_tr])
    va_s = set(unique_subs[n_tr:n_tr + n_va])
    print(f"  Subjects: total={n}, train={len(tr_s)}, val={len(va_s)}, "
          f"test={n - len(tr_s) - len(va_s)}", flush=True)
    by_split = {"train": [], "val": [], "test": []}
    for s in samples:
        if s[2] in tr_s:
            by_split["train"].append(s)
        elif s[2] in va_s:
            by_split["val"].append(s)
        else:
            by_split["test"].append(s)
    return by_split


def build_arrays(samples, landmarker):
    n = len(samples)
    imgs = np.zeros((n, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    lms  = np.zeros((n, 136), dtype=np.float32)
    ys   = np.zeros(n, dtype=np.int64)
    subs = []
    valid = np.ones(n, dtype=bool)
    skip_load = skip_face = 0

    for i, (path, label, sub) in enumerate(samples):
        img = load_image(path)
        if img is None:
            valid[i] = False; skip_load += 1; subs.append(""); continue
        lm = extract_landmarks(landmarker, img)
        if lm is None:
            valid[i] = False; skip_face += 1; subs.append(""); continue
        imgs[i] = img.astype(np.float32) / 255.0
        lms[i]  = lm
        ys[i]   = label
        subs.append(sub)
        if (i + 1) % 100 == 0 or (i + 1) == n:
            print(f"    extract {i+1}/{n}  (no_load={skip_load}, no_face={skip_face})", flush=True)

    subs = np.array(subs)
    return imgs[valid], lms[valid], ys[valid], subs[valid]


def main():
    print("="*60)
    print("  CK+ Kaggle Original (shawon10/ckplus, 981 img, 7c)")
    print("="*60)
    print(f"  Raw dir: {RAW_DIR}", flush=True)

    samples = collect_samples()
    print(f"  Total samples: {len(samples)}", flush=True)
    for emo, idx in FOLDER_TO_LABEL.items():
        n = sum(1 for s in samples if s[1] == idx)
        print(f"    {emo}: {n}")

    by_split = subject_split(samples)
    for sp, lst in by_split.items():
        print(f"  {sp}: {len(lst)} samples", flush=True)

    landmarker = init_landmarker()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    info = {
        "dataset": "ckplus_kaggle_orig",
        "source": "Kaggle shawon10/ckplus (CK+48, 48×48 resized to 224×224)",
        "note": "Kaggle 7-class: anger, contempt, disgust, fear, happy, sadness, surprise. "
                "Neutral tidak ada. Digunakan oleh Grover 2024, Singh 2025.",
        "num_classes": 7,
        "emotions": EMOTIONS,
        "splits": {},
        "image_shape": [IMG_SIZE, IMG_SIZE, 3],
        "landmark_dim": 136,
        "facs_dim": len(FACS_PAIRS),
        "heatmap_sigma": HEATMAP_SIGMA,
    }

    for split in ("train", "val", "test"):
        print(f"\n  Building {split} ({len(by_split[split])} samples)...", flush=True)
        imgs, lms, ys, subs = build_arrays(by_split[split], landmarker)
        print(f"    valid: {len(imgs)}", flush=True)
        facs = compute_facs(lms)
        print(f"    computing heatmaps...", flush=True)
        hms = gen_heatmaps(lms)

        np.save(OUT_DIR / f"X_{split}_images.npy",    imgs)
        np.save(OUT_DIR / f"X_{split}_landmarks.npy", lms)
        np.save(OUT_DIR / f"X_{split}_heatmaps.npy",  hms)
        np.save(OUT_DIR / f"X_{split}_facs.npy",      facs)
        np.save(OUT_DIR / f"y_{split}.npy",            ys)
        np.save(OUT_DIR / f"subjects_{split}.npy",     subs)

        counts = Counter(ys.tolist())
        dist = {EMOTIONS[k]: int(v) for k, v in sorted(counts.items())}
        info["splits"][split] = {
            "samples": int(len(ys)),
            "subjects": int(len(set(subs.tolist()))),
            "distribution": dist,
        }
        print(f"    {split}: N={len(ys)}, dist={dist}", flush=True)

    with open(OUT_DIR / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)
    with open(OUT_DIR / "label_map.json", "w") as f:
        json.dump({e: i for i, e in enumerate(EMOTIONS)}, f, indent=2)

    total = sum(v["samples"] for v in info["splits"].values())
    print(f"\n  Saved to: {OUT_DIR}, total valid: {total}")
    print("\nDone.")


if __name__ == "__main__":
    main()
