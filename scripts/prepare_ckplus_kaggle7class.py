#!/usr/bin/env python3
"""Prepare CK+ dataset — Kaggle 7-class scheme (neutral dropped, contempt included).

Scheme ini digunakan oleh paper:
  - Grover & Bansal (2024): anger, contempt, disgust, fear, happiness, sadness, surprise
  - Singh et al. (2025) / MMSAD: sama
  - Gautam & Seeja (2023): sama

Strategi efisien: reuse data yang sudah diproses di ckplus_7class (6 kelas
yang overlap), lalu proses 18 gambar contempt via MediaPipe, gabungkan.

Input:
  data/benchmark/ckplus_7class/    (sudah diproses, 7c tanpa contempt)
  data/benchmark/ck+/Contempt/     (raw gambar contempt, 18 file)

Output:
  data/benchmark/ckplus_kaggle7class/
    X_{train,val,test}_images.npy
    X_{train,val,test}_heatmaps.npy
    X_{train,val,test}_landmarks.npy
    X_{train,val,test}_facs.npy
    y_{train,val,test}.npy
    subjects_{train,val,test}.npy
    dataset_info.json, label_map.json

Label scheme (Kaggle 7c):
  0=anger, 1=contempt, 2=disgust, 3=fear, 4=happy, 5=sadness, 6=surprised

Usage:
    python scripts/prepare_ckplus_kaggle7class.py
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

BENCHMARK_DIR   = PROJECT_ROOT / "data" / "benchmark"
SRC_DIR         = BENCHMARK_DIR / "ckplus_7class"
CONTEMPT_DIR    = BENCHMARK_DIR / "ck+" / "Contempt"
OUT_DIR         = BENCHMARK_DIR / "ckplus_kaggle7class"
IMG_SIZE        = 224
SEED            = 42
HEATMAP_SIGMA   = 3.0

# Kaggle 7-class label scheme (NO neutral)
EMOTIONS_KAGGLE = ["anger", "contempt", "disgust", "fear", "happy", "sadness", "surprised"]

# Map dari label ckplus_7class (neutral=0,happy=1,sad=2,angry=3,fearful=4,disgusted=5,surprised=6)
# ke Kaggle 7c (anger=0,contempt=1,disgust=2,fear=3,happy=4,sadness=5,surprised=6)
# neutral (0) → -1 (drop)
OLD_TO_NEW = {
    0: -1,  # neutral → drop
    1:  4,  # happy   → happy(4)
    2:  5,  # sad     → sadness(5)
    3:  0,  # angry   → anger(0)
    4:  3,  # fearful → fear(3)
    5:  2,  # disgusted → disgust(2)
    6:  6,  # surprised → surprised(6)
}

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
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
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
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE))


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
    return heatmaps


def parse_subject(filename):
    m = re.match(r"(S\d+)_", filename)
    return m.group(1) if m else "S000"


def remap_existing():
    """Load ckplus_7class, drop neutral, remap labels ke Kaggle scheme."""
    print("\n[1] Remapping existing ckplus_7class (drop neutral, remap labels)...")
    parts = {}
    for split in ("train", "val", "test"):
        imgs = np.load(SRC_DIR / f"X_{split}_images.npy")
        lms  = np.load(SRC_DIR / f"X_{split}_landmarks.npy")
        hms  = np.load(SRC_DIR / f"X_{split}_heatmaps.npy")
        facs = np.load(SRC_DIR / f"X_{split}_facs.npy")
        ys   = np.load(SRC_DIR / f"y_{split}.npy")
        subs = np.load(SRC_DIR / f"subjects_{split}.npy")

        new_y = np.array([OLD_TO_NEW[int(l)] for l in ys])
        keep  = new_y >= 0
        n_before, n_after = len(ys), keep.sum()
        print(f"  {split}: {n_before} → {n_after} (dropped {n_before - n_after} neutral)")
        parts[split] = (imgs[keep], lms[keep], hms[keep], facs[keep],
                        new_y[keep].astype(np.int64), subs[keep])
    return parts


def process_contempt(landmarker):
    """Process 18 contempt images from raw folder, assign subject IDs."""
    print("\n[2] Processing contempt images from raw folder...")
    files = sorted(CONTEMPT_DIR.iterdir())
    imgs, lms_list, subs_list = [], [], []
    skip = 0
    for f in files:
        img = load_image(f)
        if img is None:
            skip += 1
            continue
        lm = extract_landmarks(landmarker, img)
        if lm is None:
            skip += 1
            print(f"  WARN: no face in {f.name}")
            continue
        imgs.append(img.astype(np.float32) / 255.0)
        lms_list.append(lm)
        subs_list.append(parse_subject(f.name))

    print(f"  Processed: {len(imgs)} valid, {skip} skipped")
    imgs = np.stack(imgs)
    lms  = np.stack(lms_list)
    facs = compute_facs(lms)
    hms  = gen_heatmaps(lms)
    ys   = np.full(len(imgs), 1, dtype=np.int64)  # contempt = 1
    subs = np.array(subs_list)
    return imgs, lms, hms, facs, ys, subs


def split_contempt_by_subject(imgs, lms, hms, facs, ys, subs):
    """Split contempt samples by subject, consistent with SEED=42."""
    unique_subs = sorted(set(subs.tolist()))
    rng = np.random.RandomState(SEED)
    rng.shuffle(unique_subs)
    n = len(unique_subs)
    # 80/10/10 subject split
    n_tr = max(1, int(n * 0.8))
    n_va = max(1, int(n * 0.1))
    if n_tr + n_va >= n:
        n_tr, n_va = max(1, n - 2), 1
    tr_s = set(unique_subs[:n_tr])
    va_s = set(unique_subs[n_tr:n_tr + n_va])

    parts = {"train": [], "val": [], "test": []}
    for i, s in enumerate(subs):
        if s in tr_s:
            parts["train"].append(i)
        elif s in va_s:
            parts["val"].append(i)
        else:
            parts["test"].append(i)

    result = {}
    for split, idxs in parts.items():
        if not idxs:
            # edge case: put contempt in train if empty split
            idxs = list(range(len(imgs)))[:1] if split == "train" else []
        if idxs:
            idx = np.array(idxs)
            result[split] = (imgs[idx], lms[idx], hms[idx], facs[idx],
                             ys[idx], subs[idx])
        else:
            result[split] = tuple(np.zeros((0, *arr.shape[1:]), dtype=arr.dtype)
                                  for arr in [imgs, lms, hms, facs, ys, subs])
    return result


def merge_and_save(existing, contempt_splits):
    """Concatenate existing + contempt per split, save to OUT_DIR."""
    print("\n[3] Merging and saving...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    info = {
        "dataset": "ckplus",
        "source": "Extended Cohn-Kanade",
        "note": "Kaggle 7-class scheme: neutral dropped, contempt included. "
                "Matches Grover 2024, Singh 2025, Gautam 2023.",
        "num_classes": 7,
        "emotions": EMOTIONS_KAGGLE,
        "splits": {},
        "image_shape": [IMG_SIZE, IMG_SIZE, 3],
        "landmark_dim": 136,
        "facs_dim": len(FACS_PAIRS),
        "heatmap_sigma": HEATMAP_SIGMA,
    }

    for split in ("train", "val", "test"):
        e_imgs, e_lms, e_hms, e_facs, e_ys, e_subs = existing[split]
        c = contempt_splits.get(split)

        if c and len(c[0]) > 0:
            c_imgs, c_lms, c_hms, c_facs, c_ys, c_subs = c
            imgs = np.concatenate([e_imgs, c_imgs])
            lms  = np.concatenate([e_lms,  c_lms])
            hms  = np.concatenate([e_hms,  c_hms])
            facs = np.concatenate([e_facs, c_facs])
            ys   = np.concatenate([e_ys,   c_ys])
            subs = np.concatenate([e_subs, c_subs])
        else:
            imgs, lms, hms, facs, ys, subs = e_imgs, e_lms, e_hms, e_facs, e_ys, e_subs

        np.save(OUT_DIR / f"X_{split}_images.npy",    imgs)
        np.save(OUT_DIR / f"X_{split}_landmarks.npy", lms)
        np.save(OUT_DIR / f"X_{split}_heatmaps.npy",  hms)
        np.save(OUT_DIR / f"X_{split}_facs.npy",      facs)
        np.save(OUT_DIR / f"y_{split}.npy",            ys)
        np.save(OUT_DIR / f"subjects_{split}.npy",     subs)

        counts = Counter(ys.tolist())
        dist = {EMOTIONS_KAGGLE[k]: int(v) for k, v in sorted(counts.items())}
        n_subjects = len(set(subs.tolist()))
        info["splits"][split] = {"samples": int(len(ys)), "subjects": n_subjects,
                                 "distribution": dist}
        dist_str = ", ".join(f"{k}={v}" for k, v in dist.items())
        print(f"  {split:>5s}: N={len(ys):>4d}, subjects={n_subjects}, [{dist_str}]")

    with open(OUT_DIR / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)
    with open(OUT_DIR / "label_map.json", "w") as f:
        json.dump({e: i for i, e in enumerate(EMOTIONS_KAGGLE)}, f, indent=2)

    total = sum(v["samples"] for v in info["splits"].values())
    print(f"\n  Saved to: {OUT_DIR}")
    print(f"  Total samples: {total}")


def main():
    print("="*60)
    print("  CK+ Kaggle 7-class (contempt in, neutral out)")
    print("="*60)

    existing = remap_existing()

    landmarker = init_landmarker()
    c_imgs, c_lms, c_hms, c_facs, c_ys, c_subs = process_contempt(landmarker)

    contempt_splits = split_contempt_by_subject(c_imgs, c_lms, c_hms, c_facs, c_ys, c_subs)
    for split, data in contempt_splits.items():
        print(f"  contempt → {split}: {len(data[0])} samples")

    merge_and_save(existing, contempt_splits)
    print("\nDone.")


if __name__ == "__main__":
    main()
