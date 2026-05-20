"""
Prepare CK+ & JAFFE benchmarks dengan format kompatibel KDEF/RAF-DB
====================================================================

Input (raw): folder per emosi berisi PNG/TIFF.
  data/benchmark/ck+/{Anger,Contempt,Disgust,Fear,Happy,Neutral,Sadness,Surprised}/
  data/benchmark/jaffe/{Anger,Disgust,Fear,Happy,Neutral,Sadness,Surprised}/

Output (sama dengan format kdef_7class / rafdb_7class):
  data/benchmark/ckplus_7class/
    X_{train,val,test}_images.npy       (N,224,224,3) float32 [0,1]
    X_{train,val,test}_landmarks.npy    (N,136) float32 normalized [0,1]
    X_{train,val,test}_facs.npy         (N,28) float32 FACS Euclidean distances
    X_{train,val,test}_heatmaps.npy     (N,224,224) float32 [0,1]
    y_{train,val,test}.npy              (N,) int (label 0-6)
    subjects_{train,val,test}.npy       (N,) str subject ID
    label_map.json, dataset_info.json
  data/benchmark/jaffe_7class/
    (sama)

Filename parsing:
  - CK+:   S<sub>_<seq>_<frame>.png       subject = S<sub>
  - JAFFE: <SUBJ>.<EMO><run>.<idx>.tiff   subject = first token before '.'

Class scheme (7c, matching KDEF/RAF-DB):
  0=neutral, 1=happy, 2=sad, 3=angry, 4=fearful, 5=disgusted, 6=surprised

CK+ Contempt class **di-drop** (tidak ada di benchmark lain).

Split: subject-wise, seed=42.
  - CK+ (118 subjek):  80/10/10
  - JAFFE (10 subjek): 70/10/20  → 7/1/2 subjek

Usage:
    python scripts/prepare_ckplus_jaffe.py                # both datasets
    python scripts/prepare_ckplus_jaffe.py --dataset ckplus
    python scripts/prepare_ckplus_jaffe.py --dataset jaffe
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"
IMG_SIZE = 224
SEED = 42

EMOTIONS_7 = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]
# Map folder name (case-insensitive) → 7c index. Contempt = -1 (drop).
FOLDER_TO_7 = {
    "neutral": 0, "happy": 1, "sad": 2, "sadness": 2,
    "angry": 3, "anger": 3, "fear": 4, "fearful": 4,
    "disgust": 5, "disgusted": 5, "surprise": 6, "surprised": 6,
    "contempt": -1,
}

# 478 → 68 dlib-mapping (sama persis dengan prepare_kdef.py)
LANDMARKS_68_MAP = [
    162, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365,
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
    168, 6, 197, 195, 5, 4, 1, 275, 281,
    33, 160, 158, 133, 153, 144,
    362, 385, 387, 263, 373, 380,
    61, 39, 37, 0, 267, 269, 291, 321, 314, 17, 84, 91, 78,
    82, 13, 312, 308, 317, 14, 87,
]

# FACS pairs (sama dengan run_unified_landmark.py)
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

# Heatmap config (sama dengan generate_landmark_heatmaps.py default)
HEATMAP_SIGMA = 3.0


# --------------------- MediaPipe FaceLandmarker ---------------------
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
        if idx < len(face_lm):
            coords.append(face_lm[idx].x)
            coords.append(face_lm[idx].y)
        else:
            coords.extend([0.0, 0.0])
    return np.array(coords, dtype=np.float32)


def load_image(path, target_size=IMG_SIZE):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (target_size, target_size))


# --------------------- FACS + heatmap ---------------------
def compute_facs_distances(landmarks_flat):
    pts = landmarks_flat.reshape(-1, 68, 2)
    a, b = INTEROCULAR_PAIR
    iod = np.maximum(np.linalg.norm(pts[:, a] - pts[:, b], axis=1), 1e-6)
    out = np.zeros((len(pts), len(FACS_PAIRS)), dtype=np.float32)
    for i, (_, p1, p2) in enumerate(FACS_PAIRS):
        d = np.linalg.norm(pts[:, p1] - pts[:, p2], axis=1)
        out[:, i] = (d / iod).astype(np.float32)
    return out


def gen_heatmap_batch(landmarks, img_size=IMG_SIZE, sigma=HEATMAP_SIGMA, log_every=200):
    N = len(landmarks)
    y_grid, x_grid = np.ogrid[:img_size, :img_size]
    heatmaps = np.zeros((N, img_size, img_size), dtype=np.float32)
    denom = 2.0 * sigma * sigma
    for i in range(N):
        coords = landmarks[i].reshape(-1, 2)
        hm = np.zeros((img_size, img_size), dtype=np.float32)
        for x_norm, y_norm in coords:
            cx = x_norm * img_size
            cy = y_norm * img_size
            g = np.exp(-((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / denom)
            hm = np.maximum(hm, g.astype(np.float32))
        heatmaps[i] = hm
        if (i + 1) % log_every == 0 or (i + 1) == N:
            print(f"    heatmap {i + 1}/{N}")
    return heatmaps


# --------------------- Subject extraction ---------------------
def parse_subject_ckplus(filename):
    """S113_001_00000001.png → 'S113'."""
    m = re.match(r"(S\d+)_", filename)
    return m.group(1) if m else None


def parse_subject_jaffe(filename):
    """KM.AN2.18.tiff → 'KM'."""
    return filename.split(".")[0]


# --------------------- Sample collection ---------------------
def collect_samples_per_emotion(root, parse_subject_fn):
    """Return list of (image_path, emotion_idx (7c), subject_id).
    Skip Contempt (idx = -1).
    """
    samples = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        key = folder.name.lower()
        emo_idx = FOLDER_TO_7.get(key, None)
        if emo_idx is None:
            print(f"  WARN: unknown folder '{folder.name}' — skipping")
            continue
        if emo_idx < 0:
            print(f"  INFO: dropping '{folder.name}' ({sum(1 for _ in folder.iterdir())} images)")
            continue
        files = sorted(folder.iterdir())
        for f in files:
            sub = parse_subject_fn(f.name)
            if sub is None:
                print(f"  WARN: cannot parse subject from {f.name}")
                continue
            samples.append((f, emo_idx, sub))
    return samples


# --------------------- Split ---------------------
def subject_split(unique_subjects, seed, ratios=(0.8, 0.1, 0.1)):
    rng = np.random.RandomState(seed)
    subs = sorted(unique_subjects)
    rng.shuffle(subs)
    n = len(subs)
    n_tr = max(1, int(n * ratios[0]))
    n_va = max(1, int(n * ratios[1]))
    if n_tr + n_va >= n:  # leave at least 1 for test
        n_tr = max(1, n - 2)
        n_va = 1
    return (set(subs[:n_tr]),
            set(subs[n_tr:n_tr + n_va]),
            set(subs[n_tr + n_va:]))


# --------------------- Build + save ---------------------
def build_arrays(samples, landmarker):
    """Process samples → (images, landmarks, labels, subjects). Drop samples
    where face detection failed or image load failed."""
    n = len(samples)
    images = np.zeros((n, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    landmarks = np.zeros((n, 136), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int64)
    subjects = []
    valid = np.ones(n, dtype=bool)
    skip_load = skip_face = 0

    for i, (path, emo_idx, sub) in enumerate(samples):
        img = load_image(path)
        if img is None:
            valid[i] = False
            skip_load += 1
            subjects.append("")
            continue
        lm = extract_landmarks(landmarker, img)
        if lm is None:
            valid[i] = False
            skip_face += 1
            subjects.append("")
            continue
        images[i] = img.astype(np.float32) / 255.0
        landmarks[i] = lm
        labels[i] = emo_idx
        subjects.append(sub)

        if (i + 1) % 100 == 0 or (i + 1) == n:
            print(f"    extract {i + 1}/{n}  (load_fail={skip_load}, no_face={skip_face})")

    subjects = np.array(subjects)
    return images[valid], landmarks[valid], labels[valid], subjects[valid]


def save_dataset(out_dir, split_data, source_info):
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {
        **source_info,
        "num_classes": 7,
        "emotions": EMOTIONS_7,
        "splits": {},
        "image_shape": [IMG_SIZE, IMG_SIZE, 3],
        "landmark_dim": 136,
        "facs_dim": 28,
        "heatmap_sigma": HEATMAP_SIGMA,
    }
    for split_name, (imgs, lms, facs, hms, ys, subs) in split_data.items():
        np.save(out_dir / f"X_{split_name}_images.npy", imgs)
        np.save(out_dir / f"X_{split_name}_landmarks.npy", lms)
        np.save(out_dir / f"X_{split_name}_facs.npy", facs)
        np.save(out_dir / f"X_{split_name}_heatmaps.npy", hms)
        np.save(out_dir / f"y_{split_name}.npy", ys)
        np.save(out_dir / f"subjects_{split_name}.npy", subs)
        counts = Counter(ys.tolist())
        info["splits"][split_name] = {
            "samples": int(len(ys)),
            "subjects": int(len(set(subs.tolist()))),
            "distribution": {EMOTIONS_7[k]: int(v) for k, v in sorted(counts.items())},
        }

    with open(out_dir / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)
    with open(out_dir / "label_map.json", "w") as f:
        json.dump({emo: i for i, emo in enumerate(EMOTIONS_7)}, f, indent=2)

    print(f"\n  Saved: {out_dir}")
    for split_name, d in info["splits"].items():
        dist = ", ".join(f"{k}={v}" for k, v in d["distribution"].items())
        print(f"    {split_name:>5s}: N={d['samples']:>4d}, subjects={d['subjects']}, "
              f"dist=[{dist}]")


# --------------------- Per-dataset orchestration ---------------------
def prepare_dataset(name, raw_root, parse_subject_fn, out_dir, ratios,
                    landmarker, source_info):
    print(f"\n{'='*60}\n  PREPARE {name}\n{'='*60}")
    if not raw_root.exists():
        print(f"  ERROR: raw root not found: {raw_root}")
        return

    samples = collect_samples_per_emotion(raw_root, parse_subject_fn)
    print(f"  Total samples (after dropping unsupported classes): {len(samples)}")

    unique_subjects = sorted({s[2] for s in samples})
    print(f"  Unique subjects: {len(unique_subjects)}")
    train_subs, val_subs, test_subs = subject_split(unique_subjects, SEED, ratios)
    print(f"  Subject split  →  train:{len(train_subs)}  val:{len(val_subs)}  test:{len(test_subs)}")

    samples_by_split = {"train": [], "val": [], "test": []}
    for s in samples:
        sub = s[2]
        if sub in train_subs:
            samples_by_split["train"].append(s)
        elif sub in val_subs:
            samples_by_split["val"].append(s)
        else:
            samples_by_split["test"].append(s)
    for split_name, lst in samples_by_split.items():
        print(f"    {split_name:>5s}: {len(lst)} samples")

    built = {}
    for split_name in ("train", "val", "test"):
        print(f"\n  Building {split_name} ({len(samples_by_split[split_name])} samples) ...")
        imgs, lms, ys, subs = build_arrays(samples_by_split[split_name], landmarker)
        print(f"    extracted {len(imgs)} valid samples")
        if len(imgs) == 0:
            built[split_name] = (imgs,
                                  np.zeros((0, 136), dtype=np.float32),
                                  np.zeros((0, 28), dtype=np.float32),
                                  np.zeros((0, IMG_SIZE, IMG_SIZE), dtype=np.float32),
                                  ys, subs)
            continue
        facs = compute_facs_distances(lms)
        print(f"    computing heatmaps ...")
        hms = gen_heatmap_batch(lms)
        built[split_name] = (imgs, lms, facs, hms, ys, subs)

    save_dataset(out_dir, built, source_info)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ckplus", "jaffe", "both"], default="both")
    ap.add_argument("--ratios", type=float, nargs=3, default=None,
                    help="train val test split ratios (default: 0.8 0.1 0.1 for ckplus, "
                         "0.7 0.1 0.2 for jaffe)")
    args = ap.parse_args()

    landmarker = init_landmarker()

    if args.dataset in ("ckplus", "both"):
        prepare_dataset(
            name="CK+ (7-class, Contempt dropped)",
            raw_root=BENCHMARK_DIR / "ck+",
            parse_subject_fn=parse_subject_ckplus,
            out_dir=BENCHMARK_DIR / "ckplus_7class",
            ratios=tuple(args.ratios) if args.ratios else (0.8, 0.1, 0.1),
            landmarker=landmarker,
            source_info={"dataset": "ckplus", "source": "Extended Cohn-Kanade",
                         "note": "Contempt class dropped to match KDEF/RAF-DB 7-class scheme."},
        )

    if args.dataset in ("jaffe", "both"):
        prepare_dataset(
            name="JAFFE (7-class)",
            raw_root=BENCHMARK_DIR / "jaffe",
            parse_subject_fn=parse_subject_jaffe,
            out_dir=BENCHMARK_DIR / "jaffe_7class",
            ratios=tuple(args.ratios) if args.ratios else (0.7, 0.1, 0.2),
            landmarker=landmarker,
            source_info={"dataset": "jaffe", "source": "Japanese Female Facial Expression"},
        )

    print(f"\n{'='*60}\nDONE!\n{'='*60}")


if __name__ == "__main__":
    main()
