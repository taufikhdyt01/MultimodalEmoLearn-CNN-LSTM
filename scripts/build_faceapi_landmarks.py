#!/usr/bin/env python3
"""
Build face-api.js landmark dataset aligned with existing MediaPipe pipeline.

Source: data/emotions.csv — dump lengkap database (20,110 rows, 80 users).
Filter ke 37 user yang dipakai dataset_frontonly_conf60, lalu match per-sample
via (user_id, soft_emotion_probs) nearest neighbor (tolerance 1e-4).

Landmark JSON face-api.js: {_imgDims, _shift, _positions[68]}
Normalization → comparable to MediaPipe pipeline:
  x_norm = (pos_x - shift_x) / imgDims.width
  y_norm = (pos_y - shift_y) / imgDims.height
→ Range [0, 1] dalam face-crop face-api.js detector.

Output per split (aligned dengan urutan existing dataset):
  X_{split}_faceapi_landmarks.npy  — (N, 136) float32, NaN untuk yang tidak ter-match
  mask_{split}_faceapi.npy         — (N,) bool, True untuk yang ter-match
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "emotions.csv"
DATA_DIR = PROJECT_ROOT / "data" / "dataset_frontonly_conf60"
OUT_DIR = DATA_DIR

MATCH_TOL = 1e-4
EMOTION_COLS = [4, 5, 6, 7, 8, 9, 10]  # neutral..surprised (0-indexed)
LANDMARK_COL = 11
MIN_CONF = 0.6  # filter conf60: max(emotion_probs) >= 0.6


def parse_landmark_json(raw: str):
    """face-api.js double-encoded JSON → (68, 2) normalized to [0,1] in face-crop."""
    if raw is None or raw == "" or raw == "NULL":
        return None
    try:
        d = json.loads(raw)
        if isinstance(d, str):
            d = json.loads(d)
        shift_x = d["_shift"]["_x"]
        shift_y = d["_shift"]["_y"]
        W = d["_imgDims"]["_width"]
        H = d["_imgDims"]["_height"]
        pts = np.array([[p["_x"], p["_y"]] for p in d["_positions"]], dtype=np.float32)
        if pts.shape != (68, 2):
            return None
        pts[:, 0] = (pts[:, 0] - shift_x) / W
        pts[:, 1] = (pts[:, 1] - shift_y) / H
        return pts.flatten()
    except Exception:
        return None


def main():
    # 1. Collect 37 users used by dataset
    users_used = set()
    for split in ("train", "val", "test"):
        arr = np.load(DATA_DIR / f"user_ids_{split}.npy", allow_pickle=True)
        for u in arr:
            users_used.add(str(u))
    print(f"Dataset users (n={len(users_used)}): {sorted(users_used, key=lambda x: int(x))}")

    # 2. Read emotions.csv, filter to 37 users, parse landmarks
    print(f"\nReading {CSV_PATH} ...")
    by_uid = defaultdict(list)  # uid -> [(probs, lm_normalized)]
    skipped_uid = 0
    parse_fail = 0
    rows_kept = 0
    with open(CSV_PATH, "r") as f:
        reader = csv.reader(f, delimiter=";", quotechar='"')
        header = next(reader)
        for row in reader:
            if not row or len(row) < 12:
                continue
            uid = row[1]
            if uid not in users_used:
                skipped_uid += 1
                continue
            try:
                probs = np.array([float(row[c]) for c in EMOTION_COLS], dtype=np.float64)
            except ValueError:
                continue
            if probs.max() < MIN_CONF:
                continue  # filter conf60 explicitly
            lm = parse_landmark_json(row[LANDMARK_COL])
            if lm is None:
                parse_fail += 1
                continue
            by_uid[uid].append((probs, lm))
            rows_kept += 1
    print(f"  rows kept (37 users, valid landmark): {rows_kept}")
    print(f"  rows skipped (non-target user): {skipped_uid}")
    print(f"  parse failures: {parse_fail}")
    for u in sorted(by_uid.keys(), key=lambda x: int(x)):
        print(f"    {u}: {len(by_uid[u])}")

    # 3. Match per split (strict + nearest-neighbor fallback)
    print("\nMatching to dataset...")
    for split in ("train", "val", "test"):
        soft = np.load(DATA_DIR / f"y_{split}_soft.npy").astype(np.float64)
        uids = np.load(DATA_DIR / f"user_ids_{split}.npy", allow_pickle=True)
        N = len(soft)
        out_lm = np.full((N, 136), np.nan, dtype=np.float32)
        match_kind = np.zeros(N, dtype=np.int8)  # 0=miss, 1=strict, 2=nn_fallback
        max_fallback_dist = 0.0

        for i in range(N):
            uid = str(uids[i])
            if uid not in by_uid:
                continue
            target = soft[i]
            best_dist = float("inf")
            best_lm = None
            for probs, lm in by_uid[uid]:
                d = float(np.abs(probs - target).max())
                if d < best_dist:
                    best_dist = d
                    best_lm = lm
            if best_lm is None:
                continue
            out_lm[i] = best_lm
            if best_dist < MATCH_TOL:
                match_kind[i] = 1
            else:
                match_kind[i] = 2
                max_fallback_dist = max(max_fallback_dist, best_dist)

        mask = match_kind > 0
        n_strict = int((match_kind == 1).sum())
        n_fallback = int((match_kind == 2).sum())
        n_miss = int((match_kind == 0).sum())

        out_path = OUT_DIR / f"X_{split}_faceapi_landmarks.npy"
        mask_path = OUT_DIR / f"mask_{split}_faceapi.npy"
        kind_path = OUT_DIR / f"match_kind_{split}_faceapi.npy"
        np.save(out_path, out_lm)
        np.save(mask_path, mask)
        np.save(kind_path, match_kind)
        print(f"  {split}: total={N}  strict={n_strict}  nn_fallback={n_fallback}  miss={n_miss}"
              f"  max_fallback_dist={max_fallback_dist:.3e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
