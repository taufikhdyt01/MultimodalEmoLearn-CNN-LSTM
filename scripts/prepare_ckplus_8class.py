#!/usr/bin/env python3
"""Prepare CK+ 8-class dataset (semua kelas: neutral + contempt).

Strategi: gabungkan ckplus_7class (neutral in, contempt out) dengan
data contempt dari ckplus_kaggle7class (yang sudah diproses MediaPipe).
Contempt diberi label baru = 7.

Label scheme (8c):
  0=neutral, 1=happy, 2=sad, 3=angry, 4=fearful, 5=disgusted, 6=surprised, 7=contempt

Output: data/benchmark/ckplus_8class/

Usage:
    python scripts/prepare_ckplus_8class.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"

SRC_7C      = BENCHMARK_DIR / "ckplus_7class"
SRC_KAGGLE  = BENCHMARK_DIR / "ckplus_kaggle7class"
OUT_DIR     = BENCHMARK_DIR / "ckplus_8class"

EMOTIONS_8 = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised", "contempt"]
# contempt label di kaggle7class = 1, akan di-remap ke 7
CONTEMPT_OLD_LABEL = 1
CONTEMPT_NEW_LABEL = 7


def main():
    print("="*60)
    print("  CK+ 8-class (neutral + contempt, semua kelas)")
    print("="*60)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    info = {
        "dataset": "ckplus",
        "source": "Extended Cohn-Kanade",
        "note": "8-class scheme: semua kelas termasuk neutral dan contempt. "
                "Label 0-6 dari ckplus_7class, contempt=7 dari ckplus_kaggle7class.",
        "num_classes": 8,
        "emotions": EMOTIONS_8,
        "splits": {},
        "image_shape": [224, 224, 3],
        "landmark_dim": 136,
        "facs_dim": 28,
        "heatmap_sigma": 3.0,
    }

    for split in ("train", "val", "test"):
        # Load ckplus_7class (labels 0-6)
        imgs7 = np.load(SRC_7C / f"X_{split}_images.npy")
        lms7  = np.load(SRC_7C / f"X_{split}_landmarks.npy")
        hms7  = np.load(SRC_7C / f"X_{split}_heatmaps.npy")
        facs7 = np.load(SRC_7C / f"X_{split}_facs.npy")
        ys7   = np.load(SRC_7C / f"y_{split}.npy")
        subs7 = np.load(SRC_7C / f"subjects_{split}.npy")

        # Load contempt dari kaggle7class, ambil hanya label contempt (=1)
        imgs_k = np.load(SRC_KAGGLE / f"X_{split}_images.npy")
        lms_k  = np.load(SRC_KAGGLE / f"X_{split}_landmarks.npy")
        hms_k  = np.load(SRC_KAGGLE / f"X_{split}_heatmaps.npy")
        facs_k = np.load(SRC_KAGGLE / f"X_{split}_facs.npy")
        ys_k   = np.load(SRC_KAGGLE / f"y_{split}.npy")
        subs_k = np.load(SRC_KAGGLE / f"subjects_{split}.npy")

        mask_contempt = (ys_k == CONTEMPT_OLD_LABEL)
        n_contempt = mask_contempt.sum()

        if n_contempt > 0:
            imgs_c = imgs_k[mask_contempt]
            lms_c  = lms_k[mask_contempt]
            hms_c  = hms_k[mask_contempt]
            facs_c = facs_k[mask_contempt]
            ys_c   = np.full(n_contempt, CONTEMPT_NEW_LABEL, dtype=np.int64)
            subs_c = subs_k[mask_contempt]

            imgs = np.concatenate([imgs7, imgs_c])
            lms  = np.concatenate([lms7,  lms_c])
            hms  = np.concatenate([hms7,  hms_c])
            facs = np.concatenate([facs7, facs_c])
            ys   = np.concatenate([ys7,   ys_c])
            subs = np.concatenate([subs7, subs_c])
        else:
            imgs, lms, hms, facs, ys, subs = imgs7, lms7, hms7, facs7, ys7, subs7

        np.save(OUT_DIR / f"X_{split}_images.npy",    imgs)
        np.save(OUT_DIR / f"X_{split}_landmarks.npy", lms)
        np.save(OUT_DIR / f"X_{split}_heatmaps.npy",  hms)
        np.save(OUT_DIR / f"X_{split}_facs.npy",      facs)
        np.save(OUT_DIR / f"y_{split}.npy",            ys)
        np.save(OUT_DIR / f"subjects_{split}.npy",     subs)

        counts = Counter(ys.tolist())
        dist = {EMOTIONS_8[k]: int(v) for k, v in sorted(counts.items())}
        info["splits"][split] = {
            "samples": int(len(ys)),
            "subjects": int(len(set(subs.tolist()))),
            "distribution": dist,
        }
        dist_str = ", ".join(f"{k}={v}" for k, v in dist.items())
        print(f"  {split:>5s}: N={len(ys):>4d}, subjects={len(set(subs.tolist()))}, [{dist_str}]")

    with open(OUT_DIR / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)
    with open(OUT_DIR / "label_map.json", "w") as f:
        json.dump({e: i for i, e in enumerate(EMOTIONS_8)}, f, indent=2)

    total = sum(v["samples"] for v in info["splits"].values())
    print(f"\n  Saved to: {OUT_DIR}")
    print(f"  Total samples: {total}")
    print("\nDone.")


if __name__ == "__main__":
    main()
