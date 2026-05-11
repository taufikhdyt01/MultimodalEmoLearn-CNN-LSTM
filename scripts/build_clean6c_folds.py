"""Build 5-fold user-level CV definition for clean 6-class dataset.

Strategy:
- Filter out hand-occluded samples (using deploy/data_cleaning/hand_occlusion_*.npz)
- Drop fearful class (had 0 clean samples)
- Remap labels: orig {0..6} \ {4=fearful} -> new {0..5}
- Greedy fold assignment: distribute users across 5 folds while balancing minority class
  representation per fold.

Output:
- data/dataset_frontonly_conf60_clean6c/folds.json
- data/dataset_frontonly_conf60_clean6c/clean_index.npz
    (orig_split, orig_index, user_id, label_6c) per clean sample, length=5482
- data/dataset_frontonly_conf60_clean6c/dataset_info.json
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "data" / "dataset_frontonly_conf60"
CLEAN_DIR = ROOT / "deploy" / "data_cleaning"
OUT_DIR = ROOT / "data" / "dataset_frontonly_conf60_clean6c"

ORIG_EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]
DROP_LABEL = 4  # fearful
KEEP_LABELS = [c for c in range(7) if c != DROP_LABEL]  # [0,1,2,3,5,6]
NEW_EMOTIONS = [ORIG_EMOTIONS[c] for c in KEEP_LABELS]   # 6 classes
LABEL_REMAP = {orig: new for new, orig in enumerate(KEEP_LABELS)}  # {0:0,1:1,2:2,3:3,5:4,6:5}

N_FOLDS = 5
SEED = 42


def build_clean_index():
    """Concat clean samples from train/val/test into a single flat index."""
    rows = []  # (orig_split, orig_index, user_id, label_6c)
    for sp in ("train", "val", "test"):
        y = np.load(SRC_DIR / f"y_{sp}.npy")
        u = np.load(SRC_DIR / f"user_ids_{sp}.npy").astype(str)
        occ = np.load(CLEAN_DIR / f"hand_occlusion_{sp}.npz")["occluded_mask"]
        keep = (~occ) & (y != DROP_LABEL)
        idx = np.where(keep)[0]
        for i in idx:
            rows.append((sp, int(i), str(u[i]), int(LABEL_REMAP[int(y[i])])))
    print(f"Clean index built: {len(rows)} samples")
    return rows


def stratified_user_folds(rows):
    """Assign each user to one of N_FOLDS folds.

    Two-stage algorithm:
    1. For each minority class (angry, disgusted, surprised) in ascending order of total count,
       snake-draft holders to folds that have the fewest samples of that class so far.
       This guarantees the rarest classes are spread across folds first.
    2. Remaining users (only neutral/happy/sad) are snake-drafted by total count,
       assigning to the smallest fold first to balance fold sizes.
    """
    user_class = defaultdict(lambda: np.zeros(len(NEW_EMOTIONS), dtype=np.int32))
    for _, _, u, y in rows:
        user_class[u][y] += 1
    users = sorted(user_class.keys(), key=lambda x: int(x) if x.isdigit() else x)

    minority_classes_idx = [3, 4, 5]  # angry, disgusted, surprised (in NEW_EMOTIONS)
    minority_names = [NEW_EMOTIONS[i] for i in minority_classes_idx]

    rng = random.Random(SEED)
    fold_users: list[list[str]] = [[] for _ in range(N_FOLDS)]
    fold_counts = [np.zeros(len(NEW_EMOTIONS), dtype=np.int32) for _ in range(N_FOLDS)]
    assigned: set[str] = set()

    # Stage 1: distribute minority holders, rarest class first
    totals = sum((user_class[u] for u in users), np.zeros(len(NEW_EMOTIONS), dtype=np.int32))
    # Order minority classes by ascending total — rarest first (e.g. disgusted=13 before angry=25 before surprised=30)
    minority_order = sorted(minority_classes_idx, key=lambda i: int(totals[i]))
    print(f"Minority distribution order: {[NEW_EMOTIONS[i] for i in minority_order]} "
          f"(totals: {[int(totals[i]) for i in minority_order]})")

    for cls_idx in minority_order:
        holders = [u for u in users if u not in assigned and user_class[u][cls_idx] > 0]
        # Sort holders by sample count desc, tiebreak random
        holders.sort(key=lambda u: (-int(user_class[u][cls_idx]), rng.random()))
        for u in holders:
            # Assign to fold with fewest of this class so far; tiebreak by smaller fold size
            scores = [(int(fold_counts[k][cls_idx]), int(fold_counts[k].sum()), rng.random(), k)
                      for k in range(N_FOLDS)]
            scores.sort()
            k = scores[0][3]
            fold_users[k].append(u)
            fold_counts[k] += user_class[u]
            assigned.add(u)

    # Stage 2: remaining users (no minorities) — balance fold sizes
    remaining = [u for u in users if u not in assigned]
    remaining.sort(key=lambda u: (-int(user_class[u].sum()), rng.random()))
    for u in remaining:
        scores = [(int(fold_counts[k].sum()), rng.random(), k) for k in range(N_FOLDS)]
        scores.sort()
        k = scores[0][2]
        fold_users[k].append(u)
        fold_counts[k] += user_class[u]
        assigned.add(u)

    return users, fold_users, user_class


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = build_clean_index()
    users, fold_users, user_class = stratified_user_folds(rows)

    # Save flat clean index as compressed npz
    orig_splits = np.array([r[0] for r in rows])
    orig_indices = np.array([r[1] for r in rows], dtype=np.int32)
    user_ids = np.array([r[2] for r in rows])
    labels_6c = np.array([r[3] for r in rows], dtype=np.int8)
    np.savez_compressed(
        OUT_DIR / "clean_index.npz",
        orig_split=orig_splits,
        orig_index=orig_indices,
        user_id=user_ids,
        label_6c=labels_6c,
    )
    print(f"clean_index.npz -> {OUT_DIR / 'clean_index.npz'}")

    # Build folds.json
    folds = {}
    for k in range(N_FOLDS):
        test_users = sorted(fold_users[k], key=lambda x: int(x) if x.isdigit() else x)
        train_users = sorted(
            [u for k2 in range(N_FOLDS) if k2 != k for u in fold_users[k2]],
            key=lambda x: int(x) if x.isdigit() else x,
        )
        # Class counts for this fold
        test_counts = np.zeros(len(NEW_EMOTIONS), dtype=np.int32)
        for u in test_users:
            test_counts += user_class[u]
        train_counts = np.zeros(len(NEW_EMOTIONS), dtype=np.int32)
        for u in train_users:
            train_counts += user_class[u]
        folds[f"fold_{k}"] = {
            "test_users": test_users,
            "train_users": train_users,
            "test_size": int(test_counts.sum()),
            "train_size": int(train_counts.sum()),
            "test_class_counts": {NEW_EMOTIONS[i]: int(test_counts[i]) for i in range(len(NEW_EMOTIONS))},
            "train_class_counts": {NEW_EMOTIONS[i]: int(train_counts[i]) for i in range(len(NEW_EMOTIONS))},
        }
    (OUT_DIR / "folds.json").write_text(json.dumps(folds, indent=2), encoding="utf-8")
    print(f"folds.json -> {OUT_DIR / 'folds.json'}")

    info = {
        "source_dataset": "data/dataset_frontonly_conf60/",
        "cleaning": "MediaPipe HandLandmarker (min_conf=0.3) -> drop occluded samples",
        "dropped_class": "fearful (5 original samples, all occluded)",
        "num_classes": len(NEW_EMOTIONS),
        "emotions": NEW_EMOTIONS,
        "label_remap_orig_to_new": {int(k): int(v) for k, v in LABEL_REMAP.items()},
        "total_clean_samples": len(rows),
        "num_users": len(users),
        "n_folds": N_FOLDS,
        "fold_strategy": "user-level, greedy stratification by minority class weight",
        "seed": SEED,
    }
    (OUT_DIR / "dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"dataset_info.json -> {OUT_DIR / 'dataset_info.json'}")

    # Print summary table
    print("\n=== Fold summary ===")
    header = f"{'Fold':<6}{'N_users':>8}{'N_samples':>10}{'neut':>7}{'happy':>7}{'sad':>5}{'angr':>5}{'disg':>5}{'surp':>5}"
    print(header)
    print("-" * len(header))
    for k in range(N_FOLDS):
        f = folds[f"fold_{k}"]
        cc = f["test_class_counts"]
        print(f"fold_{k:<2}{len(f['test_users']):>8}{f['test_size']:>10}"
              f"{cc['neutral']:>7}{cc['happy']:>7}{cc['sad']:>5}"
              f"{cc['angry']:>5}{cc['disgusted']:>5}{cc['surprised']:>5}")


if __name__ == "__main__":
    main()
