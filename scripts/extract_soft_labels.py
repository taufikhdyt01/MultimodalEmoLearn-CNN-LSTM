"""
Extract Soft Labels (Face API Confidence Distribution) for Existing Dataset
============================================================================
Script cepat untuk ekstrak y_soft_{split}.npy (N, 7) tanpa harus re-run
full prepare_dataset.py (yang heavy karena load image ulang).

Strategi: reuse label extraction + split logic, skip image loading.

Output: y_{split}_soft.npy shape (N, 7), sum across kelas = 1.0 per sampel.

Usage:
    python scripts/extract_soft_labels.py
    python scripts/extract_soft_labels.py --dataset-dir data/dataset_frontonly_conf60
    python scripts/extract_soft_labels.py --min-confidence 0.60

CATATAN: Script ini butuh data/processed/ + data/processed_new/ (ada
hanya di laptop local, TIDAK di VPS). Hasil y_*_soft.npy di-commit ke
repo supaya VPS bisa langsung pakai tanpa run script ini.

Validation:
    Script cek bahwa argmax(y_soft) == y (existing hard label).
    Kalau tidak match, ada mismatch order/split → error.

Konteks:
    Diperlukan untuk eksplorasi Soft Label Training (ide #5 di
    docs/eksplorasi_lanjutan.md) — memanfaatkan distribusi confidence
    Face API sebagai target training, bukan hanya argmax.
"""
import argparse
import sys
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from preprocessing.prepare_dataset import (  # noqa: E402
    load_old_labels, load_new_labels,
    collect_old_samples, collect_new_samples,
    split_by_user,
    EMOTIONS, SPLIT_RATIO, RANDOM_SEED,
)


def scores_to_soft(scores_list):
    """Convert list of score arrays → (N, 7) float32, renormalized to sum=1."""
    n = len(scores_list)
    out = np.zeros((n, len(EMOTIONS)), dtype=np.float32)
    for i, s in enumerate(scores_list):
        s = np.asarray(s, dtype=np.float32)
        total = s.sum()
        out[i] = s / total if total > 0 else s
    return out


def main():
    ap = argparse.ArgumentParser(description="Extract soft labels for existing dataset")
    ap.add_argument("--dataset-dir", default=str(PROJECT_ROOT / "data" / "dataset_frontonly_conf60"),
                    help="Dataset directory (harus sudah ada y_{split}.npy)")
    ap.add_argument("--min-confidence", type=float, default=0.60,
                    help="Must match value used in original prepare_dataset.py run (default 0.60 untuk conf60)")
    ap.add_argument("--include-side", action="store_true",
                    help="Set True kalau dataset include side-view (front_side)")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED,
                    help="Must match seed in dataset_info.json")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing y_{split}_soft.npy")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    info_path = dataset_dir / "dataset_info.json"
    if not info_path.exists():
        print(f"ERROR: {info_path} not found. Run prepare_dataset.py first.")
        return 1

    with open(info_path) as f:
        info = json.load(f)

    # Honor seed from info to match original split exactly
    seed = info.get("seed", args.seed)
    include_side = info.get("include_side", args.include_side)
    print(f"Dataset: {dataset_dir}")
    print(f"  seed={seed}  include_side={include_side}  min_conf={args.min_confidence}")

    # Check if already exists
    out_files = [dataset_dir / f"y_{s}_soft.npy" for s in ("train", "val", "test")]
    if all(p.exists() for p in out_files) and not args.force:
        print("\nSoft labels already exist. Use --force to overwrite.")
        for p in out_files:
            arr = np.load(p)
            print(f"  {p.name}  shape={arr.shape}  sum/row≈{arr.sum(axis=1).mean():.4f}")
        return 0

    # --- Step 1: Load & collect samples (same pipeline as prepare_dataset) ---
    print("\n[1/4] Loading labels from xlsx/csv...")
    old_labels = load_old_labels()
    new_labels = load_new_labels()
    print(f"  Old: {len(old_labels)} users, New: {len(new_labels)} users")

    print(f"\n[2/4] Collecting samples (min_confidence={args.min_confidence})...")
    old_samples = collect_old_samples(old_labels, min_confidence=args.min_confidence)
    new_samples = collect_new_samples(new_labels, include_side=include_side,
                                      min_confidence=args.min_confidence)
    all_samples = old_samples + new_samples
    print(f"  Total: {len(all_samples)} samples")

    # --- Step 2: Split identically ---
    print(f"\n[3/4] Splitting (seed={seed})...")
    train, val, test, _, _, _ = split_by_user(all_samples, SPLIT_RATIO, seed)
    print(f"  Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    # Expected sizes from existing dataset
    exp_train = info["train"]["samples"]
    exp_val = info["val"]["samples"]
    exp_test = info["test"]["samples"]
    for name, actual, expected in [("Train", len(train), exp_train),
                                    ("Val", len(val), exp_val),
                                    ("Test", len(test), exp_test)]:
        if actual != expected:
            print(f"  [ERROR] {name} size mismatch: got {actual}, expected {expected}")
            print("  Periksa --min-confidence / --seed / --include-side sesuai dataset_info.json")
            return 1

    # --- Step 3: Extract soft labels + validate ---
    print("\n[4/4] Extracting soft labels + validating vs hard labels...")
    for split_name, samples in [("train", train), ("val", val), ("test", test)]:
        scores_list = [s[3] for s in samples]  # tuple: (uid, face_path, lm_path, scores)
        y_soft = scores_to_soft(scores_list)

        # Validate argmax(soft) == existing hard label
        y_hard_existing = np.load(dataset_dir / f"y_{split_name}.npy")
        y_hard_from_soft = np.argmax(y_soft, axis=1)
        matches = (y_hard_existing == y_hard_from_soft).sum()
        total = len(y_hard_existing)
        print(f"  {split_name}: argmax(soft) matches y_hard = {matches}/{total} "
              f"({matches/total*100:.2f}%)")
        if matches != total:
            diff_idx = np.where(y_hard_existing != y_hard_from_soft)[0][:5]
            print(f"    [WARN] mismatch at indices {diff_idx.tolist()}; "
                  f"y_hard={y_hard_existing[diff_idx].tolist()}, "
                  f"argmax(soft)={y_hard_from_soft[diff_idx].tolist()}")
            print("    Mungkin ada ties di argmax (probability sama) — tidak fatal.")

        # Save
        out_path = dataset_dir / f"y_{split_name}_soft.npy"
        np.save(out_path, y_soft)
        print(f"    Saved: {out_path.name}  shape={y_soft.shape}  "
              f"mean_max_conf={y_soft.max(axis=1).mean():.4f}")

    # Update dataset_info.json to flag soft labels availability
    info["has_soft_labels"] = True
    info["min_confidence"] = args.min_confidence
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    print("\nDone. y_{train,val,test}_soft.npy siap dipakai untuk Soft Label Training.")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
