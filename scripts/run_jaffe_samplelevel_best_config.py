#!/usr/bin/env python3
"""JAFFE sample-level benchmark — konfigurasi terbaik: Intermediate Fusion TL,
blendshape_52, B3.

Konfigurasi ini adalah hasil terbaik pada protokol subject-wise (acc=55%, B3).
Script ini mengulangi evaluasi dengan protokol sample-level untuk
perbandingan apple-to-apple dengan literatur.

Konfigurasi persis sama dengan unified protocol B3:
  - Model     : IntermediateFusionTransfer (ResNet-18 pretrained ImageNet)
  - Feature   : blendshape_52 = ARKit blendshapes MediaPipe = 52-dim
  - Scenario  : B3 — WeightedRandomSampler + augmentasi (photometric only,
                karena blendshape_52 adalah scalar features bukan koordinat)
  - Batch     : 32, Epochs: 50, Patience: 15, LR: 1e-4, Seed: 42

Protokol:
  - randomsplit : StratifiedShuffleSplit 80:20, 5 seed
  - cv10        : StratifiedKFold 10-fold sample-level

Output: models/benchmark/jaffe_samplelevel/jaffe_7c_{randomsplit,cv10}_intermediate_bs52_tl_b3.json
Usage:  CUDA_VISIBLE_DEVICES=1 python scripts/run_jaffe_samplelevel_best_config.py
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from training.models import IntermediateFusionTransfer
from training.utils import train_model, full_evaluation
from training.fusion_aug import IntermediateFusionDataset, make_balanced_sampler

DATA_DIR = PROJECT_ROOT / "data" / "benchmark" / "jaffe_7class"
OUT_DIR  = PROJECT_ROOT / "models" / "benchmark" / "jaffe_samplelevel"

EMOTIONS_7 = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]
BATCH, EPOCHS, PATIENCE, LR, SEED = 32, 50, 15, 1e-4, 42
LANDMARK_DIM = 52  # blendshape_52
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_all():
    """Load semua split, gabungkan, ambil blendshape_52."""
    imgs, lms, ys = [], [], []
    for sp in ["train", "val", "test"]:
        img = np.load(DATA_DIR / f"X_{sp}_images.npy").astype(np.float32)
        bs  = np.load(DATA_DIR / f"X_{sp}_mp_blendshapes.npy").astype(np.float32)
        bs  = np.nan_to_num(bs, nan=0.0)  # (N, 52)
        y   = np.load(DATA_DIR / f"y_{sp}.npy").astype(np.int64)
        if img.max() > 1.5:
            img = img / 255.0
        imgs.append(img); lms.append(bs); ys.append(y)
    return np.concatenate(imgs), np.concatenate(lms), np.concatenate(ys)


def make_loader(img, lm, y, *, augment=False, sampler=None, seed=42):
    # facs_plus_bs_80 adalah scalar features — landmark_is_coord=False
    ds = IntermediateFusionDataset(img, lm, y, augment=augment, seed=seed,
                                   landmark_is_coord=False)
    shuffle = (sampler is None) and augment
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, sampler=sampler,
                      num_workers=2, pin_memory=True, drop_last=(sampler is not None))


def train_eval_fold(img, lm, y, tr_idx, te_idx, fdir):
    n_val = max(1, int(len(tr_idx) * 0.15))
    perm = np.random.RandomState(SEED).permutation(len(tr_idx))
    vi, ti = tr_idx[perm[:n_val]], tr_idx[perm[n_val:]]

    # B3: WeightedRandomSampler + augmentasi (photometric only, blendshape bukan koordinat)
    sampler = make_balanced_sampler(y[ti], 7)
    tr_l = make_loader(img[ti], lm[ti], y[ti], augment=True, sampler=sampler, seed=SEED)
    va_l = make_loader(img[vi], lm[vi], y[vi], augment=False)
    te_l = make_loader(img[te_idx], lm[te_idx], y[te_idx], augment=False)

    torch.manual_seed(SEED); np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    model = IntermediateFusionTransfer(num_classes=7, landmark_dim=LANDMARK_DIM).to(device)
    sp = str(fdir / "m.pth")
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5,
                                                     patience=8, min_lr=1e-7)
    train_model(model, tr_l, va_l, nn.CrossEntropyLoss(), opt, sch, device,
                "fusion", EPOCHS, PATIENCE, sp)
    model.load_state_dict(torch.load(sp, map_location=device, weights_only=True))
    r = full_evaluation(model, te_l, nn.CrossEntropyLoss(), device, "fusion", EMOTIONS_7)
    os.remove(sp)
    return {"accuracy": float(r["test_accuracy"]),
            "macro_f1": float(r["test_macro_f1"]),
            "weighted_f1": float(r["test_weighted_f1"])}


def run_protocol(name, folds, img, lm, y):
    print(f"\n{'='*64}", flush=True)
    print(f"  JAFFE 7c — Intermediate facs_plus_bs_80 scratch B2 — {name} ({len(folds)} folds)",
          flush=True)
    print(f"{'='*64}", flush=True)

    fdir = OUT_DIR / "_tmp"; fdir.mkdir(parents=True, exist_ok=True)
    accs, f1s, wf1s, per_fold = [], [], [], []

    for fi, (tr_idx, te_idx) in enumerate(folds):
        t0 = time.time()
        r = train_eval_fold(img, lm, y, np.asarray(tr_idx), np.asarray(te_idx), fdir)
        accs.append(r["accuracy"]); f1s.append(r["macro_f1"]); wf1s.append(r["weighted_f1"])
        per_fold.append({"fold_idx": fi, "n_train": int(len(tr_idx)),
                         "n_test": int(len(te_idx)), **r})
        print(f"    [fold {fi+1}/{len(folds)}] macro_f1={r['macro_f1']:.4f} "
              f"acc={r['accuracy']:.4f} ({time.time()-t0:.0f}s)", flush=True)

    res = {
        "model": "IntermediateFusionTransfer_blendshape52_TL_B3",
        "protocol": name,
        "feature": "blendshape_52",
        "scenario": "B3",
        "macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
        "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
        "weighted_f1_mean": float(np.mean(wf1s)), "weighted_f1_std": float(np.std(wf1s)),
        "n_folds": len(folds), "per_fold": per_fold,
    }
    save = OUT_DIR / f"jaffe_7c_{name}_intermediate_bs52_tl_b3.json"
    json.dump(res, open(save, "w"), indent=2)
    print(f"\n  {name}: acc={res['accuracy_mean']:.4f}±{res['accuracy_std']:.4f}  "
          f"macro_f1={res['macro_f1_mean']:.4f}±{res['macro_f1_std']:.4f}  "
          f"-> saved {save.name}", flush=True)
    try: fdir.rmdir()
    except OSError: pass
    return res


def main():
    print(f"Device: {device}", flush=True)
    img, lm, y = load_all()
    print(f"\nJAFFE: N={len(y)}, kelas={len(set(y))}, landmark_dim={lm.shape[1]} (blendshape_52, TL, B3)", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rs = list(StratifiedShuffleSplit(n_splits=5, test_size=0.20,
                                     random_state=SEED).split(np.zeros(len(y)), y))
    run_protocol("randomsplit", rs, img, lm, y)

    cv = list(StratifiedKFold(n_splits=10, shuffle=True,
                               random_state=SEED).split(np.zeros(len(y)), y))
    run_protocol("cv10", cv, img, lm, y)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
