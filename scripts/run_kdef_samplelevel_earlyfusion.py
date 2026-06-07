#!/usr/bin/env python3
"""KDEF sample-level benchmark — konfigurasi terbaik: Early Fusion concat TL,
raw_136, B1 (no augmentation, no sampler).

Konfigurasi terbaik pada protokol subject-wise (acc=91,50%, macro_f1=91,40%).
Dijalankan dengan protokol sample-level untuk perbandingan apple-to-apple
dengan literatur:
  - Akhand et al. (2021): DenseNet-161, 10-fold CV → 96,51%
  - Grover & Bansal (2024): CNN, Holdout → 94,00%
  - Singh et al. (2025): CNN MMSAD, Holdout 70:30 → 88,01%

Konfigurasi B1:
  - Model   : EmotionEarlyFusionTransfer (ResNet-18 TL)
  - Feature : citra RGB + heatmap landmark raw_136 (4-channel)
  - Scenario: B1 — tanpa augmentasi, tanpa WeightedRandomSampler
  - LR      : 5e-5, Batch: 32, Epochs: 50, Patience: 15, Seed: 42

Protokol sample-level:
  - holdout8020 : StratifiedShuffleSplit 80:20, 5 seed
  - holdout7030 : StratifiedShuffleSplit 70:30, 5 seed (sesuai Singh 2025)
  - cv10        : StratifiedKFold 10-fold sample-level (sesuai Akhand 2021)

Output: models/benchmark/kdef_samplelevel/
Usage:  CUDA_VISIBLE_DEVICES=1 python scripts/run_kdef_samplelevel_earlyfusion.py
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from training.models import EmotionEarlyFusionTransfer
from training.utils import train_model, full_evaluation

DATA_DIR = PROJECT_ROOT / "data" / "benchmark" / "kdef_7class"
OUT_DIR  = PROJECT_ROOT / "models" / "benchmark" / "kdef_samplelevel"

EMOTIONS_7 = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]
BATCH, EPOCHS, PATIENCE, LR, SEED = 32, 50, 15, 5e-5, 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_all():
    imgs, hms, ys = [], [], []
    for sp in ["train", "val", "test"]:
        img = np.load(DATA_DIR / f"X_{sp}_images.npy").astype(np.float32)
        hm  = np.load(DATA_DIR / f"X_{sp}_heatmaps.npy").astype(np.float32)
        y   = np.load(DATA_DIR / f"y_{sp}.npy").astype(np.int64)
        if img.max() > 1.5:
            img = img / 255.0
        imgs.append(img); hms.append(hm); ys.append(y)
    return np.concatenate(imgs), np.concatenate(hms), np.concatenate(ys)


def make_loader(img, hm, y, shuffle=True):
    """B1: TensorDataset biasa, tanpa augmentasi, tanpa sampler."""
    a = torch.from_numpy(img).permute(0, 3, 1, 2).float()
    b = torch.from_numpy(hm).unsqueeze(1).float()
    t = torch.cat([a, b], dim=1)  # (N, 4, H, W)
    ds = TensorDataset(t, torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle,
                      num_workers=2, pin_memory=True, drop_last=shuffle)


def train_eval_fold(img, hm, y, tr_idx, te_idx, fdir):
    n_val = max(1, int(len(tr_idx) * 0.15))
    perm = np.random.RandomState(SEED).permutation(len(tr_idx))
    vi, ti = tr_idx[perm[:n_val]], tr_idx[perm[n_val:]]

    tr_l = make_loader(img[ti], hm[ti], y[ti], shuffle=True)
    va_l = make_loader(img[vi], hm[vi], y[vi], shuffle=False)
    te_l = make_loader(img[te_idx], hm[te_idx], y[te_idx], shuffle=False)

    torch.manual_seed(SEED); np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    model = EmotionEarlyFusionTransfer(num_classes=7).to(device)
    sp = str(fdir / "m.pth")
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5,
                                                     patience=8, min_lr=1e-7)
    train_model(model, tr_l, va_l, nn.CrossEntropyLoss(), opt, sch, device,
                "cnn", EPOCHS, PATIENCE, sp)
    model.load_state_dict(torch.load(sp, map_location=device, weights_only=True))
    r = full_evaluation(model, te_l, nn.CrossEntropyLoss(), device, "cnn", EMOTIONS_7)
    os.remove(sp)
    return {"accuracy": float(r["test_accuracy"]),
            "macro_f1": float(r["test_macro_f1"]),
            "weighted_f1": float(r["test_weighted_f1"])}


def run_protocol(name, folds, img, hm, y):
    print(f"\n{'='*64}", flush=True)
    print(f"  KDEF 7c — Early Fusion concat TL B1 — {name} ({len(folds)} folds)",
          flush=True)
    print(f"{'='*64}", flush=True)

    fdir = OUT_DIR / "_tmp"; fdir.mkdir(parents=True, exist_ok=True)
    accs, f1s, wf1s, per_fold = [], [], [], []

    for fi, (tr_idx, te_idx) in enumerate(folds):
        t0 = time.time()
        r = train_eval_fold(img, hm, y,
                            np.asarray(tr_idx), np.asarray(te_idx), fdir)
        accs.append(r["accuracy"]); f1s.append(r["macro_f1"])
        wf1s.append(r["weighted_f1"])
        per_fold.append({"fold_idx": fi, "n_train": int(len(tr_idx)),
                         "n_test": int(len(te_idx)), **r})
        print(f"    [fold {fi+1}/{len(folds)}] macro_f1={r['macro_f1']:.4f} "
              f"acc={r['accuracy']:.4f} ({time.time()-t0:.0f}s)", flush=True)

    res = {
        "model": "EarlyFusion_concat_TL_B1",
        "protocol": name, "scenario": "B1",
        "macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
        "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
        "weighted_f1_mean": float(np.mean(wf1s)), "weighted_f1_std": float(np.std(wf1s)),
        "n_folds": len(folds), "per_fold": per_fold,
    }
    save = OUT_DIR / f"kdef_7c_{name}_earlyfusion_b1.json"
    json.dump(res, open(save, "w"), indent=2)
    print(f"\n  {name}: acc={res['accuracy_mean']:.4f}±{res['accuracy_std']:.4f}  "
          f"macro_f1={res['macro_f1_mean']:.4f}±{res['macro_f1_std']:.4f}  "
          f"-> saved {save.name}", flush=True)
    try: fdir.rmdir()
    except OSError: pass
    return res


def main():
    print(f"Device: {device}", flush=True)
    img, hm, y = load_all()
    print(f"\nKDEF: N={len(y)}, kelas={len(set(y))} (sample-level, B1)", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Holdout 80:20 — comparable dengan Grover (2024)
    rs8020 = list(StratifiedShuffleSplit(n_splits=5, test_size=0.20,
                  random_state=SEED).split(np.zeros(len(y)), y))
    run_protocol("holdout8020", rs8020, img, hm, y)

    # Holdout 70:30 — comparable dengan Singh (2025) yang pakai 70:30
    rs7030 = list(StratifiedShuffleSplit(n_splits=5, test_size=0.30,
                  random_state=SEED).split(np.zeros(len(y)), y))
    run_protocol("holdout7030", rs7030, img, hm, y)

    # 10-fold CV — comparable dengan Akhand (2021)
    cv = list(StratifiedKFold(n_splits=10, shuffle=True,
              random_state=SEED).split(np.zeros(len(y)), y))
    run_protocol("cv10", cv, img, hm, y)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
