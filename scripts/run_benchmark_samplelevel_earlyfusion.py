#!/usr/bin/env python3
"""Early Fusion concat (ResNet-18 TL) di CK+ & KDEF dgn SPLIT LEVEL-SAMPEL — Recipe B3.

Apple-to-apple dengan literatur yang memakai holdout / k-fold level-sampel
(Grover 2024, Singh 2025, Gautam 2023, Akhand 2021). Method = Early Fusion concat
(RGB + channel heatmap landmark, 4-channel), backbone ResNet-18 transfer learning.

Recipe B3 (konsisten dengan Unified Protocol): Adam lr=5e-5, batch 32, 50 epoch,
patience 15, internal val 15%, CrossEntropy + WeightedRandomSampler + synced aug
(hflip p=0.5, rotate ±10°, brightness ±10%, contrast ×0.9–1.1). Hanya pembentukan
fold yang berbeda:
  - randomsplit : StratifiedShuffleSplit 80:20, 5 seed (mean +/- std)
  - cv10        : StratifiedKFold 10-fold sample-level

Output: models/benchmark/{ds}_samplelevel/{ds}_7c_{randomsplit,cv10}_earlyfusion_b3.json
Usage:  CUDA_VISIBLE_DEVICES=1 python scripts/run_benchmark_samplelevel_earlyfusion.py
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
from training.models import EmotionEarlyFusionTransfer
from training.utils import train_model, full_evaluation
from training.fusion_aug import EarlyFusionDataset, make_balanced_sampler

DATASETS = {
    "ckplus_kaggle_orig": PROJECT_ROOT / "data" / "benchmark" / "ckplus_kaggle_orig",
}
EMOTIONS_7 = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]
BATCH, EPOCHS, PATIENCE, LR_TL = 32, 50, 15, 5e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_all(ddir):
    img, hm, y = [], [], []
    for sp in ["train", "val", "test"]:
        img.append(np.load(ddir / f"X_{sp}_images.npy").astype(np.float32))
        hm.append(np.load(ddir / f"X_{sp}_heatmaps.npy").astype(np.float32))
        y.append(np.load(ddir / f"y_{sp}.npy").astype(np.int64))
    img = np.concatenate(img); hm = np.concatenate(hm); y = np.concatenate(y)
    if img.max() > 1.5:
        img = img / 255.0
    return img, hm, y


def make_loader(img, hm, y, *, augment=False, sampler=None, seed=42):
    ds = EarlyFusionDataset(img, hm, y, augment=augment, seed=seed)
    shuffle = (sampler is None) and augment
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, sampler=sampler,
                      num_workers=2, pin_memory=True, drop_last=augment)


def train_eval_fold(img, hm, y, tr_idx, te_idx, nc, fdir):
    n_val = max(1, int(len(tr_idx) * 0.15))
    perm = np.random.RandomState(42).permutation(len(tr_idx))
    vi, ti = tr_idx[perm[:n_val]], tr_idx[perm[n_val:]]
    sampler = make_balanced_sampler(y[ti], nc)
    tr_l = make_loader(img[ti], hm[ti], y[ti], augment=True, sampler=sampler, seed=42)
    vl_l = make_loader(img[vi], hm[vi], y[vi], augment=False)
    te_l = make_loader(img[te_idx], hm[te_idx], y[te_idx], augment=False)
    model = EmotionEarlyFusionTransfer(num_classes=nc).to(device)
    sp = str(fdir / "m.pth")
    opt = torch.optim.Adam(model.parameters(), lr=LR_TL)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5,
                                                     patience=8, min_lr=1e-7)
    train_model(model, tr_l, vl_l, nn.CrossEntropyLoss(), opt, sch, device, "cnn", EPOCHS, PATIENCE, sp)
    model.load_state_dict(torch.load(sp, map_location=device, weights_only=True))
    r = full_evaluation(model, te_l, nn.CrossEntropyLoss(), device, "cnn", EMOTIONS_7)
    os.remove(sp)
    return {"accuracy": float(r["test_accuracy"]), "macro_f1": float(r["test_macro_f1"]),
            "weighted_f1": float(r["test_weighted_f1"])}


def run_protocol(ds, name, folds, img, hm, y, out_dir, nc=7):
    print(f"\n{'='*64}\n  {ds.upper()} 7c — Early Fusion concat — {name} ({len(folds)} folds)\n{'='*64}", flush=True)
    fdir = out_dir / "_tmp"; fdir.mkdir(parents=True, exist_ok=True)
    accs, f1s, wf1s, per_fold = [], [], [], []
    for fi, (tr_idx, te_idx) in enumerate(folds):
        torch.manual_seed(42); np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        t0 = time.time()
        r = train_eval_fold(img, hm, y, np.asarray(tr_idx), np.asarray(te_idx), nc, fdir)
        accs.append(r["accuracy"]); f1s.append(r["macro_f1"]); wf1s.append(r["weighted_f1"])
        per_fold.append({"fold_idx": fi, "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)), **r})
        print(f"    [fold {fi+1}/{len(folds)}] macro_f1={r['macro_f1']:.4f} acc={r['accuracy']:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    res = {"model": "EarlyFusion_concat_TL_B3", "protocol": name,
           "macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
           "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
           "weighted_f1_mean": float(np.mean(wf1s)), "weighted_f1_std": float(np.std(wf1s)),
           "n_folds": len(folds), "per_fold": per_fold}
    save = out_dir / f"{ds}_7c_{name}_earlyfusion_b3.json"
    json.dump(res, open(save, "w"), indent=2)
    print(f"  {name}: macro_f1={res['macro_f1_mean']:.4f}±{res['macro_f1_std']:.4f}  "
          f"acc={res['accuracy_mean']:.4f}±{res['accuracy_std']:.4f}  -> saved {save.name}", flush=True)
    try:
        fdir.rmdir()
    except OSError:
        pass
    return res


def main():
    print(f"Device: {device}", flush=True)
    for ds, ddir in DATASETS.items():
        img, hm, y = load_all(ddir)
        print(f"\n##### {ds.upper()}: N={len(y)}, kelas={len(set(y))} (sample-level) #####", flush=True)
        out_dir = PROJECT_ROOT / "models" / "benchmark" / f"{ds}_samplelevel"
        out_dir.mkdir(parents=True, exist_ok=True)
        rs = list(StratifiedShuffleSplit(n_splits=5, test_size=0.20, random_state=42).split(np.zeros(len(y)), y))
        run_protocol(ds, "randomsplit", rs, img, hm, y, out_dir)
        cv = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
        run_protocol(ds, "cv10", cv, img, hm, y, out_dir)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
