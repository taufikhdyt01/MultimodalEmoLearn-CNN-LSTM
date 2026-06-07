#!/usr/bin/env python3
"""Early Fusion concat (ResNet-18 TL) — Subject-wise holdout, Recipe B3.

Menggunakan split train/val/test yang sudah subject-wise dari dataset dir.
Dijalankan untuk tiga skema kelas CK+:
  - ckplus_7class      (neutral in, contempt out) — 7c, 636 sampel
  - ckplus_kaggle7class (contempt in, neutral out) — 7c, 327 sampel
  - ckplus_8class      (neutral + contempt)       — 8c, 654 sampel

Recipe B3: Adam lr=5e-5, batch 32, 50 epoch, patience 15,
CrossEntropy + WeightedRandomSampler + synced aug.

Output: models/benchmark/ckplus_subjectwise/{ds}_earlyfusion_b3_subjectwise.json
Usage:  CUDA_VISIBLE_DEVICES=1 python scripts/run_benchmark_subjectwise_earlyfusion.py
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from training.models import EmotionEarlyFusionTransfer
from training.utils import train_model, full_evaluation
from training.fusion_aug import EarlyFusionDataset, make_balanced_sampler

DATASETS = {
    "ckplus_7class": {
        "dir": PROJECT_ROOT / "data" / "benchmark" / "ckplus_7class",
        "emotions": ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"],
        "nc": 7,
    },
    "ckplus_kaggle7class": {
        "dir": PROJECT_ROOT / "data" / "benchmark" / "ckplus_kaggle7class",
        "emotions": ["anger", "contempt", "disgust", "fear", "happy", "sadness", "surprised"],
        "nc": 7,
    },
    "ckplus_8class": {
        "dir": PROJECT_ROOT / "data" / "benchmark" / "ckplus_8class",
        "emotions": ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised", "contempt"],
        "nc": 8,
    },
    "ckplus_kaggle_orig": {
        "dir": PROJECT_ROOT / "data" / "benchmark" / "ckplus_kaggle_orig",
        "emotions": ["anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"],
        "nc": 7,
    },
}

OUT_DIR = PROJECT_ROOT / "models" / "benchmark" / "ckplus_subjectwise"
BATCH, EPOCHS, PATIENCE, LR_TL = 32, 50, 15, 5e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_split(ddir, split):
    img = np.load(ddir / f"X_{split}_images.npy").astype(np.float32)
    hm  = np.load(ddir / f"X_{split}_heatmaps.npy").astype(np.float32)
    y   = np.load(ddir / f"y_{split}.npy").astype(np.int64)
    if img.max() > 1.5:
        img = img / 255.0
    return img, hm, y


def make_loader(img, hm, y, *, augment=False, sampler=None, seed=42):
    ds = EarlyFusionDataset(img, hm, y, augment=augment, seed=seed)
    shuffle = (sampler is None) and augment
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, sampler=sampler,
                      num_workers=2, pin_memory=True, drop_last=augment)


def run_one(ds_name, cfg):
    ddir     = cfg["dir"]
    emotions = cfg["emotions"]
    nc       = cfg["nc"]

    print(f"\n{'='*64}", flush=True)
    print(f"  {ds_name}  ({nc} classes, subject-wise holdout, B3)", flush=True)
    print(f"{'='*64}", flush=True)

    img_tr, hm_tr, y_tr = load_split(ddir, "train")
    img_va, hm_va, y_va = load_split(ddir, "val")
    img_te, hm_te, y_te = load_split(ddir, "test")

    print(f"  train N={len(y_tr)}, val N={len(y_va)}, test N={len(y_te)}", flush=True)

    sampler = make_balanced_sampler(y_tr, nc)
    tr_l = make_loader(img_tr, hm_tr, y_tr, augment=True, sampler=sampler, seed=42)
    va_l = make_loader(img_va, hm_va, y_va, augment=False)
    te_l = make_loader(img_te, hm_te, y_te, augment=False)

    torch.manual_seed(42); np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    model = EmotionEarlyFusionTransfer(num_classes=nc).to(device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sp = str(OUT_DIR / f"_tmp_{ds_name}.pth")

    opt = torch.optim.Adam(model.parameters(), lr=LR_TL)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5,
                                                     patience=8, min_lr=1e-7)
    t0 = time.time()
    train_model(model, tr_l, va_l, nn.CrossEntropyLoss(), opt, sch, device,
                "cnn", EPOCHS, PATIENCE, sp)
    model.load_state_dict(torch.load(sp, map_location=device, weights_only=True))
    r = full_evaluation(model, te_l, nn.CrossEntropyLoss(), device, "cnn", emotions)
    os.remove(sp)

    elapsed = time.time() - t0
    res = {
        "model": "EarlyFusion_concat_TL_B3",
        "protocol": "subject-wise holdout",
        "dataset": ds_name,
        "num_classes": nc,
        "emotions": emotions,
        "accuracy":    float(r["test_accuracy"]),
        "macro_f1":    float(r["test_macro_f1"]),
        "weighted_f1": float(r["test_weighted_f1"]),
        "n_train": int(len(y_tr)),
        "n_val":   int(len(y_va)),
        "n_test":  int(len(y_te)),
        "elapsed_s": round(elapsed),
    }
    save = OUT_DIR / f"{ds_name}_earlyfusion_b3_subjectwise.json"
    json.dump(res, open(save, "w"), indent=2)
    print(f"\n  RESULT: acc={res['accuracy']:.4f}  macro_f1={res['macro_f1']:.4f}  "
          f"weighted_f1={res['weighted_f1']:.4f}  ({elapsed:.0f}s)", flush=True)
    print(f"  Saved: {save.name}", flush=True)
    return res


def main():
    print(f"Device: {device}", flush=True)
    results = {}
    only = os.environ.get("ONLY_DS")
    items = [(k, v) for k, v in DATASETS.items() if not only or k == only]
    for ds_name, cfg in items:
        if not cfg["dir"].exists():
            print(f"\nSKIP {ds_name}: folder tidak ditemukan ({cfg['dir']})", flush=True)
            continue
        results[ds_name] = run_one(ds_name, cfg)

    print(f"\n{'='*64}", flush=True)
    print("  SUMMARY — Subject-wise holdout, Early Fusion concat TL B3", flush=True)
    print(f"{'='*64}", flush=True)
    print(f"  {'Dataset':<25s}  {'NC':>3}  {'Accuracy':>9}  {'Macro F1':>9}  {'WF1':>9}", flush=True)
    for ds_name, r in results.items():
        print(f"  {ds_name:<25s}  {r['num_classes']:>3}  "
              f"{r['accuracy']:>9.4f}  {r['macro_f1']:>9.4f}  {r['weighted_f1']:>9.4f}", flush=True)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
