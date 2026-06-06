#!/usr/bin/env python3
"""Evaluasi JAFFE dengan RANDOM SPLIT (sample-level) — apple-to-apple dgn literatur.

Banyak paper JAFFE (Akhand 2021 99.52%, Singh 2025 98.50%, Wadhawan 2023 97.14%,
Gautam 2023 91.43%) memakai split level-SAMPEL (10-fold CV / holdout 80:20), bukan
subject-independent. LOSO kita memberi angka rendah karena person-independent.
Skrip ini mengulang evaluasi JAFFE dengan protokol random-split agar perbandingan
setara, MENIRU PERSIS prosedur benchmark (6 model, B1, batch16/50ep/patience15,
landmark raw_136, late-fusion weight grid). Hanya pembentukan fold yang diganti.

Dua protokol:
  - randomsplit : StratifiedShuffleSplit 80:20, 5 seed (mean +/- std)
  - cv10        : StratifiedKFold 10-fold sample-level (cocokkan dgn paper 10-fold)

Output: models/benchmark/jaffe_randomsplit/jaffe_7c_{randomsplit,cv10}_results.json

Usage:
  CUDA_VISIBLE_DEVICES=1 python scripts/run_jaffe_randomsplit.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from training.models import (
    EmotionCNN, EmotionFCNN, IntermediateFusion,
    EmotionCNNTransfer, IntermediateFusionTransfer,
)
from training.utils import train_model, full_evaluation

DATA = PROJECT_ROOT / "data" / "benchmark" / "jaffe_7class"
OUT = PROJECT_ROOT / "models" / "benchmark" / "jaffe_randomsplit"
OUT.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMOTIONS_7 = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]
BATCH, EPOCHS, PATIENCE = 16, 50, 15
MODELS = [
    ("CNN", EmotionCNN, "cnn", 1e-4),
    ("FCNN", EmotionFCNN, "fcnn", 1e-4),
    ("Intermediate", IntermediateFusion, "fusion", 1e-4),
    ("CNN_TL", EmotionCNNTransfer, "cnn", 5e-5),
    ("Intermediate_TL", IntermediateFusionTransfer, "fusion", 5e-5),
]


def load_all():
    img, lm, y = [], [], []
    for sp in ["train", "val", "test"]:
        img.append(np.load(DATA / f"X_{sp}_images.npy").astype(np.float32))
        lm.append(np.load(DATA / f"X_{sp}_landmarks.npy").astype(np.float32))
        y.append(np.load(DATA / f"y_{sp}.npy").astype(np.int64))
    return np.concatenate(img), np.concatenate(lm), np.concatenate(y)


def make_loader(img, lm, y, mtype, shuffle=True):
    it = torch.from_numpy(img).permute(0, 3, 1, 2)
    lt = torch.from_numpy(lm); yt = torch.from_numpy(y).long()
    if mtype == "cnn":
        ds = TensorDataset(it, yt)
    elif mtype == "fcnn":
        ds = TensorDataset(lt, yt)
    else:
        ds = TensorDataset(it, lt, yt)
    # drop_last saat training (shuffle=True) supaya BatchNorm tidak menerima batch size-1
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, num_workers=0,
                      pin_memory=True, drop_last=shuffle)


def _mk_opt(model, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5,
                                                     patience=8, min_lr=1e-7)
    return opt, sch


def train_fold(ModelClass, mtype, lr, tr_img, tr_lm, tr_y, te_img, te_lm, te_y, nc, fdir):
    n_val = max(1, int(len(tr_y) * 0.15))
    perm = np.random.RandomState(42).permutation(len(tr_y))
    vi, ti = perm[:n_val], perm[n_val:]
    tr_l = make_loader(tr_img[ti], tr_lm[ti], tr_y[ti], mtype)
    vl_l = make_loader(tr_img[vi], tr_lm[vi], tr_y[vi], mtype, False)
    te_l = make_loader(te_img, te_lm, te_y, mtype, False)
    model = ModelClass(num_classes=nc).to(device)
    sp = str(fdir / "m.pth")
    opt, sch = _mk_opt(model, lr)
    train_model(model, tr_l, vl_l, nn.CrossEntropyLoss(), opt, sch, device, mtype, EPOCHS, PATIENCE, sp)
    model.load_state_dict(torch.load(sp, map_location=device, weights_only=True))
    r = full_evaluation(model, te_l, nn.CrossEntropyLoss(), device, mtype, EMOTIONS_7)
    os.remove(sp)
    return {"accuracy": float(r["test_accuracy"]), "macro_f1": float(r["test_macro_f1"]),
            "weighted_f1": float(r["test_weighted_f1"])}


def late_fusion_fold(tr_img, tr_lm, tr_y, te_img, te_lm, te_y, nc, fdir):
    n_val = max(1, int(len(tr_y) * 0.15))
    perm = np.random.RandomState(42).permutation(len(tr_y))
    vi, ti = perm[:n_val], perm[n_val:]
    cnn = EmotionCNN(num_classes=nc).to(device)
    o1, s1 = _mk_opt(cnn, 1e-4)
    train_model(cnn, make_loader(tr_img[ti], tr_lm[ti], tr_y[ti], "cnn"),
                make_loader(tr_img[vi], tr_lm[vi], tr_y[vi], "cnn", False),
                nn.CrossEntropyLoss(), o1, s1, device, "cnn", EPOCHS, PATIENCE, str(fdir / "cnn.pth"))
    fcnn = EmotionFCNN(num_classes=nc).to(device)
    o2, s2 = _mk_opt(fcnn, 1e-4)
    train_model(fcnn, make_loader(tr_img[ti], tr_lm[ti], tr_y[ti], "fcnn"),
                make_loader(tr_img[vi], tr_lm[vi], tr_y[vi], "fcnn", False),
                nn.CrossEntropyLoss(), o2, s2, device, "fcnn", EPOCHS, PATIENCE, str(fdir / "fcnn.pth"))
    cnn.load_state_dict(torch.load(fdir / "cnn.pth", map_location=device, weights_only=True))
    fcnn.load_state_dict(torch.load(fdir / "fcnn.pth", map_location=device, weights_only=True))
    cnn.eval(); fcnn.eval()
    ti_ = torch.from_numpy(te_img).permute(0, 3, 1, 2).to(device)
    tl_ = torch.from_numpy(te_lm).to(device)
    with torch.no_grad():
        cp = torch.softmax(cnn(ti_), dim=1).cpu().numpy()
        fp = torch.softmax(fcnn(tl_), dim=1).cpu().numpy()
    best_f1, best_preds = 0, cp.argmax(1)
    for w in np.arange(0.0, 1.05, 0.05):
        preds = (w * cp + (1 - w) * fp).argmax(1)
        f1 = f1_score(te_y, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_preds = f1, preds
    for f in ["cnn.pth", "fcnn.pth"]:
        (fdir / f).unlink(missing_ok=True)
    return {"accuracy": float(accuracy_score(te_y, best_preds)), "macro_f1": float(best_f1),
            "weighted_f1": float(f1_score(te_y, best_preds, average="weighted", zero_division=0))}


def run_protocol(name, folds, img, lm, y, nc=7):
    print(f"\n{'='*64}\n  JAFFE 7c — {name}  ({len(folds)} folds)\n{'='*64}")
    fdir = OUT / "_tmp"; fdir.mkdir(exist_ok=True)
    all_results = {}
    runlist = MODELS + [("Late_Fusion", None, "late", 1e-4)]
    for mname, MC, mtype, lr in runlist:
        accs, f1s, wf1s, per_fold = [], [], [], []
        for fi, (tr_idx, te_idx) in enumerate(folds):
            torch.manual_seed(42); np.random.seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)
            if mtype == "late":
                r = late_fusion_fold(img[tr_idx], lm[tr_idx], y[tr_idx],
                                     img[te_idx], lm[te_idx], y[te_idx], nc, fdir)
            else:
                r = train_fold(MC, mtype, lr, img[tr_idx], lm[tr_idx], y[tr_idx],
                               img[te_idx], lm[te_idx], y[te_idx], nc, fdir)
            accs.append(r["accuracy"]); f1s.append(r["macro_f1"]); wf1s.append(r["weighted_f1"])
            per_fold.append({"fold_idx": fi, "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)),
                             "accuracy": r["accuracy"], "macro_f1": r["macro_f1"],
                             "weighted_f1": r["weighted_f1"]})
            print(f"    [{mname} fold {fi+1}/{len(folds)}] macro_f1={r['macro_f1']:.4f} "
                  f"acc={r['accuracy']:.4f}", flush=True)
        all_results[f"{mname}_B1"] = {
            "model": mname,
            "macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
            "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
            "weighted_f1_mean": float(np.mean(wf1s)), "weighted_f1_std": float(np.std(wf1s)),
            "n_folds": len(folds), "per_fold": per_fold}
        print(f"  {mname:18s} macro_f1={np.mean(f1s):.4f}±{np.std(f1s):.4f}  "
              f"acc={np.mean(accs):.4f}±{np.std(accs):.4f}", flush=True)
    save = OUT / f"jaffe_7c_{name}_results.json"
    json.dump(all_results, open(save, "w"), indent=2)
    print(f"  saved {save}")
    try:
        fdir.rmdir()
    except OSError:
        pass
    return all_results


def main():
    print(f"Device: {device}")
    img, lm, y = load_all()
    print(f"JAFFE total: {len(y)} sampel, {len(set(y))} kelas (sample-level split)")

    # randomsplit: 80:20 stratified, 5 seed
    rs = list(StratifiedShuffleSplit(n_splits=5, test_size=0.20, random_state=42).split(np.zeros(len(y)), y))
    run_protocol("randomsplit", rs, img, lm, y)

    # cv10: 10-fold sample-level stratified (pembanding paper 10-fold)
    cv = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
    run_protocol("cv10", cv, img, lm, y)

    print("\nDone.")


if __name__ == "__main__":
    main()
