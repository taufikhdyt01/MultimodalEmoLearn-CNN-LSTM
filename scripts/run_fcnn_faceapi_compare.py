#!/usr/bin/env python3
"""
Apples-to-apples comparison: MediaPipe vs face-api.js landmarks untuk FCNN B1.

Same setup sebagai run_cnn1d_faceapi_compare.py tapi pakai EmotionFCNN
(flat 136-dim input, 5 Dense layers). Tujuannya: cek apakah lift face-api.js
robust across arsitektur (FCNN ablation).

Usage:
  CUDA_VISIBLE_DEVICES=1 python scripts/run_fcnn_faceapi_compare.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from training.models import EmotionFCNN  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "dataset_frontonly_conf60"

REMAP_3 = np.array([1, 0, 2, 2, 2, 2, 0], dtype=np.int64)

BATCH = 32
EPOCHS = 50
PATIENCE = 15
LR = 1e-3
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load_data(source: str, num_classes: int):
    out = {}
    fname = "X_{}_landmarks.npy" if source == "mediapipe" else "X_{}_faceapi_landmarks.npy"
    for split in ("train", "val", "test"):
        mask = np.load(DATA_DIR / f"mask_{split}_faceapi.npy")
        assert mask.all()
        X = np.load(DATA_DIR / fname.format(split)).astype(np.float32)
        y = np.load(DATA_DIR / f"y_{split}.npy").astype(np.int64)
        assert not np.isnan(X).any()
        if num_classes == 3:
            y = REMAP_3[y]
        out[split] = (X, y)
    return out


def make_loader(X, y, shuffle):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, drop_last=shuffle,
                      num_workers=0, pin_memory=True)


@torch.no_grad()
def evaluate(model, loader, num_classes):
    model.eval()
    preds, targets = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        preds.append(model(xb).argmax(1).cpu().numpy())
        targets.append(yb.numpy())
    preds = np.concatenate(preds); targets = np.concatenate(targets)
    return {
        "accuracy": float(accuracy_score(targets, preds)),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(targets, preds, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(targets, preds, labels=list(range(num_classes))).tolist(),
    }


def train_run(data, source: str, num_classes: int):
    print(f"\n========== FCNN  {source} × {num_classes}c ==========")
    Xtr, ytr = data["train"]; Xva, yva = data["val"]; Xte, yte = data["test"]
    print(f"  train: {Xtr.shape}  counts: {np.bincount(ytr, minlength=num_classes).tolist()}")

    set_seed(SEED)
    model = EmotionFCNN(input_dim=136, num_classes=num_classes).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()

    tl = make_loader(Xtr, ytr, True)
    vl = make_loader(Xva, yva, False)
    el = make_loader(Xte, yte, False)

    best_val = -1.0; best_state = None; best_epoch = -1; no_imp = 0
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0; n = 0
        for xb, yb in tl:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            optim.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward(); optim.step()
            total += loss.item() * xb.size(0); n += xb.size(0)
        vm = evaluate(model, vl, num_classes)
        if epoch == 1 or epoch % 5 == 0:
            print(f"  epoch {epoch:3d}  loss={total/n:.4f}  val_mf1={vm['macro_f1']:.4f}  val_acc={vm['accuracy']:.4f}")
        if vm["macro_f1"] > best_val:
            best_val = vm["macro_f1"]; best_epoch = epoch; no_imp = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                print(f"  early stop at {epoch}")
                break
    elapsed = time.time() - t0
    model.load_state_dict(best_state)
    test_m = evaluate(model, el, num_classes)
    print(f"  best epoch {best_epoch}  val_mf1={best_val:.4f}  ({elapsed:.0f}s)")
    print(f"  TEST: macro_f1={test_m['macro_f1']:.4f}  wf1={test_m['weighted_f1']:.4f}  acc={test_m['accuracy']:.4f}")
    return {
        "source": source, "num_classes": num_classes, "model": "EmotionFCNN",
        "best_epoch": best_epoch, "elapsed_sec": elapsed,
        "n_train": int(len(Xtr)), "n_val": int(len(Xva)), "n_test": int(len(Xte)),
        "val_macro_f1_best": best_val,
        "test": test_m,
    }


def main():
    print(f"Device: {device}")
    all_results = {}
    for nc in (3, 7):
        for src in ("mediapipe", "faceapi"):
            data = load_data(src, nc)
            all_results[f"{src}_{nc}c"] = train_run(data, src, nc)

    for nc in (3, 7):
        out_dir = PROJECT_ROOT / "models" / "frontonly_conf60" / f"{nc}class" / "FCNN_compare"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_json = out_dir / "compare_landmark_source.json"
        with open(out_json, "w") as f:
            json.dump({
                "config": f"fcnn_b1_{nc}c_landmark_source_compare",
                "mediapipe": all_results[f"mediapipe_{nc}c"],
                "faceapi": all_results[f"faceapi_{nc}c"],
            }, f, indent=2)
        print(f"\nSaved: {out_json}")

    print("\n" + "=" * 60)
    print("Summary: FCNN B1, MediaPipe vs face-api.js (full 6795 samples)")
    print("=" * 60)
    for nc in (3, 7):
        mp = all_results[f"mediapipe_{nc}c"]["test"]
        fa = all_results[f"faceapi_{nc}c"]["test"]
        print(f"\n  --- {nc}-class ---")
        print(f"  {'metric':<20s}  {'MediaPipe':>10s}  {'face-api.js':>12s}  {'delta':>8s}")
        for k in ("macro_f1", "weighted_f1", "accuracy"):
            d = fa[k] - mp[k]
            print(f"  test_{k:<14s}  {mp[k]:>10.4f}  {fa[k]:>12.4f}  {d:>+8.4f}")


if __name__ == "__main__":
    main()
