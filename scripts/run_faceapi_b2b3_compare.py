#!/usr/bin/env python3
"""
Lengkapi gap: B2 & B3 untuk MediaPipe vs face-api.js, CNN1D & FCNN, 3c & 7c.

Sebelumnya hanya B1 yang ada di:
  - models/frontonly_conf60/{3,7}class/CNN1D_geom_compare/compare_landmark_source.json
  - models/frontonly_conf60/{3,7}class/FCNN_compare/compare_landmark_source.json

Script ini menjalankan 16 run (2 models x 2 scenarios x 2 sources x 2 schemes)
di subset 6795 sampel matched (sama seperti B1 compare).

Usage:
  CUDA_VISIBLE_DEVICES=1 python scripts/run_faceapi_b2b3_compare.py
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
from sklearn.utils.class_weight import compute_class_weight

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from training.models import EmotionCNN1D, EmotionFCNN  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "dataset_frontonly_conf60"

REMAP_3 = np.array([1, 0, 2, 2, 2, 2, 0], dtype=np.int64)

BATCH = 32
EPOCHS = 50
PATIENCE = 15
LR = 1e-3
SEED = 42

AUG_TECHNIQUES = ["hflip", "rotate_pos", "rotate_neg", "flip_rot"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data(source: str, num_classes: int):
    out = {}
    fname = "X_{}_landmarks.npy" if source == "mediapipe" else "X_{}_faceapi_landmarks.npy"
    for split in ("train", "val", "test"):
        mask = np.load(DATA_DIR / f"mask_{split}_faceapi.npy")
        assert mask.all(), f"{split} mask not 100% (got {mask.sum()}/{len(mask)})"
        X = np.load(DATA_DIR / fname.format(split)).astype(np.float32)
        y = np.load(DATA_DIR / f"y_{split}.npy").astype(np.int64)
        assert not np.isnan(X).any(), f"NaN in {source}/{split}"
        if num_classes == 3:
            y = REMAP_3[y]
        out[split] = (X, y)
    return out


def augment_landmark(lm_136: np.ndarray, technique: str) -> np.ndarray:
    pts = lm_136.reshape(-1, 2).copy()
    if technique == "hflip":
        pts[:, 0] = 1.0 - pts[:, 0]
        return pts.flatten().astype(np.float32)
    if technique in ("rotate_pos", "rotate_neg", "flip_rot"):
        if technique == "rotate_pos":
            angle = np.deg2rad(10)
        elif technique == "rotate_neg":
            angle = np.deg2rad(-10)
        else:
            pts[:, 0] = 1.0 - pts[:, 0]
            angle = np.deg2rad(5)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        centered = pts - 0.5
        rotated = np.stack([
            centered[:, 0] * cos_a - centered[:, 1] * sin_a,
            centered[:, 0] * sin_a + centered[:, 1] * cos_a,
        ], axis=1)
        return (rotated + 0.5).flatten().astype(np.float32)
    raise ValueError(technique)


def build_augmented_train(X, y, num_classes, seed=SEED):
    rng = np.random.default_rng(seed)
    counts = np.bincount(y, minlength=num_classes)
    target = int(counts.max())
    aug_X, aug_y, meta = [X], [y], {}
    for c in range(num_classes):
        deficit = target - counts[c]
        if deficit <= 0 or counts[c] == 0:
            meta[c] = 0
            continue
        idx = np.where(y == c)[0]
        new_X = np.zeros((deficit, X.shape[1]), dtype=np.float32)
        for k in range(deficit):
            src_i = idx[rng.integers(0, len(idx))]
            tech = AUG_TECHNIQUES[k % len(AUG_TECHNIQUES)]
            new_X[k] = augment_landmark(X[src_i], tech)
        new_y = np.full(deficit, c, dtype=np.int64)
        aug_X.append(new_X)
        aug_y.append(new_y)
        meta[c] = int(deficit)
    X_out = np.concatenate(aug_X)
    y_out = np.concatenate(aug_y)
    perm = rng.permutation(len(X_out))
    return X_out[perm], y_out[perm], meta


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
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    return {
        "accuracy": float(accuracy_score(targets, preds)),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(targets, preds, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(targets, preds, labels=list(range(num_classes))).tolist(),
    }


def build_model(model_kind: str, num_classes: int):
    if model_kind == "cnn1d":
        return EmotionCNN1D(num_classes=num_classes)
    if model_kind == "fcnn":
        return EmotionFCNN(input_dim=136, num_classes=num_classes)
    raise ValueError(model_kind)


def train_run(data, *, model_kind: str, source: str, num_classes: int, scenario: str):
    print(f"\n========== {model_kind.upper()}  {source}  {num_classes}c  {scenario.upper()} ==========")
    Xtr, ytr = data["train"]
    Xva, yva = data["val"]
    Xte, yte = data["test"]

    set_seed(SEED)

    aug_meta = None
    if scenario == "b3":
        Xtr, ytr, aug_meta = build_augmented_train(Xtr, ytr, num_classes)
        print(f"  post-aug train counts: {np.bincount(ytr, minlength=num_classes).tolist()}  added: {aug_meta}")

    print(f"  train: {Xtr.shape}  counts: {np.bincount(ytr, minlength=num_classes).tolist()}")

    if scenario in ("b2", "b3"):
        class_w = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(num_classes),
            y=ytr,
        ).astype(np.float32)
        print(f"  class weights: {class_w.round(3).tolist()}")
        crit = nn.CrossEntropyLoss(weight=torch.from_numpy(class_w).to(device))
    else:
        crit = nn.CrossEntropyLoss()

    model = build_model(model_kind, num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    tl = make_loader(Xtr, ytr, True)
    vl = make_loader(Xva, yva, False)
    el = make_loader(Xte, yte, False)

    best_val = -1.0
    best_state = None
    best_epoch = -1
    no_imp = 0
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        n = 0
        for xb, yb in tl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optim.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            optim.step()
            total += loss.item() * xb.size(0)
            n += xb.size(0)
        vm = evaluate(model, vl, num_classes)
        if epoch == 1 or epoch % 5 == 0:
            print(f"  epoch {epoch:3d}  loss={total/n:.4f}  val_mf1={vm['macro_f1']:.4f}  val_acc={vm['accuracy']:.4f}")
        if vm["macro_f1"] > best_val:
            best_val = vm["macro_f1"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
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
        "model": "EmotionCNN1D" if model_kind == "cnn1d" else "EmotionFCNN",
        "source": source,
        "num_classes": num_classes,
        "scenario": scenario.upper(),
        "n_params": int(n_params),
        "best_epoch": int(best_epoch),
        "elapsed_sec": float(elapsed),
        "n_train": int(len(Xtr)),
        "n_val": int(len(Xva)),
        "n_test": int(len(Xte)),
        "val_macro_f1_best": float(best_val),
        "aug_per_class_added": aug_meta,
        "test": test_m,
    }


def main():
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"  PyTorch: {torch.__version__}  CUDA: {torch.version.cuda}")

    all_results = {}
    for model_kind in ("cnn1d", "fcnn"):
        for nc in (3, 7):
            data_cache = {src: load_data(src, nc) for src in ("mediapipe", "faceapi")}
            for scenario in ("b2", "b3"):
                for src in ("mediapipe", "faceapi"):
                    key = f"{model_kind}_{src}_{nc}c_{scenario}"
                    all_results[key] = train_run(
                        data_cache[src],
                        model_kind=model_kind,
                        source=src,
                        num_classes=nc,
                        scenario=scenario,
                    )

    # Save: extend existing compare JSONs with b2/b3 entries
    for model_kind in ("cnn1d", "fcnn"):
        comp_dir_name = "CNN1D_geom_compare" if model_kind == "cnn1d" else "FCNN_compare"
        for nc in (3, 7):
            out_dir = PROJECT_ROOT / "models" / "frontonly_conf60" / f"{nc}class" / comp_dir_name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_json = out_dir / "compare_landmark_source_b2b3.json"
            payload = {
                "config": f"{model_kind}_b2b3_{nc}c_landmark_source_compare",
                "b2": {
                    "mediapipe": all_results[f"{model_kind}_mediapipe_{nc}c_b2"],
                    "faceapi": all_results[f"{model_kind}_faceapi_{nc}c_b2"],
                },
                "b3": {
                    "mediapipe": all_results[f"{model_kind}_mediapipe_{nc}c_b3"],
                    "faceapi": all_results[f"{model_kind}_faceapi_{nc}c_b3"],
                },
            }
            with open(out_json, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"\nSaved: {out_json}")

    # Final summary
    print("\n" + "=" * 78)
    print("SUMMARY — face-api.js B2/B3 extension (test_macro_f1)")
    print("=" * 78)
    header = f"  {'model':<8s} {'scheme':<6s} {'scenario':<9s} {'MP':>8s} {'face-api':>9s} {'delta':>8s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for model_kind in ("cnn1d", "fcnn"):
        for nc in (3, 7):
            for scenario in ("b2", "b3"):
                mp = all_results[f"{model_kind}_mediapipe_{nc}c_{scenario}"]["test"]["macro_f1"]
                fa = all_results[f"{model_kind}_faceapi_{nc}c_{scenario}"]["test"]["macro_f1"]
                d = fa - mp
                print(f"  {model_kind:<8s} {nc}c     {scenario.upper():<9s} {mp:>8.4f} {fa:>9.4f} {d:>+8.4f}")


if __name__ == "__main__":
    main()
