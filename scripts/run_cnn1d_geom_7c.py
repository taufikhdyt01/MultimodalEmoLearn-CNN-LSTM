#!/usr/bin/env python3
"""
Eksplorasi: CNN 1D dengan fitur geometrik baseline (raw landmark coordinates).

Arahan dosen: "Coba model CNN dengan fitur geometrik (baseline - hanya koordinat titik)"

Setup:
  - Primer conf60 (train/val/test split asli, single split)
  - Input: 68 landmark × 2 (x, y) → Conv1d sequence
  - Skenario:
      B1 = no augmentation, no class weights
      B2 = balanced class weights
      B3 = balanced class weights + on-the-fly landmark augmentation
  - Bisa pakai 7-class (default) atau 3-class (remap)

Baseline pembanding:
  - 7c FCNN: B1=0.2317, B2=0.2437, B3=0.2224 (macro_f1)
  - 3c FCNN: B1=0.5893, B2=0.5750, B3=0.6342 (macro_f1)

Usage:
  CUDA_VISIBLE_DEVICES=1 python scripts/run_cnn1d_geom_7c.py --scenario b1
  CUDA_VISIBLE_DEVICES=1 python scripts/run_cnn1d_geom_7c.py --scenario b3 --classes 3
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from training.models import EmotionCNN1D  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "dataset_frontonly_conf60"

# 7-class -> 3-class remap (matching pipeline elsewhere)
# Original: angry=0, disgust=1, fear=2, happy=3, neutral=4, sad=5, surprise=6
# Note: y_*.npy in this dataset uses different ordering — see REMAP below.
# We replicate the convention from nb 81 / scripts/make_nb81_geometric.py:
#   REMAP_3 = [1, 0, 2, 2, 2, 2, 0]  (positive=0, neutral=1, negative=2 in *original* label order
#   used by this repo's stored y_*.npy — see EMOTIONS list).
EMOTIONS_7 = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]
EMOTIONS_3 = ["positive", "neutral", "negative"]
# Map: neutral->1(neutral), happy->0(positive), sad/angry/fearful/disgusted->2(negative), surprised->0(positive)
REMAP_3 = np.array([1, 0, 2, 2, 2, 2, 0], dtype=np.int64)

BATCH = 32
EPOCHS = 50
PATIENCE = 15
LR = 1e-3
SEED = 42

# B3 on-the-fly augmentation parameters (matches augment_conf60_3class.py spirit)
AUG_TECHNIQUES = ["hflip", "rotate_pos", "rotate_neg", "flip_rot"]
AUG_TARGET_MIN_FRAC = 1.0  # augment minority classes up to majority size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split(split: str, num_classes: int):
    X = np.load(DATA_DIR / f"X_{split}_landmarks.npy").astype(np.float32)
    y = np.load(DATA_DIR / f"y_{split}.npy").astype(np.int64)
    if num_classes == 3:
        y = REMAP_3[y]
    return X, y


def augment_landmark(lm_136: np.ndarray, technique: str) -> np.ndarray:
    """Same geometric augmentation as src/preprocessing/augment_conf60_3class.py."""
    pts = lm_136.reshape(-1, 2).copy()  # (68, 2)
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


def build_augmented_train(X: np.ndarray, y: np.ndarray, num_classes: int, seed: int = SEED):
    """Offline expansion of training set: augment minority classes to match majority.

    For each minority class, cycle through AUG_TECHNIQUES to fill samples.
    """
    rng = np.random.default_rng(seed)
    counts = np.bincount(y, minlength=num_classes)
    target = int(counts.max() * AUG_TARGET_MIN_FRAC)

    aug_X = [X]
    aug_y = [y]
    aug_meta = {}
    for c in range(num_classes):
        deficit = target - counts[c]
        if deficit <= 0 or counts[c] == 0:
            aug_meta[c] = 0
            continue
        idx = np.where(y == c)[0]
        # Cycle techniques across source samples
        new_X = np.zeros((deficit, X.shape[1]), dtype=np.float32)
        for k in range(deficit):
            src_i = idx[rng.integers(0, len(idx))]
            tech = AUG_TECHNIQUES[k % len(AUG_TECHNIQUES)]
            new_X[k] = augment_landmark(X[src_i], tech)
        new_y = np.full(deficit, c, dtype=np.int64)
        aug_X.append(new_X)
        aug_y.append(new_y)
        aug_meta[c] = int(deficit)
    X_out = np.concatenate(aug_X)
    y_out = np.concatenate(aug_y)
    perm = rng.permutation(len(X_out))
    return X_out[perm], y_out[perm], aug_meta


def make_loader(X: np.ndarray, y: np.ndarray, shuffle: bool):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(
        ds,
        batch_size=BATCH,
        shuffle=shuffle,
        drop_last=shuffle,  # avoid 1-sample batch breaking BN
        num_workers=0,
        pin_memory=True,
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, num_classes: int):
    model.eval()
    preds, targets = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        preds.append(logits.argmax(1).cpu().numpy())
        targets.append(yb.numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    return {
        "accuracy": float(accuracy_score(targets, preds)),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(targets, preds, average="weighted", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            targets, preds, labels=list(range(num_classes))
        ).tolist(),
    }


def train_one(X_tr, y_tr, X_val, y_val, X_te, y_te, *, scenario: str, num_classes: int,
              log_every: int = 5):
    set_seed(SEED)

    model = EmotionCNN1D(num_classes=num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params:,}")

    optim = torch.optim.Adam(model.parameters(), lr=LR)

    aug_meta = None
    if scenario == "b3":
        X_tr, y_tr, aug_meta = build_augmented_train(X_tr, y_tr, num_classes)
        print(f"  post-aug train: X={X_tr.shape}  per-class added: {aug_meta}")
        print(f"  post-aug class counts: {np.bincount(y_tr, minlength=num_classes).tolist()}")

    if scenario in ("b2", "b3"):
        class_w = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(num_classes),
            y=y_tr,
        ).astype(np.float32)
        print(f"  class weights: {class_w.round(3).tolist()}")
        crit = nn.CrossEntropyLoss(weight=torch.from_numpy(class_w).to(device))
    else:
        crit = nn.CrossEntropyLoss()

    train_loader = make_loader(X_tr, y_tr, shuffle=True)
    val_loader = make_loader(X_val, y_val, shuffle=False)
    test_loader = make_loader(X_te, y_te, shuffle=False)

    best_val = -1.0
    best_epoch = -1
    best_state = None
    epochs_no_improve = 0
    history = []

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optim.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            optim.step()
            total_loss += loss.item() * xb.size(0)
            n += xb.size(0)
        train_loss = total_loss / max(n, 1)

        val_metrics = evaluate(model, val_loader, num_classes)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "val_accuracy": val_metrics["accuracy"],
        })
        if epoch == 1 or epoch % log_every == 0 or epoch == EPOCHS:
            print(
                f"  epoch {epoch:3d}  train_loss={train_loss:.4f}  "
                f"val_macro_f1={val_metrics['macro_f1']:.4f}  "
                f"val_wf1={val_metrics['weighted_f1']:.4f}  "
                f"val_acc={val_metrics['accuracy']:.4f}"
            )

        if val_metrics["macro_f1"] > best_val:
            best_val = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"  early stop at epoch {epoch} (no improve {PATIENCE}).")
                break

    elapsed = time.time() - t0
    print(f"  best epoch {best_epoch}  val_macro_f1={best_val:.4f}  ({elapsed:.1f}s)")

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, num_classes)
    val_metrics_best = evaluate(model, val_loader, num_classes)

    return {
        "best_epoch": best_epoch,
        "elapsed_sec": elapsed,
        "n_params": n_params,
        "aug_per_class_added": aug_meta,
        "val": val_metrics_best,
        "test": test_metrics,
        "history": history,
    }, model


FCNN_BASELINES = {
    7: {
        "b1": {"macro_f1": 0.2317, "weighted_f1": 0.7650, "accuracy": 0.7675},
        "b2": {"macro_f1": 0.2437, "weighted_f1": 0.7674, "accuracy": 0.7653},
        "b3": {"macro_f1": 0.2224, "weighted_f1": 0.7580, "accuracy": 0.7395},
    },
    3: {
        "b1": {"macro_f1": 0.5893, "weighted_f1": 0.7596, "accuracy": 0.7406},
        "b2": {"macro_f1": 0.5750, "weighted_f1": 0.7302, "accuracy": 0.7018},
        "b3": {"macro_f1": 0.6342, "weighted_f1": 0.7636, "accuracy": 0.7492},
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["b1", "b2", "b3"], default="b1",
                        help="B1=no weight, B2=weights, B3=weights+on-the-fly aug.")
    parser.add_argument("--classes", type=int, choices=[3, 7], default=7,
                        help="7-class (default) or 3-class (remapped positive/neutral/negative).")
    parser.add_argument("--save-ckpt", action="store_true")
    args = parser.parse_args()

    num_classes = args.classes
    out_dir = (
        PROJECT_ROOT / "models" / "frontonly_conf60"
        / f"{num_classes}class" / "CNN1D_geom"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Data dir: {DATA_DIR}")
    print(f"Output dir: {out_dir}")
    print(f"Num classes: {num_classes}\n")

    X_tr, y_tr = load_split("train", num_classes)
    X_val, y_val = load_split("val", num_classes)
    X_te, y_te = load_split("test", num_classes)
    print(f"train: X={X_tr.shape}  y={y_tr.shape}")
    print(f"val:   X={X_val.shape}  y={y_val.shape}")
    print(f"test:  X={X_te.shape}  y={y_te.shape}")
    print(f"class counts (train): {np.bincount(y_tr, minlength=num_classes).tolist()}\n")

    scen_label = args.scenario.upper()
    desc = {
        "b1": "no aug, no class weight",
        "b2": "no aug, balanced class weights",
        "b3": "on-the-fly landmark aug + balanced class weights",
    }[args.scenario]
    print(f"[CNN1D {scen_label} {num_classes}c — raw landmark coordinates, {desc}]")
    result, model = train_one(X_tr, y_tr, X_val, y_val, X_te, y_te,
                              scenario=args.scenario, num_classes=num_classes)

    result["config"] = f"cnn1d_geom_{args.scenario}_{num_classes}c"
    result["model"] = "EmotionCNN1D"
    result["scenario"] = scen_label
    result["num_classes"] = num_classes
    result["input"] = {
        "kind": "raw_landmark_coords",
        "shape_per_sample": [68, 2],
        "flat_dim": 136,
    }
    result["hyperparams"] = {
        "batch": BATCH, "epochs": EPOCHS, "patience": PATIENCE,
        "lr": LR, "optimizer": "Adam", "loss": "CrossEntropyLoss",
        "seed": SEED,
    }

    out_json = out_dir / f"cnn1d_geom_{args.scenario}_results.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved: {out_json}")

    if args.save_ckpt:
        ckpt_path = out_dir / f"cnn1d_geom_{args.scenario}.pth"
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")

    fcnn = FCNN_BASELINES[num_classes][args.scenario]
    print(f"\n=== Hasil vs FCNN {scen_label} baseline ({num_classes}-class) ===")
    print(f"  metric              FCNN {scen_label}   CNN1D {scen_label}   delta")
    for k in ("macro_f1", "weighted_f1", "accuracy"):
        v = result["test"][k]
        d = v - fcnn[k]
        print(f"  test_{k:<13s}  {fcnn[k]:.4f}     {v:.4f}      {d:+.4f}")


if __name__ == "__main__":
    main()
