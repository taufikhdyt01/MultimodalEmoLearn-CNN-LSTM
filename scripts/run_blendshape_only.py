#!/usr/bin/env python3
"""
Eksperimen: Blendshape-only sebagai geometric feature.

52-dim MediaPipe blendshape coefficients dipakai langsung sebagai input
(no landmark, no head pose). Blendshape adalah aktivasi otot wajah
ARKit-style yang serupa FACS Action Units, di-extract via deep learning
(bukan dari manual landmark distance).

Setup:
  - Input: 52-dim blendshape coefficients dari MediaPipe FaceLandmarker v2
  - Model: dua arsitektur (kita CNN1D_FACS-style + Bachtiar Multi-stacked CNN)
  - Skema: 3c & 7c
  - Scenario: B1 (no class weight, no aug)
  - Total: 2 archs × 2 schemes = 4 runs

Usage:
  CUDA_VISIBLE_DEVICES=2 python scripts/run_blendshape_only.py
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
from training.models import EmotionCNN1D_FACS, MultiStackedCNN_Bachtiar  # noqa: E402
from training.exp_utils import (  # noqa: E402
    class_counts, evaluate_full, make_run_record,
)

DATA_DIR = PROJECT_ROOT / "data" / "dataset_frontonly_conf60"
REMAP_3 = np.array([1, 0, 2, 2, 2, 2, 0], dtype=np.int64)
CLASS_NAMES_7 = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]
CLASS_NAMES_3 = ["positive", "neutral", "negative"]

# 52 ARKit blendshape category names (urutan dari MediaPipe FaceLandmarker v2)
BLENDSHAPE_NAMES = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight", "cheekPuff", "cheekSquintLeft",
    "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft",
    "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
    "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft",
    "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft",
    "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft",
    "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
    "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
]
assert len(BLENDSHAPE_NAMES) == 52

BATCH = 32
EPOCHS = 50
PATIENCE = 15
LR = 1e-3
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load_data(num_classes: int):
    out = {}
    Xs = {s: np.load(DATA_DIR / f"X_{s}_mp_blendshapes.npy").astype(np.float32)
          for s in ("train", "val", "test")}
    # Impute NaN with median of training data per feature
    median_train = np.nanmedian(Xs["train"], axis=0)
    median_train = np.where(np.isnan(median_train), 0.0, median_train)
    for s in ("train", "val", "test"):
        X = Xs[s]
        nan_rows = int(np.isnan(X).any(axis=1).sum())
        X = np.where(np.isnan(X), median_train, X).astype(np.float32)
        y = np.load(DATA_DIR / f"y_{s}.npy").astype(np.int64)
        if num_classes == 3:
            y = REMAP_3[y]
        out[s] = (X, y, nan_rows)
    return out


def make_loader(X, y, shuffle):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, drop_last=shuffle,
                      num_workers=0, pin_memory=True)


def train_run(data, num_classes: int, arch_name: str, model_factory):
    cls_names = CLASS_NAMES_3 if num_classes == 3 else CLASS_NAMES_7
    Xtr, ytr, _ = data["train"]; Xva, yva, _ = data["val"]; Xte, yte, _ = data["test"]
    print(f"\n========== {arch_name} × {num_classes}c × B1 (blendshape only) ==========")
    print(f"  feature dim: {Xtr.shape[1]}  range: [{Xtr.min():.3f}, {Xtr.max():.3f}]  mean: {Xtr.mean():.3f}")

    set_seed(SEED)
    model = model_factory(num_classes=num_classes, in_dim=Xtr.shape[1]).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()

    tl = make_loader(Xtr, ytr, True)
    vl = make_loader(Xva, yva, False)
    el = make_loader(Xte, yte, False)

    record = make_run_record(
        config=f"blendshape_{arch_name}_{num_classes}c_b1",
        notes="Pure 52-dim MediaPipe blendshape coefficients as input (no landmark, no head pose).",
        hyperparams={
            "batch_size": BATCH, "epochs_max": EPOCHS, "patience": PATIENCE,
            "lr": LR, "optimizer": "Adam", "loss": "CrossEntropyLoss",
            "seed": SEED, "scenario": "B1",
        },
        dataset={
            "data_dir": str(DATA_DIR.relative_to(PROJECT_ROOT)),
            "feature_kind": "mp_blendshape_52",
            "feature_dim": Xtr.shape[1],
            "feature_names": BLENDSHAPE_NAMES,
            "landmark_source": "mediapipe_facelandmarker_v2_blendshapes",
            "num_classes": num_classes,
            "class_names": cls_names,
            "n_train": int(len(Xtr)), "n_val": int(len(Xva)), "n_test": int(len(Xte)),
            "class_counts_train": class_counts(ytr, num_classes),
            "class_counts_val": class_counts(yva, num_classes),
            "class_counts_test": class_counts(yte, num_classes),
            "nan_imputed": {s: data[s][2] for s in ("train", "val", "test")},
        },
        model=model,
    )
    record["arch_name"] = arch_name

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    best_val = -1.0; best_state = None; best_epoch = -1; no_imp = 0
    history = []
    t0 = time.time()
    early_stopped = False
    epochs_done = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        et0 = time.time()
        total = 0.0; correct = 0; n = 0
        for xb, yb in tl:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            optim.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward(); optim.step()
            total += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            n += xb.size(0)
        epoch_time = time.time() - et0
        train_loss = total / max(n, 1)
        train_acc = correct / max(n, 1)
        vm = evaluate_full(model, vl, num_classes, cls_names, device=device)
        history.append({
            "epoch": epoch, "epoch_time_sec": epoch_time,
            "train_loss": train_loss, "train_accuracy": train_acc,
            "val_macro_f1": vm["macro_f1"], "val_weighted_f1": vm["weighted_f1"],
            "val_accuracy": vm["accuracy"],
        })
        epochs_done = epoch
        if epoch == 1 or epoch % 10 == 0:
            print(f"  ep {epoch:3d}  loss={train_loss:.4f}  tr_acc={train_acc:.4f}  val_mf1={vm['macro_f1']:.4f}  val_acc={vm['accuracy']:.4f}")
        if vm["macro_f1"] > best_val:
            best_val = vm["macro_f1"]; best_epoch = epoch; no_imp = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                early_stopped = True
                break
    elapsed = time.time() - t0
    peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)
                    if device.type == "cuda" else 0.0)

    model.load_state_dict(best_state)
    test_m = evaluate_full(model, el, num_classes, cls_names, device=device)
    val_m_best = evaluate_full(model, vl, num_classes, cls_names, device=device)

    print(f"  best epoch {best_epoch}  val_mf1={best_val:.4f}  ({elapsed:.0f}s)")
    print(f"  TEST: macro_f1={test_m['macro_f1']:.4f}  wf1={test_m['weighted_f1']:.4f}  acc={test_m['accuracy']:.4f}")

    record["training"] = {
        "elapsed_sec": elapsed, "epochs_completed": epochs_done,
        "best_epoch": best_epoch, "early_stopped": early_stopped,
        "peak_vram_mb": float(peak_vram_mb), "history": history,
    }
    record["test"] = test_m
    record["val_at_best"] = val_m_best
    return record


ARCHITECTURES = {
    "ours_cnn1d_facs": EmotionCNN1D_FACS,
    "bachtiar_multistack": MultiStackedCNN_Bachtiar,
}


def main():
    print(f"Device: {device}")
    all_records = {}
    for nc in (3, 7):
        data = load_data(nc)
        for arch_name, factory in ARCHITECTURES.items():
            all_records[f"{arch_name}_{nc}c"] = train_run(data, nc, arch_name, factory)

    for nc in (3, 7):
        out_dir = PROJECT_ROOT / "models" / "frontonly_conf60" / f"{nc}class" / "Blendshape_compare"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_json = out_dir / "compare_archs.json"
        with open(out_json, "w") as f:
            json.dump({
                "config": f"blendshape_{nc}c_arch_compare",
                "runs": {k: v for k, v in all_records.items() if f"_{nc}c" in k},
            }, f, indent=2)
        print(f"Saved: {out_json}")

    print("\n" + "=" * 70)
    print("Summary: Blendshape-only (52-dim), B1 — Ours vs Bachtiar")
    print("=" * 70)
    print(f"  {'scheme':<6s}  {'ours_macro_f1':>16s}  {'bachtiar_macro_f1':>20s}  {'delta':>8s}")
    for nc in (3, 7):
        ours = all_records[f"ours_cnn1d_facs_{nc}c"]["test"]["macro_f1"]
        bach = all_records[f"bachtiar_multistack_{nc}c"]["test"]["macro_f1"]
        print(f"  {nc}c     {ours:>16.4f}  {bach:>20.4f}  {bach - ours:>+8.4f}")


if __name__ == "__main__":
    main()
