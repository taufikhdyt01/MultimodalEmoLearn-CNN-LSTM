#!/usr/bin/env python3
"""
Eksplorasi: CNN dengan fitur geometrik FACS-decomposed (jarak Euclidean).

Arahan dosen: "Coba model CNN dengan fitur geometrik (yang didekomposisi -
mencari jarak euclid dari acuan FACS - Facial Action Coding System)".

Setup:
  - Input: 28 jarak Euclidean antar landmark pair sesuai FACS Action Units
  - Normalisasi: dibagi inter-ocular distance d(36, 45) untuk scale invariance
  - Model: EmotionCNN1D_FACS (3 Conv1d blocks + GAP + FC) — 96K params
  - Full sweep: 2 sources × 2 schemes × 2 scenarios = 8 runs
      sources:    {mediapipe, faceapi}
      schemes:    {3-class, 7-class}
      scenarios:  {B1 no-weight, B2 balanced-weight}

Usage:
  CUDA_VISIBLE_DEVICES=1 python scripts/run_cnn1d_facs.py
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
from training.models import EmotionCNN1D_FACS  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "dataset_frontonly_conf60"

# 28 FACS-decomposed landmark pairs (dlib 68-point indexing).
# Each entry maps a FACS Action Unit (AU) to a landmark pair whose
# Euclidean distance encodes that AU's geometric activation.
FACS_PAIRS = [
    # AU1 — Inner Brow Raiser (brow ↑↑ relatif eye)
    ("AU1_inner_brow_R", 21, 39),
    ("AU1_inner_brow_L", 22, 42),
    # AU2 — Outer Brow Raiser
    ("AU2_outer_brow_R", 17, 36),
    ("AU2_outer_brow_L", 26, 45),
    # AU4 — Brow Lowerer (frown / glabella)
    ("AU4_frown_inner_brows", 21, 22),
    ("AU4_brow_to_eye_R", 19, 37),
    ("AU4_brow_to_eye_L", 24, 44),
    # AU5/AU7 — Eye Opening / Lid Tightener
    ("AU5_eye_open_R_1", 37, 41),
    ("AU5_eye_open_R_2", 38, 40),
    ("AU5_eye_open_L_1", 43, 47),
    ("AU5_eye_open_L_2", 44, 46),
    # AU6 — Cheek Raiser
    ("AU6_cheek_R", 36, 31),
    ("AU6_cheek_L", 45, 35),
    # AU9 — Nose Wrinkler
    ("AU9_nose_width", 31, 35),
    ("AU9_nose_bridge", 27, 30),
    # AU10 — Upper Lip Raiser
    ("AU10_nose_to_upperlip", 33, 51),
    ("AU10_lip_top_to_nose_R", 50, 33),
    ("AU10_lip_top_to_nose_L", 52, 33),
    # AU12 — Lip Corner Puller (smile)
    ("AU12_corner_R_to_eye", 48, 36),
    ("AU12_corner_L_to_eye", 54, 45),
    # AU15 — Lip Corner Depressor
    ("AU15_corner_R_to_chin", 48, 8),
    ("AU15_corner_L_to_chin", 54, 8),
    # AU20/23 — Mouth Stretcher / Tightener
    ("AU20_mouth_width_outer", 48, 54),
    ("AU23_mouth_width_inner", 60, 64),
    # AU25/26 — Lips Part / Jaw Drop
    ("AU25_lip_open_outer", 51, 57),
    ("AU25_lip_open_inner", 62, 66),
    ("AU26_nose_to_chin", 33, 8),
    # Face shape (reference normalization)
    ("face_height", 8, 27),
]

# Inter-ocular distance for normalization (outer eye corners)
INTEROCULAR_PAIR = (36, 45)

REMAP_3 = np.array([1, 0, 2, 2, 2, 2, 0], dtype=np.int64)

BATCH = 32
EPOCHS = 50
PATIENCE = 15
LR = 1e-3
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def compute_facs_distances(landmarks_flat: np.ndarray) -> np.ndarray:
    """(N, 136) → (N, len(FACS_PAIRS)) normalized Euclidean distances."""
    pts = landmarks_flat.reshape(-1, 68, 2)  # (N, 68, 2)
    # Inter-ocular distance per sample
    a, b = INTEROCULAR_PAIR
    iod = np.linalg.norm(pts[:, a] - pts[:, b], axis=1)  # (N,)
    iod = np.maximum(iod, 1e-6)

    n_pairs = len(FACS_PAIRS)
    out = np.zeros((len(pts), n_pairs), dtype=np.float32)
    for i, (_, p1, p2) in enumerate(FACS_PAIRS):
        d = np.linalg.norm(pts[:, p1] - pts[:, p2], axis=1)  # (N,)
        out[:, i] = (d / iod).astype(np.float32)
    return out


def load_data(source: str, num_classes: int):
    out = {}
    fname = "X_{}_landmarks.npy" if source == "mediapipe" else "X_{}_faceapi_landmarks.npy"
    for split in ("train", "val", "test"):
        mask = np.load(DATA_DIR / f"mask_{split}_faceapi.npy")
        assert mask.all()
        X_raw = np.load(DATA_DIR / fname.format(split)).astype(np.float32)
        assert not np.isnan(X_raw).any()
        X = compute_facs_distances(X_raw)
        y = np.load(DATA_DIR / f"y_{split}.npy").astype(np.int64)
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


def train_run(data, source: str, num_classes: int, scenario: str):
    print(f"\n========== FACS-CNN1D  {source} × {num_classes}c × {scenario.upper()} ==========")
    Xtr, ytr = data["train"]; Xva, yva = data["val"]; Xte, yte = data["test"]
    print(f"  train: {Xtr.shape}  counts: {np.bincount(ytr, minlength=num_classes).tolist()}")
    print(f"  feature stats: min={Xtr.min():.3f} max={Xtr.max():.3f} mean={Xtr.mean():.3f}")

    set_seed(SEED)
    model = EmotionCNN1D_FACS(num_classes=num_classes, in_dim=Xtr.shape[1]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if scenario == "b2":
        cw = compute_class_weight("balanced", classes=np.arange(num_classes), y=ytr).astype(np.float32)
        print(f"  class weights: {cw.round(2).tolist()}")
        crit = nn.CrossEntropyLoss(weight=torch.from_numpy(cw).to(device))
    else:
        crit = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=LR)

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
        if epoch == 1 or epoch % 10 == 0:
            print(f"  epoch {epoch:3d}  loss={total/n:.4f}  val_mf1={vm['macro_f1']:.4f}  val_acc={vm['accuracy']:.4f}")
        if vm["macro_f1"] > best_val:
            best_val = vm["macro_f1"]; best_epoch = epoch; no_imp = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                break
    elapsed = time.time() - t0
    model.load_state_dict(best_state)
    test_m = evaluate(model, el, num_classes)
    print(f"  best epoch {best_epoch}  val_mf1={best_val:.4f}  ({elapsed:.0f}s)  params={n_params:,}")
    print(f"  TEST: macro_f1={test_m['macro_f1']:.4f}  wf1={test_m['weighted_f1']:.4f}  acc={test_m['accuracy']:.4f}")
    return {
        "source": source, "num_classes": num_classes, "scenario": scenario.upper(),
        "model": "EmotionCNN1D_FACS", "n_params": int(n_params),
        "best_epoch": best_epoch, "elapsed_sec": elapsed,
        "n_train": int(len(Xtr)), "n_val": int(len(Xva)), "n_test": int(len(Xte)),
        "val_macro_f1_best": best_val,
        "test": test_m,
    }


def main():
    print(f"Device: {device}")
    print(f"FACS pairs: {len(FACS_PAIRS)}")
    all_results = {}
    for nc in (3, 7):
        for src in ("mediapipe", "faceapi"):
            data = load_data(src, nc)
            for scen in ("b1", "b2"):
                key = f"{src}_{nc}c_{scen}"
                all_results[key] = train_run(data, src, nc, scen)

    # Save per (class, scenario) for easy diff
    for nc in (3, 7):
        out_dir = PROJECT_ROOT / "models" / "frontonly_conf60" / f"{nc}class" / "CNN1D_FACS"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = {
            "config": f"cnn1d_facs_{nc}c_sweep",
            "facs_pairs": [{"name": n, "p1": p1, "p2": p2} for n, p1, p2 in FACS_PAIRS],
            "interocular_pair": list(INTEROCULAR_PAIR),
            "results": {k: v for k, v in all_results.items() if f"_{nc}c_" in k},
        }
        out_json = out_dir / "facs_sweep_results.json"
        with open(out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved: {out_json}")

    # Final summary table
    print("\n" + "=" * 80)
    print("Summary: FACS-CNN1D macro_f1")
    print("=" * 80)
    print(f"  {'scheme':<6s}  {'scen':<3s}  {'mediapipe':>10s}  {'face-api.js':>12s}  {'delta':>8s}")
    for nc in (3, 7):
        for scen in ("b1", "b2"):
            mp = all_results[f"mediapipe_{nc}c_{scen}"]["test"]["macro_f1"]
            fa = all_results[f"faceapi_{nc}c_{scen}"]["test"]["macro_f1"]
            print(f"  {nc}c    {scen.upper():<3s}  {mp:>10.4f}  {fa:>12.4f}  {fa - mp:>+8.4f}")


if __name__ == "__main__":
    main()
