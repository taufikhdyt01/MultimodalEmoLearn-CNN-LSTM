#!/usr/bin/env python3
"""
Eksplorasi: kombinasi FACS-decomposed distance (28-dim) + MediaPipe blendshape (52-dim) = 80-dim.

Arahan dosen: "eksplorasi kombinasi dekomposisi FACS + Blendshape".

Setup:
  - FACS landmark source: {mediapipe, faceapi}
  - Blendshape: selalu dari MediaPipe (face-api.js tidak punya native blendshape)
  - Arsitektur: {CNN1D Ours (EmotionCNN1D_FACS), FCNN (EmotionFCNN)}
  - Skema: {3c, 7c}
  - Skenario: B1 only
  - Total: 2 sources × 2 archs × 2 schemes = 8 runs

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/run_facs_plus_blendshape.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from training.models import EmotionCNN1D_FACS, EmotionFCNN  # noqa: E402
from training.exp_utils import class_counts, evaluate_full, make_run_record  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "dataset_frontonly_conf60"
REMAP_3 = np.array([1, 0, 2, 2, 2, 2, 0], dtype=np.int64)
CLASS_NAMES_7 = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]
CLASS_NAMES_3 = ["positive", "neutral", "negative"]

FACS_PAIRS = [
    ("AU1_inner_brow_R", 21, 39), ("AU1_inner_brow_L", 22, 42),
    ("AU2_outer_brow_R", 17, 36), ("AU2_outer_brow_L", 26, 45),
    ("AU4_frown_inner_brows", 21, 22), ("AU4_brow_to_eye_R", 19, 37),
    ("AU4_brow_to_eye_L", 24, 44),
    ("AU5_eye_open_R_1", 37, 41), ("AU5_eye_open_R_2", 38, 40),
    ("AU5_eye_open_L_1", 43, 47), ("AU5_eye_open_L_2", 44, 46),
    ("AU6_cheek_R", 36, 31), ("AU6_cheek_L", 45, 35),
    ("AU9_nose_width", 31, 35), ("AU9_nose_bridge", 27, 30),
    ("AU10_nose_to_upperlip", 33, 51), ("AU10_lip_top_to_nose_R", 50, 33),
    ("AU10_lip_top_to_nose_L", 52, 33),
    ("AU12_corner_R_to_eye", 48, 36), ("AU12_corner_L_to_eye", 54, 45),
    ("AU15_corner_R_to_chin", 48, 8), ("AU15_corner_L_to_chin", 54, 8),
    ("AU20_mouth_width_outer", 48, 54), ("AU23_mouth_width_inner", 60, 64),
    ("AU25_lip_open_outer", 51, 57), ("AU25_lip_open_inner", 62, 66),
    ("AU26_nose_to_chin", 33, 8), ("face_height", 8, 27),
]
INTEROCULAR_PAIR = (36, 45)

BATCH = 32
EPOCHS = 50
PATIENCE = 15
LR = 1e-3
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def compute_facs_distances(landmarks_flat):
    pts = landmarks_flat.reshape(-1, 68, 2)
    a, b = INTEROCULAR_PAIR
    iod = np.maximum(np.linalg.norm(pts[:, a] - pts[:, b], axis=1), 1e-6)
    out = np.zeros((len(pts), len(FACS_PAIRS)), dtype=np.float32)
    for i, (_, p1, p2) in enumerate(FACS_PAIRS):
        d = np.linalg.norm(pts[:, p1] - pts[:, p2], axis=1)
        out[:, i] = (d / iod).astype(np.float32)
    return out


def load_data(facs_source, num_classes):
    """Build feature: 28 FACS distance (from `facs_source`) + 52 blendshape (always MP)."""
    out = {}
    fname = ("X_{}_landmarks.npy" if facs_source == "mediapipe"
             else "X_{}_faceapi_landmarks.npy")
    for split in ("train", "val", "test"):
        raw_lm = np.load(DATA_DIR / fname.format(split)).astype(np.float32)
        facs = compute_facs_distances(raw_lm)  # (N, 28)
        bs = np.load(DATA_DIR / f"X_{split}_mp_blendshapes.npy").astype(np.float32)  # (N, 52)
        # Impute NaN in blendshape with training median
        if split == "train":
            bs_median = np.nanmedian(bs, axis=0)
            bs_median = np.where(np.isnan(bs_median), 0.0, bs_median)
        bs = np.where(np.isnan(bs), bs_median, bs).astype(np.float32)
        # Concat: 28 + 52 = 80
        X = np.concatenate([facs, bs], axis=1)
        y = np.load(DATA_DIR / f"y_{split}.npy").astype(np.int64)
        if num_classes == 3:
            y = REMAP_3[y]
        out[split] = (X, y)
    return out


def make_loader(X, y, shuffle):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, drop_last=shuffle,
                      num_workers=0, pin_memory=True)


def train_run(data, facs_source, num_classes, arch_name, model_factory):
    cls_names = CLASS_NAMES_3 if num_classes == 3 else CLASS_NAMES_7
    Xtr, ytr = data["train"]; Xva, yva = data["val"]; Xte, yte = data["test"]
    print(f"\n========== {arch_name} × facs={facs_source} × {num_classes}c × B1 (FACS+Blendshape 80-dim) ==========")
    print(f"  feature dim: {Xtr.shape[1]}  range: [{Xtr.min():.3f}, {Xtr.max():.3f}]")

    set_seed(SEED)
    if arch_name == "fcnn":
        model = model_factory(input_dim=Xtr.shape[1], num_classes=num_classes).to(device)
    else:
        model = model_factory(num_classes=num_classes, in_dim=Xtr.shape[1]).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()

    tl = make_loader(Xtr, ytr, True)
    vl = make_loader(Xva, yva, False)
    el = make_loader(Xte, yte, False)

    record = make_run_record(
        config=f"facs+blendshape_{arch_name}_{facs_source}_{num_classes}c_b1",
        notes="Concat 28-dim FACS Euclidean distance + 52-dim MediaPipe blendshape = 80-dim feature.",
        hyperparams={"batch_size": BATCH, "epochs_max": EPOCHS, "patience": PATIENCE,
                     "lr": LR, "optimizer": "Adam", "loss": "CrossEntropyLoss",
                     "seed": SEED, "scenario": "B1"},
        dataset={"data_dir": str(DATA_DIR.relative_to(PROJECT_ROOT)),
                 "feature_kind": "facs_distance_28_+_blendshape_52",
                 "feature_dim": Xtr.shape[1],
                 "feature_breakdown": {"facs_distance": 28, "blendshape": 52},
                 "facs_landmark_source": facs_source,
                 "blendshape_source": "mediapipe",
                 "num_classes": num_classes, "class_names": cls_names,
                 "n_train": int(len(Xtr)), "n_val": int(len(Xva)), "n_test": int(len(Xte)),
                 "class_counts_train": class_counts(ytr, num_classes),
                 "class_counts_val": class_counts(yva, num_classes),
                 "class_counts_test": class_counts(yte, num_classes)},
        model=model,
    )
    record["arch_name"] = arch_name
    record["facs_source"] = facs_source

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    best_val = -1.0; best_state = None; best_epoch = -1; no_imp = 0
    history = []; t0 = time.time(); early_stopped = False; epochs_done = 0
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
        vm = evaluate_full(model, vl, num_classes, cls_names, device=device)
        history.append({"epoch": epoch, "epoch_time_sec": time.time() - et0,
                        "train_loss": total / max(n, 1), "train_accuracy": correct / max(n, 1),
                        "val_macro_f1": vm["macro_f1"], "val_weighted_f1": vm["weighted_f1"],
                        "val_accuracy": vm["accuracy"]})
        epochs_done = epoch
        if epoch == 1 or epoch % 10 == 0:
            print(f"  ep {epoch:3d}  loss={total/max(n,1):.4f}  val_mf1={vm['macro_f1']:.4f}")
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

    record["training"] = {"elapsed_sec": elapsed, "epochs_completed": epochs_done,
                          "best_epoch": best_epoch, "early_stopped": early_stopped,
                          "peak_vram_mb": float(peak_vram_mb), "history": history}
    record["test"] = test_m
    record["val_at_best"] = val_m_best
    return record


ARCHS = {
    "ours_cnn1d_facs": EmotionCNN1D_FACS,
    "fcnn": EmotionFCNN,
}


def main():
    print(f"Device: {device}")
    all_recs = {}
    for facs_src in ("mediapipe", "faceapi"):
        for nc in (3, 7):
            data = load_data(facs_src, nc)
            for arch_name, factory in ARCHS.items():
                key = f"{arch_name}_{facs_src}_{nc}c"
                all_recs[key] = train_run(data, facs_src, nc, arch_name, factory)

    for nc in (3, 7):
        out_dir = PROJECT_ROOT / "models" / "frontonly_conf60" / f"{nc}class" / "FACS_plus_Blendshape"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_json = out_dir / "sweep_results.json"
        with open(out_json, "w") as f:
            json.dump({
                "config": f"facs+blendshape_{nc}c_sweep",
                "runs": {k: v for k, v in all_recs.items() if f"_{nc}c" in k},
            }, f, indent=2)
        print(f"Saved: {out_json}")

    print("\n" + "=" * 80)
    print("Summary: FACS + Blendshape (80-dim), B1")
    print("=" * 80)
    print(f"  {'scheme':<6s}  {'arch':<22s}  {'facs_src':<10s}  {'macro_f1':>10s}")
    for nc in (3, 7):
        for arch in ("ours_cnn1d_facs", "fcnn"):
            for src in ("mediapipe", "faceapi"):
                v = all_recs[f"{arch}_{src}_{nc}c"]["test"]["macro_f1"]
                print(f"  {nc}c     {arch:<22s}  {src:<10s}  {v:>10.4f}")


if __name__ == "__main__":
    main()
