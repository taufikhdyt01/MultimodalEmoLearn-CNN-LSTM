#!/usr/bin/env python3
"""
Retrain 4 model Primer 7-class untuk keperluan Grad-CAM error analysis.
Checkpoint disimpan ke models/frontonly_conf60/gradcam_ckpts_7c/.

Model yang dilatih (best B scenario per family):
  1. CNN              — B1 (no class weights, val=0.2774)
  2. EarlyFusion_TL   — B1 (no class weights, belum pernah dilatih 7-class)
  3. Intermediate_TL  — B3 (class weights, val=0.2917)
  4. LateFusion       — B1 (no class weights, scratch CNN+FCNN, val=0.2887)

Usage:
  CUDA_VISIBLE_DEVICES=1 python scripts/retrain_for_gradcam_7c.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from training.models import (
    EmotionCNN,
    EmotionCNNTransfer,
    EmotionEarlyFusionTransfer,
    EmotionFCNN,
    IntermediateFusionTransfer,
)

DATA_DIR  = PROJECT_ROOT / "data" / "dataset_frontonly_conf60"
CKPT_DIR  = PROJECT_ROOT / "models" / "frontonly_conf60" / "gradcam_ckpts_7c"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 7
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

BATCH    = 32
EPOCHS   = 60
PATIENCE = 15
LR_TL    = 5e-5
LR_SCRATCH = 1e-4
SEED     = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Data ────────────────────────────────────────────────────────────────────
def load_split(split):
    img = np.load(DATA_DIR / f"X_{split}_images.npy").astype(np.float32)
    lm  = np.load(DATA_DIR / f"X_{split}_landmarks.npy").astype(np.float32)
    hm  = np.load(DATA_DIR / f"X_{split}_heatmaps.npy").astype(np.float32)
    y   = np.load(DATA_DIR / f"y_{split}.npy").astype(np.int64)
    return img, lm, hm, y


def class_weights(y):
    counts = np.bincount(y, minlength=NUM_CLASSES).astype(np.float32)
    w = counts.sum() / (NUM_CLASSES * np.maximum(counts, 1))
    return torch.FloatTensor(w / w.sum() * NUM_CLASSES).to(device)


def make_loader(arch, img, lm, hm, y, shuffle=False):
    y_t = torch.from_numpy(y).long()
    if arch == "cnn":
        t = torch.from_numpy(img).permute(0, 3, 1, 2).float()
        ds = TensorDataset(t, y_t)
    elif arch == "early_fusion":
        t_img = torch.from_numpy(img).permute(0, 3, 1, 2).float()
        t_hm  = torch.from_numpy(hm).unsqueeze(1).float()
        t = torch.cat([t_img, t_hm], dim=1)
        ds = TensorDataset(t, y_t)
    elif arch == "fcnn":
        ds = TensorDataset(torch.from_numpy(lm).float(), y_t)
    elif arch == "fusion":
        t_img = torch.from_numpy(img).permute(0, 3, 1, 2).float()
        t_lm  = torch.from_numpy(lm).float()
        ds = TensorDataset(t_img, t_lm, y_t)
    else:
        raise ValueError(arch)
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle,
                      num_workers=2, pin_memory=True, drop_last=shuffle)


# ─── Training ─────────────────────────────────────────────────────────────────
def train_model(model, arch, tr_loader, va_loader, criterion, lr, save_path):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=8, min_lr=1e-7)
    best_val, stale = 0.0, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch in tr_loader:
            *x, y = batch
            x = [xi.to(device) for xi in x]
            y = y.to(device)
            out = model(*x) if arch == "fusion" else model(x[0])
            loss = criterion(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        yt, yp = [], []
        with torch.no_grad():
            for batch in va_loader:
                *x, y = batch
                x = [xi.to(device) for xi in x]
                out = model(*x) if arch == "fusion" else model(x[0])
                yt.append(y.numpy()); yp.append(out.argmax(1).cpu().numpy())
        vf1 = f1_score(np.concatenate(yt), np.concatenate(yp),
                       average="macro", zero_division=0)
        sch.step(vf1)
        if vf1 > best_val:
            best_val, stale = vf1, 0
            torch.save(model.state_dict(), save_path)
        else:
            stale += 1
            if stale >= PATIENCE: break
    return best_val


def evaluate(model, arch, loader):
    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for batch in loader:
            *x, y = batch
            x = [xi.to(device) for xi in x]
            out = model(*x) if arch == "fusion" else model(x[0])
            yt.append(y.numpy()); yp.append(out.argmax(1).cpu().numpy())
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    return {
        "test_macro_f1":    float(f1_score(yt, yp, average="macro",    zero_division=0)),
        "test_weighted_f1": float(f1_score(yt, yp, average="weighted", zero_division=0)),
        "test_accuracy":    float(accuracy_score(yt, yp)),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    print(f"Device: {device}")

    img_tr, lm_tr, hm_tr, y_tr = load_split("train")
    img_va, lm_va, hm_va, y_va = load_split("val")
    img_te, lm_te, hm_te, y_te = load_split("test")
    print(f"Train={len(y_tr)}  Val={len(y_va)}  Test={len(y_te)}")
    print(f"Class dist train: {np.bincount(y_tr, minlength=NUM_CLASSES).tolist()}")

    results = {}

    # ── 1. CNN B1 (no class weights) ─────────────────────────────────────────
    name = "cnn"
    ckpt = CKPT_DIR / f"{name}.pth"
    if ckpt.exists():
        print(f"\n[{name}] checkpoint exists, skip")
    else:
        print(f"\n[{name}] training...")
        t0 = time.time()
        model = EmotionCNN(num_classes=NUM_CLASSES).to(device)
        crit  = nn.CrossEntropyLoss()  # B1: no class weights
        tr_l  = make_loader("cnn", img_tr, lm_tr, hm_tr, y_tr, shuffle=True)
        va_l  = make_loader("cnn", img_va, lm_va, hm_va, y_va)
        te_l  = make_loader("cnn", img_te, lm_te, hm_te, y_te)
        best_val = train_model(model, "cnn", tr_l, va_l, crit, LR_SCRATCH, ckpt)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        m = evaluate(model, "cnn", te_l)
        m["val_macro_f1"] = float(best_val)
        m["elapsed_sec"]  = round(time.time() - t0, 1)
        results[name] = m
        print(f"  val={best_val:.4f}  test_macro={m['test_macro_f1']:.4f}  ({m['elapsed_sec']}s)")

    # ── 2. EarlyFusion_TL B1 (no class weights) ──────────────────────────────
    name = "early_fusion_tl"
    ckpt = CKPT_DIR / f"{name}.pth"
    if ckpt.exists():
        print(f"\n[{name}] checkpoint exists, skip")
    else:
        print(f"\n[{name}] training...")
        t0 = time.time()
        model = EmotionEarlyFusionTransfer(num_classes=NUM_CLASSES).to(device)
        crit  = nn.CrossEntropyLoss()  # B1: no class weights
        tr_l  = make_loader("early_fusion", img_tr, lm_tr, hm_tr, y_tr, shuffle=True)
        va_l  = make_loader("early_fusion", img_va, lm_va, hm_va, y_va)
        te_l  = make_loader("early_fusion", img_te, lm_te, hm_te, y_te)
        best_val = train_model(model, "cnn", tr_l, va_l, crit, LR_TL, ckpt)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        m = evaluate(model, "cnn", te_l)
        m["val_macro_f1"] = float(best_val)
        m["elapsed_sec"]  = round(time.time() - t0, 1)
        results[name] = m
        print(f"  val={best_val:.4f}  test_macro={m['test_macro_f1']:.4f}  ({m['elapsed_sec']}s)")

    # ── 3. Intermediate_TL B3 (class weights) ────────────────────────────────
    name = "intermediate_tl"
    ckpt = CKPT_DIR / f"{name}.pth"
    if ckpt.exists():
        print(f"\n[{name}] checkpoint exists, skip")
    else:
        print(f"\n[{name}] training...")
        t0 = time.time()
        model = IntermediateFusionTransfer(num_classes=NUM_CLASSES).to(device)
        crit  = nn.CrossEntropyLoss(weight=class_weights(y_tr))
        tr_l  = make_loader("fusion", img_tr, lm_tr, hm_tr, y_tr, shuffle=True)
        va_l  = make_loader("fusion", img_va, lm_va, hm_va, y_va)
        te_l  = make_loader("fusion", img_te, lm_te, hm_te, y_te)
        best_val = train_model(model, "fusion", tr_l, va_l, crit, LR_TL, ckpt)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        m = evaluate(model, "fusion", te_l)
        m["val_macro_f1"] = float(best_val)
        m["elapsed_sec"]  = round(time.time() - t0, 1)
        results[name] = m
        print(f"  val={best_val:.4f}  test_macro={m['test_macro_f1']:.4f}  ({m['elapsed_sec']}s)")

    # ── 4. LateFusion scratch B1 (no class weights — CNN + FCNN) ─────────────
    name_cnn  = "late_fusion_cnn"
    name_fcnn = "late_fusion_fcnn"
    ckpt_cnn  = CKPT_DIR / f"{name_cnn}.pth"
    ckpt_fcnn = CKPT_DIR / f"{name_fcnn}.pth"
    if ckpt_cnn.exists() and ckpt_fcnn.exists():
        print(f"\n[late_fusion] checkpoints exist, skip")
    else:
        print(f"\n[late_fusion] training CNN branch...")
        t0 = time.time()
        crit = nn.CrossEntropyLoss()  # B1: no class weights

        cnn  = EmotionCNN(num_classes=NUM_CLASSES).to(device)
        tr_l = make_loader("cnn", img_tr, lm_tr, hm_tr, y_tr, shuffle=True)
        va_l = make_loader("cnn", img_va, lm_va, hm_va, y_va)
        val_cnn = train_model(cnn, "cnn", tr_l, va_l, crit, LR_SCRATCH, ckpt_cnn)
        print(f"  CNN val={val_cnn:.4f}")

        print(f"\n[late_fusion] training FCNN branch...")
        fcnn  = EmotionFCNN(num_classes=NUM_CLASSES).to(device)
        tr_lf = make_loader("fcnn", img_tr, lm_tr, hm_tr, y_tr, shuffle=True)
        va_lf = make_loader("fcnn", img_va, lm_va, hm_va, y_va)
        val_fcnn = train_model(fcnn, "fcnn", tr_lf, va_lf, crit, LR_SCRATCH, ckpt_fcnn)
        print(f"  FCNN val={val_fcnn:.4f}")

        cnn.load_state_dict(torch.load(ckpt_cnn, map_location=device, weights_only=True))
        fcnn.load_state_dict(torch.load(ckpt_fcnn, map_location=device, weights_only=True))
        cnn.eval(); fcnn.eval()

        def get_probs(model, arch, loader):
            probs = []
            with torch.no_grad():
                for batch in loader:
                    *x, _ = batch
                    x = [xi.to(device) for xi in x]
                    probs.append(F.softmax(model(x[0]), dim=1).cpu().numpy())
            return np.concatenate(probs)

        p_va_cnn  = get_probs(cnn,  "cnn",  make_loader("cnn",  img_va, lm_va, hm_va, y_va))
        p_va_fcnn = get_probs(fcnn, "fcnn", make_loader("fcnn", img_va, lm_va, hm_va, y_va))
        p_te_cnn  = get_probs(cnn,  "cnn",  make_loader("cnn",  img_te, lm_te, hm_te, y_te))
        p_te_fcnn = get_probs(fcnn, "fcnn", make_loader("fcnn", img_te, lm_te, hm_te, y_te))

        best_w, best_vf1 = 0.5, 0.0
        for w in np.arange(0.0, 1.05, 0.05):
            fused = w * p_va_cnn + (1 - w) * p_va_fcnn
            f1 = f1_score(y_va, fused.argmax(1), average="macro", zero_division=0)
            if f1 > best_vf1:
                best_vf1, best_w = f1, float(w)

        fused_te = best_w * p_te_cnn + (1 - best_w) * p_te_fcnn
        m = {
            "test_macro_f1":    float(f1_score(y_te, fused_te.argmax(1), average="macro",    zero_division=0)),
            "test_weighted_f1": float(f1_score(y_te, fused_te.argmax(1), average="weighted", zero_division=0)),
            "test_accuracy":    float(accuracy_score(y_te, fused_te.argmax(1))),
            "val_macro_f1":     float(best_vf1),
            "best_w_cnn":       best_w,
            "elapsed_sec":      round(time.time() - t0, 1),
        }
        results["late_fusion"] = m
        print(f"  ensemble val={best_vf1:.4f}  test_macro={m['test_macro_f1']:.4f}  best_w_cnn={best_w:.2f}")

    out_path = CKPT_DIR / "retrain_results.json"
    if results:
        existing = json.load(open(out_path)) if out_path.exists() else {}
        existing.update(results)
        json.dump(existing, open(out_path, "w"), indent=2)

    print("\n=== Checkpoints saved to:", CKPT_DIR)
    for f in sorted(CKPT_DIR.glob("*.pth")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
