#!/usr/bin/env python3
"""Lengkapi checkpoint 7c untuk Grad-CAM 5.7.2 dengan backbone konsisten (ResNet18-TL).

7c gradcam_ckpts_7c yang sudah ada: cnn.pth (SCRATCH EmotionCNN), early_fusion_tl.pth
(CONCAT). Untuk perbandingan CNN_TL vs EF-concat vs EF-gated dengan backbone sama,
script ini menambah:
  - cnn_tl.pth       (EmotionCNNTransfer, ResNet18-TL, B1)
  - ef_gated_tl.pth  (EmotionEarlyFusionTransferGated, B1)

Mirror training loop scripts/retrain_for_gradcam.py (B1, no class weights, TL lr).
EF-concat 7c dipakai dari early_fusion_concat_tl.pth (hasil retrain_ef_concat_gradcam.py).

Usage:
  CUDA_VISIBLE_DEVICES=1 python scripts/retrain_gradcam_7c_extras.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from training.models import EmotionCNNTransfer, EmotionEarlyFusionTransferGated

DATA_DIR = PROJECT_ROOT / "data" / "dataset_frontonly_conf60"
CKPT_DIR = PROJECT_ROOT / "models" / "frontonly_conf60" / "gradcam_ckpts_7c"
NUM_CLASSES = 7
BATCH, EPOCHS, PATIENCE, LR_TL, SEED = 32, 60, 15, 5e-5, 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_split(split):
    img = np.load(DATA_DIR / f"X_{split}_images.npy").astype(np.float32)
    hm = np.load(DATA_DIR / f"X_{split}_heatmaps.npy").astype(np.float32)
    y = np.load(DATA_DIR / f"y_{split}.npy").astype(np.int64)
    return img, hm, y


def make_loader(arch, img, hm, y, shuffle=False):
    y_t = torch.from_numpy(y).long()
    if arch == "cnn":
        t = torch.from_numpy(img).permute(0, 3, 1, 2).float()
    else:  # early_fusion (4-channel)
        t_img = torch.from_numpy(img).permute(0, 3, 1, 2).float()
        t_hm = torch.from_numpy(hm).unsqueeze(1).float()
        t = torch.cat([t_img, t_hm], dim=1)
    ds = TensorDataset(t, y_t)
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle,
                      num_workers=2, pin_memory=True, drop_last=shuffle)


def train_model(model, tr, va, crit, lr, save_path):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5,
                                                     patience=8, min_lr=1e-7)
    best, stale = 0.0, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            loss = crit(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); yt, yp = [], []
        with torch.no_grad():
            for x, y in va:
                out = model(x.to(device))
                yt.append(y.numpy()); yp.append(out.argmax(1).cpu().numpy())
        vf1 = f1_score(np.concatenate(yt), np.concatenate(yp), average="macro", zero_division=0)
        sch.step(vf1)
        if vf1 > best:
            best, stale = vf1, 0
            torch.save(model.state_dict(), save_path)
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    return best


def evaluate(model, loader):
    model.eval(); yt, yp = [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device))
            yt.append(y.numpy()); yp.append(out.argmax(1).cpu().numpy())
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    return {"test_macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
            "test_weighted_f1": float(f1_score(yt, yp, average="weighted", zero_division=0)),
            "test_accuracy": float(accuracy_score(yt, yp))}


def run(name, model_fn, arch):
    ckpt = CKPT_DIR / f"{name}.pth"
    if ckpt.exists():
        print(f"[{name}] sudah ada, skip")
        return None
    print(f"\n[{name}] training ({arch}, B1)...")
    tr = make_loader(arch, *load_split("train"), shuffle=True)
    va = make_loader(arch, *load_split("val"))
    te = make_loader(arch, *load_split("test"))
    t0 = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    model = model_fn().to(device)
    best = train_model(model, tr, va, nn.CrossEntropyLoss(), LR_TL, ckpt)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    m = evaluate(model, te); m["val_macro_f1"] = float(best); m["elapsed_sec"] = round(time.time()-t0, 1)
    print(f"  [{name}] val={best:.4f} test_macro={m['test_macro_f1']:.4f} acc={m['test_accuracy']:.4f} ({m['elapsed_sec']}s)")
    rp = CKPT_DIR / "retrain_results.json"
    existing = json.load(open(rp)) if rp.exists() else {}
    existing[name] = m
    json.dump(existing, open(rp, "w"), indent=2)
    return m


def main():
    print(f"Device: {device}")
    # Hanya CNN_TL yang genuinely belum ada untuk 7c.
    # EF-concat -> gradcam_ckpts_7c/early_fusion_tl.pth (sudah ada)
    # EF-gated  -> 7class/Unified/fusion_early_gated_tl/checkpoints/b1.pt (sudah ada)
    run("cnn_tl", lambda: EmotionCNNTransfer(num_classes=NUM_CLASSES), "cnn")
    print("\nDone.")


if __name__ == "__main__":
    main()
