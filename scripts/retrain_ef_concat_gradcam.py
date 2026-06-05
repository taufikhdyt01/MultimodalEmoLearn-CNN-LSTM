#!/usr/bin/env python3
"""Retrain Early Fusion *concat* (TL) untuk Grad-CAM sub-bab 5.7.2.

Melengkapi checkpoint gated yang sudah ada (early_fusion_tl.pth) dengan varian
concat (EmotionEarlyFusionTransfer, 4-channel: RGB + heatmap landmark diperlakukan
setara). Training mirror scripts/retrain_for_gradcam.py (B1, no class weights, TL lr).

Output:
  3c -> models/frontonly_conf60/gradcam_ckpts/early_fusion_concat_tl.pth
  7c -> models/frontonly_conf60/gradcam_ckpts_7c/early_fusion_concat_tl.pth

Usage:
  CUDA_VISIBLE_DEVICES=1 python scripts/retrain_ef_concat_gradcam.py
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
from training.models import EmotionEarlyFusionTransfer

DATA_DIR = PROJECT_ROOT / "data" / "dataset_frontonly_conf60"
REMAP_3 = np.array([1, 0, 2, 2, 2, 2, 0], dtype=np.int64)

BATCH, EPOCHS, PATIENCE, LR_TL, SEED = 32, 60, 15, 5e-5, 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_split(split, num_classes):
    img = np.load(DATA_DIR / f"X_{split}_images.npy").astype(np.float32)
    hm = np.load(DATA_DIR / f"X_{split}_heatmaps.npy").astype(np.float32)
    y7 = np.load(DATA_DIR / f"y_{split}.npy").astype(np.int64)
    y = REMAP_3[y7].astype(np.int64) if num_classes == 3 else y7
    return img, hm, y


def make_loader(img, hm, y, shuffle=False):
    t_img = torch.from_numpy(img).permute(0, 3, 1, 2).float()
    t_hm = torch.from_numpy(hm).unsqueeze(1).float()
    t = torch.cat([t_img, t_hm], dim=1)  # (N,4,H,W)
    ds = TensorDataset(t, torch.from_numpy(y).long())
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


def run_scheme(num_classes, ckpt_dir):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / "early_fusion_concat_tl.pth"
    tag = f"{num_classes}c"
    if ckpt.exists():
        print(f"[{tag}] checkpoint sudah ada, skip: {ckpt}")
        return None
    print(f"\n[{tag}] training Early Fusion CONCAT (TL, B1)...")
    tr = make_loader(*load_split("train", num_classes), shuffle=True)
    va = make_loader(*load_split("val", num_classes))
    te = make_loader(*load_split("test", num_classes))
    t0 = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    model = EmotionEarlyFusionTransfer(num_classes=num_classes).to(device)
    crit = nn.CrossEntropyLoss()  # B1: no class weights
    best = train_model(model, tr, va, crit, LR_TL, ckpt)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    m = evaluate(model, te)
    m["val_macro_f1"] = float(best)
    m["elapsed_sec"] = round(time.time() - t0, 1)
    print(f"  [{tag}] val={best:.4f}  test_macro={m['test_macro_f1']:.4f}  "
          f"test_acc={m['test_accuracy']:.4f}  ({m['elapsed_sec']}s)")
    # simpan ke retrain_results.json
    rp = ckpt_dir / "retrain_results.json"
    existing = json.load(open(rp)) if rp.exists() else {}
    existing["early_fusion_concat_tl"] = m
    json.dump(existing, open(rp, "w"), indent=2)
    return m


def main():
    print(f"Device: {device}")
    run_scheme(3, PROJECT_ROOT / "models" / "frontonly_conf60" / "gradcam_ckpts")
    run_scheme(7, PROJECT_ROOT / "models" / "frontonly_conf60" / "gradcam_ckpts_7c")
    print("\nDone.")


if __name__ == "__main__":
    main()
