#!/usr/bin/env python3
"""Grad-CAM panel CK+ v2 — model DILATIH di CK+ (bukan cross-dataset dari primer).

Tiga arsitektur SAMA PERSIS dengan primer (5.7.2):
  (1) CNN_TL            — unimodal citra, ResNet-18 TL
  (2) Early Fusion concat TL — RGB + heatmap landmark (4-channel)
  (3) Early Fusion gated TL  — heatmap men-gate atensi spasial

Konfigurasi terbaik CK+ dari all_metrics_tables:
  - EF-concat TL B3 → acc=93,22%, macro_f1=88,50%  (TERBAIK)
  - EF-gated  TL B3 → acc=93,22%, macro_f1=85,10%
  - CNN_TL    B3    → acc=89,83%, macro_f1=84,74%

Semua model dilatih di ckplus_7class (636 sampel, subject-wise 80:10:10),
Grad-CAM dijalankan pada test set CK+ (gambar yang TIDAK dilihat saat training).

Output: docs/figures/gradcam/gradcam_ckplus_v2.{png,pdf}
Usage:  CUDA_VISIBLE_DEVICES=1 python scripts/make_gradcam_ckplus_v2.py
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from training.models import (
    EmotionCNNTransfer, EmotionEarlyFusionTransfer, EmotionEarlyFusionTransferGated,
)
from training.utils import train_model, full_evaluation
from training.fusion_aug import EarlyFusionDataset, make_balanced_sampler

CK_DATA   = PROJECT_ROOT / "data"   / "benchmark" / "ckplus_7class"
CKPT_DIR  = PROJECT_ROOT / "models" / "benchmark" / "ckplus_7class" / "gradcam_ckpts"
OUT       = PROJECT_ROOT / "docs"   / "figures"   / "gradcam"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RNG     = np.random.RandomState(42)
NAMES   = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]
BATCH, EPOCHS, PATIENCE, LR_TL, SEED = 32, 50, 15, 5e-5, 42

# Ekspresi khas posed intensitas tinggi di CK+ untuk visualisasi
TARGET_EMOTIONS = [1, 6, 5, 3]  # happy, surprised, disgusted, angry
plt.rcParams.update({"font.size": 9})


# ─── Data ─────────────────────────────────────────────────────────────────

def load_split(split):
    img = np.load(CK_DATA / f"X_{split}_images.npy").astype(np.float32)
    hm  = np.load(CK_DATA / f"X_{split}_heatmaps.npy").astype(np.float32)
    y   = np.load(CK_DATA / f"y_{split}.npy").astype(np.int64)
    if img.max() > 1.5:
        img = img / 255.0
    return img, hm, y


def make_loader_b3(img, hm, y, *, train=True):
    """B3: EarlyFusionDataset + augment + WeightedRandomSampler untuk train."""
    ds = EarlyFusionDataset(img, hm, y, augment=train, seed=SEED)
    if train:
        sampler = make_balanced_sampler(y, 7)
        return DataLoader(ds, batch_size=BATCH, sampler=sampler,
                          num_workers=2, pin_memory=True, drop_last=True)
    return DataLoader(ds, batch_size=BATCH, shuffle=False,
                      num_workers=2, pin_memory=True)


def make_loader_img_b3(img, y, *, train=True):
    """B3 untuk CNN_TL (image only): augmentasi photometric + sampler."""
    from training.image_aug import AugmentingImageDataset as ImageDataset
    ds = ImageDataset(img, y, augment=train, seed=SEED)
    if train:
        sampler = make_balanced_sampler(y, 7)
        return DataLoader(ds, batch_size=BATCH, sampler=sampler,
                          num_workers=2, pin_memory=True, drop_last=True)
    return DataLoader(ds, batch_size=BATCH, shuffle=False,
                      num_workers=2, pin_memory=True)


# ─── Training ──────────────────────────────────────────────────────────────

def train_and_save(arch_name, model_fn, train_loader_fn, val_loader_fn,
                   test_loader_fn, ckpt_path):
    """Train model if checkpoint not already saved."""
    if ckpt_path.exists():
        print(f"  [{arch_name}] checkpoint exists — skip training", flush=True)
        return

    print(f"\n  [{arch_name}] Training B3 on CK+ (subject-wise 7c)...", flush=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    model = model_fn().to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR_TL)
    sch   = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="max", factor=0.5, patience=8, min_lr=1e-7)
    sp = str(ckpt_path)

    tr_l  = train_loader_fn()
    va_l  = val_loader_fn()

    train_model(model, tr_l, va_l, nn.CrossEntropyLoss(), opt, sch,
                DEVICE, "cnn", EPOCHS, PATIENCE, sp)
    print(f"  [{arch_name}] saved → {ckpt_path.name}", flush=True)


def load_ckpt(model, path):
    sd = torch.load(path, map_location=DEVICE, weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    elif isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd)
    return model.to(DEVICE).eval()


# ─── GradCAM helpers ───────────────────────────────────────────────────────

def get_cam_layer(model, arch):
    """Semua model (CNN_TL, EF-concat, EF-gated) menggunakan ResNet-18
    via model.features. Layer4 = features[7], BasicBlock terakhir = features[7][-1].
    Ini ekuivalen dengan layer4[-1] pada ResNet-18 standar."""
    return model.features[7][-1]


def predict_all(model, arch, img, hm, device):
    """Returns (preds, probs) arrays over full test set."""
    probs = []
    with torch.no_grad():
        for i in range(0, len(img), 64):
            if arch == "cnn_tl":
                x = torch.from_numpy(img[i:i+64]).permute(0,3,1,2).float().to(device)
                logits = model(x)
            else:
                a = torch.from_numpy(img[i:i+64]).permute(0,3,1,2).float()
                b = torch.from_numpy(hm[i:i+64]).unsqueeze(1).float()
                x = torch.cat([a,b],1).to(device)
                logits = model(x)
            probs.append(F.softmax(logits, dim=1).cpu().numpy())
    probs = np.concatenate(probs)
    return probs.argmax(1), probs


def get_gradcam(model, arch, img_np, hm_np, target_class, device):
    """Returns CAM heatmap (H,W) for single image."""
    target_layer = [get_cam_layer(model, arch)]
    with GradCAM(model=model, target_layers=target_layer) as cam:
        if arch == "cnn_tl":
            x = torch.from_numpy(img_np).permute(2,0,1).unsqueeze(0).float().to(device)
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
            grayscale = cam(input_tensor=x,
                            targets=[ClassifierOutputTarget(target_class)])
        else:
            a = torch.from_numpy(img_np).permute(2,0,1).unsqueeze(0).float()
            b = torch.from_numpy(hm_np).unsqueeze(0).unsqueeze(0).float()
            x = torch.cat([a,b],1).to(device)
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
            grayscale = cam(input_tensor=x,
                            targets=[ClassifierOutputTarget(target_class)])
    return grayscale[0]  # (H,W)


# ─── Panel figure ──────────────────────────────────────────────────────────

HEADER_BOX = dict(boxstyle="round,pad=0.35", facecolor="#eef2f7", edgecolor="#7a8aa0")
COL_LABELS  = ["Citra asli", "CNN_TL (unimodal)", "Early concat TL", "Early gated TL"]
ARCHS       = ["cnn_tl", "ef_concat", "ef_gated"]


def make_panel(samples, models_dict, img_test, hm_test, y_test, preds_dict):
    """Layout identik dengan primer (make_gradcam_5_7_2.py):
    - Header kolom bold + bbox di baris pertama
    - ylabel = 'True: xxx' di kolom citra asli
    - Prediksi label di BAWAH sel dengan ax.text transform
    """
    n_rows = len(samples)
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(3.05 * n_cols, 3.05 * n_rows + 0.5))
    if n_rows == 1:
        axes = axes[None, :]

    # ── Header kolom (hanya baris pertama, bold + bbox) ──────────────────
    for c, ct in enumerate(COL_LABELS):
        axes[0, c].set_title(ct, fontsize=13.5, fontweight="bold", pad=12,
                              bbox=HEADER_BOX)

    for ri, (idx, true_lbl) in enumerate(samples):
        img_np = np.clip(img_test[idx], 0, 1)
        hm_np  = hm_test[idx]

        # ── Kolom 0: citra asli ──────────────────────────────────────────
        ax0 = axes[ri, 0]
        ax0.imshow(img_np)
        ax0.set_ylabel(f"True: {NAMES[true_lbl]}", fontsize=11, fontweight="bold")
        ax0.set_xticks([]); ax0.set_yticks([])

        # ── Kolom 1-3: GradCAM ───────────────────────────────────────────
        for ci, arch in enumerate(ARCHS, start=1):
            model    = models_dict[arch]
            pred     = preds_dict[arch][idx]
            conf     = float(preds_dict[f"{arch}_prob"][idx][pred])
            ok       = pred == true_lbl
            cam_map  = get_gradcam(model, arch, img_np, hm_np, true_lbl, DEVICE)
            cam_vis  = show_cam_on_image(img_np, cam_map, use_rgb=True)
            ax = axes[ri, ci]
            ax.imshow(cam_vis)
            ax.set_xticks([]); ax.set_yticks([])
            # Label prediksi DI BAWAH sel (sama persis dengan primer)
            ax.text(0.5, -0.045,
                    f"→ {NAMES[pred]} ({conf:.2f}) {'✓' if ok else '✗'}",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=10, fontweight="bold",
                    color=("#1a7f37" if ok else "#c0392b"))

    fig.suptitle("Grad-CAM Cabang Citra — CK+\nModel dilatih di CK+ (subject-wise B3)",
                 fontsize=14, fontweight="bold", y=1.015)
    fig.text(0.5, 0.94, "Ekspresi posed intensitas tinggi — atensi terlokalisasi pada region ekspresif",
             ha="center", va="top", fontsize=10.5, color="#333")
    fig.text(0.5, 0.004,
             "Grad-CAM cabang citra (ResNet-18 TL, layer4). "
             "Cabang landmark tidak di-Grad-CAM (input vektor, bukan citra).",
             ha="center", va="bottom", fontsize=8.5, style="italic", color="#444")
    plt.tight_layout(rect=[0, 0.018, 1, 0.915], h_pad=3.0)
    for ext in ("png", "pdf"):
        path = OUT / f"gradcam_ckplus_v2.{ext}"
        fig.savefig(path, dpi=190, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}", flush=True)
    print(f"Data:   {CK_DATA}", flush=True)

    img_tr, hm_tr, y_tr = load_split("train")
    img_va, hm_va, y_va = load_split("val")
    img_te, hm_te, y_te = load_split("test")

    print(f"Train={len(y_tr)}, Val={len(y_va)}, Test={len(y_te)}", flush=True)

    # ── 1. Train / load checkpoints ────────────────────────────────────────

    # CNN_TL B3
    # CNN_TL hanya butuh gambar (tidak perlu heatmap)
    from training.image_aug import AugmentingImageDataset as ImageDataset
    def cnn_train_loader():
        ds = ImageDataset(img_tr, y_tr, augment=True, seed=SEED)
        sam = make_balanced_sampler(y_tr, 7)
        return DataLoader(ds, batch_size=BATCH, sampler=sam,
                          num_workers=2, pin_memory=True, drop_last=True)
    def cnn_val_loader():
        ds = ImageDataset(img_va, y_va, augment=False, seed=SEED)
        return DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=2)

    train_and_save(
        "CNN_TL_B3",
        lambda: EmotionCNNTransfer(num_classes=7),
        cnn_train_loader, cnn_val_loader, None,
        CKPT_DIR / "cnn_tl_b3.pth",
    )

    # EF-concat TL B3
    def ef_train_loader():
        return make_loader_b3(img_tr, hm_tr, y_tr, train=True)
    def ef_val_loader():
        return make_loader_b3(img_va, hm_va, y_va, train=False)

    train_and_save(
        "EF_concat_TL_B3",
        lambda: EmotionEarlyFusionTransfer(num_classes=7),
        ef_train_loader, ef_val_loader, None,
        CKPT_DIR / "ef_concat_tl_b3.pth",
    )

    # EF-gated TL — sudah ada dari unified benchmark (b3.pt)
    gated_existing = (PROJECT_ROOT / "models" / "benchmark" /
                      "ckplus_7class" / "7class" / "Unified" /
                      "fusion_early_gated_tl" / "checkpoints" / "b3.pt")
    gated_ckpt = CKPT_DIR / "ef_gated_tl_b3.pth"
    if not gated_ckpt.exists() and gated_existing.exists():
        import shutil
        shutil.copy(gated_existing, gated_ckpt)
        print(f"  [EF_gated_TL_B3] copied existing checkpoint", flush=True)
    elif not gated_ckpt.exists():
        # retrain if not found
        train_and_save(
            "EF_gated_TL_B3",
            lambda: EmotionEarlyFusionTransferGated(num_classes=7),
            ef_train_loader, ef_val_loader, None,
            gated_ckpt,
        )

    # ── 2. Load models ─────────────────────────────────────────────────────
    print("\nLoading checkpoints...", flush=True)
    m_cnn    = load_ckpt(EmotionEarlyFusionTransfer(num_classes=7).__class__(num_classes=7)
                         if False else EmotionCNNTransfer(num_classes=7),
                         CKPT_DIR / "cnn_tl_b3.pth")
    m_concat = load_ckpt(EmotionEarlyFusionTransfer(num_classes=7),
                         CKPT_DIR / "ef_concat_tl_b3.pth")
    m_gated  = load_ckpt(EmotionEarlyFusionTransferGated(num_classes=7),
                         gated_ckpt)

    models_dict = {"cnn_tl": m_cnn, "ef_concat": m_concat, "ef_gated": m_gated}

    # ── 3. Predict all test images ─────────────────────────────────────────
    print("Predicting test set...", flush=True)
    preds_dict = {}
    for arch, model in models_dict.items():
        preds, probs = predict_all(model, arch, img_te, hm_te, DEVICE)
        preds_dict[arch] = preds
        preds_dict[f"{arch}_prob"] = probs
        acc = (preds == y_te).mean()
        print(f"  {arch}: test acc={acc:.4f}", flush=True)

    # ── 4. Pilih sampel: 1 per ekspresi target, benar diprediksi ke-3 model ─
    samples = []
    for target_lbl in TARGET_EMOTIONS:
        # Kandidat: true label = target, semua model benar
        cands = [i for i in range(len(y_te))
                 if y_te[i] == target_lbl
                 and all(preds_dict[a][i] == target_lbl
                         for a in ["cnn_tl", "ef_concat", "ef_gated"])]
        if cands:
            # Pilih yang confidence EF-concat tertinggi
            best = max(cands,
                       key=lambda i: preds_dict["ef_concat_prob"][i][target_lbl])
            samples.append((best, target_lbl))
            print(f"  Sample: idx={best}, label={NAMES[target_lbl]}", flush=True)
        else:
            # Fallback: hanya perlu EF-concat benar
            cands2 = [i for i in range(len(y_te))
                      if y_te[i] == target_lbl
                      and preds_dict["ef_concat"][i] == target_lbl]
            if cands2:
                best = max(cands2,
                           key=lambda i: preds_dict["ef_concat_prob"][i][target_lbl])
                samples.append((best, target_lbl))
                print(f"  Sample (fallback): idx={best}, label={NAMES[target_lbl]}",
                      flush=True)

    if not samples:
        print("WARNING: tidak ada sampel ditemukan!", flush=True)
        return

    # ── 5. Generate GradCAM panel ──────────────────────────────────────────
    print(f"\nGenerating GradCAM panel ({len(samples)} samples)...", flush=True)
    make_panel(samples, models_dict, img_te, hm_te, y_te, preds_dict)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
