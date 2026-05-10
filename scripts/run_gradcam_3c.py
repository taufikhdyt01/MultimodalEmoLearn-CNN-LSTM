#!/usr/bin/env python3
"""
Grad-CAM error analysis — 3-class Primer conf60.
4 model families: CNN_TL, EarlyFusion_TL, Intermediate_TL, LateFusion_TL.

Usage:
  CUDA_VISIBLE_DEVICES=1 python scripts/run_gradcam_3c.py
"""
import sys
import json
import functools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from sklearn.metrics import f1_score, confusion_matrix, classification_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from training.models import (
    EmotionCNNTransfer,
    EmotionEarlyFusionTransfer,
    EmotionFCNN,
    IntermediateFusionTransfer,
)

DATA_DIR  = PROJECT_ROOT / "data" / "dataset_frontonly_conf60"
CKPT_DIR  = PROJECT_ROOT / "models" / "frontonly_conf60" / "gradcam_ckpts"
OUT_DIR   = PROJECT_ROOT / "outputs" / "gradcam"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 3
REMAP_3     = np.array([1, 0, 2, 2, 2, 2, 0], dtype=np.int64)
CLASS_NAMES = ["positive", "neutral", "negative"]
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RNG         = np.random.RandomState(42)
BATCH       = 64

print(f"Device: {DEVICE}")


# ─── Data ─────────────────────────────────────────────────────────────────────
def load_split(split):
    img = np.load(DATA_DIR / f"X_{split}_images.npy").astype(np.float32)
    lm  = np.load(DATA_DIR / f"X_{split}_landmarks.npy").astype(np.float32)
    hm  = np.load(DATA_DIR / f"X_{split}_heatmaps.npy").astype(np.float32)
    y7  = np.load(DATA_DIR / f"y_{split}.npy")
    y   = REMAP_3[y7].astype(np.int64)
    return img, lm, hm, y


img_tr, lm_tr, hm_tr, y_tr = load_split("train")
img_va, lm_va, hm_va, y_va = load_split("val")
img_te, lm_te, hm_te, y_te = load_split("test")
print(f"Train={len(y_tr)}  Val={len(y_va)}  Test={len(y_te)}")
print(f"Test class dist: {np.bincount(y_te, minlength=NUM_CLASSES).tolist()}")


# ─── Tensor helpers ────────────────────────────────────────────────────────────
def img_t(arr, idx):
    return torch.from_numpy(arr[idx]).permute(0, 3, 1, 2).float().to(DEVICE)

def ef_t(img_arr, hm_arr, idx):
    t_img = torch.from_numpy(img_arr[idx]).permute(0, 3, 1, 2).float()
    t_hm  = torch.from_numpy(hm_arr[idx]).unsqueeze(1).float()
    return torch.cat([t_img, t_hm], dim=1).to(DEVICE)

def lm_t(arr, idx):
    return torch.from_numpy(arr[idx]).float().to(DEVICE)


# ─── Load checkpoint ──────────────────────────────────────────────────────────
def load_ckpt(model, path):
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


# ─── Predictions ──────────────────────────────────────────────────────────────
def get_preds(model, arch, img_arr, lm_arr, hm_arr):
    model.eval()
    n = len(img_arr)
    all_probs = []
    with torch.no_grad():
        for i in range(0, n, BATCH):
            idx = list(range(i, min(i + BATCH, n)))
            if arch == "cnn":
                out = model(img_t(img_arr, idx))
            elif arch == "early_fusion":
                out = model(ef_t(img_arr, hm_arr, idx))
            elif arch == "fcnn":
                out = model(lm_t(lm_arr, idx))
            elif arch == "fusion":
                out = model(img_t(img_arr, idx), lm_t(lm_arr, idx))
            all_probs.append(F.softmax(out, dim=1).cpu().numpy())
    probs = np.concatenate(all_probs)
    return probs.argmax(1), probs


# ─── Load models ──────────────────────────────────────────────────────────────
print("\nLoading models...")
cnn_tl = load_ckpt(EmotionCNNTransfer(num_classes=NUM_CLASSES).to(DEVICE),
                   CKPT_DIR / "cnn_tl.pth")
ef_tl  = load_ckpt(EmotionEarlyFusionTransfer(num_classes=NUM_CLASSES).to(DEVICE),
                   CKPT_DIR / "early_fusion_tl.pth")
itl    = load_ckpt(IntermediateFusionTransfer(num_classes=NUM_CLASSES).to(DEVICE),
                   CKPT_DIR / "intermediate_tl.pth")
lf_cnn = load_ckpt(EmotionCNNTransfer(num_classes=NUM_CLASSES).to(DEVICE),
                   CKPT_DIR / "late_fusion_tl_cnn.pth")
lf_fcnn= load_ckpt(EmotionFCNN(num_classes=NUM_CLASSES).to(DEVICE),
                   CKPT_DIR / "late_fusion_tl_fcnn.pth")
print("All checkpoints loaded.")


# ─── LateFusion best_w (grid search on val) ───────────────────────────────────
results_path = CKPT_DIR / "retrain_results.json"
if results_path.exists():
    best_w = json.load(open(results_path)).get("late_fusion_tl", {}).get("best_w_cnn", None)
else:
    best_w = None

if best_w is None:
    print("Computing LateFusion best_w from val set...")
    _, p_va_cnn  = get_preds(lf_cnn,  "cnn",  img_va, lm_va, hm_va)
    _, p_va_fcnn = get_preds(lf_fcnn, "fcnn", img_va, lm_va, hm_va)
    best_w, best_f1 = 0.5, 0.0
    for w in np.arange(0.0, 1.05, 0.05):
        fused = w * p_va_cnn + (1 - w) * p_va_fcnn
        f1 = f1_score(y_va, fused.argmax(1), average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_w = f1, float(w)
    print(f"  best_w_cnn={best_w:.2f}  val_f1={best_f1:.4f}")

print(f"LateFusion best_w_cnn = {best_w:.2f}")


# ─── Compute all predictions ───────────────────────────────────────────────────
print("\nComputing test predictions...")
_, p_cnn  = get_preds(cnn_tl, "cnn",          img_te, lm_te, hm_te)
_, p_ef   = get_preds(ef_tl,  "early_fusion",  img_te, lm_te, hm_te)
_, p_itl  = get_preds(itl,    "fusion",        img_te, lm_te, hm_te)
_, p_lfc  = get_preds(lf_cnn,  "cnn",          img_te, lm_te, hm_te)
_, p_lffc = get_preds(lf_fcnn, "fcnn",         img_te, lm_te, hm_te)
p_lf = best_w * p_lfc + (1 - best_w) * p_lffc

MODEL_NAMES   = ["CNN_TL", "EarlyFusion_TL", "Intermediate_TL", "LateFusion_TL"]
MODEL_PROBS   = {"CNN_TL": p_cnn, "EarlyFusion_TL": p_ef,
                 "Intermediate_TL": p_itl, "LateFusion_TL": p_lf}
MODEL_PREDS   = {k: v.argmax(1) for k, v in MODEL_PROBS.items()}
MODEL_OBJECTS = {"CNN_TL": cnn_tl, "EarlyFusion_TL": ef_tl,
                 "Intermediate_TL": itl, "LateFusion_TL": lf_cnn}
MODEL_ARCH    = {"CNN_TL": "cnn", "EarlyFusion_TL": "early_fusion",
                 "Intermediate_TL": "fusion", "LateFusion_TL": "cnn"}

print("\n=== Test Metrics ===")
for name in MODEL_NAMES:
    f1 = f1_score(y_te, MODEL_PREDS[name], average="macro", zero_division=0)
    fw = f1_score(y_te, MODEL_PREDS[name], average="weighted", zero_division=0)
    print(f"  {name:20s}: macro={f1:.4f}  weighted={fw:.4f}")


# ─── Grad-CAM helpers ─────────────────────────────────────────────────────────
def get_target_layer(model_obj, model_name):
    if model_name in ("CNN_TL", "LateFusion_TL", "EarlyFusion_TL"):
        return model_obj.features[-2][-1]
    elif model_name == "Intermediate_TL":
        return model_obj.image_features[-2][-1]
    raise ValueError(model_name)


class FusionWrapper(torch.nn.Module):
    def __init__(self, m, lm):
        super().__init__()
        self.m  = m
        self.lm = lm
    def forward(self, x):
        return self.m(x, self.lm)


def run_gradcam(model_obj, model_name, sample_idx):
    """Return (rgb float32 H×W×3, overlay float32 H×W×3)."""
    rgb = img_te[sample_idx].copy()
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    rgb = np.clip(rgb, 0, 1)

    if model_name == "Intermediate_TL":
        lm_fixed = lm_t(lm_te, [sample_idx])
        wrapper  = FusionWrapper(model_obj, lm_fixed)
        target_layer = model_obj.image_features[-2][-1]
        cam = GradCAM(model=wrapper, target_layers=[target_layer])
        inp = img_t(img_te, [sample_idx])
    else:
        target_layer = get_target_layer(model_obj, model_name)
        cam = GradCAM(model=model_obj, target_layers=[target_layer])
        if model_name == "EarlyFusion_TL":
            inp = ef_t(img_te, hm_te, [sample_idx])
        else:
            inp = img_t(img_te, [sample_idx])

    grayscale = cam(input_tensor=inp, targets=None)[0]
    overlay   = show_cam_on_image(rgb, grayscale, use_rgb=True)
    return rgb, overlay


# ─── Sample helpers ───────────────────────────────────────────────────────────
def sample_idx(preds, true_cls, pred_cls, n=1):
    mask = (y_te == true_cls) & (preds == pred_cls)
    idx  = np.where(mask)[0]
    if len(idx) == 0:
        return []
    return RNG.choice(idx, size=min(n, len(idx)), replace=False).tolist()


# ─── Plot 1: per model — benar vs salah per kelas ─────────────────────────────
print("\n=== Generating per-model Grad-CAM plots ===")
for model_name in MODEL_NAMES:
    model_obj = MODEL_OBJECTS[model_name]
    preds     = MODEL_PREDS[model_name]
    probs     = MODEL_PROBS[model_name]

    fig, axes = plt.subplots(NUM_CLASSES * 2, 4, figsize=(14, NUM_CLASSES * 4))
    fig.suptitle(f"{model_name} — Grad-CAM Error Analysis (3-class Primer)",
                 fontsize=13, fontweight="bold")
    for ax, lbl in zip(axes[0], ["Original", "Grad-CAM (benar)", "Original", "Grad-CAM (salah)"]):
        ax.set_title(lbl, fontsize=9, fontweight="bold")

    for cls in range(NUM_CLASSES):
        rb = cls * 2
        correct_idx = sample_idx(preds, cls, cls, n=1)
        wrong_idx, wrong_pred = [], None
        for pc in range(NUM_CLASSES):
            if pc == cls: continue
            w = sample_idx(preds, cls, pc, n=1)
            if w:
                wrong_idx, wrong_pred = w, pc
                break

        # Row A: correct
        axes[rb][0].set_ylabel(f"True: {CLASS_NAMES[cls]}\n(benar)", fontsize=8)
        if correct_idx:
            i = correct_idx[0]
            rgb, ov = run_gradcam(model_obj, model_name, i)
            axes[rb][0].imshow(rgb); axes[rb][0].axis("off")
            axes[rb][0].set_title(f"conf={probs[i][cls]:.2f}", fontsize=7)
            axes[rb][1].imshow(ov);  axes[rb][1].axis("off")
            axes[rb][1].set_title(f"→ {CLASS_NAMES[cls]} ✓", fontsize=7, color="green")
        else:
            for a in axes[rb][:2]: a.axis("off")
        axes[rb][2].axis("off"); axes[rb][3].axis("off")

        # Row B: wrong
        axes[rb+1][0].set_ylabel(f"True: {CLASS_NAMES[cls]}\n(salah)", fontsize=8)
        if wrong_idx:
            i = wrong_idx[0]
            rgb, ov = run_gradcam(model_obj, model_name, i)
            axes[rb+1][0].imshow(rgb); axes[rb+1][0].axis("off")
            axes[rb+1][0].set_title(f"conf={probs[i][wrong_pred]:.2f}", fontsize=7)
            axes[rb+1][1].imshow(ov);  axes[rb+1][1].axis("off")
            axes[rb+1][1].set_title(f"→ {CLASS_NAMES[wrong_pred]} ✗", fontsize=7, color="red")
        else:
            for a in axes[rb+1][:2]: a.axis("off")
        axes[rb+1][2].axis("off"); axes[rb+1][3].axis("off")

    plt.tight_layout()
    save_path = OUT_DIR / f"gradcam_{model_name.lower()}_error.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.name}")


# ─── Plot 2: cross-model comparison (sampel salah di SEMUA model) ─────────────
print("\n=== Generating cross-model comparison plots ===")
wrong_all = np.ones(len(y_te), dtype=bool)
for name in MODEL_NAMES:
    wrong_all &= (MODEL_PREDS[name] != y_te)
print(f"Sampel salah di semua 4 model: {wrong_all.sum()}")

for cls in range(NUM_CLASSES):
    cls_mask = (y_te == cls) & wrong_all
    idxs = np.where(cls_mask)[0]
    if len(idxs) == 0:
        print(f"  {CLASS_NAMES[cls]}: tidak ada sampel salah di semua model")
        continue
    sel = RNG.choice(idxs, size=min(3, len(idxs)), replace=False)

    for sample_idx_ in sel:
        fig, axes = plt.subplots(1, 5, figsize=(18, 3))
        fig.suptitle(
            f"True: {CLASS_NAMES[cls]} | Sample #{sample_idx_} — semua model salah",
            fontsize=11, fontweight="bold"
        )
        rgb = np.clip(img_te[sample_idx_].copy(), 0, 1)
        if rgb.max() > 1.0: rgb = rgb / 255.0
        axes[0].imshow(rgb); axes[0].axis("off"); axes[0].set_title("Original", fontsize=9)

        for col, model_name in enumerate(MODEL_NAMES, start=1):
            pc   = MODEL_PREDS[model_name][sample_idx_]
            conf = MODEL_PROBS[model_name][sample_idx_][pc]
            _, ov = run_gradcam(MODEL_OBJECTS[model_name], model_name, sample_idx_)
            axes[col].imshow(ov); axes[col].axis("off")
            axes[col].set_title(f"{model_name}\n→ {CLASS_NAMES[pc]} ({conf:.2f})",
                                fontsize=8, color="red")

        plt.tight_layout()
        save_path = OUT_DIR / f"gradcam_comparison_cls{cls}_sample{sample_idx_}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path.name}")


# ─── Ringkasan metrics ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("RINGKASAN METRICS — 3-class Primer test set")
print("="*60)
for name in MODEL_NAMES:
    preds = MODEL_PREDS[name]
    f1m   = f1_score(y_te, preds, average="macro",    zero_division=0)
    f1w   = f1_score(y_te, preds, average="weighted", zero_division=0)
    print(f"\n{name}  macro_f1={f1m:.4f}  weighted_f1={f1w:.4f}")
    print(classification_report(y_te, preds, target_names=CLASS_NAMES, zero_division=0))

print(f"\nSemua output disimpan di: {OUT_DIR}")
print("Done.")
