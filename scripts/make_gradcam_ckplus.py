#!/usr/bin/env python3
"""Grad-CAM panel kontras CK+ untuk tesis 5.7.2.

CK+ = ekspresi posed intensitas tinggi (lab-controlled). Dipakai sebagai pembanding
"ideal": atensi Grad-CAM cabang citra jauh lebih tajam/terbaca pada region ekspresif
dibanding kondisi natural webcam (primer). Model = CNN_TL 7c terlatih di primer
(cross-dataset), Grad-CAM di layer4.

Output: docs/figures/gradcam/gradcam_5_7_2_ckplus.{png,pdf}

Usage: CUDA_VISIBLE_DEVICES=1 python scripts/make_gradcam_ckplus.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from training.models import EmotionCNNTransfer

CK_DATA = PROJECT_ROOT / "data" / "benchmark" / "ckplus_7class"
CK7 = PROJECT_ROOT / "models" / "frontonly_conf60" / "gradcam_ckpts_7c"
OUT = PROJECT_ROOT / "docs" / "figures" / "gradcam"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RNG = np.random.RandomState(42)
NAMES = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]
# kelas intensitas tinggi yang paling terbaca atensinya
HIGH_INTENSITY = [1, 6, 5, 3]  # happy, surprised, disgusted, angry
plt.rcParams.update({"font.size": 9})


def main():
    print(f"Device: {DEVICE}")
    img = np.load(CK_DATA / "X_test_images.npy").astype(np.float32)
    y = np.load(CK_DATA / "y_test.npy").astype(np.int64)
    if img.max() > 1.5:
        img = img / 255.0
    model = EmotionCNNTransfer(num_classes=7)
    sd = torch.load(CK7 / "cnn_tl.pth", map_location=DEVICE, weights_only=False)
    model.load_state_dict(sd); model = model.to(DEVICE).eval()

    # prediksi
    probs = []
    with torch.no_grad():
        for i in range(0, len(img), 64):
            x = torch.from_numpy(img[i:i+64]).permute(0, 3, 1, 2).float().to(DEVICE)
            probs.append(F.softmax(model(x), dim=1).cpu().numpy())
    probs = np.concatenate(probs); preds = probs.argmax(1)

    # Pilih 3 sampel berdasarkan TRUE label ekspresi dramatis (happy/surprised/disgusted).
    # Tujuan CK+ = menunjukkan atensi tajam pada ekspresi posed — kebenaran prediksi
    # cross-dataset bukan fokus; pred->true ditampilkan apa adanya. Ambil sampel dgn
    # max-confidence tertinggi (peta atensi paling 'peaked'/terbaca).
    DRAMATIC = [1, 6, 5, 3]  # happy, surprised, disgusted, angry
    chosen = []
    for cls in DRAMATIC:
        cand = np.where(y == cls)[0]
        if len(cand):
            best = cand[np.argmax(probs[cand].max(1))]
            chosen.append(int(best))
        if len(chosen) >= 3:
            break
    print("Sampel terpilih:", [(i, NAMES[y[i]], NAMES[preds[i]]) for i in chosen])

    target = model.features[-2][-1]
    n = len(chosen)
    fig, axes = plt.subplots(n, 2, figsize=(6.2, 3.0 * n))
    if n == 1:
        axes = axes[None, :]
    axes[0, 0].set_title("Citra asli (CK+)", fontsize=11, fontweight="bold", pad=8)
    axes[0, 1].set_title("Grad-CAM CNN_TL", fontsize=11, fontweight="bold", pad=8)
    for r, idx in enumerate(chosen):
        rgb = np.clip(img[idx], 0, 1).astype(np.float32)
        cam = GradCAM(model=model, target_layers=[target])
        x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        gray = cam(input_tensor=x, targets=None)[0]
        ov = show_cam_on_image(rgb, gray, use_rgb=True)
        pred = int(preds[idx]); conf = float(probs[idx, pred])
        axes[r, 0].imshow(rgb); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        axes[r, 0].set_ylabel(f"True: {NAMES[y[idx]]}", fontsize=10, fontweight="bold")
        axes[r, 1].imshow(ov); axes[r, 1].axis("off")
        ok = pred == y[idx]
        axes[r, 1].set_title(f"→ {NAMES[pred]} ({conf:.2f}) {'✓' if ok else '✗'}",
                             fontsize=9, fontweight="bold",
                             color=("#1a7f37" if ok else "#c0392b"))
    fig.suptitle("Grad-CAM Kontras 'Ideal' — CK+ (ekspresi posed intensitas tinggi)",
                 fontsize=12, fontweight="bold", y=0.998)
    fig.text(0.5, 0.004,
             "Model CNN_TL 7c (terlatih di primer) diterapkan pada citra CK+. Atensi "
             "lebih tajam & terlokalisasi pada region ekspresif dibanding kondisi webcam primer.",
             ha="center", va="bottom", fontsize=7.5, style="italic", color="#444")
    plt.tight_layout(rect=[0, 0.015, 1, 0.98])
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"gradcam_5_7_2_ckplus.{ext}", dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(f"saved gradcam_5_7_2_ckplus.{{png,pdf}} ({n} sampel)")


if __name__ == "__main__":
    main()
