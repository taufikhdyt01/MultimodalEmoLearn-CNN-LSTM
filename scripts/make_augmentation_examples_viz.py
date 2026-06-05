"""
Visualisasi contoh augmentation B3 — sync apply ke image, landmark, dan heatmap.

5 kolom augmentation:
  (a) Original
  (b) Horizontal flip (sync flip landmark left/right swap + heatmap mirror)
  (c) Rotate +10° (sync rotate image + landmark + heatmap)
  (d) Brightness/Contrast (photometric, image-only — landmark & heatmap unchanged)
  (e) Combined (B3 realistic example: hflip + rotate + photometric)

3 baris channel:
  (1) Image (RGB)
  (2) Landmark overlay (titik 68 di atas RGB)
  (3) Heatmap (grayscale)

Output:
  docs/figures/augmentation_examples_b3.png
  docs/figures/augmentation_examples_b3.pdf

Usage:
    python scripts/make_augmentation_examples_viz.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
DATA_DIR = PROJECT_ROOT / "data" / "dataset_frontonly_conf60"
OUT_DIR = PROJECT_ROOT / "docs" / "figures"

# Reuse aug functions dari src/training/fusion_aug.py (jangan duplikasi)
from training.fusion_aug import (
    _hflip_image, _hflip_heatmap, _hflip_landmark_136,
    _rotate_image, _rotate_heatmap, _rotate_landmark_136,
    _augment_image_photometric,
)

REGION_COLORS = {
    "Jaw":      ("#3b7dd8", list(range(0, 17))),
    "Eyebrows": ("#1a7f37", list(range(17, 27))),
    "Nose":     ("#9b59b6", list(range(27, 36))),
    "Eyes":     ("#fac858", list(range(36, 48))),
    "Mouth":    ("#ee6666", list(range(48, 68))),
}


def color_for_idx(i):
    for c, rng in REGION_COLORS.values():
        if i in rng:
            return c
    return "#888888"


CONNECTIONS = [
    list(range(0, 17)), list(range(17, 22)), list(range(22, 27)),
    list(range(27, 31)), list(range(31, 36)),
    list(range(36, 42)) + [36], list(range(42, 48)) + [42],
    list(range(48, 60)) + [48], list(range(60, 68)) + [60],
]


def overlay_landmarks(ax, img, lm_flat):
    H, W = img.shape[:2]
    pts = lm_flat.reshape(68, 2) * np.array([W, H])
    ax.imshow(img)
    for grp in CONNECTIONS:
        c = color_for_idx(grp[0])
        ax.plot(pts[grp, 0], pts[grp, 1], "-", color=c,
                linewidth=0.9, alpha=0.75)
    for i in range(68):
        ax.scatter(pts[i, 0], pts[i, 1], s=8, c=color_for_idx(i),
                   edgecolors="white", linewidths=0.3, zorder=5)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    images = np.load(DATA_DIR / "X_train_images.npy")
    # Source: MediaPipe 68-point landmarks (478->68 dlib mapping), bukan face-api.js
    mp_lm  = np.load(DATA_DIR / "X_train_landmarks.npy")
    heatmaps = np.load(DATA_DIR / "X_train_heatmaps.npy")
    y_hard = np.load(DATA_DIR / "y_train.npy")
    soft_p = DATA_DIR / "y_train_soft.npy"
    y_soft = np.load(soft_p) if soft_p.exists() else None

    # Pilih sample neutral confidence tinggi (sample yang sama dengan figure lain)
    candidates = np.where(y_hard == 0)[0]
    good = []
    for i in candidates:
        lm = mp_lm[i].reshape(68, 2)
        if lm.min() >= 0.05 and lm.max() <= 0.95:
            good.append(i)
    if y_soft is not None:
        good = sorted(good, key=lambda i: -y_soft[i, 0])
    sample_idx = good[0]
    print(f"Using sample idx={sample_idx}")

    img_orig = np.clip(images[sample_idx], 0, 1).astype(np.float32)
    lm_orig = mp_lm[sample_idx].astype(np.float32)
    hm_orig = heatmaps[sample_idx].astype(np.float32)

    # ── Apply augmentations ──────────────────────────────────────────────────
    # Hflip
    img_hf  = _hflip_image(img_orig)
    lm_hf   = _hflip_landmark_136(lm_orig)
    hm_hf   = _hflip_heatmap(hm_orig)

    # Rotate +10° (CCW)
    angle = 10.0
    img_rot = _rotate_image(img_orig, angle)
    lm_rot  = _rotate_landmark_136(lm_orig, angle)
    hm_rot  = _rotate_heatmap(hm_orig, angle)

    # Brightness/Contrast (deterministic example with fixed b=+0.08, c=1.10)
    # We construct a custom photometric to make the change visually obvious
    img_bc = np.clip((img_orig - 0.5) * 1.10 + 0.5 + 0.08, 0, 1).astype(np.float32)
    # Photometric tidak ubah landmark/heatmap
    lm_bc = lm_orig.copy()
    hm_bc = hm_orig.copy()

    # Combined (B3 realistic): hflip + rotate -8° + photometric
    img_co = _hflip_image(img_orig)
    lm_co  = _hflip_landmark_136(lm_orig)
    hm_co  = _hflip_heatmap(hm_orig)
    img_co = _rotate_image(img_co, -8.0)
    lm_co  = _rotate_landmark_136(lm_co, -8.0)
    hm_co  = _rotate_heatmap(hm_co, -8.0)
    img_co = np.clip((img_co - 0.5) * 1.08 + 0.5 + 0.05, 0, 1).astype(np.float32)

    col_data = [
        ("(a) Original",                 img_orig, lm_orig, hm_orig),
        ("(b) Horizontal flip",          img_hf,   lm_hf,   hm_hf),
        ("(c) Rotate +10°",              img_rot,  lm_rot,  hm_rot),
        ("(d) Brightness/Contrast",      img_bc,   lm_bc,   hm_bc),
        ("(e) Combined (B3 example)",    img_co,   lm_co,   hm_co),
    ]

    # ── Plot 3 rows × 5 cols ─────────────────────────────────────────────────
    n_cols = len(col_data)
    # left margin diperbesar (0.10→0.16) supaya label rata kiri tidak menabrak axes
    fig, axes = plt.subplots(3, n_cols,
                              figsize=(2.4 * n_cols, 2.4 * 3 + 0.8),
                              gridspec_kw={"wspace": 0.06, "hspace": 0.12,
                                           "left": 0.16, "right": 0.99,
                                           "top": 0.90, "bottom": 0.05})

    # Caption diperpendek supaya muat di margin kiri
    row_captions = [
        "(1) RGB image",
        "(2) Landmark",
        "(3) Heatmap",
    ]

    for ci, (col_title, img, lm, hm) in enumerate(col_data):
        # Row 0: Image only
        axes[0, ci].imshow(img)
        axes[0, ci].set_xticks([]); axes[0, ci].set_yticks([])
        axes[0, ci].set_title(col_title, fontsize=10, fontweight="bold", pad=6)

        # Row 1: Landmark overlay
        overlay_landmarks(axes[1, ci], img, lm)

        # Row 2: Heatmap (hot)
        axes[2, ci].imshow(hm, cmap="hot", vmin=0,
                            vmax=hm.max() if hm.max() > 0 else 1)
        axes[2, ci].set_xticks([]); axes[2, ci].set_yticks([])

        for r in range(3):
            for spine in axes[r, ci].spines.values():
                spine.set_edgecolor("#888888"); spine.set_linewidth(0.6)

    # Row labels (left-aligned, figure-level)
    for di, caption in enumerate(row_captions):
        ax_pos = axes[di, 0].get_position()
        y_center = (ax_pos.y0 + ax_pos.y1) / 2
        fig.text(0.005, y_center, caption, fontsize=11,
                 ha="left", va="center", fontweight="bold")

    fig.suptitle("Augmentation Examples (B3 — synced per-batch augmentation)",
                 fontsize=14, fontweight="bold", y=0.97)
    fig.text(0.5, 0.01,
             "Catatan: Photometric (d) hanya diaplikasikan ke image — landmark & heatmap unchanged. "
             "Spatial aug (b, c, e) sync ke ketiga channel.",
             ha="center", va="bottom", fontsize=9, color="#444444", style="italic")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "augmentation_examples_b3.png"
    out_pdf = OUT_DIR / "augmentation_examples_b3.pdf"
    plt.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.15,
                facecolor="white")
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    plt.close(fig)
    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
