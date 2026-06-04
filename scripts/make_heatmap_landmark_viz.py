"""
Visualisasi landmark heatmap (input channel ke-4 di Early Fusion).

Tujuan: pembaca tesis paham bagaimana 68 landmark di-render jadi heatmap 224×224
yang di-stack sebagai channel ke-4 RGB di Early Fusion.

Layout:
  Row 1 (a): RGB image asli
  Row 2 (b): Landmark heatmap (grayscale, hot colormap)
  Row 3 (c): Overlay (heatmap di-blend di atas RGB)
  Row 4 (d): 4-channel stacked tensor (R, G, B, Heatmap) sebagai 4 grayscale panel
             — input Early Fusion model.
  Kolom: 1 sample per kelas emosi (7 kelas).

Output:
  docs/figures/heatmap_landmark_overview.png
  docs/figures/heatmap_landmark_overview.pdf

Usage:
    python scripts/make_heatmap_landmark_viz.py
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "dataset_frontonly_conf60"
OUT_DIR = PROJECT_ROOT / "docs" / "figures"

EMOTIONS_7 = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]

# Konsisten dengan figure class_samples_all_datasets sebelumnya
SAMPLE_OVERRIDES = {2: 7, 3: 7, 4: 7}


def pick_sample(y_hard, y_soft, class_idx, seed=42, rank=None):
    idx_class = np.where(y_hard == class_idx)[0]
    if len(idx_class) == 0:
        return None
    if rank is not None and y_soft is not None:
        confs = y_soft[idx_class, class_idx]
        return idx_class[np.argsort(-confs)[min(rank, len(confs) - 1)]]
    if y_soft is not None:
        confs = y_soft[idx_class, class_idx]
        top_k = min(5, len(idx_class))
        top_idx_local = np.argsort(-confs)[:top_k]
        rng = np.random.RandomState(seed + class_idx)
        return idx_class[top_idx_local[rng.randint(top_k)]]
    rng = np.random.RandomState(seed + class_idx)
    return idx_class[rng.randint(len(idx_class))]


def make_overlay(rgb, hm, alpha=0.55, cmap_name="hot"):
    """Blend heatmap (gray) di atas RGB image."""
    cmap = plt.get_cmap(cmap_name)
    hm_norm = hm / max(hm.max(), 1e-6)
    hm_rgb = cmap(hm_norm)[..., :3]  # (H, W, 3)
    # alpha-blend hanya untuk pixel where hm > threshold
    mask = (hm_norm > 0.05)[..., None]
    overlay = np.where(mask, alpha * hm_rgb + (1 - alpha) * rgb, rgb)
    return np.clip(overlay, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    images = np.load(DATA_DIR / "X_train_images.npy")
    heatmaps = np.load(DATA_DIR / "X_train_heatmaps.npy")
    y_hard = np.load(DATA_DIR / "y_train.npy")
    soft_p = DATA_DIR / "y_train_soft.npy"
    y_soft = np.load(soft_p) if soft_p.exists() else None

    n_emo = len(EMOTIONS_7)
    n_row = 4  # a) RGB, b) Heatmap, c) Overlay, d) 4-channel stacked (compact)
    fig, axes = plt.subplots(n_row, n_emo,
                             figsize=(2.0 * n_emo, 2.0 * n_row + 0.4),
                             gridspec_kw={"wspace": 0.05, "hspace": 0.18,
                                          "left": 0.20, "right": 0.99,
                                          "top": 0.94, "bottom": 0.04})

    # Caption diperpendek supaya muat di margin kiri tanpa overlap ke image
    row_captions = [
        "(a) RGB image",
        "(b) Landmark heatmap",
        "(c) Heatmap overlay",
        "(d) Channel-4 input\n     (Early Fusion)",
    ]

    for ci, emo in enumerate(EMOTIONS_7):
        idx = pick_sample(y_hard, y_soft, ci, seed=args.seed,
                          rank=SAMPLE_OVERRIDES.get(ci))
        if idx is None:
            for r in range(n_row):
                axes[r, ci].axis("off")
            continue
        rgb = np.clip(images[idx], 0, 1)
        hm = heatmaps[idx]
        hm_norm = hm / max(hm.max(), 1e-6)

        # Row (a) RGB
        axes[0, ci].imshow(rgb)
        axes[0, ci].set_title(emo, fontsize=10, pad=4)

        # Row (b) Heatmap (hot colormap)
        axes[1, ci].imshow(hm, cmap="hot", vmin=0, vmax=hm.max() if hm.max() > 0 else 1)

        # Row (c) Overlay
        axes[2, ci].imshow(make_overlay(rgb, hm, alpha=0.6))

        # Row (d) Heatmap intensity di grayscale + Gaussian contour
        axes[3, ci].imshow(hm_norm, cmap="gray", vmin=0, vmax=1)

        # Common axis styling
        for r in range(n_row):
            ax = axes[r, ci]
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#888888"); spine.set_linewidth(0.6)

    # Row labels (left-aligned, figure-level)
    for di, caption in enumerate(row_captions):
        ax_pos = axes[di, 0].get_position()
        y_center = (ax_pos.y0 + ax_pos.y1) / 2
        fig.text(0.005, y_center, caption, fontsize=11,
                 ha="left", va="center", fontweight="bold")

    fig.suptitle("Landmark Heatmap Visualization — Input Channel ke-4 untuk Early Fusion",
                 fontsize=13, y=0.985, fontweight="bold")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "heatmap_landmark_overview.png"
    out_pdf = OUT_DIR / "heatmap_landmark_overview.pdf"
    plt.savefig(out_png, dpi=args.dpi, bbox_inches="tight", pad_inches=0.15,
                facecolor="white")
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    plt.close(fig)
    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
