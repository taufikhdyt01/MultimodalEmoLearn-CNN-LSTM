"""
Extract 3 file input siap-pakai untuk diagram arsitektur Claude Design.

Output:
  docs/figures/_archviz_input_face.png       — single 224×224 face (RGB)
  docs/figures/_archviz_input_landmark.png   — 68 landmark dots, no face background
  docs/figures/_archviz_input_heatmap.png    — single heatmap (hot colormap)

Semua dari sample yang sama (idx 4467, neutral, user 209) supaya visual konsisten
lintas diagram arsitektur.

Usage:
    python scripts/make_archviz_inputs.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "dataset_frontonly_conf60"
OUT_DIR = PROJECT_ROOT / "docs" / "figures"

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


def main():
    SAMPLE_IDX = 4467
    images = np.load(DATA_DIR / "X_train_images.npy")
    fa_lm  = np.load(DATA_DIR / "X_train_faceapi_landmarks.npy")
    heatmaps = np.load(DATA_DIR / "X_train_heatmaps.npy")

    face_img = np.clip(images[SAMPLE_IDX], 0, 1)
    lm = fa_lm[SAMPLE_IDX].reshape(68, 2)
    hm = heatmaps[SAMPLE_IDX]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ───────────────────────────────────────────────────────────────────────
    # 1. Face image (clean 224×224 RGB, no annotations)
    # ───────────────────────────────────────────────────────────────────────
    face_out = OUT_DIR / "_archviz_input_face.png"
    face_uint8 = (face_img * 255).astype(np.uint8)
    Image.fromarray(face_uint8).save(face_out)
    print(f"Saved {face_out}  ({face_uint8.shape[1]}×{face_uint8.shape[0]})")

    # ───────────────────────────────────────────────────────────────────────
    # 2. Landmark dots only (no face bg, white canvas, region-colored)
    # ───────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    H, W = 224, 224
    pts = lm * np.array([W, H])
    # Connection lines
    connections = [
        list(range(0, 17)), list(range(17, 22)), list(range(22, 27)),
        list(range(27, 31)), list(range(31, 36)),
        list(range(36, 42)) + [36], list(range(42, 48)) + [42],
        list(range(48, 60)) + [48], list(range(60, 68)) + [60],
    ]
    for grp in connections:
        c = color_for_idx(grp[0])
        ax.plot(pts[grp, 0], pts[grp, 1], "-", color=c, linewidth=1.5, alpha=0.85)
    for i in range(68):
        ax.scatter(pts[i, 0], pts[i, 1], s=30,
                   c=color_for_idx(i), edgecolors="white",
                   linewidths=0.5, zorder=5)
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    lm_out = OUT_DIR / "_archviz_input_landmark.png"
    plt.tight_layout(pad=0)
    plt.savefig(lm_out, dpi=200, bbox_inches="tight", pad_inches=0.05,
                facecolor="white")
    plt.close(fig)
    print(f"Saved {lm_out}")

    # ───────────────────────────────────────────────────────────────────────
    # 3. Heatmap (hot colormap, no axes, no annotations)
    # ───────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(hm, cmap="hot", vmin=0, vmax=hm.max() if hm.max() > 0 else 1)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    hm_out = OUT_DIR / "_archviz_input_heatmap.png"
    plt.tight_layout(pad=0)
    plt.savefig(hm_out, dpi=200, bbox_inches="tight", pad_inches=0.05,
                facecolor="white")
    plt.close(fig)
    print(f"Saved {hm_out}")

    print("\nSemua file siap dipakai sebagai input image di Claude Design.")
    print(f"Sample source: idx {SAMPLE_IDX}, user 209, class=neutral (high confidence).")


if __name__ == "__main__":
    main()
