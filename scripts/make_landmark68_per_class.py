"""
Visualisasi 68-point landmark per kelas emosi × 2 source (face-api.js vs MediaPipe).

Layout: 2 baris × 7 kolom.
  Baris (a) = face-api.js (native 68 dlib)
  Baris (b) = MediaPipe (478→68 dlib mapping)
  Kolom = neutral / happy / sad / angry / fearful / disgusted / surprised

Tujuan: ilustrasi visual perbedaan kualitas landmark detector antara FA dan MP
yang diuji di hasil eksperimen unimodal (FA > MP konsisten di top results).

Output:
  docs/figures/landmark68_per_class.png
  docs/figures/landmark68_per_class.pdf

Usage:
    python scripts/make_landmark68_per_class.py
    python scripts/make_landmark68_per_class.py --seed 7
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "dataset_frontonly_conf60"
OUT_DIR = PROJECT_ROOT / "docs" / "figures"

EMOTIONS_7 = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]

# Region grouping (dlib 68-point standard)
REGION_COLORS = {
    "Jaw":      ("#3b7dd8", list(range(0, 17))),
    "Eyebrows": ("#1a7f37", list(range(17, 27))),
    "Nose":     ("#9b59b6", list(range(27, 36))),
    "Eyes":     ("#fac858", list(range(36, 48))),
    "Mouth":    ("#ee6666", list(range(48, 68))),
}

CONNECTION_GROUPS = [
    (list(range(0, 17)),  False),  # jaw
    (list(range(17, 22)), False),  # right brow
    (list(range(22, 27)), False),  # left brow
    (list(range(27, 31)), False),  # nose bridge
    (list(range(31, 36)), False),  # nose tip
    (list(range(36, 42)) + [36], True),  # right eye closed
    (list(range(42, 48)) + [42], True),  # left eye closed
    (list(range(48, 60)) + [48], True),  # outer mouth closed
    (list(range(60, 68)) + [60], True),  # inner mouth closed
]


def color_for_index(idx):
    for c, rng in REGION_COLORS.values():
        if idx in rng:
            return c
    return "#888888"


def draw_landmarks(ax, img_shape, lm68, point_size=14, line_width=1.0):
    """Plot landmark points + connection lines TANPA citra wajah background.

    img_shape: (H, W) atau (H, W, 3) — dipakai untuk skala koordinat saja.
    lm68: (136,) flat atau (68, 2), normalized [0, 1].
    """
    H, W = img_shape[:2]
    pts = lm68.reshape(68, 2) * np.array([W, H])
    # Connection lines
    for group, _closed in CONNECTION_GROUPS:
        c = color_for_index(group[0])
        ax.plot(pts[group, 0], pts[group, 1], "-",
                color=c, linewidth=line_width, alpha=0.85)
    # Points
    for i in range(68):
        ax.scatter(pts[i, 0], pts[i, 1], s=point_size,
                   c=color_for_index(i), edgecolors="white",
                   linewidths=0.4, zorder=5)
    # Aspect & framing — supaya layout mirror image dimensions
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)             # invert y so origin = top-left (image convention)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#888888"); spine.set_linewidth(0.6)


def pick_sample(y_hard, y_soft, class_idx, seed, rank=None):
    idx_class = np.where(y_hard == class_idx)[0]
    if len(idx_class) == 0:
        return None
    if rank is not None and y_soft is not None:
        confs = y_soft[idx_class, class_idx]
        sorted_local = np.argsort(-confs)
        return idx_class[sorted_local[min(rank, len(sorted_local) - 1)]]
    if y_soft is not None:
        confs = y_soft[idx_class, class_idx]
        top_k = min(5, len(idx_class))
        top_idx_local = np.argsort(-confs)[:top_k]
        rng = np.random.RandomState(seed + class_idx)
        return idx_class[top_idx_local[rng.randint(top_k)]]
    rng = np.random.RandomState(seed + class_idx)
    return idx_class[rng.randint(len(idx_class))]


# Sama dengan class_samples_all_datasets: shift sad/angry/fearful supaya
# konsisten dengan figure sebelumnya
SAMPLE_OVERRIDES = {2: 7, 3: 7, 4: 7}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    images = np.load(DATA_DIR / "X_train_images.npy")
    fa_lm  = np.load(DATA_DIR / "X_train_faceapi_landmarks.npy")
    mp_lm  = np.load(DATA_DIR / "X_train_landmarks.npy")
    y_hard = np.load(DATA_DIR / "y_train.npy")
    soft_p = DATA_DIR / "y_train_soft.npy"
    y_soft = np.load(soft_p) if soft_p.exists() else None

    n_emo = len(EMOTIONS_7)
    fig, axes = plt.subplots(2, n_emo,
                             figsize=(2.0 * n_emo, 2.0 * 2 + 0.8),
                             gridspec_kw={"wspace": 0.05, "hspace": 0.20,
                                          "left": 0.13, "right": 0.99,
                                          "top": 0.84, "bottom": 0.04})

    # Per-emotion: pick sample, plot FA (row 0) and MP (row 1)
    for ci, emo in enumerate(EMOTIONS_7):
        idx = pick_sample(y_hard, y_soft, ci, args.seed,
                          rank=SAMPLE_OVERRIDES.get(ci))
        if idx is None:
            for r in range(2):
                axes[r, ci].axis("off")
            continue
        img_shape = images[idx].shape  # (H, W, 3)
        draw_landmarks(axes[0, ci], img_shape, fa_lm[idx])
        draw_landmarks(axes[1, ci], img_shape, mp_lm[idx])
        axes[0, ci].set_title(emo, fontsize=10, pad=4)

    # Row labels (left-aligned, figure-level — outside axes biar tidak overlap)
    # x=0.01 → mulai dari edge kiri figure; axes mulai dari left=0.13 → ada buffer
    for di, caption in enumerate(["(a) face-api.js", "(b) MediaPipe"]):
        ax_pos = axes[di, 0].get_position()
        y_center = (ax_pos.y0 + ax_pos.y1) / 2
        fig.text(0.01, y_center, caption, fontsize=11,
                 ha="left", va="center", fontweight="bold")

    # Legend di atas dengan jarak yang cukup
    # top axes ada di y=0.84 (gridspec), titik nama emosi di atasnya ~0.86,
    # legend ditempat di y=0.95 jadi ada gap visual jelas
    handles = [mpatches.Patch(color=c, label=name)
               for name, (c, _) in REGION_COLORS.items()]
    fig.legend(handles=handles, loc="upper center", ncol=len(REGION_COLORS),
               bbox_to_anchor=(0.5, 0.97), fontsize=9, frameon=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "landmark68_per_class.png"
    out_pdf = OUT_DIR / "landmark68_per_class.pdf"
    # bbox_inches='tight' may crop our manual padding; use a small pad explicitly
    plt.savefig(out_png, dpi=args.dpi, bbox_inches="tight", pad_inches=0.15,
                facecolor="white")
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    plt.close(fig)
    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
