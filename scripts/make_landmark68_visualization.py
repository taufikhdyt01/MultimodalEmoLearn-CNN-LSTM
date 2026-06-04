"""
Visualisasi 68-point facial landmark (dlib scheme).

Output:
  docs/figures/landmark68_overlay.png  — overlay 68 numbered landmarks pada wajah
  docs/figures/landmark68_overlay.pdf
  docs/figures/landmark68_regions.png  — color-coded by anatomical region
  docs/figures/landmark68_comparison.png — MP vs FA side-by-side comparison

Source: face-api.js landmarks (FA, native 68 dlib) + MediaPipe (MP, 478→68 mapping).
Data: data/dataset_frontonly_conf60/X_train_{faceapi_,}landmarks.npy + X_train_images.npy

Usage:
    python scripts/make_landmark68_visualization.py
    python scripts/make_landmark68_visualization.py --sample-idx 100  # pakai sample lain
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "dataset_frontonly_conf60"
OUT_DIR = PROJECT_ROOT / "docs" / "figures"

# 68-point dlib region grouping
REGIONS = [
    ("Jaw (0–16)",           list(range(0, 17)),  "#3b7dd8"),  # blue
    ("Right Eyebrow (17–21)", list(range(17, 22)), "#1a7f37"),  # green
    ("Left Eyebrow (22–26)",  list(range(22, 27)), "#1a7f37"),
    ("Nose Bridge (27–30)",   list(range(27, 31)), "#9b59b6"),  # purple
    ("Nose Tip (31–35)",      list(range(31, 36)), "#9b59b6"),
    ("Right Eye (36–41)",     list(range(36, 42)), "#fac858"),  # gold
    ("Left Eye (42–47)",      list(range(42, 48)), "#fac858"),
    ("Outer Mouth (48–59)",   list(range(48, 60)), "#ee6666"),  # red
    ("Inner Mouth (60–67)",   list(range(60, 68)), "#ee6666"),
]


def get_region_color(idx):
    for _, rng, c in REGIONS:
        if idx in rng:
            return c
    return "#888888"


def draw_landmarks(ax, img, lm68, numbered=False, color_by_region=False,
                    point_size=18, line_width=1.0, show_connections=True):
    """Plot wajah + 68 landmark.

    img: (H, W, 3) float32 [0, 1]
    lm68: shape (136,) flat or (68, 2) — assumed normalized [0,1] relative to image
    """
    H, W = img.shape[:2]
    pts = lm68.reshape(68, 2) * np.array([W, H])

    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])

    if show_connections:
        # Draw region polylines / closed curves
        connection_groups = [
            (list(range(0, 17)),  False),  # jaw open
            (list(range(17, 22)), False),  # right brow
            (list(range(22, 27)), False),  # left brow
            (list(range(27, 31)), False),  # nose bridge
            (list(range(31, 36)), False),  # nose tip arc
            (list(range(36, 42)) + [36], True),  # right eye closed
            (list(range(42, 48)) + [42], True),  # left eye closed
            (list(range(48, 60)) + [48], True),  # outer mouth closed
            (list(range(60, 68)) + [60], True),  # inner mouth closed
        ]
        for group, _closed in connection_groups:
            xs = pts[group, 0]; ys = pts[group, 1]
            c = get_region_color(group[0]) if color_by_region else "#22ee22"
            ax.plot(xs, ys, "-", color=c, linewidth=line_width, alpha=0.6)

    # Draw points
    if color_by_region:
        for i in range(68):
            ax.scatter(pts[i, 0], pts[i, 1], s=point_size,
                       c=get_region_color(i), edgecolors="white",
                       linewidths=0.5, zorder=5)
    else:
        ax.scatter(pts[:, 0], pts[:, 1], s=point_size,
                   c="#22ee22", edgecolors="white", linewidths=0.5, zorder=5)

    # Number labels
    if numbered:
        for i in range(68):
            ax.annotate(str(i), (pts[i, 0], pts[i, 1]),
                         textcoords="offset points", xytext=(3, 3),
                         fontsize=6, color="white",
                         bbox=dict(boxstyle="round,pad=0.1",
                                   fc="#000000", alpha=0.55, ec="none"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-idx", type=int, default=None,
                    help="Index sample dari train set (default: pilih neutral confidence tertinggi)")
    ap.add_argument("--split", choices=["train", "val", "test"], default="train")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    images = np.load(DATA_DIR / f"X_{args.split}_images.npy")
    fa_lm  = np.load(DATA_DIR / f"X_{args.split}_faceapi_landmarks.npy")
    mp_lm  = np.load(DATA_DIR / f"X_{args.split}_landmarks.npy")
    y_hard = np.load(DATA_DIR / f"y_{args.split}.npy")

    if args.sample_idx is None:
        # Pilih sample neutral dengan confidence tertinggi (proxy: koordinat FA dalam
        # bounding [0,1], tidak ada landmark out-of-frame)
        soft_p = DATA_DIR / f"y_{args.split}_soft.npy"
        candidates = np.where(y_hard == 0)[0]  # neutral
        # Filter: landmarks fully inside [0.05, 0.95]
        good = []
        for i in candidates:
            lm = fa_lm[i].reshape(68, 2)
            if lm.min() >= 0.05 and lm.max() <= 0.95:
                good.append(i)
        if not good:
            good = candidates.tolist()
        if soft_p.exists():
            soft = np.load(soft_p)
            good = sorted(good, key=lambda i: -soft[i, 0])
        sample_idx = good[0]
    else:
        sample_idx = args.sample_idx
    print(f"Using sample idx={sample_idx} (class={y_hard[sample_idx]}, split={args.split})")

    img = np.clip(images[sample_idx], 0, 1)
    lm_fa = fa_lm[sample_idx]
    lm_mp = mp_lm[sample_idx]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Figure 1: numbered overlay (single panel, big) ────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8))
    draw_landmarks(ax, img, lm_fa, numbered=True, color_by_region=False,
                   point_size=28, line_width=1.2)
    ax.set_title("68-point Facial Landmark (face-api.js, dlib scheme)",
                 fontsize=12, pad=8)
    out = OUT_DIR / "landmark68_overlay.png"
    plt.tight_layout()
    plt.savefig(out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")

    # ── Figure 2: color-coded by region (with legend) ─────────────────────────
    fig, ax = plt.subplots(figsize=(9, 8))
    draw_landmarks(ax, img, lm_fa, numbered=False, color_by_region=True,
                   point_size=36, line_width=1.5)
    ax.set_title("68-point Facial Landmark — Color-coded by Anatomical Region",
                 fontsize=12, pad=8)
    # Legend (unique by color)
    seen = {}
    for label, _rng, color in REGIONS:
        # Use main region name (Jaw, Eyebrow, Nose, Eye, Mouth)
        main = label.split(" (")[0]
        if main.endswith(" Eyebrow"): main = "Eyebrows"
        elif main.endswith(" Eye"):    main = "Eyes"
        elif main.startswith("Nose"):  main = "Nose"
        elif "Mouth" in main:          main = "Mouth"
        seen[main] = color
    handles = [mpatches.Patch(color=c, label=lbl) for lbl, c in seen.items()]
    ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.85,
              edgecolor="#888888")
    out2 = OUT_DIR / "landmark68_regions.png"
    plt.tight_layout()
    plt.savefig(out2, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.savefig(out2.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out2}")

    # ── Figure 3: MP vs FA side-by-side comparison ────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    draw_landmarks(axes[0], img, lm_mp, numbered=False, color_by_region=True,
                   point_size=30, line_width=1.3)
    axes[0].set_title("(a) MediaPipe (478→68 dlib mapping)", fontsize=11, pad=6)
    draw_landmarks(axes[1], img, lm_fa, numbered=False, color_by_region=True,
                   point_size=30, line_width=1.3)
    axes[1].set_title("(b) face-api.js (native 68 dlib)", fontsize=11, pad=6)
    out3 = OUT_DIR / "landmark68_comparison.png"
    plt.tight_layout()
    plt.savefig(out3, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.savefig(out3.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out3}")


if __name__ == "__main__":
    main()
