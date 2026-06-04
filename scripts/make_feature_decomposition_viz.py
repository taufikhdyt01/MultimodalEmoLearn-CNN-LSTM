"""
Visualisasi 4 representasi feature landmark (RQ2):
  (a) raw_136  — 68 koordinat 2D landmark (136-dim flat)
  (b) facs_28  — 28 Euclidean distance antar pasangan landmark berbasis FACS-AU
  (c) blendshape_52 — 52 koefisien ARKit Blendshape (MediaPipe)
  (d) facs_plus_bs_80 — concat (facs_28 + blendshape_52)

Layout: 2 baris × 2 kolom.
  Atas: spatial (titik & garis) berdasarkan 1 sample wajah.
  Bawah: bar chart vektor feature.

Output:
  docs/figures/feature_decomposition_overview.png
  docs/figures/feature_decomposition_overview.pdf

Usage:
    python scripts/make_feature_decomposition_viz.py
    python scripts/make_feature_decomposition_viz.py --sample-idx 100
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "dataset_frontonly_conf60"
OUT_DIR = PROJECT_ROOT / "docs" / "figures"

# ── FACS-28 pair definitions (mirror run_unified_landmark.py) ────────────────
FACS_PAIRS = [
    ("AU1_left",  17, 21), ("AU1_right", 22, 26),
    ("AU2_left",  20, 39), ("AU2_right", 23, 42),
    ("AU4_left",  21, 39), ("AU4_right", 22, 42),
    ("AU5_left",  37, 41), ("AU5_right", 43, 47),
    ("AU6_left",   1, 41), ("AU6_right", 15, 47),
    ("AU7_left",  37, 40), ("AU7_right", 43, 46),
    ("AU9_left",  31, 39), ("AU9_right", 35, 42),
    ("AU10_top",  33, 51), ("AU10_left", 31, 48), ("AU10_right", 35, 54),
    ("AU12_left", 48, 36), ("AU12_right", 54, 45),
    ("AU15_left", 48,  4), ("AU15_right", 54, 12),
    ("AU17_left",  8, 57), ("AU17_right",  8, 51),
    ("AU20_left", 48, 60), ("AU20_right", 54, 64),
    ("AU23_outer", 48, 54), ("AU24_top",   51, 57),
    ("AU25",      62, 66),
]
INTEROCULAR_PAIR = (36, 45)

# ── 52 ARKit Blendshape names (urutan dari MediaPipe FaceLandmarker v2) ──────
BLENDSHAPE_NAMES = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight", "cheekPuff", "cheekSquintLeft",
    "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft",
    "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
    "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft",
    "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft",
    "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft",
    "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
    "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
]
assert len(BLENDSHAPE_NAMES) == 52

# Region grouping (dlib 68-point)
REGION_COLORS = {
    "Jaw":      ("#3b7dd8", list(range(0, 17))),
    "Eyebrows": ("#1a7f37", list(range(17, 27))),
    "Nose":     ("#9b59b6", list(range(27, 36))),
    "Eyes":     ("#fac858", list(range(36, 48))),
    "Mouth":    ("#ee6666", list(range(48, 68))),
}


def color_for_index(idx):
    for c, rng in REGION_COLORS.values():
        if idx in rng:
            return c
    return "#888888"


def compute_facs28(lm_136):
    pts = lm_136.reshape(-1, 68, 2)
    a, b = INTEROCULAR_PAIR
    iod = np.maximum(np.linalg.norm(pts[:, a] - pts[:, b], axis=1), 1e-6)
    out = np.zeros((len(pts), len(FACS_PAIRS)), dtype=np.float32)
    for i, (_, p1, p2) in enumerate(FACS_PAIRS):
        d = np.linalg.norm(pts[:, p1] - pts[:, p2], axis=1)
        out[:, i] = (d / iod).astype(np.float32)
    return out


# ── Panel (a): raw_136 — 68 dots only ────────────────────────────────────────
def draw_raw136(ax, img_shape, lm68):
    H, W = img_shape[:2]
    pts = lm68.reshape(68, 2) * np.array([W, H])
    for i in range(68):
        ax.scatter(pts[i, 0], pts[i, 1], s=22,
                   c=color_for_index(i), edgecolors="white",
                   linewidths=0.4, zorder=5)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("(a) raw_136 — 68 koordinat 2D (136-dim)", fontsize=11, pad=8)


# ── Panel (b): facs_28 — 28 distance lines between landmark pairs ────────────
def draw_facs28(ax, img_shape, lm68):
    H, W = img_shape[:2]
    pts = lm68.reshape(68, 2) * np.array([W, H])
    # Draw FACS distance lines (each = Euclidean distance / interocular)
    used_pts = set()
    for name, p1, p2 in FACS_PAIRS:
        ax.plot([pts[p1, 0], pts[p2, 0]], [pts[p1, 1], pts[p2, 1]],
                "-", color="#cf222e", linewidth=1.2, alpha=0.85)
        used_pts.add(p1); used_pts.add(p2)
    # Draw landmark endpoints used in FACS pairs
    for i in used_pts:
        ax.scatter(pts[i, 0], pts[i, 1], s=24,
                   c=color_for_index(i), edgecolors="white",
                   linewidths=0.4, zorder=5)
    # Highlight interocular (denominator) in dashed gray
    a, b = INTEROCULAR_PAIR
    ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]],
            "--", color="#666666", linewidth=1.0, alpha=0.6, zorder=2)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("(b) facs_28 — 28 jarak Euclidean antar pasangan FACS-AU\n(garis abu putus-putus = jarak interocular sebagai normalizer)",
                 fontsize=10, pad=6)


# ── Panel (c): blendshape_52 — bar chart 52 coefficients ─────────────────────
def draw_blendshape52(ax, bs_vec):
    n = len(bs_vec)
    xs = np.arange(n)
    bars = ax.bar(xs, bs_vec, color="#3b7dd8", edgecolor="#1a4578",
                  linewidth=0.3, alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels(BLENDSHAPE_NAMES, rotation=90, fontsize=6.5)
    ax.set_ylabel("coefficient value", fontsize=9)
    ax.set_xlim(-0.7, n - 0.3)
    ax.set_ylim(0, max(bs_vec.max() * 1.15, 0.05))
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_title("(c) blendshape_52 — 52 koefisien ARKit (MediaPipe FaceLandmarker v2)",
                 fontsize=11, pad=8)


# ── Panel (d): facs_plus_bs_80 — concat bar chart, color-coded ───────────────
def draw_fb80(ax, fb_vec):
    n = len(fb_vec)
    colors = ["#cf222e"] * 28 + ["#3b7dd8"] * 52
    xs = np.arange(n)
    ax.bar(xs, fb_vec, color=colors, edgecolor="black", linewidth=0.2, alpha=0.85)
    # Vertical separator between FACS and Blendshape segments
    ax.axvline(27.5, color="#222222", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.text(13.5, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else 0.5,
            "FACS_28\n(distances)", ha="center", va="top", fontsize=8,
            color="#cf222e", fontweight="bold")
    ax.text(53.5, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else 0.5,
            "Blendshape_52\n(coefficients)", ha="center", va="top", fontsize=8,
            color="#3b7dd8", fontweight="bold")
    ax.set_xlim(-0.7, n - 0.3)
    ax.set_xticks([0, 14, 28, 41, 54, 67, 80])
    ax.set_xticklabels(["0", "14", "28", "41", "54", "67", "80"], fontsize=8)
    ax.set_xlabel("feature index (0–27 FACS, 28–79 Blendshape)", fontsize=9)
    ax.set_ylabel("value", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_title("(d) facs_plus_bs_80 — concat 28 jarak FACS + 52 koefisien Blendshape",
                 fontsize=11, pad=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-idx", type=int, default=None,
                    help="Index sample (default: pilih neutral confidence tertinggi tanpa NaN blendshape)")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    images = np.load(DATA_DIR / "X_train_images.npy")
    fa_lm  = np.load(DATA_DIR / "X_train_faceapi_landmarks.npy")
    bs_all = np.load(DATA_DIR / "X_train_mp_blendshapes.npy")
    y_hard = np.load(DATA_DIR / "y_train.npy")

    if args.sample_idx is None:
        # Cari sample neutral dengan landmark dalam frame & blendshape non-NaN
        candidates = np.where(y_hard == 0)[0]
        soft_p = DATA_DIR / "y_train_soft.npy"
        soft = np.load(soft_p) if soft_p.exists() else None
        good = []
        for i in candidates:
            lm = fa_lm[i].reshape(68, 2)
            if lm.min() >= 0.05 and lm.max() <= 0.95 and not np.isnan(bs_all[i]).any():
                good.append(i)
        if not good:
            good = candidates.tolist()
        if soft is not None:
            good = sorted(good, key=lambda i: -soft[i, 0])
        sample_idx = good[0]
    else:
        sample_idx = args.sample_idx
    print(f"Using sample idx={sample_idx} (class={y_hard[sample_idx]})")

    img_shape = images[sample_idx].shape
    lm = fa_lm[sample_idx]
    bs = np.nan_to_num(bs_all[sample_idx], nan=0.0)
    facs28 = compute_facs28(lm.reshape(1, -1))[0]
    fb80 = np.concatenate([facs28, bs])

    # Layout: legend di atas, dua baris panel dengan height_ratios supaya
    # row spasial (a/b square) tidak menyisakan banyak whitespace vertikal.
    # top=0.86 → axes mulai dari y=0.86, beri buffer ~0.10 untuk legend di atas
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5),
                             gridspec_kw={"hspace": 0.20, "wspace": 0.18,
                                          "height_ratios": [1.4, 1.0],
                                          "left": 0.07, "right": 0.99,
                                          "top": 0.86, "bottom": 0.10})
    draw_raw136(axes[0, 0], img_shape, lm)
    draw_facs28(axes[0, 1], img_shape, lm)
    draw_blendshape52(axes[1, 0], bs)
    draw_fb80(axes[1, 1], fb80)

    # Region legend di atas figure — jauh dari label blendshape (yang ada di bawah)
    handles = [mpatches.Patch(color=c, label=name)
               for name, (c, _) in REGION_COLORS.items()]
    fig.legend(handles=handles, loc="upper center", ncol=len(REGION_COLORS),
               bbox_to_anchor=(0.5, 0.96), fontsize=9, frameon=False,
               title="Landmark region (untuk panel a & b)", title_fontsize=9)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "feature_decomposition_overview.png"
    out_pdf = OUT_DIR / "feature_decomposition_overview.pdf"
    plt.savefig(out_png, dpi=args.dpi, bbox_inches="tight", pad_inches=0.15,
                facecolor="white")
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    plt.close(fig)
    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
