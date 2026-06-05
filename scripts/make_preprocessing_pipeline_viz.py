"""
Visualisasi alur preprocessing data (Bab 3 Metodologi tesis):
  raw frame → face detection → crop → landmark extraction → derived features

Output:
  docs/figures/preprocessing_pipeline.png
  docs/figures/preprocessing_pipeline.pdf

Usage:
    python scripts/make_preprocessing_pipeline_viz.py
    python scripts/make_preprocessing_pipeline_viz.py --sample-idx 100
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "dataset_frontonly_conf60"
OUT_DIR = PROJECT_ROOT / "docs" / "figures"

# Region colors konsisten dengan figure lain
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


def simulate_raw_frame(face_img, target_aspect=16/9, face_scale=0.35):
    """Simulasi 'raw video frame' HD 16:9 dengan wajah berukuran realistis.

    face_scale: ukuran wajah relatif terhadap tinggi frame
                (0.35 = wajah 35% tinggi frame, mendekati realistic webcam framing)
    """
    H_face, W_face = face_img.shape[:2]
    # Tinggi frame ditentukan supaya face_scale terpenuhi
    canvas_h = int(H_face / face_scale)
    canvas_w = int(canvas_h * target_aspect)
    # Background — abu gelap dengan gradient halus supaya tidak monoton flat
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    # Subtle vertical gradient (gelap di atas, sedikit lebih terang di bawah)
    for y in range(canvas_h):
        v = 0.08 + 0.05 * (y / canvas_h)
        canvas[y, :] = v
    # Tambah subtle horizontal vignette
    for x in range(canvas_w):
        dist = abs(x - canvas_w / 2) / (canvas_w / 2)
        canvas[:, x] *= 1.0 - 0.3 * dist
    # Posisikan wajah di tengah frame
    y0 = (canvas_h - H_face) // 2
    x0 = (canvas_w - W_face) // 2
    canvas[y0:y0 + H_face, x0:x0 + W_face] = face_img
    return canvas, (x0, y0, x0 + W_face, y0 + H_face)  # bbox xyxy


def draw_arrow(fig, ax_from, ax_to, color="#333333", text=None):
    """Draw arrow between two axes (in figure coords)."""
    bbox_from = ax_from.get_position()
    bbox_to = ax_to.get_position()
    x0 = bbox_from.x1
    y0 = (bbox_from.y0 + bbox_from.y1) / 2
    x1 = bbox_to.x0
    y1 = (bbox_to.y0 + bbox_to.y1) / 2
    arrow = FancyArrowPatch((x0, y0), (x1, y1),
                            transform=fig.transFigure,
                            arrowstyle="-|>", mutation_scale=18,
                            color=color, linewidth=1.6, zorder=10)
    fig.add_artist(arrow)
    if text:
        fig.text((x0 + x1) / 2, y0 + 0.015, text,
                 ha="center", va="bottom", fontsize=8, color=color,
                 fontweight="bold")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-idx", type=int, default=None,
                    help="Index sample. Default: neutral confidence tertinggi.")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    images = np.load(DATA_DIR / "X_train_images.npy")
    fa_lm  = np.load(DATA_DIR / "X_train_faceapi_landmarks.npy")  # dipakai utk seleksi sample (sample tetap sama)
    mp_lm  = np.load(DATA_DIR / "X_train_landmarks.npy")          # MediaPipe 68-titik — dipakai utk overlay stage 4
    bs_all = np.load(DATA_DIR / "X_train_mp_blendshapes.npy")
    heatmaps = np.load(DATA_DIR / "X_train_heatmaps.npy")
    y_hard = np.load(DATA_DIR / "y_train.npy")

    if args.sample_idx is None:
        candidates = np.where(y_hard == 0)[0]
        soft_p = DATA_DIR / "y_train_soft.npy"
        soft = np.load(soft_p) if soft_p.exists() else None
        good = []
        for i in candidates:
            lm = fa_lm[i].reshape(68, 2)
            if lm.min() >= 0.05 and lm.max() <= 0.95 and not np.isnan(bs_all[i]).any():
                good.append(i)
        if not good: good = candidates.tolist()
        if soft is not None:
            good = sorted(good, key=lambda i: -soft[i, 0])
        sample_idx = good[0]
    else:
        sample_idx = args.sample_idx
    print(f"Using sample idx={sample_idx} (class={y_hard[sample_idx]})")

    face_img = np.clip(images[sample_idx], 0, 1)
    lm = mp_lm[sample_idx]  # overlay stage 4 pakai MediaPipe
    hm = heatmaps[sample_idx]
    H, W = face_img.shape[:2]

    # ── Build figure: 1 row × 5 main stages + output annotation ─────────────
    fig = plt.figure(figsize=(18, 5.5))
    # 5 stage panels — equal spacing
    n_stage = 5
    pad = 0.02
    panel_w = (1 - 2 * pad - (n_stage - 1) * 0.025) / n_stage
    panel_h = 0.62
    y_panel = 0.18
    axes = []
    for i in range(n_stage):
        x = pad + i * (panel_w + 0.025)
        ax = fig.add_axes([x, y_panel, panel_w, panel_h])
        axes.append(ax)

    # Stage labels (above panels)
    stage_titles = [
        "(1) Raw video frame",
        "(2) Face detection",
        "(3) Crop 224×224",
        "(4) Landmark extraction",
        "(5) Heatmap rendering",
    ]
    stage_descs = [
        "frame dari\nrekaman webcam",
        "MediaPipe FaceLandmarker\nbounding box",
        "face crop\nresize ke 224×224",
        "68 titik MediaPipe (MP)\n(FA di-extract paralel)",
        "Gaussian blob\nper landmark → channel-4",
    ]

    # === Stage 1 & 2: Pakai raw frame asli kalau tersedia ────────────────
    raw_frame_path = PROJECT_ROOT / "docs" / "figures" / "raw_frame_user209_2025-11-27_15-07-43.png.jpg"
    use_real_raw = raw_frame_path.exists()
    if use_real_raw:
        from PIL import Image as PILImage
        raw_frame = np.array(PILImage.open(raw_frame_path)).astype(np.float32) / 255.0
        Hc, Wc = raw_frame.shape[:2]
        # Bbox dari Haar cascade face detection (lihat preprocessing_pipeline.md)
        x0, y0, bbox_w, bbox_h = 393, 84, 358, 358
    else:
        raw_frame, (x0, y0, x1, y1) = simulate_raw_frame(face_img,
                                                          target_aspect=16/9,
                                                          face_scale=0.35)
        Hc, Wc = raw_frame.shape[:2]
        bbox_w, bbox_h = x1 - x0, y1 - y0

    # Stage 1: raw frame tanpa bbox
    axes[0].imshow(raw_frame)
    # "REC ●" indicator di kiri-atas
    axes[0].text(Wc * 0.02, Hc * 0.06, "● REC", color="#ee2222",
                 fontsize=9, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.2",
                           facecolor="black", alpha=0.55, edgecolor="none"))
    # Resolution annotation di kanan-atas
    res_text = f"{Wc}×{Hc}  30fps" if use_real_raw else "1920×1080  30fps"
    axes[0].text(Wc * 0.98, Hc * 0.06, res_text,
                 color="#dddddd", fontsize=8, ha="right",
                 fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.2",
                           facecolor="black", alpha=0.55, edgecolor="none"))
    axes[0].set_xticks([]); axes[0].set_yticks([])

    # Stage 2: raw frame + bbox highlighted
    axes[1].imshow(raw_frame)
    rect = mpatches.Rectangle((x0, y0), bbox_w, bbox_h,
                              edgecolor="#ee6666", facecolor="none",
                              linewidth=2.5)
    axes[1].add_patch(rect)
    axes[1].text(x0, max(y0 - 14, 14), "face  conf=0.92",
                 color="#ee6666", fontsize=9, fontweight="bold")
    axes[1].set_xticks([]); axes[1].set_yticks([])

    # === Stage 3: Cropped face 224×224 ===
    axes[2].imshow(face_img)
    axes[2].set_xticks([]); axes[2].set_yticks([])

    # === Stage 4: Landmark extraction ===
    axes[3].imshow(face_img)
    pts = lm.reshape(68, 2) * np.array([W, H])
    # Connection lines (region color)
    connection_groups = [
        list(range(0, 17)), list(range(17, 22)), list(range(22, 27)),
        list(range(27, 31)), list(range(31, 36)),
        list(range(36, 42)) + [36], list(range(42, 48)) + [42],
        list(range(48, 60)) + [48], list(range(60, 68)) + [60],
    ]
    for grp in connection_groups:
        c = color_for_index(grp[0])
        axes[3].plot(pts[grp, 0], pts[grp, 1], "-", color=c,
                     linewidth=0.9, alpha=0.75)
    for i in range(68):
        axes[3].scatter(pts[i, 0], pts[i, 1], s=10,
                        c=color_for_index(i), edgecolors="white",
                        linewidths=0.3, zorder=5)
    axes[3].set_xticks([]); axes[3].set_yticks([])

    # === Stage 5: Heatmap ===
    axes[4].imshow(hm, cmap="hot", vmin=0, vmax=hm.max() if hm.max() > 0 else 1)
    axes[4].set_xticks([]); axes[4].set_yticks([])

    # Stage title & description above each panel
    for i, (ax, title, desc) in enumerate(zip(axes, stage_titles, stage_descs)):
        bbox = ax.get_position()
        x_center = (bbox.x0 + bbox.x1) / 2
        fig.text(x_center, bbox.y1 + 0.025, title,
                 ha="center", va="bottom", fontsize=11, fontweight="bold")
        fig.text(x_center, bbox.y0 - 0.015, desc,
                 ha="center", va="top", fontsize=8.5,
                 color="#444444", style="italic")

    # Arrows between consecutive stages
    for i in range(n_stage - 1):
        draw_arrow(fig, axes[i], axes[i + 1])

    # ── Output annotation row di bawah ───────────────────────────────────────
    fig.text(0.5, 0.04,
             "OUTPUT yang disimpan ke disk per sample:\n"
             "• RGB image  (224×224×3)   → input CNN / image branch\n"
             "• Landmark raw_136  (68×2) — 2 source paralel: face-api.js (FA) + MediaPipe (MP)\n"
             "  → derive ke FACS_28 (Euclidean dist), Blendshape_52 (MP only), atau FACS+BS_80\n"
             "• Heatmap  (224×224)  → channel ke-4 input Early Fusion",
             ha="center", va="top", fontsize=9, color="#222222",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f4f8",
                       edgecolor="#aaaaaa", linewidth=0.8))

    # Main title
    fig.suptitle("Data Preprocessing Pipeline — Primer dataset (frontonly_conf60)",
                 fontsize=14, fontweight="bold", y=0.99)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "preprocessing_pipeline.png"
    out_pdf = OUT_DIR / "preprocessing_pipeline.pdf"
    plt.savefig(out_png, dpi=args.dpi, bbox_inches="tight", pad_inches=0.15,
                facecolor="white")
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    plt.close(fig)
    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
