"""
End-to-end pipeline diagram untuk tesis (Bab 3 Metodologi, overview gambar).

Alur: Input → Preprocessing → Feature representations →
      5 model paths (2 unimodal + 3 fusion) → Output classifier.

Output:
  docs/figures/pipeline_diagram_e2e.png
  docs/figures/pipeline_diagram_e2e.pdf

Usage:
    python scripts/make_pipeline_diagram_e2e.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "docs" / "figures"

# Palette
C_INPUT  = "#dcdcdc"   # gray
C_PREP   = "#cfe4f4"   # light blue
C_FEAT   = "#b9e5b8"   # light green
C_MODEL  = "#fde4a0"   # light yellow
C_FUSION = "#ffc8a0"   # light orange
C_OUTPUT = "#f7b8b8"   # light red
C_TEXT   = "#222222"
C_ARROW  = "#444444"


def add_box(ax, x, y, w, h, text, color, fontsize=9, fontweight="normal",
            zorder=2, edge="#666666", linewidth=0.9, padding=0.18):
    """Tambahkan rounded box dengan text di tengah."""
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad={padding},rounding_size=0.05",
                         linewidth=linewidth, edgecolor=edge,
                         facecolor=color, zorder=zorder)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=C_TEXT, fontweight=fontweight,
            zorder=zorder + 1, wrap=True)


def add_arrow(ax, x0, y0, x1, y1, color=C_ARROW, lw=1.3,
              style="-|>", mutation=15, zorder=1):
    arrow = FancyArrowPatch((x0, y0), (x1, y1),
                            arrowstyle=style, mutation_scale=mutation,
                            color=color, linewidth=lw, zorder=zorder)
    ax.add_patch(arrow)


def main():
    fig, ax = plt.subplots(figsize=(16, 11))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.axis("off")

    # =========================================================================
    # SECTION 1 (atas): INPUT + PREPROCESSING
    # =========================================================================
    add_box(ax, 2, 88, 14, 8,
            "Raw video frame\n(webcam, 1920×1080)",
            C_INPUT, fontsize=9, fontweight="bold")
    add_box(ax, 22, 87, 28, 10,
            "Preprocessing\n• Face detection (MediaPipe, conf ≥ 0.6)\n"
            "• Crop & resize → 224×224\n"
            "• Landmark extraction (MP + FA)\n"
            "• Heatmap rendering (Gaussian per landmark)",
            C_PREP, fontsize=8, fontweight="bold")
    add_arrow(ax, 16, 92, 22, 92, lw=1.6)

    # =========================================================================
    # SECTION 2: 6 FEATURE REPRESENTATIONS (row di tengah-atas)
    # =========================================================================
    feat_y = 70
    feat_h = 7
    feat_w = 13
    feat_spacing = 14.5
    feat_x0 = 4
    features = [
        ("RGB image\n(224×224×3)", "#3b7dd8"),
        ("raw_136\n(68×2 coords)", "#5470c6"),
        ("FACS_28\n(Euclidean dist)", "#91cc75"),
        ("Blendshape_52\n(MP only)", "#fac858"),
        ("FACS+BS_80\n(concat)", "#1a7f37"),
        ("Heatmap 224×224\n(landmark blob)", "#9b59b6"),
    ]
    feat_centers = []
    for i, (label, edge) in enumerate(features):
        x = feat_x0 + i * feat_spacing
        add_box(ax, x, feat_y, feat_w, feat_h, label, C_FEAT,
                fontsize=8, fontweight="bold", edge=edge, linewidth=1.4)
        feat_centers.append(x + feat_w / 2)

    # Arrow from preprocessing block to feature row (one wide bracket)
    # Center of preproc block
    px = 22 + 28 / 2
    py = 87
    # branch line down
    add_arrow(ax, px, py - 0.5, px, feat_y + feat_h + 5, lw=1.6, style="-")
    # horizontal spread to each feature
    spread_y = feat_y + feat_h + 5
    ax.plot([feat_centers[0], feat_centers[-1]], [spread_y, spread_y],
            color=C_ARROW, linewidth=1.6, zorder=1)
    for cx in feat_centers:
        add_arrow(ax, cx, spread_y, cx, feat_y + feat_h + 0.5, lw=1.3)

    # =========================================================================
    # SECTION 3: 5 MODEL PATHS
    # =========================================================================
    model_y = 38
    model_h = 18
    paths = [
        # (x, label, color_edge, used_features (indices into features), description)
        (3, "Unimodal\nImage",      "#ee6666",
         [0],
         "CNN scratch / CNN_TL\n(ResNet-18)"),
        (21, "Unimodal\nLandmark",  "#5470c6",
         [1, 2, 3, 4],
         "FCNN / CNN1D\n(per feature variant)"),
        (39, "Early Fusion",        "#a4262c",
         [0, 5],
         "RGB ⊕ Heatmap\n→ 4-ch → CNN/CNN_TL\n(concat / gated)"),
        (57, "Intermediate Fusion", "#3ba272",
         [0, 1, 2, 3, 4],
         "CNN feat ⊕ FCNN feat\n→ concat → MLP head\n(scratch / TL)"),
        (75, "Late Fusion",         "#9b59b6",
         [0, 1, 2, 3, 4],
         "w·softmax(CNN) +\n(1−w)·softmax(FCNN)\n(scratch / TL × feature)"),
    ]
    path_w = 17
    path_centers = []
    for i, (x, title, edge, feat_idx, desc) in enumerate(paths):
        # Header strip
        add_box(ax, x, model_y + model_h - 4, path_w, 4,
                title, C_MODEL, fontsize=9.5, fontweight="bold",
                edge=edge, linewidth=1.8, padding=0.12)
        # Body
        add_box(ax, x, model_y, path_w, model_h - 4.5, desc,
                "#fff8dc", fontsize=8, fontweight="normal",
                edge=edge, linewidth=1.0, padding=0.18)
        path_centers.append(x + path_w / 2)
        # Connect from features to path (arrows from each used feature)
        for fi in feat_idx:
            src_x = feat_centers[fi]
            src_y = feat_y
            dst_x = x + path_w / 2 + (fi - 2.5) * 1.5  # spread arrows to avoid pure overlap
            dst_y = model_y + model_h
            add_arrow(ax, src_x, src_y - 0.3, src_x, src_y - 4, lw=0.7,
                       color="#888888", style="-")
            ax.plot([src_x, dst_x], [src_y - 4, model_y + model_h + 2],
                    color="#888888", linewidth=0.7, alpha=0.7, zorder=1)
            add_arrow(ax, dst_x, model_y + model_h + 2,
                      dst_x, model_y + model_h + 0.5,
                      lw=0.9, color="#888888")

    # =========================================================================
    # SECTION 4 (bawah): CLASSIFIER + OUTPUT
    # =========================================================================
    cls_y = 12
    cls_w = 30
    cls_h = 10
    add_box(ax, 35, cls_y, cls_w, cls_h,
            "Softmax classifier\n→ 3-class (positive / neutral / negative)\n"
            "→ 7-class (neutral / happy / sad / angry / fearful / disgusted / surprised)",
            C_OUTPUT, fontsize=9, fontweight="bold", edge="#a4262c", linewidth=1.4)

    # Arrows: each model path → softmax
    cls_cx = 35 + cls_w / 2
    cls_top = cls_y + cls_h
    for cx in path_centers:
        ax.plot([cx, cls_cx], [model_y - 0.5, cls_top + 4],
                color=C_ARROW, linewidth=1.1, zorder=1)
    add_arrow(ax, cls_cx, cls_top + 4, cls_cx, cls_top + 0.3, lw=1.4)

    # =========================================================================
    # Section labels (kiri, vertikal)
    # =========================================================================
    section_lbl_x = -1
    for y, txt in [(92, "Input"), (74.5, "Feature\nrepresentation"),
                    (47, "Model\nfamilies"), (17, "Output")]:
        ax.text(section_lbl_x, y, txt, ha="center", va="center",
                fontsize=10, color="#444444", fontweight="bold",
                rotation=90)

    # Title
    fig.suptitle("End-to-End Pipeline — Multimodal Emotion Recognition (CNN + Landmark Fusion)",
                 fontsize=14, fontweight="bold", y=0.97)

    # Legend (footnote at bottom)
    ax.text(50, 3.5,
            "Catatan: Setiap model family (kolom) di-evaluasi pada 2 skema kelas "
            "(3c, 7c) × 3 skenario imbalance (B1/B2/B3). Detail arsitektur per komponen di "
            "src/training/models.py — referensi visual feature representation di "
            "feature_decomposition_overview.png.",
            ha="center", va="center", fontsize=8.5, color="#444444",
            style="italic",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f8f8",
                      edgecolor="#cccccc", linewidth=0.6))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "pipeline_diagram_e2e.png"
    out_pdf = OUT_DIR / "pipeline_diagram_e2e.pdf"
    plt.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.20,
                facecolor="white")
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.20, facecolor="white")
    plt.close(fig)
    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
