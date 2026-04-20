"""
Draft Pipeline Overview Figure for JITeCS Paper (Fig 1).

⚠️  DRAFT/REFERENCE ONLY — finalisasi manual di PowerPoint.

Diagram ini berfungsi sebagai:
  1. Struktur referensi dengan visualisasi real data (face, landmark, heatmap)
  2. Daftar informasi yang harus ada di figure final
  3. Sanity check konten (54 configs, 5 arch, 3 scenarios, etc.)

Untuk paper: import draft ini ke PowerPoint/Inkscape, lalu:
  - Pertahankan struktur + labels
  - Refine alignment arrow manual untuk presisi
  - Adjust warna sesuai style IEEE (saat ini pastel colored — ganti muted kalau perlu)

Stages ditampilkan:
  (1) Input & Extraction — video frames → face detection → face crop
  (2) Preprocess + Data — landmark overlay, image 224×224×3, heatmap 224×224
  (3) 5 Fusion Architectures — CNN, FCNN, Early, Intermediate, Late
  (4) Training + Evaluation — optimizer spec + 4 metrics

Output:
  docs/figures/pipeline_overview.pdf   (draft reference)
  docs/figures/pipeline_overview.png   (draft preview)

Usage:
    python scripts/make_pipeline_figure.py
    python scripts/make_pipeline_figure.py --sample-idx 2523  # happy
    python scripts/make_pipeline_figure.py --anonymize
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Academic palette — navy + deep teal accent + warm highlight (IEEE journal friendly)
NAVY = '#1F3A5F'           # primary (structure, borders, labels)
NAVY_LIGHT = '#D6DEE9'     # light navy fill
# Multimodal accent = navy (monochrome scheme — user preference)
TEAL = NAVY                # alias untuk kemudahan code
TEAL_LIGHT = NAVY_LIGHT
CORAL = '#E76F51'          # annotation highlight (bbox, landmark dots)
GRAY_DARK = '#222222'      # body text
GRAY_MED = '#555555'       # secondary text
GRAY_LIGHT = '#888888'     # muted borders
WHITE_BG = '#FFFFFF'
OFFWHITE = '#F5F5F5'

# Legacy vars aliased to navy palette (no hue variation between stages)
C_INPUT = NAVY
C_PRE = NAVY
C_ARCH = NAVY
C_TRAIN = NAVY
C_EVAL = NAVY
C_BG_INPUT = OFFWHITE
C_BG_PRE = OFFWHITE
C_BG_ARCH = OFFWHITE
C_BG_TRAIN = OFFWHITE
C_BG_EVAL = OFFWHITE
C_ARROW = GRAY_DARK

IMG_SIZE = 224


def gaussian_heatmap(landmarks_136, img_size=IMG_SIZE, sigma=3.0):
    y_grid, x_grid = np.ogrid[:img_size, :img_size]
    coords = landmarks_136.reshape(-1, 2)
    heatmap = np.zeros((img_size, img_size), dtype=np.float32)
    denom = 2.0 * sigma * sigma
    for x_norm, y_norm in coords:
        cx = x_norm * img_size
        cy = y_norm * img_size
        g = np.exp(-((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / denom)
        heatmap = np.maximum(heatmap, g.astype(np.float32))
    return heatmap


def blur_face(img, sigma=8):
    try:
        from scipy.ndimage import gaussian_filter
        return np.stack([gaussian_filter(img[..., c], sigma=sigma) for c in range(3)], axis=-1)
    except ImportError:
        return img


def fig_arrow(fig, xy_from, xy_to, color=C_ARROW, lw=1.3):
    arr = FancyArrowPatch(xy_from, xy_to,
                           transform=fig.transFigure,
                           arrowstyle='-|>', mutation_scale=15,
                           color=color, linewidth=lw,
                           connectionstyle='arc3,rad=0')
    fig.patches.append(arr)


def mid_y(ax):
    bb = ax.get_position()
    return (bb.y0 + bb.y1) / 2


def top_y(ax):
    return ax.get_position().y1


def bottom_y(ax):
    return ax.get_position().y0


def stage_label(fig, y, label, color):
    fig.text(0.07, y, label, ha='center', va='center', fontsize=8.5,
             fontweight='bold', color=color, style='italic')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dpi', type=int, default=300)
    ap.add_argument('--sample-idx', type=int, default=2523,
                    help='Default 2523 = happy (more expressive)')
    ap.add_argument('--anonymize', action='store_true')
    ap.add_argument('--no-pdf', action='store_true')
    args = ap.parse_args()

    data_dir = PROJECT_ROOT / 'data' / 'dataset_frontonly_conf60'
    X = np.load(data_dir / 'X_train_images.npy')
    L = np.load(data_dir / 'X_train_landmarks.npy')
    y = np.load(data_dir / 'y_train.npy')
    emotions = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised']

    idx = min(args.sample_idx, len(X) - 1)
    sample_img = np.clip(X[idx], 0, 1)
    if args.anonymize:
        sample_img = np.clip(blur_face(sample_img, sigma=6), 0, 1)
    sample_lm = L[idx]
    sample_label = emotions[int(y[idx])]
    heatmap = gaussian_heatmap(sample_lm, sigma=3.0)
    coords = sample_lm.reshape(-1, 2) * IMG_SIZE

    fig = plt.figure(figsize=(7.16, 8.6))
    gs = gridspec.GridSpec(
        4, 12, figure=fig,
        height_ratios=[1.7, 1.7, 1.35, 2.1],
        hspace=0.7, wspace=0.3,
        left=0.17, right=0.98, top=0.97, bottom=0.03)

    # ═══ STAGE 1: Input & Face Extraction ═══
    # Panel 1a: Raw frames — stack 3 real face crops dari dataset
    ax_a = fig.add_subplot(gs[0, 0:4])
    ax_a.set_xlim(0, 1); ax_a.set_ylim(0, 1)

    # Pilih 3 sample diverse dari dataset untuk represent "frames"
    rng = np.random.RandomState(42)
    stack_indices = [idx]  # first = main sample
    other_candidates = rng.choice(len(X), size=50, replace=False)
    for cand in other_candidates:
        if len(stack_indices) >= 3:
            break
        if cand != idx:
            stack_indices.append(int(cand))
    stack_imgs = [np.clip(X[i], 0, 1) for i in stack_indices[:3]]
    if args.anonymize:
        stack_imgs = [np.clip(blur_face(im, sigma=6), 0, 1) for im in stack_imgs]

    # Stack with offset — reverse order so main sample is on top
    positions = [(0.32, 0.15), (0.26, 0.22), (0.20, 0.29)]
    sizes = [(0.42, 0.52), (0.42, 0.52), (0.42, 0.52)]
    for i, (img_s, (px, py), (pw, ph)) in enumerate(
            zip(reversed(stack_imgs), reversed(positions), reversed(sizes))):
        # imshow pada specific sub-position menggunakan extent
        ax_a.imshow(img_s, extent=(px, px + pw, py, py + ph),
                    aspect='auto', zorder=2 + i)
        # Border
        border = Rectangle((px, py), pw, ph, fill=False,
                            edgecolor=GRAY_MED, linewidth=0.8, zorder=2 + i + 0.5)
        ax_a.add_patch(border)

    ax_a.set_title('Raw frames', fontsize=8, color=GRAY_DARK, pad=3)
    ax_a.set_xticks([]); ax_a.set_yticks([])
    for s in ax_a.spines.values(): s.set_visible(False)

    # Panel 1b: Face detection bbox overlay
    ax_b = fig.add_subplot(gs[0, 4:8])
    ax_b.imshow(sample_img)
    rect = Rectangle((18, 28), IMG_SIZE - 36, IMG_SIZE - 46,
                      fill=False, edgecolor=CORAL, linewidth=2.5)
    ax_b.add_patch(rect)
    ax_b.set_xticks([]); ax_b.set_yticks([])
    ax_b.set_title('Face detection', fontsize=8, color=GRAY_DARK, pad=3)
    for s in ax_b.spines.values():
        s.set_edgecolor(GRAY_MED); s.set_linewidth(0.8)

    # Panel 1c: Face crop
    ax_c = fig.add_subplot(gs[0, 8:12])
    ax_c.imshow(sample_img)
    ax_c.set_xticks([]); ax_c.set_yticks([])
    ax_c.set_title('Face crop', fontsize=8, color=GRAY_DARK, pad=3)
    for s in ax_c.spines.values():
        s.set_edgecolor(GRAY_MED); s.set_linewidth(0.8)

    # ═══ STAGE 2: Data Representations ═══
    ax_d = fig.add_subplot(gs[1, 0:4])
    ax_d.imshow(sample_img)
    ax_d.scatter(coords[:, 0], coords[:, 1],
                 s=8, c=CORAL, edgecolors='white', linewidths=0.5)
    ax_d.set_xticks([]); ax_d.set_yticks([])
    ax_d.set_title('Landmarks (68)', fontsize=8, color=GRAY_DARK, pad=3)
    ax_d.set_xlim(0, IMG_SIZE); ax_d.set_ylim(IMG_SIZE, 0)
    for s in ax_d.spines.values():
        s.set_edgecolor(GRAY_MED); s.set_linewidth(0.8)

    ax_e = fig.add_subplot(gs[1, 4:8])
    ax_e.imshow(sample_img)
    ax_e.set_xticks([]); ax_e.set_yticks([])
    ax_e.set_title('Image (224×224×3)', fontsize=8, color=GRAY_DARK, pad=3)
    for s in ax_e.spines.values():
        s.set_edgecolor(GRAY_MED); s.set_linewidth(0.8)

    ax_f = fig.add_subplot(gs[1, 8:12])
    # Custom cmap white→coral biar matching dengan landmark dots (coral)
    coral_cmap = LinearSegmentedColormap.from_list(
        'coral_heat', ['white', '#FDE4D9', CORAL, '#A8401F'])
    ax_f.imshow(heatmap, cmap=coral_cmap, vmin=0, vmax=1)
    ax_f.set_xticks([]); ax_f.set_yticks([])
    ax_f.set_title('Heatmap (224×224)', fontsize=8, color=GRAY_DARK, pad=3)
    for s in ax_f.spines.values():
        s.set_edgecolor(GRAY_MED); s.set_linewidth(0.8)

    # ═══ STAGE 3: 5 Architectures ═══
    ax_arch = fig.add_subplot(gs[2, :])
    ax_arch.set_xlim(0, 10); ax_arch.set_ylim(0, 4)
    ax_arch.axis('off')

    # Opsi B + C: subtle fill (multimodal = navy_light) + fusion point indicator
    #   fusion_pos: None=single modal, 0=input, 1=feature, 2=decision
    arch_defs = [
        ('CNN',          None),
        ('FCNN',         None),
        ('Early Fusion', 0),
        ('Intermediate', 1),
        ('Late Fusion',  2),
    ]
    bw = 1.82; bh = 2.3; by = 0.8

    def fusion_indicator(cx, cy, fusion_pos, bar_w=1.1):
        """3-dot indicator: ●─○─○ (filled = fusion position, teal accent)."""
        xs = [cx - bar_w / 2, cx, cx + bar_w / 2]
        ax_arch.plot([xs[0], xs[2]], [cy, cy], color=GRAY_MED, linewidth=0.7, zorder=1)
        for i_pos, x in enumerate(xs):
            filled = (i_pos == fusion_pos)
            circ = plt.Circle((x, cy), 0.075,
                               facecolor=TEAL if filled else WHITE_BG,
                               edgecolor=NAVY, linewidth=0.9, zorder=2)
            ax_arch.add_patch(circ)
        active_label = ['input', 'feature', 'decision'][fusion_pos]
        ax_arch.text(xs[fusion_pos], cy - 0.24, active_label,
                      ha='center', va='top', fontsize=7,
                      fontweight='bold', color=TEAL)

    for i, (name, fpos) in enumerate(arch_defs):
        bx = 0.25 + i * 1.95
        is_multimodal = fpos is not None
        rect = FancyBboxPatch((bx, by), bw, bh,
                               boxstyle='round,pad=0.03,rounding_size=0.1',
                               facecolor=TEAL_LIGHT if is_multimodal else WHITE_BG,
                               edgecolor=TEAL if is_multimodal else NAVY,
                               linewidth=1.0)
        ax_arch.add_patch(rect)
        # Name (teal for multimodal, navy for single)
        ax_arch.text(bx + bw / 2, by + bh - 0.4, name,
                     ha='center', va='center', fontsize=8.5,
                     fontweight='bold',
                     color=TEAL if is_multimodal else NAVY)
        # Indicator or single-modality label
        cy_ind = by + bh / 2 - 0.35
        if is_multimodal:
            fusion_indicator(bx + bw / 2, cy_ind, fpos)
        else:
            ax_arch.text(bx + bw / 2, cy_ind, 'single modality',
                          ha='center', va='center', fontsize=7,
                          style='italic', color=GRAY_MED)

    # ═══ STAGE 4: Training + Evaluation ═══
    ax_te = fig.add_subplot(gs[3, :])
    ax_te.set_xlim(0, 10); ax_te.set_ylim(0, 2)
    ax_te.axis('off')

    # ── Training box (enlarged) dengan 2×3 config matrix ──
    tr_rect = FancyBboxPatch((0.15, 0.1), 4.55, 1.85,
                              boxstyle='round,pad=0.05,rounding_size=0.1',
                              facecolor=WHITE_BG, edgecolor=NAVY, linewidth=1.0)
    ax_te.add_patch(tr_rect)
    ax_te.text(0.4, 1.75, 'Training', ha='left', va='center',
               fontsize=10, fontweight='bold', color=NAVY)

    # Matrix 2×3: columns = backbones, rows = scenarios
    mat_x0, mat_y0 = 1.55, 0.35
    col_w, row_h = 1.45, 0.38
    # Column headers
    ax_te.text(mat_x0 + col_w / 2, mat_y0 + 3 * row_h + 0.08, 'scratch',
               ha='center', va='bottom', fontsize=7.5, color=NAVY, fontweight='bold')
    ax_te.text(mat_x0 + col_w + col_w / 2, mat_y0 + 3 * row_h + 0.08, 'ResNet18 TL',
               ha='center', va='bottom', fontsize=7.5, color=NAVY, fontweight='bold')
    # Row headers + cells (B1 at top, B3 at bottom)
    scenarios = ['B3', 'B2', 'B1']  # bottom-up
    for r, sc in enumerate(scenarios):
        cy = mat_y0 + r * row_h
        ax_te.text(mat_x0 - 0.2, cy + row_h / 2, sc,
                    ha='right', va='center', fontsize=8,
                    color=NAVY, fontweight='bold')
        for c in range(2):
            cx = mat_x0 + c * col_w
            cell = Rectangle((cx + 0.04, cy + 0.04),
                              col_w - 0.08, row_h - 0.08,
                              facecolor=TEAL_LIGHT, edgecolor=TEAL, linewidth=0.6)
            ax_te.add_patch(cell)

    # ── Evaluation box (enlarged) ──
    ev_rect = FancyBboxPatch((5.2, 0.1), 4.65, 1.85,
                              boxstyle='round,pad=0.05,rounding_size=0.1',
                              facecolor=WHITE_BG, edgecolor=NAVY, linewidth=1.0)
    ax_te.add_patch(ev_rect)
    ax_te.text(5.45, 1.75, 'Evaluation', ha='left', va='center',
               fontsize=10, fontweight='bold', color=NAVY)

    # Mini confusion matrix icon (4×4 grid) — larger
    cm_x0, cm_y0 = 5.5, 0.55
    cm_side = 1.1
    cell_s = cm_side / 4
    for r in range(4):
        for c in range(4):
            cx = cm_x0 + c * cell_s
            cy = cm_y0 + (3 - r) * cell_s
            is_diag = (r == c)
            alpha = 0.9 if is_diag else (0.25 if abs(r - c) == 1 else 0.10)
            cell = Rectangle((cx, cy), cell_s * 0.9, cell_s * 0.9,
                              facecolor=TEAL if is_diag else NAVY,
                              alpha=alpha,
                              edgecolor='white', linewidth=0.4)
            ax_te.add_patch(cell)
    ax_te.text(cm_x0 + cm_side / 2, cm_y0 - 0.12,
               'confusion matrix',
               ha='center', va='top', fontsize=7, style='italic', color=GRAY_MED)

    # 4 metric pills — stack 2×2 supaya pills lebih lebar (tidak mepet)
    # Kasih right-padding biar tidak menempel ke box edge
    metrics = ['Macro F1', 'Micro F1', 'Weighted F1', 'Per-class']
    pill_x0 = 6.85
    pill_w, pill_h = 1.28, 0.42
    pill_gap_x = 0.10
    # Row 1 (atas) & Row 2 (bawah)
    positions = [
        (pill_x0, 1.20),                                # Macro F1
        (pill_x0 + pill_w + pill_gap_x, 1.20),          # Micro F1
        (pill_x0, 0.65),                                # Weighted F1
        (pill_x0 + pill_w + pill_gap_x, 0.65),          # Per-class
    ]
    for m, (mx, my) in zip(metrics, positions):
        bubble = FancyBboxPatch((mx, my), pill_w, pill_h,
                                 boxstyle='round,pad=0.02,rounding_size=0.08',
                                 facecolor='white', edgecolor=NAVY, linewidth=0.9)
        ax_te.add_patch(bubble)
        ax_te.text(mx + pill_w / 2, my + pill_h / 2, m,
                   ha='center', va='center', fontsize=7.5, color=GRAY_DARK)

    # Force draw to compute axes positions
    fig.canvas.draw()

    # ═══ Position-aware sidebar labels (tanpa arrows — add manual di PowerPoint) ═══
    stage_label(fig, mid_y(ax_a), 'Input\n&\nExtraction', C_INPUT)
    stage_label(fig, mid_y(ax_d), 'Data\nRepresentations', C_PRE)
    stage_label(fig, mid_y(ax_arch), '5 Fusion\nArchitectures', C_ARCH)
    stage_label(fig, mid_y(ax_te), 'Train\n+ Eval', C_TRAIN)

    out_dir = PROJECT_ROOT / 'docs' / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / 'pipeline_overview.png'
    plt.savefig(png_path, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    print(f'Saved: {png_path}')
    if not args.no_pdf:
        pdf_path = out_dir / 'pipeline_overview.pdf'
        plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
        print(f'Saved: {pdf_path}')
    print(f'\nSample: idx={idx}, label={sample_label}, anonymize={args.anonymize}')


if __name__ == '__main__':
    main()
