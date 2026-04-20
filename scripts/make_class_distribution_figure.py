"""
Generate Class Distribution Figure for JITeCS Paper (Section 3.1).

Side-by-side bar chart 7-class vs 4-class dataset Primer conf60,
highlighting imbalance ratio (1:1138 di 7-class) untuk justifikasi
penggunaan macro-F1 sebagai metrik utama.

Output:
  docs/figures/class_distribution.pdf   (IEEE paper insert)
  docs/figures/class_distribution.png   (presentation/preview)

Data source: data/dataset_frontonly_conf60/dataset_info.json
  - neutral: 5691, happy: 651, sad: 361,
  - angry: 32, fearful: 5, disgusted: 16, surprised: 39
  - 4-class: negative = angry + fearful + disgusted + surprised = 92

Usage:
    python scripts/make_class_distribution_figure.py
    python scripts/make_class_distribution_figure.py --dpi 300
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Color palette — colorblind-friendly, print-safe
COLOR_MAJOR = '#1f77b4'      # neutral, happy — blue shades
COLOR_MINOR_BG = '#d62728'   # highlight minority class with red
COLOR_MINOR = '#ff7f0e'      # orange for medium minorities
COLOR_GRID = '#e0e0e0'

# Class orderings
EMOTIONS_7 = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised']
EMOTIONS_4 = ['neutral', 'happy', 'sad', 'negative']


def load_counts():
    info_path = PROJECT_ROOT / 'data' / 'dataset_frontonly_conf60' / 'dataset_info.json'
    with open(info_path) as f:
        info = json.load(f)
    dist = info['emotion_distribution']
    counts_7 = [dist[e] for e in EMOTIONS_7]
    # 4-class remap: negative = angry + fearful + disgusted + surprised
    counts_4 = [
        dist['neutral'],
        dist['happy'],
        dist['sad'],
        dist['angry'] + dist['fearful'] + dist['disgusted'] + dist['surprised'],
    ]
    return counts_7, counts_4, info


def format_bar(ax, bars, counts, total, color_threshold_pct=1.0):
    """Annotate bars with count + %. Color minority classes distinctly."""
    for bar, c in zip(bars, counts):
        pct = c / total * 100
        label = f'{c:,}\n({pct:.1f}%)' if c >= 10 else f'{c}\n({pct:.2f}%)'
        # Color: red for <1%, orange for 1-5%, blue for >5%
        if pct < color_threshold_pct:
            bar.set_color(COLOR_MINOR_BG)
        elif pct < 5:
            bar.set_color(COLOR_MINOR)
        else:
            bar.set_color(COLOR_MAJOR)
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                label, ha='center', va='bottom', fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dpi', type=int, default=300)
    ap.add_argument('--no-pdf', action='store_true', help='Skip PDF output')
    args = ap.parse_args()

    counts_7, counts_4, info = load_counts()
    total = info['total_samples']

    # Ratio calc
    ratio_7 = counts_7[0] / min(counts_7)
    ratio_4 = counts_4[0] / min(counts_4)

    # IEEE single-column width ~3.5 in, double-column ~7.16 in. Use 7in × 3.5in.
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.4), sharey=False)

    # ── Left: 7-class ──
    ax1 = axes[0]
    x7 = np.arange(len(EMOTIONS_7))
    bars7 = ax1.bar(x7, counts_7, edgecolor='black', linewidth=0.4)
    format_bar(ax1, bars7, counts_7, total)
    ax1.set_yscale('log')
    ax1.set_xticks(x7)
    ax1.set_xticklabels(EMOTIONS_7, rotation=30, ha='right', fontsize=9)
    ax1.set_ylabel('Number of Samples (log scale)', fontsize=9)
    ax1.set_title('(a) 7-Class', fontsize=10)
    ax1.grid(axis='y', color=COLOR_GRID, linestyle='--', linewidth=0.5, alpha=0.7)
    ax1.set_axisbelow(True)
    ax1.set_ylim(1, max(counts_7) * 3)

    # (Annotation narrative moved to LaTeX figure caption)

    # ── Right: 4-class ──
    ax2 = axes[1]
    x4 = np.arange(len(EMOTIONS_4))
    bars4 = ax2.bar(x4, counts_4, edgecolor='black', linewidth=0.4)
    format_bar(ax2, bars4, counts_4, total)
    ax2.set_yscale('log')
    ax2.set_xticks(x4)
    ax2.set_xticklabels(EMOTIONS_4, rotation=0, fontsize=9)
    ax2.set_ylabel('Number of Samples (log scale)', fontsize=9)
    ax2.set_title('(b) 4-Class (remap)', fontsize=10)
    ax2.grid(axis='y', color=COLOR_GRID, linestyle='--', linewidth=0.5, alpha=0.7)
    ax2.set_axisbelow(True)
    ax2.set_ylim(1, max(counts_4) * 3)

    # (Ratio note moved to LaTeX figure caption)

    # Legend for color coding (sekali saja, di atas figure)
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=COLOR_MAJOR, edgecolor='black', label='Majority (≥5%)'),
        Patch(facecolor=COLOR_MINOR, edgecolor='black', label='Minority (1–5%)'),
        Patch(facecolor=COLOR_MINOR_BG, edgecolor='black', label='Severe minority (<1%)'),
    ]
    fig.legend(handles=legend_handles, loc='lower center',
               ncol=3, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.02))

    # (Suptitle moved to LaTeX figure caption)
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    out_dir = PROJECT_ROOT / 'docs' / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)

    png_path = out_dir / 'class_distribution.png'
    plt.savefig(png_path, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    print(f'Saved: {png_path}')

    if not args.no_pdf:
        pdf_path = out_dir / 'class_distribution.pdf'
        plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
        print(f'Saved: {pdf_path}')

    print(f'\n7-class counts: {dict(zip(EMOTIONS_7, counts_7))}')
    print(f'4-class counts: {dict(zip(EMOTIONS_4, counts_4))}')
    print(f'Total: {total}')
    print(f'\nImbalance ratios:')
    print(f'  7-class: majority/minority = {counts_7[0]}/{min(counts_7)} = 1:{int(ratio_7)}')
    print(f'  4-class: majority/minority = {counts_4[0]}/{min(counts_4)} = 1:{int(ratio_4)}')


if __name__ == '__main__':
    main()
