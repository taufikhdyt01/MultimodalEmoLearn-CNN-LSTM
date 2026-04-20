"""
Generate Confusion Matrix Figures for JITeCS Paper (Fig 6).

Side-by-side 7-class dan 4-class confusion matrix dari best config:
  - 7-class: Late Fusion TL B1 (Macro F1 = 0.301)
  - 4-class: Late Fusion TL B3 (Macro F1 = 0.567) — best overall

Input: hasil dari `scripts/export_new_best_predictions.py` (run di VPS):
  models/frontonly_conf60/predictions/best_7c_early_fusion_tl_b3.json
  models/frontonly_conf60/predictions/best_4c_intermediate_tl_b3.json

(Legacy: best_7c_late_fusion_tl_b1.json / best_4c_late_fusion_tl_b3.json
 juga masih didukung via --legacy flag, untuk compat paper draft lama.)

Output:
  docs/figures/confusion_matrix.pdf   (IEEE paper)
  docs/figures/confusion_matrix.png   (preview)

Usage:
    python scripts/make_confusion_matrix_figure.py
    python scripts/make_confusion_matrix_figure.py --normalize row
    python scripts/make_confusion_matrix_figure.py --dpi 300
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PRED_DIR = PROJECT_ROOT / 'models' / 'frontonly_conf60' / 'predictions'
FIG_DIR = PROJECT_ROOT / 'docs' / 'figures'


def load_predictions(json_path):
    if not json_path.exists():
        raise FileNotFoundError(
            f'{json_path}\n'
            f'Run scripts/export_best_predictions.py di VPS dulu untuk generate file ini.')
    with open(json_path) as f:
        return json.load(f)


def plot_cm(ax, cm, labels, title, normalize=None):
    """Draw single confusion matrix heatmap on given axes."""
    cm = np.array(cm, dtype=np.float64)
    raw_cm = cm.copy().astype(int)

    if normalize == 'row':  # per-row normalisasi (recall per true class)
        row_sum = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum > 0)
        fmt = '.2f'
        vmax = 1.0
    elif normalize == 'col':
        col_sum = cm.sum(axis=0, keepdims=True)
        cm = np.divide(cm, col_sum, out=np.zeros_like(cm), where=col_sum > 0)
        fmt = '.2f'
        vmax = 1.0
    else:  # raw count
        fmt = 'd'
        vmax = cm.max()

    im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=vmax)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Predicted', fontsize=9)
    ax.set_ylabel('True', fontsize=9)
    ax.set_title(title, fontsize=10, pad=6)

    # Annotate cells
    threshold = vmax * 0.6
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm[i, j]
            display = f'{val:{fmt}}' if normalize else f'{raw_cm[i, j]}'
            color = 'white' if val > threshold else 'black'
            ax.text(j, i, display, ha='center', va='center',
                    fontsize=7, color=color)

    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dpi', type=int, default=300)
    ap.add_argument('--normalize', choices=['raw', 'row', 'col'], default='raw',
                    help='Normalize per row (recall), per col (precision), or raw counts')
    ap.add_argument('--no-pdf', action='store_true')
    ap.add_argument('--test-tuned', action='store_true',
                    help='(Legacy Late Fusion only) Pakai grid-search w di test. '
                         'Default: val-tuned (proper out-of-sample).')
    ap.add_argument('--legacy', action='store_true',
                    help='Pakai Late Fusion predictions lama (pre val-tuning fix).')
    args = ap.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load predictions — default pakai NEW best (Intermediate TL 4c + Early Fusion TL 7c)
    if args.legacy:
        p7 = load_predictions(PRED_DIR / 'best_7c_late_fusion_tl_b1.json')
        p4 = load_predictions(PRED_DIR / 'best_4c_late_fusion_tl_b3.json')
    else:
        p7 = load_predictions(PRED_DIR / 'best_7c_early_fusion_tl_b3.json')
        p4 = load_predictions(PRED_DIR / 'best_4c_intermediate_tl_b3.json')

    def pick(p):
        """Return (cm, metrics, w_or_none).
        - New models (Intermediate/Early Fusion): no fusion weight, w=None
        - Legacy Late Fusion: has best_cnn_tl_weight + optional test_tuned variant
        """
        if args.test_tuned and 'test_tuned' in p:
            t = p['test_tuned']
            return t['confusion_matrix'], t['test_metrics'], t.get('best_cnn_tl_weight')
        return p['confusion_matrix'], p['test_metrics'], p.get('best_cnn_tl_weight')

    cm7, m7, w7 = pick(p7)
    cm4, m4, w4 = pick(p4)

    if args.legacy:
        tune_label = '(test-tuned w — paper-compat)' if args.test_tuned else '(val-tuned w — proper)'
    else:
        tune_label = '(val-tuned proper — new best models)'

    # Figure: 2 panels side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.6),
                              gridspec_kw={'width_ratios': [1.3, 1]})

    arch7 = p7.get('architecture', 'Late Fusion TL B1')
    sc7 = p7.get('scenario', 'b1').upper()
    w7_str = f'  w={w7:.2f}' if w7 is not None else ''
    title_7 = (f"(a) 7-Class — {arch7} {sc7}\n"
               f"Macro F1={m7['macro_f1']:.3f}  Acc={m7['accuracy']:.3f}{w7_str}")
    im7 = plot_cm(axes[0], cm7, p7['emotions'],
                  title_7, normalize=None if args.normalize == 'raw' else args.normalize)

    arch4 = p4.get('architecture', 'Late Fusion TL B3')
    sc4 = p4.get('scenario', 'b3').upper()
    w4_str = f'  w={w4:.2f}' if w4 is not None else ''
    title_4 = (f"(b) 4-Class — {arch4} {sc4} (best)\n"
               f"Macro F1={m4['macro_f1']:.3f}  Acc={m4['accuracy']:.3f}{w4_str}")
    im4 = plot_cm(axes[1], cm4, p4['emotions'],
                  title_4, normalize=None if args.normalize == 'raw' else args.normalize)

    fig.suptitle(f'Confusion Matrix {tune_label}', fontsize=9, y=1.01)

    # Colorbar shared (only on right panel untuk compactness)
    cbar_label = 'Count' if args.normalize == 'raw' else 'Proportion'
    fig.colorbar(im4, ax=axes[1], fraction=0.046, pad=0.04, label=cbar_label)

    plt.tight_layout()

    suffix = '' if args.normalize == 'raw' else f'_{args.normalize}norm'
    if args.test_tuned:
        suffix += '_testtuned'
    png_path = FIG_DIR / f'confusion_matrix{suffix}.png'
    plt.savefig(png_path, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    print(f'Saved: {png_path}')

    if not args.no_pdf:
        pdf_path = FIG_DIR / f'confusion_matrix{suffix}.pdf'
        plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
        print(f'Saved: {pdf_path}')

    # Ringkas summary
    print(f'\n7-class ({p7["tag"]}):')
    print(f'  w(CNN_TL) = {p7["best_cnn_tl_weight"]:.2f}  Macro F1 = {m7["macro_f1"]:.4f}')
    print(f'  Per-class F1:')
    for emo in p7['emotions']:
        if emo in p7['classification_report']:
            r = p7['classification_report'][emo]
            print(f'    {emo:>10}: P={r["precision"]:.3f}  R={r["recall"]:.3f}  F1={r["f1-score"]:.3f}  support={int(r["support"])}')

    print(f'\n4-class ({p4["tag"]}):')
    print(f'  w(CNN_TL) = {p4["best_cnn_tl_weight"]:.2f}  Macro F1 = {m4["macro_f1"]:.4f}')
    print(f'  Per-class F1:')
    for emo in p4['emotions']:
        if emo in p4['classification_report']:
            r = p4['classification_report'][emo]
            print(f'    {emo:>10}: P={r["precision"]:.3f}  R={r["recall"]:.3f}  F1={r["f1-score"]:.3f}  support={int(r["support"])}')


if __name__ == '__main__':
    main()
