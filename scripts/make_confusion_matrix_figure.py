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
    ap.add_argument('--include-4class', action='store_true',
                    help='Tambah panel 4-class (ablation). Default: skip — paper pakai 7c + 3c saja.')
    args = ap.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load 7-class predictions
    if args.legacy:
        p7 = load_predictions(PRED_DIR / 'best_7c_late_fusion_tl_b1.json')
    else:
        p7 = load_predictions(PRED_DIR / 'best_7c_early_fusion_tl_b3.json')

    # Load 3-class from all_results_3class.json (nb 79) — Late Fusion TL B3 juara
    p3 = None
    r3_path = PROJECT_ROOT / 'models' / 'frontonly_conf60' / '3class' / 'all_results_3class.json'
    if r3_path.exists():
        all3 = json.load(open(r3_path))
        r3 = all3.get('Late_Fusion_TL_B3')
        if r3 is not None:
            p3 = {
                'tag': 'Late Fusion TL 3c B3 (juara val-tuned)',
                'emotions': ['positive', 'neutral', 'negative'],
                'confusion_matrix': r3['confusion_matrix'],
                'test_metrics': {
                    'macro_f1': r3['test_macro_f1'],
                    'accuracy': r3['test_accuracy'],
                    'weighted_f1': r3['test_weighted_f1'],
                },
                'best_cnn_tl_weight': r3.get('best_cnn_weight'),
            }
    if p3 is None:
        raise FileNotFoundError(f'{r3_path} missing atau Late_Fusion_TL_B3 tidak ada — run nb 79 dulu')

    # Optional 4-class (ablation)
    p4 = None
    if args.include_4class:
        if args.legacy:
            p4 = load_predictions(PRED_DIR / 'best_4c_late_fusion_tl_b3.json')
        else:
            p4 = load_predictions(PRED_DIR / 'best_4c_intermediate_tl_b3.json')

    def pick(p):
        """Return (cm, metrics, w_or_none)."""
        if args.test_tuned and 'test_tuned' in p:
            t = p['test_tuned']
            return t['confusion_matrix'], t['test_metrics'], t.get('best_cnn_tl_weight')
        return p['confusion_matrix'], p['test_metrics'], p.get('best_cnn_tl_weight')

    cm7, m7, w7 = pick(p7)
    cm3, m3, w3 = p3['confusion_matrix'], p3['test_metrics'], p3.get('best_cnn_tl_weight')

    if args.legacy:
        tune_label = '(test-tuned w — paper-compat)' if args.test_tuned else '(val-tuned w — proper)'
    else:
        tune_label = '(val-tuned proper — new best models)'

    # Figure: default 2-panel (7c + 3c). With --include-4class: 3-panel
    if p4 is not None:
        cm4, m4, w4 = pick(p4)
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6),
                                  gridspec_kw={'width_ratios': [1.3, 1, 0.85]})
        im7 = plot_cm(axes[0], cm7, p7['emotions'], '(a) 7-Class',
                      normalize=None if args.normalize == 'raw' else args.normalize)
        im4 = plot_cm(axes[1], cm4, p4['emotions'], '(b) 4-Class',
                      normalize=None if args.normalize == 'raw' else args.normalize)
        im3 = plot_cm(axes[2], cm3, p3['emotions'], '(c) 3-Class',
                      normalize=None if args.normalize == 'raw' else args.normalize)
        im_last = im3
    else:
        fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6),
                                  gridspec_kw={'width_ratios': [1.3, 0.85]})
        im7 = plot_cm(axes[0], cm7, p7['emotions'], '(a) 7-Class',
                      normalize=None if args.normalize == 'raw' else args.normalize)
        im3 = plot_cm(axes[1], cm3, p3['emotions'], '(b) 3-Class',
                      normalize=None if args.normalize == 'raw' else args.normalize)
        im_last = im3

    # Colorbar shared (on rightmost panel)
    cbar_label = 'Count' if args.normalize == 'raw' else 'Proportion'
    fig.colorbar(im_last, ax=axes[-1], fraction=0.046, pad=0.04, label=cbar_label)

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
    def print_summary(p, m, w, label):
        print(f'\n{label} ({p["tag"]}):')
        w_line = f'w(CNN_TL) = {w:.2f}  ' if w is not None else ''
        print(f'  {w_line}Macro F1 = {m["macro_f1"]:.4f}')
        if 'classification_report' in p:
            print(f'  Per-class F1:')
            for emo in p['emotions']:
                if emo in p['classification_report']:
                    r = p['classification_report'][emo]
                    print(f'    {emo:>10}: P={r["precision"]:.3f}  R={r["recall"]:.3f}  '
                          f'F1={r["f1-score"]:.3f}  support={int(r["support"])}')
        else:
            # Derive per-class from CM (for 3c where classification_report wasn't saved)
            import numpy as _np
            cm = _np.array(p['confusion_matrix'], dtype=float)
            print(f'  Per-class F1 (from CM):')
            for i, emo in enumerate(p['emotions']):
                tp = cm[i, i]
                p_den = cm[:, i].sum()
                r_den = cm[i, :].sum()
                prec = tp / p_den if p_den > 0 else 0
                rec  = tp / r_den if r_den > 0 else 0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
                print(f'    {emo:>10}: P={prec:.3f}  R={rec:.3f}  '
                      f'F1={f1:.3f}  support={int(r_den)}')

    print_summary(p7, m7, w7, '7-class')
    print_summary(p3, p3['test_metrics'], p3.get('best_cnn_tl_weight'), '3-class')
    if p4 is not None:
        cm4_, m4_, w4_ = pick(p4)
        print_summary(p4, m4_, w4_, '4-class (ablation)')


if __name__ == '__main__':
    main()
