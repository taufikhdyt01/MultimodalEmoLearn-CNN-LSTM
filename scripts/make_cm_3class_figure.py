"""
Generate confusion matrix figure for Late Fusion TL 3-class B3 (juara val-tuned).

Input:  models/frontonly_conf60/3class/all_results_3class.json
Output: docs/figures/confusion_matrix_3class.{pdf,png}

Single panel: 3×3 CM dengan annotations per-class precision/recall/F1.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_JSON = PROJECT_ROOT / 'models' / 'frontonly_conf60' / '3class' / 'all_results_3class.json'
FIG_DIR = PROJECT_ROOT / 'docs' / 'figures'
EMOTIONS_3 = ['positive', 'neutral', 'negative']


def compute_per_class(cm):
    """Compute precision, recall, F1 per class from confusion matrix."""
    cm = np.array(cm, dtype=np.float64)
    per_class = {}
    for i, cls in enumerate(EMOTIONS_3):
        tp = cm[i, i]
        p_denom = cm[:, i].sum()
        r_denom = cm[i, :].sum()
        precision = tp / p_denom if p_denom > 0 else 0.0
        recall = tp / r_denom if r_denom > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[cls] = {
            'precision': precision, 'recall': recall, 'f1': f1,
            'support': int(r_denom),
        }
    return per_class


def plot_cm(ax, cm, labels, title, normalize='raw'):
    cm = np.array(cm, dtype=np.float64)
    raw_cm = cm.copy().astype(int)

    if normalize == 'row':
        row_sum = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum > 0)
        fmt = '.2f'
        vmax = 1.0
    else:
        fmt = 'd'
        vmax = cm.max()

    im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=vmax)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('True', fontsize=10)
    ax.set_title(title, fontsize=11, pad=8)

    threshold = vmax * 0.6
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm[i, j]
            disp = f'{val:{fmt}}' if normalize == 'row' else f'{raw_cm[i, j]}'
            color = 'white' if val > threshold else 'black'
            ax.text(j, i, disp, ha='center', va='center', fontsize=10, color=color)

    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dpi', type=int, default=300)
    ap.add_argument('--normalize', choices=['raw', 'row'], default='raw')
    ap.add_argument('--no-pdf', action='store_true')
    ap.add_argument('--config', default='Late_Fusion_TL_B3',
                    help='Key di all_results_3class.json (default: juara Late_Fusion_TL_B3)')
    args = ap.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if not RESULTS_JSON.exists():
        raise FileNotFoundError(
            f'{RESULTS_JSON}\nRun nb 79 di VPS dulu untuk generate results.')

    all_results = json.load(open(RESULTS_JSON))
    if args.config not in all_results:
        raise KeyError(f'{args.config} not in {list(all_results.keys())}')

    r = all_results[args.config]
    cm = r['confusion_matrix']
    macro_f1 = r['test_macro_f1']
    acc = r['test_accuracy']
    weighted_f1 = r['test_weighted_f1']
    val_f1 = r['val_macro_f1']
    w_best = r.get('best_cnn_weight')
    per_class = compute_per_class(cm)

    # Figure: single panel CM + metrics summary
    fig, ax = plt.subplots(1, 1, figsize=(5.0, 4.2))

    arch_name = args.config.replace('_', ' ')
    title = f'3-Class CM — {arch_name}'
    im = plot_cm(ax, cm, EMOTIONS_3, title, normalize=args.normalize)

    cbar_label = 'Count' if args.normalize == 'raw' else 'Proportion'
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)

    plt.tight_layout()

    suffix = '' if args.normalize == 'raw' else f'_{args.normalize}norm'
    png_path = FIG_DIR / f'confusion_matrix_3class{suffix}.png'
    plt.savefig(png_path, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    print(f'Saved: {png_path}')

    if not args.no_pdf:
        pdf_path = FIG_DIR / f'confusion_matrix_3class{suffix}.pdf'
        plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
        print(f'Saved: {pdf_path}')

    # Summary print
    print(f"\n{'='*60}")
    print(f'  {args.config}')
    print(f"{'='*60}")
    w_line = f'  w_best (CNN) = {w_best:.2f}' if w_best is not None else ''
    print(f'  val_macro_f1 = {val_f1:.4f}{w_line}')
    print(f'  test_macro_f1 = {macro_f1:.4f}  acc = {acc:.4f}  weighted_f1 = {weighted_f1:.4f}')
    print(f'\n  Per-class metrics:')
    print(f"    {'class':>10}  {'P':>6}  {'R':>6}  {'F1':>6}  {'support':>8}")
    for cls, m in per_class.items():
        print(f"    {cls:>10}  {m['precision']:>6.3f}  {m['recall']:>6.3f}  {m['f1']:>6.3f}  {m['support']:>8d}")


if __name__ == '__main__':
    main()
