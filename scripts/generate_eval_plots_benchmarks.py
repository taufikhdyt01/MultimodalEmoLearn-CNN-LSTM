"""
Generate post-hoc evaluation plots untuk benchmark + cross-dataset.

Sumber:
  models/benchmark/{ckplus,jaffe,rafdb,kdef}/{4c,7c}_results.json   (Skema 1)
  models/benchmark/crossdataset/cross_*.json                         (Skema 2)
  models/benchmark/{ckplus_cv10,jaffe_loso}/*.json                   (CV/LOSO)

Output:
  docs/figures/benchmark_evaluation/
    ├── skema1_macro_f1_per_dataset.png       (bar chart Skema 1 per dataset)
    ├── skema1_per_arch_avg.png               (mean per arch lintas dataset)
    ├── skema2_cross_macro_f1.png             (cross-dataset → Primer)
    ├── ckplus_cv10_mean_std.png              (10-fold CV CK+ mean±std)
    └── jaffe_loso_mean_std.png               (LOSO JAFFE mean±std)
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = PROJECT_ROOT / 'models' / 'benchmark'
OUT_DIR = PROJECT_ROOT / 'docs' / 'figures' / 'benchmark_evaluation'
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ['ckplus', 'jaffe', 'rafdb', 'kdef']
ARCHS = ['CNN', 'FCNN', 'Intermediate', 'CNN_TL', 'Intermediate_TL', 'Late_Fusion', 'Late_Fusion_TL', 'EarlyFusion', 'EarlyFusion_TL']
ARCH_LABELS = ['CNN', 'FCNN', 'Intermediate', 'CNN TL', 'Intermediate TL', 'Late Fusion', 'Late Fusion TL', 'Early Fusion', 'Early Fusion TL']
ARCH_COLORS = ['#4A6A8A', '#5A8055', '#A87143', '#3D5A7A', '#496B47', '#8E5C34', '#9E5070', '#6F5A8F', '#5C3D71']


def load_skema1(dataset, nc):
    p = BENCH_DIR / dataset / f'{dataset}_{nc}c_results.json'
    return json.load(open(p)) if p.exists() else None


def load_cross(source, nc):
    p = BENCH_DIR / 'crossdataset' / f'cross_{source}_{nc}c.json'
    return json.load(open(p)) if p.exists() else None


def get_macro(d, arch_key):
    """Try multiple key formats."""
    for k in [f'{arch_key}_B1', arch_key]:
        if k in d:
            return d[k].get('macro_f1', 0)
    return 0


# ─────────────────────────────────────────────────────────────────────
# 1. Skema 1 — Macro F1 per dataset (grouped by arch)
# ─────────────────────────────────────────────────────────────────────
def plot_skema1_per_dataset():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax_idx, nc in enumerate([7, 4]):
        ax = axes[ax_idx]
        x = np.arange(len(DATASETS))
        bar_w = 0.09
        for i, (arch, color) in enumerate(zip(ARCHS, ARCH_COLORS)):
            vals = []
            for ds in DATASETS:
                d = load_skema1(ds, nc)
                vals.append(get_macro(d, arch) if d else 0)
            offset = (i - len(ARCHS)/2 + 0.5) * bar_w
            ax.bar(x + offset, vals, bar_w, color=color, edgecolor='black', linewidth=0.3,
                   label=ARCH_LABELS[i] if ax_idx == 0 else None)
        ax.set_xticks(x)
        ax.set_xticklabels([d.upper() for d in DATASETS], fontsize=10)
        ax.set_ylabel('Macro F1' if ax_idx == 0 else '', fontsize=10)
        ax.set_title(f'{nc}-Class', fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=5, fontsize=8, frameon=False)
    fig.suptitle('Skema 1 Self-Train-Test — Macro F1 per Benchmark Dataset', fontsize=12, y=1.06)
    plt.tight_layout()
    out = OUT_DIR / 'skema1_macro_f1_per_dataset.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: {out.name}')


# ─────────────────────────────────────────────────────────────────────
# 2. Skema 1 — average per arch lintas dataset
# ─────────────────────────────────────────────────────────────────────
def plot_skema1_per_arch_avg():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax_idx, nc in enumerate([7, 4]):
        ax = axes[ax_idx]
        avgs, stds = [], []
        for arch in ARCHS:
            vals = []
            for ds in DATASETS:
                d = load_skema1(ds, nc)
                if d:
                    v = get_macro(d, arch)
                    if v > 0:
                        vals.append(v)
            avgs.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals) if len(vals) > 1 else 0)
        x = np.arange(len(ARCHS))
        ax.bar(x, avgs, yerr=stds, color=ARCH_COLORS, edgecolor='black', linewidth=0.4,
               capsize=3, error_kw={'linewidth': 0.8})
        ax.set_xticks(x)
        ax.set_xticklabels(ARCH_LABELS, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('Mean Macro F1 ± std' if ax_idx == 0 else '', fontsize=9)
        ax.set_title(f'{nc}-Class — Mean Across Benchmarks', fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out = OUT_DIR / 'skema1_per_arch_avg.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: {out.name}')


# ─────────────────────────────────────────────────────────────────────
# 3. Skema 2 — cross-dataset bar chart
# ─────────────────────────────────────────────────────────────────────
def plot_skema2_cross():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax_idx, nc in enumerate([7, 4]):
        ax = axes[ax_idx]
        x = np.arange(len(DATASETS))
        bar_w = 0.09
        for i, (arch, color) in enumerate(zip(ARCHS, ARCH_COLORS)):
            vals = []
            for src in DATASETS:
                d = load_cross(src, nc)
                vals.append(get_macro(d, arch) if d else 0)
            offset = (i - len(ARCHS)/2 + 0.5) * bar_w
            ax.bar(x + offset, vals, bar_w, color=color, edgecolor='black', linewidth=0.3,
                   label=ARCH_LABELS[i] if ax_idx == 0 else None)
        ax.set_xticks(x)
        ax.set_xticklabels([d.upper() + ' → Primer' for d in DATASETS], fontsize=8.5, rotation=15)
        ax.set_ylabel('Macro F1' if ax_idx == 0 else '', fontsize=10)
        ax.set_title(f'{nc}-Class', fontsize=11)
        ax.set_ylim(0, 0.5)
        ax.grid(axis='y', alpha=0.3)
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=5, fontsize=8, frameon=False)
    fig.suptitle('Skema 2 Cross-Dataset → Primer — Macro F1 per Source', fontsize=12, y=1.06)
    plt.tight_layout()
    out = OUT_DIR / 'skema2_cross_macro_f1.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: {out.name}')


# ─────────────────────────────────────────────────────────────────────
# 4. CK+ 10-fold CV mean±std
# ─────────────────────────────────────────────────────────────────────
def plot_cv_loso(name, json_path, suptitle):
    if not json_path.exists():
        print(f'  [SKIP] {json_path}')
        return
    d = json.load(open(json_path))
    archs = list(d.keys())
    means = [d[a].get('macro_f1_mean', 0) for a in archs]
    stds  = [d[a].get('macro_f1_std', 0) for a in archs]

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(archs))
    colors = ARCH_COLORS[:len(archs)]
    ax.bar(x, means, yerr=stds, color=colors, edgecolor='black', linewidth=0.4,
           capsize=4, error_kw={'linewidth': 1.0})
    ax.set_xticks(x)
    ax.set_xticklabels([a.replace('_', ' ') for a in archs], rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Macro F1 (mean ± std)', fontsize=10)
    ax.set_title(suptitle, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    for xi, m, s in zip(x, means, stds):
        ax.text(xi, m + s + 0.02, f'{m:.3f}±{s:.3f}', ha='center', fontsize=7)
    plt.tight_layout()
    out = OUT_DIR / name
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: {out.name}')


def main():
    print(f'Generating benchmark plots → {OUT_DIR}')
    plot_skema1_per_dataset()
    plot_skema1_per_arch_avg()
    plot_skema2_cross()
    plot_cv_loso('ckplus_cv10_mean_std_4c.png',
                  BENCH_DIR / 'ckplus_cv10' / 'ckplus_4c_cv10_results.json',
                  'CK+ 10-Fold CV (4-class) — Macro F1 mean ± std')
    plot_cv_loso('ckplus_cv10_mean_std_7c.png',
                  BENCH_DIR / 'ckplus_cv10' / 'ckplus_7c_cv10_results.json',
                  'CK+ 10-Fold CV (7-class) — Macro F1 mean ± std')
    plot_cv_loso('jaffe_loso_mean_std_4c.png',
                  BENCH_DIR / 'jaffe_loso' / 'jaffe_4c_loso_results.json',
                  'JAFFE LOSO (4-class) — Macro F1 mean ± std')
    plot_cv_loso('jaffe_loso_mean_std_7c.png',
                  BENCH_DIR / 'jaffe_loso' / 'jaffe_7c_loso_results.json',
                  'JAFFE LOSO (7-class) — Macro F1 mean ± std')
    print(f'\nAll benchmark plots saved to: {OUT_DIR}')


if __name__ == '__main__':
    main()
