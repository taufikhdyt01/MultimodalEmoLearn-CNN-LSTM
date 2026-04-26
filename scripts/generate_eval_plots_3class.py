"""
Generate post-hoc evaluation plots dari JSON results 3-class.

Sumber data:
  models/frontonly_conf60/3class/all_results_3class.json   (15 configs nb 79)
  models/frontonly_conf60/3class/scratch_all_results.json  (12 configs nb 82)
  models/frontonly_conf60/3class/CBAM/cbam_3class_results.json   (4 configs nb 80)
  models/frontonly_conf60/3class/Geometric/geometric_3class_results.json  (5 configs nb 81)

Generates:
  docs/figures/3class_evaluation/
    ├── comparison_macro_f1_27configs.png    (bar chart all configs)
    ├── comparison_per_arch.png              (per-arch best F1)
    ├── per_class_f1_top_configs.png         (per-class F1 untuk top configs)
    ├── cm_heatmap_juara.png                 (CM Late Fusion TL B3)
    └── w_best_distribution.png              (w_best across Late Fusion configs)
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / 'models' / 'frontonly_conf60' / '3class'
OUT_DIR = PROJECT_ROOT / 'docs' / 'figures' / '3class_evaluation'
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMOTIONS_3 = ['positive', 'neutral', 'negative']
COLORS_3 = ['#5A8055', '#4A6A8A', '#A87143']


def load_all_results():
    """Merge semua JSON results jadi master dict."""
    master = {}
    sources = [
        ('all_results_3class.json',          ''),
        ('scratch_all_results.json',         ''),
        ('CBAM/cbam_3class_results.json',    ''),
        ('Geometric/geometric_3class_results.json', ''),
    ]
    for f, _ in sources:
        path = RESULTS_DIR / f
        if path.exists():
            d = json.load(open(path))
            master.update(d)
            print(f'  Loaded {len(d)} from {f}')
        else:
            print(f'  [SKIP] {path}')
    print(f'Master: {len(master)} configs total')
    return master


def parse_arch_scenario(cfg):
    """Heuristic to split 'CNN_TL_B3' → ('CNN_TL', 'B3')."""
    if cfg.endswith(('_B1', '_B2', '_B3')):
        return cfg[:-3], cfg[-2:]
    return cfg, '?'


# ─────────────────────────────────────────────────────────────────────
# 1. Comparison bar chart — all configs Macro F1
# ─────────────────────────────────────────────────────────────────────
def plot_comparison_all(results):
    cfgs = sorted(results.keys(), key=lambda k: -results[k].get('val_macro_f1', 0))
    val_f1  = [results[c].get('val_macro_f1', 0) for c in cfgs]
    test_f1 = [results[c].get('test_macro_f1', 0) for c in cfgs]

    fig, ax = plt.subplots(figsize=(11, max(5, len(cfgs) * 0.28)))
    y = np.arange(len(cfgs))
    bar_h = 0.4
    ax.barh(y - bar_h/2, val_f1, height=bar_h, color='#4A6A8A', label='Val Macro F1', edgecolor='black', linewidth=0.4)
    ax.barh(y + bar_h/2, test_f1, height=bar_h, color='#A87143', label='Test Macro F1', edgecolor='black', linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(cfgs, fontsize=7.5)
    ax.set_xlabel('Macro F1', fontsize=10)
    ax.set_xlim(0, 0.8)
    ax.invert_yaxis()
    ax.axvline(0.6229, color='red', linestyle='--', linewidth=0.8, alpha=0.6, label='Juara baseline (Late Fusion TL B3 val=0.623)')
    ax.legend(fontsize=8, loc='lower right')
    ax.set_title(f'3-Class Configs — Val vs Test Macro F1 (sorted by val, n={len(cfgs)})', fontsize=11)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    out = OUT_DIR / 'comparison_macro_f1_all_configs.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: {out.name}')


# ─────────────────────────────────────────────────────────────────────
# 2. Per-arch best F1 (group by architecture)
# ─────────────────────────────────────────────────────────────────────
def plot_per_arch_best(results):
    arch_groups = {}
    for cfg in results:
        arch, sc = parse_arch_scenario(cfg)
        arch_groups.setdefault(arch, []).append((sc, results[cfg]))

    archs = sorted(arch_groups.keys(), key=lambda a: -max(r.get('val_macro_f1', 0) for _, r in arch_groups[a]))
    scenarios = ['B1', 'B2', 'B3']
    bar_w = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(archs))
    for i, sc in enumerate(scenarios):
        vals = [next((r['val_macro_f1'] for s, r in arch_groups[a] if s == sc), 0) for a in archs]
        ax.bar(x + i * bar_w - bar_w, vals, bar_w, label=f'{sc}',
               color=['#4A6A8A', '#5A8055', '#A87143'][i], edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(archs, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Val Macro F1', fontsize=10)
    ax.set_xlabel('Architecture', fontsize=10)
    ax.set_ylim(0, 0.75)
    ax.axhline(0.6229, color='red', linestyle='--', linewidth=0.8, alpha=0.6, label='Juara (LF TL B3 val=0.623)')
    ax.set_title('3-Class Val Macro F1 per Architecture × Scenario', fontsize=11)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out = OUT_DIR / 'per_arch_val_macro_f1.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: {out.name}')


# ─────────────────────────────────────────────────────────────────────
# 3. Per-class F1 — top-5 configs
# ─────────────────────────────────────────────────────────────────────
def plot_per_class_top(results, top_k=5):
    sorted_cfgs = sorted(results.items(), key=lambda kv: -kv[1].get('val_macro_f1', 0))[:top_k]

    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    cfgs = [c for c, _ in sorted_cfgs]
    bar_w = 0.25
    x = np.arange(len(cfgs))

    for i, cls in enumerate(EMOTIONS_3):
        f1s = []
        for c, r in sorted_cfgs:
            cr = r.get('classification_report', {})
            if cls in cr:
                f1s.append(cr[cls]['f1-score'])
            else:
                # Derive from CM if missing classification_report
                cm = np.array(r.get('confusion_matrix', [[0]*3]*3), dtype=float)
                if cm.sum() > 0:
                    tp = cm[i, i]
                    p_d = cm[:, i].sum()
                    r_d = cm[i, :].sum()
                    prec = tp / p_d if p_d > 0 else 0
                    rec  = tp / r_d if r_d > 0 else 0
                    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
                    f1s.append(f1)
                else:
                    f1s.append(0)
        ax.bar(x + i * bar_w - bar_w, f1s, bar_w, label=cls, color=COLORS_3[i], edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(cfgs, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Test F1 per class', fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_title(f'Per-Class Test F1 — Top-{top_k} Configs (sorted by val)', fontsize=11)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out = OUT_DIR / f'per_class_f1_top{top_k}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: {out.name}')


# ─────────────────────────────────────────────────────────────────────
# 4. CM heatmap — juara
# ─────────────────────────────────────────────────────────────────────
def plot_cm_juara(results, cfg='Late_Fusion_TL_B3'):
    if cfg not in results:
        print(f'  [SKIP] {cfg} not in results')
        return
    cm = np.array(results[cfg]['confusion_matrix'], dtype=int)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=cm.max())
    ax.set_xticks(np.arange(3)); ax.set_yticks(np.arange(3))
    ax.set_xticklabels(EMOTIONS_3, fontsize=10)
    ax.set_yticklabels(EMOTIONS_3, fontsize=10)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)

    threshold = cm.max() * 0.6
    for i in range(3):
        for j in range(3):
            color = 'white' if cm[i, j] > threshold else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=11, color=color)

    val = results[cfg].get('val_macro_f1', 0)
    test = results[cfg].get('test_macro_f1', 0)
    ax.set_title(f'{cfg.replace("_", " ")} — val F1={val:.3f}, test F1={test:.3f}', fontsize=10, pad=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Count')
    plt.tight_layout()
    out = OUT_DIR / f'cm_{cfg.lower()}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: {out.name}')


# ─────────────────────────────────────────────────────────────────────
# 5. w_best distribution (Late Fusion configs)
# ─────────────────────────────────────────────────────────────────────
def plot_w_best(results):
    lf_configs = {k: v for k, v in results.items()
                  if 'Late_Fusion' in k and v.get('best_cnn_weight') is not None}
    if not lf_configs:
        print('  [SKIP] No Late Fusion configs with w_best')
        return

    cfgs = sorted(lf_configs.keys())
    ws   = [lf_configs[c]['best_cnn_weight'] for c in cfgs]
    val  = [lf_configs[c].get('val_macro_f1', 0) for c in cfgs]

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ['#5A8055' if w <= 0.20 else '#A87143' for w in ws]
    bars = ax.bar(range(len(cfgs)), ws, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(cfgs)))
    ax.set_xticklabels(cfgs, rotation=25, ha='right', fontsize=8)
    ax.set_ylabel('w_best (CNN weight)', fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.axhline(0.20, color='red', linestyle='--', linewidth=0.8, alpha=0.6, label='w=0.20 (FCNN-dominant threshold)')
    ax.set_title(f'Late Fusion w_best (val-tuned) — {len(cfgs)} configs', fontsize=11)
    for bar, w_val, vf in zip(bars, ws, val):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{w_val:.2f}\n(F1 {vf:.3f})', ha='center', fontsize=7)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out = OUT_DIR / 'w_best_distribution.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: {out.name}')


def main():
    print(f'Loading results from {RESULTS_DIR}...')
    results = load_all_results()
    print(f'\nGenerating plots → {OUT_DIR}')
    plot_comparison_all(results)
    plot_per_arch_best(results)
    plot_per_class_top(results, top_k=5)
    plot_cm_juara(results, 'Late_Fusion_TL_B3')
    plot_cm_juara(results, 'CNN_TL_B3')        # secondary: best test F1
    plot_w_best(results)
    print(f'\nAll plots saved to: {OUT_DIR}')


if __name__ == '__main__':
    main()
