#!/usr/bin/env python3
"""Build notebooks/89_robustness_documentation.ipynb.

Dokumentasi visual untuk eksperimen robustness (LOSO / 5-CV / RandomSplit)
atas 6 config top-modality (A-F), mirror gaya build_unimodal_notebook.py /
build_multimodal_notebook.py. Semua angka diambil langsung dari
models/frontonly_conf60/robustness/{loso,cv5,randomsplit}/*_summary.json +
*_per_fold.json. Notebook bersifat self-contained: tabel & plot di-generate
inline dari JSON, tidak bergantung pada figur pra-render.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB_PATH = HERE.parent / "notebooks" / "89_robustness_documentation.ipynb"


def md(*src):
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(src)}


def code(*src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": _lines(src)}


def _lines(src):
    text = "\n".join(src)
    lines = text.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


cells = []

# 0 — Title
cells.append(md(
    "# Dokumentasi Eksperimen Robustness (LOSO / 5-CV / Random Split)",
    "",
    "**Notebook ini di-generate dari `scripts/build_robustness_notebook.py`** sebagai "
    "dokumentasi visual untuk evaluasi robustness top-modality models di bawah tiga "
    "protokol cross-validation subject-wise.",
    "",
    "## Scope notebook",
    "1. Setup & protokol robustness (LOSO, 5-Fold CV subject-wise, Random Split stratified)",
    "2. Loading semua `*_summary.json` + `*_per_fold.json`",
    "3. Master table per strategi (mean ± std: macro_f1, weighted_f1, accuracy)",
    "4. Visualisasi (di-generate inline dari JSON):",
    "   - Bar chart mean ± std macro_f1 per config × strategi",
    "   - Distribusi per-fold (boxplot) tiap config",
    "   - Perbandingan 3 strategi side-by-side",
    "   - Stabilitas (std) per config",
    "5. Tabel detail per-fold (per subjek / fold / seed)",
    "6. Insight utama",
    "",
    "> Eksperimen dijalankan via `scripts/run_robustness_unified.py`. Notebook ini "
    "bersifat dokumentasi & visualisasi — semua angka diambil dari hasil JSON.",
    "",
    "---",
))

# 1 — Setup & protokol
cells.append(md(
    "## 1. Setup & Protokol Eksperimen",
    "",
    "### Tiga protokol validasi",
    "| Strategi | Unit fold | Jumlah | Deskripsi |",
    "|---|---|---|---|",
    "| **LOSO** | subjek | 37 | Leave-One-Subject-Out: 1 subjek test, 2 subjek val, sisanya train. Estimasi paling konservatif untuk generalisasi ke subjek baru. |",
    "| **5-CV** | fold | 5 | 5-Fold CV subject-wise: subjek dipartisi ke 5 fold disjoint (tidak ada bocoran subjek antar train/test). |",
    "| **Random Split** | seed | 5 | Stratified split 80/10/10 di level sampel (subjek boleh muncul di train & test), diulang 5 seed. Estimasi paling optimis (upper bound). |",
    "",
    "### Config yang dievaluasi (top-modality per scheme)",
    "Subset 6 config dipilih dari hasil primer unimodal/multimodal sebagai representatif terbaik per (scheme × arsitektur).",
    "",
    "| Key | Scheme | Arch | Feature | Source | Label |",
    "|:---:|:---:|---|---|:---:|---|",
    "| A | 3c | fcnn | facs_28 | FA | Landmark FCNN (facs_28, FA) |",
    "| B | 3c | intermediate | facs_28 | FA | Intermediate scratch (facs_28, FA) |",
    "| C | 3c | late_tl | facs_28 | FA | Late Fusion TL (facs_28, FA) |",
    "| D | 7c | cnn1d | facs_plus_bs_80 | FA | Landmark CNN1D (FB80, FA) |",
    "| E | 7c | intermediate | facs_28 | FA | Intermediate scratch (facs_28, FA) |",
    "| F | 7c | late_tl | raw_136 | FA | Late Fusion TL (raw_136, FA) |",
    "",
    "### Hyperparameters (sama dengan eksperimen primer)",
    "```",
    "Adam, lr per-config (1e-4 / 5e-5 untuk variant TL), batch=32",
    "epochs_max=50, patience=15, seed=42, loss=CrossEntropyLoss",
    "model selection: best val macro_f1",
    "```",
    "",
    "### Dataset",
    "`data/dataset_frontonly_conf60/` — confidence ≥ 0.6, total 6795 sampel / 37 subjek. "
    "Fold di-split di level subjek (kecuali Random Split yang di level sampel).",
    "",
    "### Metrik",
    "Per fold dihitung **macro_f1**, **weighted_f1**, **accuracy** pada test set fold. "
    "Ringkasan config = **mean ± std** lintas fold. macro_f1 adalah metrik utama "
    "(robust terhadap class imbalance, terutama untuk 7c).",
))

# 2 — Imports
cells.append(md("### 1.1 Imports & paths"))
cells.append(code(
    "%matplotlib inline",
    "import json",
    "from pathlib import Path",
    "import numpy as np",
    "import pandas as pd",
    "import matplotlib.pyplot as plt",
    "from IPython.display import display, Markdown",
    "",
    "PROJECT = Path('..').resolve()",
    "ROBUST = PROJECT / 'models' / 'frontonly_conf60' / 'robustness'",
    "",
    "STRATS = [",
    "    ('loso',        'LOSO (Leave-One-Subject-Out)', 'subjek'),",
    "    ('cv5',         '5-Fold CV (subject-wise)',     'fold'),",
    "    ('randomsplit', 'Random Split (stratified)',    'seed'),",
    "]",
    "CONFIG_ORDER = list('ABCDEF')",
    "",
    "plt.rcParams['figure.dpi'] = 110",
    "plt.rcParams['font.size'] = 10",
    "print('PROJECT:', PROJECT)",
    "print('Robustness dir:', ROBUST)",
    "assert ROBUST.exists(), 'robustness dir tidak ditemukan'",
))

# 3 — Loading
cells.append(md(
    "## 2. Loading Semua Hasil Robustness",
    "",
    "Loader mengiterasi `models/frontonly_conf60/robustness/{strat}/?_*_summary.json` "
    "(ringkasan mean/std) dan `*_per_fold.json` (metrik per fold).",
))
cells.append(code(
    "def load_strategy(strat):",
    "    \"\"\"Return (summaries_df, per_fold_dict) untuk satu strategi.\"\"\"",
    "    d = ROBUST / strat",
    "    rows, per_fold = [], {}",
    "    for sf in sorted(d.glob('?_*_summary.json')):",
    "        s = json.load(open(sf))",
    "        c = s['config']",
    "        rows.append({",
    "            'config': s['config_key'], 'label': c['label'],",
    "            'scheme': c['scheme'], 'arch': c['arch'],",
    "            'feature': c['feature'], 'source': c['source'],",
    "            'folds': f\"{s['n_folds_done']}/{s['n_folds_total']}\",",
    "            'macro_f1_mean': s['test_macro_f1_mean'], 'macro_f1_std': s['test_macro_f1_std'],",
    "            'weighted_f1_mean': s['test_weighted_f1_mean'], 'weighted_f1_std': s['test_weighted_f1_std'],",
    "            'accuracy_mean': s['test_accuracy_mean'], 'accuracy_std': s['test_accuracy_std'],",
    "        })",
    "        pf = json.load(open(str(sf).replace('_summary.json', '_per_fold.json')))",
    "        per_fold[s['config_key']] = pf",
    "    df = pd.DataFrame(rows).set_index('config').reindex(CONFIG_ORDER)",
    "    return df, per_fold",
    "",
    "DATA = {strat: load_strategy(strat) for strat, _, _ in STRATS}",
    "for strat, title, _ in STRATS:",
    "    df, pf = DATA[strat]",
    "    n_err = sum('error' in r for cfg in pf.values() for r in cfg)",
    "    print(f'{title:32} | {len(df)} config | '",
    "          f\"{sum(len(v) for v in pf.values())} fold total | {n_err} error\")",
))

# 4 — Master tables
cells.append(md(
    "## 3. Master Table per Strategi",
    "",
    "Setiap baris = satu config. Nilai = **mean ± std** lintas fold. "
    "Highlight = nilai tertinggi per kolom dalam scheme yang sama.",
))
cells.append(code(
    "def fmt_pm(m, s):",
    "    return f'{m:.4f} ± {s:.4f}'",
    "",
    "def master_table(strat):",
    "    df, _ = DATA[strat]",
    "    out = pd.DataFrame(index=df.index)",
    "    out['label'] = df['label']",
    "    out['scheme'] = df['scheme']",
    "    out['folds'] = df['folds']",
    "    out['macro_f1']    = [fmt_pm(m, s) for m, s in zip(df['macro_f1_mean'], df['macro_f1_std'])]",
    "    out['weighted_f1'] = [fmt_pm(m, s) for m, s in zip(df['weighted_f1_mean'], df['weighted_f1_std'])]",
    "    out['accuracy']    = [fmt_pm(m, s) for m, s in zip(df['accuracy_mean'], df['accuracy_std'])]",
    "    return out",
    "",
    "for strat, title, _ in STRATS:",
    "    display(Markdown(f'### {title}'))",
    "    display(master_table(strat))",
    "    # best macro_f1 per scheme",
    "    df, _ = DATA[strat]",
    "    for sch in ['3c', '7c']:",
    "        sub = df[df['scheme'] == sch]",
    "        best = sub['macro_f1_mean'].idxmax()",
    "        print(f'  [{sch}] best macro_f1: config {best} '",
    "              f\"({sub.loc[best, 'label']}) = {sub.loc[best, 'macro_f1_mean']:.4f}\")",
))

# 5 — Visualisasi: bar mean±std macro_f1
cells.append(md(
    "## 4. Visualisasi",
    "",
    "### 4.1 macro_f1 mean ± std per config × strategi",
    "",
    "Grouped bar dengan error bar = std lintas fold. Memperlihatkan urutan optimis→konservatif: "
    "Random Split (subject leakage) > LOSO/5-CV (subject-disjoint).",
))
cells.append(code(
    "fig, ax = plt.subplots(figsize=(11, 5))",
    "x = np.arange(len(CONFIG_ORDER))",
    "w = 0.26",
    "colors = {'loso': '#4C72B0', 'cv5': '#55A868', 'randomsplit': '#C44E52'}",
    "for i, (strat, title, _) in enumerate(STRATS):",
    "    df, _ = DATA[strat]",
    "    means = df['macro_f1_mean'].values",
    "    stds = df['macro_f1_std'].values",
    "    ax.bar(x + (i - 1) * w, means, w, yerr=stds, capsize=3,",
    "           label=title, color=colors[strat], alpha=0.88)",
    "labels = [f\"{c}\\n({DATA['loso'][0].loc[c, 'scheme']})\" for c in CONFIG_ORDER]",
    "ax.set_xticks(x); ax.set_xticklabels(labels)",
    "ax.set_ylabel('macro_f1 (mean ± std)')",
    "ax.set_xlabel('Config (scheme)')",
    "ax.set_title('Robustness: macro_f1 per config across validation strategies')",
    "ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)",
    "plt.tight_layout(); plt.show()",
))

# 6 — Boxplot per-fold distribution
cells.append(md(
    "### 4.2 Distribusi per-fold (boxplot) — LOSO",
    "",
    "LOSO punya 37 fold sehingga distribusinya informatif. Box lebar = sensitif terhadap "
    "identitas subjek (generalisasi tidak merata).",
))
cells.append(code(
    "def per_fold_metric(strat, metric='macro_f1'):",
    "    _, pf = DATA[strat]",
    "    key = {'macro_f1': 'macro_f1', 'weighted_f1': 'weighted_f1', 'accuracy': 'acc'}[metric]",
    "    return {c: [r[key] for r in pf[c] if 'error' not in r] for c in CONFIG_ORDER}",
    "",
    "fig, ax = plt.subplots(figsize=(10, 5))",
    "dist = per_fold_metric('loso', 'macro_f1')",
    "ax.boxplot([dist[c] for c in CONFIG_ORDER], labels=CONFIG_ORDER, showmeans=True)",
    "ax.set_ylabel('macro_f1 (per subjek)')",
    "ax.set_xlabel('Config')",
    "ax.set_title('LOSO — distribusi macro_f1 antar 37 subjek')",
    "ax.grid(axis='y', alpha=0.3)",
    "plt.tight_layout(); plt.show()",
))

# 7 — Strategy comparison per scheme
cells.append(md(
    "### 4.3 Perbandingan 3 strategi per scheme",
    "",
    "Rata-rata macro_f1 antar config dalam tiap scheme, dibandingkan lintas strategi. "
    "Memvisualkan gap optimis (RandomSplit) vs konservatif (LOSO).",
))
cells.append(code(
    "fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)",
    "for ax, sch in zip(axes, ['3c', '7c']):",
    "    vals, errs, names = [], [], []",
    "    for strat, title, _ in STRATS:",
    "        df, _ = DATA[strat]",
    "        sub = df[df['scheme'] == sch]",
    "        vals.append(sub['macro_f1_mean'].mean())",
    "        errs.append(sub['macro_f1_mean'].std())",
    "        names.append(strat)",
    "    bars = ax.bar(names, vals, yerr=errs, capsize=4,",
    "                  color=[colors[n] for n in names], alpha=0.88)",
    "    for b, v in zip(bars, vals):",
    "        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f'{v:.3f}',",
    "                ha='center', fontsize=9)",
    "    ax.set_title(f'{sch} — mean macro_f1 antar config')",
    "    ax.grid(axis='y', alpha=0.3)",
    "axes[0].set_ylabel('macro_f1')",
    "plt.tight_layout(); plt.show()",
))

# 8 — Stability (std) heatmap-ish bar
cells.append(md(
    "### 4.4 Stabilitas — std macro_f1 per config",
    "",
    "Std rendah = model stabil lintas fold (robust). LOSO biasanya paling tinggi std-nya "
    "karena variasi subjek; RandomSplit paling rendah.",
))
cells.append(code(
    "fig, ax = plt.subplots(figsize=(11, 4.5))",
    "for i, (strat, title, _) in enumerate(STRATS):",
    "    df, _ = DATA[strat]",
    "    ax.bar(x + (i - 1) * w, df['macro_f1_std'].values, w,",
    "           label=title, color=colors[strat], alpha=0.88)",
    "ax.set_xticks(x); ax.set_xticklabels(labels)",
    "ax.set_ylabel('std macro_f1 (lower = more stable)')",
    "ax.set_xlabel('Config (scheme)')",
    "ax.set_title('Robustness: variansi macro_f1 lintas fold')",
    "ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)",
    "plt.tight_layout(); plt.show()",
))

# 9 — Per-fold detail tables
cells.append(md(
    "## 5. Tabel Detail Per-Fold",
    "",
    "Metrik per fold/seed untuk setiap config. Baris terakhir = mean ± std agregat. "
    "Berguna untuk lampiran tesis (reproducibility per subjek).",
))
cells.append(code(
    "def detail_table(strat, cfg):",
    "    _, pf = DATA[strat]",
    "    unit = dict((s, u) for s, _, u in STRATS)[strat]",
    "    rows = []",
    "    for r in pf[cfg]:",
    "        if 'error' in r:",
    "            rows.append({unit: r.get('fold_label'), 'n_train': None, 'n_val': None,",
    "                         'n_test': None, 'macro_f1': None, 'weighted_f1': None, 'accuracy': None})",
    "            continue",
    "        rows.append({unit: r.get('fold_label'), 'n_train': r.get('n_tr'),",
    "                     'n_val': r.get('n_val'), 'n_test': r.get('n_te'),",
    "                     'macro_f1': r.get('macro_f1'), 'weighted_f1': r.get('weighted_f1'),",
    "                     'accuracy': r.get('acc')})",
    "    df = pd.DataFrame(rows)",
    "    s = DATA[strat][0].loc[cfg]",
    "    mean_row = {unit: 'mean ± std', 'n_train': '', 'n_val': '', 'n_test': '',",
    "                'macro_f1': fmt_pm(s['macro_f1_mean'], s['macro_f1_std']),",
    "                'weighted_f1': fmt_pm(s['weighted_f1_mean'], s['weighted_f1_std']),",
    "                'accuracy': fmt_pm(s['accuracy_mean'], s['accuracy_std'])}",
    "    return pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)",
    "",
    "# Contoh: tampilkan semua config untuk LOSO. Ganti 'loso' ke 'cv5'/'randomsplit' sesuai kebutuhan.",
    "STRAT_SHOW = 'loso'",
    "for cfg in CONFIG_ORDER:",
    "    s = DATA[STRAT_SHOW][0].loc[cfg]",
    "    display(Markdown(f\"### {cfg} — {s['label']} ({s['scheme']})\"))",
    "    display(detail_table(STRAT_SHOW, cfg))",
))
cells.append(md(
    "> Untuk melihat detail **5-CV** atau **Random Split**, ubah `STRAT_SHOW` di cell di atas "
    "ke `'cv5'` atau `'randomsplit'` lalu re-run. Versi statis lengkap (ketiga strategi) "
    "tersedia di [`docs/robustness_metrics_tables.md`](../docs/robustness_metrics_tables.md)."
))

# 10 — Insight
cells.append(md(
    "## 6. Insight Utama",
    "",
    "### Hierarki estimasi (optimis → konservatif)",
    "- **Random Split > LOSO ≈ 5-CV** untuk macro_f1 3c. Random Split mengizinkan subjek "
    "yang sama muncul di train & test (subject leakage), sehingga over-estimasi generalisasi. "
    "LOSO/5-CV subject-disjoint memberi estimasi realistik untuk subjek baru.",
    "- Gap antara Random Split dan LOSO ≈ besarnya **overfitting ke identitas subjek**.",
    "",
    "### 3c vs 7c",
    "- macro_f1 3c jauh lebih tinggi dari 7c di semua strategi — konsisten dengan eksperimen "
    "primer; kelas minoritas di 7c sulit dipisah dan menyeret macro_f1 ke bawah.",
    "- weighted_f1 & accuracy 7c tetap tinggi (~0.85) → model akurat di kelas mayoritas tetapi "
    "lemah di kelas minoritas (gap macro vs weighted = sinyal imbalance).",
    "",
    "### Stabilitas",
    "- **LOSO punya std tertinggi** (±0.17–0.25) — performa sangat bergantung pada identitas "
    "subjek test; beberapa subjek mencapai macro_f1 1.0, sebagian < 0.35.",
    "- **Random Split paling stabil** (std kecil) karena distribusi train/test mirip tiap seed.",
    "",
    "### Catatan kualitas (perlu dicek)",
    "- **Config D (CNN1D 7c) di Random Split**: std ≈ 0 di ketiga metrik → kelima seed "
    "menghasilkan output identik. Indikasi kemungkinan model collapse / prediksi kelas "
    "mayoritas saja. Layak diverifikasi terpisah sebelum dipakai sebagai klaim di tesis.",
    "",
    "### Implikasi untuk tesis",
    "- Laporkan **LOSO sebagai metrik generalisasi utama** (paling konservatif & relevan untuk "
    "deployment ke subjek baru), dengan Random Split sebagai upper bound dan 5-CV sebagai "
    "estimasi tengah.",
    "- Sertakan **mean ± std** (bukan hanya mean) untuk menunjukkan variansi antar subjek.",
    "",
    "---",
    "",
    "*Auto-generated oleh `scripts/build_robustness_notebook.py`. Regenerate setelah sweep "
    "robustness baru selesai.*",
))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.write_text(json.dumps(nb, indent=1))
print("WROTE:", NB_PATH, f"({len(cells)} cells)")
