"""Generate notebooks/88_multimodal_documentation.ipynb."""
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": text.splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.splitlines(keepends=True)})


md("""# Dokumentasi Eksperimen Multimodal (Fusion)

**Notebook ini di-generate dari `scripts/build_multimodal_notebook.py`** sebagai dokumentasi visual untuk semua eksperimen multimodal fusion (citra wajah + facial landmark).

## Scope notebook
1. Setup & protokol fusion (Early / Intermediate / Late, scratch/TL variant, source MP/FA, feature variants)
2. Loading semua hasil `results.json` fusion → tabular
3. Master table fusion per scheme
4. Visualisasi:
   - Heatmap master fusion (Primer)
   - Top-10 fusion leaderboard
   - Confusion matrices top fusion per kategori
   - Late fusion weights (w_image vs w_landmark)
   - Training time comparison
   - Fusion vs unimodal terbaik per dataset (cross-dataset)
   - Best fusion per dataset (Primer + 4 benchmark)
5. Insight utama

> Eksperimen utama dijalankan via Python scripts di `scripts/`. Notebook ini bersifat dokumentasi & visualisasi — semua angka diambil dari `results.json` hasil eksperimen.

---
""")

md("""## 1. Setup & Protokol Eksperimen

### Skema fusion
- **Early Fusion (input-level)** — landmark heatmap di-stack sebagai channel ke-4 RGB → CNN single-branch (raw_136 only).
  - Mode: `concat` (channel-stack langsung) vs `gated` (spatial gating sebelum conv).
- **Intermediate Fusion (feature-level)** — image branch (CNN/CNN_TL) + landmark branch (FCNN), feature concat sebelum head.
- **Late Fusion (decision-level)** — train image branch + landmark branch independent, lalu weighted softmax (w_image swept di val set).

### Variant arsitektur
| Variant | Image branch | Landmark branch |
|---|---|---|
| `scratch` | CNN 4-block (27M) | FCNN 5-dense |
| `tl` | ResNet-18 ImageNet | FCNN 5-dense |

### Feature variants di landmark branch
`raw_136`, `facs_28`, `blendshape_52` (MP only), `facs_plus_bs_80`.

### Source landmark
- **MediaPipe (MP)**: post face-crop 224×224, 478→68 dlib-mapping.
- **face-api.js (FA)**: koordinat frame asli, native 68 dlib via TinyFace+Landmark68.

### Definisi B1 / B2 / B3 (sama dengan unimodal)

| Scenario | Sampler | Augmentation per batch |
|---|---|---|
| **B1** | shuffle uniform | none |
| **B2** | `WeightedRandomSampler` (prob ∝ 1/class_count) | none |
| **B3** | `WeightedRandomSampler` | synced per-batch aug (hflip + landmark_swap + heatmap_flip, rotate ±10°, brightness/contrast ±10%) |

### Hyperparameters

```
Adam, lr=1e-3 (lr=1e-4 untuk variant TL ResNet-18 finetune)
batch=32, epochs_max=50, patience=15, seed=42
loss=CrossEntropyLoss (no class weight — sampler yang handle)
```

### Scripts utama
- `scripts/run_unified_fusion.py` — early & intermediate fusion sweep
- `scripts/compute_late_fusion_unified.py` — late fusion (combine cached softmax dari unimodal)

### Dataset utama (Primer)
`data/dataset_frontonly_conf60/` — confidence ≥ 0.6, per-user split.

- train 5287 / 29 users · val 579 / 5 users · test 929 / 3 users · total 6795 / 37 users

### Benchmark cross-dataset
KDEF, RAF-DB, CK+, JAFFE (7-class, 3-class derivation via valence mapping).
""")

code("""# Imports
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display

PROJECT = Path('..').resolve()
PRIMER = PROJECT / 'models' / 'frontonly_conf60'
FIG_ROOT = PROJECT / 'docs' / 'figures' / 'multimodal'

BENCHMARKS = [
    ('KDEF',   PROJECT / 'models/benchmark/kdef_7class'),
    ('RAF-DB', PROJECT / 'models/benchmark/rafdb_7class'),
    ('CK+',    PROJECT / 'models/benchmark/ckplus_7class'),
    ('JAFFE',  PROJECT / 'models/benchmark/jaffe_7class'),
]

# Fusion result directory prefixes (under {scheme}class/Unified/)
FUSION_PREFIXES = ('fusion_early_', 'fusion_intermediate_', 'fusion_late_')
print('PROJECT:', PROJECT)
print('Figure dir:', FIG_ROOT)""")

md("""## 2. Loading Semua Hasil Multimodal

Loader mirror `build_multimodal_figures.py` — iterasi `models/.../{3,7}class/Unified/fusion_*/results.json`.
""")

code("""def load_fusion_results(scheme_dir):
    out = []
    if not scheme_dir.exists(): return out
    for results_file in scheme_dir.glob('*/results.json'):
        method = results_file.parent.name
        if not method.startswith(FUSION_PREFIXES):
            continue
        try:
            d = json.load(open(results_file))
        except Exception:
            continue
        for run_key, run in d.get('runs', {}).items():
            run['_method_dir'] = method
            run['_run_key'] = run_key
            out.append(run)
    return out

primer_fusion = {}
for scheme, sk in [('3class','3c'), ('7class','7c')]:
    primer_fusion[sk] = load_fusion_results(PRIMER / scheme / 'Unified')
    print(f'Primer fusion {sk}: {len(primer_fusion[sk])} runs')

bench_fusion = {}
for bname, bdir in BENCHMARKS:
    bench_fusion[bname] = {}
    for scheme, sk in [('3class','3c'), ('7class','7c')]:
        p = bdir / scheme / 'Unified'
        bench_fusion[bname][sk] = load_fusion_results(p) if p.exists() else []
        print(f'{bname} fusion {sk}: {len(bench_fusion[bname][sk])} runs')""")

md("""## 3. Master Table Fusion (Primer)

Setiap baris = (fusion_type, mode/variant, feature, source). Kolom = scheme × scenario (6 kolom). Cell = test macro_f1.
""")

code("""def parse_fusion_key(method_dir, run_key):
    \"\"\"Decompose run_key like 'fusion_<type>_<...>_<scenario>_<scheme>'.

    run_key examples:
      fusion_early_gated_scratch_b1_3c              → early/gated/scratch/MP/raw_136
      fusion_early_gated_scratch_faceapi_b1_3c      → early/gated/scratch/FA/raw_136
      fusion_intermediate_scratch_facs_28_b1_3c     → intermediate/-/scratch/MP/facs_28
      fusion_intermediate_tl_facs_plus_bs_80_faceapi_b3_7c → intermediate/-/tl/FA/facs_plus_bs_80
      fusion_late_scratch_blendshape_52_b1_3c       → late/-/scratch/MP/blendshape_52
    \"\"\"
    parts = run_key.split('_')
    if len(parts) < 4 or parts[0] != 'fusion':
        return None
    scheme = parts[-1]                    # '3c' / '7c'
    scenario = parts[-2].upper()          # 'B1' / 'B2' / 'B3'
    mid = parts[1:-2]                     # without 'fusion' prefix and trailing scenario/scheme
    if not mid:
        return None
    ftype = mid[0]                        # early / intermediate / late
    body = mid[1:]
    if body and body[-1] == 'faceapi':
        source = 'FA'; body = body[:-1]
    else:
        source = 'MP'

    if ftype == 'early':
        # body = [mode, variant]; feature is always raw_136 for early
        mode = body[0] if body else 'concat'
        variant = body[1] if len(body) > 1 else 'scratch'
        feature = 'raw_136'
    else:
        # body = [variant] or [variant, *feature_parts]
        variant = body[0] if body else 'scratch'
        feat_parts = body[1:]
        feature = '_'.join(feat_parts) if feat_parts else 'raw_136'
        mode = ''

    return {'ftype': ftype, 'mode': mode, 'variant': variant,
            'feature': feature, 'source': source,
            'scheme': scheme, 'scenario': scenario}


def build_master_table_fusion():
    rows = {}
    for scheme in ('3c', '7c'):
        for r in primer_fusion[scheme]:
            mf1 = r.get('test', {}).get('macro_f1')
            if mf1 is None:
                continue
            info = parse_fusion_key(r['_method_dir'], r['_run_key'])
            if not info:
                continue
            key = (info['ftype'], info['mode'] or info['variant'],
                   info['variant'] if info['mode'] else '—',
                   info['feature'], info['source'])
            col = f\"{info['scheme']}-{info['scenario']}\"
            rows.setdefault(key, {})[col] = mf1
    df = pd.DataFrame.from_dict(rows, orient='index')
    df.index.names = ['ftype', 'mode/variant', 'variant', 'feature', 'source']
    df = df.reindex(columns=['3c-B1','3c-B2','3c-B3','7c-B1','7c-B2','7c-B3'])
    return df.sort_index()

master_fusion = build_master_table_fusion()
print(f'Total rows: {len(master_fusion)}')
master_fusion.style.format('{:.4f}', na_rep='-').background_gradient(cmap='RdYlGn', axis=None)""")

md("""### Highlight cell terbaik per kolom
""")

code("""print('Best per kolom:')
for col in master_fusion.columns:
    col_data = master_fusion[col].dropna()
    if col_data.empty:
        print(f'  {col}: (no data)')
        continue
    idx = col_data.idxmax()
    print(f'  {col}: {idx}  →  {col_data.loc[idx]:.4f}')""")

md("""## 4. Visualisasi

Semua figure dihasilkan oleh `scripts/build_multimodal_figures.py`. Untuk regenerate setelah eksperimen baru selesai, jalankan ulang script tersebut.

### 4.0 Pembahasan Research Question 3

**RQ3 tesis:** Bagaimana kinerja tiga strategi fusi multimodal (Early Fusion, Intermediate Fusion, Late Fusion) beserta dua mode Early Fusion (concat dan learned gated), dan kombinasi mana paling optimal pada 3-class dan 7-class?

Scope RQ3 mencakup **fusion secara keseluruhan**, termasuk variasi feature representation (raw_136, facs_28, blendshape_52, facs_plus_bs_80) di Intermediate & Late Fusion. Early Fusion secara desain terbatas pada raw_136 (input adalah RGB + landmark heatmap).

Section ini punya 4 angle pembahasan:

#### 4.0.1 RQ3.a — Early Fusion: concat vs learned gated (paired comparison)

Dipasangkan per (variant × source × scenario), jadi setiap pasangan beda hanya pada mekanisme fusion (concat = simple channel-stack, gated = spatial gating sebelum conv).
""")

code("""display(Image(filename=str(FIG_ROOT/'comparisons'/'rq3_early_concat_vs_gated_3c.png')))
display(Image(filename=str(FIG_ROOT/'comparisons'/'rq3_early_concat_vs_gated_7c.png')))""")

md("""#### 4.0.2 RQ3.b — Perbandingan 3 strategi fusion (Early / Intermediate / Late)

Distribusi mf1 per fusion strategy (Early dipecah menjadi concat & gated). Boxplot di kiri menunjukkan range/median; bar di kanan menampilkan mean ± std + max sebagai ringkasan numerik.
""")

code("""display(Image(filename=str(FIG_ROOT/'comparisons'/'rq3_fusion_strategy_comparison_3c.png')))
display(Image(filename=str(FIG_ROOT/'comparisons'/'rq3_fusion_strategy_comparison_7c.png')))""")

md("""#### 4.0.3 RQ3.c — Fusion × feature decomposition (heatmap)

Heatmap menunjukkan **best mf1 lintas scenario** untuk tiap kombinasi (strategy + variant + source) × feature. Catatan: Early Fusion hanya support raw_136 (baris Early hanya terisi di kolom raw_136).
""")

code("""display(Image(filename=str(FIG_ROOT/'comparisons'/'rq3_fusion_feature_decomposition_3c.png')))
display(Image(filename=str(FIG_ROOT/'comparisons'/'rq3_fusion_feature_decomposition_7c.png')))""")

md("""#### 4.0.4 RQ3.d — Kombinasi optimal per scheme × scenario

Pemenang absolut (top mf1) untuk tiap kombinasi (scheme × scenario), menampilkan strategi + variant + feature + source.
""")

code("""# RQ3 optimal-combination table
def _parse(k):
    parts = k.split('_')
    if len(parts)<4 or parts[0]!='fusion': return None
    scheme = parts[-1]; scn = parts[-2].upper()
    mid = parts[1:-2]
    if not mid: return None
    ftype = mid[0]; body = mid[1:]
    if body and body[-1]=='faceapi': source='FA'; body=body[:-1]
    else: source='MP'
    if ftype=='early':
        mode = body[0] if body else 'concat'
        variant = body[1] if len(body)>1 else 'scratch'
        feature = 'raw_136'
    else:
        mode=''; variant = body[0] if body else 'scratch'
        feature = '_'.join(body[1:]) if len(body)>1 else 'raw_136'
    return ftype, mode, variant, feature, source, scn, scheme

rq3_rows = []
for scheme_key in ('3c','7c'):
    for scn in ('B1','B2','B3'):
        cands = []
        for r in primer_fusion[scheme_key]:
            info = _parse(r['_run_key'])
            mf1 = r.get('test',{}).get('macro_f1')
            if info is None or mf1 is None: continue
            ftype, mode, variant, feature, source, r_scn, r_scheme = info
            if r_scheme != scheme_key or r_scn != scn: continue
            cands.append((ftype, mode, variant, feature, source, mf1))
        if not cands: continue
        top = max(cands, key=lambda t: t[5])
        ftype, mode, variant, feature, source, mf1 = top
        strat = f'Early-{mode}' if ftype=='early' else ftype.capitalize()
        rq3_rows.append({'scheme':scheme_key,'scenario':scn,
                         'top_strategy':strat,'variant':variant,
                         'feature':feature,'source':source,'top_mf1':mf1})
rq3_df = pd.DataFrame(rq3_rows)
rq3_df.style.format({'top_mf1':'{:.4f}'})""")

md("""#### 4.0.5 Multi-metric fusion — accuracy, macro_f1, weighted_f1

Pelengkap RQ3: best run per fusion strategy × variant dievaluasi dengan 3 metrik sekaligus.
""")

code("""display(Image(filename=str(FIG_ROOT/'comparisons'/'multi_metric_fusion_3c.png')))
display(Image(filename=str(FIG_ROOT/'comparisons'/'multi_metric_fusion_7c.png')))""")

md("""#### 4.0.6 Inference throughput per fusion strategy
""")

code("""display(Image(filename=str(FIG_ROOT/'comparisons'/'inference_throughput_fusion_3c.png')))
display(Image(filename=str(FIG_ROOT/'comparisons'/'inference_throughput_fusion_7c.png')))""")

md("""#### 4.0.7 Per-class Precision / Recall / F1 — top fusion per strategy
""")

code("""display(Image(filename=str(FIG_ROOT/'confusion_matrices'/'per_class_metrics_fusion_3c.png')))
display(Image(filename=str(FIG_ROOT/'confusion_matrices'/'per_class_metrics_fusion_7c.png')))""")

md("""**Tabel per-class precision/recall/f1 — top fusion models (Primer):**
""")

code("""rows = []
for scheme_key in ('3c','7c'):
    cands = [(r['test']['macro_f1'], r) for r in primer_fusion[scheme_key]
             if r.get('test',{}).get('macro_f1') is not None]
    if not cands: continue
    cands.sort(key=lambda t: -t[0])
    for rank, (mf1, r) in enumerate(cands[:3], start=1):
        rep = r.get('test',{}).get('classification_report', {})
        drop = {'accuracy','macro avg','weighted avg'}
        for cls, m in rep.items():
            if cls in drop or not isinstance(m, dict): continue
            rows.append({'scheme':scheme_key,'rank':rank,
                         'method':r.get('_run_key',''),
                         'class':cls,
                         'precision':m.get('precision'),
                         'recall':m.get('recall'),
                         'f1':m.get('f1-score'),
                         'support':int(m.get('support',0))})
per_class_fusion_df = pd.DataFrame(rows)
per_class_fusion_df.style.format({'precision':'{:.3f}','recall':'{:.3f}','f1':'{:.3f}'}, na_rep='-') \\
                         .background_gradient(subset=['precision','recall','f1'], cmap='RdYlGn', vmin=0, vmax=1)""")

md("""#### 4.0.8 Grad-CAM — Interpretability image branch (CNN_TL → Fusion)

Grad-CAM membandingkan attention map **CNN_TL standalone** vs **Early Fusion TL** vs **Intermediate Fusion TL** image branch. Visualisasi apakah landmark guidance (heatmap di Early, coord features di Intermediate) **menggeser attention** ke region yang lebih meaningful.

> **Sumber:** outputs/gradcam (3c), outputs/gradcam_7c (7c) via `scripts/run_gradcam_3c.py` & `scripts/run_gradcam_7c.py`. Late Fusion juga di-gen (combined per-branch).

**Side-by-side 3c (positive / neutral / negative):**
""")

code("""GRADCAM_3C = PROJECT/'outputs'/'gradcam'
for f in ['gradcam_comparison_cls0_sample273.png',
          'gradcam_comparison_cls1_sample129.png',
          'gradcam_comparison_cls2_sample337.png']:
    print(f'--- {f} ---')
    display(Image(filename=str(GRADCAM_3C/f)))""")

md("""**Side-by-side 7c — sampel per kelas:**
""")

code("""GRADCAM_7C = PROJECT/'outputs'/'gradcam_7c'
for f in ['gradcam_comparison_angry_s339.png',
          'gradcam_comparison_disgust_s25.png',
          'gradcam_comparison_fear_s562.png',
          'gradcam_comparison_happy_s762.png',
          'gradcam_comparison_neutral_s851.png',
          'gradcam_comparison_sad_s337.png',
          'gradcam_comparison_surprise_s289.png']:
    print(f'--- {f} ---')
    display(Image(filename=str(GRADCAM_7C/f)))""")

md("""**Error case analysis — Early/Intermediate/Late Fusion (3c & 7c):**

Sample yang misclassified untuk tiap arsitektur fusion. Diagnosa mengapa fusion gagal di kasus tertentu.
""")

code("""print('--- Early Fusion TL error (3c) ---')
display(Image(filename=str(GRADCAM_3C/'gradcam_earlyfusion_tl_error.png')))
print('--- Intermediate Fusion TL error (3c) ---')
display(Image(filename=str(GRADCAM_3C/'gradcam_intermediate_tl_error.png')))
print('--- Late Fusion TL error (3c) ---')
display(Image(filename=str(GRADCAM_3C/'gradcam_latefusion_tl_error.png')))""")

code("""print('--- Early Fusion TL error (7c) ---')
display(Image(filename=str(GRADCAM_7C/'gradcam_earlyfusion_tl_error.png')))
print('--- Intermediate Fusion TL error (7c) ---')
display(Image(filename=str(GRADCAM_7C/'gradcam_intermediate_tl_error.png')))
print('--- Late Fusion error (7c) ---')
display(Image(filename=str(GRADCAM_7C/'gradcam_latefusion_error.png')))""")

md("""**Insight Grad-CAM untuk RQ3:**

- **Early Fusion TL** dengan landmark heatmap channel cenderung membuat attention **lebih fokus** di region landmark-rich (mata, mulut) dibanding CNN_TL polos — bukti kualitatif bahwa landmark prior berguna.
- **Intermediate Fusion TL** image branch menampilkan attention yang **berbeda dari CNN_TL standalone** — model belajar pakai branch image untuk pattern berbeda saat ada concurrent landmark feature.
- **Late Fusion** tidak menggeser attention CNN (karena training terpisah) — keunggulannya hanya di decision-level combination.
- Untuk minority class 7c (disgust, fear), fusion Grad-CAM masih diffuse → modality combination tidak fully menyelesaikan few-shot challenge.

### 4.0.9 Deep-dive notebook
Untuk analisis Grad-CAM yang lebih komprehensif (semua sample, multi-class confusion case), lihat:
- `notebooks/73_gradcam_analysis.ipynb` — overview
- `notebooks/86_gradcam_error_analysis.ipynb` — error case studi 3c

---

**Insight RQ3 dari data Primer:**

- **3-class (3c)**: Late Fusion paling sering juara di B1; Intermediate menang di B2 (TL); Late TL menang di B3. Pola: **decision-level fusion (Late) optimal saat class-count rendah** karena weighted softmax robust terhadap mismatch skala antar branch.
- **7-class (7c)**: Intermediate Fusion menang di B1; Late di B2 & B3. Pada class-count tinggi, **feature-level fusion (Intermediate)** menunjukkan keunggulan di skenario standar (B1), tapi Late lebih stabil di scenario dengan sampler imbalance.
- **Early concat vs gated**: gated tidak konsisten unggul dari concat — varies per variant/source/scenario. Spatial gating overhead tidak selalu terbayar di dataset kecil.

### 4.1 Master Heatmap Fusion (Primer)

Heatmap macro_f1 untuk semua kombinasi fusion di Primer.
""")

code("""display(Image(filename=str(FIG_ROOT/'comparisons'/'heatmap_fusion_primer_3c.png')))
display(Image(filename=str(FIG_ROOT/'comparisons'/'heatmap_fusion_primer_7c.png')))""")

md("""### 4.2 Top-10 Fusion Leaderboard (Primer)

Top-10 fusion run berdasarkan macro_f1. Warna = jenis fusion (early/intermediate/late).
""")

code("""display(Image(filename=str(FIG_ROOT/'leaderboards'/'top10_fusion_primer_3c.png')))
display(Image(filename=str(FIG_ROOT/'leaderboards'/'top10_fusion_primer_7c.png')))""")

md("""### 4.3 Confusion Matrices — Top Fusion Models per Kategori

Confusion matrix untuk fusion terbaik per kategori (Early / Intermediate / Late × scratch / TL × feature variant).
""")

code("""import os
cm_dir = FIG_ROOT/'confusion_matrices'
cm_files = sorted([f for f in os.listdir(cm_dir) if f.startswith('top_fusion_')])
for f in cm_files:
    print(f'--- {f} ---')
    display(Image(filename=str(cm_dir/f)))""")

md("""### 4.4 Late Fusion Weights (w_image vs w_landmark)

Late fusion melakukan weighted softmax: `p_fused = w_image · p_image + (1 − w_image) · p_landmark`, dengan `w_image` di-sweep di val set untuk pilih optimal. Plot ini menunjukkan distribusi w_image yang dipilih untuk semua late fusion run.
""")

code("""display(Image(filename=str(FIG_ROOT/'comparisons'/'late_fusion_weights.png')))""")

md("""### 4.5 Training Time Comparison

Wall-clock training time per fusion type — early & intermediate yang train end-to-end vs late yang reuse cached unimodal predictions.
""")

code("""display(Image(filename=str(FIG_ROOT/'comparisons'/'training_time_comparison.png')))""")

md("""### 4.6 Fusion vs Unimodal Terbaik per Dataset (Cross-dataset)

Apakah fusion betul-betul mengalahkan unimodal di setiap dataset? Bandingkan top fusion vs top unimodal per dataset, scheme B1.
""")

code("""display(Image(filename=str(FIG_ROOT/'comparisons'/'fusion_vs_unimodal_cross_dataset_3c.png')))
display(Image(filename=str(FIG_ROOT/'comparisons'/'fusion_vs_unimodal_cross_dataset_7c.png')))""")

md("""### 4.7 Best Fusion per Dataset (Primer + 4 Benchmark)

Top fusion macro_f1 per dataset (Primer, KDEF, RAF-DB, CK+, JAFFE) — kedua skema dalam satu chart.
""")

code("""display(Image(filename=str(FIG_ROOT/'comparisons'/'best_fusion_per_dataset.png')))""")

md("""## 5. Insight Utama

### Best per scheme (Primer)
- **3c top**: Late Fusion `facs_28` × FA × **B1 = 0.7604** (sedikit lebih baik dari Late `facs_28` FA scratch B1 = 0.7604; FCNN unimodal `facs_28` FA = 0.7585)
- **7c top**: Intermediate scratch `facs_28` × FA × **B1 = 0.3363** (slim margin atas Landmark CNN1D `fb80` FA = 0.3279)

### Milestone — Fusion mengalahkan Unimodal
17 Mei 2026: Late Fusion AKHIRNYA outperform unimodal terbaik di Primer 3c (Δ +0.0019). Untuk 7c gap masih sangat tipis dan secara statistik kemungkinan tied.

### Pola yang konsisten
- **FA-source > MP-source** di fusion (sama seperti unimodal) — koordinat frame asli lebih informatif daripada koordinat pasca face-crop.
- **`facs_28` dominasi di fusion**: top model di hampir semua kombinasi fusion × scheme pakai facs_28 FA, mengindikasikan hand-crafted FACS distance + image branch saling melengkapi.
- **Late > Intermediate > Early** untuk 3c (urutan macro_f1 top per fusion type). Decision-level fusion paling tahan terhadap mismatch dimensi/skala antar modality.
- **Intermediate menang di 7c**: feature-level fusion lebih cocok ketika class-count tinggi karena tidak terkunci skala softmax single-prediction.
- **Variant TL > scratch** di image branch untuk dataset kecil (~5K train sample), konsisten dengan finding unimodal.

### Cross-dataset
- Ranking method **konsisten** lintas dataset (top performer di Primer juga top di KDEF/RAF-DB/CK+/JAFFE), walaupun absolute mf1 berbeda karena variasi distribusi kelas dan kondisi citra.
- CK+ & JAFFE (lab-controlled, balanced, small) → mf1 sangat tinggi (0.9+ di 3c).
- RAF-DB (in-the-wild, mild imbalance) → paling challenging di antara benchmark, masih outperform Primer at scale.

### Limitations
- Hyperparameter shared antar fusion variant (lr, batch, patience) → tidak ada per-fusion tuning.
- Image branch fix di ResNet-18 / CNN 4-block; belum eksplorasi backbone lain (EfficientNet, ViT).
- Early fusion saat ini hanya `raw_136` (heatmap channel). Belum coba FACS-distance/blendshape sebagai input channel tambahan.
""")

# Write notebook
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}
    },
    "nbformat": 4, "nbformat_minor": 5,
}
out = PROJECT / "notebooks" / "88_multimodal_documentation.ipynb"
out.parent.mkdir(exist_ok=True)
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Wrote {out} ({len(cells)} cells)")
