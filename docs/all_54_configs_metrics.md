# Tabel Lengkap 54 Konfigurasi — Primer conf60 (3-class & 7-class)

Dataset: `data/dataset_frontonly_conf60/` (37 users, 6795 samples — 5287 train / 579 val / 929 test).

**Catatan metrik:**
- `Val Macro F1` = macro-F1 pada validation set (3 user) — dipakai untuk model selection / early stopping. Tidak dilaporkan sebagai hasil utama.
- `Test Macro F1` = macro-F1 pada test set (5 user) — **angka utama untuk laporan**.
- `Test Weighted F1` = test F1 dengan bobot proporsional terhadap support tiap kelas (mengakomodasi class imbalance).
- `Test Acc` = accuracy. **Catatan:** untuk single-label multi-class, `micro_f1 ≡ accuracy ≡ micro_precision ≡ micro_recall` — kolom Test Micro F1 dihilangkan karena redundant.
- `Best Epoch` = epoch dengan val_f1 terbaik (early stopping).
- `—` = metrik tidak tersimpan di JSON saat eksperimen lama (lihat catatan per section).

Sumber:
- 3-class: `models/frontonly_conf60/3class/{all_results_3class.json, scratch_all_results.json}`
- 7-class: `models/frontonly_conf60/{7class/, 7class_tl/, early_fusion/}*.json`

---

## A. 3-Class (Negative / Neutral / Positive) — 27 konfigurasi

**Class scheme:** REMAP_3 = `{0:Neutral→1, 1:Happy→0, 2-6:Sad/Angry/Fearful/Disgusted/Surprised→2}` (Positive=0, Neutral=1, Negative=2).

**Konvensi B1/B2/B3:** Backbone variants (3 arsitektur backbone berbeda).

Diurutkan berdasar **val_macro_f1 descending**.

| Rank | Config | Category | Val Macro F1 | Test Macro F1 | Test Weighted F1 | Test Acc | Best Epoch |
|:----:|:-------|:--------:|-------------:|--------------:|-----------------:|---------:|-----------:|
| 1 | `Late_Fusion_TL_B3` | TL | 0.6229 | 0.6370 | 0.7947 | 0.7836 | — |
| 2 | `FCNN_B3` | Hybrid | 0.6193 | 0.6342 | 0.7636 | 0.7492 | 45 |
| 3 | `Late_Fusion_TL_B1` | TL | 0.6093 | 0.6526 | 0.8128 | 0.7966 | — |
| 4 | `Late_Fusion_scratch_B3` | Scratch | 0.6085 | 0.5393 | 0.7061 | 0.6825 | — |
| 5 | `Late_Fusion_scratch_B1` | Scratch | 0.6065 | 0.6713 | 0.8280 | 0.8224 | — |
| 6 | `FCNN_B1` | Hybrid | 0.6025 | 0.5893 | 0.7596 | 0.7406 | 16 |
| 7 | `Late_Fusion_scratch_B2` | Scratch | 0.6000 | 0.5876 | 0.7443 | 0.7244 | — |
| 8 | `Late_Fusion_TL_B2` | TL | 0.5914 | 0.6456 | 0.7878 | 0.7740 | — |
| 9 | `Intermediate_scratch_B3` | Scratch | 0.5871 | 0.6235 | 0.7981 | 0.7912 | 6 |
| 10 | `Intermediate_scratch_B1` | Scratch | 0.5673 | 0.5910 | 0.8013 | 0.8019 | 13 |
| 11 | `FCNN_B2` | Hybrid | 0.5641 | 0.5750 | 0.7302 | 0.7018 | 49 |
| 12 | `Intermediate_TL_B2` | TL | 0.5593 | 0.6645 | 0.8185 | 0.8149 | 17 |
| 13 | `Intermediate_TL_B1` | TL | 0.5537 | 0.6856 | 0.8034 | 0.7901 | 20 |
| 14 | `Early_Fusion_TL_B1` | TL | 0.5371 | 0.5872 | 0.7735 | 0.7826 | 13 |
| 15 | `Early_Fusion_scratch_B1` | Scratch | 0.5216 | 0.5158 | 0.7433 | 0.7427 | 45 |
| 16 | `Early_Fusion_scratch_B2` | Scratch | 0.5173 | 0.5699 | 0.7504 | 0.7352 | 17 |
| 17 | `CNN_TL_B2` | TL | 0.5150 | 0.5161 | 0.6099 | 0.5813 | 3 |
| 18 | `Early_Fusion_TL_B3` | TL | 0.5088 | 0.6988 | 0.8348 | 0.8202 | 1 |
| 19 | `Early_Fusion_TL_B2` | TL | 0.5039 | 0.5840 | 0.6759 | 0.6502 | 4 |
| 20 | `Intermediate_TL_B3` | TL | 0.5005 | 0.6891 | 0.8342 | 0.8278 | 8 |
| 21 | `CNN_TL_B3` | TL | 0.4953 | 0.7055 | 0.8499 | 0.8396 | 2 |
| 22 | `CNN_TL_B1` | TL | 0.4927 | 0.6340 | 0.7996 | 0.7912 | 1 |
| 23 | `CNN_scratch_B1` | Scratch | 0.4874 | 0.6122 | 0.8047 | 0.8052 | 22 |
| 24 | `Early_Fusion_scratch_B3` | Scratch | 0.4825 | 0.5070 | 0.7356 | 0.7266 | 39 |
| 25 | `Intermediate_scratch_B2` | Scratch | 0.4810 | 0.4942 | 0.7291 | 0.7212 | 22 |
| 26 | `CNN_scratch_B3` | Scratch | 0.4531 | 0.5165 | 0.6861 | 0.6588 | 5 |
| 27 | `CNN_scratch_B2` | Scratch | 0.4477 | 0.4644 | 0.6112 | 0.5759 | 28 |

**Ringkasan 3-class val_macro_f1:** min=0.4477, max=0.6229, mean=0.5421 ± 0.0542 (n=27).

**Ringkasan 3-class test_macro_f1:** min=0.4644, max=0.7055, mean=0.5997 ± 0.0679 (n=27).

### Top-3 (by val_macro_f1)

1. **`Late_Fusion_TL_B3`** — val=0.6229, test=0.6370
2. **`FCNN_B3`** — val=0.6193, test=0.6342
3. **`Late_Fusion_TL_B1`** — val=0.6093, test=0.6526

### Top-3 (by test_macro_f1)

1. **`CNN_TL_B3`** — test=0.7055, val=0.4953
2. **`Early_Fusion_TL_B3`** — test=0.6988, val=0.5088
3. **`Intermediate_TL_B3`** — test=0.6891, val=0.5005

> Konsistensi val vs test ranking lemah (Spearman ρ=0.350, top-3 overlap 0/3) — kemungkinan karena val set kecil (3 user, 579 sampel) tidak representatif.

---

## B. 7-Class (Neutral/Happy/Sad/Angry/Fearful/Disgusted/Surprised) — 27 konfigurasi

**Konvensi B1/B2/B3:** Training scenarios — `B1 Baseline` (no class weights), `B2 Class Weights`, `B3 Weights+Aug` (class weights + augmentation). EarlyFusion 7c memakai konvensi backbone (B1/B2/B3 = backbone variants), inkonsisten dengan 6 family lain.

**⚠️ Val Macro F1 hanya tersimpan untuk LateFusion (scratch & TL)** — notebook awal 7-class (nb 43, 44, 21, 26, 28) hanya menyimpan test metrics, tidak val_f1.

Diurutkan berdasar **test_macro_f1 descending** (val tidak konsisten tersedia).

| Rank | Config | Category | Val Macro F1 | Test Macro F1 | Test Weighted F1 | Test Acc |
|:----:|:-------|:--------:|-------------:|--------------:|-----------------:|---------:|
| 1 | `EarlyFusion_TL_B3` | TL | — | 0.3330 | 0.7729 | 0.7535 |
| 2 | `Intermediate_TL / B3 Weights+Aug` | TL | — | 0.2917 | 0.8256 | 0.8245 |
| 3 | `Intermediate_TL / B2 Class Weights` | TL | — | 0.2833 | 0.8249 | 0.8245 |
| 4 | `CNN scratch / B1 Baseline` | Scratch | — | 0.2774 | 0.8092 | 0.8105 |
| 5 | `Intermediate_TL / B1 Baseline` | TL | — | 0.2769 | 0.7997 | 0.7922 |
| 6 | `CNN_TL / B1 Baseline` | TL | — | 0.2734 | 0.7820 | 0.7933 |
| 7 | `LateFusion scratch / B1 Baseline` | Scratch | 0.2887 | 0.2701 | 0.8117 | 0.8159 |
| 8 | `EarlyFusion_B3` | Scratch | — | 0.2638 | 0.7260 | 0.6803 |
| 9 | `Intermediate scratch / B1 Baseline` | Scratch | — | 0.2612 | 0.7914 | 0.7922 |
| 10 | `EarlyFusion_TL_B1` | TL | — | 0.2533 | 0.7224 | 0.7126 |
| 11 | `CNN scratch / B3 Weights+Aug` | Scratch | — | 0.2531 | 0.7820 | 0.7847 |
| 12 | `LateFusion_TL / B2 Class Weights` | TL | 0.2706 | 0.2490 | 0.7810 | 0.7815 |
| 13 | `LateFusion scratch / B2 Class Weights` | Scratch | 0.2697 | 0.2478 | 0.7776 | 0.7772 |
| 14 | `Intermediate scratch / B2 Class Weights` | Scratch | — | 0.2469 | 0.7843 | 0.7793 |
| 15 | `EarlyFusion_TL_B2` | TL | — | 0.2467 | 0.6634 | 0.6362 |
| 16 | `EarlyFusion_B1` | Scratch | — | 0.2464 | 0.7861 | 0.7944 |
| 17 | `FCNN / B2 Class Weights` | Scratch | — | 0.2437 | 0.7674 | 0.7653 |
| 18 | `CNN_TL / B2 Class Weights` | TL | — | 0.2433 | 0.7462 | 0.7503 |
| 19 | `CNN_TL / B3 Weights+Aug` | TL | — | 0.2405 | 0.7969 | 0.8073 |
| 20 | `CNN scratch / B2 Class Weights` | Scratch | — | 0.2396 | 0.7666 | 0.7740 |
| 21 | `LateFusion_TL / B1 Baseline` | TL | 0.2663 | 0.2382 | 0.7836 | 0.7901 |
| 22 | `LateFusion_TL / B3 Weights+Aug` | TL | 0.2735 | 0.2319 | 0.7795 | 0.7621 |
| 23 | `FCNN / B1 Baseline` | Scratch | — | 0.2317 | 0.7650 | 0.7675 |
| 24 | `Intermediate scratch / B3 Weights+Aug` | Scratch | — | 0.2294 | 0.7541 | 0.7750 |
| 25 | `FCNN / B3 Weights+Aug` | Scratch | — | 0.2224 | 0.7580 | 0.7395 |
| 26 | `LateFusion scratch / B3 Weights+Aug` | Scratch | 0.2690 | 0.2224 | 0.7580 | 0.7395 |
| 27 | `EarlyFusion_B2` | Scratch | — | 0.2049 | 0.5516 | 0.5199 |

**Ringkasan 7-class val_macro_f1:** min=0.2663, max=0.2887, mean=0.2730 ± 0.0081 (n=6/27 — hanya LateFusion variants).

**Ringkasan 7-class test_macro_f1:** min=0.2049, max=0.3330, mean=0.2527 ± 0.0259 (n=27).

### Top-3 (by test_macro_f1)

1. **`EarlyFusion_TL_B3`** — test=0.3330, val=—
2. **`Intermediate_TL / B3 Weights+Aug`** — test=0.2917, val=—
3. **`Intermediate_TL / B2 Class Weights`** — test=0.2833, val=—

> Macro F1 7-class sangat rendah (~0.23-0.33) karena class imbalance ekstrim: dari 6795 sampel, neutral 5691, happy 651, sad 361, dan 4 kelas minoritas total <100 sampel (angry 32, surprised 39, disgusted 16, fearful 5).

---

## C. Catatan & Limitasi

1. **Micro F1 = Accuracy** untuk single-label multi-class. Beberapa file JSON menyimpan `test_micro_f1` secara eksplisit; nilainya selalu identik dengan `test_accuracy`.
2. **Single-seed runs.** Semua angka di tabel adalah hasil 1 kali training (seed=42). Variance single-seed estimasi 0.05-0.09 untuk macro F1 (sumber: `docs/bimbingan_progress.md:2000`). Gap antar config kecil sering kali within-noise.
3. **Val set kecil (3 user, 579 sampel)** — ranking val tidak prediktif untuk test (ρ=0.35 di 3-class). Justifikasi perlu LOSO/5-Fold CV untuk hasil yang stabil.
4. **7-class metrik incomplete** — val_macro_f1 tidak konsisten tersimpan karena kelemahan pipeline notebook awal. Kalau perlu val_f1 lengkap untuk model selection 7c, harus re-train (~10-15 jam GPU).
5. **EarlyFusion 7c menggunakan konvensi backbone** (B1/B2/B3 = ResNet18/34/50) sedangkan 6 family lain memakai konvensi training scenarios — perbandingan langsung antar B1/B2/B3 di 7-class harus hati-hati.
6. **`best_cnn_weight`** (kolom yang ada di Late Fusion saja) = bobot optimal untuk weighted ensemble CNN+FCNN, di-tune di val.

## D. Sumber Data

```
models/frontonly_conf60/
├── 3class/
│   ├── all_results_3class.json     # 15 TL+hybrid configs (FCNN + 4 TL × B1/B2/B3)
│   └── scratch_all_results.json    # 12 scratch configs (CNN, Intermediate, EarlyFusion, LateFusion × B1/B2/B3)
├── 7class/                         # scratch (CNN, FCNN, Intermediate, LateFusion × 3 scenario)
├── 7class_tl/                      # TL (CNN_TL, Intermediate_TL, LateFusion_TL × 3 scenario)
└── early_fusion/early_fusion_7c_results.json   # EarlyFusion + EarlyFusion_TL × B1/B2/B3 (6 configs)
```

Generated: 2026-05-09 by Claude Opus 4.7.