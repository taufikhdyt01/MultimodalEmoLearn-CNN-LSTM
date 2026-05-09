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

**Class scheme:** `REMAP_3 = np.array([1, 0, 2, 2, 2, 2, 0])` →
- **Positive (0):** happy, surprised
- **Neutral (1):** neutral
- **Negative (2):** sad, angry, fearful, disgusted

(Surprised dipetakan ke positive — mengikuti konvensi Russell circumplex untuk surprise sebagai high-arousal positive emotion. Sumber: nb 79, 82, 84.)

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

---

## E. Benchmark Dataset Sekunder — CK+, JAFFE, RAF-DB, KDEF (3c/7c)

**Konteks:** Skema 1 benchmark — train & test di dataset sekunder masing-masing (bukan cross-dataset).
**Sumber data:**
- 3-class: `models/benchmark/{ds}/3class/{ds}_3c_results.json`
- 7-class: `models/benchmark/{ds}/7class/{ds}_7c_results.json`

**Catatan umum:**
- 7-class: 1 run (B1) — val_f1 tidak tersimpan di format JSON lama
- CK+ dan JAFFE 3-class: 1 run (B1) — val_f1 tersimpan
- RAF-DB dan KDEF 3-class: mean 3 runs (B1/B2/B3) — val_f1 tersimpan hanya untuk EarlyFusion ke atas
- JAFFE 7-class sangat kecil (10 subjects, ~20 samples test) — angka tidak stabil

_Update: 10 Mei 2026_

---

### CK+ (Extended Cohn-Kanade)
_3-class B1 only. No EarlyFusion checkpoint._

**3-Class** *(B1 only)*

| Arch | Val F1 | Test Macro F1 | Test Weighted F1 | Test Acc |
|:---|:---:|:---:|:---:|:---:|
| `CNN` | 0.7434 | 0.7104 | 0.7297 | 0.7458 |
| `FCNN` | 0.8008 | 0.5498 | 0.5387 | 0.5424 |
| `Intermediate` | 0.6834 | 0.5752 | 0.6055 | 0.6949 |
| `CNN_TL` | 0.9700 | 0.9560 | 0.9499 | 0.9492 |
| `Intermediate_TL` | 0.9487 | 0.9704 | 0.9665 | 0.9661 |
| `LateFusion` | 0.8164 | 0.7084 | 0.6835 | 0.6780 |
| `LateFusion_TL` | 0.9449 | 0.9365 | 0.9497 | 0.9492 |

> Best test macro F1: **`Intermediate_TL`** = 0.9704

**7-Class** *(B1 only)*

| Arch | Test Macro F1 | Test Weighted F1 | Test Acc |
|:---|:---:|:---:|:---:|
| `CNN` | 0.4611 | 0.6593 | 0.7288 |
| `FCNN` | 0.3947 | 0.6143 | 0.6780 |
| `Intermediate` | 0.3160 | 0.5846 | 0.6949 |
| `CNN_TL` | 0.9127 | 0.9461 | 0.9492 |
| `Intermediate_TL` | 0.8333 | 0.8855 | 0.8814 |
| `EarlyFusion` | 0.4458 | 0.6646 | 0.6949 |
| `EarlyFusion_TL` | 0.7624 | 0.8471 | 0.8475 |
| `LateFusion` | 0.4942 | 0.6914 | 0.7797 |
| `LateFusion_TL` | 0.8352 | 0.8905 | 0.8814 |

> Best test macro F1: **`CNN_TL`** = 0.9127

---

### JAFFE (Japanese Female Facial Expression)
_3-class B1 only. Hanya 10 subjects — angka test tidak stabil (20 sampel test)._

**3-Class** *(B1 only)*

| Arch | Val F1 | Test Macro F1 | Test Weighted F1 | Test Acc |
|:---|:---:|:---:|:---:|:---:|
| `CNN` | 0.7475 | 0.4452 | 0.5632 | 0.6000 |
| `FCNN` | 0.4841 | 0.3596 | 0.4570 | 0.5000 |
| `Intermediate` | 0.4195 | 0.2837 | 0.3744 | 0.4000 |
| `CNN_TL` | 0.4710 | 0.8116 | 0.7963 | 0.8000 |
| `Intermediate_TL` | 0.7172 | 0.4382 | 0.5867 | 0.6500 |
| `LateFusion` | 0.9496 | 0.4841 | 0.6321 | 0.7000 |
| `LateFusion_TL` | 0.9552 | 0.3988 | 0.5152 | 0.5500 |

> Best test macro F1: **`CNN_TL`** = 0.8116

**7-Class** *(B1 only)*

| Arch | Test Macro F1 | Test Weighted F1 | Test Acc |
|:---|:---:|:---:|:---:|
| `CNN` | 0.3040 | 0.3192 | 0.4500 |
| `FCNN` | 0.2088 | 0.1692 | 0.2500 |
| `Intermediate` | 0.0373 | 0.0391 | 0.1500 |
| `CNN_TL` | 0.4639 | 0.4371 | 0.5000 |
| `Intermediate_TL` | 0.4473 | 0.4197 | 0.4500 |
| `EarlyFusion` | 0.2857 | 0.2667 | 0.3500 |
| `EarlyFusion_TL` | 0.0408 | 0.0429 | 0.1500 |
| `LateFusion` | 0.3143 | 0.2900 | 0.4000 |
| `LateFusion_TL` | 0.1457 | 0.1196 | 0.2000 |

> Best test macro F1: **`CNN_TL`** = 0.4639

---

### RAF-DB (Real-world Affective Faces Database)
_~15,000 samples. 3-class: mean 3 runs (B1/B2/B3). 7c: B1 only._

**3-Class** *(mean 3 runs)*

| Arch | Val F1 | Test Macro F1 | Test Weighted F1 | Test Acc |
|:---|:---:|:---:|:---:|:---:|
| `CNN` | — | 0.8028 | 0.8248 | 0.8235 |
| `FCNN` | — | 0.6938 | 0.7254 | 0.7242 |
| `Intermediate` | — | 0.7792 | 0.8049 | 0.8059 |
| `CNN_TL` | — | 0.8119 | 0.8326 | 0.8327 |
| `Intermediate_TL` | — | 0.7703 | 0.7962 | 0.7960 |
| `EarlyFusion` | 0.7905 | 0.7903 | 0.8138 | 0.8127 |
| `EarlyFusion_TL` | 0.7648 | 0.7550 | 0.7812 | 0.7807 |
| `LateFusion` | 0.8123 | 0.8078 | 0.8293 | 0.8282 |
| `LateFusion_TL` | 0.8295 | 0.8141 | 0.8344 | 0.8345 |

> Best test macro F1: **`LateFusion_TL`** = 0.8141

**7-Class** *(B1 only)*

| Arch | Test Macro F1 | Test Weighted F1 | Test Acc |
|:---|:---:|:---:|:---:|
| `CNN` | 0.7294 | 0.8128 | 0.8152 |
| `FCNN` | 0.5781 | 0.7031 | 0.7136 |
| `Intermediate` | 0.6958 | 0.7827 | 0.7854 |
| `CNN_TL` | 0.7407 | 0.8265 | 0.8304 |
| `Intermediate_TL` | 0.7440 | 0.8322 | 0.8329 |
| `EarlyFusion` | 0.7098 | 0.8044 | 0.8079 |
| `EarlyFusion_TL` | 0.6929 | 0.7864 | 0.7902 |
| `LateFusion` | 0.7191 | 0.8046 | 0.8093 |
| `LateFusion_TL` | 0.7350 | 0.8231 | 0.8294 |

> Best test macro F1: **`Intermediate_TL`** = 0.7440

---

### KDEF (Karolinska Directed Emotional Faces)
_4,900 samples, 70 subjects. 3-class: mean 3 runs (B1/B2/B3). 7c: B1 only._

**3-Class** *(mean 3 runs)*

| Arch | Val F1 | Test Macro F1 | Test Weighted F1 | Test Acc |
|:---|:---:|:---:|:---:|:---:|
| `CNN` | 0.9541 | 0.8332 | 0.8479 | 0.8481 |
| `FCNN` | 0.8304 | 0.7080 | 0.7387 | 0.7404 |
| `Intermediate` | 0.9103 | 0.8194 | 0.8321 | 0.8311 |
| `CNN_TL` | 0.9722 | 0.9059 | 0.9136 | 0.9127 |
| `Intermediate_TL` | 0.9753 | 0.9111 | 0.9169 | 0.9161 |
| `EarlyFusion` | 0.8908 | 0.7854 | 0.8097 | 0.8095 |
| `EarlyFusion_TL` | 0.9417 | 0.8770 | 0.8942 | 0.8946 |
| `LateFusion` | 0.9573 | 0.8380 | 0.8523 | 0.8526 |
| `LateFusion_TL` | 0.9724 | 0.9139 | 0.9168 | 0.9161 |

> Best test macro F1: **`LateFusion_TL`** = 0.9139

**7-Class** *(B1 only)*

| Arch | Test Macro F1 | Test Weighted F1 | Test Acc |
|:---|:---:|:---:|:---:|
| `CNN` | 0.7984 | 0.7979 | 0.8012 |
| `FCNN` | 0.6657 | 0.6629 | 0.6795 |
| `Intermediate` | 0.6710 | 0.6681 | 0.6736 |
| `CNN_TL` | 0.8333 | 0.8329 | 0.8309 |
| `Intermediate_TL` | 0.8431 | 0.8426 | 0.8427 |
| `EarlyFusion` | 0.6665 | 0.6633 | 0.6736 |
| `EarlyFusion_TL` | 0.7987 | 0.7971 | 0.7953 |
| `LateFusion` | 0.7757 | 0.7753 | 0.7774 |
| `LateFusion_TL` | 0.8358 | 0.8356 | 0.8338 |

> Best test macro F1: **`Intermediate_TL`** = 0.8431

---

## F. Ringkasan Komparatif Benchmark (Best per Dataset per Class)

> Primer tidak dimasukkan — sudah dibahas lengkap di Section A (3-class) dan Section B (7-class).

| Dataset | 3-class best | Macro F1 | 7-class best | Macro F1 |
|:---|:---:|:---:|:---:|:---:|
| **CK+** | `Intermediate_TL` | 0.9704 | `CNN_TL` | 0.9127 |
| **JAFFE** | `CNN_TL` | 0.8116 | `CNN_TL` | 0.4639 |
| **RAF-DB** | `LateFusion_TL` | 0.8141 | `Intermediate_TL` | 0.7440 |
| **KDEF** | `LateFusion_TL` | 0.9139 | `Intermediate_TL` | 0.8431 |

**Pola konsisten:**
- `Intermediate_TL` dan `LateFusion_TL` mendominasi di CK+, RAF-DB, KDEF (dataset besar + terstruktur)
- JAFFE sangat kecil (10 subjects) — hasil tidak stabil, best model berbeda-beda antar class granularity
- TL variant secara konsisten mengalahkan scratch — pre-trained ImageNet backbone membantu generalisasi
- 7-class Primer sangat rendah (0.29) karena class imbalance ekstrim; 7-class RAF-DB dan KDEF jauh lebih baik (0.74–0.84) karena distribusi lebih seimbang

_Update: 10 Mei 2026_

---

## G. Skema 2 — Cross-Dataset Inference (3-class, nb 85)

**Konteks:** Model dilatih di dataset sekunder (Skema 1), lalu di-*inference* ke test set Primer conf60 (3-class).
**Sumber:** `models/benchmark/all_3c_skema2_cross_results.json` — 36 konfigurasi (4 dataset × 9 arsitektur).

_Update: 10 Mei 2026_

---

### CK+ → Primer

| Arch | Test Macro F1 | Test Weighted F1 | Test Acc |
|:---|:---:|:---:|:---:|
| `CNN` | 0.3690 | 0.6557 | 0.6997 |
| `FCNN` | 0.3223 | 0.3029 | 0.3003 |
| `Intermediate` | 0.2731 | 0.3500 | 0.3628 |
| `CNN_TL` | 0.2795 | 0.5523 | 0.4898 |
| `Intermediate_TL` | 0.1399 | 0.2194 | 0.1830 |
| `EarlyFusion` | 0.3890 | 0.6139 | 0.5414 |
| `EarlyFusion_TL` | 0.4127 | 0.6438 | 0.6792 |
| `Late_Fusion` | **0.5052** | 0.6772 | 0.6125 |
| `Late_Fusion_TL` | 0.4448 | 0.6731 | 0.6555 |

> Best: **`Late_Fusion`** = 0.5052

---

### JAFFE → Primer

| Arch | Test Macro F1 | Test Weighted F1 | Test Acc |
|:---|:---:|:---:|:---:|
| `CNN` | 0.0373 | 0.0066 | 0.0592 |
| `FCNN` | 0.1681 | 0.0830 | 0.2002 |
| `Intermediate` | 0.1710 | 0.0847 | 0.1905 |
| `CNN_TL` | 0.0373 | 0.0066 | 0.0592 |
| `Intermediate_TL` | 0.0682 | 0.0264 | 0.0646 |
| `EarlyFusion` | 0.0373 | 0.0066 | 0.0592 |
| `EarlyFusion_TL` | 0.1122 | 0.0953 | 0.1819 |
| `Late_Fusion` | 0.1524 | 0.0790 | 0.1442 |
| `Late_Fusion_TL` | **0.2107** | 0.1065 | 0.2110 |

> Best: **`Late_Fusion_TL`** = 0.2107

---

### RAF-DB → Primer

| Arch | Test Macro F1 | Test Weighted F1 | Test Acc |
|:---|:---:|:---:|:---:|
| `CNN` | 0.1288 | 0.0955 | 0.1281 |
| `FCNN` | 0.2235 | 0.1735 | 0.1485 |
| `Intermediate` | 0.2491 | 0.3297 | 0.2605 |
| `CNN_TL` | **0.4442** | 0.5929 | 0.5285 |
| `Intermediate_TL` | 0.2363 | 0.3491 | 0.2982 |
| `EarlyFusion` | 0.2858 | 0.3081 | 0.3046 |
| `EarlyFusion_TL` | 0.4395 | 0.5765 | 0.5350 |
| `Late_Fusion` | 0.1254 | 0.0860 | 0.1238 |
| `Late_Fusion_TL` | 0.4442 | 0.5929 | 0.5285 |

> Best: **`CNN_TL`** = 0.4442 (tie dengan `Late_Fusion_TL`)

---

### KDEF → Primer

| Arch | Test Macro F1 | Test Weighted F1 | Test Acc |
|:---|:---:|:---:|:---:|
| `CNN` | 0.1245 | 0.1032 | 0.1238 |
| `FCNN` | 0.1128 | 0.0678 | 0.2002 |
| `Intermediate` | 0.1112 | 0.0668 | 0.2002 |
| `CNN_TL` | 0.0445 | 0.0123 | 0.0603 |
| `Intermediate_TL` | 0.2072 | 0.2206 | 0.1873 |
| `EarlyFusion` | 0.0868 | 0.0548 | 0.1012 |
| `EarlyFusion_TL` | 0.0925 | 0.0477 | 0.1066 |
| `Late_Fusion` | 0.1142 | 0.0668 | 0.1206 |
| `Late_Fusion_TL` | **0.2179** | 0.1112 | 0.1970 |

> Best: **`Late_Fusion_TL`** = 0.2179

---

### Ringkasan Skema 2 (Best per Source)

| Source Dataset | Best Arch | Macro F1 | Keterangan |
|:---|:---:|:---:|:---|
| **CK+** | `Late_Fusion` | **0.5052** | Tertinggi — karakteristik lab-controlled mirip Primer |
| **RAF-DB** | `CNN_TL` | 0.4442 | Dataset besar in-the-wild, TL membantu |
| **KDEF** | `Late_Fusion_TL` | 0.2179 | Rendah — Scandinavian subjects, domain gap besar |
| **JAFFE** | `Late_Fusion_TL` | 0.2107 | Sangat rendah — hanya 10 subjects Jepang |

**Analisis:**
- CK+ memberikan generalisasi terbaik ke Primer — setting lab-controlled paling mirip kondisi pengambilan data Primer
- RAF-DB cukup baik (0.44) meskipun in-the-wild karena ukuran dataset besar → feature lebih robust
- JAFFE dan KDEF rendah karena domain gap etnis + ukuran kecil
- `Late_Fusion` / `Late_Fusion_TL` dominan di 3/4 source dataset — weighted ensemble membantu adaptasi cross-domain
- Semua nilai << Skema 1 self-trained (0.81–0.97) → domain gap benchmark→Primer nyata dan signifikan