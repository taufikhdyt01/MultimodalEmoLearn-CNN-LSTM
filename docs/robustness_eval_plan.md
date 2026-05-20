# Robustness Evaluation — Gap Analysis & Plan

> Status snapshot per **2026-05-20**. Sumber: `all_54_configs_metrics.md` §H,
> `all_metrics_tables.md`, `docs/significance_tests.md`.

---

## A. Yang Sudah Dilakukan

### §H di `all_54_configs_metrics.md` — Primer conf60 (single split + multi-split)

- **Single-split** (seed=42, 29 train / 3 val / 5 test user) untuk semua 54 config.
- **Multi-split** untuk **3 arsitektur top-3 lama** (raw_136 + MediaPipe basis):

| Strategi | n | Arsitektur (3c) | Arsitektur (7c) |
|---|:---:|---|---|
| LOSO | 33 fold | FCNN, Intermediate_TL, Late_Fusion_TL | CNN, Intermediate_TL, Late_Fusion |
| 5-Fold CV (subject-wise) | 5 | (sda) | (sda) |
| Random Split (per-sample) | 5 seed | (sda) | (sda) |

Output: `models/frontonly_conf60/{3class,7class}/{loso,crossval,randomsplit}/*.json`

**Bukti hasil yang sudah ada di §H.1 & §H.2:**
- 3c LOSO mean ± std: 0.44–0.46 ± 0.10–0.12 macro F1
- 7c LOSO mean ± std: 0.17–0.31 ± 0.09–0.14 macro F1
- Trade-off ranking: Random > 5-Fold > LOSO (subject-leakage gap konsisten)

### `docs/significance_tests.md` — Statistical test (baru, 2026-05-20)

McNemar + paired bootstrap CI untuk **Late Fusion vs Best Unimodal** di 5 dataset
(primer + 4 sekunder), 6 group per dataset = 30 perbandingan total.

Implementasi: `scripts/compute_significance_tests.py` — rekonstruksi Late Fusion
softmax dari unimodal cache + `best_image_weight`/`best_landmark_weight`. No
re-training.

---

## B. Gap yang Tersisa

### B1. Top-3 arsitektur di §H **bukan lagi top-3 setelah unified sweep**

§H pakai naming lama (raw_136 + MP basis). Setelah unified sweep, top-3 primer
berubah signifikan:

| Scheme | Top-3 LAMA (§H) | Top-3 BARU (dari unified sweep) |
|---|---|---|
| 3c | FCNN, Intermediate_TL, Late_Fusion_TL (semua raw_136 MP) | **Late Fusion tl facs_28 FA** (mf1 0.7604), **Landmark facs_28 FA FCNN** (0.7585), **Intermediate scratch facs_28 FA** (0.7581) |
| 7c | CNN, Intermediate_TL, Late_Fusion (raw_136 MP) | **Intermediate scratch facs_28 FA** (mf1 0.3363), **Late Fusion tl FB80 FA** (0.3332), **Landmark FB80 FA FCNN** (0.3331) |

**Implikasi:** LOSO/CV/Random hasil §H **tidak mewakili top performer aktual**.
Reviewer yang teliti akan menemukan diskrepansi ini.

### B2. Sekunder (KDEF/RAF-DB/CK+/JAFFE) tidak punya LOSO/CV

Sekunder hanya punya single-split eval. Acceptable karena fungsinya sebagai
**cross-dataset benchmark** (external validation), bukan internal robustness.
Tetap, beberapa reviewer mungkin tanya stabilitas hasil sekunder dengan multi-seed.

### B3. Statistical test untuk Early/Intermediate fusion

`significance_tests.md` saat ini cuma cover **Late Fusion vs Unimodal** karena
Early/Intermediate tidak menyimpan test softmax cache (cuma summary metrics di
`results.json`). Untuk extend, perlu re-run inference dari `.pt` checkpoint.

---

## C. Rekomendasi Aksi

### C.1 Prioritas TINGGI (wajib sebelum submit)

**[ ] Update §H dengan top-3 baru — LOSO + 5-Fold CV untuk arsitektur baru**

Top-3 baru per scheme (6 config total):
- **3c**: Late Fusion tl facs_28 FA, Landmark facs_28 FA FCNN, Intermediate scratch facs_28 FA
- **7c**: Intermediate scratch facs_28 FA, Late Fusion tl FB80 FA, Landmark FB80 FA FCNN

Compute estimate:
- LOSO: 33 fold × 6 config = 198 runs. Avg ~10 min/run = ~33 jam (1 GPU)
- 5-Fold CV: 5 fold × 6 config = 30 runs ≈ 5 jam
- Random Split: SKIP (sudah ada untuk old top-3, dan reviewer biasanya hepi dengan CV+LOSO)

Script: ekstensi `scripts/run_loso.py` untuk support `--feature facs_28 --source faceapi`.
Saat ini script hardcode ke raw_136 MP. Perlu adapter ke unified data pipeline.

**[x] Statistical significance test cross-dataset — DONE**

`docs/significance_tests.md` — McNemar + bootstrap untuk Late Fusion vs Unimodal
di 5 dataset.

### C.2 Prioritas SEDANG (kalau ada waktu)

**[ ] Re-run inference Early/Intermediate fusion dari checkpoint .pt**

Buat script `scripts/cache_fusion_predictions.py` yang load `.pt` checkpoint per
fusion config, jalankan inference di test set, dump softmax ke
`models/.../intermediate_fusion_cache/*_test.npy`. Setelah itu re-run
`compute_significance_tests.py` untuk cover early/intermediate juga.

Compute: ~180 fusion configs × inference (~30 detik/config dengan batch). ~2 jam
total. Bisa dijalankan di GPU manapun (inference bukan training).

**[ ] Multi-seed sekunder (3 seed sanity check)**

Run B1 single-split untuk top fusion config di 4 sekunder dengan 3 seed berbeda.
~12 runs total, ~3 jam.

### C.3 Prioritas RENDAH (skip kecuali reviewer minta)

**[ ] LOSO untuk sekunder** — multiplies compute 30–70x per dataset. Skip;
cross-dataset benchmark sudah lebih kuat.

**[ ] Random Split sweep dengan 10+ seed** — diminishing return setelah ada
5-Fold CV.

---

## D. Workflow Eksekusi yang Diusulkan

```
[Phase 1: cepat] (~1 hari kerja)
  1. Cache Early/Intermediate fusion predictions dari .pt (~2 jam)
  2. Re-run compute_significance_tests.py (extended) — McNemar+bootstrap
     untuk semua fusion family vs unimodal (~5 menit)

[Phase 2: heavy compute] (~2 hari GPU)
  3. Adapter LOSO/CV ke unified data pipeline (~half day code)
  4. Jalankan LOSO untuk top-3 baru per scheme — paralel 3 GPU (~12 jam wall)
  5. Jalankan 5-Fold CV (~5 jam paralel)
  6. Update §H tabel dengan angka baru

[Phase 3: dokumentasi]
  7. Update `feature_design_qa.md` dengan robustness discussion
  8. Generate confidence interval plots untuk paper figures
```

---

## E. Komparasi: "Cukup" vs "Strong" defense

| Skenario | Cukup untuk thesis | Strong untuk top-tier paper |
|---|---|---|
| Single-split top-1 saja | ❌ | ❌ |
| Single-split semua + best-effort cross-dataset | ⚠️ | ❌ |
| **Yang sekarang: §H lama + cross-dataset + significance test (Late Fusion)** | ✅ | ⚠️ |
| **Setelah Phase 1+2 selesai** | ✅✅ | ✅ |
| Phase 1+2 + multi-seed sekunder + per-class CI | ✅✅ | ✅✅ |

---

## F. Lihat juga

- `feature_design_qa.md` — desain feature × source × fusion (kenapa kombinasi
  tertentu tidak ada di tabel)
- `all_metrics_tables.md` — master tabel hasil eksperimen
- `all_54_configs_metrics.md` §H — robustness existing (top-3 lama)
- `significance_tests.md` — McNemar/bootstrap Late Fusion vs Unimodal
