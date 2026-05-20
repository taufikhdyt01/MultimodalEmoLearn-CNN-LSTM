# Hasil Unimodal: Citra Wajah dan Facial Landmark

> 📋 **Detail metric lengkap (macro_f1 + weighted_f1 + accuracy) untuk semua run + placeholder ⏳ untuk yang belum dijalankan:** lihat **[`docs/all_metrics_tables.md`](all_metrics_tables.md)** (auto-generated, regenerate dengan `python scripts/build_results_tables.py`).
>
> 🎯 **Note:** unimodal **FACS_28 × FA × FCNN × B1 = 0.7585** (3c) dan **FB80 × FA × FCNN × B2 = 0.3331** (7c) **sekarang sudah BUKAN ceiling absolut** — Late Fusion + feature variants berhasil mengalahkan (slightly): 3c → 0.7604, 7c → 0.3332. Lihat `multimodal_results.md` Section 3.3. Tapi gap kecil (<0.5%), confirming bahwa **landmark FA dengan FACS_28/FB80 = backbone yang sangat strong** untuk task ini.
>
> ✅ **Unimodal primer selesai 16 Mei 2026** — 96 cell master table fully filled (Section 2).
>
> ✅ **Cross-dataset B1 (KDEF, RAF-DB, CK+, JAFFE) selesai 17 Mei 2026** — Section 6.
>
> ✅ **Derived sweep (blendshape_52 + FB80) untuk 4 dataset sekunder selesai 17 Mei 2026** (32 runs).
>
> 🟡 **B2/B3 extension untuk 4 dataset sekunder** sedang queued di GPU 2 chain (akan selesai ~10 jam).

**Dataset primer:** `data/dataset_frontonly_conf60/` (Primer conf60, train/val/test asli per-user split, confidence ≥ 0.6) — train 5287/29 user, val 579/5 user, test 929/3 user, total 6795 sample dari 37 user.

**Dataset sekunder:** KDEF (2353/294/294, balanced), RAF-DB (9404/2347/2936, ~5:1 imbalance), CK+ (508/69/59, neutral-heavy ~5:1), JAFFE (152/21/40, perfectly balanced).

---

## 1. Setup & Protokol

### Dimensi eksperimen

| Dimensi | Nilai |
|---|---|
| **Skema kelas** | 3c (positive/neutral/negative), 7c (neutral/happy/sad/angry/fearful/disgusted/surprised) |
| **Sumber landmark** | MediaPipe FaceLandmarker v2 (478→68 dlib-mapping), face-api.js (native 68 dlib via TinyFace+Landmark68) |
| **Sumber citra** | Face crop 224×224×3 dari MediaPipe bbox (OBS recording). face-api.js tidak punya data citra independen |
| **Arsitektur citra** | CNN Scratch (4 conv blocks, 27M params), CNN_TL (ResNet-18 ImageNet, 11M params) |
| **Arsitektur landmark** | FCNN (5-dense), CNN1D (Conv1d on (2, 68) sequence), CNN1D-FACS (decomposed input untuk 28-dim FACS) |
| **Representasi landmark** | Raw 2D coords (136), FACS Euclidean distance (28), ARKit Blendshape (52, MP only), FACS+Blendshape concat (80) |

### Definisi B1 / B2 / B3 (Unified Protocol, 14 Mei 2026)

| Skenario | Sampler | Augmentation per batch |
|---|---|---|
| **B1** | Shuffle uniform | None |
| **B2** | `WeightedRandomSampler` (class-balanced, prob ∝ 1/class_count) | None |
| **B3** | `WeightedRandomSampler` | Random aug per `__getitem__` call (true on-the-fly) |

Separasi concern: **balancing pakai sampler** (probabilistic oversample minoritas), **diversity pakai aug** (random transform). Tidak campur.

**Augmentation per modalitas (B3):**

- **Landmark** (`src/training/landmark_aug.py`): hflip with proper 68-pt left/right index swap (p=0.5) → rotate ±10° around face center → scale 0.95-1.05 → translate ±2% → per-coord Gaussian noise σ=0.005
- **Image** (`src/training/image_aug.py`): hflip (p=0.5) → rotate ±10° reflect-pad → brightness ±10% → contrast ×0.9-1.1

### Hyperparameters (semua run Unified Protocol)

Adam, lr=1e-3 (lr=1e-4 untuk CNN_TL ResNet-18 finetune), batch=32, epochs_max=50, patience=15, seed=42, loss=CrossEntropyLoss (no class weight — sampler yang handle).

### Scripts

| Sweep | Script | Runs | Status |
|---|---|---|---|
| Landmark (raw_136, facs_28) | `scripts/run_unified_landmark.py` | 48 | ✅ done 14 Mei |
| Derived (blendshape_52, fb_80) | `scripts/run_unified_derived.py` | 36 | ✅ done 14 Mei |
| Image (CNN scratch, CNN_TL) | `scripts/run_unified_image.py` | 12 | ✅ done 16 Mei |

---

## 2. Master Table — Semua Hasil Unimodal (macro_f1)

Tabel ini adalah **single source of truth** untuk semua kombinasi feature × source × arch × scenario × scheme. Angka = test macro_f1. Bold = best per row. ⭐ = top global per scheme.

| # | Feature | Dim | Source | Arch | 3c-B1 | 3c-B2 | 3c-B3 | 7c-B1 | 7c-B2 | 7c-B3 |
|---:|---|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | Raw coords | 136 | MP | FCNN | 0.5087 | 0.5104 | **0.5652** | 0.2255 | 0.1827 | **0.2392** |
| 2 | Raw coords | 136 | MP | CNN1D | 0.5336 | 0.5163 | **0.7005** | 0.2471 | **0.2673** | 0.2226 |
| 3 | Raw coords | 136 | FA | FCNN | **0.7119** | 0.6827 | 0.6928 | **0.3122** | 0.2386 | 0.2885 |
| 4 | Raw coords | 136 | FA | CNN1D | 0.6627 | 0.6885 | **0.7358** | ⭐ **0.3256** | 0.2309 | 0.2218 |
| 5 | FACS distance | 28 | MP | FCNN | 0.5594 | 0.6122 | **0.6246** | 0.1972 | **0.2351** | 0.2038 |
| 6 | FACS distance | 28 | MP | CNN1D | 0.5274 | 0.5495 | **0.5613** | **0.2239** | 0.2077 | 0.1769 |
| 7 | FACS distance | 28 | FA | FCNN | ⭐ **0.7585** | 0.7489 | 0.7372 | **0.3090** | 0.3036 | 0.2907 |
| 8 | FACS distance | 28 | FA | CNN1D | 0.7143 | 0.6997 | **0.7120** | 0.3114 | **0.3211** | 0.3117 |
| 9 | Blendshape | 52 | MP | FCNN | 0.5688 | **0.6474** | 0.6211 | **0.2782** | 0.2177 | 0.2654 |
| 10 | Blendshape | 52 | MP | CNN1D | 0.5311 | 0.6016 | **0.6340** | **0.2411** | 0.2390 | 0.2178 |
| 11 | FACS+Blendshape | 80 | MP | FCNN | 0.5600 | 0.6363 | **0.6376** | 0.2631 | **0.2969** | 0.2654 |
| 12 | FACS+Blendshape | 80 | MP | CNN1D | 0.6175 | **0.6382** | 0.5910 | **0.2849** | 0.2496 | 0.2244 |
| 13 | FACS+Blendshape | 80 | FA | FCNN | **0.7350** | 0.7116 | 0.6603 | 0.2687 | ⭐ **0.3331** | 0.2886 |
| 14 | FACS+Blendshape | 80 | FA | CNN1D | 0.6936 | 0.7156 | **0.7261** | **0.3279** | 0.2820 | 0.3079 |
| 15 | Image 224×224×3 | — | MP-crop | CNN scratch | 0.5095 | 0.4124 | **0.5700** | 0.2432 | 0.1903 | **0.2582** |
| 16 | Image 224×224×3 | — | MP-crop | CNN_TL (ResNet-18) | 0.6348 | **0.7107** | 0.6911 | 0.2763 | **0.2964** | 0.2833 |

**Yang langsung terlihat dari tabel:**
- **96 dari 96 cell terisi** (100% coverage). Image sweep selesai 16 Mei 2026.
- **Top global:** 3c-B1 FA × FACS×FCNN = 0.7585. 7c-B2 FA × FB80×FCNN = 0.3331. 7c-B1 FA × Raw×CNN1D = 0.3256.
- **Best image-based:** 3c CNN_TL × B2 = 0.7107 (peringkat ~11 di leaderboard 3c, tidak masuk top-10). 7c CNN_TL × B2 = 0.2964 (tidak masuk top-10 7c). **Image-based < landmark-based** di kedua skema — landmark (terutama FA) lebih informatif untuk task emosi ini.
- Detail metrik lengkap (`weighted_f1`, `accuracy`, per-class report, confusion matrix, per-epoch history) tersimpan di JSON — lihat Section **Sumber data**.

---

## 3. Top-10 Leaderboard

Diturunkan dari master table.

### 3-class macro_f1

| # | Feature | Source | Arch | Scn | mf1 |
|---|---|---|---|---|:---:|
| 1 ⭐ | FACS distance (28) | FA | FCNN | B1 | **0.7585** |
| 2 | FACS distance (28) | FA | FCNN | B2 | 0.7489 |
| 3 | FACS distance (28) | FA | FCNN | B3 | 0.7372 |
| 4 | Raw coords (136) | FA | CNN1D | B3 | 0.7358 |
| 5 | FACS+Blendshape (80) | FA | FCNN | B1 | 0.7350 |
| 6 | FACS+Blendshape (80) | FA | CNN1D | B3 | 0.7261 |
| 7 | FACS+Blendshape (80) | FA | CNN1D | B2 | 0.7156 |
| 8 | FACS distance (28) | FA | CNN1D | B1 | 0.7143 |
| 9 | FACS distance (28) | FA | CNN1D | B3 | 0.7120 |
| 10 | Raw coords (136) | FA | FCNN | B1 | 0.7119 |

### 7-class macro_f1

| # | Feature | Source | Arch | Scn | mf1 |
|---|---|---|---|---|:---:|
| 1 ⭐ | FACS+Blendshape (80) | FA | FCNN | B2 | **0.3331** |
| 2 | FACS+Blendshape (80) | FA | CNN1D | B1 | 0.3279 |
| 3 | Raw coords (136) | FA | CNN1D | B1 | 0.3256 |
| 4 | FACS distance (28) | FA | CNN1D | B2 | 0.3211 |
| 5 | Raw coords (136) | FA | FCNN | B1 | 0.3122 |
| 6 | FACS distance (28) | FA | CNN1D | B3 | 0.3117 |
| 7 | FACS distance (28) | FA | CNN1D | B1 | 0.3114 |
| 8 | FACS distance (28) | FA | FCNN | B1 | 0.3090 |
| 9 | FACS+Blendshape (80) | FA | CNN1D | B3 | 0.3079 |
| 10 | FACS distance (28) | FA | FCNN | B2 | 0.3036 |

**Catatan:** image-based (baris 15-16 master table) sudah lengkap tapi **tidak ada yang masuk top-10** di kedua skema (best image 3c = CNN_TL × B2 = 0.7107 ≈ rank #11; best image 7c = CNN_TL × B2 = 0.2964 < 0.3036 threshold top-10). Landmark-based (terutama FA-source) dominan di leaderboard.

---

## 4. Insight & Interpretasi

### 4.1 Landmark source: face-api.js konsisten unggul MediaPipe
Top-10 di **kedua skema** 3c & 7c semuanya pakai FA-landmark. Δ macro_f1 (FA - MP) di 3c B1 berkisar +0.11 sampai +0.20 antar representasi.

**Hipotesis:** face-api.js (TinyFace+Landmark68) pakai koordinat di **frame asli** sebelum crop. MediaPipe (478→68 mapping) pakai koordinat di **face-crop 224×224**. Frame asli punya info scale/pose preserved; face-crop kompresi info. Plus FA native dlib 68 convention (semantically konsisten), MP 478→68 cuma index mapping approximate.

### 4.2 Trade-off antara dimensi feature dan semantik
- **FACS distance (28-dim)** dengan FA + FCNN — paling efisien, best 3c (0.7585) dengan parameter sedikit
- **Raw coords (136-dim)** dengan FA + CNN1D + B3 — second best 3c (0.7358), aug membantu
- **Blendshape (52-dim)** standalone underperform — terikat akurasi MP, no FA equivalent
- **FACS+Blendshape (80-dim)** combo — menang di 7c B2 (0.3331) tapi tidak konsisten unggul di 3c

### 4.3 Arsitektur per representasi
- **Raw coords (136):** CNN1D > FCNN (Conv1d ambil locality antar titik berurutan)
- **FACS distance (28):** FCNN > CNN1D di FA (hand-crafted features tidak punya locality, MLP natural)
- **Blendshape (52):** mixed (FCNN di 7c, CNN1D di 3c) — unordered features
- **FACS+Blendshape (80):** FCNN dominan di FA, CNN1D dominan di MP

### 4.4 Behavior B1/B2/B3 (Unified Protocol)
- **B1 paling stabil untuk 7c MP-source.** Class imbalance ekstrem (2263:1) bikin sampler + aug menambah noise > info gain
- **B3 baru menonjol di CNN1D + raw coords.** Lokalitas titik adjacent + aug = synergy. MP raw CNN1D B3 = 0.7005 (B1 cuma 0.5336)
- **B2 menang untuk derived features 7c.** FB80 × FA × FCNN × B2 = **0.3331** mengalahkan B1 yang sebelumnya dianggap optimal
- **B2 menang untuk image CNN_TL.** 3c CNN_TL × B2 = 0.7107 dan 7c CNN_TL × B2 = 0.2964 — image task lebih responsif ke sampler dibanding aug (B3 sedikit lebih rendah). Pretrained ResNet-18 sudah robust ke variasi visual, sehingga photometric aug kurang berdampak
- **CNN_TL >> CNN scratch konsisten** di semua scenario × skema. Pretrained ImageNet features esensial untuk dataset kecil (5287 train sample)
- **MP-source + B2/B3 sering jatuh** (terutama 7c). Force-balance + aug di source noisy → distorsi loss
- **B3 untuk derived features dengan blendshape ter-tahan** — aug landmark hanya cover FACS part (dim 0-27), blendshape coef (dim 28-79) statis (lihat L3 Section 5.3)

### 4.5 Class imbalance — bottleneck struktural di 7c
Train counts 7c: `[4526 neutral, 416 happy, 287 sad, 27 angry, 2 fearful, 13 disgusted, 16 surprised]`. Rasio 2263:1. Test set kelas minoritas 1-3 sampel → metric noise tinggi, best 7c mf1 = 0.3331 dengan acc 0.81+. 3c remap (positive=happy+surprised, neutral=neutral, negative=sad+angry+fearful+disgusted) jauh lebih wajar.

---

## 5. Yang masih kurang & roadmap eksekusi

Section ini adalah **single source of truth** untuk gap, instruksi eksekusi, dan limitations. Tidak ada doc/runbook terpisah — semua di sini.

### 5.1 Gap unimodal: ✅ closed (16 Mei 2026)

Image-based sweep selesai. CNN scratch + CNN_TL × B1/B2/B3 × 3c/7c = **12 runs** sudah di master table baris 15-16. Semua 96 cell unimodal terisi.

**Selanjutnya:** explorasi multimodal — lihat `docs/multimodal_results.md`.

### 5.2 Status cakupan B1/B2/B3 per representasi (✅ all complete)

| Representasi | B1 | B2 | B3 |
|---|:---:|:---:|:---:|
| Raw coords (136) MP & FA | ✓ | ✓ | ✓ |
| FACS distance (28) MP & FA | ✓ | ✓ | ✓ |
| Blendshape (52) MP | ✓ | ✓ | ✓ |
| FACS+Blendshape (80) MP & FA | ✓ | ✓ | ✓ |
| Image (224×224×3) MP-crop | ✓ | ✓ | ✓ |

96 cell unimodal fully complete dengan protokol B3 konsisten.

### 5.3 Limitations sengaja diterima (per arahan, tidak akan di-fix)

| # | Limitation | Alasan |
|---|---|---|
| L1 | Single split, 1 seed per kombinasi (no multi-seed / CV / LOSO) | "cukup single split saja" |
| L2 | Blendshape part (dim 28-79) di FB80 tidak ter-augment di B3 | Effort vs payoff rendah; aug hanya kena FACS part |
| L3 | Tidak ada CNN × face-api.js | FA cuma deteksi landmark, tidak punya data citra. Citra selalu dari OBS+MediaPipe |

**Implikasi L1:** angka 7c sangat sensitif ke random init (kelas minoritas 1-3 sampel di test). Geser ±0.02-0.05 antar seed. Untuk publish/skripsi, perlu di-disclose.

### 5.4 Catatan historis (1 baris)

Sebelum Unified Protocol (≤14 Mei 2026), B3 punya 2 arti berbeda antar script (offline-augmented vs on-the-fly). Dataset offline-aug `dataset_frontonly_conf60_augmented/` sudah hilang dari server. Semua angka di doc ini pakai Unified Protocol — legacy results sudah deprecated.

---

## 6. Cross-dataset: KDEF & RAF-DB (B1, MP-source)

Replication finding primer di dataset benchmark. Hanya scenario B1 (B2/B3 belum), source MediaPipe (FA tidak tersedia — image sekunder sudah pre-cropped).

### 6.1 KDEF 7c (balanced, ~334-338/class)

| Method | 3c-B1 | 7c-B1 |
|---|:---:|:---:|
| FCNN raw_136 MP | 0.6937 | 0.5078 |
| CNN1D raw_136 MP | 0.7099 | 0.6042 |
| FCNN FACS_28 MP | 0.7223 | 0.6066 |
| CNN1D FACS_28 MP | 0.7063 | **0.6341** |
| CNN scratch (image) | 0.7845 | 0.7920 |
| CNN_TL (image, ResNet-18) | ⭐ **0.9454** | ⭐ **0.8966** |

**KDEF pattern:** CNN_TL menang absolut, gap besar dari landmark (~0.30). Balanced dataset = image-friendly. Best landmark method (FACS_28 CNN1D 7c = 0.6341) jauh di bawah image.

### 6.2 RAF-DB 7c (~5:1 imbalance)

| Method | 3c-B1 | 7c-B1 |
|---|:---:|:---:|
| FCNN raw_136 MP | 0.6724 | **0.5373** |
| CNN1D raw_136 MP | 0.6720 | 0.5015 |
| FCNN FACS_28 MP | 0.6856 | 0.4734 |
| CNN1D FACS_28 MP | 0.6709 | 0.4722 |
| CNN scratch (image) | 0.7809 | 0.6887 |
| CNN_TL (image, ResNet-18) | ⭐ **0.8254** | ⭐ **0.7255** |

**RAF-DB pattern:** CNN_TL menang konsisten. Untuk landmark 7c, raw_136 FCNN > FACS_28 (kebalikan dari KDEF).

### 6.3 Cross-dataset ranking (3c-B1)

| Ranking | Primer | KDEF | RAF-DB |
|---|---|---|---|
| 1 (best) | FACS_28 FA FCNN (0.7585) | CNN_TL (0.9454) | CNN_TL (0.8254) |
| Landmark best | landmark FA (~0.75) | landmark MP (~0.72) | landmark MP (~0.69) |
| Image best | CNN_TL B2 (0.7107) | CNN_TL (0.9454) | CNN_TL (0.8254) |

**Implikasi:** finding primer "landmark FA menang" **tidak generalize** ke KDEF & RAF-DB. Kemungkinan karena (1) FA-landmark tidak ada di sekunder, (2) image sekunder lebih ekspresif (KDEF lab-quality, RAF-DB real-world tapi tetap close-shot).

### 6.4 Status sekunder lain

- **CK+ 7c** (508 train, 118 subjek, neutral-heavy): 🟡 sedang B1 chain (landmark + image)
- **JAFFE 7c** (152 train, 7 subjek, perfectly balanced): 🟡 sedang B1 chain
- **Derived sweep (blendshape_52 + FB80)** untuk 4 sekunder: 🟡 sedang jalan di GPU 0
- **B2/B3 extension** untuk semua sekunder: belum dijalankan

---

## Sumber data

```
models/frontonly_conf60/{3class,7class}/Unified/
├── raw_136/results.json            (12 runs: MP+FA × FCNN+CNN1D × B1+B2+B3)
├── facs_28/results.json            (12 runs: MP+FA × FCNN+CNN1D × B1+B2+B3)
├── blendshape_52/results.json      (6 runs: MP × FCNN+CNN1D × B1+B2+B3)
├── facs_plus_bs_80/results.json    (12 runs: MP+FA × FCNN+CNN1D × B1+B2+B3)
├── cnn_scratch/results.json        (3 runs: B1+B2+B3 × per-scheme)
└── cnn_tl/results.json             (3 runs: B1+B2+B3 × per-scheme)
```

Setiap JSON: `config`, `runs[<key>].{test,val_at_best,training,hyperparams,hardware,model,dataset}`. Field `test` punya `macro_f1`, `weighted_f1`, `accuracy`, `confusion_matrix`, `classification_report` (per-class). Field `training.history` punya per-epoch (`train_loss`, `val_macro_f1`, dst).

> **Multimodal results** (IntermediateFusion, Late Fusion, Early Fusion) ada di tree yang sama tapi **out-of-scope doc unimodal ini** — lihat doc multimodal terpisah.

---

*Dokumen dibuat: 14 Mei 2026. Update: 16 Mei 2026 (restrukturisasi konsolidatif + image sweep selesai → 96 cell master table fully filled).*
