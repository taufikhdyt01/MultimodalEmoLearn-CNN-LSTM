# Status Data Sekunder (Benchmark Datasets)

> Audit 16 Mei 2026. Tujuan: identifikasi apa yang sudah ada vs yang perlu di-preprocess untuk extend Unified Protocol sweep ke benchmark datasets (cross-dataset claim).
>
> **TL;DR — banyak hasil sudah ada (legacy protocol)**: KDEF, RAF-DB, CK+ (CV10), JAFFE (LOSO) sudah punya B1 results untuk 9 arch (CNN, FCNN, Intermediate, CNN_TL, Intermediate_TL, EarlyFusion, EarlyFusion_TL, LateFusion, LateFusion_TL). KDEF & RAF-DB **3c bahkan multi-seed (3 seeds, ada mean±std)**. Plus cross-dataset transfer evaluation (`all_cross_results.json`). Yang **belum** = B2/B3 scenarios + Unified Protocol port + FA-landmark variant + FACS/Blendshape features.

---

## 1. Dataset sekunder yang tersedia di server

`data/benchmark/` — 18 GB total.

| Dataset | Skema | Size | Status preprocessing | Sumber |
|---|---|---|---|---|
| **KDEF** | 4c, 7c | 1.4 GB × 2 | ✅ **Fully preprocessed** (image + landmark + heatmap + subjects) | Karolinska Directed Emotional Faces |
| **RAF-DB** | 4c, 7c | 7.5 GB × 2 | ✅ **Mostly preprocessed** (image + landmark + heatmap, no val split explicit) | Real-world Affective Faces Database |
| CK+ | 7 classes | 112 MB | ⏳ Raw PNG di 8 folder emosi (S010_*, S011_*, dst). Belum di-preprocess ke npy | Extended Cohn-Kanade |
| JAFFE | 7 classes | 15 MB | ⏳ Raw images per emotion folder. Belum di-preprocess | Japanese Female Facial Expression |

**Yang langsung pakai (KDEF + RAF-DB):**
- KDEF: train 2353, val 294, test 294 (subject-aware split via `subjects_*.npy`)
- KDEF balance: **perfectly balanced** ~334-338 per class (vs primer 2263:1 imbalance!) — bagus untuk fair 7c comparison
- RAF-DB: train + test (no explicit val di sini, mungkin train-only split di train data)

**Distribusi per dataset (penting untuk class balance analysis):**

| Dataset | Train | Distribution 7c |
|---|---|---|
| KDEF 7c | 2353 | neutral 334, happy 337, sad 336, angry 337, fearful 338, disgusted 334, surprised 337 (≈balanced) |
| Primer 7c | 5287 | neutral 4526, happy 416, sad 287, angry 27, fearful 2, disgusted 13, surprised 16 (rasio 2263:1) |

> 💡 KDEF balance jadi opportunity untuk validate hipotesis: kalau primer 7c lemah (best 0.3331) karena class imbalance ekstrem, KDEF 7c **harus** kasih angka lebih bagus karena balanced.

---

## 2. File yang ada vs yang missing per dataset

### KDEF 4c & 7c (lengkap untuk fusion)

```
data/benchmark/kdef_{4,7}class/
├── X_{train,val,test}_images.npy       (N, 224, 224, 3) float32     ✅ ada
├── X_{train,val,test}_landmarks.npy    (N, 136) float32              ✅ ada (MP raw coords)
├── X_{train,val,test}_heatmaps.npy     (N, 224, 224) float32         ✅ ada
├── y_{train,val,test}.npy              (N,) int                      ✅ ada
├── subjects_{train,val,test}.npy       (N,) int                      ✅ ada (subject-aware)
├── label_map.json                                                    ✅ ada
└── dataset_info.json                                                 ✅ ada
```

**Belum ada di KDEF:**
- ❌ FACS distance (28-dim)
- ❌ Blendshape coefficient (52-dim, MP)
- ❌ FACS+Blendshape (80-dim concat)
- ❌ face-api.js landmark (alternatif source)

### RAF-DB 4c & 7c (lengkap kecuali val split)

```
data/benchmark/rafdb_{4,7}class/
├── X_{train,test}_images.npy           ✅ ada
├── X_{train,test}_landmarks.npy        ✅ ada (MP raw coords)
├── X_{train,test}_heatmaps.npy         ✅ ada
├── y_{train,test}.npy                  ✅ ada
└── (no val split — train/test only)
```

**Belum ada di RAF-DB:**
- ❌ Val split (perlu split dari train untuk early stopping / model selection)
- ❌ FACS distance, Blendshape, FB80, face-api.js landmark (sama dengan KDEF)
- ❌ Subject info (tidak ada subjects_*.npy)

### CK+ dan JAFFE (raw, butuh preprocessing penuh)

Raw PNG per emotion folder. Untuk masuk ke pipeline butuh:
- Face detection + crop ke 224×224
- Landmark extraction (MediaPipe atau face-api.js)
- Heatmap generation dari landmark
- Subject-aware split (CK+: pakai subject ID dari filename `S010_*`, JAFFE: subject ID dari filename)
- Save sebagai npy

CK+ punya 8 kelas (Anger, Contempt, Disgust, Fear, Happy, Neutral, Sadness, Surprised) — ada Contempt tambahan di luar 7c standar (perlu mapping).

JAFFE punya 7 kelas (Anger, Disgust, Fear, Happy, Neutral, Sadness, Surprised).

---

## 2.5. Hasil eksperimen yang SUDAH ADA di sekunder (legacy protocol)

> Disimpan di `models/benchmark/*`. Semua **B1-only** kecuali KDEF/RAF-DB 3c yang multi-seed (3 seeds, mean ± std).

### KDEF — 3c multi-seed (n=3) | 4c B1 | 7c B1

**3c (mean ± std macro_f1):**

| Method | 3c | 4c (B1 only) | 7c (B1 only) |
|---|:---:|:---:|:---:|
| CNN | 0.8332 ± 0.0098 | 0.8407 | 0.7984 |
| FCNN | 0.7080 ± 0.0231 | 0.6784 | 0.6657 |
| Intermediate | 0.8194 ± 0.0169 | 0.7759 | 0.6710 |
| CNN_TL | 0.9059 ± 0.0109 | 0.9179 | 0.8333 |
| **Intermediate_TL** | 0.9111 ± 0.0090 | **0.9232** ⭐ | **0.8431** ⭐ |
| EarlyFusion | 0.7854 ± 0.0226 | 0.6934 | 0.6665 |
| EarlyFusion_TL | 0.8770 ± 0.0198 | 0.8157 | 0.7987 |
| LateFusion | 0.8380 ± 0.0085 | 0.8589 | 0.7757 |
| **LateFusion_TL** | **0.9139 ± 0.0121** ⭐ | 0.9195 | 0.8358 |

**Top 3c KDEF: LateFusion_TL = 0.9139 ± 0.0121** (sangat stabil multi-seed!) >> primer best 0.7585 (FACS×FA×FCNN). **Fusion sangat efektif di KDEF.**

### RAF-DB — 3c multi-seed (n=3) | 4c B1 | 7c B1

| Method | 3c | 4c (B1 only) | 7c (B1 only) |
|---|:---:|:---:|:---:|
| CNN | 0.8028 ± 0.0093 | 0.8084 | 0.7294 |
| FCNN | 0.6938 ± 0.0057 | 0.6940 | 0.5781 |
| Intermediate | 0.7792 ± 0.0104 | 0.7924 | 0.6958 |
| CNN_TL | 0.8119 ± 0.0014 | 0.8269 | 0.7407 |
| Intermediate_TL | 0.7703 ± 0.0001 | **0.8356** ⭐ | **0.7440** ⭐ |
| EarlyFusion | 0.7903 ± 0.0018 | 0.7925 | 0.7098 |
| EarlyFusion_TL | 0.7550 ± 0.0032 | 0.7992 | 0.6929 |
| LateFusion | 0.8078 ± 0.0037 | 0.8194 | 0.7191 |
| **LateFusion_TL** | **0.8141 ± 0.0015** ⭐ | 0.8322 | 0.7350 |

**Top 3c RAF-DB: LateFusion_TL = 0.8141 ± 0.0015** — close gap dengan CNN_TL (0.8119), fusion adds value.

### CK+ — 10-fold cross-validation (CV10)

`models/benchmark/ckplus_cv10/`. Mean ± std atas 10 folds.

| Method | 4c | 7c |
|---|:---:|:---:|
| CNN_B1 | 0.5839 ± 0.163 | 0.4044 ± 0.049 |
| FCNN_B1 | 0.5978 ± 0.036 | 0.4778 ± 0.022 |
| Intermediate_B1 | 0.4581 ± 0.172 | 0.2261 ± 0.082 |
| **CNN_TL_B1** | **0.7545 ± 0.079** | 0.7335 ± 0.082 |
| **Intermediate_TL_B1** | 0.7151 ± 0.054 | **0.7827 ± 0.107** ⭐ |
| Late_Fusion_B1 | 0.6214 ± 0.031 | 0.5436 ± 0.060 |

**Top CK+ 7c: Intermediate_TL_B1 = 0.7827 ± 0.107** — variance besar karena dataset kecil per fold.

### JAFFE — Leave-One-Subject-Out (LOSO)

`models/benchmark/jaffe_loso/`. Mean ± std atas subject splits.

| Method | 7c (LOSO) |
|---|:---:|
| CNN_B1 | 0.2486 ± 0.111 |
| FCNN_B1 | 0.3043 ± 0.157 |
| Intermediate_B1 | 0.1294 ± 0.070 |
| CNN_TL_B1 | 0.4260 ± 0.143 |
| Intermediate_TL_B1 | 0.2925 ± 0.156 |
| **Late_Fusion_B1** | **0.4667 ± 0.092** ⭐ |

**JAFFE 7c paling sulit** — LOSO + dataset kecil (213 sample) bikin variance tinggi. Late Fusion bertahan paling stabil.

### Cross-dataset transfer evaluation

`models/benchmark/crossdataset/all_cross_results.json`. **Train di primer, test di sekunder** (zero-shot transfer, no fine-tuning).

| Target | Best transfer | mf1 | Implikasi |
|---|---|:---:|---|
| ckplus_7c | Late_Fusion_TL_B1 (srcval) | 0.2528 | Domain gap besar — primer → CK+ jauh |
| ckplus_4c | CNN_B1 | 0.3958 | 4c lebih bertahan dari 7c |
| jaffe_7c | Late_Fusion_TL_B1 | 0.0501 | Hampir random — primer → JAFFE sangat jauh |
| jaffe_4c | Intermediate_TL_B1 | 0.0929 | Same |
| rafdb_7c | FCNN_B1 | 0.1827 | Lebih baik dari JAFFE tapi masih lemah |
| rafdb_4c | EarlyFusion_TL_B1 | 0.3115 | Sedikit reasonable |
| kdef_7c | Late_Fusion_TL_B1 | 0.0546 | Hampir random |
| kdef_4c | EarlyFusion_TL_B1 | 0.1037 | Lemah |

**Implikasi cross-dataset:** primer-trained models **tidak generalize** ke benchmark datasets tanpa fine-tuning. Domain gap (camera setup, lighting, subjects, emotion expression style) terlalu besar. Untuk cross-dataset claim, perlu evaluate **after fine-tuning** atau pakai domain adaptation method.

### File summary di `models/benchmark/`

```
models/benchmark/
├── kdef/{3,4,7}class/kdef_{3,4,7}c_results.json     # 9 methods × B1
├── rafdb/{3,4,7}class/rafdb_{3,4,7}c_results.json   # 9 methods × B1
├── ckplus_cv10/ckplus_{4,7}c_cv10_results.json      # 6 methods × CV10
├── jaffe_loso/jaffe_{4,7}c_loso_results.json        # 6 methods × LOSO
├── primer/{3,4,7}class/                              # primer reference (legacy)
├── crossdataset/all_cross_results.json              # primer → target transfer
├── crossdataset/cross_{ckplus,jaffe,rafdb,kdef}_{4,7}c.json   # detailed per target
├── all_3c_skema1_results.json                       # RAF-DB + KDEF 3c summary
└── all_3c_skema2_cross_results.json                 # cross-dataset 3c
```

### Yang BELUM ada di hasil sekunder

| Item | Status |
|---|---|
| **B2 / B3 scenarios** untuk KDEF, RAF-DB | ❌ semua B1-only kecuali multi-seed B1 di 3c |
| **Unified Protocol** versi (sampler + on-the-fly aug) | ❌ semua legacy protocol |
| **FA-landmark variant** di sekunder | ❌ — semua MP-only (FA landmark belum di-extract) |
| **FACS distance, Blendshape, FB80** di sekunder | ❌ — feature derivation belum dilakukan |
| **Multi-seed untuk 4c & 7c** | ❌ — cuma 3c yang multi-seed (n=3); 4c & 7c single run |
| **CK+ 3c, JAFFE 3c** | ❌ — cuma 4c & 7c |

---

## 3. Gap matrix: apa yang bisa dijalankan langsung vs butuh kerja preprocessing

| Eksperimen | KDEF 4c/7c | RAF-DB 4c/7c | CK+ | JAFFE |
|---|---|---|---|---|
| Unimodal Raw landmark 136 (MP) | ✅ langsung | ⚠️ butuh val split | ❌ butuh preprocessing | ❌ butuh preprocessing |
| Unimodal FACS distance (28) | ❌ butuh derive | ❌ butuh derive | ❌ butuh preprocessing | ❌ butuh preprocessing |
| Unimodal Blendshape (52) | ❌ butuh re-run MP | ❌ butuh re-run MP | ❌ | ❌ |
| Unimodal FACS+BS (80) | ❌ butuh both | ❌ butuh both | ❌ | ❌ |
| Unimodal Image CNN scratch/TL | ✅ langsung | ⚠️ butuh val split | ❌ | ❌ |
| Multimodal Early Fusion | ✅ langsung | ⚠️ butuh val split | ❌ | ❌ |
| Multimodal Intermediate Fusion | ✅ langsung | ⚠️ butuh val split | ❌ | ❌ |
| Multimodal Late Fusion | ✅ langsung | ⚠️ butuh val split | ❌ | ❌ |
| face-api.js variant (semua) | ❌ butuh extract FA landmark | ❌ | ❌ | ❌ |

**Effort estimasi per kerja preprocessing:**

| Kerja | Effort | Output |
|---|---|---|
| FACS distance derive dari raw_136 landmark | **~10 menit** — sudah ada fungsi `compute_facs_distances` di `run_unified_landmark.py`, tinggal panggil per dataset | `X_{train,val,test}_facs.npy` per benchmark dataset |
| RAF-DB val split | ~20 menit — split train 80/20 dengan stratify per class, save subjects_val.npy fake (or use random) | `X_val_*.npy`, `y_val.npy` di rafdb_*/ |
| Blendshape extract via MediaPipe | ~30-60 menit/dataset — run MP FaceLandmarker v2 on each image, extract blendshape coeffs | `X_*_mp_blendshapes.npy` |
| FACS+BS concat | ~5 menit (concat 2 arrays) | `X_*_mp_facs_plus_bs.npy` (atau gabung at runtime) |
| face-api.js landmark extract | ~1-2 jam/dataset — pakai face-api.js JS pipeline, native dlib 68-pt | `X_*_faceapi_landmarks.npy` |
| CK+ preprocessing penuh | ~2-3 jam — face crop, landmark, heatmap, label map (7 atau 8 class), subject split | KDEF-like structure |
| JAFFE preprocessing penuh | ~1-2 jam (lebih kecil dari CK+) | KDEF-like structure |

---

## 4. Rekomendasi prioritas untuk cross-dataset claim

Untuk thesis claim "method works across datasets", level effort vs payoff:

### Quick win (jam-an, langsung impactful)

**Step 1:** Parameterize Unified Protocol scripts (`run_unified_landmark.py`, `run_unified_image.py`, `run_unified_fusion.py`) untuk accept `--data-dir` flag.

**Effort:** ~30 menit modifikasi (cuma replace hardcoded `DATA_DIR` dengan CLI arg).

**Step 2:** Jalankan Unified sweep di KDEF (sudah lengkap):
- Unimodal landmark raw_136 MP × FCNN+CNN1D × B1/B2/B3 × 3c+7c — wait KDEF cuma 7c, jadi 6 runs (atau 4c kalau 4-class), atau 6+6=12
- Unimodal image CNN scratch + CNN_TL × B1/B2/B3 × {4c,7c} = 12 runs
- Multimodal fusion early + intermediate × scratch + TL × B1/B2/B3 × {4c,7c} = 24 runs (+ late 12 post-hoc)
- **Total: ~48 runs di KDEF**

**Estimasi durasi KDEF sweep:** ~2-4 jam (KDEF lebih kecil dari primer: 2353 vs 5287 train).

**Impact:** validate temuan primer (FA landmark best, Intermediate TL B3 best fusion 7c) di KDEF. **Cross-dataset claim langsung jadi.**

### Medium effort (1-2 hari)

**Step 3:** Derive FACS distance (28) untuk KDEF + RAF-DB → run feature variation experiments.

**Step 4:** RAF-DB val split + run full KDEF-style sweep di RAF-DB.

### High effort (week+)

**Step 5:** Re-extract MediaPipe blendshape untuk KDEF + RAF-DB → enable FB80 feature.

**Step 6:** face-api.js landmark extract untuk semua benchmark → enable FA-source experiments di sekunder.

**Step 7:** CK+ dan JAFFE full preprocessing pipeline.

---

## 5. Konkret next step untuk multimodal exploration (paralel dengan Phase 1B)

Sambil Phase 1B Late Fusion sedang running di GPU 2 (~3 jam lagi), bisa kerjakan ini di GPU 0 yang sudah idle:

### Opsi cepat (1-2 jam total): Parametrize + KDEF sweep

1. Modifikasi `run_unified_landmark.py`, `run_unified_image.py`, `run_unified_fusion.py` untuk accept `--data-dir <path>` (default = primer)
2. Modifikasi REMAP_3 → opsional (KDEF native 7c, kalau perlu 3c remap pakai mapping serupa)
3. Sketch `scripts/derive_facs_features.py` (one-shot util untuk derive FACS distance dari raw_136 untuk dataset baru) — opsional
4. Launch KDEF 7c sweep di GPU 0:
   ```bash
   $PY scripts/run_unified_landmark.py --data-dir data/benchmark/kdef_7class --features raw_136 --sources mediapipe --classes 7
   $PY scripts/run_unified_image.py --data-dir data/benchmark/kdef_7class --classes 7
   $PY scripts/run_unified_fusion.py --data-dir data/benchmark/kdef_7class --classes 7
   ```

**Hasil yang diharapkan:** angka KDEF 7c untuk landmark MP × FCNN/CNN1D × B1/B2/B3, image CNN/TL × B1/B2/B3, fusion early/intermediate × scratch/TL × B1/B2/B3 — total ~36 runs di sekunder.

**Validasi yang bisa dibuat:**
- Bandingkan KDEF 7c vs primer 7c — apakah angka KDEF lebih tinggi (karena balanced)?
- Apakah ranking method konsisten (e.g., Intermediate TL B3 menang juga di KDEF)?
- Confirm class imbalance bottleneck claim di primer

---

*Dokumen audit dibuat: 16 Mei 2026 sore. Updated berdasarkan eksplorasi file `data/benchmark/*` actual contents.*
