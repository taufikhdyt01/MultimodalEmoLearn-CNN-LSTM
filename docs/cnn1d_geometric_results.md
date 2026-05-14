# Hasil Eksplorasi CNN1D Fitur Geometrik

**Arahan dosen (13 Mei 2026):** "Coba model CNN dengan fitur geometrik (baseline – hanya koordinat titik)."

**Interpretasi:** CNN 1D di koordinat landmark mentah — 68 titik × 2 (x, y) sebagai sequence, bukan flat 136-dim vector seperti FCNN. "Baseline" = raw coordinates dulu, sebelum tambah fitur turunan (eccentricity / distance ratio à la Liliana 2019).

---

## Setup eksperimen

| Item | Nilai |
|---|---|
| Dataset | `data/dataset_frontonly_conf60/` (Primer conf60, single train/val/test split) |
| Model | `EmotionCNN1D` (`src/training/models.py:14`) |
| Arsitektur | 4 Conv1d blocks (32→64→128→256, k=5/5/3/3) + GAP + FC(128) + head |
| Params | 441,735 (7c) / 441,219 (3c) |
| Input | (B, 2, 68) — 2 channel (x, y) × 68 titik landmark |
| Optimizer | Adam, lr=1e-3 |
| Batch / Epochs | 32 / max 50 (early stop patience 15, val macro-F1) |
| Seed | 42 |
| Hardware | NVIDIA L40 (GPU 1), shared server |

**Skema kelas:**
- **7c:** angry, disgust, fear, happy, neutral, sad, surprise (label asli dataset)
- **3c:** positive / neutral / negative (remap dari 7c via `REMAP_3 = [1, 0, 2, 2, 2, 2, 0]`)

**Skenario training:**

| Kode | Augmentasi | Class weights |
|---|---|---|
| B1 | none | none (uniform) |
| B2 | none | sklearn `balanced` |
| B3 | on-the-fly landmark aug (hflip + rotate ±10° + flip_rot, minoritas → match majority count) | balanced (= 1.0 setelah aug) |

**Catatan B3:** karena augmented dataset offline (`data/dataset_frontonly_conf60_3class_augmented/`) tidak ada di server, B3 dijalankan dengan augmentasi on-the-fly di training script. Implementasi mirror `src/preprocessing/augment_conf60_3class.py:73` (`augment_landmark`). Untuk kelas minoritas, oversampling dengan cycle teknik (hflip → rotate_pos → rotate_neg → flip_rot) sampai count sama dengan kelas mayoritas.

---

## Distribusi kelas

### 7-class (label asli)

| Split | neutral | happy | sad | angry | fearful | disgusted | surprised | total |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| train | 4526 | 416 | 287 | 27 | **2** | 13 | 16 | 5287 |
| val | 477 | 52 | 24 | 3 | 2 | 1 | 20 | 579 |
| test | 688 | 183 | 50 | 2 | 1 | 2 | 3 | 929 |

> Imbalance ekstrem: rasio neutral:fearful = **2263:1** di train. Inilah sumber masalah B2 7c.

### 3-class (positive / neutral / negative)

| Split | positive | neutral | negative | total |
|---|:---:|:---:|:---:|:---:|
| train | 432 | 4526 | 329 | 5287 |
| val | 72 | 477 | 30 | 579 |
| test | 186 | 688 | 55 | 929 |

> Imbalance moderate: rasio max ≈ 14:1. Class weights B2 = [4.08, 0.39, 5.36] (jauh lebih jinak dari 7c).

---

## Hasil utama

### 7-class

| Metric | FCNN B1 | **CNN1D B1** | FCNN B2 | CNN1D B2 | FCNN B3 | CNN1D B3 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| test_macro_f1 | 0.2317 | **0.2766** ✓ | 0.2437 | 0.1631 ✗ | 0.2224 | 0.2235 ≈ |
| test_weighted_f1 | 0.7650 | **0.8014** | 0.7674 | 0.5994 | 0.7580 | 0.7003 |
| test_accuracy | 0.7675 | **0.8084** | 0.7653 | 0.5608 | 0.7395 | 0.6609 |
| val_macro_f1 (best) | — | 0.2777 | — | 0.2184 | — | 0.2409 |
| best epoch | — | 32 | — | 36 | — | 19 |
| training time (s) | — | 171 | — | 142 | — | 551 |
| **delta macro_f1** | — | **+0.0449** | — | **−0.0806** | — | **+0.0011** |

### 3-class

| Metric | FCNN B1 | **CNN1D B1** | FCNN B2 | CNN1D B2 | FCNN B3 ⭐ | CNN1D B3 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| test_macro_f1 | 0.5893 | **0.6183** ✓ | 0.5750 | 0.5935 ✓ | **0.6342** | 0.5744 ✗ |
| test_weighted_f1 | 0.7596 | **0.8203** | 0.7302 | 0.7686 | 0.7636 | 0.7072 |
| test_accuracy | 0.7406 | **0.8299** | 0.7018 | 0.7524 | 0.7492 | 0.6598 |
| val_macro_f1 (best) | — | 0.5977 | — | 0.6039 | — | 0.5617 |
| best epoch | — | 24 | — | 24 | — | 25 |
| training time (s) | — | 110 | — | 109 | — | 279 |
| **delta macro_f1** | — | **+0.0290** | — | **+0.0185** | — | **−0.0598** |

**Best overall:** FCNN 3c B3 = **0.6342 macro_f1** (raja sejauh ini).
**Best CNN1D:** 3c B1 = **0.6183 macro_f1** (close second, +0.029 vs FCNN B1).

---

## Confusion matrix (CNN1D)

### 7c — labels: neutral, happy, sad, angry, fearful, disgusted, surprised

**B1** (best 7c):
```
            pred:  N   H   S   A   F   D   Su
neutral   (688): [622, 36, 30,  0,  0,  0,  0]
happy     (183): [ 72,108,  3,  0,  0,  0,  0]
sad        (50): [ 29,  0, 21,  0,  0,  0,  0]
angry       (2): [  2,  0,  0,  0,  0,  0,  0]
fearful     (1): [  0,  0,  1,  0,  0,  0,  0]
disgusted   (2): [  2,  0,  0,  0,  0,  0,  0]
surprised   (3): [  2,  1,  0,  0,  0,  0,  0]
```
Model hanya predict 3 kelas teratas (N/H/S). Kelas dengan 1-3 sampel test mustahil di-recover dari 0 prediksi.

**B2** (collapse):
```
            pred:  N   H   S   A   F   D   Su
neutral   (688): [470,  7,179, 31,  0,  0,  1]
happy     (183): [123, 41, 12,  7,  0,  0,  0]
sad        (50): [ 37,  1, 10,  1,  0,  0,  1]
...
```
Class weight 377.6 untuk fear (hanya 2 train) mendistorsi loss; akurasi neutral hancur (470/688 = 68% recall vs 90% di B1).

### 3c — labels: positive, neutral, negative

**B1** (best CNN1D 3c):
```
              pred:  Pos  Neu  Neg
positive (186): [132,  52,   2]
neutral  (688): [ 41, 629,  18]
negative  (55): [  4,  41,  10]
```
Recall positive 71%, neutral 91%, negative 18%. Negative tetap susah (hanya 329 train sample, banyak campuran sad/angry/fearful/disgusted dengan ciri ambigu).

**B2:**
```
              pred:  Pos  Neu  Neg
positive (186): [101,  75,  10]
neutral  (688): [ 23, 571,  94]
negative  (55): [  0,  28,  27]
```
Trade-off: negative recall naik ke 49%, tapi positive jatuh ke 54%.

**B3:**
```
              pred:  Pos  Neu  Neg
positive (186): [153,  27,   6]
neutral  (688): [ 70, 424, 194]
negative  (55): [  4,  15,  36]
```
Positive recall 82%, negative recall 65%, tapi neutral jatuh ke 62% → weighted-F1 hancur.

---

## Interpretasi

1. **CNN1D B1 konsisten menang vs FCNN B1.** Di 7c (+4.5%) dan 3c (+2.9%) macro-F1. Struktur sequential antar titik landmark (Conv1d) bisa di-leverage; flat MLP membuang informasi adjacency.

2. **B2 di 7c collapse.** `class_weight='balanced'` dengan rasio 2263:1 (neutral:fearful) menghasilkan weight 377.6 untuk fearful. Loss didominasi 2 sampel fear di train → model over-correct, accuracy jatuh dari 0.81 ke 0.56. FCNN juga tidak diuntungkan banyak di B2 7c (cuma +0.012 vs B1) — naive balanced weights memang **bukan strategi yang tepat** untuk imbalance ekstrem seperti ini.

3. **B2 di 3c aman.** Rasio max 14:1, weights moderate (5.4:1:0.4), CNN1D B2 marginal unggul FCNN B2 (+0.019). Tapi val_macro_f1 B2 (0.604) lebih tinggi dari B1 (0.598) → model selection by val mungkin pilih B2; di test B1 menang tipis.

4. **B3 di 3c kalah dari FCNN B3.** Augmentasi on-the-fly yang dipakai (hflip + rotate, oversampling sampai match majority) berbeda dengan offline augmented data yang dilatih FCNN B3. Kemungkinan penyebab:
   - On-the-fly aug terlalu agresif (every epoch sees fully balanced 13.5k samples vs original 5.3k)
   - hflip landmark tanpa swap index left/right secara semantik bisa rusak (mata kanan sekarang di posisi mata kiri, tapi index tetap → model belajar pola yang inconsistent)
   - Class weights 1:1:1 setelah balancing mungkin terlalu agresif vs offline aug yang biasanya partial fill

5. **Class imbalance ekstrem adalah bottleneck struktural untuk 7c.** Bahkan model bagus pun stuck di macro-F1 ~0.28 karena kelas dengan 2-16 train sample mustahil dipelajari. Ini konsisten dengan eksperimen clean-6c sebelumnya (drop fear → macro_f1 6c masih 0.27).

6. **Macro-F1 tidak selalu align dengan accuracy/weighted-F1.** Di 7c B1, CNN1D unggul di semua tiga metrik. Tapi di 3c B3, FCNN B3 unggul macro-F1 tapi CNN1D B3 worse di semua weighted/acc → macro-F1 menangkap performa minoritas, weighted/acc menangkap mayoritas.

---

## Output files

```
models/frontonly_conf60/7class/CNN1D_geom/
  ├── cnn1d_geom_b1_results.json
  ├── cnn1d_geom_b2_results.json
  └── cnn1d_geom_b3_results.json
models/frontonly_conf60/3class/CNN1D_geom/
  ├── cnn1d_geom_b1_results.json
  ├── cnn1d_geom_b2_results.json
  └── cnn1d_geom_b3_results.json
logs/
  ├── cnn1d_geom_7c.log         (B1 7c)
  ├── cnn1d_geom_7c_b2.log      (B2 7c)
  └── cnn1d_geom_chain.log      (B3 7c + B1/B2/B3 3c)
```

JSON fields: `test`, `val`, `history`, `best_epoch`, `elapsed_sec`, `n_params`, `aug_per_class_added` (B3 only), `hyperparams`, `config`, `model`, `scenario`, `num_classes`, `input`.

---

## Cara reproduce

```bash
cd /mnt/extended-home/fitra_dosen/2025_iris_fer_taufik/MultimodalEmoLearn
source /mnt/extended-home/fitra_dosen/2025_iris_fer_taufik/miniconda3/bin/activate 2025_iris_fer_taufik

# 7-class
CUDA_VISIBLE_DEVICES=1 python scripts/run_cnn1d_geom_7c.py --scenario b1 --classes 7
CUDA_VISIBLE_DEVICES=1 python scripts/run_cnn1d_geom_7c.py --scenario b2 --classes 7
CUDA_VISIBLE_DEVICES=1 python scripts/run_cnn1d_geom_7c.py --scenario b3 --classes 7

# 3-class
CUDA_VISIBLE_DEVICES=1 python scripts/run_cnn1d_geom_7c.py --scenario b1 --classes 3
CUDA_VISIBLE_DEVICES=1 python scripts/run_cnn1d_geom_7c.py --scenario b2 --classes 3
CUDA_VISIBLE_DEVICES=1 python scripts/run_cnn1d_geom_7c.py --scenario b3 --classes 3

# Opsional: simpan checkpoint best
CUDA_VISIBLE_DEVICES=1 python scripts/run_cnn1d_geom_7c.py --scenario b1 --classes 3 --save-ckpt
```

---

## Eksperimen ablation: MediaPipe vs face-api.js landmark source (14 Mei 2026)

**Latar belakang:** Pipeline existing pakai landmark hasil **MediaPipe FaceLandmarker v2** (478 titik → mapping ke 68). Tapi database project punya kolom `landmarks` JSON face-api.js (dlib 68-point native) yang di-extract live di browser saat data collection. Eksperimen: bandingkan dua source di model arsitektur yang sama.

### Setup matching
- **Source full data:** `data/emotions.csv` (20,110 baris, 80 user)
- **Filter:** 37 user yang dipakai `dataset_frontonly_conf60`, conf60 (max emotion prob ≥ 0.6)
- **Matching:** strict (uid, soft_label vector) tol=1e-4; nearest-neighbor fallback untuk 13 sampel borderline (max distance 8.7e-3)
- **Coverage:** **100% (6795/6795)** — strict match 6782, nn fallback 13
- **Normalisasi face-api.js coords:** `(pos - shift) / imgDims` per axis → comparable dengan MediaPipe yang sudah di-normalize ke face-crop [0,1]
- Output: `X_{split}_faceapi_landmarks.npy`, `mask_{split}_faceapi.npy`, `match_kind_{split}_faceapi.npy` (1=strict, 2=fallback)
- Script: `scripts/build_faceapi_landmarks.py`

### Hasil — CNN1D vs FCNN, semuanya B1 (no aug, no class weight)

| Model | Skema | MediaPipe | **face-api.js** | Δ |
|---|---|:---:|:---:|:---:|
| **FCNN** | 3c | 0.5087 | **0.7119** | **+0.2032** |
| **FCNN** | 7c | 0.2255 | **0.3122** | **+0.0867** |
| **CNN1D** | 3c | 0.6380 | **0.7524** | **+0.1144** |
| **CNN1D** | 7c | 0.2679 | **0.3036** | **+0.0357** |

(Angka di atas adalah test_macro_f1. Untuk weighted_f1 dan accuracy, face-api.js juga konsisten unggul.)

**Best overall sebelumnya:** FCNN 3c B3 = 0.6342 (full pipeline dengan augmentation).
**Best baru:** **CNN1D 3c B1 + face-api.js = 0.7524** — pecah rekor +12% tanpa augmentation, hanya ganti landmark source.

### Interpretasi

1. **Landmark source > arsitektur model.** Gap face-api.js konsisten muncul di kedua arsitektur (FCNN +0.20 di 3c, CNN1D +0.11 di 3c). FCNN+face-api.js (0.7119) bahkan beat CNN1D+MediaPipe (0.6380), padahal CNN1D arsitektur lebih canggih.

2. **Kenapa face-api.js menang?** Hipotesis (perlu validasi lanjut):
   - **Coords preservation:** face-api.js extract dari frame asli (sebelum face-crop). MediaPipe extract dari face-crop 224×224 yang sudah resize. Frame asli punya info skala/pose yang hilang setelah crop+resize.
   - **Native dlib 68-point:** face-api.js output 68 titik native dlib convention. MediaPipe 478→68 via index mapping fix — semantik approximate, beberapa titik mungkin tidak persis di posisi konvensi dlib.
   - **Temporal consistency:** face-api.js extract live saat user beraktivitas (kondisi sesuai konteks). MediaPipe extract offline dari face-crop tersimpan (bisa ada artifact resize/kompresi JPEG 224×224).

3. **Kenaikan lebih besar di 3-class.** Mungkin karena di 3c, presisi landmark lebih critical untuk membedakan positive vs negative emotion (subtle geometric difference), sedangkan di 7c bottleneck utama adalah class imbalance ekstrem (rasio 2263:1), bukan kualitas feature.

4. **FCNN dapat lift terbesar.** FCNN simpler dan langsung memetakan koordinat → kelas. Jadi sangat sensitif terhadap **konsistensi & akurasi koordinat absolut**. CNN1D extract pattern lokal antar titik, sehingga slightly more robust terhadap variasi koordinat absolut.

### Output files

```
data/dataset_frontonly_conf60/
  ├── X_{train,val,test}_faceapi_landmarks.npy
  ├── mask_{train,val,test}_faceapi.npy
  └── match_kind_{train,val,test}_faceapi.npy
models/frontonly_conf60/3class/CNN1D_geom_compare/compare_landmark_source.json
models/frontonly_conf60/7class/CNN1D_geom_compare/compare_landmark_source.json
models/frontonly_conf60/3class/FCNN_compare/compare_landmark_source.json
models/frontonly_conf60/7class/FCNN_compare/compare_landmark_source.json
logs/cnn1d_faceapi_compare.log
logs/fcnn_faceapi_compare.log
```

### Cara reproduce

```bash
# 1. Build face-api.js landmark dari emotions.csv (sekali saja)
python scripts/build_faceapi_landmarks.py

# 2. Comparison CNN1D
CUDA_VISIBLE_DEVICES=1 python scripts/run_cnn1d_faceapi_compare.py

# 3. Comparison FCNN
CUDA_VISIBLE_DEVICES=1 python scripts/run_fcnn_faceapi_compare.py
```

### Caveat
- 1 seed per kombinasi → ada variance run-to-run (~±0.02 macro-F1). Multi-seed direkomendasikan untuk error bar. Tapi gap face-api.js (+0.09 sampai +0.20) jauh di atas noise floor.
- Match fallback untuk 13/6795 sampel (0.19%) — dampak negligible.

---

## Eksplorasi lanjutan yang disarankan

1. **Multi-seed (3-5 seeds)** — dapat error bar. Test set imbalanced (1-3 sampel di kelas minoritas) membuat 1 run noisy.
2. **5-fold CV subject-wise / LOSO** — pakai pattern dari `scripts/run_eval_7c.py`. Robustness vs random split.
3. **Regenerate offline augmented 3c dataset** dengan `src/preprocessing/augment_conf60_3class.py`, lalu re-run B3 dengan augmented data persis sama dengan FCNN B3 → comparison yang fair.
4. **Landmark hflip dengan index swap** — saat hflip horizontal, swap left/right pair indices (mata kiri ↔ mata kanan, alis kiri ↔ alis kanan, dll) supaya geometri tetap valid secara semantik. Saat ini hflip cuma mirror x, indeks tetap.
5. **CNN1D di clean-6c dataset** — drop fearful (2 sampel), class imbalance jauh lebih jinak, class weights B2 tidak ekstrem.
6. **Coordinate normalization** — center-of-face + inter-ocular distance scaling. Sering bantu landmark-based models robust terhadap variasi pose/scale.
7. **Compare vs FCNN + Liliana 20-dim geometric features** (nb 81) — apakah CNN1D di raw 136-dim coords > FCNN di derived 20-dim?
8. **Hybrid: Conv1d branch + 20-dim Liliana branch → concat → head** — gabungkan low-level (raw points) + high-level (eccentricity, distance ratios) features.
9. **Focal loss** sebagai alternatif class weights — biasanya lebih stabil di imbalance ekstrem.
10. **Multi-seed (3-5) untuk face-api.js comparison** — confirm gap +0.09 sampai +0.20 di atas noise floor empirically.
11. **Re-run B2/B3 dengan face-api.js landmark source** — apakah scenario lain juga konsisten naik?
12. **Apply face-api.js ke pipeline image-based juga** (CNN, IntermediateFusion, EarlyFusion) — re-extract image features dari face crop yang dihasilkan face-api.js bbox, lihat apakah lift bertahan saat digabung dengan visual features.
13. **FACS-decomposed Euclidean distances** (sedang berjalan via `scripts/run_cnn1d_facs.py`) — bandingkan dengan raw coords dan Liliana 20-dim untuk lihat representasi geometric mana yang paling efektif.

---

*Tanggal generate: 13 Mei 2026. Author: eksplorasi mandiri dengan Claude.*
