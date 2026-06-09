# Analisis Perbandingan Kinerja pada Dataset JAFFE

> **Catatan:** Pada protokol subject-wise, konfigurasi terbaik berdasarkan
> **accuracy** dan **macro F1** menunjuk pada model yang berbeda. Oleh karena itu,
> **kedua konfigurasi dijalankan** pada protokol untuk perbandingan
> lengkap:
> - **Konfigurasi A** — terbaik berdasarkan **Macro F1**: Intermediate scratch,
>   facs_plus_bs_80, B2 → acc=50,00%, macro_f1=**52,87%**
> - **Konfigurasi B** — terbaik berdasarkan **Accuracy**: Intermediate TL,
>   blendshape_52, B3 → acc=**55,00%**, macro_f1=52,55%

---

## 1. Dataset JAFFE

| Aspek | Nilai |
|---|---|
| Sumber | Japanese Female Facial Expression (JAFFE) |
| Total gambar | 213 |
| Jumlah subjek | 10 (semua perempuan Jepang) |
| Kelas | 7: neutral, happy, sad, angry, fearful, disgusted, surprised |
| Distribusi | Sangat seimbang (~30 gambar per kelas) |
| Split subject-wise | 7 train / 1 val / 2 test subjek |
| Image shape | 224×224×3 |
| Landmark | raw_136 (MediaPipe) → facs_plus_bs_80 (FACS_28 + blendshape_52) |

---

## 2. Konfigurasi Model

Dua konfigurasi diuji karena pada protokol subject-wise, best accuracy dan
best macro F1 berasal dari model yang berbeda.

### Konfigurasi A — Best Macro F1 pada Subject-wise

| Aspek | Nilai |
|---|---|
| Model | IntermediateFusion (**scratch**, tanpa TL) |
| Feature landmark | facs_plus_bs_80 = concat(FACS_28, blendshape_52) = 80-dim |
| Scenario | B2 — WeightedRandomSampler, tanpa augmentasi |
| Learning rate | 1e-3 |
| Subject-wise: Accuracy | 50,00% |
| Subject-wise: Macro F1 | **52,87%** ← terbaik |

### Konfigurasi B — Best Accuracy pada Subject-wise

| Aspek | Nilai |
|---|---|
| Model | IntermediateFusionTransfer (**ResNet-18 TL**) |
| Feature landmark | blendshape_52 = ARKit blendshapes MediaPipe = 52-dim |
| Scenario | B3 — WeightedRandomSampler + augmentasi (photometric only*) |
| Learning rate | 1e-4 |
| Subject-wise: Accuracy | **55,00%** ← terbaik |
| Subject-wise: Macro F1 | 52,55% |

*Augmentasi geometrik (hflip, rotate) di-skip otomatis karena blendshape_52
adalah scalar features, bukan koordinat spasial.

**Hyperparameter sama untuk keduanya:** Batch=32, Epochs=50, Patience=15, Seed=42.

---

## 3. Hasil Penelitian Ini pada JAFFE

### 3.1 Subject-wise Train/test split — Protokol Utama

Dataset: `jaffe_7class` (213 sampel), split subject-wise 70:10:20.
Sumber: `docs/all_metrics_tables.md` §6.4

| Konfigurasi | Protokol | Accuracy | Macro F1 | Weighted F1 |
|---|---|:---:|:---:|:---:|
| A — scratch, facs_plus_bs_80, B2 | Subject-independent split | 50,00% | **52,87%** | 51,99% |
| B — TL, blendshape_52, B3 | Subject-independent split | **55,00%** | 52,55% | 53,04% |

### 3.2 Sample-level

Script: `scripts/run_jaffe_samplelevel_best_config.py`
Output: `models/benchmark/jaffe_samplelevel/`

| Konfigurasi | Protokol | Accuracy | Macro F1 | Weighted F1 |
|---|---|:---:|:---:|:---:|
| A — scratch, facs_plus_bs_80, B2 | Train/test split 80:20 (5× rata-rata) | 73,49% ± 6,99% | 72,00% ± 8,21% | 71,34% |
| A — scratch, facs_plus_bs_80, B2 | 10-fold CV | 75,26% ± 11,04% | 75,07% ± 11,03% | 74,97% |
| B — TL, blendshape_52, B3 | Train/test split 80:20 (5× rata-rata) | **84,19% ± 6,14%** | **83,62% ± 6,20%** | 83,65% |
| B — TL, blendshape_52, B3 | 10-fold CV | **82,21% ± 10,30%** | **81,10% ± 10,65%** | 81,40% |

---

## 4. Catatan Konfigurasi Lama (Tidak Dipakai)

File `docs/jaffe_randomsplit_results.md` dan `models/benchmark/jaffe_randomsplit/`
menggunakan konfigurasi yang berbeda:
- Feature: raw_136 (bukan facs_plus_bs_80 atau blendshape_52)
- Scenario: B1 (tanpa sampler, tanpa augmentasi)

Hasil tersebut **tidak digunakan** karena tidak konsisten dengan konfigurasi
terbaik yang dipilih dari all_metrics_tables.

---

## 5. Tabel Perbandingan dengan Penelitian Terdahulu

| Penelitian | Metode | Protokol | Akurasi |
|---|---|---|:---:|
| Akhand et al. (2021) | DenseNet-161, Transfer Learning | 10-fold CV | 99,52% |
| Singh et al. (2025) / MMSAD | CNN ringan (modul emosi MMSAD) | Train/test split 80:20 | 98,50% |
| Lasri et al. (2022) | VGG-16, Transfer Learning | 10-fold CV | 98,00% |
| Lasri et al. (2022) | VGG-16, Transfer Learning | Train/test split 80:20 | 97,70% |
| Wadhawan & Gandhi (2023) | Ensemble TL landmark-aware | 10-fold CV subject-independent | 97,14% |
| Gautam & Seeja (2023) | HOG + CNN | Train/test split (rasio tidak dirinci) | 91,43% |
| **Penelitian ini** (Konfigurasi A) | Intermediate Fusion scratch (citra + facs_plus_bs_80), B2 | Train/test split 80:20 (5× rata-rata) | **73,49%** |
| **Penelitian ini** (Konfigurasi A) | Intermediate Fusion scratch (citra + facs_plus_bs_80), B2 | 10-fold CV | **75,26%** |
| **Penelitian ini** (Konfigurasi B) | Intermediate Fusion TL (citra + blendshape_52), ResNet-18 TL, B3 | Train/test split 80:20 (5× rata-rata) | **84,19%** |
| **Penelitian ini** (Konfigurasi B) | Intermediate Fusion TL (citra + blendshape_52), ResNet-18 TL, B3 | 10-fold CV | **82,21%** |
| **Penelitian ini** *(protokol utama, Konfigurasi A)* | Intermediate Fusion scratch (citra + facs_plus_bs_80), B2 | Subject-independent split | 50,00% |
| **Penelitian ini** *(protokol utama, Konfigurasi B)* | Intermediate Fusion TL (citra + blendshape_52), ResNet-18 TL, B3 | Subject-independent split | **55,00%** |

---

## 6. Verifikasi Paper Pembanding

| Paper | Kelas | Nama Kelas | Protokol | Terverifikasi? |
|---|:---:|---|---|:---:|
| Akhand et al. (2021) | 7 | Afraid, Angry, Disgusted, Sad, Happy, Surprised, Neutral | 10-fold CV | ✅ |
| Singh et al. (2025) | 7 | Anger, Fear, Disgust, Happiness, Neutral, Sadness, Surprise | Train/test split 80:20 | ✅ |
| Lasri et al. (2022) | 7 | Angry, Disgust, Fear, Happiness, Sad, Surprise, Neutral | Train/test split 80:20 + 10-fold CV | ✅ |
| Wadhawan & Gandhi (2023) | 7 | Anger, Disgust, Fear, Happiness, Sadness, Surprise, Neutral | 10-fold **subject-independent** | ✅ |
| Gautam & Seeja (2023) | 7 | 6 basic + neutral (tidak eksplisit) | Tidak dirinci (train/test Kaggle) | ⚠️ |

**Catatan kelas:** Semua paper menggunakan **7 kelas termasuk neutral, tanpa contempt** — sama dengan dataset JAFFE penelitian ini. ✅

**Catatan Singh:** Dalam teks eksperimen disebutkan akurasi **98,1%**, sedangkan abstract dan tabel comparison menyebut **98,50%**. Yang digunakan di tabel adalah 98,50% (angka tabel comparison Singh).

**Catatan Akhand:** Train/test split 90:10 menghasilkan 100% (terlalu kecil test set — ~21 gambar), 10-fold CV lebih valid → 99,52% yang dipakai.

---

## 7. Analisis

### 7.1 Mengapa Subject-wise Rendah (50%)?

JAFFE hanya memiliki **10 subjek**. Dengan split 70:10:20:
- Train: 7 subjek (~152 sampel)
- Test: 2 subjek (~40 sampel)

Model yang dilatih pada ekspresi 7 orang diuji pada 2 orang yang karakteristik
wajahnya sangat berbeda. Ini adalah tantangan genuine dari dataset yang sangat
kecil dan homogen (semua perempuan Jepang).

### 7.2 Mengapa Angka Literatur Sangat Tinggi (97–99%)?

Hampir seluruh paper pembanding (Akhand, Singh, Lasri, Gautam) menggunakan
**random split** — gambar dari subjek yang sama tersebar di train dan test.
Dengan hanya 213 gambar dari 10 orang, model dapat "mengenali wajah" subjek
yang sudah dilihat saat training → angka menjadi sangat tinggi (97–100%).

**Wadhawan (97,14%)** adalah satu-satunya yang menggunakan **subject-independent**
— paling comparable dengan kita — namun tetap lebih tinggi karena:
- Ensemble 5 subnetwork (masing-masing fokus pada region wajah berbeda)
- Data augmentation masif
- Backbone lebih dalam

### 7.3 Pernyataan untuk Tesis

> Rendahnya akurasi pada protokol subject-wise (50–55%) mencerminkan tantangan
> genuine generalisasi lintas-subjek pada dataset JAFFE yang sangat kecil
> (213 gambar, 10 subjek). Dengan protokol yang sama dengan
> literatur, model terbaik (Konfigurasi B) mencapai **84,19%**, masih terdapat
> gap ~15% dibanding paper terbaik (99,52%). Gap ini terutama disebabkan oleh:
> (1) model penelitian ini dioptimalkan untuk dataset primer dan bersifat
> generalis, bukan khusus JAFFE; (2) paper pembanding menggunakan backbone lebih
> dalam (DenseNet-161, VGG-16) yang lebih sesuai untuk dataset kecil;
> (3) JAFFE hanya 213 gambar — dataset yang terlalu kecil untuk model multimodal
> yang lebih kompleks.

---

## Lampiran A: Hasil Lengkap Benchmark JAFFE

### A.1 Subject-wise Train/test split — Kedua Konfigurasi

Sumber: `docs/all_metrics_tables.md` §6.4

| Konfigurasi | Model | Feature | Scenario | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|:---:|:---:|:---:|:---:|
| A | Intermediate scratch | facs_plus_bs_80 | B2 | 50,00% | **52,87%** | 51,99% |
| B | Intermediate TL | blendshape_52 | B3 | **55,00%** | 52,55% | 53,04% |

### A.2 Sample-level — Kedua Konfigurasi

Output: `models/benchmark/jaffe_samplelevel/`

| Konfigurasi | Protokol | Accuracy | Macro F1 | Weighted F1 |
|---|---|:---:|:---:|:---:|
| A — scratch, facs_plus_bs_80, B2 | Train/test split 80:20 (5× rata-rata) | 73,49% ± 6,99% | 72,00% ± 8,21% | 71,34% |
| A — scratch, facs_plus_bs_80, B2 | 10-fold CV | 75,26% ± 11,04% | 75,07% ± 11,03% | 74,97% |
| **B — TL, blendshape_52, B3** | **Train/test split 80:20 (5× rata-rata)** | **84,19% ± 6,14%** | **83,62% ± 6,20%** | 83,65% |
| **B — TL, blendshape_52, B3** | **10-fold CV** | **82,21% ± 10,30%** | **81,10% ± 10,65%** | 81,40% |

---

## Lampiran B: Daftar Pustaka (Harvard Anglia)

Akhand, M.A.H., Roy, S., Siddique, N., Kamal, M.A.S. and Shimamura, T. (2021)
'Facial emotion recognition using transfer learning in the deep CNN', *Electronics*,
10(9), p. 1036. Available at: https://doi.org/10.3390/electronics10091036.

Gautam, C. and Seeja, K.R. (2023) 'Facial emotion recognition using handcrafted
features and CNN', *Procedia Computer Science*, 218, pp. 1295–1303. Available at:
https://doi.org/10.1016/j.procs.2023.01.108.

Lasri, I., Riadsolh, A. and Elbelkacemi, M. (2022) 'Facial emotion recognition of
deaf and hard-of-hearing students for engagement detection using deep learning',
*Education and Information Technologies*, 28, pp. 4069–4092. Available at:
https://doi.org/10.1007/s10639-022-11370-4.

Singh, R., Ramanujam, E. and Naresh Babu, M. (2025) 'MMSAD — A multi-modal student
attentiveness detection in smart education using facial features and landmarks',
*Journal of Ambient Intelligence and Smart Environments*, 17(3). Available at:
https://doi.org/10.1177/18761364251315239.

Wadhawan, R. and Gandhi, T.K. (2023) 'Landmark-aware and part-based ensemble
transfer learning network for static facial expression recognition from images',
*IEEE Transactions on Artificial Intelligence*, 4(2), pp. 349–361. Available at:
https://doi.org/10.1109/TAI.2022.3172272.
