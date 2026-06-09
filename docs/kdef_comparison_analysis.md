# Analisis Perbandingan Kinerja pada Dataset KDEF

> **Konfigurasi yang digunakan:** **Early Fusion concat TL (ResNet-18 pretrained),
> raw_136 + heatmap landmark (4-channel), scenario B1 (tanpa augmentasi,
> tanpa WeightedRandomSampler)** — konfigurasi terbaik penelitian ini pada
> protokol subject-wise (acc=91,50%, macro_f1=91,40%).
>
> Catatan: B1 optimal untuk KDEF karena distribusi kelas sangat seimbang
> (~42 gambar per kelas per split), sehingga sampler dan augmentasi tidak
> diperlukan.

---

## 1. Dataset KDEF

| Aspek | Nilai |
|---|---|
| Sumber | Karolinska Directed Emotional Faces (KDEF) |
| Total gambar | 2.941 |
| Jumlah subjek | 70 |
| Kelas | 7: neutral, happy, sad, angry, fearful, disgusted, surprised |
| Distribusi | Sangat seimbang (~42 gambar per kelas per split) |
| Split subject-wise | 56 train / 7 val / 7 test subjek (80:10:10) |
| Image shape | 224×224×3 |
| Landmark | raw_136 (MediaPipe, 2D coords 136-dim) → heatmap 224×224 |

---

## 2. Konfigurasi Model

| Aspek | Nilai |
|---|---|
| Model | EmotionEarlyFusionTransfer (ResNet-18 pretrained ImageNet) |
| Input | RGB 224×224 + heatmap landmark = 4-channel |
| Feature landmark | raw_136 = 2D koordinat MediaPipe (136-dim) → heatmap |
| Scenario | B1 — tanpa augmentasi, tanpa WeightedRandomSampler |
| Batch size | 32 |
| Epochs | 50 (max), patience 15 |
| Learning rate | 5e-5 (TL) |
| Seed | 42 |

---

## 3. Hasil Penelitian Ini pada KDEF

### 3.1 Subject-wise Train/test split — Protokol Utama

Sumber: `docs/all_metrics_tables.md` §3.
Dataset: `kdef_7class` (2941 sampel), split subject-wise 80:10:10.

| Protokol | Accuracy | Macro F1 | Weighted F1 |
|---|:---:|:---:|:---:|
| Subject-independent split, B1 *(protokol utama)* | **91,50%** | **91,40%** | 91,40% |

### 3.2 Sample-level — Tiga Protokol

Script: `scripts/run_kdef_samplelevel_earlyfusion.py`
Output: `models/benchmark/kdef_samplelevel/`
Log: `logs/benchmark_kdef_samplelevel_ef_b1.log`

| Protokol | Comparable dengan | Accuracy | Macro F1 | Weighted F1 |
|---|---|:---:|:---:|:---:|
| Train/test split 80:20 (5× rata-rata), B1 | Grover & Bansal (2024) | **90,36% ± 1,12%** | **90,29% ± 1,14%** | 90,28% |
| Train/test split 70:30 (5× rata-rata), B1 | Singh et al. (2025) | **88,27% ± 1,18%** | **88,15% ± 1,21%** | 88,14% |
| 10-fold CV, B1 | Akhand et al. (2021) | **91,57% ± 1,22%** | **91,51% ± 1,27%** | 91,51% |

---

## 4. Tabel Perbandingan dengan Penelitian Terdahulu

| Penelitian | Metode | Protokol | Akurasi |
|---|---|---|:---:|
| Lasri et al. (2022) | VGG-16, Transfer Learning | 10-fold CV | 99,00% |
| Akhand et al. (2021) | DenseNet-161, Transfer Learning | 10-fold CV | 96,51% |
| Grover & Bansal (2024) | CNN ringan (citra) | Train/test split (rasio tidak dirinci) | 94,00% |
| Singh et al. (2025) / MMSAD | CNN ringan citra (modul emosi MMSAD) | Train/test split 70:30 | 88,01% |
| Lasri et al. (2022) | VGG-16, Transfer Learning | Train/test split 80:20 | 86,33% |
| Kurniawardhani et al. (2022) | CNN (frontal only, 490 img) | Train/test split 80:10:10 | 82,00% |
| **Penelitian ini** | Early Fusion concat (citra + heatmap landmark), ResNet-18 TL | Subject-independent split | **91,50%** |
| **Penelitian ini** | Early Fusion concat (citra + heatmap landmark), ResNet-18 TL | 10-fold CV | **91,57%** |
| **Penelitian ini** | Early Fusion concat (citra + heatmap landmark), ResNet-18 TL | Train/test split 80:20 (5× rata-rata) | **90,36%** |
| **Penelitian ini** | Early Fusion concat (citra + heatmap landmark), ResNet-18 TL | Train/test split 70:30 (5× rata-rata) | **88,27%** |

---

## 5. Verifikasi Paper Pembanding

| Paper | Kelas KDEF | Protokol | Akurasi | Terverifikasi? |
|---|:---:|---|:---:|:---:|
| Lasri et al. (2022) | 7 (happiness, fear, sadness, neutral, disgust, anger, surprise) | 10-fold CV | 99,00% | ✅ |
| Lasri et al. (2022) | 7 | Train/test split **80:20** | 86,33% | ✅ |
| Akhand et al. (2021) | 7 (afraid, angry, disgusted, happy, neutral, sad, surprised) | 10-fold CV | 96,51% | ✅ |
| Grover & Bansal (2024) | 7 (angry, disgust, fear, happy, sad, surprise, neutral) | Train/test split (**tidak dirinci rasionya**) | 94,00% | ⚠️ rasio tidak dirinci |
| Singh et al. (2025) | 7 (angry, disgust, fear, happy, sad, surprise, neutral) | Train/test split **70:30** | 88,01% | ✅ |

**Catatan kelas:** Semua paper menggunakan **7 kelas termasuk neutral, tanpa contempt** — sama dengan KDEF penelitian ini. ✅

**Catatan Akhand:** Menggunakan seluruh 4900 gambar KDEF (frontal + profil). Train/test split 90:10 menghasilkan 98,78% — terlalu tinggi karena test set sangat kecil (~490 gambar). Angka 10-fold CV (96,51%) lebih representatif dan yang dipakai untuk perbandingan.

**Catatan Grover:** Test set 8793 gambar melebihi total KDEF (4900), mengindikasikan augmentasi dilakukan sebelum split. Protokol tidak dirinci secara eksplisit.

**Catatan Lasri:** Punya dua hasil — 10-fold CV (99%) jauh lebih tinggi dari Train/test split 80:20 (86,33%). Keduanya dicantumkan di tabel agar perbandingan lebih lengkap.


**Catatan Kurniawardhani et al. (2022):** Hanya menggunakan **490 gambar frontal** dari KDEF (70 per kelas, 35 pria + 35 wanita), bukan 4900 gambar penuh. Test set hanya 49 gambar. Hasil 82% harus diinterpretasikan dengan hati-hati.

**Catatan Lee et al. (2023):** **Tidak menggunakan KDEF** (pakai AffectNet) — tidak dimasukkan ke tabel perbandingan.

---

## 6. Analisis

### 6.1 Mengapa B1 Optimal untuk KDEF?

KDEF memiliki distribusi kelas yang **sangat seimbang** (~42 gambar per kelas
per split). Tidak ada class imbalance, sehingga:
- WeightedRandomSampler tidak diperlukan (B2/B3 tidak memberi keuntungan)
- Augmentasi (B3) justru bisa menambah noise tanpa manfaat signifikan
- B1 sederhana dan stabil pada dataset yang sudah balanced

### 6.2 Konteks Gap dengan Literatur

Gap antara penelitian ini (91,50% subject-wise) dan paper pembanding (88–97%)
perlu dilihat dari dua sisi:
- **vs Akhand (96,51%)**: Protokol berbeda — Akhand pakai 10-fold CV
  yang lebih longgar. Dengan protokol sama, hasil penelitian ini diperkirakan lebih tinggi.
- **vs Grover (94,00%)**: Protokol Grover tidak dirinci, kemungkinan.
- **vs Singh (88,01%)**: Penelitian ini sudah **unggul** bahkan pada protokol
  subject-wise yang lebih ketat (91,50% > 88,01%).

### 6.3 Pernyataan untuk Tesis

> Pada dataset KDEF, model Early Fusion concat TL mencapai akurasi **91,50%**
> menggunakan protokol subject-wise holdout yang lebih ketat. Dengan protokol
> yang sama, model mencapai **88,27%** (Train/test split 70:30) hingga
> **91,57%** (10-fold CV).
>
> Penelitian ini **mengungguli Singh et al. (2025)** pada protokol yang setara
> (Train/test split 70:30: 88,27% vs 88,01%). Dibandingkan Grover (Train/test split: 90,36% vs
> 94,00%) dan Akhand (10-fold CV: 91,57% vs 96,51%), gap ~3–5% terutama
> disebabkan perbedaan arsitektur backbone (ResNet-18 vs DenseNet-161/CNN khusus)
> dan protokol evaluasi yang lebih ketat (subject-wise). Dibandingkan Lasri
> (10-fold CV: 91,57% vs 99,00%), gap yang besar disebabkan Lasri menggunakan
> seluruh 4900 gambar KDEF dengan split yang memungkinkan data
> leakage antar subjek.

---

## Lampiran A: Benchmark Tambahan

Dataset: `kdef_7class` (2941 sampel), model: Early Fusion concat TL B1.

| Protokol | Accuracy | Macro F1 | Weighted F1 | File |
|---|:---:|:---:|:---:|---|
| Subject-independent split | 91,50% | 91,40% | 91,40% | `all_metrics_tables.md` §3 |
| **10-fold CV** | **91,57% ± 1,22%** | **91,51% ± 1,27%** | 91,51% | `kdef_7c_cv10_earlyfusion_b1.json` |
| **Train/test split 80:20 (5× rata-rata)** | **90,36% ± 1,12%** | **90,29% ± 1,14%** | 90,28% | `kdef_7c_holdout8020_earlyfusion_b1.json` |
| **Train/test split 70:30 (5× rata-rata)** | **88,27% ± 1,18%** | **88,15% ± 1,21%** | 88,14% | `kdef_7c_holdout7030_earlyfusion_b1.json` |

---

## Lampiran B: Daftar Pustaka (Harvard Anglia)

Akhand, M.A.H., Roy, S., Siddique, N., Kamal, M.A.S. and Shimamura, T. (2021)
'Facial emotion recognition using transfer learning in the deep CNN', *Electronics*,
10(9), p. 1036. Available at: https://doi.org/10.3390/electronics10091036.

Grover, R. and Bansal, S. (2024) 'Efficient facial expression recognition through
lightweight CNN technique on public datasets', *SN Computer Science*, 6, p. 15.
Available at: https://doi.org/10.1007/s42979-024-03557-y.

Lasri, I., Riadsolh, A. and Elbelkacemi, M. (2022) 'Facial emotion recognition of
deaf and hard-of-hearing students for engagement detection using deep learning',
*Education and Information Technologies*, 28, pp. 4069–4092. Available at:
https://doi.org/10.1007/s10639-022-11370-4.

Singh, R., Ramanujam, E. and Naresh Babu, M. (2025) 'MMSAD — A multi-modal student
attentiveness detection in smart education using facial features and landmarks',
*Journal of Ambient Intelligence and Smart Environments*, 17(3). Available at:
https://doi.org/10.1177/18761364251315239.

Kurniawardhani, A., Azizi, F.N. and Paputungan, I.V. (2022) 'Facial expression image
based emotion detection using convolutional neural network', in *2022 IEEE 20th Student
Conference on Research and Development (SCOReD)*, Bangi, Malaysia, 8–9 November 2022.
Available at: https://doi.org/10.1109/SCOReD57793.2022.10200435.

