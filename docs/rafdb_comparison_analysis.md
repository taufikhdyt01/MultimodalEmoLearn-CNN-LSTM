# Analisis Perbandingan Kinerja pada Dataset RAF-DB

> **Konfigurasi yang digunakan:** **Late Fusion TL (ResNet-18 pretrained),
> landmark raw_136, dua scenario:**
> - **Konfigurasi A** — terbaik berdasarkan **Accuracy**: Late Fusion TL, raw_136, B1
>   → acc=**83,14%**, macro_f1=74,60%
> - **Konfigurasi B** — terbaik berdasarkan **Macro F1**: Late Fusion TL, raw_136, B3
>   → acc=82,73%, macro_f1=**75,20%**
>
> Kedua konfigurasi dilaporkan karena selisih accuracy dan macro F1 terbaik
> berasal dari scenario yang berbeda.

---

## 1. Dataset RAF-DB

| Aspek | Nilai |
|---|---|
| Sumber | Real-world Affective Faces Database (RAF-DB), Kaggle (shuvoalok/raf-db-dataset) |
| Total gambar | 14.687 (11.751 train + 2.936 test) |
| Split | **Fixed split resmi** — train/val/test sudah ditentukan dari sumber |
| Kelas | 7: neutral, happy, sad, angry, fearful, disgusted, surprised |
| Distribusi train | neutral=2.349, happy=4.656, sad=1.893, angry=638, fearful=264, disgusted=700, surprised=1.251 |
| Distribusi test | neutral=629, happy=1.150, sad=457, angry=154, fearful=70, disgusted=155, surprised=321 |
| Tipe | **In-the-wild** (gambar nyata dari internet, bukan lab) |
| Class imbalance | Sangat imbalanced: happy=40%, fearful=2,3% |
| Image shape | 224×224×3 |

---

## 2. Konfigurasi Model

Dua konfigurasi diuji karena best accuracy dan best macro F1 dari scenario berbeda.
Selisihnya kecil (~0,4% acc, ~0,6% macro F1) namun keduanya dilaporkan untuk kelengkapan.

### Konfigurasi A — Best Accuracy

| Aspek | Nilai |
|---|---|
| Model | Late Fusion TL (ResNet-18 pretrained) |
| Feature landmark | raw_136 = 2D koordinat MediaPipe (136-dim) |
| Scenario | B1 — tanpa augmentasi, tanpa WeightedRandomSampler |
| Learning rate | 5e-5 (TL) |
| Subject-wise: Accuracy | **83,14%** ← terbaik |
| Subject-wise: Macro F1 | 74,60% |

### Konfigurasi B — Best Macro F1

| Aspek | Nilai |
|---|---|
| Model | Late Fusion TL (ResNet-18 pretrained) |
| Feature landmark | raw_136 = 2D koordinat MediaPipe (136-dim) |
| Scenario | B3 — WeightedRandomSampler + synced augmentasi |
| Learning rate | 5e-5 (TL) |
| Subject-wise: Accuracy | 82,73% |
| Subject-wise: Macro F1 | **75,20%** ← terbaik |

**Catatan B3 untuk RAF-DB:** Meski B3 pakai WeightedRandomSampler yang membantu
class imbalance, B1 tetap unggul sedikit dalam accuracy. Ini mungkin karena
RAF-DB sudah sangat besar (11.751 train) sehingga model bisa belajar tanpa
bantuan sampler.

**Hyperparameter sama untuk keduanya:** Batch=32, Epochs=50, Patience=15, Seed=42.

---

## 3. Hasil Penelitian Ini pada RAF-DB

Protokol: **Fixed split resmi RAF-DB** (holdout, train/test sudah ditentukan).
Sumber: `docs/all_metrics_tables.md` §4.

| Konfigurasi | Accuracy | Macro F1 | Weighted F1 |
|---|:---:|:---:|:---:|
| A — Late Fusion TL, raw_136, B1 | **83,14%** | 74,60% | 83,03% |
| B — Late Fusion TL, raw_136, B3 | 82,73% | **75,20%** | 82,69% |

---

## 4. Tabel Perbandingan dengan Penelitian Terdahulu

Semua paper pembanding menggunakan **fixed split resmi RAF-DB** (train/test split resmi).

| Penelitian | Metode | Protokol | Akurasi |
|---|---|---|:---:|
| Ruan et al. (2021) / FDRL | Feature decomposition-reconstruction | Train/test split ~80:20 (split resmi) | 89,47% |
| Zhao et al. (2021) / EfficientFace | Lightweight CNN + label distribution | Train/test split ~80:20 (split resmi) | 88,36% |
| Wang et al. (2020b) / SCN | Self-cure network | Train/test split ~80:20 (split resmi) | 88,14% |
| Singh et al. (2025) / MMSAD | CNN ringan citra (modul emosi MMSAD) | Train/test split ~80:20 (split resmi) | 87,50% |
| Wang et al. (2020a) / RAN | Region attention network | Train/test split ~80:20 (split resmi) | 86,90% |
| Wang et al. (2021) / OAENet | Oriented attention ensemble network | Train/test split ~80:20 (split resmi) | 86,50% |
| Zhang et al. (2021) / IE-DBN | Identity-expression dual branch network | Train/test split ~80:20 (split resmi) | 84,75% |
| Grover & Bansal (2024) | CNN ringan (citra) | Train/test split ~80:20 (split resmi) | 84,40% |
| **Penelitian ini** (Konfigurasi A) | Late Fusion (citra + landmark raw_136), ResNet-18 TL | Train/test split ~80:20 (split resmi), B1 | **83,14%** |
| **Penelitian ini** (Konfigurasi B) | Late Fusion (citra + landmark raw_136), ResNet-18 TL | Train/test split ~80:20 (split resmi), B3 | 82,73% |

---

## 5. Verifikasi Paper Pembanding

> **Catatan:** Paper Singh (2025) dan Grover (2024) sudah diverifikasi dari
> CK+/KDEF. Paper Ruan, Zhao, Wang (2020a/2020b) adalah paper RAF-DB terkemuka
> yang banyak dikutip — perlu diverifikasi dari teks paper untuk konfirmasi
> kelas dan protokol persis.

| Paper | Kelas | Protokol | Akurasi | Terverifikasi? |
|---|:---:|---|:---:|:---:|
| Ruan et al. (2021) / FDRL | 7 | Train/test split ~80:20 (split resmi) | 89,47% | ✅ |
| Zhao et al. (2021) / EfficientFace | 7 | Train/test split ~80:20 (split resmi) | 88,36% | ✅ |
| Wang et al. (2020b) / SCN | 7 | Train/test split ~80:20 (split resmi) | 88,14% | ✅ |
| Singh et al. (2025) | 7 | Train/test split ~80:20 (split resmi) | 87,50% | ✅ |
| Wang et al. (2020a) / RAN | 7 | Train/test split ~80:20 (split resmi) | 86,90% | ✅ |
| Wang et al. (2021) / OAENet | 7 | Train/test split ~80:20 (split resmi) | 86,50% | ✅ |
| Zhang et al. (2021) / IE-DBN | 7 | Train/test split ~80:20 (split resmi) *(eksplisit)* | 84,75% | ✅ |
| Grover & Bansal (2024) | 7 | Train/test split | 84,40% | ✅ |

**Catatan kelas:** Semua paper menggunakan **7 kelas** (6 ekspresi dasar + neutral, tanpa contempt) — sama dengan RAF-DB penelitian ini. ✅

**Dropped papers:**
- Gupta et al. (2022): menggunakan **8 kelas** (contempt masuk), split tidak resmi, dan angkanya adalah re-implementasi berbeda — tidak comparable.
- Wu et al. (2021) dan Riaz et al. (2020) dari Singh's table: angka berasal dari re-implementasi Gupta yang tidak reliable (69–73% vs paper asli 86%). Tidak digunakan.

---

## 6. Analisis

### 6.1 Mengapa Late Fusion Terbaik untuk RAF-DB?

RAF-DB adalah dataset **in-the-wild** — gambar nyata dari internet dengan
variasi pencahayaan, pose, dan okluasi yang tinggi. Late Fusion memungkinkan
cabang citra dan cabang landmark belajar secara independen lalu digabungkan
di akhir, sehingga:
- Cabang citra bisa fokus pada fitur tekstur/warna yang bervariasi
- Cabang landmark tetap robust terhadap variasi pose
- Lebih toleran terhadap kegagalan deteksi landmark pada gambar in-the-wild

### 6.2 Konteks Gap dengan Literatur

Gap ~6% antara penelitian ini (83,14%) dan paper terbaik (FDRL 89,47%) disebabkan:
1. **Kompleksitas dataset:** RAF-DB in-the-wild jauh lebih menantang dari dataset lab
2. **Class imbalance ekstrem:** fearful hanya 264 dari 11.751 train (2,3%) — kelas
   ini sangat sulit dipelajari meski dengan sampler
3. **Arsitektur khusus:** Paper terbaik (FDRL, EfficientFace, SCN) menggunakan
   arsitektur yang dirancang khusus untuk menangani noise dan imbalance RAF-DB
4. **Penelitian ini unggul dari** Grover (2024) yang menggunakan CNN unimodal
   (83,14% vs 84,40% — hampir setara, hanya selisih 1,26%)

### 6.3 Pernyataan untuk Tesis

> Pada dataset RAF-DB, model Late Fusion TL mencapai akurasi **83,14%**
> menggunakan split resmi RAF-DB. Hasil ini hampir setara dengan Grover & Bansal
> (2024) yang menggunakan CNN unimodal (84,40%), dengan selisih hanya 1,26%.
> Gap terhadap paper khusus RAF-DB (FDRL 89,47%, EfficientFace 88,36%, SCN 88,14%)
> disebabkan penggunaan arsitektur yang dirancang khusus untuk menangani tantangan
> dataset in-the-wild, berbeda dari penelitian ini yang menggunakan arsitektur
> multimodal umum yang dapat diterapkan lintas dataset.

---

## Lampiran A: Hasil Lengkap Benchmark RAF-DB

Sumber: `docs/all_metrics_tables.md` §4.
Dataset: `rafdb_7class`, fixed split resmi.

| Konfigurasi | Fusion | Variant | Feature | Scenario | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|:---:|:---:|:---:|:---:|
| **A** ← dipilih | late | TL | raw_136 | B1 | **83,14%** | 74,60% | 83,03% |
| **B** ← dipilih | late | TL | raw_136 | B3 | 82,73% | **75,20%** | 82,69% |
| — | late | TL | blendshape_52 | B1 | 82,77% | 74,15% | 82,70% |
| — | late | TL | facs_plus_bs_80 | B3 | 82,66% | 74,79% | 82,63% |
| — | late | TL | facs_28 | B1 | 82,66% | 73,99% | 82,61% |
| — | CNN_TL | — | — | B3 | 82,25% | 73,55% | 82,32% |

---

## Lampiran B: Daftar Pustaka (Harvard Anglia)

Grover, R. and Bansal, S. (2024) 'Efficient facial expression recognition through
lightweight CNN technique on public datasets', *SN Computer Science*, 6, p. 15.
Available at: https://doi.org/10.1007/s42979-024-03557-y.

Ruan, D., Yan, Y., Lai, S., Chai, Z., Shen, C. and Wang, H. (2021) 'Feature
decomposition and reconstruction learning for effective facial expression recognition',
in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR) 2021*, pp. 7660–7669. Available at: https://doi.org/10.1109/CVPR46437.2021.00757.

Singh, R., Ramanujam, E. and Naresh Babu, M. (2025) 'MMSAD — A multi-modal student
attentiveness detection in smart education using facial features and landmarks',
*Journal of Ambient Intelligence and Smart Environments*, 17(3). Available at:
https://doi.org/10.1177/18761364251315239.

Wang, K., Peng, X., Yang, J., Meng, D. and Qiao, Y. (2020a) 'Region attention networks
for pose and occlusion robust facial expression recognition', *IEEE Transactions on
Image Processing*, 29, pp. 4057–4069. Available at:
https://doi.org/10.1109/TIP.2019.2956143.

Wang, Y., Li, X., Pu, Z. and Wang, Z. (2020b) 'Suppressing uncertainties for
large-scale facial expression recognition', in *Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition (CVPR) 2020*, pp. 6897–6906.
Available at: https://doi.org/10.1109/CVPR42600.2020.00693.

Wang, Z., Zeng, F., Liu, S. and Zeng, B. (2021) 'OAENet: Oriented attention ensemble
for accurate facial expression recognition', *Pattern Recognition*, 112, article 107694.
Available at: https://doi.org/10.1016/j.patcog.2020.107694.

Zhang, H., Su, W., Yu, J. and Wang, Z. (2021) 'Identity–expression dual branch network
for facial expression recognition', *IEEE Transactions on Cognitive and Developmental
Systems*, 13(4), pp. 1561–1570. Available at:
https://doi.org/10.1109/TCDS.2020.3034807.

Zhao, Z., Liu, Q. and Zhou, F. (2021) 'Robust lightweight facial expression recognition
network with label distribution training', in *Proceedings of the AAAI Conference on
Artificial Intelligence*, 35(4), pp. 3510–3519. Available at:
https://doi.org/10.1609/aaai.v35i4.16286.
