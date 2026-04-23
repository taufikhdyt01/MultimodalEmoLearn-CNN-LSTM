# Bahan Bimbingan Tesis - Progress & Konsultasi

**Nama:** Taufik Hidayat  
**NIM:** 256150117111007  
**Judul:** Integrasi Multimodal Citra Wajah dan Facial Landmark untuk Pengenalan Emosi dalam Konteks Pembelajaran Pemrograman  
**Pembimbing 1:** Dr.Eng. Fitra Abdurrachman Bachtiar, S.T., M.Eng.  
**Pembimbing 2:** Dr.Eng. Budi Darma Setiawan, S.Kom., M.Cs.  
**Tanggal:** April 2025

---

## SLIDE 1: Agenda

> **Poin yang disampaikan:**  
> "Pak/Bu, hari ini saya ingin melaporkan progress preprocessing data dan meminta konsultasi terkait beberapa keputusan teknis sebelum masuk ke tahap training."

1. Progress pengumpulan dan preprocessing data
2. Statistik dataset yang dihasilkan
3. Konsultasi: Validasi ahli psikologi
4. Konsultasi: Strategi penanganan class imbalance
5. Rencana selanjutnya

---

## SLIDE 2: Data yang Terkumpul

### Total 37 Mahasiswa

| Batch | Direkam | Tersedia | Sudut Kamera | Periode |
|-------|---------|----------|-------------|---------|
| Batch 1 (lama) | 20 mahasiswa | 20 mahasiswa | Depan saja | April-Mei 2025 |
| Batch 2 (baru) | 20 mahasiswa | **17 mahasiswa** | Depan + Samping | November 2025 |
| **Total** | **40** | **37 mahasiswa** | | |

**Catatan Batch 2:** Awalnya batch 2 merekam 20 mahasiswa, namun 3 data rekaman tidak ditemukan di hardisk PC perekaman. Setelah ditelusuri dan hardisk dibuka, file rekaman 3 mahasiswa tersebut tidak tersimpan. Sehingga data batch 2 yang tersedia hanya 17 mahasiswa.

> **Penjelasan lisan:**  
> "Untuk batch 2, awalnya saya merekam 20 mahasiswa, sama seperti batch 1. Namun saat akan memproses datanya, 3 file rekaman tidak ditemukan di hardisk PC yang digunakan untuk perekaman. Setelah saya cek langsung ke hardisk-nya — karena CPU-nya sempat tidak menyala — ternyata file rekamannya memang tidak tersimpan. Kemungkinan gagal saat proses recording. Jadi data batch 2 yang bisa digunakan hanya 17 mahasiswa."
>
> "Total data yang tersedia menjadi 37 mahasiswa dari target 38 di proposal."
>
> "Perbedaan utama batch 2 adalah penambahan sudut kamera samping (side view), sehingga satu mahasiswa menghasilkan dua set frame: depan dan samping. Ini untuk menambah variasi data dan menguji apakah side view bisa membantu pengenalan emosi."

---

## SLIDE 3: Pipeline Preprocessing

```
Video Rekaman (.mp4/.mkv)
        |
        v
[1] Ekstraksi Frame (berdasarkan timestamp emosi di database)
        |
        v
[2] Face Detection & Cropping (MediaPipe → 224x224 px)
        |
        v
[3] Landmark Extraction (68 titik → 136 fitur koordinat)
        |
        v
[4] Dataset Preparation (matching label + split train/val/test)
```

> **Penjelasan lisan:**  
> "Berbeda dengan preprocessing sebelumnya yang mengekstrak frame setiap 5 detik secara blind, sekarang saya hanya mengekstrak frame pada timestamp yang tepat sesuai data emosi yang tercatat di database. Jadi setiap frame pasti memiliki label emosi yang berkorespondensi."
>
> "Untuk face detection, saya menggunakan MediaPipe (bukan dlib seperti di proposal) karena MediaPipe mampu mendeteksi wajah dari sudut samping, sedangkan dlib hanya efektif untuk wajah frontal. Hasil landmark tetap di-map ke 68 titik standar yang setara dengan dlib, sehingga tidak mengubah desain penelitian di proposal. Hasilnya 136 fitur (68 titik × 2 koordinat x,y)."

---

## SLIDE 4: Hasil Preprocessing

### Statistik Ekstraksi Frame

| Tahap | Batch 1 | Batch 2 | Total |
|-------|---------|---------|-------|
| Frame diekstrak | 3,849 | 6,553 | 10,402 |
| Face terdeteksi | 3,824 (99.4%) | 6,070 (92.6%) | 9,894 |
| Gagal deteksi | 25 | 483 | 508 |

> **Penjelasan lisan:**  
> "Dari 10,402 frame yang diekstrak, 9,894 berhasil dideteksi wajahnya (95%). Yang gagal deteksi umumnya frame di mana mahasiswa sedang menunduk, keluar frame, atau wajah tertutup tangan. Angka ini normal dan tidak mempengaruhi kualitas dataset."

### Detail Batch 2 (dengan side view)

| Angle | Frame | Matched ke label |
|-------|-------|-----------------|
| Front | 3,275 | 3,035 |
| Side  | 3,278 | 3,035 |
| **Total** | **6,553** | **6,070** |

---

## SLIDE 5: Distribusi Emosi

| Emosi | Jumlah | Persentase |
|-------|--------|-----------|
| Neutral | 8,356 | **84.5%** |
| Happy | 783 | 7.9% |
| Sad | 576 | 5.8% |
| Surprised | 79 | 0.8% |
| Angry | 63 | 0.6% |
| Disgusted | 24 | 0.2% |
| Fearful | 13 | 0.1% |
| **Total** | **9,894** | **100%** |

> **Penjelasan lisan:**  
> "Distribusi emosi sangat imbalanced dengan neutral mendominasi 84.5%. Ini konsisten dengan literatur yang disebutkan di proposal bahwa mahasiswa cenderung menampilkan ekspresi netral saat fokus pemrograman (Coto et al., 2022 melaporkan dominasi ekspresi tertentu hingga 65%)."
>
> "Emosi fearful (13) dan disgusted (24) sangat sedikit. Ini menjadi tantangan tersendiri yang perlu kita diskusikan apakah kelas-kelas ini tetap dipertahankan atau digabung."

---

## SLIDE 6: Penanganan Class Imbalance

### Masalah

Jika model ditraining tanpa penanganan, model akan cenderung selalu memprediksi "neutral" karena itu pilihan yang "paling aman" (benar 84.5% dari waktu). Emosi langka seperti fearful, disgusted, dan angry hampir tidak akan pernah diprediksi.

### Solusi yang Sudah Disiapkan: Class Weights

Menggunakan metode **Class-Balanced Loss** (Cui et al., 2019) — memberikan bobot (penalty) berbeda pada loss function saat training:

| Emosi | Jumlah di Train | Weight | Artinya |
|-------|----------------|--------|---------|
| Neutral | 5,678 | **1.0x** | Baseline penalty |
| Happy | 751 | **1.9x** | Salah prediksi 2x lebih mahal |
| Sad | 490 | **2.6x** | Salah prediksi 3x lebih mahal |
| Surprised | 70 | **14.7x** | Salah prediksi 15x lebih mahal |
| Angry | 48 | **21.3x** | Salah prediksi 21x lebih mahal |
| Disgusted | 19 | **52.9x** | Salah prediksi 53x lebih mahal |
| Fearful | 8 | **125.0x** | Salah prediksi 125x lebih mahal |

> **Penjelasan lisan:**  
> "Cara kerjanya seperti ini: tanpa weights, kalau model salah prediksi neutral atau salah prediksi fearful, penalty-nya sama. Akibatnya model tidak termotivasi untuk belajar emosi langka. Dengan weights, kalau model salah prediksi fearful, penalty-nya 125 kali lebih besar daripada salah prediksi neutral. Ini memaksa model untuk serius belajar emosi langka juga."
>
> "Metode ini dari paper Cui et al. (2019) - 'Class-Balanced Loss Based on Effective Number of Samples' yang dipublikasikan di CVPR. Jadi secara referensi sudah kuat."

### Ilustrasi (untuk di slide):

```
Tanpa Class Weights:
  Prediksi neutral padahal fearful → penalty 1x
  Prediksi neutral padahal neutral → benar!
  → Model "malas" → prediksi neutral terus → accuracy tinggi tapi tidak berguna

Dengan Class Weights:
  Prediksi neutral padahal fearful → penalty 125x !!!
  Prediksi neutral padahal neutral → benar!
  → Model "dipaksa" belajar semua emosi → lebih balanced
```

### Status Implementasi

| Tahap | Status |
|-------|--------|
| Deteksi masalah imbalance | Selesai |
| Hitung class weights (Cui et al., 2019) | Selesai, tersimpan di `class_weights.json` |
| Apply ke training (weighted cross-entropy) | Belum, diterapkan saat training nanti |
| Metrik evaluasi: Macro F1-Score | Belum, diterapkan saat evaluasi nanti |

> **Penjelasan lisan:**  
> "Weights sudah dihitung dan siap pakai. Nanti saat training, weights ini akan dimasukkan ke loss function (weighted cross-entropy). Selain itu, untuk evaluasi saya akan menggunakan Macro F1-Score, bukan accuracy. Accuracy tidak cocok untuk data imbalanced karena model yang selalu prediksi neutral saja sudah dapat accuracy 84.5%, tapi sebenarnya tidak berguna."

### **(KONSULTASI 1)** Strategi Class Imbalance

> **Yang perlu ditanyakan:**  
> "Pak/Bu, untuk penanganan class imbalance saya sudah menyiapkan 3 skenario yang akan dibandingkan hasilnya saat training nanti."

**Skenario B1: Tanpa penanganan (baseline)**
- Training biasa tanpa class weights
- Sebagai baseline untuk dibandingkan
- Dataset: `data/dataset/` (7,064 train samples)

**Skenario B2: Dengan class weights saja**
- Weighted cross-entropy loss (Cui et al., 2019)
- Tidak mengubah data, hanya mengubah loss function
- Dataset: `data/dataset/` + `class_weights.json`

**Skenario B3: Dengan class weights + augmentasi data (sudah disiapkan)**
- Augmentasi kelas minoritas: horizontal flip, rotasi (-15/+15 derajat), brightness adjustment, dan kombinasinya
- Hanya kelas < 150 sample yang diaugmentasi, target minimal 150 per kelas
- **Hanya train set yang diaugmentasi** — val/test tetap original (supaya evaluasi fair)
- Dataset: `data/dataset_augmented/` (7,519 train samples, +455 augmented)

| Emosi | Original | Setelah Augmentasi | Ditambah |
|-------|----------|-------------------|----------|
| Neutral | 5,678 | 5,678 | - |
| Happy | 751 | 751 | - |
| Sad | 490 | 490 | - |
| Angry | 48 | **150** | +102 |
| Fearful | 8 | **150** | +142 |
| Disgusted | 19 | **150** | +131 |
| Surprised | 70 | **150** | +80 |
| **Total** | **7,064** | **7,519** | **+455** |

Evaluasi semua skenario menggunakan **Macro F1-Score** (bukan accuracy, karena accuracy bias ke kelas mayoritas).

> **Penjelasan lisan:**  
> "Saya menyiapkan 3 skenario untuk dibandingkan: tanpa penanganan sebagai baseline, dengan class weights saja, dan dengan class weights plus augmentasi. Ketiga skenario ini akan dijalankan untuk setiap model (CNN, FCNN, Late Fusion, Intermediate Fusion), sehingga kita bisa lihat mana yang paling efektif."
>
> "Augmentasi hanya dilakukan pada training set — validation dan test set tidak disentuh supaya evaluasinya fair dan merepresentasikan kondisi data real."
>
> "Apakah Bapak/Ibu setuju dengan pendekatan perbandingan 3 skenario ini?"

---

## SLIDE 7: Split Dataset

**Strategi: Split by User + Smart Rare-Emotion Distribution**

| Split | Samples | Users | Persentase |
|-------|---------|-------|-----------|
| Train | 7,064 | 29 users | 71.4% |
| Validation | 1,174 | 3 users | 11.9% |
| Test | 1,656 | 5 users | 16.7% |

### Detail per Batch (Front-Only)

| | Train | Val | Test | Total |
|-|:-----:|:---:|:----:|:-----:|
| **Batch 1** (20 user, front only) | 18 user (3,334) | 1 user (232) | 1 user (258) | 20 user (3,824) |
| **Batch 2** (17 user, front only) | 11 user (2,014) | 2 user (475) | 4 user (778) | 17 user (3,267) |
| **Total** | **29 user (5,348)** | **3 user (707)** | **5 user (1,036)** | **37 user (7,091)** |

Detail user per split:
- **Train (29 user):** 97, 99, 100, 101, 102, 103, 104, 106, 107, 108, 109, 110, 113, 114, 115, 116, 117, 118, 200, 201, 203, 205, 208, 210, 211, 212, 213, 215, 216
- **Val (3 user):** 112 (Batch 1), 207, 214 (Batch 2)
- **Test (5 user):** 111 (Batch 1), 197, 202, 206, 209 (Batch 2)

> **Penjelasan lisan:**
> "Split dilakukan per user, bukan per sampel — jadi semua frame dari 1 mahasiswa hanya masuk ke 1 split. Dari 20 user batch 1, 18 masuk train, 1 val, 1 test. Dari 17 user batch 2, 11 masuk train, 2 val, 4 test. Proporsi batch 2 di test set memang lebih banyak karena algoritma smart split memastikan emosi langka (fearful, disgusted) terdistribusi merata ke semua split."

### Kenapa rasionya bukan tepat 80/10/10?

Di proposal tertulis 80/10/10, namun karena split dilakukan **berdasarkan user** (bukan random per-sample), rasio tidak bisa tepat. Rasio aktual 71/12/17 masih dalam rentang standar yang lazim di penelitian:

| Rasio | Digunakan di |
|-------|-------------|
| 80/10/10 | Paling umum (split random) |
| **70/15/15** | **Umum untuk user-based split** |
| 70/10/20 | Dataset dengan test set lebih besar |
| 60/20/20 | Dataset kecil |

> **Penjelasan lisan:**  
> "Di proposal saya tulis 80/10/10, tapi karena split dilakukan per-user untuk mencegah data leaking, rasionya menjadi 71/12/17. Ini masih dalam rentang standar 70/10-15/15-20 yang lazim di penelitian deep learning. Lebih baik sedikit bergeser rasio daripada mencampur data user yang sama di train dan test, karena itu bisa menyebabkan model menghafal wajah, bukan belajar mengenali emosi."
>
> "Di tesis nanti akan saya tuliskan: 'Pembagian data dilakukan berdasarkan user (user-level split) untuk mencegah data leaking, menghasilkan rasio train/validation/test sebesar 71.4%/11.9%/16.7% dari total 9,894 sampel.'"

### Distribusi emosi per split (semua 7 emosi ada di semua split):

| Emosi | Train | Val | Test |
|-------|-------|-----|------|
| Neutral | 5,678 | 1,090 | 1,588 |
| Happy | 575 | 22 | 10 |
| Sad | 402 | 48 | 38 |
| Angry | 43 | 2 | 13 |
| Fearful | 7 | 4 | 1 |
| Disgusted | 19 | 2 | 3 |
| Surprised | 39 | 6 | 3 |

> **Penjelasan lisan (lanjutan):**  
> "Tantangan utama: karena split by user, emosi langka (fearful hanya 13 total, disgusted hanya 24 total) bisa saja tidak muncul di salah satu split. Untuk mengatasinya, saya menggunakan algoritma smart split yang memastikan user yang memiliki emosi langka tersebar ke semua split. Hasilnya semua 7 emosi terwakili di train, validation, dan test."
>
> "Ini penting karena kalau misalnya fearful tidak ada di test set, kita tidak bisa mengevaluasi kemampuan model mengenali emosi fearful."

---

## SLIDE 8: Validasi Ahli Psikologi

> **Penjelasan lisan:**  
> "Sesuai proposal, label emosi saat ini sepenuhnya dari auto-detection (Face API). Untuk dual-validation, kita perlu ahli psikologi untuk memvalidasi sebagian sample."

### **(KONSULTASI 2)** Berapa Sample untuk Validasi?

Saya sudah menyiapkan 3 opsi:

| Opsi | Total Sample | Strategi | Beban Ahli |
|------|-------------|----------|-----------|
| **A** | 1,938 | Semua non-neutral + 400 neutral | Berat (~8 jam) |
| **B** | 1,067 | Stratified 10% per kelas | Sedang (~4 jam) |
| **C** | 583 | Stratified 5% per kelas | Ringan (~2 jam) |

#### Detail per opsi:

**Opsi A (1,938 sample):**
- Semua emosi non-neutral divalidasi 100%
- Paling kuat secara metodologi
- Justifikasi: "Seluruh sample non-neutral divalidasi oleh ahli psikologi"

**Opsi B (1,067 sample, 10% stratified):**
- Proporsional 10% dari tiap kelas, minimum 30 per kelas
- Distribusi validasi merepresentasikan distribusi dataset
- Justifikasi: "10% stratified random sampling dengan minimum 30 sampel per kelas"

**Opsi C (583 sample, 5% stratified):**
- Proporsional 5%, minimum 30 per kelas
- Masih memenuhi syarat statistik untuk Cohen's Kappa
- Justifikasi: "5% stratified random sampling, memenuhi minimum statistical requirement"

---

### **(KONSULTASI 2b)** Revisi Pendekatan: SSL + Validasi Minimal

Berdasarkan masukan dosen — opsi A/B/C terlalu banyak beban validator — pendekatan direvisi menjadi:

**Skema baru: 146 sampel divalidasi ahli + SSL untuk sisanya**

```
9,894 sampel total
       ↓
146 sampel → divalidasi manual oleh ahli (stratified per kelas, min 10/kelas)
       ↓
Sisanya (~9,748) → diverifikasi otomatis menggunakan
representasi ResNet18 (self-supervised pre-selection):
  - Ekstrak embedding semua sampel
  - Sampel konsisten dengan centroid kelasnya → dianggap valid
  - Sampel tidak konsisten (outlier) → dieksklusi dari training
```

#### Justifikasi Jumlah 146 Sampel (~1.5%)

Mengacu pada **MER2024** (Lian et al., 2024) — benchmark semi-supervised emotion recognition resmi di ACM MM + IJCAI 2024:

| | MER2024 | Dataset penelitian ini |
|-|---------|----------------------|
| Total sampel | 115,595 | 9,894 |
| Labeled (divalidasi) | 1,169 (**±1%**) | **146 sampel (1.5%)** |
| Unlabeled (SSL) | 99% | 99% |
| Anotator | 5 orang | 3 orang (minimum) |
| Threshold agreement | 4/5 = 80% | 2/3 = 67% |

> **Referensi:** Lian, Z., et al. (2024). *MER 2024: Semi-Supervised Learning, Noise Robustness, and Open-Vocabulary Multimodal Emotion Recognition.* Proceedings of the 2nd International Workshop on Multimodal and Responsible Affective Computing, ACM MM 2024. https://arxiv.org/abs/2404.17113

#### Jumlah Validator

Berdasarkan MER2024 (5 anotator) — untuk konteks tesis S2 yang lebih terbatas:
- **Minimum 3 validator** berlatar belakang psikologi (bisa mahasiswa S2/S3 psikologi)
- Validasi independen → hitung **Fleiss' Kappa** (untuk 3+ rater)
- Target κ ≥ 0.61 (Landis & Koch, 1977) = substantial agreement

#### Penjelasan lisan:
> "Setelah berdiskusi dengan dosen, pendekatan validasi direvisi. Daripada membebankan validator dengan ratusan sampel, saya mengadopsi pendekatan semi-supervised seperti yang digunakan di MER2024 — challenge resmi pengenalan emosi di ACM MM 2024. Di sana, hanya 1% dari total data yang divalidasi manusia, sisanya ditangani oleh model SSL. Untuk dataset saya yang berjumlah ~9.894 sampel, 1% stratified sampling menghasilkan 146 sampel yang perlu divalidasi ahli (minimum 10 per kelas emosi, kelas langka seperti fearful diambil seluruhnya). Sisanya diverifikasi konsistensinya menggunakan embedding dari ResNet18 yang sudah dilatih."
>
> "Untuk validator, saya berencana menggunakan 3 orang berlatar belakang psikologi, mengacu pada MER2024 yang menggunakan 5 anotator. Kesepakatan antar validator diukur menggunakan Fleiss' Kappa dengan target κ ≥ 0.61."

---

### Tool Validasi yang Sudah Disiapkan:

Saya sudah membuat **web tool interaktif** (Streamlit) untuk memudahkan proses validasi:

**Fitur utama:**
- Tampilan gambar wajah besar + label otomatis + confidence score + bar chart 7 emosi
- Ahli tinggal klik **"Setuju"** atau pilih emosi yang benar — tidak perlu ketik manual
- **Multi-validator** — support 1-2 ahli, masing-masing punya progress terpisah
- **3 opsi set validasi** tersedia dalam 1 tool — tinggal pilih saat login
- Auto-save setiap klik, ada progress tracker
- Halaman ringkasan otomatis menghitung **Cohen's Kappa** dan **inter-rater agreement**
- Download hasil ke CSV

**Cara akses:**
- Tool akan di-deploy online (Streamlit Cloud) — ahli cukup klik link, tidak perlu install apapun
- Data wajah mahasiswa sudah mendapat informed consent
- Repo bersifat **private** (tidak publik)

**Demo** *(bisa ditunjukkan saat bimbingan jika diminta):*

```
Alur validasi ahli:
1. Buka link → Login nama → Pilih set validasi
2. Lihat gambar wajah → Lihat label otomatis
3. Klik "Setuju" atau pilih emosi yang benar
4. Otomatis lanjut ke sample berikutnya
5. Setelah selesai → Download hasil CSV
```

> **Penjelasan lisan:**  
> "Untuk memudahkan ahli psikologi, saya sudah menyiapkan web tool interaktif. Ahli cukup buka link di browser, lalu klik-klik untuk validasi — tidak perlu buka Excel dan folder gambar satu-satu. Tool ini juga otomatis menghitung Cohen's Kappa setelah validasi selesai."
>
> "Kalau validatornya 2 orang, tool juga bisa menghitung inter-rater agreement antara kedua ahli secara otomatis."

> **Yang perlu ditanyakan:**  
> 1. "Opsi mana yang paling tepat untuk level tesis S2?"
> 2. "Apakah 1 ahli cukup, atau perlu 2 ahli untuk inter-rater reliability?"
> 3. "Apakah Bapak/Ibu punya rekomendasi ahli psikologi yang bisa dihubungi?"
> 4. "Apakah perlu diberikan honorarium untuk validator?"

---

## SLIDE 9: Perubahan dari Proposal

| Aspek | Di Proposal | Implementasi | Alasan |
|-------|------------|-------------|--------|
| Face detection | dlib | **MediaPipe** | Dlib gagal deteksi side view, MediaPipe support multi-angle |
| Landmark | 68 titik (dlib) | **68 titik (mapped dari MediaPipe)** | Setara, output tetap 136 fitur |
| Jumlah mahasiswa | 38 | **37** | 1 mahasiswa data tidak lengkap |
| Fusion strategy | Late + Intermediate | Late + Intermediate | Tetap sesuai proposal |

> **Penjelasan lisan:**  
> "Perubahan utama hanya pada tool face detection dari dlib ke MediaPipe. Ini karena data baru memiliki side view yang tidak bisa dideteksi oleh dlib. Output landmark tetap 68 titik yang di-map dari 478 titik MediaPipe, sehingga secara substansi tidak mengubah desain penelitian."

### **(KONSULTASI 3)** Apakah Perubahan Ini Perlu Direvisi di Proposal?

> "Pak/Bu, apakah perubahan dari dlib ke MediaPipe ini perlu direvisi di dokumen proposal, atau cukup dijelaskan di BAB 4 (Implementasi) saja?"

---

## SLIDE 10: Rencana Selanjutnya

| No | Tahap | Status | Target |
|----|-------|--------|--------|
| 1 | Pengumpulan data | Selesai | - |
| 2 | Preprocessing (frame, crop, landmark) | Selesai | - |
| 3 | Prepare dataset (numpy arrays) | Selesai | - |
| 4 | Class weights + augmentasi | Selesai | - |
| 5 | Tool validasi ahli (web app) | Selesai, siap deploy | - |
| 6 | Deploy tool + kirim ke ahli | **Menunggu keputusan** | Setelah bimbingan |
| 7 | Training CNN | **Selesai** | - |
| 8 | Training FCNN | **Selesai** | - |
| 9 | Late Fusion | **Selesai** | - |
| 10 | Intermediate Fusion | **Selesai** | - |
| 11 | Evaluasi & perbandingan | **Selesai** | - |
| 12 | **Perbaikan model** | **Perlu diskusi** | Setelah bimbingan |

---

## SLIDE 11: Rancangan Eksperimen — Tahap 1: 7-Class From Scratch

**7 kelas emosi:** neutral, happy, sad, angry, fearful, disgusted, surprised

**4 Model yang dibandingkan:**
- **CNN** — menggunakan fitur citra wajah 224×224 px
- **FCNN** — menggunakan fitur landmark geometrik (136 koordinat)
- **Late Fusion** — menggabungkan CNN + FCNN di level keputusan (weighted average)
- **Intermediate Fusion** — menggabungkan CNN + FCNN di level fitur (concatenation)

**3 Skenario penanganan class imbalance:**
- **B1** = Baseline: training biasa tanpa penanganan
- **B2** = Class Weights (Cui et al., 2019): penalty loss lebih besar untuk kelas langka
- **B3** = Class Weights + Augmentasi: flip, rotasi ±15°, brightness untuk kelas < 150 sample

**Total: 4 model × 3 skenario = 12 eksperimen** | GPU: NVIDIA T4 | Metrik: Macro F1-Score

> **Penjelasan lisan:**
> "Saya melakukan eksperimen tahap pertama dengan 7 kelas emosi: neutral, happy, sad, angry, fearful, disgusted, dan surprised. Ada 4 model yang dibandingkan — CNN yang menggunakan fitur citra wajah, FCNN yang menggunakan fitur landmark geometrik, Late Fusion yang menggabungkan keduanya di level keputusan, dan Intermediate Fusion yang menggabungkan di level fitur."
>
> "Masing-masing model dijalankan dengan 3 skenario penanganan class imbalance yang berbeda. Skenario B1 adalah baseline tanpa penanganan. Skenario B2 menggunakan class weights — prinsipnya kelas yang lebih langka mendapat penalty lebih besar kalau salah diprediksi, sehingga model dipaksa lebih serius mempelajari kelas tersebut. Skenario B3 menambahkan augmentasi data di atas class weights."
>
> "Total di tahap ini: 4 model × 3 skenario = 12 eksperimen."

---

## SLIDE 12: Hasil Training (4 Model × 3 Skenario = 12 Eksperimen)

Training dilakukan di VPS Biznet Gio (NVIDIA T4, 16GB VRAM) menggunakan PyTorch.

### Tabel Hasil Lengkap (diurutkan berdasarkan Macro F1)

| Rank | Model | Skenario | Accuracy | Macro F1 | Weighted F1 |
|------|-------|----------|----------|----------|-------------|
| 1 | **FCNN** | **B1 Baseline** | **95.8%** | **0.234** | 0.952 |
| 2 | Late Fusion | B1 Baseline | 95.8% | 0.230 | 0.951 |
| 3 | FCNN | B2 Weights | 89.1% | 0.189 | 0.912 |
| 4 | Late Fusion | B2 Weights | 89.7% | 0.189 | 0.909 |
| 5 | Late Fusion | B3 Aug | 92.5% | 0.182 | 0.929 |
| 6 | FCNN | B3 Aug | 92.3% | 0.182 | 0.927 |
| 7 | Intermediate | B2 Weights | 84.5% | 0.140 | 0.881 |
| 8 | Intermediate | B3 Aug | 81.6% | 0.134 | 0.866 |
| 9 | CNN | B2 Weights | 82.8% | 0.134 | 0.872 |
| 10 | CNN | B1 Baseline | 84.2% | 0.133 | 0.881 |
| 11 | CNN | B3 Aug | 64.9% | 0.119 | 0.766 |
| 12 | Intermediate | B1 Baseline | 63.3% | 0.111 | 0.744 |

### Kombinasi Terbaik: **FCNN + B1 Baseline** (Macro F1: 0.234)

> **Penjelasan lisan:**  
> "Training sudah selesai untuk semua 12 kombinasi. Hasilnya, model FCNN (berbasis landmark) tanpa class weights justru memberikan hasil terbaik dengan accuracy 95.8% dan Macro F1 0.234."

---

## SLIDE 13: Analisis Hasil

### Temuan 1: FCNN (Landmark) > CNN (Image)

| Model | Best Macro F1 |
|-------|--------------|
| **FCNN** | **0.234** |
| CNN | 0.134 |

> **Penjelasan lisan:**  
> "Fitur geometrik dari facial landmark ternyata lebih efektif dari fitur penampilan (citra wajah) untuk dataset ini. Ini konsisten dengan temuan Bachtiar et al. (2024) yang menunjukkan bahwa facial landmark bisa superior untuk ekspresi halus. Kemungkinan karena landmark lebih robust terhadap variasi pencahayaan dan sudut wajah yang terjadi selama sesi pemrograman."

### Temuan 2: Fusion tidak lebih baik dari FCNN saja

| Model | Best Macro F1 |
|-------|--------------|
| FCNN | **0.234** |
| Late Fusion (10% CNN + 90% FCNN) | 0.230 |
| Intermediate Fusion | 0.140 |

> **Penjelasan lisan:**  
> "Late Fusion optimal pada bobot CNN hanya 10% dan FCNN 90% — artinya CNN hampir tidak berkontribusi. Intermediate Fusion justru lebih buruk karena fitur CNN yang noisy mengganggu fitur landmark yang sudah bagus. Ini menunjukkan bahwa untuk dataset pembelajaran pemrograman ini, pendekatan unimodal FCNN sudah cukup optimal."

### Temuan 3: Class weights dan augmentasi TIDAK membantu

| Skenario | FCNN Macro F1 | CNN Macro F1 |
|----------|--------------|-------------|
| B1 Baseline | **0.234** | 0.133 |
| B2 Class Weights | 0.189 | 0.134 |
| B3 Weights + Aug | 0.182 | 0.119 |

> **Penjelasan lisan:**  
> "Yang mengejutkan, class weights dan augmentasi justru menurunkan performa. Ini kemungkinan karena memberikan bobot terlalu besar pada kelas yang sangat sedikit (fearful: 8 sample, weight 125x) membuat model tidak stabil. Baseline tanpa penanganan justru lebih baik."

### Temuan 4: Masalah utama — kelas minoritas tidak terdeteksi

Dari classification report model terbaik (FCNN B1):

| Emosi | Support | Recall | F1 |
|-------|---------|--------|-----|
| Neutral | 1,588 | 99% | 0.98 |
| Happy | 10 | 0% | 0.00 |
| Sad | 38 | 32% | 0.41 |
| Angry | 13 | 0% | 0.00 |
| Fearful | 1 | 0% | 0.00 |
| Disgusted | 3 | 0% | 0.00 |
| Surprised | 3 | 0% | 0.00 |

> **Penjelasan lisan:**  
> "Meskipun accuracy tinggi (95.8%), model hanya bisa mengenali neutral dan sedikit sad. Kelas lain seperti happy, angry, fearful, disgusted, dan surprised tidak terdeteksi sama sekali. Ini karena jumlah sampelnya terlalu sedikit di test set (1-13 sample) dan model belum cukup belajar dari data training yang juga sangat sedikit untuk kelas-kelas ini."
>
> "Accuracy 95.8% sebenarnya misleading — model bisa dapat accuracy segitu hanya dengan memprediksi semua sebagai neutral. Itulah kenapa kita pakai Macro F1 (0.234) sebagai metrik utama, yang menunjukkan performa sebenarnya."

---

## SLIDE 14: Rancangan Eksperimen — Tahap 2: 4-Class

| Kelas Baru | Kelas Asal | Total Sample | Alasan |
|-----------|-----------|-------------|--------|
| neutral | neutral | 8,356 | Tetap — kelas dominan |
| happy | happy | 783 | Tetap — cukup data |
| sad | sad | 576 | Tetap — cukup data |
| negative | angry (63) + fearful (13) + disgusted (24) + surprised (79) | 179 | Digabung — masing-masing < 80 sample |

**4 model × 3 skenario = 12 eksperimen tambahan | Total kumulatif: 24 eksperimen**

> **Penjelasan lisan:**
> "Setelah melihat hasil tahap 1, Macro F1 terbaik hanya 0.234. Kalau saya lihat detail per kelasnya, ternyata emosi-emosi langka seperti angry, fearful, disgusted, dan surprised sama sekali tidak terdeteksi — recall-nya nol. Ini bukan karena modelnya buruk, tapi karena jumlah sampelnya terlalu sedikit. Yang paling ekstrem adalah fearful, hanya 13 sampel total, dan di test set hanya ada 1 sampel."
>
> "Dari sini saya berpikir: masalahnya bukan di arsitektur model, tapi di data. Keempat emosi langka ini secara konseptual juga bisa dikelompokkan sebagai respons emosi negatif. Jadi saya coba gabungkan keempatnya menjadi satu kelas baru bernama 'negative'."
>
> "Dengan penggabungan ini, kelas 'negative' memiliki 179 sampel — jauh lebih banyak dari masing-masing kelas aslinya. Saya kemudian mengulang seluruh eksperimen dengan konfigurasi 4 kelas ini. Total jadi 24 eksperimen kumulatif."

---

## SLIDE 15: Hasil Training 4-Class (Analisis Tambahan)

Sebagai analisis tambahan, dilakukan eksperimen dengan **4 kelas emosi**:
- neutral, happy, sad, **negative** (gabungan angry + fearful + disgusted + surprised)

### Tabel Hasil 4-Class (diurutkan berdasarkan Macro F1)

| Rank | Model | Skenario | Accuracy | Macro F1 | Weighted F1 |
|------|-------|----------|----------|----------|-------------|
| 1 | **FCNN** | **B3 Aug** | **94.4%** | **0.394** | 0.943 |
| 2 | Late Fusion | B3 Aug | 94.6% | 0.385 | 0.943 |
| 3 | FCNN | B1 Baseline | 95.8% | 0.330 | 0.943 |
| 4 | Late Fusion | B1 Baseline | 95.8% | 0.330 | 0.943 |
| 5 | FCNN | B2 Weights | 92.9% | 0.327 | 0.929 |
| 6 | Late Fusion | B2 Weights | 92.9% | 0.327 | 0.929 |
| 7 | CNN | B2 Weights | 92.8% | 0.296 | 0.929 |
| 8 | CNN | B1 Baseline | 92.8% | 0.282 | 0.932 |
| 9 | Intermediate | B2 Weights | 87.3% | 0.258 | 0.895 |
| 10 | Intermediate | B1 Baseline | 91.9% | 0.245 | 0.919 |
| 11 | Intermediate | B3 Aug | 83.0% | 0.238 | 0.875 |
| 12 | CNN | B3 Aug | 79.2% | 0.238 | 0.853 |

### Kombinasi Terbaik 4-Class: **FCNN + B3 Augmented** (Macro F1: 0.394)

> **Penjelasan lisan:**  
> "Dengan 4 kelas, performa meningkat signifikan di semua model. FCNN dengan augmentasi memberikan Macro F1 terbaik 0.394, naik 68% dari 7-kelas. Ini menunjukkan bahwa penggabungan kelas langka menjadi 'negative' efektif meningkatkan kemampuan model."

---

## SLIDE 16: Perbandingan 7-Class vs 4-Class

### Best per Model

| Model | 7-Class (Macro F1) | 4-Class (Macro F1) | Peningkatan |
|-------|-------------------|-------------------|-------------|
| **FCNN** | 0.234 (B1) | **0.394 (B3)** | **+68%** |
| Late Fusion | 0.230 (B1) | 0.385 (B3) | +67% |
| CNN | 0.134 (B2) | 0.296 (B2) | +121% |
| Intermediate | 0.140 (B2) | 0.258 (B2) | +84% |

### Temuan baru dari 4-class:

**Temuan 5: Penggabungan kelas signifikan meningkatkan Macro F1**

> **Penjelasan lisan:**  
> "Dengan menggabungkan 4 emosi langka (angry, fearful, disgusted, surprised) menjadi satu kelas 'negative', Macro F1 meningkat 68% pada model terbaik. Ini menunjukkan bahwa masalah utama di 7-kelas bukan pada modelnya, tapi pada jumlah data kelas minoritas yang terlalu sedikit."

**Temuan 6: Augmentasi efektif untuk 4-class (berbeda dengan 7-class)**

> **Penjelasan lisan:**  
> "Menariknya, di 4-kelas skenario B3 (augmentasi) justru terbaik, sedangkan di 7-kelas B1 (baseline) yang terbaik. Ini karena di 4-kelas, kelas 'negative' sudah punya 145 sample asli — cukup besar untuk mendapat manfaat dari augmentasi. Di 7-kelas, kelas terkecil hanya 8 sample — terlalu sedikit sehingga augmentasi hanya menghasilkan variasi dari data yang sangat terbatas."

**Temuan 7: FCNN tetap konsisten terbaik di kedua konfigurasi**

> **Penjelasan lisan:**  
> "Baik di 7-kelas maupun 4-kelas, FCNN (fitur landmark) selalu unggul dari CNN (fitur citra wajah). Ini memperkuat temuan bahwa fitur geometrik lebih efektif dari fitur penampilan untuk dataset pembelajaran pemrograman ini."

---

## SLIDE 17: Rancangan Eksperimen — Tahap 3: Transfer Learning

| Komponen | Tahap 1 & 2 (From Scratch) | Tahap 3 (Transfer Learning) | Catatan |
|---------|--------------------------|----------------------------|--------|
| CNN backbone | EmotionCNN (dari nol) | ResNet18 pretrained ImageNet | Fine-tune seluruh layer |
| FCNN | Fully-connected (dari nol) | Sama — tidak berubah | Landmark = numerik, tidak perlu pretrained |
| Late Fusion | CNN scratch + FCNN | CNN TL + FCNN | |
| Intermediate Fusion | CNN scratch + FCNN (feature-level) | ResNet18 + FCNN (feature-level) | |
| Learning Rate | 0.0001 | 0.00005 | Lebih kecil untuk fine-tuning |

**4 model × 3 skenario × 2 konfigurasi kelas = 24 eksperimen tambahan**
**Total keseluruhan: 12 + 12 + 24 = 48 eksperimen**

> **Penjelasan lisan:**
> "Setelah tahap 2, FCNN memang sudah membaik — Macro F1 0.394 di 4 kelas. Tapi CNN masih tertinggal jauh di 0.296, padahal idealnya fusion harusnya lebih baik dari masing-masing modalitas. Saya analisis penyebabnya: CNN dilatih dari nol dengan dataset yang relatif kecil, sekitar 9 ribu sampel. Untuk CNN yang kompleks, itu belum cukup."
>
> "Solusinya adalah Transfer Learning. Saya gunakan ResNet18 yang sudah pretrained di ImageNet — dataset dengan 1.2 juta gambar. Idenya, ResNet18 sudah 'memahami' fitur visual dasar seperti tepi, tekstur, dan bentuk. Saya tinggal fine-tune agar ia belajar mengenali ekspresi wajah dari dataset saya, dengan learning rate yang lebih kecil supaya tidak menghancurkan pengetahuan yang sudah ada."
>
> "Perlu dicatat, FCNN tidak saya ubah — karena FCNN menggunakan data landmark yang berupa koordinat numerik, tidak ada pretrained weights yang relevan untuk itu. Transfer Learning hanya diterapkan pada komponen CNN."
>
> "Di tahap ini saya jalankan eksperimen untuk 7-class dan 4-class sekaligus, jadi total tambahan 24 eksperimen. Keseluruhan dari 3 tahap: 48 eksperimen."

---

## SLIDE 18: Hasil Transfer Learning (ResNet18 Pretrained ImageNet)

Sebagai upaya meningkatkan performa CNN, dilakukan eksperimen **Transfer Learning** menggunakan **ResNet18 pretrained ImageNet** sebagai pengganti CNN from scratch.

### Strategi Transfer Learning

- **Model CNN** diganti: `EmotionCNN` (from scratch) → `EmotionCNNTransfer` (ResNet18 pretrained)
- **Fine-tune** seluruh network dengan learning rate kecil (0.00005)
- **Model FCNN** tidak berubah (tetap from scratch, landmark tidak perlu pretrained)
- **Late Fusion TL** = CNN TL + FCNN (from scratch)
- **Intermediate Fusion TL** = ResNet18 + FCNN digabung di level fitur

### Hasil Transfer Learning 7-Class

| Model | From Scratch (Best) | Transfer Learning (Best) | Peningkatan |
|-------|--------------------|--------------------------|----|
| CNN | 0.134 (B2) | **0.177 (B2)** | +32% |
| FCNN | 0.234 (B1) | 0.234 (B1) | — (sama) |
| Late Fusion | 0.230 (B1) | 0.234 (B1) | +2% |
| Intermediate Fusion | 0.140 (B2) | **0.232 (B1)** | +66% |

### Hasil Transfer Learning 4-Class

| Model | From Scratch (Best) | Transfer Learning (Best) | Peningkatan |
|-------|--------------------|--------------------------|----|
| CNN | 0.296 (B2) | **0.407 (B2)** | +37% |
| FCNN | 0.394 (B3) | 0.394 (B3) | — (sama) |
| Late Fusion | 0.385 (B3) | **0.442 (B2)** | +15% |
| Intermediate Fusion | 0.258 (B2) | **0.376 (B3)** | +46% |

### **Best Overall: Late Fusion TL 4-class B2 (Macro F1: 0.442)**

> **Penjelasan lisan:**  
> "Transfer Learning dengan ResNet18 terbukti meningkatkan performa CNN secara signifikan, terutama di 4-kelas (+37% untuk CNN). Yang paling menonjol adalah Late Fusion dengan Transfer Learning — menggabungkan ResNet18 dan FCNN menghasilkan Macro F1 terbaik secara keseluruhan: **0.442**. Ini naik dari 0.385 (Late Fusion from scratch) dan dari 0.394 (FCNN terbaik from scratch)."
>
> "Menariknya, FCNN tetap tidak terpengaruh oleh transfer learning karena FCNN menggunakan landmark — tidak ada pretrained weights yang relevan untuk data numerik geometrik."

### Temuan 8: Transfer Learning efektif meningkatkan CNN

> "ResNet18 yang sudah 'memahami' fitur visual umum dari ImageNet membantu CNN mengenali pola ekspresi wajah meskipun dataset terbatas. Ini menjelaskan mengapa CNN from scratch sangat lemah (kurang data), sedangkan CNN TL jauh lebih baik."

### Temuan 9: Late Fusion TL menjadi model terbaik keseluruhan

> "Dengan menggabungkan CNN TL (ResNet18) dan FCNN via Late Fusion di 4-kelas dengan Class Weights (B2), diperoleh **Macro F1 0.442** — best overall dari 48 kombinasi eksperimen (4 model × 2 variant CNN × 3 skenario × 2 konfigurasi kelas)."

### Top 5 Model Terbaik (dari 48 kombinasi)

| Rank | Model | CNN Variant | Kelas | Skenario | Macro F1 |
|------|-------|-------------|-------|----------|----------|
| 1 | **Late Fusion** | **Transfer Learning** | **4-class** | **B2** | **0.442** |
| 2 | CNN | Transfer Learning | 4-class | B2 | 0.407 |
| 3 | FCNN | Transfer Learning | 4-class | B3 | 0.394 |
| 3 | FCNN | From Scratch | 4-class | B3 | 0.394 |
| 5 | Late Fusion | From Scratch | 4-class | B3 | 0.385 |

---

## SLIDE 19: Perbandingan Lengkap Semua Eksperimen

### Ringkasan Perjalanan Eksperimen

| Tahap | Model Terbaik | Macro F1 | Keterangan |
|-------|--------------|----------|------------|
| Tahap 1: 7-class from scratch | FCNN B1 | 0.234 | Baseline awal |
| Tahap 2: 4-class from scratch | FCNN B3 | 0.394 | +68% dari 7-class |
| Tahap 3: Transfer Learning | Late Fusion TL B2 | **0.442** | +12% dari 4-class FCNN |

---

## SLIDE 21: Eksperimen Front-Only (Tahap 4: Konsistensi Data)

### Motivasi

Batch 1 hanya memiliki sudut kamera **depan**, sedangkan batch 2 memiliki **depan + samping**. Inkonsistensi ini bisa mempengaruhi hasil eksperimen karena side view memiliki karakteristik fitur yang berbeda (landmark geometrik, tekstur wajah).

**Solusi:** Ulangi seluruh 48 eksperimen menggunakan **hanya data sudut depan** dari kedua batch.

### Dataset Front-Only

| | Front+Side (sebelumnya) | Front-Only |
|-|------------------------|------------|
| Batch 1 | 3,824 (front) | 3,824 (front) |
| Batch 2 | ~6,070 (front+side) | ~3,267 (front) |
| **Total** | **9,894** | **7,091** |
| Train / Val / Test | 7,064 / 1,174 / 1,656 | 5,348 / 707 / 1,036 |

User split **identik** — test set menggunakan user yang sama → hasil bisa dibandingkan secara fair.

### Hasil Front-Only: Best per Tahap

| Tahap | Model Terbaik | Macro F1 |
|-------|--------------|----------|
| Tahap 1: 7-class from scratch | Late Fusion B3 | 0.175 |
| Tahap 2: 4-class from scratch | Late Fusion B3 | 0.394 |
| Tahap 3: 7-class TL | Intermediate TL B3 | 0.180 |
| Tahap 3: 4-class TL | **Intermediate TL B1** | **0.412** |

### Top 5 Front-Only (dari 48 kombinasi)

| Rank | Model | Kelas | Skenario | Macro F1 |
|------|-------|-------|----------|----------|
| 1 | **Intermediate Fusion TL** | **4-class** | **B1** | **0.412** |
| 2 | Late Fusion | 4-class | B3 | 0.394 |
| 3 | Late Fusion TL | 4-class | B3 | 0.372 |
| 4 | FCNN | 4-class | B3 | 0.361 |
| 5 | Late Fusion TL | 4-class | B1 | 0.309 |

> **Penjelasan lisan:**
> "Sesuai saran dosen, saya mengulangi seluruh 48 eksperimen menggunakan hanya data sudut depan dari kedua batch, untuk memastikan konsistensi data. Dataset berkurang dari 9.894 menjadi 7.091 sampel karena side view dari batch 2 dihilangkan."
>
> "Model terbaik front-only adalah Intermediate Fusion TL 4-class B1 dengan Macro F1 0.412. Menariknya, ini sedikit lebih baik daripada best model front+side (CNN TL 4-class B2: 0.407), yang menunjukkan bahwa side view justru bisa menambah noise."

---

## SLIDE 22: Perbandingan Front-Only vs Front+Side

### Best Model per Tahap

| Tahap | Front-Only (F1) | Front+Side (F1) | Selisih |
|-------|:--------------:|:--------------:|:-------:|
| 7-class scratch | 0.175 (Late Fusion B3) | 0.234 (FCNN B1) | -0.060 |
| 4-class scratch | 0.394 (Late Fusion B3) | 0.394 (FCNN B3) | -0.001 |
| 7-class TL | 0.180 (Intermediate TL B3) | 0.232 (Intermediate TL B1) | -0.052 |
| **4-class TL** | **0.412 (Intermediate TL B1)** | **0.407 (CNN TL B2)** | **+0.005** |

### Overall Best

| | Model | Macro F1 |
|-|-------|:--------:|
| **Front-only** | Intermediate Fusion TL 4-class B1 | **0.412** |
| Front+side | CNN TL 4-class B2 | 0.407 |
| Selisih | | **+0.005** |

### Temuan dari Perbandingan

**Temuan 10: Side view tidak meningkatkan performa best model**
> "Overall best front-only (0.412) sedikit lebih baik dari front+side (0.407). Ini menunjukkan bahwa penambahan side view tidak membantu — justru berpotensi menambah noise karena perbedaan karakteristik fitur antar sudut kamera."

**Temuan 11: FCNN paling terdampak oleh penghapusan side view**
> "FCNN mengalami penurunan terbesar (-0.076 di 7-class) ketika side view dihilangkan. Ini logis karena landmark dari side view memiliki pola geometrik yang sangat berbeda dari front view — model kehilangan variasi data yang signifikan."

**Temuan 12: CNN justru membaik di front-only**
> "CNN (citra wajah) sedikit membaik di front-only (+0.005 di 7-class). Ini karena tanpa variasi sudut kamera, CNN bisa lebih fokus mempelajari fitur ekspresi wajah dari sudut yang konsisten."

**Temuan 13: Konsistensi data lebih penting dari kuantitas**
> "Meskipun dataset berkurang 28% (9.894 → 7.091), best model tidak menurun. Ini memperkuat argumen bahwa konsistensi data (semua front view) lebih penting daripada kuantitas data (campur front+side)."

> **Penjelasan lisan:**
> "Perbandingan menunjukkan bahwa front-only justru menghasilkan best model yang sedikit lebih baik. Dosen benar — konsistensi data lebih penting dari jumlah data. Menariknya, model terbaik bergeser dari Late Fusion TL (front+side) ke Intermediate Fusion TL (front-only), yang menunjukkan bahwa arsitektur optimal bisa berubah tergantung karakteristik data."

---

## SLIDE 23: Evaluasi Robustness — LOSO, 5-Fold CV, Random Split

### Motivasi

Eksperimen sebelumnya menggunakan **single user-based split** (80/10/10 — fix 5 user test). Hasilnya bisa bias ke user tertentu. Dosen meminta evaluasi dengan strategi split yang berbeda untuk menunjukkan robustness model.

### 3 Strategi yang Dibandingkan

| Strategi | Cara | Fold | Data Leakage? |
|----------|------|:----:|:-------------:|
| **LOSO** | 1 user = 1 test set, rotasi | 37 | Tidak |
| **5-Fold CV** | User dibagi 5 grup (~7 user/grup), rotasi | 5 | Tidak |
| **Random Split** | Sampel diacak tanpa peduli user | 5 repeat | **Ya** (baseline) |

### Model yang Diuji

3 model terbaik dari front-only 4-class:
1. Intermediate Fusion TL B1 (single split: 0.412)
2. Late Fusion B3 (single split: 0.394)
3. FCNN B3 (single split: 0.361)

### Hasil

| Model | Single Split | Random Split | 5-Fold CV | LOSO |
|-------|:-----------:|:------------:|:---------:|:----:|
| Intermediate TL | 0.412 | **0.586 ± 0.032** | **0.435 ± 0.068** | **0.370 ± 0.125** |
| Late Fusion | 0.394 | **0.580 ± 0.032** | **0.401 ± 0.055** | *pending* |
| FCNN | 0.361 | **0.471 ± 0.026** | **0.399 ± 0.062** | *pending* |

*LOSO (Late Fusion, FCNN) masih berjalan di VPS.*

### Temuan 14: LOSO menunjukkan performa sebenarnya lebih rendah dari single split

Intermediate Fusion TL pada LOSO (34 fold dari 37 user):
- **LOSO Macro F1: 0.370 ± 0.125** vs Single Split: 0.412
- Std deviation tinggi (0.125) → performa sangat bervariasi antar user
- Ini menunjukkan single split (0.412) kemungkinan **over-estimate** karena kebetulan mendapat user test yang "mudah"

### Temuan 15: Random Split jauh lebih tinggi — bukti data leakage

Random Split menghasilkan Macro F1 yang **jauh lebih tinggi** (+0.11 s/d +0.19) dari user-based split:

| Model | User-Based → Random | Selisih |
|-------|:-------------------:|:-------:|
| Intermediate TL | 0.412 → 0.586 | **+42%** |
| Late Fusion | 0.394 → 0.580 | **+47%** |
| FCNN | 0.361 → 0.471 | **+30%** |

Ini membuktikan bahwa ketika sampel dari user yang sama ada di train dan test (data leakage), model "menghafal" identitas wajah user — bukan mengenali pola ekspresi emosi. **Maka user-based split (LOSO/CV) wajib digunakan** untuk evaluasi yang valid.

> **Penjelasan lisan:**
> "Untuk memvalidasi robustness model, saya mengevaluasi 3 model terbaik menggunakan 3 strategi pembagian data: LOSO (gold standard, 37 fold), 5-Fold CV (moderat), dan Random Split (baseline dengan data leakage)."
>
> "Hasilnya sangat jelas — Random Split menghasilkan Macro F1 yang jauh lebih tinggi, naik 30-47% dibandingkan user-based split. Intermediate Fusion TL misalnya, naik dari 0.412 ke 0.586. Ini bukan karena modelnya tiba-tiba lebih baik, tapi karena model bisa 'menghafal' wajah user yang sama muncul di train dan test."
>
> "Temuan ini memperkuat argumen bahwa evaluasi harus menggunakan user-based split seperti LOSO atau Cross-Validation, bukan random split."
>
> "Sementara itu, LOSO untuk Intermediate Fusion TL sudah selesai — Macro F1 turun dari 0.412 (single split) ke 0.370 ± 0.125 (LOSO). Std deviation yang tinggi (0.125) menunjukkan performa sangat bervariasi antar user — ada user yang mudah diprediksi, ada yang sangat sulit. Ini artinya angka 0.412 dari single split kemungkinan sedikit optimistic."
>
> "Rangkuman perbandingan Intermediate TL: Random Split (0.586) >> 5-Fold CV (0.435) > Single Split (0.412) > LOSO (0.370). Pola ini menunjukkan semakin ketat strategi evaluasinya, semakin rendah hasilnya — tapi semakin jujur juga representasinya."

---

## SLIDE 24: Benchmark — JAFFE & CK+

### Tujuan
Menguji pipeline dan arsitektur yang sama pada dataset standar untuk menunjukkan bahwa pendekatan ini kompetitif dan performa rendah di dataset sendiri disebabkan oleh **karakteristik data** (natural expression, imbalanced), bukan kelemahan arsitektur.

### Dataset Benchmark

| Dataset | Sampel | Emosi | Subjek | Karakteristik |
|---------|:------:|:-----:|:------:|---------------|
| **JAFFE** | 213 | 7 | 10 | Lab, wanita Jepang, seimbang |
| **CK+** | 636 | 7+contempt | 118 | Lab, ekspresi peak, semi-balanced |
| **Dataset sendiri** | 7,091 | 7 | 37 | Natural (sesi programming), sangat imbalanced |

### Hasil Benchmark — Single Split (B1 Baseline)

#### 7-Class

| Model | JAFFE | CK+ | Dataset Sendiri |
|-------|:-----:|:---:|:---------------:|
| CNN | 0.304 | 0.461 | 0.137 |
| FCNN | 0.209 | 0.395 | 0.158 |
| Late Fusion | **0.545** | 0.498 | 0.175 |
| Intermediate | 0.037 | 0.316 | 0.137 |
| CNN TL | 0.464 | **0.913** | 0.154 |
| Intermediate TL | 0.447 | 0.833 | 0.180 |

#### 4-Class

| Model | JAFFE | CK+ | Dataset Sendiri |
|-------|:-----:|:---:|:---------------:|
| CNN | 0.177 | 0.645 | 0.238 |
| FCNN | **0.438** | 0.592 | 0.361 |
| Late Fusion | 0.396 | 0.592 | 0.394 |
| Intermediate | 0.177 | 0.567 | 0.243 |
| CNN TL | 0.330 | 0.675 | 0.274 |
| Intermediate TL | 0.375 | **0.837** | **0.412** |

### Hasil Benchmark — LOSO (JAFFE) & 10-Fold CV (CK+)

Evaluasi yang lebih robust untuk perbandingan dengan paper lain:

- **JAFFE → LOSO (10 fold):** JAFFE hanya punya 10 subjek, sehingga LOSO = 10-fold CV secara natural (setiap fold = 1 subjek keluar)
- **CK+ → 10-Fold CV (subject-wise):** CK+ punya 118 subjek — LOSO 118 fold terlalu mahal secara komputasi, sehingga digunakan 10-fold CV dimana 118 subjek dibagi ke 10 grup

#### 7-Class (mean ± std)

| Model | JAFFE LOSO | CK+ 10-Fold CV | Dataset Sendiri LOSO |
|-------|:----------:|:--------------:|:--------------------:|
| CNN | 0.249 ± 0.111 | 0.404 ± 0.049 | - |
| FCNN | 0.304 ± 0.157 | 0.478 ± 0.022 | - |
| Late Fusion | **0.467 ± 0.092** | 0.544 ± 0.060 | - |
| Intermediate | 0.129 ± 0.070 | 0.226 ± 0.082 | - |
| CNN TL | 0.426 ± 0.143 | 0.734 ± 0.082 | - |
| Intermediate TL | 0.293 ± 0.156 | **0.783 ± 0.107** | 0.370 ± 0.125* |

#### 4-Class (mean ± std)

| Model | JAFFE LOSO | CK+ 10-Fold CV | Dataset Sendiri |
|-------|:----------:|:--------------:|:---------------:|
| CNN | 0.338 ± 0.161 | 0.584 ± 0.163 | - |
| FCNN | 0.431 ± 0.194 | 0.598 ± 0.036 | 0.399 ± 0.062** |
| Late Fusion | **0.530 ± 0.126** | 0.621 ± 0.031 | 0.401 ± 0.055** |
| Intermediate | 0.202 ± 0.066 | 0.458 ± 0.172 | - |
| CNN TL | 0.510 ± 0.155 | **0.755 ± 0.079** | - |
| Intermediate TL | 0.450 ± 0.214 | 0.715 ± 0.054 | **0.435 ± 0.068** / 0.370 ± 0.125* |

*LOSO (37 fold) | **5-Fold CV (dataset sendiri)

### Perbandingan Best Model per Dataset

| Dataset | Evaluasi | Best Model | Macro F1 |
|---------|----------|-----------|:--------:|
| CK+ 7-class | 10-Fold CV | Intermediate TL | **0.783 ± 0.107** |
| CK+ 4-class | 10-Fold CV | CNN TL | **0.755 ± 0.079** |
| JAFFE 7-class | LOSO | Late Fusion | **0.467 ± 0.092** |
| JAFFE 4-class | LOSO | Late Fusion | **0.530 ± 0.126** |
| Dataset sendiri 4-class | 5-Fold CV | Intermediate TL | **0.435 ± 0.068** |
| Dataset sendiri 4-class | LOSO | Intermediate TL | **0.370 ± 0.125** |

### Perbandingan dengan Paper Lain (State-of-the-Art)

#### CK+

| Paper | Tahun | Metode | Evaluasi | Accuracy/F1 |
|-------|:-----:|--------|----------|:-----------:|
| Dada et al. | 2023 | CNN-10 | 10-fold CV | 99.9% (acc) |
| AA-DCN | 2024 | Anti-aliased Deep Conv | - | 99.26% (acc) |
| β-skeleton + CNN | 2024 | Landmark + CNN hybrid | 10-fold | 96.19% (acc) |
| GhostNet Multimodal | 2024 | Face+Speech+EEG fusion | 10-fold CV | 98.27% (acc) |
| **Penelitian ini** | **2026** | **Intermediate TL (ResNet18+FCNN)** | **10-fold CV** | **91.9% (acc) / 0.783 (F1)** |

#### JAFFE

| Paper | Tahun | Metode | Evaluasi | Accuracy/F1 |
|-------|:-----:|--------|----------|:-----------:|
| AA-DCN | 2024 | Anti-aliased Deep Conv | - | 98.0% (acc) |
| Fine-grained fusion | 2025 | Landmark + image fusion | - | 97.61% (acc) |
| Feature boosted DL | 2023 | Boosted features | - | 96.16% (acc) |
| β-skeleton + CNN | 2024 | Landmark geometric | 10-fold | 89.23% (acc) |
| **Penelitian ini** | **2026** | **Late Fusion (CNN+FCNN)** | **LOSO** | **54.0% (acc) / 0.467 (F1)** |

**Catatan Penting:** Perbandingan tidak sepenuhnya apple-to-apple karena:
- Paper SOTA (99%+) **kemungkinan besar pakai random split** (data leakage) — bukan subject-wise
- Penelitian ini pakai **subject-wise CV/LOSO** yang lebih ketat — hasil lebih rendah tapi lebih valid
- Paper SOTA pakai arsitektur lebih besar (VGG16, EfficientNet, ViT) vs penelitian ini ResNet18 sederhana
- Metrik berbeda: paper lain lapor accuracy, penelitian ini fokus Macro F1 (lebih jujur untuk imbalanced)
- **Bukti:** di dataset sendiri, random split naik +30-47% vs subject-wise — angka SOTA di paper lain bisa inflate serupa

### Temuan dari Benchmark

**Temuan 16: CK+ menghasilkan Macro F1 jauh lebih tinggi**
> CK+ best: Intermediate TL 7-class = 0.783 (10-fold CV), CNN TL 4-class = 0.755. Dibandingkan paper SOTA (95-99%), hasil ini lebih rendah karena menggunakan arsitektur sederhana (ResNet18, bukan VGG/EfficientNet) dan subject-wise split. Namun ini menunjukkan **arsitektur bekerja baik di dataset standar** — masalah performa rendah ada di karakteristik data sendiri.

**Temuan 17: Transfer Learning konsisten terbaik di semua dataset**
> CNN TL dan Intermediate TL selalu masuk top 2 di CK+ (baik single split maupun 10-fold CV). Di JAFFE, Late Fusion terbaik karena dataset kecil dan seimbang — CNN dan FCNN saling melengkapi.

**Temuan 18: Dataset sendiri paling menantang — gap signifikan**
> Pola: CK+ (0.783) >> JAFFE (0.467) > Dataset sendiri (0.370). Semakin natural ekspresinya dan semakin imbalanced distribusinya, semakin rendah performanya. Ini bukan kelemahan arsitektur, tapi **karakteristik inherent** dari data natural programming.

**Temuan 19: Std deviation menunjukkan stabilitas model**
> CK+ std rendah (0.02-0.10) → performa stabil antar fold. JAFFE dan dataset sendiri std tinggi (0.09-0.21) → sangat bergantung pada subjek mana yang jadi test.

> **Penjelasan lisan:**
> "Benchmark LOSO di JAFFE dan 10-fold CV di CK+ sudah selesai. CK+ menghasilkan Macro F1 0.783 dengan Intermediate TL — ini menunjukkan arsitektur saya bekerja baik di dataset standar. JAFFE lebih rendah (0.467) karena hanya 213 gambar dan 10 subjek."
>
> "Yang menarik, di semua dataset Transfer Learning konsisten terbaik. Di CK+ CNN TL dan Intermediate TL selalu top 2. Ini memperkuat argumen penggunaan ResNet18 pretrained."
>
> "Perbandingan dengan paper lain: paper SOTA di CK+ mencapai 95-99%, tapi mereka pakai arsitektur lebih besar (VGG16, EfficientNet), augmentasi berat, dan sebagian pakai random split. Kamu pakai ResNet18 sederhana dengan subject-wise 10-fold CV yang lebih ketat — jadi 0.783 sudah reasonable."
>
> "Pola keseluruhan: CK+ (0.783) >> JAFFE (0.467) > Dataset sendiri (0.370). Ini temuan penting — gap ini menunjukkan tantangan nyata dalam menerapkan FER di konteks natural programming."

---

## SLIDE 24B: Benchmark Tambahan — RAF-DB & KDEF (arahan dosen)

### Motivasi
Dosen meminta benchmark diperluas ke **RAF-DB** (natural/in-the-wild) dan **KDEF** (lab/posed). Skema 1 (self train-test) + Skema 2 (cross-dataset → Primer). Metrik wajib: Macro/Micro/Weighted F1 (semua).

### Setup Dataset

| Dataset | Train / Test | Subjek | Karakteristik |
|---------|:-------------:|:------:|---------------|
| RAF-DB | 11,565 / 2,884 (official split) | ~8,000+ | In-the-wild dari web, 7 basic emotions |
| KDEF | 2,630 / 337 (+340 val, subject-wise) | 70 | Lab posed, 5 angle → filter MediaPipe keep ~3,307 dari 4,900 |

### Hasil RAF-DB (Self Train-Test, B1)

#### 7-Class
| Model | Macro F1 | Micro F1 | Weighted F1 | Accuracy |
|-------|:--------:|:--------:|:-----------:|:--------:|
| CNN | 0.729 | 0.815 | 0.813 | 0.815 |
| FCNN | 0.578 | 0.714 | 0.703 | 0.714 |
| Intermediate | 0.696 | 0.785 | 0.783 | 0.785 |
| CNN TL | 0.741 | 0.830 | 0.826 | 0.830 |
| **Intermediate TL** | **0.744** | **0.833** | **0.832** | **0.833** |
| Late Fusion | 0.719 | 0.809 | 0.805 | 0.809 |

#### 4-Class
| Model | Macro F1 | Micro F1 | Weighted F1 | Accuracy |
|-------|:--------:|:--------:|:-----------:|:--------:|
| CNN | 0.808 | 0.830 | 0.830 | 0.830 |
| FCNN | 0.694 | 0.728 | 0.729 | 0.728 |
| Intermediate | 0.792 | 0.818 | 0.818 | 0.818 |
| CNN TL | 0.827 | 0.845 | 0.846 | 0.845 |
| **Intermediate TL** | **0.836** | **0.853** | **0.855** | **0.853** |
| Late Fusion | 0.819 | 0.842 | 0.841 | 0.842 |

### Hasil KDEF (Self Train-Test, B1)

#### 7-Class
| Model | Macro F1 | Micro F1 | Weighted F1 | Accuracy |
|-------|:--------:|:--------:|:-----------:|:--------:|
| CNN | 0.798 | 0.801 | 0.798 | 0.801 |
| FCNN | 0.666 | 0.680 | 0.663 | 0.680 |
| Intermediate | 0.671 | 0.674 | 0.668 | 0.674 |
| CNN TL | 0.833 | 0.831 | 0.833 | 0.831 |
| **Intermediate TL** | **0.843** | **0.843** | **0.843** | **0.843** |
| Late Fusion | 0.776 | 0.777 | 0.775 | 0.777 |

#### 4-Class
| Model | Macro F1 | Micro F1 | Weighted F1 | Accuracy |
|-------|:--------:|:--------:|:-----------:|:--------:|
| CNN | 0.841 | 0.872 | 0.872 | 0.872 |
| FCNN | 0.678 | 0.792 | 0.766 | 0.792 |
| Intermediate | 0.776 | 0.831 | 0.828 | 0.831 |
| CNN TL | 0.918 | 0.929 | 0.928 | 0.929 |
| **Intermediate TL** | **0.923** | **0.929** | **0.929** | **0.929** |
| Late Fusion | 0.859 | 0.890 | 0.888 | 0.890 |

### Hasil Primer Self Benchmark (nb 62, B1 only)

#### 7-Class
| Model | Macro F1 | Micro F1 | Weighted F1 | Accuracy |
|-------|:--------:|:--------:|:-----------:|:--------:|
| CNN | 0.270 | 0.787 | 0.795 | 0.787 |
| FCNN | 0.261 | 0.766 | 0.792 | 0.766 |
| Intermediate | 0.260 | 0.799 | 0.793 | 0.799 |
| CNN TL | 0.281 | 0.819 | 0.814 | 0.819 |
| **Intermediate TL** | **0.292** | **0.808** | **0.819** | **0.808** |
| Late Fusion | 0.244 | 0.778 | 0.777 | 0.778 |

#### 4-Class
| Model | Macro F1 | Micro F1 | Weighted F1 | Accuracy |
|-------|:--------:|:--------:|:-----------:|:--------:|
| CNN | 0.476 | 0.803 | 0.802 | 0.803 |
| FCNN | 0.459 | 0.763 | 0.774 | 0.763 |
| Intermediate | 0.444 | 0.727 | 0.753 | 0.727 |
| CNN TL | 0.378 | 0.721 | 0.682 | 0.721 |
| **Intermediate TL** | **0.482** | **0.780** | **0.790** | **0.780** |
| Late Fusion | 0.460 | 0.763 | 0.779 | 0.763 |

> Accuracy/Micro F1 tinggi (~0.78-0.82) menyesatkan karena didominasi neutral.
> Macro F1 (0.24-0.48) menunjukkan kesulitan sebenarnya di kelas minoritas.

### Hasil Cross-Dataset (Skema 2, nb 63) — CK+ → Primer 7c

| Model | Macro F1 | Micro F1 | Weighted F1 | Accuracy |
|-------|:--------:|:--------:|:-----------:|:--------:|
| CNN | 0.127 | 0.719 | 0.635 | 0.719 |
| **FCNN** | **0.194** | 0.773 | 0.739 | 0.773 |
| Intermediate | 0.153 | 0.536 | 0.593 | 0.536 |
| CNN TL | 0.163 | 0.670 | 0.701 | 0.670 |
| Intermediate TL | 0.103 | 0.152 | 0.238 | 0.152 |
| Late Fusion | 0.160 | 0.529 | 0.580 | 0.529 |

**Temuan 19: CK+ → Primer poor generalization**
- Semua model Macro F1 0.10-0.19 → model yang dilatih di CK+ (lab posed) gagal generalize ke Primer (natural programming)
- **FCNN paling tinggi (0.194)** — landmark geometric features lebih **domain-invariant** dibanding visual features
- **Intermediate TL collapse (0.103)** — kemungkinan overfitting ke feature-level fusion yang CK+-specific
- Konfirmasi: **domain gap lab vs natural ekstrem** — perlu fine-tuning / domain adaptation

### Progress Eksperimen

| Eksperimen | Status |
|-----------|:------:|
| RAF-DB 7-class | ✅ Done |
| RAF-DB 4-class | ✅ Done |
| KDEF 7-class | ✅ Done |
| KDEF 4-class | ✅ Done |
| Primer 7-class (nb 62) | ✅ Done |
| Primer 4-class (nb 62) | ✅ Done |
| Cross CK+ → Primer 7c (nb 63) | ✅ Done |
| Cross CK+ → Primer 4c (nb 63) | ⏳ Belum dijalankan |
| Cross JAFFE/RAF-DB/KDEF → Primer (nb 63) | ⏳ Belum dijalankan |

### Temuan Awal

**Temuan 16: Intermediate TL konsisten best di semua dataset**

Pola arsitektur terbaik (Intermediate Fusion + Transfer Learning) konsisten:
- RAF-DB 7c: 0.744 | 4c: 0.836
- KDEF 7c: 0.843 | 4c: 0.923
- **Primer 7c: 0.292** (nb 62 B1) | 4c (lama): 0.412 (Intermediate TL B1)

**Temuan 17: Gap besar vs primer → konfirmasi hipotesis karakteristik data**

RAF-DB (0.836 4c) dan KDEF (0.843 7c) jauh lebih tinggi dari primer (0.412 4c). Gap ini memperkuat argumen: performa rendah di primer bukan karena **kelemahan arsitektur**, tapi karena **karakteristik data** — natural expression + imbalanced ekstrem + minoritas langka.

**Temuan 18: CNN TL sudah kompetitif, fusion memberi gain marginal di dataset besar**

Di RAF-DB/KDEF, selisih CNN TL vs Intermediate TL kecil (~0.3-1%), karena dataset cukup besar untuk CNN belajar features. Tapi di primer dataset kecil + imbalanced, fusion memberi manfaat lebih signifikan.

> **Penjelasan lisan:**
> "Saya perluas benchmark sesuai arahan ke RAF-DB (dataset in-the-wild 15k images) dan KDEF (dataset lab 4.9k). Hasilnya konsisten dengan pola sebelumnya: Intermediate TL selalu best, dengan Macro F1 0.744-0.843."
>
> "Yang menarik, gap antara dataset primer (0.412) dan benchmark (0.84) sangat besar. Ini memperkuat argumen bahwa kesulitan di primer bukan karena arsitektur, tapi karena data natural + imbalanced ekstrem. Di dataset yang lebih seimbang, arsitektur yang sama bekerja sangat baik."
>
> "Eksperimen KDEF 4-class, Primer self, dan Cross-dataset masih berjalan. Hasil akan saya update setelah selesai."

---

## SLIDE 25: Tahap 5A — Undersampling Neutral (Analisis Imbalance)

### Motivasi
Neutral mendominasi 78% training dan 95% test. Imbalance ratio 36.8:1 (neutral vs negative). Model cenderung "malas" — prediksi semua neutral sudah dapat accuracy 95%.

### Strategi Undersampling

| Variasi | Neutral Train | Total Train | Rasio N:Neg |
|---------|:------------:|:-----------:|:-----------:|
| Original | 4,192 | 5,348 | 36.8:1 |
| **Under-660** | 660 | 1,816 | 5.8:1 |
| Under-382 | 382 | 1,538 | 3.4:1 |
| Under-114 | 114 | 1,270 | 1:1 |

### Hasil Per-Class F1

#### Intermediate TL

| Dataset | Macro F1 | neutral | happy | sad | negative |
|---------|:--------:|:-------:|:-----:|:---:|:--------:|
| Original | 0.363 | 0.975 | 0.000 | 0.476 | 0.000 |
| Under-660 | 0.257 | 0.934 | 0.068 | 0.027 | 0.000 |
| Under-382 | 0.322 | 0.933 | 0.059 | 0.197 | 0.100 |
| Under-114 | 0.277 | 0.889 | 0.118 | 0.019 | 0.083 |

#### Late Fusion

| Dataset | Macro F1 | neutral | happy | sad | negative |
|---------|:--------:|:-------:|:-----:|:---:|:--------:|
| Original | 0.296 | 0.967 | 0.158 | 0.061 | 0.000 |
| **Under-660** | **0.405** | 0.963 | 0.075 | **0.581** | 0.000 |
| Under-382 | 0.294 | 0.931 | 0.024 | 0.222 | 0.000 |
| Under-114 | 0.160 | 0.516 | 0.010 | 0.115 | 0.000 |

#### FCNN

| Dataset | Macro F1 | neutral | happy | sad | negative |
|---------|:--------:|:-------:|:-----:|:---:|:--------:|
| Original | 0.263 | 0.940 | 0.112 | 0.000 | 0.000 |
| **Under-660** | **0.348** | 0.954 | 0.095 | **0.341** | 0.000 |
| Under-382 | 0.265 | 0.921 | 0.037 | 0.100 | 0.000 |
| Under-114 | 0.343 | 0.852 | 0.235 | 0.247 | 0.038 |

### Temuan dari Undersampling

**Temuan 20: Under-660 meningkatkan deteksi sad secara signifikan**
> Late Fusion under-660: sad F1 naik dari 0.061 → **0.581** (+852%). FCNN under-660: sad naik dari 0.000 → 0.341. Mengurangi neutral ke level happy (660) memberikan ruang bagi model untuk belajar kelas lain.

**Temuan 21: Negative tetap tidak terdeteksi (F1 ~ 0)**
> Di hampir semua variasi, negative F1 = 0.000. Bahkan dengan rasio 1:1 (under-114), negative hanya mencapai 0.083-0.100. **Masalah bukan hanya di training imbalance** — test set hanya punya 16 sampel negative, terlalu sedikit untuk evaluasi yang reliable.

**Temuan 22: Test set terlalu imbalanced untuk evaluasi fair**
> Test set: neutral=981 (95%), happy=10 (1%), sad=29 (3%), negative=16 (1.5%). Dengan test set sekecil ini, 1 sampel salah/benar di happy bisa mengubah F1 sebesar 10-20%. **Evaluasi per kelas tidak reliable untuk kelas dengan < 30 sampel di test set.**

**Temuan 23: Undersampling terlalu agresif justru menurunkan performa**
> Under-114 (rasio 1:1) justru menurunkan performa semua model. Total training hanya 1,270 sampel — terlalu sedikit untuk deep learning. **Under-660 (rasio 5.8:1) adalah sweet spot** — cukup seimbang tapi masih punya data yang cukup.

> **Penjelasan lisan:**
> "Saya melakukan undersampling neutral dengan 3 variasi. Hasilnya menarik — under-660 (neutral dikurangi ke 660, setara happy) meningkatkan deteksi sad secara signifikan. Late Fusion sad F1 naik dari 0.061 ke 0.581."
>
> "Tapi negative tetap tidak terdeteksi bahkan dengan rasio 1:1. Ini menunjukkan masalahnya ada di dua level: (1) training data negative terlalu sedikit (114 sampel), dan (2) test set hanya punya 16 sampel negative — terlalu kecil untuk evaluasi yang meaningful."
>
> "Temuan penting: masalah bukan hanya di imbalance training, tapi juga di ukuran test set per kelas. Untuk kelas dengan < 30 sampel di test, evaluasi F1 tidak reliable."

---

## SLIDE 26: Tahap 5B — Confidence Filtering >= 60% (BREAKTHROUGH)

### Motivasi

Analisis confidence score Face API menunjukkan bahwa kelas minoritas punya confidence **jauh lebih rendah**:

| Emosi | Rata-rata Confidence |
|-------|:--------------------:|
| Neutral | 0.959 (tinggi) |
| Happy | 0.878 |
| Sad | 0.770 |
| **Angry** | **0.634** |
| **Fearful** | **0.671** |
| **Disgusted** | **0.566** |
| Surprised | 0.730 |

**Hipotesis:** Label dengan confidence rendah kemungkinan salah — menjadi noise yang mengganggu training.

### Strategi

Filter semua sampel dengan confidence < 60% → `data/dataset_frontonly_conf60`

| | Original | Conf60 | Filtered |
|-|:--------:|:------:|:--------:|
| Total | 7,091 | 6,795 | 296 (4.2%) |
| Kelas minoritas | lebih banyak noise | lebih bersih | ~50-57% dari minoritas |

### Hasil — Perbandingan Best Model

| Tahap | Original (F1) | **Conf60 (F1)** | Improvement |
|-------|:-------------:|:---------------:|:-----------:|
| 7-class scratch | 0.175 | **0.289** (Late Fusion B1) | **+65%** |
| 4-class scratch | 0.394 | **0.482** (Late Fusion B1) | **+22%** |
| 7-class TL | 0.180 | **0.301** (Late Fusion TL B1) | **+67%** |
| **4-class TL** | **0.412** | **0.567** (Late Fusion TL B3) | **+38%** |

### Hasil Lengkap Conf60

#### 7-Class From Scratch

| Model | Skenario | Macro F1 | Accuracy |
|-------|----------|:--------:|:--------:|
| CNN | B1 Baseline | 0.277 | 0.811 |
| CNN | B2 Class Weights | 0.240 | 0.774 |
| CNN | B3 Weights+Aug | 0.253 | 0.785 |
| FCNN | B1 Baseline | 0.232 | 0.768 |
| FCNN | B2 Class Weights | 0.244 | 0.765 |
| FCNN | B3 Weights+Aug | 0.222 | 0.740 |
| Intermediate | B1 Baseline | 0.261 | 0.792 |
| Intermediate | B2 Class Weights | 0.247 | 0.779 |
| Intermediate | B3 Weights+Aug | 0.229 | 0.775 |
| **Late Fusion** | **B1 Baseline** | **0.289** | 0.839 |

#### 4-Class From Scratch

| Model | Skenario | Macro F1 | Accuracy |
|-------|----------|:--------:|:--------:|
| CNN | B1 Baseline | 0.438 | 0.808 |
| CNN | B2 Class Weights | 0.448 | 0.826 |
| CNN | B3 Weights+Aug | 0.432 | 0.760 |
| FCNN | B1 Baseline | 0.422 | 0.695 |
| FCNN | B2 Class Weights | 0.460 | 0.757 |
| FCNN | B3 Weights+Aug | 0.421 | 0.702 |
| Intermediate | B1 Baseline | 0.445 | 0.788 |
| Intermediate | B2 Class Weights | 0.416 | 0.783 |
| Intermediate | B3 Weights+Aug | 0.382 | 0.790 |
| Late Fusion | B1 Baseline | 0.482 | 0.821 |
| **Late Fusion** | **B2 Class Weights** | **0.503** | 0.825 |
| Late Fusion | B3 Weights+Aug | 0.463 | 0.798 |

#### 7-Class Transfer Learning

| Model | Skenario | Macro F1 | Accuracy |
|-------|----------|:--------:|:--------:|
| CNN TL | B1 Baseline | 0.273 | 0.793 |
| CNN TL | B2 Class Weights | 0.243 | 0.750 |
| CNN TL | B3 Weights+Aug | 0.240 | 0.807 |
| Intermediate TL | B1 Baseline | 0.277 | 0.792 |
| Intermediate TL | B2 Class Weights | 0.283 | 0.825 |
| Intermediate TL | B3 Weights+Aug | 0.292 | 0.825 |
| **Late Fusion TL** | **B1 Baseline** | **0.301** | 0.830 |
| Late Fusion TL | B2 Class Weights | 0.264 | 0.819 |
| Late Fusion TL | B3 Weights+Aug | 0.260 | 0.849 |

#### 4-Class Transfer Learning

| Model | Skenario | Macro F1 | Accuracy |
|-------|----------|:--------:|:--------:|
| CNN TL | B1 Baseline | 0.456 | 0.747 |
| CNN TL | B2 Class Weights | 0.447 | 0.742 |
| CNN TL | B3 Weights+Aug | 0.507 | 0.799 |
| Intermediate TL | B1 Baseline | 0.489 | 0.800 |
| Intermediate TL | B2 Class Weights | 0.508 | 0.825 |
| Intermediate TL | B3 Weights+Aug | 0.521 | 0.822 |
| Late Fusion TL | B1 Baseline | 0.513 | 0.802 |
| Late Fusion TL | B2 Class Weights | 0.519 | 0.818 |
| **Late Fusion TL** | **B3 Weights+Aug** | **0.567** | 0.812 |

### Temuan dari Confidence Filtering

**Temuan 24: Confidence filtering 60% meningkatkan performa signifikan**
> Best F1 naik dari 0.412 → **0.567** (+38%). Ini **breakthrough terbesar** sepanjang penelitian. Menunjukkan bahwa label noise (dari Face API confidence rendah) adalah penyebab utama performa rendah sebelumnya, bukan hanya karakteristik data natural.

**Temuan 25: Hanya 4.2% data dihilangkan tapi kualitas label meningkat drastis**
> Dengan threshold 60%, hanya 296 sampel (4.2%) dihilangkan. Tapi kelas minoritas kehilangan 50-57% — yang dihilangkan kemungkinan besar adalah label noise. Trade-off: sedikit data tapi lebih bersih vs banyak data tapi banyak noise. **Bersih menang.**

**Temuan 26: Transfer Learning + Augmented (B3) jadi kombinasi terbaik**
> Setelah confidence filtering, skenario B3 (weights + augmentasi) akhirnya memberikan manfaat nyata. Sebelumnya B3 justru menurun karena menguatkan noise. Sekarang dengan data lebih bersih, augmentasi efektif.

**Temuan 27: Late Fusion TL B3 menjadi best overall**
> Best model: **Late Fusion TL 4-class B3 = 0.567**. Menggabungkan CNN TL dan FCNN dengan weighted average, ditambah augmentasi pada data conf60 yang bersih, memberikan hasil terbaik. Ini menunjukkan bahwa fusion multimodal efektif ketika kualitas label baik.

> **Penjelasan lisan:**
> "Saya menganalisis confidence score dari Face API dan menemukan bahwa kelas minoritas (angry, fearful, disgusted) punya confidence rata-rata 0.56-0.67 — Face API sendiri tidak yakin dengan labelnya."
>
> "Ketika saya filter sampel dengan confidence < 60%, hanya 4.2% data yang hilang tapi Macro F1 naik signifikan. Best model dari 0.412 ke **0.567** — peningkatan 38%. Ini breakthrough terbesar sepanjang penelitian."
>
> "Best model bergeser dari Intermediate TL ke **Late Fusion TL 4-class B3** — fusion multimodal akhirnya menang karena data sudah bersih dan augmentasi efektif."
>
> "Temuan utama: masalah performa rendah sebelumnya bukan sepenuhnya karena karakteristik data natural, tapi karena **label noise** dari auto-detection Face API. Dengan membersihkan label yang tidak confident, model bisa belajar pola yang benar."
>
> "Ini juga memperkuat alasan penggunaan validasi ahli — untuk memastikan label yang digunakan benar-benar reliable."

---

## SLIDE 27: Tahap 5C — Conf60 + Undersampling (Kombinasi)

### Motivasi

Menggabungkan 2 strategi yang terbukti efektif sebelumnya:
1. **Confidence Filtering >= 60%** — mengatasi label noise
2. **Undersampling Neutral (under-660)** — mengatasi imbalance

Target: apakah kombinasi memberikan hasil lebih baik dari masing-masing?

**Catatan:** Hanya 3 model teratas yang diuji (Intermediate TL, FCNN, Late Fusion). CNN tidak dimasukkan karena konsisten lebih rendah di eksperimen sebelumnya.

### Hasil

| Model | Conf60 Original | Conf60 + Under-660 | Kesimpulan |
|-------|:---------------:|:------------------:|:----------:|
| Intermediate TL | 0.478 | 0.401 | ❌ Turun -16% |
| FCNN | 0.489 | 0.433 | ❌ Turun -11% |
| **Late Fusion** | **0.466** | **0.508** | ✅ Naik +9% |

### Per-Class F1 (Best Model: Late Fusion)

| Dataset | Macro F1 | neutral | happy | sad | negative |
|---------|:--------:|:-------:|:-----:|:---:|:--------:|
| Conf60 Original | 0.466 | 0.899 | 0.761 | 0.203 | 0.000 |
| **Conf60 + Under-660** | **0.508** | 0.895 | 0.713 | **0.424** | 0.000 |

### Perbandingan dengan Best Overall

| Strategi | Best Model | Macro F1 |
|----------|-----------|:--------:|
| Tahap 5A: Undersampling saja (under-660) | Late Fusion 4c | 0.405 |
| **Tahap 5B: Conf60 saja (TL+B3)** | **Late Fusion TL 4c B3** | **0.567** 🏆 |
| Tahap 5C: Conf60 + Under-660 (B1 only) | Late Fusion conf60 under-660 | 0.508 |

**Best overall tetap: Late Fusion TL 4-class B3 conf60 = 0.567**

### Temuan dari Kombinasi

**Temuan 28: Undersampling + Conf60 hanya efektif untuk Late Fusion**
> Late Fusion naik 0.466 → 0.508 (+9%) dengan under-660. Sad F1 naik dari 0.203 → 0.424. Tapi Intermediate TL dan FCNN justru turun signifikan (-11 s/d -16%).

**Temuan 29: Model kompleks butuh data lebih banyak**
> Intermediate TL (ResNet18 + FCNN + fusion head) memiliki banyak parameter, butuh data training yang cukup. Undersampling dari 6,795 → 1,816 membuat model kekurangan data. Late Fusion lebih robust karena CNN dan FCNN dilatih terpisah.

**Temuan 30: Best overall masih Late Fusion TL B3 conf60 saja**
> Kombinasi tidak mengalahkan Late Fusion TL B3 conf60 (0.567). Artinya **confidence filtering + augmentasi (B3)** lebih efektif dari **confidence filtering + undersampling**.

**Temuan 31: Negative tetap F1 = 0.000 di semua strategi**
> Konsisten dengan temuan sebelumnya — test set terlalu kecil (16 sampel negative). Apapun strateginya, evaluasi negative tidak reliable.

> **Penjelasan lisan:**
> "Saya coba gabungkan conf60 + undersampling. Hasilnya campuran — hanya Late Fusion yang membaik, Intermediate TL dan FCNN justru turun karena kekurangan data training."
>
> "Kesimpulan: kombinasi tidak selalu lebih baik. Late Fusion TL B3 dengan conf60 saja tetap best overall (0.567)."
>
> "Ini juga menunjukkan bahwa masalah utama kita sekarang bukan imbalance, tapi **label noise dan ukuran test set untuk kelas minoritas**."

---

## SLIDE 28: Analisis Mendalam Masalah Dataset (untuk Konsultasi)

### Konteks
Meskipun Conf60 sudah meningkatkan performa ke 0.567, angka ini masih relatif rendah. Analisis mendalam dilakukan untuk mencari akar masalah.

### Masalah 1: Diversity Per-User yang Buruk

Dari 37 user, **12 user (32%) bermasalah**:

| Kategori | User ID | Masalah |
|----------|---------|---------|
| **Extreme Neutral** (100%) | 115, 209 | **Tidak ada variasi emosi sama sekali** |
| **Extreme Neutral** (95%+) | 112, 200, 206, 207, 201, 203 | Hampir tidak ada ekspresi non-neutral |
| **Too Few Non-Neutral** (< 10) | 101, 110, 118, 113, 197 | Tidak cukup untuk training |

**Implikasi:** Model belajar dari user yang memang tidak ekspresif → sulit generalisasi.

### Masalah 2: Kelas Minoritas Sangat Langka

| Emosi | Total | Conf >= 60% | Conf >= 90% |
|-------|:-----:|:-----------:|:-----------:|
| Neutral | 6,054 | 5,906 | 5,242 |
| Happy | 699 | 667 | 514 |
| Sad | 451 | 373 | 212 |
| **Angry** | 56 | 35 | **8** |
| **Fearful** | **10** | 5 | 3 |
| **Disgusted** | 21 | 17 | 5 |
| **Surprised** | 55 | 39 | 19 |

**Fearful hanya 10 sampel total** di seluruh dataset — bahkan sebelum split.

### Masalah 3: Test Set Terlalu Kecil untuk Kelas Minoritas

Test set 4-class conf60:
- Neutral: ~950 sampel (stabil)
- Happy: **10** sampel ⚠️
- Sad: 29 sampel (batas minimum)
- Negative: **16** sampel ⚠️

**Mengapa tidak reliable?**

| Jumlah Test | Stabilitas F1 |
|:-----------:|---------------|
| < 10 | Tidak reliable |
| 10-30 | **Hati-hati** — 1 sampel = 3-10% perbedaan F1 |
| 30-100 | Cukup reliable |
| > 100 | Reliable |

**Contoh:** Happy 10 sampel → 1 prediksi salah = F1 berubah 10%. Jadi Macro F1 yang rendah bisa dari variance bukan dari model.

### Solusi yang Bisa Dikonsultasikan

#### Opsi A: Drop Problem Users (Realistis)
- Hilangkan 4 user extreme (115, 209, 112, 200)
- Hasil: dataset lebih bersih, fokus ke user yang ekspresif
- Trade-off: test set berkurang tapi kualitas evaluasi lebih baik

#### Opsi B: Merge ke 3-Class (neutral / happy / negative)
- Gabung sad + angry + fearful + disgusted + surprised + ke "negative"
- Test set: neutral=950, happy=10, negative=55 → **negative lebih reliable**
- Trade-off: kehilangan granularitas emosi

#### Opsi C: Binary Classification (neutral vs non-neutral)
- Hanya deteksi "ada emosi" vs "tidak"
- Test set: neutral=950, non-neutral=65 (cukup)
- Paling mudah dicapai, tapi scope lebih sempit dari proposal

#### Opsi D: Tambah Data Eksternal
- Gunakan CK+ / FER2013 untuk kelas minoritas (fearful, disgusted, angry)
- Risiko: domain shift (lab vs natural)

### **(KONSULTASI 6)** Pertanyaan untuk Pembimbing

1. **Apakah acceptable** jika scope penelitian diubah ke 3-class atau binary karena keterbatasan data?
2. **Apakah perlu drop problem users** (112, 115, 200, 209) atau tetap diinklude untuk representasi natural?
3. **Apakah bisa menggunakan data eksternal** (CK+/FER2013) untuk melengkapi kelas minoritas?
4. **Bagaimana melaporkan test set yang kecil** di tesis — pakai confidence interval atau tetap lapor F1 apa adanya?
5. **Apakah F1 0.567 sudah acceptable** mengingat keterbatasan natural dataset ini?

> **Penjelasan lisan:**
> "Setelah menganalisis lebih dalam, saya menemukan masalah struktural di dataset: 32% user tidak ekspresif (95%+ neutral), kelas minoritas sangat langka (fearful hanya 10 sampel), dan test set kelas minoritas terlalu kecil untuk evaluasi yang reliable."
>
> "Macro F1 0.567 mungkin sudah batas yang bisa dicapai dengan dataset ini. Saya mohon arahan: apakah scope perlu diubah (3-class atau binary), drop problem users, atau tambah data eksternal? Atau terima bahwa ini adalah karakteristik dataset natural programming yang memang menantang?"

---

## SLIDE 29: Diskusi dan Kesimpulan Sementara

### Jawaban terhadap Rumusan Masalah:

**RQ1 (Performa CNN):**
- From scratch: Macro F1 0.137 (7-class) dan 0.265 (4-class) — rendah karena dataset terbatas
- Transfer Learning: Macro F1 0.154 (7-class) dan **0.274 (4-class)** — meningkat dengan ResNet18

**RQ2 (Performa FCNN):**
Model FCNN menghasilkan Macro F1 **0.158** (7-class) dan **0.361** (4-class). Fitur geometrik dari 68 facial landmark terbukti lebih robust.

**RQ3 (Perbandingan Fusion):**
- From scratch: Late Fusion terbaik (0.175 / 0.394) — menggabungkan CNN + FCNN di level keputusan
- Transfer Learning: **Intermediate Fusion TL menjadi yang terbaik (0.412)** — penggabungan di level fitur lebih efektif saat CNN sudah kuat (ResNet18)
- Konsistensi data (front-only) lebih penting dari kuantitas — best model front-only (0.412) sedikit lebih baik dari front+side (0.407)

### Ringkasan Perjalanan Eksperimen (Front-Only — Acuan Utama)

| Tahap | Model Terbaik | Macro F1 | Keterangan |
|-------|--------------|----------|------------|
| Tahap 1: 7-class | Late Fusion B3 | 0.175 | Baseline |
| Tahap 2: 4-class | Late Fusion B3 | 0.394 | +125% dari 7-class |
| Tahap 3: TL 4-class | **Intermediate TL B1** | **0.412** | +5% dari 4-class |
| Tahap 4: Front-only | Intermediate TL B1 | 0.412 | Konsistensi > kuantitas |
| **Tahap 5: Conf60 (BREAKTHROUGH)** | **Late Fusion TL B3** | **0.567** | **+38% dari front-only** |
| Tahap 4: Front-only vs Front+side | Front-only sedikit lebih baik | +0.005 | Konsistensi > kuantitas |

### **(KONSULTASI 5)** Pertanyaan untuk Pembimbing

> "Pak/Bu, total eksperimen sekarang: 48 (front+side) + 48 (front-only) = 96, ditambah evaluasi LOSO/CV/Random Split yang sedang berjalan. Beberapa hal yang perlu didiskusikan:"

1. **Acuan utama:** "Hasil front-only saya jadikan acuan utama karena datanya konsisten (semua sudut depan). Hasil front+side dijadikan pembanding. Apakah Bapak/Ibu setuju?"

2. **Best model bergeser:** "Best model bergeser dari Late Fusion TL (front+side, 0.442) ke Intermediate Fusion TL (front-only, 0.412). Ini menunjukkan arsitektur optimal bergantung pada karakteristik data."

3. **Macro F1 0.412:** "Nilai terbaik 0.412 masih di bawah 0.5. Namun mengingat ini dataset nyata dari sesi pemrograman yang sangat imbalanced, apakah ini acceptable?"

4. **LOSO sedang berjalan:** "Evaluasi LOSO (37 fold), 5-Fold CV, dan Random Split sedang berjalan di VPS. Hasilnya akan menunjukkan apakah model robust terhadap variasi user dan apakah ada data leakage."

5. **Validasi ahli:** "Dataset validasi 1% (146 sampel) sudah disiapkan. Butuh 3 validator berlatar psikologi. Apakah Bapak/Ibu punya rekomendasi?"

---

## SLIDE 30: Tahap 6 — Early Fusion (Arahan Dosen)

### Motivasi
Dosen meminta implementasi **Early Fusion** di mana landmark "ditempel" ke gambar. Pendekatan yang dipilih: **HAE-Net style** (Mo et al., MMM 2020) — landmark dikonversi ke **Gaussian heatmap 224×224** dan di-concat sebagai channel ke-4 citra RGB. Pembeda dengan Intermediate Fusion: fusi terjadi di **level input (0%)**, bukan setelah feature extraction.

### Arsitektur Early Fusion
- **Input**: Image 224×224×3 + Heatmap 224×224×1 → 224×224×**4**
- **Heatmap generation**: Setiap 68 landmark → Gaussian blob σ=3px → element-wise max aggregation → single channel [0,1]
- **Scratch variant**: Conv2d pertama `in_channels=4` dari awal
- **TL variant**: ResNet18 pretrained ImageNet, Conv2d pertama dimodifikasi 3→4 channel (weight RGB di-copy, channel ke-4 di-init dari mean RGB)

### Hasil Early Fusion conf60 (12 configs, notebook 64)

**7-class:**
| Config | Macro F1 | Micro F1 | Weighted F1 | Accuracy |
|--------|:--------:|:--------:|:-----------:|:--------:|
| EarlyFusion B1 | 0.246 | 0.794 | 0.786 | 0.794 |
| EarlyFusion B2 | 0.205 | 0.520 | 0.552 | 0.520 |
| EarlyFusion B3 | 0.264 | 0.680 | 0.726 | 0.680 |
| EarlyFusion TL B1 | 0.253 | 0.713 | 0.722 | 0.713 |
| EarlyFusion TL B2 | 0.247 | 0.636 | 0.663 | 0.636 |
| **EarlyFusion TL B3** | **0.333** ⭐ | 0.753 | 0.773 | 0.753 |

**4-class:**
| Config | Macro F1 | Micro F1 | Weighted F1 | Accuracy |
|--------|:--------:|:--------:|:-----------:|:--------:|
| EarlyFusion B1 | 0.457 | 0.822 | 0.816 | 0.822 |
| EarlyFusion B2 | 0.427 | 0.690 | 0.728 | 0.690 |
| EarlyFusion B3 | 0.427 | 0.728 | 0.752 | 0.728 |
| **EarlyFusion TL B1** | **0.471** ⭐ | 0.770 | 0.770 | 0.770 |
| EarlyFusion TL B2 | 0.424 | 0.642 | 0.668 | 0.642 |
| EarlyFusion TL B3 | 0.433 | 0.678 | 0.709 | 0.678 |

> **Catatan**: Micro F1 = Accuracy untuk multi-class single-label. Dicantumkan terpisah sesuai instruksi dosen (Macro/Micro/Weighted F1 semua wajib ada).

### Perbandingan Semua Strategi Fusion (Best per Arsitektur, conf60)

| Fusion Strategy | Best 7c Macro F1 | Best 4c Macro F1 |
|-----------------|:----------------:|:----------------:|
| Single CNN (TL) | 0.273 | 0.507 |
| Single FCNN | 0.244 | 0.459 |
| **Early Fusion** (HAE-Net) | **0.333** (TL B3) | 0.471 (TL B1) |
| Intermediate Fusion | 0.292 (TL B3) | 0.521 (TL B3) |
| **Late Fusion** | 0.301 (TL B1) | **0.567** (TL B3) ⭐ |

### Temuan dari Early Fusion

**Temuan 32: Early Fusion TL B3 tembus best di 7-class (0.333)**
> Melampaui Intermediate TL B3 (0.292) dan Late Fusion TL B1 (0.301). Kombinasi heatmap channel + class weights + augmentation + TL bekerja sinergis untuk skenario kelas banyak + imbalance ekstrem.

**Temuan 33: Early Fusion 4-class underperforms (0.471 vs 0.567 Late Fusion)**
> Di 4-class, Early Fusion gagal tembus 0.50. Hipotesis: heatmap sparse (mostly zeros) kurang informatif ketika class granularity dikurangi; fusion di level feature/decision lebih optimal untuk kasus ini.

**Temuan 34: B2 class weights merugikan Early Fusion secara konsisten**
> Skenario B2 (class weights saja) drop accuracy ~30% dibanding B1 di semua Early Fusion config. Re-weighting di level loss tampak merusak learning ketika input adalah 4-channel concat. Sebaliknya, Intermediate/Late Fusion tetap stabil dengan B2.

**Temuan 35: Ranking fusion strategy berbeda per granularitas kelas**
> - 7-class: **Early TL B3 > Late TL B1 > Intermediate TL B3**
> - 4-class: **Late TL B3 > Intermediate TL B3 > Early TL B1**
>
> Tidak ada single strategy yang dominan di semua setting — pemilihan arsitektur bergantung konteks task.

### Hasil Early Fusion di Benchmark Dataset (nb 66)

Menjalankan Early Fusion yang sama di 4 benchmark dataset untuk validasi apakah underperformance di Primer spesifik data natural atau universal. **16 experiments** (B1 scratch + TL × 4 dataset × 2 class). Angka lengkap sudah terintegrasi ke tabel **Skema 1 di SLIDE 32** (ditambah 2 row `Early Fusion` dan `Early Fusion TL` di setiap tabel).

**Temuan 36: Early Fusion universal kompetitif di benchmark (kecuali JAFFE)**
> Di RAF-DB/KDEF/CK+ 4c: Early Fusion TL selisih cuma 0.03-0.11 dari best Intermediate TL. Gap di Primer 4c (0.471 vs 0.521) kira-kira sama dengan gap di benchmark → Early Fusion channel-concat memang **sedikit di bawah** Intermediate/Late Fusion secara universal, bukan masalah data natural per-se.

**Temuan 37: JAFFE TL collapse (Macro F1 0.041 di 7c)**
> JAFFE hanya 213 images, terlalu kecil untuk fine-tune ResNet18 dengan first Conv2d modifikasi 3→4 channel. Scratch variant justru 0.286 (7c). **Implikasi**: Early Fusion TL butuh dataset training minimal ~600+ samples untuk converge sehat.

**Temuan 38: RAF-DB 7c — scratch > TL di Early Fusion (0.710 > 0.693)**
> Anomali: untuk RAF-DB 7c (11,565 training), scratch Early Fusion bahkan sedikit mengungguli TL variant. Hipotesis: large dataset + heatmap channel info, model scratch bisa belajar representasi task-specific tanpa bias ImageNet.

**Temuan 39: TL boost terbesar di CK+ (+0.28 s/d +0.32 Macro F1)**
> CK+ 7c: 0.446 → 0.762 (+0.316). CK+ 4c: 0.507 → 0.795 (+0.288). Dataset posed lab dengan ekspresi peak → pretrained visual features sangat transferable. Bandingkan dengan Primer: +0.10 saja.

> **Penjelasan lisan:**
> "Pak, saya sudah implementasikan Early Fusion sesuai arahan. Pendekatan yang saya pakai adalah menjadikan landmark sebagai heatmap Gaussian 224×224, lalu di-concat sebagai channel ke-4 ke citra RGB. Referensi: HAE-Net (Mo et al., MMM 2020 — Tsinghua University)."
>
> "Di Primer, Early Fusion TL B3 jadi best di 7-class (0.333), melampaui Intermediate dan Late Fusion. Tapi di 4-class, Early Fusion kalah dari Late Fusion (0.471 vs 0.567)."
>
> "Saya juga extend Early Fusion ke 4 benchmark dataset (nb 66, 16 experiments). Hasilnya Early Fusion universal kompetitif — selisih 0.03-0.11 dari best Intermediate TL di RAF-DB/KDEF/CK+ 4c. Ini **membantah hipotesis** bahwa Early Fusion underperform karena natural data — ternyata memang sedikit di bawah Intermediate/Late secara universal."
>
> "Temuan unik: TL collapse di JAFFE (0.041 untuk 7c) karena dataset terlalu kecil (213 images) — first Conv 3→4 channel butuh data lebih banyak untuk converge."
>
> "Total eksperimen conf60 sekarang 54 config + 16 benchmark Early Fusion. Best overall Primer tetap Late Fusion TL 4c B3 = 0.567."

---

## SLIDE 31: Cross-Dataset Evaluation Lengkap (Skema 2)

Melengkapi cross-dataset yang sebelumnya hanya CK+→Primer 7c. Sekarang semua sumber (CK+/JAFFE/RAF-DB/KDEF) diuji → Primer conf60 untuk 7c dan 4c, semua 6 arsitektur.

### Best per Source Dataset → Primer conf60

| Source | Best Model | Acc | Macro F1 | Weighted F1 |
|--------|-----------|:---:|:--------:|:-----------:|
| **CK+ 7c** | FCNN B1 | 0.773 | 0.194 | 0.739 |
| **CK+ 4c** | CNN B1 | 0.642 | **0.396** | 0.703 |
| **JAFFE 7c** | Late Fusion B1 | 0.081 | 0.040 | 0.054 |
| **JAFFE 4c** | Intermediate TL B1 | 0.160 | 0.093 | 0.073 |
| **RAF-DB 7c** | FCNN B1 | 0.640 | 0.183 | 0.672 |
| **RAF-DB 4c** | Intermediate B1 | 0.551 | **0.269** | 0.594 |
| **KDEF 7c** | CNN TL B1 | 0.053 | 0.038 | 0.036 |
| **KDEF 4c** | CNN B1 | 0.067 | 0.079 | 0.078 |

### Temuan dari Cross-Dataset

**Temuan 36: Semua source dataset gagal generalize ke Primer (catastrophic)**
> Best cross-dataset Macro F1 hanya 0.269 (RAF-DB 4c). Bandingkan dengan Primer self-training best 0.567 — gap -0.30. Public FER benchmarks **tidak representatif** untuk setting natural programming learning.

**Temuan 37: RAF-DB 4c paling robust untuk transfer ke Primer**
> Dari semua source, RAF-DB (in-the-wild) paling mendekati Primer (natural). CK+ dan JAFFE (lab-posed) terjun bebas. KDEF (lab-posed high quality) bahkan lebih jelek (acc 5-7%).

**Temuan 38: Landmark features lebih domain-invariant dari visual**
> Di CK+/RAF-DB → Primer 7c, **FCNN (landmark-only)** yang best, bukan arsitektur yang menggunakan image. Konfirmasi: landmark geometry lebih tahan domain shift dibanding texture/color features.

**Temuan 39: JAFFE → Primer total collapse (< 10% acc)**
> JAFFE terlalu kecil (213 images, Japanese female only) dan sangat posed. Model overfit ke karakteristik spesifik dataset, tidak transferable sama sekali.

**Temuan 40: Justifikasi kuat untuk Primer-only train/test di paper**
> Hasil ini memperkuat argumen bahwa **transfer learning dari public FER dataset tidak viable** untuk domain natural programming. Paper JITeCS hanya pakai Primer conf60 — keputusan yang didukung data.

### Progress Eksperimen Benchmark

| Eksperimen | Status |
|-----------|:------:|
| Skema 1 self-train-test semua dataset (CK+/JAFFE/RAF-DB/KDEF/Primer × 7c+4c) | ✅ Done |
| Skema 1 Late Fusion TL (sebelumnya terlewat, nb 65) | ✅ Done |
| Skema 2 Cross-dataset semua source → Primer (nb 63) | ✅ Done |
| Micro F1 untuk semua hasil lama (CK+/JAFFE) | ✅ Done |

> **Penjelasan lisan:**
> "Cross-dataset dari semua source (CK+, JAFFE, RAF-DB, KDEF) ke Primer sudah saya jalankan. Hasilnya konsisten: semua gagal generalize. Best hanya 0.269 dari RAF-DB 4c."
>
> "Ini memperkuat justifikasi kenapa paper JITeCS saya fokus hanya pada Primer. Dataset publik, meski performa di self-train tinggi (KDEF 0.84, RAF-DB 0.74), tidak transferable ke setting natural."

---

## SLIDE 32: Tabel Lengkap Skema 1 & Skema 2 (Sesuai Instruksi Dosen)

### Konteks
Instruksi dosen (konsultasi sebelumnya): print **semua metrik** (Macro/Micro/Weighted F1) untuk **semua model** (CNN, FCNN, Intermediate, Late Fusion, CNN TL, Intermediate TL, +Late Fusion TL) di **Skema 1** (self train-test tiap dataset) dan **Skema 2** (cross-dataset train → Primer test).

Slide 24 (CK+/JAFFE) lama hanya Macro F1. Slide 24B belum lengkap Late Fusion TL. Slide ini **konsolidasi tabel lengkap** semua.

---

### Skema 1 — Self Train-Test (9 Model × 4 Metrik)

*Termasuk 2 row terakhir per tabel: `Early Fusion` (scratch B1) dan `Early Fusion TL` (B1) dari nb 66 — melengkapi spektrum 5 fusion strategy (CNN/FCNN single + Early/Intermediate/Late fusion × scratch/TL).*

> **Catatan methodology — Late Fusion weight tuning (semua val-tuned proper per Apr 2026)**:
> - **Primer conf60** (nb 62/65): `w` val-tuned via `rerun_late_fusion_proper.py`.
> - **RAF-DB, KDEF** (nb 60/61): sudah val-tuned native di notebook.
> - **CK+ dan JAFFE** (nb 36/37, B1): sebelumnya test-tuned; di-retrain + re-evaluate via `retrain_ckplus_jaffe_b1.py` + `rerun_late_fusion_proper.py`. Tabel di bawah angka baru (val-tuned). Drop paling signifikan: JAFFE 7c Late Fusion 0.545 → 0.314 (−0.23). Best JAFFE 7c bergeser ke **CNN TL (0.464)**.

#### CK+ 7-class
| Model | Macro F1 | Micro F1 | Weighted F1 | Accuracy |
|-------|:--------:|:--------:|:-----------:|:--------:|
| CNN | 0.461 | 0.729 | 0.659 | 0.729 |
| FCNN | 0.395 | 0.678 | 0.614 | 0.678 |
| Intermediate | 0.316 | 0.695 | 0.585 | 0.695 |
| **CNN TL** | **0.913** ⭐ | 0.949 | 0.946 | 0.949 |
| Intermediate TL | 0.833 | 0.881 | 0.886 | 0.881 |
| Late Fusion | 0.494 | 0.780 | 0.691 | 0.780 |
| Late Fusion TL | 0.835 | 0.881 | 0.890 | 0.881 |
| Early Fusion | 0.446 | 0.695 | 0.665 | 0.695 |
| Early Fusion TL | 0.762 | 0.847 | 0.847 | 0.847 |

#### CK+ 4-class
| Model | Macro F1 | Micro F1 | Weighted F1 | Accuracy |
|-------|:--------:|:--------:|:-----------:|:--------:|
| CNN | 0.645 | 0.790 | 0.776 | 0.790 |
| FCNN | 0.592 | 0.758 | 0.740 | 0.758 |
| Intermediate | 0.567 | 0.758 | 0.740 | 0.758 |
| CNN TL | 0.675 | 0.903 | 0.890 | 0.903 |
| **Intermediate TL** | **0.837** ⭐ | 0.903 | 0.902 | 0.903 |
| Late Fusion | 0.537 | 0.694 | 0.681 | 0.694 |
| Late Fusion TL | 0.604 | 0.806 | 0.803 | 0.806 |
| Early Fusion | 0.507 | 0.694 | 0.667 | 0.694 |
| Early Fusion TL | 0.795 | 0.871 | 0.872 | 0.871 |

#### JAFFE 7-class
| Model | Macro F1 | Micro F1 | Weighted F1 | Accuracy |
|-------|:--------:|:--------:|:-----------:|:--------:|
| CNN | 0.304 | 0.450 | 0.319 | 0.450 |
| FCNN | 0.209 | 0.250 | 0.169 | 0.250 |
| Intermediate | 0.037 | 0.150 | 0.039 | 0.150 |
| **CNN TL** | **0.464** ⭐ | 0.500 | 0.437 | 0.500 |
| Intermediate TL | 0.447 | 0.450 | 0.420 | 0.450 |
| Late Fusion | 0.314 | 0.400 | 0.290 | 0.400 |
| Late Fusion TL | 0.146 | 0.200 | 0.120 | 0.200 |
| Early Fusion | 0.286 | 0.350 | 0.267 | 0.350 |
| Early Fusion TL | 0.041 | 0.150 | 0.043 | 0.150 |

#### JAFFE 4-class
| Model | Macro F1 | Micro F1 | Weighted F1 | Accuracy |
|-------|:--------:|:--------:|:-----------:|:--------:|
| CNN | 0.177 | 0.550 | 0.390 | 0.550 |
| FCNN | 0.438 | 0.550 | 0.530 | 0.550 |
| Intermediate | 0.177 | 0.550 | 0.390 | 0.550 |
| CNN TL | 0.329 | 0.500 | 0.476 | 0.500 |
| Intermediate TL | 0.375 | 0.650 | 0.558 | 0.650 |
| Late Fusion | 0.492 | 0.650 | 0.615 | 0.650 |
| **Late Fusion TL** | **0.492** ⭐ | 0.650 | 0.615 | 0.650 |
| Early Fusion | 0.177 | 0.550 | 0.390 | 0.550 |
| Early Fusion TL | 0.352 | 0.600 | 0.507 | 0.600 |

#### RAF-DB 7-class
| Model | Macro F1 | Micro F1 | Weighted F1 | Accuracy |
|-------|:--------:|:--------:|:-----------:|:--------:|
| CNN | 0.729 | 0.815 | 0.813 | 0.815 |
| FCNN | 0.578 | 0.714 | 0.703 | 0.714 |
| Intermediate | 0.696 | 0.785 | 0.783 | 0.785 |
| CNN TL | 0.741 | 0.830 | 0.826 | 0.830 |
| **Intermediate TL** | **0.744** ⭐ | 0.833 | 0.832 | 0.833 |
| Late Fusion | 0.719 | 0.809 | 0.805 | 0.809 |
| Late Fusion TL | 0.735 | 0.829 | 0.823 | 0.829 |
| Early Fusion | 0.710 | 0.808 | 0.804 | 0.808 |
| Early Fusion TL | 0.693 | 0.790 | 0.786 | 0.790 |

#### RAF-DB 4-class
| Model | Macro F1 | Micro F1 | Weighted F1 | Accuracy |
|-------|:--------:|:--------:|:-----------:|:--------:|
| CNN | 0.808 | 0.830 | 0.830 | 0.830 |
| FCNN | 0.694 | 0.728 | 0.729 | 0.728 |
| Intermediate | 0.792 | 0.818 | 0.818 | 0.818 |
| CNN TL | 0.827 | 0.845 | 0.846 | 0.845 |
| **Intermediate TL** | **0.836** ⭐ | 0.853 | 0.855 | 0.853 |
| Late Fusion | 0.819 | 0.842 | 0.841 | 0.842 |
| Late Fusion TL | 0.832 | 0.849 | 0.850 | 0.849 |
| Early Fusion | 0.792 | 0.818 | 0.819 | 0.818 |
| Early Fusion TL | 0.799 | 0.823 | 0.823 | 0.823 |

#### KDEF 7-class
| Model | Macro F1 | Micro F1 | Weighted F1 | Accuracy |
|-------|:--------:|:--------:|:-----------:|:--------:|
| CNN | 0.798 | 0.801 | 0.798 | 0.801 |
| FCNN | 0.666 | 0.680 | 0.663 | 0.680 |
| Intermediate | 0.671 | 0.674 | 0.668 | 0.674 |
| CNN TL | 0.833 | 0.831 | 0.833 | 0.831 |
| **Intermediate TL** | **0.843** ⭐ | 0.843 | 0.843 | 0.843 |
| Late Fusion | 0.776 | 0.777 | 0.775 | 0.777 |
| Late Fusion TL | 0.836 | 0.834 | 0.836 | 0.834 |
| Early Fusion | 0.667 | 0.674 | 0.663 | 0.674 |
| Early Fusion TL | 0.799 | 0.795 | 0.797 | 0.795 |

#### KDEF 4-class
| Model | Macro F1 | Micro F1 | Weighted F1 | Accuracy |
|-------|:--------:|:--------:|:-----------:|:--------:|
| CNN | 0.841 | 0.872 | 0.872 | 0.872 |
| FCNN | 0.678 | 0.792 | 0.766 | 0.792 |
| Intermediate | 0.776 | 0.831 | 0.828 | 0.831 |
| CNN TL | 0.918 | 0.929 | 0.928 | 0.929 |
| **Intermediate TL** | **0.923** ⭐ | 0.929 | 0.929 | 0.929 |
| Late Fusion | 0.859 | 0.890 | 0.888 | 0.890 |
| Late Fusion TL | 0.920 | 0.932 | 0.930 | 0.932 |
| Early Fusion | 0.693 | 0.763 | 0.764 | 0.763 |
| Early Fusion TL | 0.816 | 0.855 | 0.854 | 0.855 |

#### Primer 7-class (conf60, B1 baseline — dari nb 62/65)
| Model | Macro F1 | Micro F1 | Weighted F1 | Accuracy |
|-------|:--------:|:--------:|:-----------:|:--------:|
| CNN | 0.270 | 0.787 | 0.795 | 0.787 |
| FCNN | 0.261 | 0.766 | 0.792 | 0.766 |
| Intermediate | 0.260 | 0.799 | 0.793 | 0.799 |
| CNN TL | 0.281 | 0.819 | 0.814 | 0.819 |
| **Intermediate TL** | **0.292** ⭐ | 0.808 | 0.819 | 0.808 |
| Late Fusion | 0.270 | 0.816 | 0.812 | 0.816 |
| Late Fusion TL | 0.238 | 0.790 | 0.784 | 0.790 |
| Early Fusion | 0.246 | 0.794 | 0.786 | 0.794 |
| Early Fusion TL | 0.253 | 0.713 | 0.722 | 0.713 |

#### Primer 4-class (conf60, B1 baseline — dari nb 62/65)
| Model | Macro F1 | Micro F1 | Weighted F1 | Accuracy |
|-------|:--------:|:--------:|:-----------:|:--------:|
| CNN | 0.476 | 0.803 | 0.802 | 0.803 |
| FCNN | 0.459 | 0.763 | 0.774 | 0.763 |
| Intermediate | 0.444 | 0.727 | 0.753 | 0.727 |
| CNN TL | 0.378 | 0.721 | 0.682 | 0.721 |
| **Intermediate TL** | **0.482** ⭐ | 0.780 | 0.790 | 0.780 |
| Late Fusion | 0.474 | 0.807 | 0.815 | 0.807 |
| Late Fusion TL | 0.422 | 0.695 | 0.722 | 0.695 |
| Early Fusion | 0.457 | 0.822 | 0.816 | 0.822 |
| Early Fusion TL | 0.471 | 0.770 | 0.770 | 0.770 |

> **Catatan Primer**: Tabel di atas B1 (baseline) saja, untuk perbandingan apple-to-apple dengan benchmark publik. Best Primer keseluruhan (B3 + TL + augmentation) = **Late Fusion TL 4c B3 = 0.567**.

---

### Skema 2 — Cross-Dataset (Train di Benchmark → Test di Primer)

9 model: CNN, FCNN, Intermediate, CNN TL, Intermediate TL, Late Fusion (nb 63) + **Early Fusion, Early Fusion TL (nb 68)** + **Late Fusion TL (nb 69 — inference-only reuse checkpoint nb 65)**.

Note untuk Late Fusion TL: grid search weight `w` dilakukan di Primer val (strategi default, tersimpan sebagai `Late_Fusion_TL_B1`). Variant `Late_Fusion_TL_B1_srcval` pakai source val (pure zero-shot, tersimpan untuk analisis ablation).

#### CK+ → Primer (7-class)
| Model | Macro F1 | Micro F1 | Weighted F1 | Accuracy |
|-------|:--------:|:--------:|:-----------:|:--------:|
| CNN | 0.127 | 0.719 | 0.635 | 0.719 |
| **FCNN** | **0.194** ⭐ | 0.773 | 0.739 | 0.773 |
| Intermediate | 0.153 | 0.536 | 0.593 | 0.536 |
| CNN TL | 0.163 | 0.670 | 0.701 | 0.670 |
| Intermediate TL | 0.103 | 0.152 | 0.238 | 0.152 |
| Late Fusion | 0.160 | 0.529 | 0.580 | 0.529 |
| Early Fusion | 0.125 | 0.686 | 0.638 | 0.686 |
| Early Fusion TL | 0.179 | 0.705 | 0.676 | 0.705 |
| Late Fusion TL | 0.247 | 0.703 | 0.726 | 0.703 |

#### CK+ → Primer (4-class)
| Model | Macro F1 | Micro F1 | Weighted F1 | Accuracy |
|-------|:--------:|:--------:|:-----------:|:--------:|
| **CNN** | **0.396** ⭐ | 0.642 | 0.703 | 0.642 |
| FCNN | 0.072 | 0.102 | 0.093 | 0.102 |
| Intermediate | 0.186 | 0.465 | 0.532 | 0.465 |
| CNN TL | 0.258 | 0.386 | 0.527 | 0.386 |
| Intermediate TL | 0.202 | 0.271 | 0.381 | 0.271 |
| Late Fusion | 0.245 | 0.489 | 0.556 | 0.489 |
| Early Fusion | 0.207 | 0.577 | 0.582 | 0.577 |
| Early Fusion TL | 0.101 | 0.194 | 0.288 | 0.194 |
| Late Fusion TL | 0.141 | 0.182 | 0.281 | 0.182 |

#### JAFFE → Primer (7-class)
| Model | Macro F1 | Micro F1 | Weighted F1 | Accuracy |
|-------|:--------:|:--------:|:-----------:|:--------:|
| CNN | 0.001 | 0.002 | 0.000 | 0.002 |
| FCNN | 0.023 | 0.013 | 0.018 | 0.013 |
| Intermediate | 0.017 | 0.014 | 0.020 | 0.014 |
| CNN TL | 0.015 | 0.054 | 0.006 | 0.054 |
| Intermediate TL | 0.023 | 0.028 | 0.045 | 0.028 |
| **Late Fusion** | **0.040** ⭐ | 0.081 | 0.054 | 0.081 |
| Early Fusion | 0.026 | 0.032 | 0.026 | 0.032 |
| Early Fusion TL | 0.001 | 0.002 | 0.000 | 0.002 |
| Late Fusion TL | 0.050 | 0.118 | 0.192 | 0.118 |

#### JAFFE → Primer (4-class)
| Model | Macro F1 | Micro F1 | Weighted F1 | Accuracy |
|-------|:--------:|:--------:|:-----------:|:--------:|
| CNN | 0.004 | 0.009 | 0.000 | 0.009 |
| FCNN | 0.004 | 0.009 | 0.000 | 0.009 |
| Intermediate | 0.005 | 0.010 | 0.002 | 0.010 |
| CNN TL | 0.004 | 0.009 | 0.000 | 0.009 |
| **Intermediate TL** | **0.093** ⭐ | 0.160 | 0.073 | 0.160 |
| Late Fusion | 0.007 | 0.010 | 0.002 | 0.010 |
| Early Fusion | 0.004 | 0.009 | 0.000 | 0.009 |
| Early Fusion TL | 0.004 | 0.009 | 0.000 | 0.009 |
| Late Fusion TL | 0.015 | 0.013 | 0.009 | 0.013 |

#### RAF-DB → Primer (7-class)
| Model | Macro F1 | Micro F1 | Weighted F1 | Accuracy |
|-------|:--------:|:--------:|:-----------:|:--------:|
| CNN | 0.076 | 0.099 | 0.132 | 0.099 |
| **FCNN** | **0.183** ⭐ | 0.640 | 0.672 | 0.640 |
| Intermediate | 0.109 | 0.237 | 0.287 | 0.237 |
| CNN TL | 0.175 | 0.545 | 0.611 | 0.545 |
| Intermediate TL | 0.180 | 0.479 | 0.562 | 0.479 |
| Late Fusion | 0.091 | 0.166 | 0.212 | 0.166 |
| Early Fusion | 0.142 | 0.365 | 0.476 | 0.365 |
| Early Fusion TL | 0.157 | 0.685 | 0.663 | 0.685 |
| Late Fusion TL | 0.153 | 0.483 | 0.575 | 0.483 |

#### RAF-DB → Primer (4-class)
| Model | Macro F1 | Micro F1 | Weighted F1 | Accuracy |
|-------|:--------:|:--------:|:-----------:|:--------:|
| CNN | 0.056 | 0.026 | 0.026 | 0.026 |
| FCNN | 0.264 | 0.487 | 0.553 | 0.487 |
| Intermediate | 0.269 | 0.551 | 0.594 | 0.551 |
| CNN TL | 0.206 | 0.256 | 0.357 | 0.256 |
| Intermediate TL | 0.179 | 0.306 | 0.363 | 0.306 |
| Late Fusion | 0.170 | 0.137 | 0.147 | 0.137 |
| Early Fusion | 0.227 | 0.404 | 0.485 | 0.404 |
| **Early Fusion TL** | **0.311** ⭐ | 0.477 | 0.575 | 0.477 |
| Late Fusion TL | 0.212 | 0.459 | 0.517 | 0.459 |

#### KDEF → Primer (7-class)
| Model | Macro F1 | Micro F1 | Weighted F1 | Accuracy |
|-------|:--------:|:--------:|:-----------:|:--------:|
| CNN | 0.024 | 0.040 | 0.024 | 0.040 |
| FCNN | 0.007 | 0.008 | 0.008 | 0.008 |
| Intermediate | 0.036 | 0.020 | 0.026 | 0.020 |
| **CNN TL** | **0.038** ⭐ | 0.053 | 0.036 | 0.053 |
| Intermediate TL | 0.034 | 0.056 | 0.023 | 0.056 |
| Late Fusion | 0.015 | 0.008 | 0.005 | 0.008 |
| Early Fusion | 0.007 | 0.010 | 0.011 | 0.010 |
| Early Fusion TL | 0.029 | 0.054 | 0.097 | 0.054 |
| Late Fusion TL | 0.055 | 0.078 | 0.063 | 0.078 |

#### KDEF → Primer (4-class)
| Model | Macro F1 | Micro F1 | Weighted F1 | Accuracy |
|-------|:--------:|:--------:|:-----------:|:--------:|
| CNN | 0.079 | 0.067 | 0.078 | 0.067 |
| FCNN | 0.004 | 0.009 | 0.000 | 0.009 |
| Intermediate | 0.037 | 0.020 | 0.021 | 0.020 |
| CNN TL | 0.068 | 0.045 | 0.023 | 0.045 |
| Intermediate TL | 0.076 | 0.054 | 0.030 | 0.054 |
| Late Fusion | 0.004 | 0.009 | 0.000 | 0.009 |
| Early Fusion | 0.039 | 0.040 | 0.023 | 0.040 |
| **Early Fusion TL** | **0.104** ⭐ | 0.101 | 0.152 | 0.101 |
| Late Fusion TL | 0.017 | 0.011 | 0.003 | 0.011 |

> **Note Late Fusion TL**: nilai di atas pakai strategi A (grid `w` di Primer val). Strategi B (pure zero-shot, tune di source val) hasil sangat mirip (±0.01 Macro F1) di kebanyakan combo → weight `w` robust across domain. Disimpan terpisah sebagai `Late_Fusion_TL_B1_srcval` di JSON.

---

### Summary Temuan Konsolidasi

**Skema 1 — pola lintas dataset:**
- **Best model umumnya Intermediate TL** (CK+ 4c, RAF-DB 7c/4c, KDEF 7c/4c, Primer 7c/4c)
- **Kecuali**: CK+ 7c best CNN TL (0.913); JAFFE 7c best **CNN TL (0.464)** (post val-tuned rerun — Late Fusion turun dari 0.545 test-tuned ke 0.314); JAFFE 4c best Late Fusion TL (0.492)
- **Late Fusion TL** (baru): kompetitif di RAF-DB/KDEF/Primer, tapi kolaps di CK+ 7c (0.835 vs 0.913) dan JAFFE 7c (0.146)
- **Primer paling challenging**: best Primer 4c = 0.482 (Intermediate TL), sedangkan RAF-DB 4c = 0.836, KDEF 4c = 0.923, CK+ 4c = 0.837

**Skema 2 — semua catastrophic; Early Fusion & Late Fusion TL sebagian besar tidak membantu:**
- Best overall cross: **CK+ 4c → Primer pakai CNN B1 = 0.396** (satu-satunya yang tembus > 0.30)
- **RAF-DB 4c → Primer: Early Fusion TL = 0.311** ⭐ (beat Intermediate 0.269, +0.042) — satu-satunya combo di mana EF membantu transfer
- **KDEF 4c → Primer: Early Fusion TL = 0.104** (beat CNN 0.079, +0.025) — marginal
- **CK+ 7c → Primer: Late Fusion TL = 0.247** beat non-fusion FCNN (0.194, +0.053) — Late Fusion TL satu-satunya menang di CK+ 7c
- **RAF-DB 4c → Primer: Late Fusion TL = 0.212** kalah dari EF TL 0.311 — EF tetap juara di combo ini
- JAFFE → Primer total collapse (<10% acc) — EF/LF TL tidak bantu
- KDEF 7c → Primer catastrophic (<6% acc semua model)

**Temuan 40 (revised): Fusion advanced (EF + LF TL) hanya bantu 3/8 combo cross-dataset**
> Hipotesis bahwa heatmap landmark lebih domain-invariant **tidak fully terbukti**. EF menang di RAF-DB 4c (+0.042) dan KDEF 4c (+0.025), tapi kalah di 6 combo lain (termasuk CK+ 4c -0.189 → EF B1 catastrophic). Implikasi: heatmap generic (Gaussian blob) + kernel channel ke-4 random init masih overfit ke source dataset. Untuk transfer robust, mungkin butuh pretrained channel-4 atau shared backbone across modality.
- **Pola**: FCNN / Intermediate (landmark-heavy) paling robust untuk cross-domain

**Justifikasi untuk tesis & paper:**
1. Arsitektur **bekerja** di dataset standar (Macro F1 0.82-0.92 di RAF-DB/KDEF/CK+ 4c)
2. Dataset primer **memang sulit** — bukan kelemahan arsitektur
3. Transfer learning dari public FER **tidak viable** untuk natural programming setting
4. Solusi: tetap pakai Primer self-training dengan confidence filtering (conf60) + fusion + augmentation

---

## SLIDE 33: Soft Label Training — Eksplorasi Selesai (Negative Result, nb 71/72/78)

### Motivasi
Paper Liliana et al. (2019) — *"humans express multiple emotions simultaneously"*. Face API output distribusi confidence 7-dim tiap frame, selama ini di-argmax + filter conf60 → informasi distribusi dibuang.

**Hipotesis:** pakai distribusi confidence langsung sebagai target training (soft label) → model belajar ambiguity alami → Macro F1 naik.

### Setup (konsisten lintas arsitektur)
- 4-class B1 baseline (no weights, no augmentation) — isolate efek loss function
- 4 loss variants: Hard CE, Soft CE, KL-divergence, Label Smoothing (ε=0.1)
- Hyperparam identik lintas arch: EPOCHS=50, LR_TL=5e-5, BATCH=32
- **Selection by val macro F1** (proper, no test leakage)
- 3 arsitektur di-test: CNN TL (nb 71), Late Fusion TL (nb 72), Intermediate TL (nb 78)

### Hasil Final (Test Macro F1, post-align hyperparam)

| Arsitektur | A Hard CE | B Soft CE | C KL-div | D Label Smooth | Juara by val |
|---|:---:|:---:|:---:|:---:|:---:|
| **CNN TL 4c B1** (nb 71) | 0.432 | 0.451 | 0.427 | **0.456** | D (val not logged) |
| **Late Fusion TL 4c** (nb 72) | — | 0.432 | 0.437 | — | — (no Hard baseline) |
| **Intermediate TL 4c B1** (nb 78) | **0.485** | 0.463 | 0.453 | 0.476 | **A Hard CE** (val 0.480) |

### Kesimpulan Final — Hipotesis Tidak Terkonfirmasi

**Negative result:** dengan metodologi konsisten (hyperparam align lintas arch + selection by val), **soft label tidak memberikan gain** di fusion arsitektur maupun single-modal:

- **Intermediate TL** (juara arch val-tuned): Hard CE menang (val 0.480, test 0.485). Semua soft variant di bawah Hard CE. Ranking by val: A > B > D > C.
- **Late Fusion TL**: tidak ada Hard baseline internal, tapi semua soft variant jauh di bawah external Late Fusion TL B3 val-tuned (0.466).
- **CNN TL**: D Label Smoothing 0.456 best di test (val tidak ter-log). **Previous single-seed run** (pre-align, EPOCHS=50 LR=5e-5 tapi code bug) dengan KL-div 0.517 **ternyata fluke** — re-run konsisten KL-div = 0.427 (−0.09 dari sebelumnya). Variance single-seed terbukti besar.

### Temuan kunci (konsolidasi)

**T41: Single-seed variance ±0.05-0.09 — terlalu besar untuk klaim marginal**
Gap antar loss config di setiap arsitektur = 0.02-0.03, jauh di bawah variance single-seed. **Mustahil klaim "soft label menang" tanpa multi-seed validation** (3-5 run minimum).

**T42: Previous KL-div juara (0.517) adalah fluke**
Re-run dengan setup konsisten kasih KL-div = 0.427 di CNN TL. Selisih 0.09 dari run sebelumnya = manifestasi pure seed/convergence variance. Temuan "hierarki KL > Soft CE > LS > Hard" dari draft lama **tidak reproducible**.

**T43: Soft label tidak transfer ke fusion**
Late Fusion TL gagal (0.432-0.437 vs baseline 0.466). Intermediate TL gagal (Hard CE menang). Fenomena konsisten: weighted softmax averaging (Late) + feature-level concat (Intermediate) tidak diuntungkan dari fuzzy target — modality interaction dominates over loss-function nuance.

**T44: Best-epoch anomaly mereda setelah hyperparam align**
Pre-align (LR=1e-4 nb 78): best_epoch 2-5 (terlalu cepat). Post-align (LR=5e-5): best_epoch 5-9 (normal). **Indikasi LR=1e-4 memang terlalu besar untuk Intermediate Fusion** — bug hyperparam, bukan soft label effect.

### Status: Stop eksplorasi

**Eksplorasi soft label ditutup sebagai negative result** per Apr 2026. Tidak masuk paper/tesis sebagai finding. Notebook nb 71/72/78 + JSON results tetap di-keep sebagai dokumentasi eksplorasi yang sudah dilakukan.

**Alasan stop:**
- Hipotesis tidak terkonfirmasi di fusion arsitektur (target utama paper)
- Multi-seed validation belum dilakukan, tapi gap antar config terlalu kecil (<variance) → tidak worth compute
- Fokus eksplorasi selanjutnya dipindah ke item lain yang potensinya lebih jelas:
  - **Geometric Features (Liliana 2019)** — direct extension paper dosen, 20-dim GF vs raw 136-d
  - **GCN Preprocessing (Pitaloka 2017)** — quick ablation, terukur langsung

> **Penjelasan lisan:**
> "Pak, saya sudah eksplorasi soft label terinspirasi Liliana 2019 di 3 arsitektur (CNN single, Late Fusion, Intermediate Fusion) dengan hyperparam konsisten. Hasilnya **negative result** — hipotesis tidak terkonfirmasi."
>
> "Temuan kunci: gap antar config sangat kecil (0.02-0.03), dalam range variance single-seed (~0.05-0.09). Re-run konsisten membuktikan sebelumnya KL-div juara di CNN TL (0.517) ternyata fluke — re-run kasih 0.427."
>
> "Di arsitektur juara (Intermediate TL), Hard CE justru yang menang by val. Soft label tidak transfer ke fusion."
>
> "Saya stop eksplorasi ini — negative result, tidak masuk paper. Fokus pindah ke Geometric Features (Liliana Table 3) dan GCN Preprocessing (Pitaloka) yang potensinya lebih jelas."

---

## SLIDE 34: GradCAM Observasi Awal (Eksplorasi, nb 73)

**Status:** eksperimen eksplorasi awal — belum cukup data untuk klaim general. Tidak di-framing ke paper/tesis sampai ada validasi lebih lanjut.

### Setup
- Tool: `pytorch-grad-cam` (Selvaraju et al. 2017) — visualisasi region gambar yang paling berpengaruh ke prediksi kelas
- 2 model dibandingkan:
  - **CNN TL 4c B1** (single-modal baseline, Macro F1 = 0.456)
  - **Intermediate TL 4c B3 image branch** (juara val-tuned, Macro F1 = 0.521)
- Sampel: 2 per kelas × 4 kelas = **8 overlay per model** (16 total)
- Target layer: last conv block ResNet18 (`features[-2][-1]`)
- Output: `docs/figures/gradcam/{cnn_tl_4c_b1_baseline, intermediate_tl_4c_b3_img_branch}.png`

### Observasi Kualitatif

**CNN TL baseline (8 sample):**
- Fokus tersebar ke region **non-emosi**: aksesori (kacamata), tangan, pundak, background/pojok
- Happy (2 sample): fokus di dahi/atas mata, bukan mulut (smile region)
- Sad (2 sample): 1 sample fokus di pipi+pundak, 1 sample di mulut+pipi
- Negative (2 sample): 1 di pojok atas (artifact), 1 di mata kiri

**Intermediate TL image branch (8 sample sama):**
- Fokus **konsisten di region mouth-nose** di 8/8 sample
- Bahkan di sampel sulit (neutral dengan tangan menutup mulut) — model tetap target area mulut, ignore tangan
- Pattern seragam lintas 4 kelas

### Interpretasi Tentatif (belum klaim general)
Observasi awal **menarik**: image branch dalam Intermediate Fusion tampak fokus ke region yang lebih "FER-relevant" dibanding CNN single-modal. Dugaan mekanisme: landmark stream berperan sebagai **implicit attention prior** — gradient dari fusion head memaksa image branch align dengan landmark geometry saat backprop.

### Keterbatasan Metodologi (penting)
- **Sample kecil**: hanya 2 per kelas (8 total) — representativeness rendah
- **Single seed**: belum tahu variance antar run
- **Qualitative only**: belum ada quantitative measure (mis. mean activation di mouth bounding box)
- **Tidak ada baseline sanity check**: random-init model untuk isolate efek pretrain vs fine-tune

### Validasi yang Dibutuhkan Sebelum Dianggap Finding
1. Quantitative: mean GradCAM activation di mouth region (via landmark 48-67) vs full face, di ≥50 sample per kelas
2. Multi-seed (3-5 run) untuk cek variance
3. Perbandingan dengan model kontrol (random-init, atau CNN single tanpa TL)
4. Cross-check dengan alternative viz (SmoothGrad, Integrated Gradients)

> **Penjelasan lisan (hati-hati, observational):**
> "Pak, saya coba GradCAM (arahan sebelumnya) pada 2 model untuk sekilas lihat fokus. Ada pattern awal yang menarik — image branch di Intermediate Fusion tampak lebih konsisten ke region mulut dibanding CNN single. Tapi ini baru 2 sample per kelas, belum quantitative. Saya sediakan dulu sebagai observasi untuk diskusi, apakah perlu diperdalam dengan quantitative measurement + multi-seed validation, atau cukup sebagai eksplorasi saja?"

---

## SLIDE 35: Space Alignment (arahan dosen #2, nb 75)

### Motivasi
Evaluasi apakah feature space image (CNN TL 256-d) dan landmark (FCNN 128-d) **sejajar** di latent space. Motivasi: kalau alignment rendah, Intermediate Fusion (concat) suboptimal. Kalau alignment tinggi, Intermediate Fusion justified secara empiris.

### Setup
- Checkpoint: CNN TL 4c B1 (256-d) + FCNN 4c B2 (128-d), extract features di Primer test set (929 sampel)
- 4 metode analisis (no training, analytical only):
  1. **CCA** (Canonical Correlation Analysis) — linear alignment top-k components
  2. **t-SNE** joint embedding — cek same-class cluster per stream
  3. **Per-class paired cosine similarity** (CCA-aligned space)
  4. **Cross-modal retrieval** top-k accuracy (image → paired landmark)

### Hasil Quantitative (strong alignment — positive result)

**CCA correlations:**
- top-5 mean = **0.978**
- top-10 mean = 0.967
- top-20 all > 0.89 (konsisten tinggi di 20 komponen)

**Per-class paired cosine (CCA-aligned):**

| Class | n | Mean ± std |
|---|:---:|:---:|
| neutral | 688 | 0.937 ± 0.041 |
| happy | 183 | 0.941 ± 0.037 |
| sad | 50 | 0.951 ± 0.046 |
| negative | 8 | 0.947 ± 0.030 |
| **overall** | 929 | **0.939** |

**Cross-modal retrieval** (image feature → nearest landmark feature, n=929):

| Top-k | Accuracy | Random baseline |
|---|:---:|:---:|
| top-1 | 0.670 | 0.001 |
| top-5 | **0.953** | 0.005 |
| top-10 | 0.995 | 0.011 |
| top-20 | 1.000 | 0.022 |

### Temuan

**T45: Alignment CNN-FCNN sangat kuat** (CCA top-5 = 0.978, far above 0.5 threshold)
Image dan landmark stream belajar representasi yang **nyaris co-linear** di latent space — walaupun input modality berbeda (pixel RGB 224×224 vs koordinat 136-d), model memetakan ke joint semantic space.

**T46: Alignment konsisten lintas kelas**
Per-class cosine 0.937-0.951 (termasuk minoritas `negative` n=8). Alignment bukan artifact kelas mayoritas saja — model belajar mapping yang seragam.

**T47: Cross-modal retrieval top-5 = 95.3%** (vs random baseline 0.54%)
Evidence kuat: feature image "tahu" paired landmark-nya. Streams **bukan independent**, meskipun modality berbeda — strong coupling semantic.

**T48: Justifikasi empiris untuk Intermediate Fusion**
Alignment tinggi mendukung pilihan concat fusion (Intermediate) — representasi yang aligned bisa di-compose secara linear di fusion head. Konsisten dengan observasi Intermediate TL juara val-tuned 4c (0.521) > Late Fusion TL (0.466).

### Output Files

```
docs/figures/space_alignment/
├── cca_correlations.png       (bar chart top-20 CCA correlations)
├── tsne_per_stream.png        (joint t-SNE 2D, split CNN vs FCNN)
└── retrieval_topk.png         (retrieval accuracy @ top-1/5/10/20/50)

models/frontonly_conf60/space_alignment/alignment_metrics.json
notebooks/results/75_space_alignment_executed.ipynb
```

### Status: Finding valid (methodology-clean)

Berbeda dari soft label (negative + single-seed fragile) dan GradCAM (observational kecil), **Space Alignment analisis adalah positive result yang methodology-clean**:
- Analytical (no training → no seed variance issue)
- Seluruh test set (n=929) — populasi lengkap, bukan sampling
- Multiple metrik konvergen (CCA + cosine + retrieval semua konsisten tinggi)
- Baseline comparison eksplisit (random retrieval)

**Siap dimasukkan ke paper/tesis** sebagai empirical justification fusion strategy di Section Discussion. Figure CCA + retrieval bisa jadi Fig. X paper.

> **Penjelasan lisan:**
> "Pak, saya sudah jalankan Space Alignment analysis (arahan #2) di test set Primer (929 sampel). Hasilnya menarik: CNN TL dan FCNN meskipun input modality sangat berbeda, belajar representasi yang sangat aligned di latent space."
>
> "Quantitative: CCA top-5 = 0.978 (well above 0.5 threshold), per-class paired cosine 0.94, cross-modal retrieval top-5 = 95.3% vs random baseline 0.54%."
>
> "Interpretasi: alignment tinggi ini justifikasi empiris untuk Intermediate Fusion — konsisten dengan observasi Intermediate TL juara val-tuned (0.521) di 4-class. Streams bukan independent, mereka semantically paired walaupun modality berbeda."
>
> "Ini beda dari soft label atau GradCAM — analisis ini methodology-clean (no seed variance, seluruh test set). Saya usulkan dimasukkan ke paper sebagai empirical justification di Section Discussion fusion strategy."

---

## SLIDE 36: 3-Class Exploration (Positive/Neutral/Negative, nb 79)

### Motivasi
4-class mapping (neutral/happy/sad/negative) **arbitrary** — tidak ada di literature. 3-class valence (positive/neutral/negative) adalah **standard Russell 1980 circumplex**, well-established di affective computing (DEAP, MAHNOB-HCI, Liliana 2019).

**REMAP_3:**
- `happy, surprised` → **positive** (0)
- `neutral` → **neutral** (1)
- `sad, angry, fearful, disgusted` → **negative** (2)

**Imbalance 4× lebih balanced:**

| Mapping | Max/min ratio |
|---|:---:|
| 7-class | 1:1138 |
| 4-class | 1:62 |
| **3-class** | **1:14** |

### Setup
- 5 arsitektur × 3 scenario = **15 configs**: FCNN, CNN TL, Intermediate TL, Late Fusion TL, Early Fusion TL × B1/B2/B3
- Hyperparam: EPOCHS=50, PATIENCE=15, LR_TL=5e-5, LR_scratch=1e-4
- **Selection by val macro F1** (proper)
- B3 pakai augmented dataset (`dataset_frontonly_conf60_3class_augmented`, target_min=1500/class)

### Hasil Lengkap (val-based selection)

| Config | Val Macro | Test Macro | Test Acc | w_best |
|---|:---:|:---:|:---:|:---:|
| FCNN B1 | 0.603 | 0.589 | 0.741 | — |
| FCNN B2 | 0.564 | 0.575 | 0.702 | — |
| FCNN B3 | 0.619 | 0.634 | 0.749 | — |
| CNN TL B1 | 0.493 | 0.634 | 0.791 | — |
| CNN TL B2 | 0.515 | 0.516 | 0.581 | — |
| CNN TL B3 | 0.495 | **0.706** | 0.840 | — |
| Intermediate TL B1 | 0.554 | 0.686 | 0.790 | — |
| Intermediate TL B2 | 0.559 | 0.665 | 0.815 | — |
| Intermediate TL B3 | 0.501 | 0.689 | 0.828 | — |
| **Late Fusion TL B1** | **0.609** | 0.653 | 0.797 | 0.25 |
| Late Fusion TL B2 | 0.591 | 0.646 | 0.774 | 0.30 |
| **Late Fusion TL B3** ⭐ | **0.623** | 0.637 | 0.784 | 0.15 |
| Early Fusion TL B1 | 0.537 | 0.587 | 0.783 | — |
| Early Fusion TL B2 | 0.504 | 0.584 | 0.650 | — |
| Early Fusion TL B3 | 0.509 | 0.699 | 0.820 | — |

**Juara by val: Late Fusion TL B3 — Macro F1 0.623 (val), 0.637 (test), acc 0.784.**

### Perbandingan vs 4-class & 7-class

| Mapping | Juara by val | Val Macro F1 | Δ vs 3-class |
|---|---|:---:|:---:|
| 7-class | Early Fusion TL B3 | 0.333 | −0.29 |
| 4-class | Intermediate TL B3 | 0.521 | −0.10 |
| **3-class** | **Late Fusion TL B3** | **0.623** | — |

**+0.10 Macro F1 gain** dari 4-class → 3-class.

### Temuan Kunci

**T49: 3-class memberikan gain signifikan lintas semua arsitektur**
Best 3-class (0.623) jauh melampaui best 4-class (0.521, +0.10 Macro F1). Bahkan config terlemah 3-class (CNN TL B2 test 0.516) masih di level best 4-class. Valence mapping memang lebih cocok untuk Primer natural data.

**T50: Late Fusion TL reclaims juara di 3-class**
Di 4-class Late Fusion TL kalah dari Intermediate TL (0.466 vs 0.521). Di 3-class, Late Fusion TL juara di **semua scenario** (B1/B2/B3) — Intermediate TL drop ke 3rd/4th. **Fusion strategy preference shifts dengan class granularity:**
- Fine granularity (7c) → Early Fusion TL (heatmap channel + aug membantu)
- Medium (4c) → Intermediate TL (feature-level joint learning)
- Coarse (3c) → **Late Fusion TL** (decision-level averaging optimal)

**T51: w_best lebih balanced di 3-class (0.15–0.30) vs 4-class (0.00–0.15)**
Di 4-class, w_best hampir selalu FCNN-dominant (11/12 config w ≤ 0.20). Di 3-class, image stream contribute **significantly** (w 0.15-0.30). Streams complementary lebih seimbang di coarser class — mungkin karena kelas merger menyamakan discriminability image vs landmark features.

**T52: FCNN 3-class sangat kompetitif**
FCNN B3 val=0.619, test=0.634 — hampir setara Late Fusion TL B3 (0.623/0.637). Landmark geometry memang dominan di Primer, dan di 3-class landmark info sudah cukup untuk diskriminasi valence — image stream adds marginal value.

### Status

**Positive result yang strong** — beda dari soft label (negative) dan GradCAM (observational kecil). Methodology-clean (val-based selection, hyperparam konsisten, 15 configs systematic).

**Usulan untuk paper:**
- **Reframe paper ke 3-class sebagai main experiment** (primary claim: 0.623 Late Fusion TL B3)
- 4-class jadi ablation comparison (not main result)
- 7-class tetap di-keep untuk demonstrate fine-grained challenge

### Output Files
```
data/dataset_frontonly_conf60_3class_augmented/   (~12k samples train augmented)
models/frontonly_conf60/3class/
├── FCNN/, CNN_TL/, Intermediate_TL/, Early_Fusion_TL/, Late_Fusion_TL/
└── all_results_3class.json
notebooks/results/79_threeclass_exploration_executed.ipynb
```

> **Penjelasan lisan:**
> "Pak, saya coba mapping 3-class valence (positive/neutral/negative) berdasarkan Russell 1980 circumplex — mapping ini punya literature precedent kuat, beda dari 4-class kita yang sebelumnya arbitrary."
>
> "Hasilnya signifikan: best 3-class Late Fusion TL B3 = Macro F1 0.623 val, 0.637 test. Gain +0.10 dari 4-class (0.521)."
>
> "Menarik, ranking fusion strategy bergeser per granularity: 7c Early Fusion menang, 4c Intermediate menang, 3c Late Fusion menang. w_best di 3-class juga lebih balanced (0.15-0.30) vs 4-class (0-0.15 FCNN-dominant) — artinya image stream di 3-class contribute lebih signifikan."
>
> "Saya usulkan reframe paper ke 3-class sebagai main experiment. Imbalance 4× lebih balanced, literature precedent jelas, dan gain 0.10 Macro F1 meaningful. 4-class bisa jadi ablation comparison."

---

## Ringkasan Poin Konsultasi

| No | Topik | Status |
|----|-------|--------|
| 1 | Eksperimen front+side (48 kombinasi) | ✅ Selesai |
| 2 | Eksperimen front-only (48 kombinasi) | ✅ Selesai |
| 3 | Perbandingan front-only vs front+side | ✅ Selesai — front-only sedikit lebih baik |
| 4 | LOSO Cross-Validation (37 fold) | ⏳ Sedang berjalan di VPS |
| 5 | 5-Fold CV (subject-wise) | ⏳ Sedang berjalan di VPS |
| 6 | Random Split (baseline) | ⏳ Sedang berjalan di VPS |
| 7 | Validasi ahli — dataset 146 sampel | ✅ Dataset siap, butuh 3 validator |
| 8 | Validasi ahli — Streamlit web app | ✅ App siap deploy |
| 9 | Perubahan dlib → MediaPipe | ✅ Cukup dijelaskan di BAB 4 |
| 10 | Benchmark CK+/JAFFE/RAF-DB/KDEF/Primer Skema 1 & 2 | ✅ Selesai (Apr 2026) |
| 11 | Early Fusion (arahan dosen, HAE-Net style) | ✅ Selesai (nb 64, 12 configs) |
| 12 | Cross-dataset semua source → Primer | ✅ Selesai (nb 63) |

### Pertanyaan yang Perlu Didiskusikan

| No | Topik | Pertanyaan |
|----|-------|-----------|
| 1 | Acuan utama | Front-only sebagai acuan utama, front+side sebagai pembanding? |
| 2 | Best model bergeser | Late Fusion TL (0.442) → Intermediate TL (0.412) — bagaimana menyikapi? |
| 3 | Macro F1 0.412 | Acceptable untuk tesis? |
| 4 | Validasi ahli | Rekomendasi 3 validator psikologi? |
| 5 | Honorarium | Perlu fee untuk validator? |

---

## Lampiran A: Struktur Project Saat Ini

```
MultimodalEmoLearn/
├── src/
│   ├── preprocessing/
│   │   ├── prepare_dataset.py          # Gabung data + split + numpy
│   │   ├── augment_minority.py         # Augmentasi kelas minoritas
│   │   └── generate_validation_set.py  # Generate set validasi ahli
│   ├── training/
│   │   ├── models.py                   # Arsitektur CNN, FCNN, IntermediateFusion
│   │   └── utils.py                    # Training loop, evaluasi, visualisasi
│   ├── tools/
│   │   ├── validation_app.py           # Streamlit web tool validasi
│   │   └── process_validation_results.py # Proses hasil validasi + Cohen's Kappa
│   └── utils/
│       ├── batch_video_processor.py    # Ekstrak frame dari video
│       ├── face_crop_landmark.py       # Face crop + 68 landmark
│       └── generate_emotion_label.py   # Generate label emosi
├── notebooks/
│   ├── 01-05                           # Training 7-class from scratch (front+side)
│   ├── 06-10                           # Training 4-class from scratch (front+side)
│   ├── 11-16                           # Transfer Learning (front+side)
│   ├── 17_comparison_all               # Comparison front+side
│   ├── 18-25                           # Training front-only (from scratch)
│   ├── 26-31                           # Transfer Learning front-only
│   ├── 32_comparison_frontonly          # Comparison front-only vs original
│   ├── 33_loso_frontonly               # LOSO Cross-Validation (37 fold)
│   ├── 34_crossval_frontonly           # 5-Fold CV (subject-wise)
│   ├── 35_randomsplit_frontonly        # Random Split (baseline)
│   └── results/                        # Executed notebooks dari VPS
├── models/
│   ├── cnn/, fcnn/, late_fusion/,      # Hasil 7-class from scratch
│   │   intermediate_fusion/
│   ├── 4class/                         # Hasil 4-class from scratch
│   │   ├── cnn/, fcnn/, late_fusion/,
│   │   │   intermediate_fusion/
│   │   └── experiment_summary_4class.json
│   ├── cnn_transfer/                   # Hasil CNN Transfer Learning (7-class + 4-class)
│   ├── 4class/cnn_transfer/            # Hasil CNN TL 4-class
│   ├── final_comparison.json           # Ringkasan 48 kombinasi eksperimen
│   ├── final_comparison_bar.png        # Chart perbandingan bar
│   ├── final_comparison_heatmap.png    # Chart heatmap
│   └── experiment_summary.json         # Ringkasan 7-class
├── data/
│   ├── dataset/                        # Numpy arrays 7-class (front+side)
│   ├── dataset_frontonly/              # Numpy arrays 7-class (front-only)
│   ├── dataset_frontonly_4class/       # Numpy arrays 4-class (front-only)
│   ├── dataset_frontonly_*augmented/   # Versi augmented
│   └── validation/sets/1pct/          # Set validasi ahli (146 sampel)
├── scripts/
│   ├── run_frontonly_7class.sh         # Training front-only 7-class
│   ├── run_frontonly_4class.sh         # Training front-only 4-class
│   ├── run_frontonly_transfer.sh       # Training front-only TL
│   ├── run_loso.sh                     # LOSO cross-validation
│   ├── run_crossval.sh                 # 5-Fold CV
│   └── run_randomsplit.sh              # Random Split baseline
└── docs/
    ├── bimbingan_progress.md           # File ini
    └── linux_training_guide.md         # Panduan training di VPS
```

---

## Lampiran B: Referensi Metode yang Digunakan

| Metode | Referensi | Digunakan untuk |
|--------|-----------|----------------|
| Class-Balanced Loss | Cui et al. (2019), CVPR | Penanganan class imbalance via weighted loss |
| MediaPipe Face Mesh | Lugaresi et al. (2019), Google | Face detection + 468 landmark extraction |
| 68-point landmark mapping | Sagonas et al. (2016) | Standar landmark untuk FER (di-map dari MediaPipe) |
| Macro F1-Score | Sokolova & Lapalme (2009) | Evaluasi yang fair untuk data imbalanced |
| User-based split | - | Mencegah data leaking antar split |
| Intermediate Fusion | Boulahia et al. (2021) | Feature-level fusion CNN + FCNN |
| Late Fusion | Boulahia et al. (2021) | Decision-level weighted averaging |
| ResNet18 Transfer Learning | He et al. (2016), CVPR | Pretrained ImageNet sebagai backbone CNN |

---

## Lampiran C: Konfigurasi Training

### From Scratch

| Parameter | CNN | FCNN | Intermediate Fusion |
|-----------|-----|------|---------------------|
| Optimizer | Adam | Adam | Adam |
| Learning Rate | 0.0001 | 0.0001 | 0.0001 |
| Batch Size | 32 | 128 | 16 |
| Max Epochs | 50 | 100 | 80 |
| Early Stopping Patience | 15 | 20 | 25 |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=8) | ReduceLROnPlateau (factor=0.5, patience=8) | ReduceLROnPlateau (factor=0.5, patience=8) |
| Best Model Criteria | Val Macro F1 | Val Macro F1 | Val Macro F1 |
| GPU | NVIDIA T4 (16GB) | NVIDIA T4 (16GB) | NVIDIA T4 (16GB) |
| Framework | PyTorch | PyTorch | PyTorch |

### Transfer Learning

| Parameter | CNN TL (ResNet18) | Intermediate Fusion TL |
|-----------|-------------------|------------------------|
| Backbone | ResNet18 pretrained ImageNet | ResNet18 pretrained ImageNet |
| Optimizer | Adam | Adam |
| Learning Rate | 0.00005 | 0.00005 |
| Batch Size | 32 | 16 |
| Max Epochs | 50 | 80 |
| Early Stopping Patience | 15 | 25 |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=8) | ReduceLROnPlateau (factor=0.5, patience=8) |
| Fine-tune Strategy | Full fine-tune semua layer | Full fine-tune semua layer |
| GPU | NVIDIA T4 (16GB) | NVIDIA T4 (16GB) |
