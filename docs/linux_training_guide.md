# Panduan Training di Linux (NVIDIA T4 - Biznet Gio VPS)

> **Server:** Biznet Gio VPS dengan NVIDIA T4
> **Akses:** MobaXterm (SSH + file transfer)

## 1. Transfer Project ke VPS

### Option A: Git clone di VPS (RECOMMENDED)

Buka MobaXterm → SSH ke VPS → jalankan:
```bash
git clone https://github.com/taufikhdyt01/MultimodalEmoLearn.git
cd MultimodalEmoLearn
```

### Option B: Upload via MobaXterm file browser

MobaXterm punya panel file browser di sebelah kiri saat SSH session aktif.
Cukup **drag & drop** folder project dari Windows ke panel tersebut.

## 2. Transfer Data ke VPS

Data tidak ada di git (di-gitignore, terlalu besar ~10GB). Harus transfer manual.

### Langkah 1: Compress data di Windows

Buka Git Bash / terminal di folder project:
```bash
cd D:/MultimodalEmoLearn
tar -czf dataset.tar.gz data/dataset data/dataset_augmented
# Hasilnya: dataset.tar.gz (~3-4 GB setelah compress)
```

### Langkah 2: Upload ke VPS

**Cara A: Drag & drop via MobaXterm (paling mudah)**
1. Buka MobaXterm → SSH ke VPS
2. Di panel kiri (file browser), navigasi ke `/home/USER/MultimodalEmoLearn/`
3. Drag file `dataset.tar.gz` dari Windows Explorer ke panel kiri MobaXterm
4. Tunggu upload selesai

**Cara B: Via terminal MobaXterm**

MobaXterm punya terminal lokal bawaan yang sudah support SCP:
```bash
scp D:/MultimodalEmoLearn/dataset.tar.gz USER@IP_VPS:/home/USER/MultimodalEmoLearn/
```

> **Tips:** Upload ~3-4 GB bisa memakan waktu tergantung kecepatan internet.

### Langkah 3: Extract di VPS

Di terminal SSH MobaXterm:
```bash
cd MultimodalEmoLearn
tar -xzf dataset.tar.gz

# Verifikasi
ls data/dataset/*.npy | wc -l           # harus 9 files
ls data/dataset_augmented/*.npy | wc -l  # harus 9 files

# Hapus file compress (opsional, hemat disk)
rm dataset.tar.gz
```

### File yang dibutuhkan:
```
data/
├── dataset/
│   ├── X_train_images.npy      (~4.2 GB)
│   ├── X_train_landmarks.npy   (~3.8 MB)
│   ├── y_train.npy             (~28 KB)
│   ├── X_val_images.npy        (~0.7 GB)
│   ├── X_val_landmarks.npy     (~0.6 MB)
│   ├── y_val.npy               (~5 KB)
│   ├── X_test_images.npy       (~1.0 GB)
│   ├── X_test_landmarks.npy    (~0.9 MB)
│   ├── y_test.npy              (~7 KB)
│   ├── class_weights.json
│   ├── dataset_info.json
│   └── label_map.json
└── dataset_augmented/
    ├── X_train_images.npy      (~4.5 GB)
    ├── X_train_landmarks.npy   (~4.1 MB)
    ├── y_train.npy             (~30 KB)
    ├── class_weights.json
    └── ... (val/test di-copy dari dataset/)
```

## 3. Setup Environment di VPS

Semua perintah di bawah dijalankan di VPS (setelah `ssh USER@IP_VPS`):

```bash
# 1. Install Miniconda (jika belum ada)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Ikuti instruksi, jawab "yes" untuk init conda
source ~/.bashrc

# 2. Buat conda environment
conda create -n emotrain python=3.10 -y
conda activate emotrain

# 3. Install PyTorch dengan CUDA (untuk NVIDIA T4)
# Cek dulu versi CUDA di VPS:
nvidia-smi
# Lihat "CUDA Version" di pojok kanan atas output

# Untuk CUDA 12.x:
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Untuk CUDA 11.x:
# conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 4. Install dependencies lainnya
pip install numpy scikit-learn matplotlib seaborn jupyter openpyxl

# 5. Verifikasi GPU terdeteksi
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
# Expected output:
# CUDA: True
# GPU: Tesla T4
```

## 4. Verifikasi Setup

```bash
cd MultimodalEmoLearn

# Cek data ada
python -c "
import numpy as np
from pathlib import Path

for name in ['dataset', 'dataset_augmented']:
    d = Path(f'data/{name}')
    for f in sorted(d.glob('*.npy')):
        arr = np.load(f)
        print(f'{name}/{f.name}: shape={arr.shape}, dtype={arr.dtype}')
"

# Cek model bisa dibuat
python -c "
import sys; sys.path.insert(0, 'src')
from training.models import EmotionCNN, EmotionFCNN, IntermediateFusion
import torch

device = torch.device('cuda')
m1 = EmotionCNN(7).to(device)
m2 = EmotionFCNN(136, 7).to(device)
m3 = IntermediateFusion(7, 136).to(device)

# Test forward pass
img = torch.randn(2, 3, 224, 224).to(device)
lm = torch.randn(2, 136).to(device)

print('CNN:', m1(img).shape)          # [2, 7]
print('FCNN:', m2(lm).shape)          # [2, 7]
print('Fusion:', m3(img, lm).shape)   # [2, 7]
print('All models OK!')
"
```

## 5. Jalankan Training

### Opsi A: Notebook interaktif via MobaXterm (RECOMMENDED)

Jalankan Jupyter di VPS, akses dari browser di laptop:

```bash
# Di MobaXterm, buka SSH session ke VPS, lalu jalankan:
conda activate emotrain
cd MultimodalEmoLearn
jupyter notebook --no-browser --port=8888 --ip=0.0.0.0
```

MobaXterm otomatis membuat SSH tunnel. Buka browser di laptop: **http://localhost:8888**

> Jika port 8888 tidak ter-tunnel otomatis, buat manual:
> MobaXterm → menu Tunneling → New SSH tunnel → Local port 8888 → Remote port 8888

Jalankan notebook secara berurutan:
- **7-class:** `01` → `02` → `03` → `04` → `05`
- **4-class:** `06` → `07` → `08` → `09` → `10`

### Opsi B: Jalankan semua di background (RECOMMENDED kalau mau ditinggal)

Koneksi SSH bisa putus kapan saja tanpa mempengaruhi training:

```bash
# Di MobaXterm SSH session:

# Pakai tmux supaya proses tidak mati kalau koneksi putus
tmux new -s training

# Di dalam tmux:
conda activate emotrain
cd MultimodalEmoLearn
bash scripts/run_all.sh

# Setelah jalan, DETACH dari tmux:
# Tekan: Ctrl+B, lalu tekan D
# Sekarang bisa tutup MobaXterm, training tetap jalan di VPS

# Nanti kalau mau cek progress, buka MobaXterm lagi:
tmux attach -t training
```

### Opsi C: Jalankan satu-satu manual

```bash
conda activate emotrain
cd MultimodalEmoLearn

# Jalankan per notebook:
jupyter nbconvert --to notebook --execute notebooks/01_train_cnn.ipynb \
    --output 01_train_cnn_executed.ipynb --output-dir notebooks/results/ \
    --ExecutePreprocessor.timeout=7200
# Ulangi untuk 02-10
```

## 6. Estimasi Waktu Training 7-Class (NVIDIA T4)

| Notebook | Model | Estimasi per Skenario | Total (3 skenario) |
|----------|-------|-----------------------|---------------------|
| 01_train_cnn | CNN | ~15-30 menit | ~45-90 menit |
| 02_train_fcnn | FCNN | ~2-5 menit | ~6-15 menit |
| 03_late_fusion | Late Fusion | ~1-2 menit (inference only) | ~3-6 menit |
| 04_intermediate_fusion | Intermediate | ~20-40 menit | ~60-120 menit |
| 05_comparison | - | ~1 menit | ~1 menit |
| **Total** | | | **~2-4 jam** |

## 7. Setelah Training Selesai

### Transfer hasil dari VPS ke laptop Windows:

**Cara A: Drag & drop via MobaXterm**
1. Di MobaXterm SSH session, navigasi panel kiri ke `MultimodalEmoLearn/models/`
2. Select semua folder → drag ke Windows Explorer

**Cara B: Compress lalu download**

Di VPS:
```bash
cd MultimodalEmoLearn
tar -czf results.tar.gz models/ notebooks/results/
```

Lalu download `results.tar.gz` via panel kiri MobaXterm, atau dari Git Bash:
```bash
scp USER@IP_VPS:/home/USER/MultimodalEmoLearn/results.tar.gz D:/MultimodalEmoLearn/
cd D:/MultimodalEmoLearn
tar -xzf results.tar.gz
```

### File hasil yang dihasilkan:
```
models/
├── cnn/
│   ├── cnn_b1_baseline.pth
│   ├── cnn_b2_weighted.pth
│   ├── cnn_b3_augmented.pth
│   └── cnn_results.json
├── fcnn/
│   ├── fcnn_b1_baseline.pth
│   ├── fcnn_b2_weighted.pth
│   ├── fcnn_b3_augmented.pth
│   └── fcnn_results.json
├── late_fusion/
│   └── late_fusion_results.json
├── intermediate_fusion/
│   ├── intermediate_b1_baseline.pth
│   ├── intermediate_b2_weighted.pth
│   ├── intermediate_b3_augmented.pth
│   └── intermediate_fusion_results.json
└── experiment_summary.json
```

---

## 8. Menjalankan Eksperimen 4-Class (Lanjutan)

> **Prasyarat:** Notebook 01-05 (7-class) sudah selesai dijalankan.

Setelah hasil 7-class dianalisis, langkah selanjutnya adalah menjalankan eksperimen dengan 4 kelas emosi sebagai perbandingan:
- **neutral, happy, sad, negative** (angry+fearful+disgusted+surprised digabung)

### Langkah 1: Pull kode terbaru di VPS

```bash
ssh USER@IP_VPS   # atau buka MobaXterm
cd MultimodalEmoLearn
git pull origin master
```

### Langkah 2: Generate dataset 4-class

Dataset 4-class dibuat dari dataset 7-class yang sudah ada (hanya remap label, tidak perlu transfer data baru):

```bash
conda activate emotrain
python src/preprocessing/prepare_dataset_4class.py
```

### Langkah 3: Jalankan training 4-class

```bash
# Pakai tmux supaya aman kalau koneksi putus
tmux new -s training4class

conda activate emotrain
bash scripts/run_4class.sh

# Detach: Ctrl+B lalu D
# Re-attach: tmux attach -t training4class
```

Atau jalankan interaktif via Jupyter (notebook 06 → 07 → 08 → 09 → 10).

### Estimasi waktu 4-class (NVIDIA T4):

| Notebook | Model | Total (3 skenario) |
|----------|-------|---------------------|
| 06_train_cnn_4class | CNN | ~45-90 menit |
| 07_train_fcnn_4class | FCNN | ~6-15 menit |
| 08_late_fusion_4class | Late Fusion | ~3-6 menit |
| 09_intermediate_fusion_4class | Intermediate | ~60-120 menit |
| 10_comparison_4class | Comparison | ~1 menit |
| **Total** | | **~2-4 jam** |

### Langkah 4: Transfer hasil 4-class

```bash
# Di VPS:
cd MultimodalEmoLearn
tar -czf results_4class.tar.gz models/4class/ notebooks/results/
```

Download via MobaXterm (drag & drop) atau:
```bash
# Di laptop:
scp USER@IP_VPS:/home/USER/MultimodalEmoLearn/results_4class.tar.gz D:/MultimodalEmoLearn/
cd D:/MultimodalEmoLearn && tar -xzf results_4class.tar.gz
```

### File hasil 4-class:
```
models/4class/
├── cnn/
│   ├── cnn_4c_b1_baseline.pth
│   ├── cnn_4c_b2_weighted.pth
│   ├── cnn_4c_b3_augmented.pth
│   └── cnn_4class_results.json
├── fcnn/
│   ├── fcnn_4c_b1_baseline.pth
│   ├── fcnn_4c_b2_weighted.pth
│   ├── fcnn_4c_b3_augmented.pth
│   └── fcnn_4class_results.json
├── late_fusion/
│   └── late_fusion_4class_results.json
├── intermediate_fusion/
│   ├── intermediate_4c_b1_baseline.pth
│   ├── intermediate_4c_b2_weighted.pth
│   ├── intermediate_4c_b3_augmented.pth
│   └── intermediate_fusion_4class_results.json
└── experiment_summary_4class.json
```

---

## 9. Menjalankan Transfer Learning (Lanjutan)

> **Prasyarat:** Notebook 01-10 (from scratch) sudah selesai dijalankan.

Transfer Learning menggunakan ResNet18 pretrained ImageNet untuk menggantikan CNN from scratch.

### Langkah 1: Pull kode terbaru

```bash
cd MultimodalEmoLearn
git pull origin master
```

### Langkah 2: Jalankan

```bash
tmux new -s transfer

conda activate emotrain
bash scripts/run_transfer.sh

# Detach: Ctrl+B lalu D
```

Notebook yang dijalankan: `11` → `12` → `13` → `14` → `15` → `16` → `17`

### Estimasi waktu Transfer Learning (NVIDIA T4):

| Notebook | Model | Total (3 skenario) |
|----------|-------|---------------------|
| 11_train_cnn_transfer | CNN TL 7-class | ~30-60 menit |
| 12_late_fusion_transfer | Late Fusion TL 7-class | ~3-6 menit |
| 13_intermediate_fusion_transfer | Intermediate TL 7-class | ~40-80 menit |
| 14_train_cnn_transfer_4class | CNN TL 4-class | ~30-60 menit |
| 15_late_fusion_transfer_4class | Late Fusion TL 4-class | ~3-6 menit |
| 16_intermediate_fusion_transfer_4class | Intermediate TL 4-class | ~40-80 menit |
| 17_comparison_all | Final comparison | ~1 menit |
| **Total** | | **~2.5-5 jam** |

---

## 10. Menjalankan Eksperimen Front-Only (Lanjutan)

> **Prasyarat:** Notebook 01-17 (front+side) sudah selesai dijalankan.

Eksperimen front-only menggunakan **hanya data sudut depan** dari kedua batch, untuk konsistensi karena batch 1 hanya memiliki sudut depan sedangkan batch 2 memiliki depan+samping.

### Langkah 1: Pull kode terbaru

```bash
cd MultimodalEmoLearn
git pull origin master
```

### Langkah 2: Transfer dataset front-only ke VPS

Dataset front-only sudah di-generate di laptop. Cukup upload **base dataset saja** (~4 GB), sisanya di-generate di VPS.

**Di laptop (Git Bash / terminal):**
```bash
cd D:/MultimodalEmoLearn
tar -czf dataset_frontonly.tar.gz data/dataset_frontonly/
# Hasilnya: dataset_frontonly.tar.gz (~2-3 GB setelah compress)
```

**Upload ke VPS** via MobaXterm (drag & drop) atau:
```bash
scp dataset_frontonly.tar.gz USER@IP_VPS:/home/USER/MultimodalEmoLearn/
```

**Di VPS — extract dan generate dataset turunan:**
```bash
cd MultimodalEmoLearn
tar -xzf dataset_frontonly.tar.gz
rm dataset_frontonly.tar.gz  # hemat disk

# Verify
ls data/dataset_frontonly/*.npy | wc -l  # harus 9+ files

# Generate augmented + 4-class (dari base dataset, ~5 menit)
conda activate emotrain
python scripts/prepare_frontonly_all.py
```

Ini akan menghasilkan 4 dataset:
```
data/
├── dataset_frontonly/              # 7-class front-only (7,091 sampel) — dari upload
├── dataset_frontonly_augmented/    # 7-class + augmentasi kelas minoritas — di-generate
├── dataset_frontonly_4class/       # 4-class front-only — di-generate
└── dataset_frontonly_4class_augmented/  # 4-class + augmentasi — di-generate
```

Ini akan menghasilkan 4 dataset:
```
data/
├── dataset_frontonly/              # 7-class front-only (7,091 sampel)
├── dataset_frontonly_augmented/    # 7-class + augmentasi kelas minoritas
├── dataset_frontonly_4class/       # 4-class front-only
└── dataset_frontonly_4class_augmented/  # 4-class + augmentasi
```

### Langkah 3: Jalankan training front-only

```bash
tmux new -s frontonly

conda activate emotrain

# From scratch 7-class (notebook 18-21)
bash scripts/run_frontonly_7class.sh

# From scratch 4-class (notebook 22-25)
bash scripts/run_frontonly_4class.sh

# Transfer Learning (notebook 26-31)
bash scripts/run_frontonly_transfer.sh

# Comparison (notebook 32)
jupyter nbconvert --to notebook --execute notebooks/32_comparison_frontonly_vs_original.ipynb \
    --output 32_executed.ipynb --output-dir notebooks/results/ \
    --ExecutePreprocessor.timeout=7200

# Detach: Ctrl+B lalu D
```

Atau jalankan interaktif via Jupyter (notebook 18 → 19 → ... → 32).

### Notebook front-only:

| No | Notebook | Model | Kelas |
|----|----------|-------|-------|
| **From Scratch** |||
| 18 | `cnn_frontonly_7class` | CNN | 7 |
| 19 | `fcnn_frontonly_7class` | FCNN | 7 |
| 20 | `late_fusion_frontonly_7class` | Late Fusion | 7 |
| 21 | `intermediate_frontonly_7class` | Intermediate | 7 |
| 22 | `cnn_frontonly_4class` | CNN | 4 |
| 23 | `fcnn_frontonly_4class` | FCNN | 4 |
| 24 | `late_fusion_frontonly_4class` | Late Fusion | 4 |
| 25 | `intermediate_frontonly_4class` | Intermediate | 4 |
| **Transfer Learning** |||
| 26 | `cnn_tl_frontonly_7class` | CNN TL (ResNet18) | 7 |
| 27 | `late_fusion_tl_frontonly_7class` | Late Fusion TL | 7 |
| 28 | `intermediate_tl_frontonly_7class` | Intermediate TL | 7 |
| 29 | `cnn_tl_frontonly_4class` | CNN TL (ResNet18) | 4 |
| 30 | `late_fusion_tl_frontonly_4class` | Late Fusion TL | 4 |
| 31 | `intermediate_tl_frontonly_4class` | Intermediate TL | 4 |
| **Perbandingan** |||
| 32 | `comparison_frontonly_vs_original` | Semua | - |

### Estimasi waktu front-only (NVIDIA T4):

| Tahap | Notebook | Estimasi |
|-------|----------|----------|
| From scratch 7-class | 18-21 | ~2-4 jam |
| From scratch 4-class | 22-25 | ~2-4 jam |
| Transfer Learning | 26-31 | ~2.5-5 jam |
| Comparison | 32 | ~1 menit |
| **Total** | | **~6.5-13 jam** |

### Langkah 4: Transfer hasil

```bash
# Di VPS:
cd MultimodalEmoLearn
tar -czf results_frontonly.tar.gz models/frontonly/ notebooks/results/
```

Download via MobaXterm atau:
```bash
scp USER@IP_VPS:/home/USER/MultimodalEmoLearn/results_frontonly.tar.gz D:/MultimodalEmoLearn/
cd D:/MultimodalEmoLearn && tar -xzf results_frontonly.tar.gz
```

### File hasil front-only:
```
models/frontonly/
├── 7class/
│   ├── cnn_b1.pth, cnn_b2.pth, cnn_b3.pth
│   ├── fcnn_b1.pth, fcnn_b2.pth, fcnn_b3.pth
│   ├── intermediate_b1.pth, intermediate_b2.pth, intermediate_b3.pth
│   └── *_results.json
├── 4class/
│   └── (sama seperti 7class)
├── 7class_tl/
│   ├── cnn_tl_b1.pth, cnn_tl_b2.pth, cnn_tl_b3.pth
│   ├── fcnn_b1.pth, fcnn_b2.pth, fcnn_b3.pth
│   ├── intermediate_tl_b1.pth, ...
│   └── *_results.json
├── 4class_tl/
│   └── (sama seperti 7class_tl)
└── results_transfer_frontonly.json
```

---

## 11. LOSO Cross-Validation (Lanjutan)

LOSO (Leave-One-Subject-Out) mengevaluasi robustness model — setiap user jadi test set 1x.

### Prerequisite: Generate user_ids (jika belum ada)

```bash
# Jalankan di LAPTOP (butuh data/processed & data/final), lalu upload hasilnya
python scripts/generate_user_ids.py
# Upload file-file kecil ini ke VPS:
#   data/dataset_frontonly/user_ids_all.npy (~28 KB)
#   data/dataset_frontonly/y_all.npy (~28 KB)
#   data/dataset_frontonly/user_ids_train.npy
#   data/dataset_frontonly/user_ids_val.npy
#   data/dataset_frontonly/user_ids_test.npy
```

### Jalankan LOSO

```bash
tmux new -s loso
conda activate emotrain
cd MultimodalEmoLearn

# Jalankan via shell script (otomatis execute notebook 33)
bash scripts/run_loso.sh
```

**Estimasi:** ~8-15 jam untuk 3 model × 37 fold.

> **Status (10 Apr 2026):** Baru 1 model (intermediate_tl) yang selesai LOSO (34/37 fold).
> Late fusion dan FCNN belum dijalankan karena waktu terbatas.
> Hasil intermediate_tl sudah tersimpan di `models/frontonly/loso/loso_intermediate_tl_4class.json`.
> Untuk melanjutkan, jalankan ulang `bash scripts/run_loso.sh` — model yang sudah selesai akan di-overwrite,
> atau modifikasi notebook 33 agar skip model yang sudah ada.

### Output LOSO:
```
models/frontonly/loso/
├── loso_intermediate_tl_4class.json   # ✅ Selesai (34/37 fold)
├── loso_late_fusion_4class.json       # ❌ Belum dijalankan
├── loso_fcnn_4class.json              # ❌ Belum dijalankan
└── loso_comparison.png                # Bar chart per fold

notebooks/results/
└── 33_loso_frontonly_executed.ipynb    # Executed notebook dengan chart
```

---

## 12. 5-Fold Cross-Validation (Lanjutan)

5-Fold CV subject-wise — user dikelompokkan ke 5 grup, rotasi test set.
Lebih cepat dari LOSO (~2-4 jam vs ~8-15 jam).

```bash
tmux new -s crossval
conda activate emotrain
cd MultimodalEmoLearn

bash scripts/run_crossval.sh
```

### Output 5-Fold CV:
```
models/frontonly/crossval/
├── cv5_intermediate_tl_4class.json
├── cv5_late_fusion_4class.json
├── cv5_fcnn_4class.json
└── cv5_comparison.png

notebooks/results/
└── 34_crossval_frontonly_executed.ipynb
```

---

## 13. Random Split — Baseline Comparison (Lanjutan)

Random split sebagai baseline untuk menunjukkan efek data leakage.
Sampel diacak tanpa memperhatikan user → user yang sama bisa di train & test.

```bash
tmux new -s randomsplit
conda activate emotrain
cd MultimodalEmoLearn

bash scripts/run_randomsplit.sh
```

**Estimasi:** ~30-60 menit (5 repeats × 3 model).

### Output Random Split:
```
models/frontonly/randomsplit/
├── random_intermediate_tl_4class.json
├── random_late_fusion_4class.json
├── random_fcnn_4class.json
└── split_strategy_comparison.png      # Grouped bar chart semua strategi

notebooks/results/
└── 35_randomsplit_frontonly_executed.ipynb
```

### Ringkasan Semua Strategi Split:

| No | Notebook | Strategi | Fold/Repeat | Estimasi |
|----|----------|----------|:-----------:|----------|
| 33 | `loso_frontonly` | LOSO (37 fold) | 37 × 3 model | ~8-15 jam |
| 34 | `crossval_frontonly` | 5-Fold CV | 5 × 3 model | ~2-4 jam |
| 35 | `randomsplit_frontonly` | Random Split | 5 × 3 model | ~30-60 menit |

---

## 14. Benchmark JAFFE & CK+ (Lanjutan)

Benchmark menggunakan dataset standar JAFFE dan CK+ dengan skenario yang sama.
Hanya B1 (Baseline) karena kedua dataset relatif seimbang.

### Prerequisite: Upload dataset benchmark ke VPS

```bash
# Di laptop: compress
cd D:/MultimodalEmoLearn
tar -czf benchmark_data.tar.gz data/benchmark/jaffe_7class data/benchmark/jaffe_4class \
    data/benchmark/ckplus_7class data/benchmark/ckplus_4class data/benchmark/ckplus_4class_contempt

# Upload ke VPS via SCP
scp benchmark_data.tar.gz USER@IP_VPS:/home/USER/MultimodalEmoLearn/

# Di VPS: extract
cd MultimodalEmoLearn
tar -xzf benchmark_data.tar.gz
rm benchmark_data.tar.gz
```

### Jalankan Benchmark

```bash
tmux new -s benchmark
conda activate emotrain
cd MultimodalEmoLearn

bash scripts/run_benchmark.sh
```

**Estimasi:** ~1-2 jam total (12 eksperimen per dataset × 2 dataset).

### Eksperimen per dataset:

| Model | Kelas | Total |
|-------|:-----:|:-----:|
| CNN | 7 + 4 | 2 |
| FCNN | 7 + 4 | 2 |
| Late Fusion | 7 + 4 | 2 |
| Intermediate | 7 + 4 | 2 |
| CNN TL | 7 + 4 | 2 |
| Intermediate TL | 7 + 4 | 2 |
| **Total per dataset** | | **12** |

### Output Benchmark:
```
models/benchmark/
├── jaffe/
│   ├── jaffe_7c_results.json
│   └── jaffe_4c_results.json
└── ckplus/
    ├── ckplus_7c_results.json
    └── ckplus_4c_results.json

notebooks/results/
├── 36_benchmark_jaffe_executed.ipynb
└── 37_benchmark_ckplus_executed.ipynb
```

### Benchmark LOSO & 10-Fold CV

Setelah single split selesai, jalankan LOSO (JAFFE) dan 10-fold CV (CK+):

```bash
tmux new -s benchmark_cv
conda activate emotrain
cd MultimodalEmoLearn

bash scripts/run_benchmark_cv.sh
```

**Estimasi:** ~8-13 jam total (120 training runs per dataset).

### Output:
```
models/benchmark/
├── jaffe_loso/
│   ├── jaffe_7c_loso_results.json
│   └── jaffe_4c_loso_results.json
└── ckplus_cv10/
    ├── ckplus_7c_cv10_results.json
    └── ckplus_4c_cv10_results.json

notebooks/results/
├── 38_benchmark_jaffe_loso_executed.ipynb
└── 39_benchmark_ckplus_cv10_executed.ipynb
```

---

## 15. Improved Experiments: Focal Loss + FER2013 Pre-Training

### Step 1: Download & Prepare FER2013 di VPS

```bash
# Install kaggle CLI (jika belum)
pip install kaggle

# Copy kaggle.json dari laptop ke VPS
# Di laptop:
scp C:/Users/grinv/.kaggle/kaggle.json USER@IP_VPS:~/.kaggle/

# Di VPS:
chmod 600 ~/.kaggle/kaggle.json
cd MultimodalEmoLearn

# Download FER2013
python -c "
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
api.dataset_download_files('msambare/fer2013', path='data/benchmark/fer2013/', unzip=True)
print('Done!')
"

# Prepare (resize 48->224, grayscale->RGB)
python scripts/prepare_fer2013.py
```

### Step 2: Run Improved Experiments

```bash
tmux new -s improved
conda activate emotrain
cd MultimodalEmoLearn

bash scripts/run_improved.sh
```

**Estimasi:** ~3-5 jam (pre-train FER2013 + 8 experiments).

### Output:
```
models/pretrained/resnet18_fer2013.pth    # FER2013 pre-trained weights
models/frontonly/improved/
├── CNN_TL_FocalLoss.pth
├── CNN_FER2013_Focal.pth
├── IntermediateTL_FER2013_Focal.pth
├── ...
└── improved_results.json
```

---

## 16. Undersampling Neutral (Lanjutan)

Mengurangi dominasi neutral (78% train) untuk meningkatkan deteksi kelas minoritas.

### Generate dataset undersampled (jika belum)
```bash
python scripts/prepare_undersampled.py
```

Menghasilkan 3 variasi:
```
data/dataset_frontonly_under_660_4class/   # neutral=660 (rasio 5.8:1)
data/dataset_frontonly_under_382_4class/   # neutral=382 (rasio 3.4:1)
data/dataset_frontonly_under_114_4class/   # neutral=114 (rasio 1:1)
```

### Jalankan eksperimen
```bash
tmux new -s undersample
conda activate emotrain
cd MultimodalEmoLearn

bash scripts/run_undersampled.sh
```

**Estimasi:** ~2-3 jam (3 model x 4 dataset = 12 eksperimen).

### Output:
```
models/frontonly/undersampled/
├── IntTL_*.pth, FCNN_*.pth, LateFusion_*.pth
├── undersampled_results.json        # per-class F1 scores
└── undersampled_perclass.png        # comparison chart

notebooks/results/
└── 42_frontonly_undersampled_executed.ipynb
```

---

## 17. Confidence Filtering >= 60% (Lanjutan)

Menghilangkan sampel dengan confidence score rendah (< 60%) dari Face API.
Kelas minoritas (angry, fearful, disgusted) punya confidence rata-rata rendah (0.56-0.67) — kemungkinan label salah.

### Step 1: Generate dataset conf60

```bash
cd MultimodalEmoLearn
conda activate emotrain

# Generate 7-class front-only dengan confidence >= 60%
python src/preprocessing/prepare_dataset.py --min-confidence 0.6 --output data/dataset_frontonly_conf60

# Generate augmented + 4-class
python scripts/prepare_conf60_all.py
```

Menghasilkan 4 dataset:
```
data/dataset_frontonly_conf60/                  # 7-class (~6,795 sampel)
data/dataset_frontonly_conf60_augmented/         # 7-class + augmentasi
data/dataset_frontonly_conf60_4class/            # 4-class
data/dataset_frontonly_conf60_4class_augmented/  # 4-class + augmentasi
```

### Step 2: Jalankan eksperimen

```bash
tmux new -s conf60
bash scripts/run_conf60.sh
```

**Estimasi:** ~8-12 jam (14 notebook × 3 skenario per model).

### Notebooks conf60 (43-57):

| No | Model | Kelas | Skenario |
|----|-------|:-----:|----------|
| 43-46 | CNN, FCNN, Late Fusion, Intermediate | 7 | B1/B2/B3 |
| 47-50 | CNN, FCNN, Late Fusion, Intermediate | 4 | B1/B2/B3 |
| 51-53 | CNN TL, Late Fusion TL, Intermediate TL | 7 | B1/B2/B3 |
| 54-56 | CNN TL, Late Fusion TL, Intermediate TL | 4 | B1/B2/B3 |
| 57 | Comparison conf60 vs original | - | - |

### Output:
```
models/frontonly_conf60/
├── 7class/          # from scratch 7-class results
├── 4class/          # from scratch 4-class results
├── 7class_tl/       # transfer learning 7-class
└── 4class_tl/       # transfer learning 4-class

notebooks/results/
├── 43-56_*_conf60_executed.ipynb
└── 57_comparison_conf60_executed.ipynb
```

---

## 18. Update: Late Fusion conf60 B1/B2/B3

Notebook Late Fusion conf60 (45, 49, 52, 55) diupdate untuk evaluasi **B1, B2, dan B3** (sebelumnya hanya B1). Pre-trained CNN & FCNN B1/B2/B3 sudah ada, hanya perlu re-run evaluation.

### Di VPS

```bash
cd MultimodalEmoLearn
git pull  # ambil notebook Late Fusion yang sudah diupdate

conda activate emotrain

# Re-run hanya Late Fusion notebooks (cepat, ~5-10 menit karena tidak training)
for nb in 45_late_fusion_conf60_7class 49_late_fusion_conf60_4class \
          52_late_fusion_tl_conf60_7class 55_late_fusion_tl_conf60_4class; do
    echo ">> $nb"
    jupyter nbconvert --to notebook --execute "notebooks/${nb}.ipynb" \
        --output "${nb}_executed.ipynb" --output-dir "notebooks/results/" \
        --ExecutePreprocessor.timeout=1800
done

# Commit hasil
git add models/frontonly_conf60/*/late_fusion*_results.json notebooks/results/*late_fusion*conf60*
git commit -m "Update Late Fusion conf60 with B1/B2/B3 results"
git push
```

**Estimasi:** ~5-10 menit (hanya evaluasi, tidak training ulang).

Hasil baru: `late_fusion_results.json` dan `late_fusion_tl_results.json` akan berisi 3 skenario (B1, B2, B3) bukan hanya B1.

---

## 19. Undersampling + Conf60 (Kombinasi Strategi Terbaik)

Menggabungkan 2 strategi terbukti efektif: conf60 + undersampling neutral (under_660 sweet spot).

### Di VPS

```bash
cd MultimodalEmoLearn
git pull

tmux new -s under_conf60
conda activate emotrain

bash scripts/run_undersampled_conf60.sh
```

**Estimasi:** ~1-1.5 jam (3 model × 2 dataset = 6 eksperimen).

### Scope

Fokus **under_660 saja** (sweet spot dari eksperimen undersampling sebelumnya).
Under-382 dan Under-114 tidak dijalankan karena hasil sebelumnya menurun signifikan.

### Output:
```
models/frontonly_conf60/undersampled/
├── IntTL_*.pth, FCNN_*.pth, LateFusion_*_cnn/fcnn.pth
└── undersampled_conf60_results.json   # per-class F1

notebooks/results/
└── 58_undersampled_conf60_executed.ipynb
```

---

## 21. Benchmark Lengkap: RAF-DB, KDEF, Cross-Dataset → Primer (Lanjutan)

Eksperimen benchmark tambahan sesuai arahan pembimbing:

- **Skema 1 (Self Train-Test):** Tiap dataset train & test di dirinya sendiri → CK+, JAFFE, RAF-DB, KDEF, Primer.
- **Skema 2 (Cross-Dataset):** Train di dataset sekunder, test di data test **Primer**.
- **Metrik:** Macro F1, Micro F1, Weighted F1 (ketiganya di-print, lewat update `src/training/utils.py`).
- **Model:** Semua model (CNN, FCNN, Intermediate, CNN_TL, Intermediate_TL, Late_Fusion).

### Step 1: Pull kode terbaru

```bash
cd MultimodalEmoLearn
git pull origin master
# Memastikan notebook 60-63, scripts/prepare_rafdb.py, scripts/prepare_kdef.py,
# scripts/run_benchmark_all.sh, dan update utils.py (micro F1) ada di VPS.
```

### Step 2: Download data mentah ke VPS (via gdown dari Google Drive)

Semua data mentah benchmark sudah di-host di Google Drive, download langsung di VPS (skip laptop):

```bash
# Install gdown (sekali saja)
pip install gdown

cd ~/MultimodalEmoLearn
mkdir -p data/benchmark
```

**Download CK+ (zip):**
```bash
# https://drive.google.com/file/d/1KrytJOXvYlN43gdmD7HsWFqOGToU9wv0/view
gdown "https://drive.google.com/uc?id=1KrytJOXvYlN43gdmD7HsWFqOGToU9wv0" -O ckplus_raw.zip
unzip -q ckplus_raw.zip -d data/benchmark/
# Hasilnya: data/benchmark/ck+/ (berisi folder Anger, Disgust, dst)
ls data/benchmark/ck+

rm ckplus_raw.zip   # hemat disk
```

**Download JAFFE (folder):**
```bash
# https://drive.google.com/drive/folders/1ymXCRJWVEfYaCA-AU0thfv4QSX8fxFcb
gdown --folder "https://drive.google.com/drive/folders/1ymXCRJWVEfYaCA-AU0thfv4QSX8fxFcb" \
      -O data/benchmark/jaffe
# Hasilnya: data/benchmark/jaffe/ (berisi folder emosi atau file TIFF langsung)
ls data/benchmark/jaffe
```

> Kalau struktur JAFFE-nya flat (semua TIFF di satu folder tanpa subfolder emosi), perlu diorganisir ke subfolder `Anger/`, `Disgust/`, `Fear/`, `Happy/`, `Neutral/`, `Sadness/`, `Surprised/` berdasarkan kode di filename (AN, DI, FE, HA, NE, SA, SU). `prepare_benchmark.py` mengharapkan struktur folder per-emosi.

**Download KDEF (zip 524 MB):**
```bash
# https://drive.google.com/file/d/1kf9kiId-3UF3d6Xre9zgwoujAS7Tmsb4/view
gdown "https://drive.google.com/uc?id=1kf9kiId-3UF3d6Xre9zgwoujAS7Tmsb4" -O KDEF_and_AKDEF.zip
# TIDAK perlu diekstrak — script baca langsung dari zip
ls -lh KDEF_and_AKDEF.zip   # harus ~524 MB
```

**RAF-DB — auto-download dari Kaggle (tidak perlu Drive):**
```bash
# Pastikan kaggle.json sudah di VPS:
ls ~/.kaggle/kaggle.json   # harus ada (kalau belum, lihat section 15)
chmod 600 ~/.kaggle/kaggle.json

# Download + generate sekaligus di Step 3 nanti
```

**Primer conf60** — sudah diupload dari eksperimen sebelumnya. Kalau terhapus juga, download dari Drive dengan cara yang sama.

### Step 3: Generate dataset benchmark di VPS

```bash
cd MultimodalEmoLearn
conda activate emotrain

# CK+ & JAFFE (dari folder mentah data/benchmark/{ck+,jaffe}/, ~3-5 menit)
python scripts/prepare_benchmark.py

# RAF-DB (auto download dari Kaggle, ~500MB, lalu extract landmarks ~15 menit)
python scripts/prepare_rafdb.py

# KDEF (baca langsung dari KDEF_and_AKDEF.zip, ~15-20 menit)
python scripts/prepare_kdef.py
```

Output yang dihasilkan:
```
data/benchmark/
├── ckplus_7class/         # 636 samples
├── ckplus_4class/         # 636 samples (remap)
├── ckplus_4class_contempt/ # 654 samples (with contempt)
├── jaffe_7class/          # 213 samples
├── jaffe_4class/          # 213 samples
├── rafdb_7class/          # 11,565 train + 2,884 test
├── rafdb_4class/          # same, remapped
├── kdef_7class/           # 2,630 train + 340 val + 337 test
└── kdef_4class/           # same, remapped
```

> **Hemat disk:** Kalau mau hapus file hasil download setelah prepare selesai:
> ```bash
> rm KDEF_and_AKDEF.zip                      # ~524 MB
> rm -rf data/benchmark/rafdb_raw/           # ~600 MB
> # ck+/ dan jaffe/ mentah (~120 MB) boleh dihapus kalau tidak butuh regenerate
> ```

### Step 4: Jalankan eksperimen benchmark

```bash
tmux new -s benchmark_full
conda activate emotrain
cd MultimodalEmoLearn

# Jalankan semua (Skema 1 + Skema 2) — detach-able
bash scripts/run_benchmark_all.sh

# Detach: Ctrl+B lalu D
# Re-attach: tmux attach -t benchmark_full
```

Atau jalankan per-notebook manual:
```bash
# Skema 1 (Self Train-Test)
jupyter nbconvert --to notebook --execute notebooks/60_benchmark_rafdb.ipynb \
    --output 60_executed.ipynb --output-dir notebooks/results/ \
    --ExecutePreprocessor.timeout=21600
jupyter nbconvert --to notebook --execute notebooks/61_benchmark_kdef.ipynb \
    --output 61_executed.ipynb --output-dir notebooks/results/ \
    --ExecutePreprocessor.timeout=21600
jupyter nbconvert --to notebook --execute notebooks/62_benchmark_primer.ipynb \
    --output 62_executed.ipynb --output-dir notebooks/results/ \
    --ExecutePreprocessor.timeout=21600

# Skema 2 (Cross-Dataset -> Primer test) — PALING LAMA
jupyter nbconvert --to notebook --execute notebooks/63_crossdataset_to_primer.ipynb \
    --output 63_executed.ipynb --output-dir notebooks/results/ \
    --ExecutePreprocessor.timeout=43200
```

### Estimasi waktu (NVIDIA T4)

| Notebook | Deskripsi | Estimasi |
|----------|-----------|----------|
| 60_benchmark_rafdb | RAF-DB self (11k train, 6 model × 2 config) | ~2-3 jam |
| 61_benchmark_kdef | KDEF self (2.6k train, 6 model × 2 config) | ~30-45 menit |
| 62_benchmark_primer | Primer conf60 self (6 model × 2 config) | ~45-60 menit |
| 63_crossdataset_to_primer | Cross (CK+, JAFFE, RAF-DB, KDEF → Primer) × 2 config × 6 model | ~4-8 jam |
| **Total** | | **~8-12 jam** |

### Output hasil benchmark

```
models/benchmark/
├── rafdb/
│   ├── 7c/{CNN,FCNN,Intermediate,CNN_TL,Intermediate_TL,Late_Fusion}_B1/*.pth
│   ├── 4c/...
│   ├── rafdb_7c_results.json
│   └── rafdb_4c_results.json
├── kdef/
│   ├── 7c/..., 4c/...
│   ├── kdef_7c_results.json
│   └── kdef_4c_results.json
├── primer/
│   ├── 7c/..., 4c/...
│   ├── primer_7c_results.json
│   └── primer_4c_results.json
└── crossdataset/
    ├── ckplus_7c/..., ckplus_4c/...
    ├── jaffe_7c/..., jaffe_4c/...
    ├── rafdb_7c/..., rafdb_4c/...
    ├── kdef_7c/..., kdef_4c/...
    ├── cross_{dataset}_{7,4}c.json  (8 files)
    └── all_cross_results.json

notebooks/results/
├── 60_benchmark_rafdb_executed.ipynb
├── 61_benchmark_kdef_executed.ipynb
├── 62_benchmark_primer_executed.ipynb
└── 63_crossdataset_to_primer_executed.ipynb
```

### Step 5: Transfer hasil balik ke laptop

```bash
# Di VPS:
cd MultimodalEmoLearn
tar -czf benchmark_full_results.tar.gz \
    models/benchmark/rafdb/*.json models/benchmark/kdef/*.json \
    models/benchmark/primer/*.json models/benchmark/crossdataset/*.json \
    notebooks/results/6{0,1,2,3}_*_executed.ipynb
```

Download via MobaXterm atau:
```bash
scp USER@IP_VPS:/home/USER/MultimodalEmoLearn/benchmark_full_results.tar.gz D:/MultimodalEmoLearn/
cd D:/MultimodalEmoLearn && tar -xzf benchmark_full_results.tar.gz
```

> **Tips:** File `.pth` (checkpoint model) biasanya tidak perlu ditransfer balik — ukurannya besar (~50-300 MB per model) dan tidak dibutuhkan untuk analisis. Cukup JSON + notebook executed.

---

## 22. Early Fusion (Input-Level Channel Concatenation) — Lanjutan

Sesuai arahan dosen, tambah **Early Fusion** yang melengkapi spektrum arsitektur:

| Fusion Level | % Depth | Arsitektur |
|--------------|:-------:|-----------|
| **Early** (NEW) | 0% (input) | Landmark heatmap sebagai channel ke-4 pada CNN |
| Intermediate | 50% (feature) | CNN + FCNN features concat di hidden layer |
| Late | 95% (decision) | Softmax averaging 2 model terpisah |

**Referensi**: Mo, S., Yang, W., Wang, G., & Liao, Q. (2020) — *Emotion Recognition with Facial Landmark Heatmaps* (HAE-Net), MMM 2020, LNCS 11961, pp. 278-289, Springer.

### Step 1: Pull kode terbaru di VPS

```bash
cd ~/MultimodalEmoLearn
git pull origin master
# Ambil: src/training/models.py (EmotionEarlyFusion + EmotionEarlyFusionTransfer),
#        scripts/generate_landmark_heatmaps.py, notebooks/64_early_fusion_conf60.ipynb
```

### Step 2: Generate landmark heatmaps

Heatmap di-generate dari landmark 136-dim yang sudah ada → Gaussian blob 224×224 (sigma=3px).

**Wajib untuk nb 64:**
```bash
conda activate emotrain
cd ~/MultimodalEmoLearn

# Base dataset (untuk B1 dan B2, ~10 menit)
python scripts/generate_landmark_heatmaps.py --only "Primer conf60"
# Output: data/dataset_frontonly_conf60/X_{train,val,test}_heatmaps.npy (~1.4 GB total)

# Augmented dataset (untuk B3, ~5-10 menit)
python scripts/generate_landmark_heatmaps.py --only "augmented"
# Output: data/dataset_frontonly_conf60_augmented/X_train_heatmaps.npy
```

Kalau mau semua (primer + benchmarks sekaligus, ~90 menit):
```bash
python scripts/generate_landmark_heatmaps.py
```

### Step 3: Jalankan training Early Fusion

```bash
tmux new -s early_fusion
conda activate emotrain
cd ~/MultimodalEmoLearn

jupyter nbconvert --to notebook --execute notebooks/64_early_fusion_conf60.ipynb \
    --output 64_early_fusion_conf60_executed.ipynb \
    --output-dir notebooks/results/ \
    --ExecutePreprocessor.timeout=14400

# Detach: Ctrl+B lalu D
```

### Konfigurasi yang dijalankan

6 config × 2 class = **12 eksperimen** total:

| Config | Backbone | Class Weights | Augmented | Kelas |
|--------|----------|:-------------:|:---------:|:-----:|
| EarlyFusion_B1 | scratch | no | no | 7, 4 |
| EarlyFusion_B2 | scratch | yes | no | 7, 4 |
| EarlyFusion_B3 | scratch | yes | **yes** | 7, 4 |
| EarlyFusion_TL_B1 | ResNet18 TL (4-ch) | no | no | 7, 4 |
| EarlyFusion_TL_B2 | ResNet18 TL (4-ch) | yes | no | 7, 4 |
| EarlyFusion_TL_B3 | ResNet18 TL (4-ch) | yes | **yes** | 7, 4 |

**Estimasi**: ~2.5-3 jam di T4 (12 eksperimen × 10-15 menit each).

Note: B3 akan otomatis di-skip kalau `data/dataset_frontonly_conf60_augmented/` atau heatmap-nya tidak ditemukan.

### Output

```
models/frontonly_conf60/early_fusion/
├── 7c/
│   ├── EarlyFusion_B1/model.pth
│   ├── EarlyFusion_B2/model.pth
│   ├── EarlyFusion_TL_B1/model.pth
│   └── EarlyFusion_TL_B2/model.pth
├── 4c/ (same structure)
├── early_fusion_7c_results.json
└── early_fusion_4c_results.json

notebooks/results/
└── 64_early_fusion_conf60_executed.ipynb
```

### Catatan Implementasi

- **First Conv modifikasi (TL variant)**: ResNet18 `conv1` diubah dari `Conv2d(3, 64, 7)` ke `Conv2d(4, 64, 7)`. Weight channel RGB di-copy dari pretrained, channel ke-4 (heatmap) diinisialisasi dari rata-rata RGB — reasonable starting point.
- **Heatmap**: Gaussian std=3px per landmark, element-wise MAX (bukan sum) untuk menjaga range [0, 1].
- **Format**: `X_{split}_images.npy (N, 224, 224, 3)` + `X_{split}_heatmaps.npy (N, 224, 224)` → di-stack ke (N, 224, 224, 4) di notebook.

### Kenapa ini berbeda dari Intermediate Fusion?

Meski sama-sama "concat landmark + image", beda fundamental:

- **Intermediate**: landmark jadi vector 136-dim, fusi di feature level (setelah CNN dan FCNN extract features terpisah). Info spasial landmark **hilang** saat di-flatten.
- **Early**: landmark tetap 2D heatmap 224×224, fusi di input level (channel concat). Info spasial **dipertahankan**, CNN joint-learn visual + geometric dari layer pertama.

---

## 23. Late Fusion TL (melengkapi benchmark) — Lanjutan

Late Fusion TL (ResNet18 CNN_TL + FCNN, weighted softmax averaging) terlewat dari
benchmark notebooks 60-62. Notebook 65 melengkapinya **tanpa re-train** jika
checkpoint `CNN_TL_B1` dan `FCNN_B1` sudah ada dari run sebelumnya.

### Step 1: Pull kode terbaru

```bash
cd ~/MultimodalEmoLearn
git pull origin master
# Ambil: notebooks/65_late_fusion_tl_benchmarks.ipynb
```

### Step 2: Jalankan notebook 65

```bash
tmux new -s latefusion_tl
conda activate emotrain
cd ~/MultimodalEmoLearn

jupyter nbconvert --to notebook --execute notebooks/65_late_fusion_tl_benchmarks.ipynb \
    --output 65_late_fusion_tl_benchmarks_executed.ipynb \
    --output-dir notebooks/results/ \
    --ExecutePreprocessor.timeout=14400
```

### Cara kerja notebook

Untuk tiap dataset × num_classes (6 variants total):

1. **Check checkpoint existing**:
   - `models/benchmark/{dataset}/{num_class}c/CNN_TL_B1/model.pth` → load kalau ada
   - `models/benchmark/{dataset}/{num_class}c/FCNN_B1/model.pth` → load kalau ada
2. **Train kalau belum ada** (fallback)
3. **Softmax averaging**:
   - Inference di val set untuk tune weight w ∈ [0, 1]
   - Pilih w yang max Macro F1 di val
   - Evaluate di test set dengan best w
4. **Update `{dataset}_{num_class}c_results.json`** dengan key `Late_Fusion_TL_B1`

### Estimasi waktu

| Skenario | Estimasi |
|----------|----------|
| **Kalau semua checkpoint sudah ada** (most likely) | ~15 menit total (inference saja) |
| Kalau harus train beberapa | 30-90 menit |
| Worst case (train semua) | 2-3 jam |

### Output

Update file existing + tambah 1 key baru per dataset:
```
models/benchmark/rafdb/rafdb_7c_results.json   (+ Late_Fusion_TL_B1)
models/benchmark/rafdb/rafdb_4c_results.json   (+ Late_Fusion_TL_B1)
models/benchmark/kdef/kdef_7c_results.json     (+ Late_Fusion_TL_B1)
models/benchmark/kdef/kdef_4c_results.json     (+ Late_Fusion_TL_B1)
models/benchmark/primer/primer_7c_results.json (+ Late_Fusion_TL_B1)
models/benchmark/primer/primer_4c_results.json (+ Late_Fusion_TL_B1)

notebooks/results/65_late_fusion_tl_benchmarks_executed.ipynb
```

### Commit hasil

```bash
cd ~/MultimodalEmoLearn
git add models/benchmark/*/*.json notebooks/results/65_*
git commit -m "Add Late Fusion TL results across all benchmarks (nb 65)"
git push
```

---

## 24. Early Fusion di Benchmark Dataset (Lanjutan)

Melengkapi Skema 1 benchmark (CK+, JAFFE, RAF-DB, KDEF) dengan **Early Fusion** — sebelumnya hanya Primer yang sudah di-train (nb 64). Notebook 66 menambahkan row `EarlyFusion_B1` dan `EarlyFusion_TL_B1` ke `{dataset}_{num}c_results.json` existing.

### Prerequisite

Heatmap benchmark harus sudah di-generate. Cek dengan:
```bash
cd ~/MultimodalEmoLearn
ls data/benchmark/ckplus_7class/X_heatmaps.npy \
   data/benchmark/ckplus_4class_contempt/X_heatmaps.npy \
   data/benchmark/jaffe_7class/X_heatmaps.npy \
   data/benchmark/jaffe_4class/X_heatmaps.npy \
   data/benchmark/rafdb_7class/X_train_heatmaps.npy \
   data/benchmark/kdef_7class/X_train_heatmaps.npy 2>&1 | head
```

Kalau ada file yang missing, generate:
```bash
conda activate emotrain
python scripts/generate_landmark_heatmaps.py --only "CK+"
python scripts/generate_landmark_heatmaps.py --only "JAFFE"
python scripts/generate_landmark_heatmaps.py --only "RAF-DB"
python scripts/generate_landmark_heatmaps.py --only "KDEF"
# Atau semua sekaligus:
# python scripts/generate_landmark_heatmaps.py
```

### Step 1: Pull kode terbaru

```bash
cd ~/MultimodalEmoLearn
git pull origin master
# Ambil: notebooks/66_early_fusion_benchmarks.ipynb
```

### Step 2: Jalankan notebook 66

```bash
tmux new -s ef_bench
conda activate emotrain
cd ~/MultimodalEmoLearn

jupyter nbconvert --to notebook --execute notebooks/66_early_fusion_benchmarks.ipynb \
    --output 66_early_fusion_benchmarks_executed.ipynb \
    --output-dir notebooks/results/ \
    --ExecutePreprocessor.timeout=18000

# Detach: Ctrl+B lalu D
```

### Konfigurasi yang dijalankan

4 dataset × 2 class × 2 backbone = **16 eksperimen** total (B1 baseline saja, apple-to-apple dengan Skema 1 benchmark):

| Config | Backbone | Kelas |
|--------|----------|:-----:|
| EarlyFusion_B1 | scratch (`EmotionEarlyFusion`) | 7, 4 |
| EarlyFusion_TL_B1 | ResNet18 TL 4-ch (`EmotionEarlyFusionTransfer`) | 7, 4 |

Dataset yang dijalankan: **CK+, JAFFE, RAF-DB, KDEF** (Primer tidak, sudah ada di nb 64 dengan skenario lengkap B1/B2/B3 × scratch/TL).

### Estimasi waktu

| Dataset | Scratch + TL (per kelas) | Total (7c + 4c) |
|---------|:------------------------:|:---------------:|
| CK+ (subject-wise split) | ~10-15 menit | ~25-30 menit |
| JAFFE (subject-wise split) | ~5-8 menit | ~12-18 menit |
| RAF-DB (11,565 train) | ~40-60 menit | ~90-120 menit |
| KDEF (2,630 train + 340 val) | ~15-20 menit | ~35-45 menit |

**Total estimasi**: ~2.5-4 jam di T4.

### Output

```
models/benchmark/ckplus/ckplus_7c/EarlyFusion_B1/model.pth
models/benchmark/ckplus/ckplus_7c/EarlyFusion_TL_B1/model.pth
models/benchmark/ckplus/ckplus_4c/EarlyFusion_{B1,TL_B1}/model.pth
models/benchmark/jaffe/jaffe_{7c,4c}/EarlyFusion_{B1,TL_B1}/model.pth
models/benchmark/rafdb/{7c,4c}/EarlyFusion_{B1,TL_B1}/model.pth
models/benchmark/kdef/{7c,4c}/EarlyFusion_{B1,TL_B1}/model.pth

# Update file JSON existing dengan 2 key baru per dataset:
models/benchmark/ckplus/ckplus_{7,4}c_results.json  (+ EarlyFusion_B1, EarlyFusion_TL_B1)
models/benchmark/jaffe/jaffe_{7,4}c_results.json    (+ ...)
models/benchmark/rafdb/rafdb_{7,4}c_results.json    (+ ...)
models/benchmark/kdef/kdef_{7,4}c_results.json      (+ ...)

notebooks/results/66_early_fusion_benchmarks_executed.ipynb
```

### Catatan Naming Convention

Sama seperti nb 65, struktur folder berbeda antara dataset lama & baru:
- **ckplus, jaffe** (nb 36/37 style): `models/benchmark/{ds}/{ds}_{num}c/EarlyFusion_{B1,TL_B1}/`
- **rafdb, kdef** (nb 60/61 style): `models/benchmark/{ds}/{num}c/EarlyFusion_{B1,TL_B1}/`

Kedua pattern di-handle otomatis oleh helper `checkpoint_dir()` di notebook.

### Auto-skip & Idempotency

- **Skip kalau heatmap missing**: notebook cek existence `X_heatmaps.npy` (atau `X_train/val/test_heatmaps.npy` untuk rafdb/kdef) — print warning lalu skip combo tsb
- **Skip kalau checkpoint ada**: kalau `model.pth` sudah ada, load saja → re-run safe untuk resume

### Commit hasil

```bash
cd ~/MultimodalEmoLearn
git add models/benchmark/*/*.json \
        "models/benchmark/ckplus/*/EarlyFusion_*/" \
        "models/benchmark/jaffe/*/EarlyFusion_*/" \
        "models/benchmark/rafdb/*/EarlyFusion_*/" \
        "models/benchmark/kdef/*/EarlyFusion_*/" \
        notebooks/results/66_*
git commit -m "Add Early Fusion results across all benchmarks (nb 66)"
git push
```

---

## 25. Cross-Dataset Early Fusion (Skema 2 Extension) — Inference Only

Melengkapi Skema 2 cross-dataset dengan Early Fusion yang sebelumnya terlewat di nb 63. **Inference-only** (reuse checkpoint nb 66) — tidak training baru.

### Prerequisite

1. Checkpoint Early Fusion dari nb 66 sudah ada di VPS ✓ (sudah di-commit `100cd95`)
2. Heatmap Primer conf60 test: `data/dataset_frontonly_conf60/X_test_heatmaps.npy` (sudah ada)

### Step: Jalankan notebook 68

```bash
cd ~/MultimodalEmoLearn
git pull origin master
conda activate emotrain

jupyter nbconvert --to notebook --execute notebooks/68_crossdataset_early_fusion.ipynb \
    --output 68_crossdataset_early_fusion_executed.ipynb \
    --output-dir notebooks/results/ \
    --ExecutePreprocessor.timeout=1800
```

### Konfigurasi

4 source (CK+/JAFFE/RAF-DB/KDEF) × 2 class (7c/4c) × 2 backbone (EF scratch + EF TL) = **16 inference runs** (target: Primer conf60 test 929 images).

**Estimasi: ~5-10 menit total** (inference GPU only, no training).

### Output

```
models/benchmark/crossdataset/
├── cross_ckplus_7c.json    (+ EarlyFusion_B1, EarlyFusion_TL_B1)
├── cross_ckplus_4c.json    (+ ...)
├── cross_jaffe_7c.json     (+ ...)
├── cross_jaffe_4c.json     (+ ...)
├── cross_rafdb_7c.json     (+ ...)
├── cross_rafdb_4c.json     (+ ...)
├── cross_kdef_7c.json      (+ ...)
├── cross_kdef_4c.json      (+ ...)
└── all_cross_results.json  (combined, updated)

notebooks/results/68_crossdataset_early_fusion_executed.ipynb
```

### Hipotesis

Heatmap landmark = **geometric info** yang lebih domain-invariant dibanding RGB → Early Fusion **mungkin unggul** untuk cross-dataset transfer ke Primer vs CNN single-modality. Namun konsekuensinya, kalau hasil buruk → heatmap generic (Gaussian blob) tidak cukup bantu untuk transfer.

### Commit hasil

```bash
cd ~/MultimodalEmoLearn
git add models/benchmark/crossdataset/cross_*.json \
        models/benchmark/crossdataset/all_cross_results.json \
        notebooks/results/68_*
git commit -m "Add cross-dataset Early Fusion → Primer results (nb 68)"
git push
```

---

## 26. Cross-Dataset Late Fusion TL (Skema 2 Extension) — Inference Only

Melengkapi Skema 2 dengan Late Fusion TL yang terlewat di nb 63. **Inference-only** — reuse `CNN_TL_B1` + `FCNN_B1` checkpoint dari nb 65 (Skema 1 self-train-test).

### Prerequisite

- Checkpoint `CNN_TL_B1` + `FCNN_B1` per source (sudah ada dari nb 65)
- Primer conf60 val + test (sudah ada)

### Step: Jalankan notebook 69

```bash
cd ~/MultimodalEmoLearn
git pull origin master
conda activate emotrain

jupyter nbconvert --to notebook --execute notebooks/69_crossdataset_late_fusion_tl.ipynb \
    --output 69_crossdataset_late_fusion_tl_executed.ipynb \
    --output-dir notebooks/results/ \
    --ExecutePreprocessor.timeout=1800
```

### Konfigurasi

4 source × 2 class × 2 tuning strategy = **16 evaluations** (8 source-class combos × 2 strategi w):

- **Strategi A** — tune `w` di **Primer val** (579 imgs) → saved as `Late_Fusion_TL_B1`
- **Strategi B** — tune `w` di **source val** (zero-shot target-agnostic) → saved as `Late_Fusion_TL_B1_srcval`

Grid search: w ∈ [0.00, 0.05, ..., 1.00], pilih max Macro F1. **Estimasi ~5-10 menit** (inference only).

### Output

```
models/benchmark/crossdataset/
├── cross_{source}_{num}c.json   (+ Late_Fusion_TL_B1, Late_Fusion_TL_B1_srcval)
└── all_cross_results.json       (combined, updated)

notebooks/results/69_crossdataset_late_fusion_tl_executed.ipynb
```

### Interpretasi 2 Strategi

- **Strategi A (Primer val tune)**: optimistic — simulasi scenario dengan akses sedikit target data untuk adaptasi
- **Strategi B (source val tune)**: pure zero-shot — paling fair untuk klaim "trained on X, tested on Primer tanpa akses target"
- Gap A - B mengukur seberapa besar benefit dari target-side tuning

Biasanya A ≥ B (target info bantu). Kalau B → A kecil, artinya `w` optimal generalize across domain.

### Commit hasil

```bash
cd ~/MultimodalEmoLearn
git add models/benchmark/crossdataset/cross_*.json \
        models/benchmark/crossdataset/all_cross_results.json \
        notebooks/results/69_*
git commit -m "Add cross-dataset Late Fusion TL → Primer results (nb 69)"
git push
```

---

## 27. Soft Label Training (Eksplorasi Liliana-inspired)

Eksperimen Soft Label Training — target distribusi Face API confidence, bukan hard one-hot. Sumber inspirasi: Liliana et al. (2019) "Fuzzy Emotion" + Mo et al. (2020) HAE-Net konsep *natural emotions are inherently ambiguous*.

### Prerequisite: Soft Labels Files

**Penting:** `scripts/extract_soft_labels.py` butuh `data/processed_new/` yang **tidak tersedia di VPS** (hanya ada di Windows PC). Solusi: file `y_*_soft.npy` sudah di-extract di local + **di-commit ke repo** (total ~190KB, kecil).

```bash
cd ~/MultimodalEmoLearn
git pull origin master
conda activate emotrain

# Cek file tersedia:
ls -lh data/dataset_frontonly_conf60/y_*_soft.npy
# Expected:
#   y_test_soft.npy    ~26K
#   y_train_soft.npy   ~145K
#   y_val_soft.npy     ~16K
```

Kalau belum ada → `git pull` aja, file sudah di-track. Tidak perlu run `extract_soft_labels.py` di VPS (script tsb cuma untuk regenerasi di laptop local).

### Step: Jalankan notebook 71

```bash
tmux new -s softlabel
conda activate emotrain
cd ~/MultimodalEmoLearn

jupyter nbconvert --to notebook --execute notebooks/71_soft_label_training.ipynb \
    --output 71_soft_label_training_executed.ipynb \
    --output-dir notebooks/results/ \
    --ExecutePreprocessor.timeout=7200
```

### Konfigurasi

4 config × 4-class × CNN TL B1 baseline = **4 runs** (isolate efek loss function saja):

| Config | Loss | Target |
|--------|------|--------|
| A (baseline) | Hard Cross-Entropy | one-hot |
| B | Soft Cross-Entropy | Face API distribution |
| C | KL-divergence | Face API distribution |
| D | Label Smoothing CE (ε=0.1) | smoothed one-hot |

### Estimasi waktu

- CNN TL 4c 50 epochs (early stop ~25-35 epochs): ~8-12 menit per config
- Total: ~30-50 menit

### Output

```
models/frontonly_conf60/soft_label/
├── 4c/
│   ├── A_hard_CE/model.pth
│   ├── B_soft_CE/model.pth
│   ├── C_KL_div/model.pth
│   └── D_label_smooth/model.pth
└── soft_label_4c_results.json

notebooks/results/71_soft_label_training_executed.ipynb
```

### Interpretasi Hasil

- **B/C > A**: soft target membantu → extend ke Late Fusion TL (target beat 0.567)
- **A ≈ B ≈ C**: Face API confidence tidak add info → tahan, tidak lanjut
- **D > A tapi B/C < A**: label smoothing generic membantu, soft Face API noisy
- **Per-class analysis** — cek apakah F1 sad/negative improve (key test untuk ambiguity hypothesis)

### Commit hasil

```bash
cd ~/MultimodalEmoLearn
git add data/dataset_frontonly_conf60/y_*_soft.npy \
        data/dataset_frontonly_conf60/dataset_info.json \
        models/frontonly_conf60/soft_label/ \
        notebooks/results/71_*
git commit -m "Add Soft Label Training results (nb 71)"
git push
```

---

## 28. Soft Label Extended ke Late Fusion TL (Target: beat 0.567)

Lanjutan nb 71 — extend soft label yang menang di CNN TL (+0.09 Macro F1) ke arsitektur best overall (Late Fusion TL 4c B3 hard = 0.567).

### Prerequisite

Sama dengan nb 71 (soft labels sudah di-commit via `y_*_soft.npy`).

### Step: Jalankan notebook 72

```bash
tmux new -s softlf
conda activate emotrain
cd ~/MultimodalEmoLearn
git pull origin master

jupyter nbconvert --to notebook --execute notebooks/72_soft_label_late_fusion_tl.ipynb \
    --output 72_soft_label_late_fusion_tl_executed.ipynb \
    --output-dir notebooks/results/ \
    --ExecutePreprocessor.timeout=7200
```

### Konfigurasi

2 variants × 2 branches = 4 trainings baru + inference + grid search:

| Config | CNN_TL Loss | FCNN Loss | Note |
|--------|-------------|-----------|------|
| A_KL_div | KL-div soft | KL-div soft | winning loss dari nb 71 |
| B_soft_CE | Soft CE | Soft CE | runner-up dari nb 71 |

Fusion: weighted softmax averaging, grid search `w` di Primer val, eval di Primer test.

### Estimasi

- CNN TL training (KL-div / Soft CE): ~15-20 menit/run
- FCNN training (KL-div / Soft CE): ~10-15 menit/run
- **Total: ~60-70 menit** (2 config × 2 branch × ~15 min + inference fast)

### Target

- Baseline Late Fusion TL 4c B1 hard: 0.513
- **Current best: Late Fusion TL 4c B3 hard = 0.567** ⭐
- Optimis: soft label KL-div di fusion → potensi **beat 0.567** tanpa perlu B3 augmentation

### Output

```
models/frontonly_conf60/soft_label/4c/LF_TL_KL_div/
├── cnn_tl.pth
└── fcnn.pth
models/frontonly_conf60/soft_label/4c/LF_TL_soft_CE/
├── cnn_tl.pth
└── fcnn.pth
models/frontonly_conf60/soft_label/soft_lf_tl_4c_results.json

notebooks/results/72_soft_label_late_fusion_tl_executed.ipynb
```

### Commit hasil

```bash
cd ~/MultimodalEmoLearn
git add models/frontonly_conf60/soft_label/ \
        notebooks/results/72_*
git commit -m "Add Soft Label Late Fusion TL results (nb 72)"
git push
```

---

## 29. nb 73 — GradCAM Analysis (arahan dosen #1)

Visualisasi bobot NN: region citra yang paling berpengaruh ke prediksi per kelas.

### Prasyarat library (one-time)

```bash
conda activate emotrain
pip install grad-cam
```

### Prasyarat checkpoint

Butuh 2 checkpoint di `models/frontonly_conf60/4class_tl/`:
- `cnn_tl_b1.pth` (CNN TL 4c B1 baseline)
- `intermediate_tl_b3.pth` (Intermediate TL 4c B3 — best overall val-tuned)

Kedua file **sudah ada di repo** (dari nb 54 + nb 56 execution). Pastikan `git pull` sebelum run.

### Run

```bash
cd ~/MultimodalEmoLearn
tmux new -s gradcam
conda activate emotrain

jupyter nbconvert --to notebook --execute notebooks/73_gradcam_analysis.ipynb \
    --output 73_gradcam_analysis_executed.ipynb \
    --output-dir notebooks/results \
    --ExecutePreprocessor.timeout=900
```

**Estimasi:** ~5-10 menit (GradCAM inference lightweight, tidak butuh training).

### Output

```
docs/figures/gradcam/
├── cnn_tl_4c_b1_baseline.png                   ← 2×4 grid, 8 sampel
└── intermediate_tl_4c_b3_img_branch.png        ← 2×4 grid, 8 sampel (fusion image branch)

notebooks/results/73_gradcam_analysis_executed.ipynb
```

### Troubleshooting

Kalau `Intermediate TL` cell error di line `img_branch = getattr(m2, 'image_features', None)`:
- Inspect nama attribute image branch via printed `named_children()` output
- Adjust `img_branch = getattr(m2, '<nama_yang_benar>', None)` sesuai model definition di `src/training/models.py`

### Commit hasil

```bash
cd ~/MultimodalEmoLearn
git add docs/figures/gradcam/ notebooks/results/73_*
git commit -m "Add GradCAM overlays (nb 73): CNN TL + Intermediate TL image branch"
git push
```

### Analisis kualitatif (post-run)

Bandingkan 2 PNG overlay:
- **CNN TL baseline** — fokus di mana per kelas?
- **Intermediate TL image branch** — apakah fokus bergeser karena adanya landmark stream?

Catatan per kelas (expected):
- neutral → eyes/mouth (netral, tidak distinct)
- happy → mouth (smile) + eye crinkle
- sad → brow + mouth corners
- negative → heterogen (merge angry/fearful/disgusted/surprised)

Hasil masuk ke BAB Discussion tesis + section paper Qualitative Analysis (optional Fig. 7).

---

## 30. Troubleshooting

### CUDA Out of Memory untuk RAF-DB (nb 60, 63)

RAF-DB images = 11,565 × 224×224×3 × 4 byte = ~7 GB. Kalau VRAM T4 (16GB) masih OOM:
```python
# Di notebook 60 / 63, ubah BATCH_SIZE:
BATCH_SIZE = 16  # default 32
```

### Disk penuh setelah prepare RAF-DB

File `.npy` raf-db bisa >10 GB. Hapus raw data setelah prepare selesai:
```bash
rm -rf data/benchmark/rafdb_raw/
# dataset siap pakai tetap di data/benchmark/rafdb_{7,4}class/
```

### CUDA Out of Memory
```python
# Kurangi batch size di notebook:
BATCH_SIZE = 16  # atau 8 untuk intermediate fusion
```

### Training terlalu lama
```python
# Kurangi epochs dan patience:
EPOCHS = 30
PATIENCE = 10
```

### Notebook error saat import
```bash
# Pastikan working directory benar:
cd MultimodalEmoLearn
# Pastikan src/ ada di path:
ls src/training/models.py  # harus ada
```
