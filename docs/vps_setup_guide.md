# VPS Setup untuk Melanjutkan Eksperimen 3-Class yang Pending

## Pending Tasks (yang butuh VPS)

| Task | Notebook | Estimasi | Butuh data |
|---|---|---|---|
| Skema 1 benchmark RAF-DB + KDEF 3c | nb 84 | ~6-8 jam | RAF-DB + KDEF 7class |
| Skema 2 cross-dataset → Primer 3c | nb 85 | ~30 mnt (inference only) | Checkpoints nb 84 + Primer 3c test |
| History logging (optional) | nb 83 | ~5-8 jam | Primer conf60 3c (sudah ada) |

> **NB 84 prerequisite untuk 85.** nb 85 inference-only, pakai checkpoint hasil nb 84.

---

## Step 1: Clone Repo + Restore Code

```bash
cd ~
git clone https://github.com/taufikhdyt01/MultimodalEmoLearn.git
cd MultimodalEmoLearn
```

---

## Step 2: Setup Python Environment

```bash
# Gunakan venv (lebih ringan dari conda)
python3 -m venv .venv
source .venv/bin/activate

# CUDA 12.1 build (kompatibel dengan driver CUDA 12.x di kebanyakan VPS)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Dependencies lainnya
pip install numpy scikit-learn matplotlib jupyter ipykernel openpyxl pandas tqdm opencv-python mediapipe

# Verifikasi GPU
python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0)); print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory/1e9, 1), 'GB')"
```

Kalau VPS pakai Python 3.12+, tambahkan `--break-system-packages` atau gunakan miniconda:

```bash
# Alternatif: miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
~/miniconda3/bin/conda create -n train python=3.11 -y
source ~/miniconda3/bin/activate train
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install numpy scikit-learn matplotlib jupyter openpyxl pandas tqdm opencv-python mediapipe
```

---

## Step 3: Upload Data ke VPS

Pakai `rsync` atau `scp` dari laptop. **Hanya upload yang dibutuhkan untuk pending tasks.**

### 3a. Primer 3-class (wajib — untuk test set)

```bash
# Di LAPTOP:
scp -r data/dataset_frontonly_conf60 user@vps-ip:~/MultimodalEmoLearn/data/
# ~5.1 GB — sudah include X_test_images.npy, X_test_landmarks.npy, X_test_heatmaps.npy, y_test.npy
```

### 3b. Benchmark dataset yang BELUM selesai

CK+ & JAFFE sudah done di laptop via nb 84 partial — upload hasilnya saja (skip data):

```bash
# Di LAPTOP — upload existing results:
scp -r models/benchmark/ user@vps-ip:~/MultimodalEmoLearn/models/benchmark/
```

Dataset untuk RAF-DB & KDEF (wajib upload data):

```bash
# Di LAPTOP — RAF-DB (8.2 GB) + KDEF (1.9 GB):
scp -r data/benchmark/rafdb_7class user@vps-ip:~/MultimodalEmoLearn/data/benchmark/
scp -r data/benchmark/kdef_7class  user@vps-ip:~/MultimodalEmoLearn/data/benchmark/
```

> **Total upload data: ~15 GB** (Primer 5.1 + RAF-DB 8.2 + KDEF 1.9).  
> CK+ & JAFFE data tidak perlu di-upload (sudah ada hasil di `models/benchmark/`).

### 3c. Resume logic — VERY IMPORTANT

nb 84 punya resume logic: skip dataset yang sudah punya `results.json`. Kalau kamu upload `models/benchmark/` beserta `*_3c_results.json`, notebook akan **skip CK+ dan JAFFE** — langsung mulai dari RAF-DB.

---

## Step 4: Restore Checkpoint .pth Backups

Kamu sudah punya backup .pth dari VPS lama. Letakkan sesuai struktur:

```
models/frontonly_conf60/3class/
├── all_results_3class.json              # master results (15 TL configs)
├── scratch_all_results.json             # master results (12 scratch configs)
├── Late_Fusion_TL/
│   ├── cnn_tl_b1.pth
│   ├── cnn_tl_b2.pth
│   ├── cnn_tl_b3.pth
│   ├── fcnn_b1.pth
│   ├── fcnn_b2.pth
│   ├── fcnn_b3.pth
│   └── results.json
├── Late_Fusion_scratch/
│   ├── cnn_b1.pth, cnn_b2.pth, cnn_b3.pth
│   ├── fcnn_b1.pth, fcnn_b2.pth, fcnn_b3.pth
│   └── results.json
├── CNN_TL/         → cnn_tl_b{1,2,3}.pth
├── FCNN/           → fcnn_b{1,2,3}.pth
├── Intermediate_TL/→ intermediate_tl_b{1,2,3}.pth
├── Early_Fusion_TL/→ early_fusion_tl_b{1,2,3}.pth
└── history/        → training curves JSON (untuk nb 83)
```

---

## Step 5: Eksekusi

### 5a. Lanjutkan nb 84 (Skema 1 RAF-DB + KDEF 3c)

```bash
cd ~/MultimodalEmoLearn
source .venv/bin/activate

# Buka notebook, run from top
jupyter notebook notebooks/84_threeclass_skema1_benchmark.ipynb
```

Resume logic akan skip CK+ & JAFFE, langsung jalankan RAF-DB + KDEF.  
**Perkiraan: 6-8 jam di VPS 1× T4.** kalau pakai GPU lebih kecil, bisa 10-15 jam.

### 5b. Jalankan nb 85 (Skema 2 cross-dataset)

Setelah nb 84 selesai:

```bash
jupyter notebook notebooks/85_threeclass_skema2_crossdataset.ipynb
```

Inference-only, ~30 menit.

---

## Step 6: Sync Hasil Kembali ke Laptop

```bash
# Di LAPTOP, tarik hasil dari VPS:
scp -r user@vps-ip:~/MultimodalEmoLearn/models/benchmark/all_3c_skema1_results.json docs/
scp -r user@vps-ip:~/MultimodalEmoLearn/models/benchmark/all_3c_skema2_cross_results.json docs/
scp -r user@vps-ip:~/MultimodalEmoLearn/notebooks/results/84_* notebooks/results/
scp -r user@vps-ip:~/MultimodalEmoLearn/notebooks/results/85_* notebooks/results/
```

---

## Ringkasan Checklist

| # | Item | Status |
|---|---|---|
| 1 | Clone repo `MultimodalEmoLearn` | ⬜ |
| 2 | Python venv + PyTorch CUDA 12.1 | ⬜ |
| 3 | pip install scikit-learn, matplotlib, openpyxl, etc. | ⬜ |
| 4 | Upload `data/dataset_frontonly_conf60/` (5.1 GB) | ⬜ |
| 5 | Upload `data/benchmark/rafdb_7class/` (8.2 GB) | ⬜ |
| 6 | Upload `data/benchmark/kdef_7class/` (1.9 GB) | ⬜ |
| 7 | Upload `models/benchmark/` (existing CK+/JAFFE results) | ⬜ |
| 8 | Restore .pth checkpoints ke `models/frontonly_conf60/3class/` | ⬜ |
| 9 | Verify `torch.cuda.is_available()` == True | ⬜ |
| 10 | Run nb 84 → nb 85 | ⬜ |
| 11 | Download hasil balik ke laptop | ⬜ |
