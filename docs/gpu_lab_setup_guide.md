# Setup Training di Server GPU Lab FILKOM untuk Eksperimen 3-Class yang Pending

## Status Eksekusi (Update: 10 Mei 2026 ~05:40)

> **Server GPU Lab FILKOM** (bukan cloud/VPS sewaan) — 3× NVIDIA L40 47.6 GB, shared server.
> NAS: `10.34.0.124:/mnt/FILKOM-LABKC-NAS1/Dataset_L40`
> Working dir: `/mnt/extended-home/fitra_dosen/2025_iris_fer_taufik/MultimodalEmoLearn/`

| Task | Notebook/Script | Status | GPU | Catatan |
|---|---|---|---|---|
| Skema 1 benchmark RAF-DB 3c | nb 84 | ✅ **SELESAI** 21/21 | — | `rafdb_3c_results.json` saved 10 Mei 04:08 |
| Skema 1 benchmark KDEF 3c | scripts/run_kdef_gpu1.py | ✅ **SELESAI** 21/21 | — | `kdef_3c_results.json` saved 9 Mei 10:24, master JSON updated |
| **Robustness eval 3c (Random+CV5+LOSO)** | scripts/run_eval_3c_chain.sh | 🔄 **BERJALAN** | GPU 1 | PID 2235895, started 10 Mei 05:33. Random ✅ → **CV5 berjalan** |
| Skema 2 cross-dataset → Primer 3c | nb 85 | ⬜ Siap dijalankan | - | nb 84 sudah selesai — bisa langsung run |
| History logging (optional) | nb 83 | ⬜ partial (7/27) | - | - |

### Progress RAF-DB / nb 84 (per 10 Mei 2026 05:40) — SELESAI ✅
- CK+ → **SKIP** (data tidak ada, `ckplus_3c_results.json` sudah ada)
- JAFFE → **SKIP** (sama)
- RAF-DB → ✅ **21/21 runs selesai** (`rafdb_3c_results.json` saved 10 Mei 04:08):
  - CNN b1/b2/b3 ✅ | FCNN b1/b2/b3 ✅ | Intermediate b1/b2/b3 ✅
  - CNN_TL b1/b2/b3 ✅ | Intermediate_TL b1/b2/b3 ✅
  - EarlyFusion b1/b2/b3 ✅ | EarlyFusion_TL b1/b2/b3 ✅
  - LateFusion ✅ | LateFusion_TL ✅
- KDEF → ✅ **21/21 selesai** (mulai 8 Mei 13:06, selesai 9 Mei 10:24, ~21 jam):
  - CNN, FCNN, Intermediate, CNN_TL, Intermediate_TL, EarlyFusion, EarlyFusion_TL, LateFusion, LateFusion_TL — semua ✅
  - Best test_f1: LateFusion_TL_b2 = 0.9274, Intermediate_TL_b2 = 0.9236

### Penyebab crash RAF-DB
OOM di GPU 2 — saat hendak training EarlyFusion (~9 Mei 03:25 → 10:44 antara), VRAM penuh karena banyak user lain (Process 1430794: 14.86 GiB, Process 2493641: 19.27 GiB di GPU 2). Process kita kebagian 7.5 GiB lalu gagal alokasi 98 MiB.

### Perubahan konfigurasi (8-9 Mei 2026)
- **BATCH dinaikkan 32 → 128** untuk utilisasi VRAM lebih tinggi
- **Per-run skip logic** di `run_dataset()`: cek `.pth` sudah ada → skip training, langsung load+eval. Aman restart kapan saja.
- **KDEF diparalelkan** ke GPU 1 via `scripts/run_kdef_gpu1.py` — sudah selesai
- GPU monitor otomatis: `logs/gpu_monitor.sh` (PID 1828266), output di `logs/gpu_monitor.log`

### Kontensi GPU (per 10 Mei 2026 05:32)
| GPU | Util | VRAM Used | Free | Job kita |
|---|---|---|---|---|
| 0 | 100% | 37.5 GB | 8.6 GB | — (user lain) |
| 1 | 25%  | 19.3 GB | 26.8 GB | ✅ eval_3c_chain (PID 2235895) |
| 2 | 100% | 36.5 GB | 9.6 GB | — (user lain) |

> **GPU 1 dipakai untuk eval_3c_chain** (CV5 → LOSO).

### Perintah untuk cek progress
```bash
# Cek semua proses kita
ps aux | grep -E "84_three|kdef_gpu" | grep -v grep

# Cek GPU usage
nvidia-smi

# Cek checkpoint RAF-DB
ls -lht models/benchmark/rafdb/ | head -8

# Cek checkpoint KDEF (akan muncul saat training mulai)
ls -lht models/benchmark/kdef/

# Cek log RAF-DB
tail -50 logs/nb84_run.log

# Cek log KDEF
tail -50 logs/kdef_gpu1_run.log

# Cek monitor GPU (OOM alert, utilization)
tail -30 logs/gpu_monitor.log
```

### Resume RAF-DB (nb84) — ✅ SUDAH SELESAI (10 Mei 04:08)
> `rafdb_3c_results.json` sudah tersimpan. Tidak perlu dijalankan ulang.

### Resume jika KDEF (run_kdef_gpu1.py) mati
```bash
cd /mnt/extended-home/fitra_dosen/2025_iris_fer_taufik/MultimodalEmoLearn
CUDA_VISIBLE_DEVICES=1 nohup \
  /mnt/extended-home/fitra_dosen/2025_iris_fer_taufik/miniconda3/envs/2025_iris_fer_taufik/bin/python3.12 \
  scripts/run_kdef_gpu1.py \
  > logs/kdef_gpu1_run.log 2>&1 &
```

> Resume logic: skip dataset yang sudah punya `*_3c_results.json` (level dataset), DAN skip per-run yang `.pth`-nya sudah ada (level run). Aman restart kapan saja untuk kedua proses.

---

## Robustness Evaluation 3-class (Random Split + 5-Fold CV + LOSO 37-fold)

**Script:** `scripts/run_eval_3c.py` (parametrik) + `scripts/run_eval_3c_chain.sh` (chain wrapper).

**Top-3 model (val-based selection):**
1. `late_fusion_tl` — CNN_TL + FCNN, weighted ensemble (val-tuned w)
2. `fcnn` — landmark-only baseline
3. `intermediate_tl` — IntermediateFusionTransfer

**Catatan training:**
- Semua model pakai **B2 scenario** (class weights, no augmentation) — augmented dataset tidak bisa di-regenerate per-fold secara konsisten.
- Class weights dihitung per-fold dari training data fold tersebut.
- Resume-aware per (model, fold/seed).

**Status (per 10 Mei 2026 05:40):** chain berjalan di GPU 1 (PID 2235895, started 05:33), urutan Random → CV5 → LOSO.

> **Bug fix (10 Mei):** `drop_last=shuffle` ditambahkan di `build_loader()` (`scripts/run_eval_3c.py:135`) — mencegah BatchNorm error saat batch terakhir berisi 1 sample di fold kecil.

| Tahap | Status | Hasil | Output dir |
|---|---|---|---|
| Random Split (5 seeds × 3 models) | ✅ **SELESAI** | late_fusion_tl: 0.7037±0.0106 / intermediate_tl: 0.7052±0.0141 / fcnn: 0.6459±0.0261 | `models/frontonly_conf60/3class/randomsplit/` |
| 5-Fold CV subject-wise | 🔄 **BERJALAN** | ~7 jam estimasi | `models/frontonly_conf60/3class/crossval/` |
| LOSO 37-fold | ⬜ Menunggu CV5 | ~30 jam estimasi | `models/frontonly_conf60/3class/loso/` |

**Total:** ~40 jam GPU idle (3-5 hari realistis dengan kontensi).

**Cek progress:**
```bash
tail -50 logs/eval_3c_chain.log
ls models/frontonly_conf60/3class/{randomsplit,crossval,loso}/

# Ringkasan per model (running mean ± std):
for f in models/frontonly_conf60/3class/randomsplit/*.json; do
  python3 -c "
import json
d = json.load(open('$f'))
print(f\"{d['model']:<20s} {d.get('num_seeds',d.get('num_folds',0))} runs  {d.get('macro_f1_mean',0):.4f}±{d.get('macro_f1_std',0):.4f}\")
"
done
```

**Resume jika chain mati:**
> Pastikan bug fix sudah ada (`drop_last=shuffle` di `scripts/run_eval_3c.py:135`) sebelum resume.
```bash
cd /mnt/extended-home/fitra_dosen/2025_iris_fer_taufik/MultimodalEmoLearn
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/run_eval_3c_chain.sh > logs/eval_3c_chain.log 2>&1 &
```

Atau per strategy individu:
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_eval_3c.py --strategy random --seeds 5
CUDA_VISIBLE_DEVICES=1 python scripts/run_eval_3c.py --strategy cv5
CUDA_VISIBLE_DEVICES=1 python scripts/run_eval_3c.py --strategy loso
```

---

## Pending Tasks (yang butuh server GPU lab)

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

> ✅ **SUDAH SELESAI** di server lokal — conda env `2025_iris_fer_taufik` sudah tersedia.
> Python 3.12.13, PyTorch 2.5.1+cu121, CUDA available, GPU: NVIDIA L40 (47.6 GB VRAM).

Untuk activate env yang sudah ada:
```bash
source /mnt/extended-home/fitra_dosen/2025_iris_fer_taufik/miniconda3/bin/activate 2025_iris_fer_taufik
```

---

Kalau setup dari awal di server GPU lab baru:

```bash
# Gunakan venv (lebih ringan dari conda)
python3 -m venv .venv
source .venv/bin/activate

# CUDA 12.1 build (kompatibel dengan driver CUDA 12.x di kebanyakan server GPU lab)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Dependencies lainnya
pip install numpy scikit-learn matplotlib jupyter ipykernel openpyxl pandas tqdm opencv-python mediapipe

# Verifikasi GPU
python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0)); print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory/1e9, 1), 'GB')"
```

Kalau server GPU lab pakai Python 3.12+, tambahkan `--break-system-packages` atau gunakan miniconda:

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

## Step 3: Upload Data ke server GPU lab

Pakai `rsync` atau `scp` dari laptop. **Hanya upload yang dibutuhkan untuk pending tasks.**

### 3a. Primer 3-class (wajib — untuk test set)

```bash
# Di LAPTOP:
scp -r data/dataset_frontonly_conf60 user@server-ip:~/MultimodalEmoLearn/data/
# ~5.1 GB — sudah include X_test_images.npy, X_test_landmarks.npy, X_test_heatmaps.npy, y_test.npy
```

### 3b. Benchmark dataset yang BELUM selesai

CK+ & JAFFE sudah done di laptop via nb 84 partial — upload hasilnya saja (skip data):

```bash
# Di LAPTOP — upload existing results:
scp -r models/benchmark/ user@server-ip:~/MultimodalEmoLearn/models/benchmark/
```

Dataset untuk RAF-DB & KDEF (wajib upload data):

```bash
# Di LAPTOP — RAF-DB (8.2 GB) + KDEF (1.9 GB):
scp -r data/benchmark/rafdb_7class user@server-ip:~/MultimodalEmoLearn/data/benchmark/
scp -r data/benchmark/kdef_7class  user@server-ip:~/MultimodalEmoLearn/data/benchmark/
```

> **Total upload data: ~15 GB** (Primer 5.1 + RAF-DB 8.2 + KDEF 1.9).  
> CK+ & JAFFE data tidak perlu di-upload (sudah ada hasil di `models/benchmark/`).

### 3c. Resume logic — VERY IMPORTANT

nb 84 punya resume logic: skip dataset yang sudah punya `results.json`. Kalau kamu upload `models/benchmark/` beserta `*_3c_results.json`, notebook akan **skip CK+ dan JAFFE** — langsung mulai dari RAF-DB.

---

## Step 4: Restore Checkpoint .pth Backups

Kamu sudah punya backup .pth dari server GPU lab lama. Letakkan sesuai struktur:

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

> ✅ **SUDAH DIJALANKAN** — berjalan background sejak 6 Mei 2026 23:12.

**⚠️ BUG YANG SUDAH DIPERBAIKI di nb 84:**
1. **F-string literal newline** — 4 `print(f'\n...')` dengan newline literal (bukan escape `\n`)
   → Diperbaiki dengan replace di JSON source cell
2. **`run_dataset` tidak terdefinisi** — fungsi ini hilang dari notebook
   → Ditambahkan sebagai Cell index 5 (baru) dengan logika lengkap:
   load data + load heatmaps dari file + training loop 7 arch × 3 run + Late Fusion

Jika perlu run ulang dari awal (misal di server GPU lab baru), pastikan kedua bug di atas sudah ada di notebook.
Notebook yang sudah diperbaiki ada di repo (sudah ter-edit in-place).

```bash
# Jalankan background di GPU tertentu (pakai CUDA_VISIBLE_DEVICES)
cd ~/MultimodalEmoLearn
mkdir -p logs
CUDA_VISIBLE_DEVICES=0 nohup \
  /path/to/conda/envs/train/bin/jupyter nbconvert \
  --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  notebooks/84_threeclass_skema1_benchmark.ipynb \
  > logs/nb84_run.log 2>&1 &
echo "PID: $!"
```

Resume logic akan skip CK+ & JAFFE (data tidak ada, tapi results JSON sudah ada),
langsung jalankan RAF-DB + KDEF.

**Perkiraan waktu:**
- T4 dedicated: ~6-8 jam
- L40 shared (seperti server ini): **~14 jam (estimasi awal, idle)** → realisasi **3-5 hari** kalau server full kontensi seperti 6-7 Mei 2026
- Benchmark per run RAF-DB: 33 menit (idle GPU) → ~5 jam (server full, 11% SM share)

### 5b. Jalankan nb 85 (Skema 2 cross-dataset)

Setelah nb 84 selesai:

```bash
CUDA_VISIBLE_DEVICES=0 nohup \
  /path/to/conda/envs/train/bin/jupyter nbconvert \
  --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  notebooks/85_threeclass_skema2_crossdataset.ipynb \
  > logs/nb85_run.log 2>&1 &
```

Inference-only, ~30 menit.

---

## Step 6: Sync Hasil Kembali ke Laptop

```bash
# Di LAPTOP, tarik hasil dari server GPU lab:
scp -r user@server-ip:~/MultimodalEmoLearn/models/benchmark/all_3c_skema1_results.json docs/
scp -r user@server-ip:~/MultimodalEmoLearn/models/benchmark/all_3c_skema2_cross_results.json docs/
scp -r user@server-ip:~/MultimodalEmoLearn/notebooks/results/84_* notebooks/results/
scp -r user@server-ip:~/MultimodalEmoLearn/notebooks/results/85_* notebooks/results/
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
