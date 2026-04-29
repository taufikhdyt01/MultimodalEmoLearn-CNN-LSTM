# Setup GitHub Auto-Backup (Persistent Storage)

Streamlit Community Cloud (free tier) punya **filesystem ephemeral** — semua file yang ditulis runtime hilang saat app sleep + wake up. Solusi: auto-commit hasil validasi ke GitHub repo via Contents API.

> **Catatan repo:** target backup adalah repo deploy **terpisah** (default `taufikhdyt01/emotion-validation`), **bukan** repo riset utama `MultimodalEmoLearn`. Repo deploy adalah repo yang Streamlit Cloud baca untuk menjalankan app ini.

## Langkah Setup (one-time)

### 1. Buat GitHub Personal Access Token (PAT)

1. Buka: https://github.com/settings/tokens
2. Klik **"Generate new token"** → pilih **"Generate new token (classic)"**
3. **Note:** "Streamlit Validation Backup"
4. **Expiration:** 90 days (atau "No expiration" kalau mau permanent)
5. **Scopes (centang):**
   - `repo` (Full control of private repositories) — perlu untuk read+write files di repo deploy
6. Klik **"Generate token"**
7. **Copy token** (dimulai dengan `ghp_...`) — **simpan!** Tidak bisa dilihat lagi setelah leave page.

### 2. Tambahkan secrets ke Streamlit Cloud

1. Buka Streamlit Cloud dashboard: https://share.streamlit.io
2. Klik app `emotion-validation` → **Settings** → **Secrets**
3. Tambahkan baris berikut (minimal hanya `GITHUB_TOKEN`, sisanya optional kalau pakai default):
   ```toml
   GITHUB_TOKEN = "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

   # Optional override (default sudah benar untuk repo emotion-validation):
   # GITHUB_REPO        = "taufikhdyt01/emotion-validation"
   # GITHUB_BRANCH      = "main"
   # GITHUB_PATH_PREFIX = "data/results"
   ```
4. **Save** — Streamlit otomatis restart app dengan secret baru

### 3. Restore Data Validator yang Sudah Ada

Validator yang sudah punya hasil JSON download sebelumnya:

1. **Clone repo deploy** (terpisah dari repo riset):
   ```bash
   git clone https://github.com/taufikhdyt01/emotion-validation.git
   cd emotion-validation
   ```
2. **Letakkan file JSON** di `data/results/` (di root repo deploy):
   - Naming: `results_<validator_name_lower_underscore>.json`
   - Contoh: `results_dephilia_rambu_raja_uju_decky,_s.psi.json`
3. **Commit + push:**
   ```bash
   mkdir -p data/results
   cp /path/to/results_dephilia_rambu_raja_uju_decky,_s.psi.json data/results/
   git add data/results/
   git commit -m "Restore Dephilia validation results (128 validated)"
   git push
   ```
4. Streamlit akan **auto-redeploy** — file ada di repo deploy + load otomatis di app saat validator login

## Cara Kerja Auto-Backup

Setelah setup:
- **Tiap save validator** → app menulis JSON ke filesystem lokal **+** commit ke repo deploy via API
- **Tiap load app/wake-up** → app fetch dari repo deploy (always-fresh) → cache lokal
- Container restart? → tidak masalah, data ada di repo deploy

## Verifikasi

Cek di repo deploy GitHub: setelah validator save annotation, file `data/results/results_<nama>.json` akan ada commit baru dengan message `"Update <nama> validation results (N validated)"`.

## Troubleshooting

**Warning di app:** "Backup GitHub gagal: GITHUB_TOKEN belum di-set"
- → Setup step 2 belum dilakukan, atau PAT salah/expired

**Warning:** "GitHub error 401: Bad credentials"
- → PAT invalid atau expired. Generate token baru, update secret.

**Warning:** "GitHub error 404: Not Found"
- → `GITHUB_REPO` atau `GITHUB_BRANCH` salah. Default app: `taufikhdyt01/emotion-validation` di branch `main`. Kalau repo deploy pakai branch lain (misal `master`), set `GITHUB_BRANCH = "master"` di secrets.

**Warning:** "GitHub error 403: rate limit"
- → Terlalu banyak save berturut-turut. PAT punya limit 5000 req/jam — jarang kena untuk validation use case (1 validator, slow rate).

## Kalau Kebutuhan > 5 Validator Aktif Bersamaan

Pertimbangkan migrate ke **Google Sheets backend** atau **Supabase** untuk concurrent writes yang lebih baik. Untuk single-validator atau few-validator, GitHub Contents API cukup.
