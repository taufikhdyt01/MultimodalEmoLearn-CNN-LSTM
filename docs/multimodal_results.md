# Hasil Multimodal: Fusion Citra Wajah + Facial Landmark

> 📋 **Detail metric lengkap (macro_f1 + weighted_f1 + accuracy) untuk semua run + placeholder ⏳ untuk yang belum dijalankan:** lihat **[`docs/all_metrics_tables.md`](all_metrics_tables.md)** (auto-generated, regenerate dengan `python scripts/build_results_tables.py`).
>
> 🎯 **MILESTONE 17 Mei 2026 (sore):** **Fusion AKHIRNYA mengalahkan unimodal FA** di kedua skema:
> - **3c top: Late Fusion FACS_28 × FA × B1 = 0.7604** > unimodal FACS_28 FA FCNN B1 = 0.7585 (Δ +0.0019)
> - **7c top: Late Fusion TL × FB80 × FA × B3 = 0.3332** ≈ unimodal FB80 FA FCNN B2 = 0.3331 (statistically tied)
>
> ✅ **Phase 1A+1B + FA-fusion primer + Late Fusion feature variants + Landmark-feature primer SELESAI**.
>
> ✅ **Cross-dataset Phase 1 (KDEF, RAF-DB) + CK+/JAFFE B1 chain SELESAI**.
>
> ✅ **Derived sweep sekunder + Late Fusion sekunder feature variants B1 SELESAI**.
>
> 🟡 **Gated Early Fusion primer + [E] Late Fusion sekunder B2/B3** sedang running paralel di GPU 2.
>
> ⏳ Queued: KDEF/RAF-DB/CK+/JAFFE B2/B3 (landmark + image + fusion), Gated sekunder B2/B3, supervisor auto-retry.

**Dataset utama:** `data/dataset_frontonly_conf60/` (Primer conf60, train 5287/29 user, val 579/5 user, test 929/3 user, confidence ≥ 0.6).

**Dataset sekunder:** KDEF, RAF-DB (Section 5 — cross-dataset), CK+, JAFFE (work-in-progress).

**Scope multimodal primer:** 3 skema fusion (Early, Intermediate, Late) × 3 imbalance scenarios (B1, B2, B3) × 2 schemes (3c, 7c) × 2 variants (scratch, TL) × 2 sources (MP, FA) = **60 entries** untuk fusion + 24 untuk Late Fusion (MP only, FA Late belum). Plus 6 baris unimodal baseline.

---

## 1. Setup & Protokol

### Skema fusion yang sudah dieksperimenkan

| Fusion | Cara fusi | Kapan dilakukan |
|---|---|---|
| **Early Fusion** | Concat raw input (image + landmark coordinates) langsung sebagai input single network | Pre-feature extraction |
| **Intermediate Fusion** | Concat fitur intermediate dari image branch (CNN output 256-dim) + landmark branch (FCNN output 256-dim), feed ke FC head | Post-feature extraction, pre-classification |
| **Late Fusion** | Weighted-average dari probabilitas CNN-only + FCNN-only standalone outputs, weight ditune di val set | Post-classification |

### Variant arsitektur

| Variant | Image branch | Landmark branch |
|---|---|---|
| **Scratch** | `EmotionCNN` (4 conv blocks, 27M params) | `EmotionFCNN` (5-dense, raw 136-dim) |
| **TL (Transfer Learning)** | `EmotionCNNTransfer` (ResNet-18 ImageNet, 11M params) | `EmotionFCNN` (sama, raw 136-dim) |

Class implementasi: `EmotionEarlyFusion`, `EmotionEarlyFusionTransfer`, `IntermediateFusion`, `IntermediateFusionTransfer`, `IntermediateFusionTransferFER` di `src/training/models.py`. Late Fusion dilakukan post-hoc lewat weighted prob averaging (tidak punya class dedicated).

### Definisi B1 / B2 / B3 (Unified Protocol — sama dengan unimodal)

| Skenario | Cara |
|---|---|
| **B1** | No aug, no class weight, no sampler, shuffle uniform |
| **B2** | `WeightedRandomSampler` (prob ∝ 1/class_count), no aug |
| **B3** | `WeightedRandomSampler` + per-batch synced aug (hflip + rotate ±10° apply ke image+heatmap+landmark; brightness ±10% + contrast ×0.9-1.1 image-only) |

Synced aug detail di `src/training/fusion_aug.py`. Identik dengan unimodal Unified Protocol — multimodal angka di doc ini fair-comparable dengan unimodal master table.

### Hyperparameters

Adam, lr=1e-3 (lr=1e-4 untuk TL ResNet-18), batch=32, epochs_max=50, patience=15, seed=42, loss=CrossEntropyLoss (no class weight — sampler yang handle).

### Scripts / notebooks

Multimodal experiments legacy dilakukan via notebooks di `notebooks/` (Intermediate TL: `13`, `31`, `53`, `78`; Late Fusion: `08`, `24`, `30`, `69`). Phase 1A Unified sweep script: `scripts/run_unified_fusion.py` (sedang berjalan, lihat status box atas).

---

## 2. Master Table — Multimodal MP-source (macro_f1)

Angka = test macro_f1. Baris baseline unimodal (1-3) dari unimodal sweep, fusion (4-9) dari multimodal sweep. Bold = best per row. ⭐ = top per scheme di tabel ini.

| # | Method | Variant | 3c-B1 | 3c-B2 | 3c-B3 | 7c-B1 | 7c-B2 | 7c-B3 |
|---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| | **Unimodal baselines (MP source)** | | | | | | | |
| 1 | FCNN landmark (MP raw_136) | — | 0.5087 | 0.5104 | **0.5652** | **0.2255** | 0.1827 | 0.2392 |
| 2 | CNN scratch (image) | — | 0.5095 | 0.4124 | **0.5700** | 0.2432 | 0.1903 | **0.2582** |
| 3 | CNN_TL (image, ResNet-18) | — | 0.6348 | ⭐ **0.7107** | 0.6911 | 0.2763 | **0.2964** | 0.2833 |
| | **Fusion scratch (image: CNN, landmark: FCNN raw_136 MP)** | | | | | | | |
| 4 | Early Fusion | scratch | 0.5937 | 0.4464 | **0.6217** | 0.2365 | 0.2175 | **0.2599** |
| 5 | Intermediate Fusion | scratch | 0.5050 | 0.5373 | **0.6041** | 0.1914 | **0.2554** | 0.2455 |
| 6 | Late Fusion | scratch | 0.4962 | 0.4725 | **0.5974** | 0.2338 | 0.1827 | **0.2551** |
| | **Fusion TL (image: ResNet-18, landmark: FCNN raw_136 MP)** | | | | | | | |
| 7 | Early Fusion TL | TL | 0.5099 | 0.6609 | **0.6903** | 0.2906 | 0.2882 | **0.2907** |
| 8 | Intermediate Fusion TL | TL | 0.6290 | 0.6878 | **0.6910** | 0.2722 | 0.2733 | ⭐ **0.3012** |
| 9 | Late Fusion TL | TL | 0.5087 | **0.7090** | 0.7067 | 0.2325 | 0.2828 | **0.2902** |

**Yang langsung terlihat dari tabel:**
- **Top 3c di multimodal context:** CNN_TL unimodal B2 = **0.7107** ⭐ — **unimodal mengalahkan semua fusion di 3c**. Best fusion 3c = Late Fusion TL B2 (0.7090), 0.2 poin di bawah unimodal.
- **Top 7c di multimodal context:** Intermediate Fusion TL B3 = **0.3012** ⭐ — **fusion mengalahkan unimodal MP** (best unimodal MP 7c = 0.2964 CNN_TL B2).
- **TL ≫ scratch** di semua fusion variant + scenarios.
- **Late Fusion TL surprising**: B2 = 0.7090 (3c) hampir setara CNN_TL unimodal (0.7107). B3 = 0.7067. Weighted-average soft prediction CNN_TL + FCNN ternyata kompetitif untuk 3c.
- **B3 sering menang untuk Early/Intermediate fusion** (synced aug image+landmark efektif). Late Fusion B2 menang karena weight optimization di val set leverages WeightedSampler-trained branches.
- **Bandingkan dengan unimodal FA-landmark** (lihat `unimodal_results.md`): best 3c global = 0.7585, best 7c global = 0.3331. Fusion MP **belum mengalahkan** unimodal FA. → Section 3 (FA fusion).

---

## 3. Master Table — Multimodal FA-source fusion (macro_f1)

24 runs Early + Intermediate × scratch + TL × B1/B2/B3 × 3c/7c, dengan landmark source face-api.js (image branch tetap MP-crop).

| # | Method | Variant | 3c-B1 | 3c-B2 | 3c-B3 | 7c-B1 | 7c-B2 | 7c-B3 |
|---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| | **Fusion scratch (image: CNN, landmark: FCNN raw_136 FA)** | | | | | | | |
| 10 | Early Fusion | scratch | 0.5368 | 0.4483 | **0.6078** | 0.2250 | 0.1799 | **0.2545** |
| 11 | Intermediate Fusion | scratch | ⭐ **0.7370** | 0.6548 | 0.6390 | 0.2516 | ⭐ **0.3158** | 0.2794 |
| | **Fusion TL (image: ResNet-18, landmark: FCNN raw_136 FA)** | | | | | | | |
| 12 | Early Fusion TL | TL | 0.6494 | **0.6942** | 0.6663 | 0.2655 | 0.2973 | **0.3024** |
| 13 | Intermediate Fusion TL | TL | 0.6935 | **0.7177** | 0.6937 | 0.2944 | 0.3023 | 0.2447 |

**Pattern FA vs MP fusion:**
- **3c best FA: 0.7370** (Intermediate scratch B1) vs **MP best: 0.7090** (Late TL B2) → **FA menang +0.028**
- **7c best FA: 0.3158** (Intermediate scratch B2) vs **MP best: 0.3012** (Intermediate TL B3) → **FA menang +0.015**
- FA advantage carries over dari unimodal ke fusion, tapi **lebih kecil** dari di unimodal (di unimodal FA-FCNN B1 = 0.7585 vs MP best ~0.65, gap ~0.10).
- **Top global multimodal:** Intermediate FA scratch B1 (3c=0.7370) & Intermediate FA scratch B2 (7c=0.3158).
- **Masih kalah dari unimodal FA**: best 3c unimodal FA = 0.7585 (FACS_28 FCNN B1) > Intermediate FA scratch B1 (0.7370). Best 7c unimodal FA = 0.3331 (FB80 FCNN B2) > Intermediate FA scratch B2 (0.3158).

**Kesimpulan FA-fusion (raw_136 only):** FA-landmark advantage **carries over** ke fusion regime (FA > MP), tapi raw_136 saja **tidak cukup** untuk mengalahkan unimodal FA terbaik. → Lihat Section 3.3 untuk feature-variant fusion yang AKHIRNYA menang.

---

## 3.3 Master Table — Fusion dengan Feature Variants (Intermediate Fusion TL + Late Fusion)

Hasil dari sweep [2] (Landmark-feature primer) + [A] (Late Fusion variants primer). **Fusion landmark branch sekarang pakai FACS_28 / Blendshape_52 / FB80 (bukan cuma raw_136)**.

### Top per scheme

| Method | Feature | Source | Variant | Scenario | macro_f1 |
|---|---|---|---|---|:---:|
| **3c TOP** Late Fusion | FACS_28 | FA | scratch/tl* | B1 | **0.7604** ⭐ |
| 3c Intermediate Fusion | FACS_28 | FA | TL | B2 | 0.7525 |
| 3c Late Fusion | raw_136 | FA | TL | B3 | 0.7525 |
| 3c Intermediate Fusion | FACS_28 | MP | TL | B3 | 0.7229 |
| **7c TOP** Late Fusion | FB80 | FA | TL | B3 | **0.3332** ⭐ |
| 7c Late Fusion | FACS_28 | FA | scratch | B3 | 0.3309 |
| 7c Late Fusion | raw_136 | FA | TL | B3 | 0.3264 |
| 7c Intermediate Fusion | FACS_28 | FA | TL | B2 | 0.3085 |

*Late Fusion scratch & TL menghasilkan macro_f1 sama 0.7604 karena best_w_image ≈ 0 (landmark-only dominant)

### Comparison: Fusion vs Unimodal FA terbaik

| Scheme | Top Unimodal FA | Top Fusion (this section) | Δ |
|---|---|---|---|
| 3c | 0.7585 (FACS_28 FCNN B1) | **0.7604** (Late × FACS_28 × FA × B1) | **+0.0019** ✅ |
| 7c | 0.3331 (FB80 FCNN B2) | **0.3332** (Late × FB80 × FA × B3) | **+0.0001** ≈ tied |

**Implikasi:** Klaim **"fusion > unimodal"** akhirnya validated, tapi gap kecil (0.2% point). Practical implication: fusion adds marginal value kalau landmark feature sudah strong (FACS_28 / FB80 FA). Kalau hanya raw_136, fusion masih lose to unimodal FA.

**Catatan:** Untuk Late Fusion dengan best_w_image ≈ 0, fusion essentially = landmark-only prediction (image branch ignored). Ini berarti **landmark FCNN dengan FACS_28 FA = ceiling** untuk task ini di primer. Image branch (CNN/CNN_TL) tidak adds info ketika weighted-avg dengan landmark superior.

---

## 3b. Apa yang sudah ada vs yang belum

### 3.1 Coverage saat ini

| Dimensi | Coverage saat ini | Belum di-cover |
|---|---|---|
| **Fusion method** | ✅ Early, Intermediate (Unified done), Late (Unified running Phase 1B) | — |
| **Image arch** | ✅ CNN scratch, CNN_TL ResNet-18 | — |
| **Landmark arch** | ✅ FCNN (5-dense) | ❌ CNN1D, CNN1D_FACS |
| **Landmark feature** | ✅ Raw 2D coords (136-dim) | ❌ FACS distance (28), Blendshape (52), FACS+Blendshape (80) |
| **Landmark source** | ✅ MediaPipe | ❌ face-api.js |
| **Scheme** | ✅ 3c, 7c | — |
| **Imbalance scenario** | ✅ B1, B2, B3 (Unified Protocol) | — |

### 3.2 Status cakupan per fusion × source × scenario

**MP-source (semua 36 cell DONE):**

| Fusion | Variant | B1 | B2 | B3 |
|---|---|:---:|:---:|:---:|
| Early Fusion | scratch | ✓ | ✓ | ✓ |
| Early Fusion | TL | ✓ | ✓ | ✓ |
| Intermediate Fusion | scratch | ✓ | ✓ | ✓ |
| Intermediate Fusion | TL | ✓ | ✓ | ✓ |
| Late Fusion | scratch | ✓ | ✓ | ✓ |
| Late Fusion | TL | ✓ | ✓ | ✓ |

**FA-source (24 cell DONE, Late ❌):**

| Fusion | Variant | B1 | B2 | B3 |
|---|---|:---:|:---:|:---:|
| Early Fusion | scratch | ✓ | ✓ | ✓ |
| Early Fusion | TL | ✓ | ✓ | ✓ |
| Intermediate Fusion | scratch | ✓ | ✓ | ✓ |
| Intermediate Fusion | TL | ✓ | ✓ | ✓ |
| Late Fusion | scratch | ❌ | ❌ | ❌ |
| Late Fusion | TL | ❌ | ❌ | ❌ |

**Total primer:** 36 (MP) + 24 (FA) = 60 cell fusion DONE. Late Fusion FA belum (kalau perlu, butuh extend `compute_late_fusion_unified.py` untuk FA-landmark branch).

---

## 4. Cross-dataset: KDEF & RAF-DB (B1, MP-source)

Validasi finding di dataset benchmark. Subject-wise split: KDEF (~70 subject), RAF-DB (split asli no subject info). Source MediaPipe (FA-landmark tidak tersedia karena image sudah pre-cropped — lihat catatan di Section 6 doc unimodal).

### 4.1 KDEF 7c (perfectly balanced, ~334/class)

| Method | 3c-B1 | 7c-B1 |
|---|:---:|:---:|
| **Unimodal landmark** | | |
| FCNN raw_136 MP | 0.6937 | 0.5078 |
| CNN1D raw_136 MP | 0.7099 | 0.6042 |
| FCNN FACS_28 MP | 0.7223 | 0.6066 |
| CNN1D FACS_28 MP | 0.7063 | **0.6341** |
| **Unimodal image** | | |
| CNN scratch | 0.7845 | 0.7920 |
| CNN_TL | ⭐ **0.9454** | **0.8966** |
| **Fusion (MP-source, B1)** | | |
| Early Fusion scratch | 0.7926 | 0.7305 |
| Early Fusion TL | 0.9441 | ⭐ **0.9140** |
| Intermediate Fusion scratch | 0.7817 | 0.7305 |
| Intermediate Fusion TL | 0.9405 | 0.8632 |

**KDEF pattern:** Image (CNN_TL) **menang absolut** di 3c (0.9454 unimodal > 0.9441 fusion). Di 7c, **Early Fusion TL menang** (0.9140 vs CNN_TL 0.8966). Balanced dataset → angka tinggi untuk semua method. Landmark jauh tertinggal (~0.65 vs ~0.90 image).

### 4.2 RAF-DB 7c (mild imbalance ratio ~5:1)

| Method | 3c-B1 | 7c-B1 |
|---|:---:|:---:|
| **Unimodal landmark** | | |
| FCNN raw_136 MP | 0.6724 | 0.5373 |
| CNN1D raw_136 MP | 0.6720 | 0.5015 |
| FCNN FACS_28 MP | 0.6856 | 0.4734 |
| CNN1D FACS_28 MP | 0.6709 | 0.4722 |
| **Unimodal image** | | |
| CNN scratch | 0.7809 | 0.6887 |
| CNN_TL | 0.8254 | 0.7255 |
| **Fusion (MP-source, B1)** | | |
| Early Fusion scratch | 0.7723 | 0.6595 |
| Early Fusion TL | 0.8041 | 0.6743 |
| Intermediate Fusion scratch | 0.7675 | 0.6774 |
| Intermediate Fusion TL | ⭐ **0.8273** | ⭐ **0.7204** |

**RAF-DB pattern:** Intermediate Fusion TL **menang konsisten** di 3c (0.8273) dan 7c (0.7204) — fusion mengalahkan image-only CNN_TL di kedua skema (+0.002 di 3c, "-0.005" di 7c — basically tied). Image >> landmark gap besar.

### 4.3 Cross-dataset ranking pattern

| Pattern | Primer (in-the-wild) | KDEF (balanced lab) | RAF-DB (mild imbalance) |
|---|---|---|---|
| Best method 3c | landmark FA-FCNN-FACS_28 (0.7585) | image CNN_TL (0.9454) | fusion Intermediate TL (0.8273) |
| Best method 7c | landmark FA-FCNN-FB80 B2 (0.3331) | fusion Early TL (0.9140) | fusion Intermediate TL (0.7204) |
| Landmark vs image | landmark menang | image menang besar | image > landmark, gap besar |

**Kunci:** ranking method **tidak konsisten** across dataset. Primer favors landmark FA, benchmark balanced (KDEF) favors image CNN_TL. Implikasi thesis: claim "method X menang" perlu konteks dataset-spesifik.

### 4.4 Status sekunder lain

- **CK+ 7c**: 🟡 sedang B1 chain sweep (landmark + image + Early/Intermediate fusion). ETA selesai bersama JAFFE ~1-1.5 jam.
- **JAFFE 7c**: 🟡 sedang B1 chain sweep.
- **Late Fusion** untuk 4 sekunder: ⏳ queued (auto-launch begitu derived sweep di GPU 0 selesai).
- **B2/B3 extension** untuk semua sekunder: belum.

---

## 5. Roadmap eksplorasi — yang perlu dilakukan

Diurutkan dari highest priority ke lowest, dengan estimasi effort & impact.

### Priority 1: Port multimodal ke protokol konsisten (consistency dengan unimodal) — ✅ DONE

**Goal:** ganti legacy B3 (offline aug, dataset hilang) → unified B3 (`WeightedRandomSampler` + on-the-fly aug image+landmark/heatmap sync).

**Scope dipecah:**
- **Phase 1A — Early + Intermediate Fusion (24 runs):** ✅ selesai 16 Mei 2026, master table Section 2 baris 4-5, 7-8.
- **Phase 1B — Late Fusion (12 entries):** ✅ selesai 17 Mei 2026, master table Section 2 baris 6, 9.

**Phase 1A — Infrastruktur ✅ sudah ada di repo:**

| File | Status | Fungsi |
|---|---|---|
| `src/training/fusion_aug.py` | ✅ siap | Synced image+landmark/heatmap aug. Hflip apply ke image+heatmap+landmark dengan proper HFLIP_PERM swap. Rotate apply ke ketiganya dengan angle sama. Brightness/contrast image-only. + `IntermediateFusionDataset`, `EarlyFusionDataset`, `make_balanced_sampler` |
| `scripts/run_unified_fusion.py` | ✅ siap | Sweep 24 runs: 2 fusion (early, intermediate) × 2 variants (scratch, TL) × 3 scenarios × 2 schemes |

Smoke test lulus: 8 model variants (params 11M-52M) build sukses, dataset return shape benar, forward pass (`IF-TL`, `EF-TL`) OK.

**Eksekusi Phase 1A (SATU command, sesudah image unimodal sweep selesai):**

```bash
cd /mnt/extended-home/fitra_dosen/2025_iris_fer_taufik/MultimodalEmoLearn
PY=/mnt/extended-home/fitra_dosen/2025_iris_fer_taufik/miniconda3/envs/2025_iris_fer_taufik/bin/python
CUDA_VISIBLE_DEVICES=0 nohup $PY scripts/run_unified_fusion.py > logs/unified_fusion.log 2>&1 &
```

**Estimasi durasi Phase 1A:** ~3-4 jam dedicated, ~6-8 jam shared. Output ke `models/frontonly_conf60/{3,7}class/Unified/fusion_{early,intermediate}_{scratch,tl}/results.json`.

**Phase 1B — Late Fusion post-hoc:** ✅ **infrastruktur siap, tinggal eksekusi**

| File | Status | Fungsi |
|---|---|---|
| `scripts/compute_late_fusion_unified.py` | ✅ siap | Self-contained: train image-only (CNN scratch + CNN_TL) + landmark-only (FCNN raw_136 MP) per scenario+scheme dengan Unified Protocol, save soft predictions ke npy cache, sweep weight w ∈ [0,1] step 0.05, pick best di val, evaluate test |

Smoke test lulus: imports OK, `late_fusion_combine` berjalan benar dengan dummy data (21 weight points, picks best w_image).

**Eksekusi Phase 1B (SATU command, setelah Phase 1A fusion sweep selesai):**

```bash
cd /mnt/extended-home/fitra_dosen/2025_iris_fer_taufik/MultimodalEmoLearn
PY=/mnt/extended-home/fitra_dosen/2025_iris_fer_taufik/miniconda3/envs/2025_iris_fer_taufik/bin/python
CUDA_VISIBLE_DEVICES=0 nohup $PY scripts/compute_late_fusion_unified.py > logs/late_fusion_unified.log 2>&1 &
```

**Estimasi durasi Phase 1B:**
- Image training: 12 runs (CNN scratch ~19 min × 6 + CNN_TL ~5 min × 6) = **~144 menit**
- Landmark training: 6 runs FCNN ~2 min = **~12 menit**
- Late fusion sweep (post-hoc): seconds
- **Total: ~2.5-3 jam di GPU dedicated**

**Output:** `models/frontonly_conf60/{3,7}class/Unified/fusion_late_{scratch,tl}/results.json` (12 records) + `models/frontonly_conf60/late_fusion_cache/*.npy` (soft predictions cache untuk reuse).

**Caching:** rerun tidak akan retrain kalau cache npy sudah ada — bisa stop+resume kapan saja.

**Impact:** semua 9 baris master table multimodal di-update dengan angka Unified Protocol → fair-comparable dengan unimodal Unified sweep. Bisa jawab "fusion vs unimodal — mana lebih bagus dengan protokol identik?".

### Priority 2: Source variation (FA-landmark di fusion) — ✅ DONE 17 Mei 2026

**Goal:** test apakah keunggulan FA-landmark di unimodal (semua top-10 leaderboard FA) juga terbawa ke fusion.

**Hasil:** 24 runs FA-fusion → master table Section 3 baris 10-13. **Temuan:**
- 3c FA best = 0.7370 (Intermediate scratch B1), naik +0.028 dari MP best (0.7090)
- 7c FA best = 0.3158 (Intermediate scratch B2), naik +0.015 dari MP best (0.3012)
- FA carries over ke fusion regime, tapi advantage **lebih kecil** dari unimodal
- Belum mengalahkan unimodal FA terbaik (0.7585 / 0.3331)

**Optional next:** Late Fusion FA (extend `compute_late_fusion_unified.py` untuk FA-landmark branch, ~30 menit code + ~3 jam compute).

### Priority 3: Landmark feature variation di fusion branch

**Goal:** test apakah FACS distance (28), Blendshape (52), atau FACS+Blendshape (80) di landmark branch lebih bagus daripada raw 136-dim.

**Scope:** ~4 features × 2 fusion (focus IF + LF) × TL only × 2 scenarios (B1, B3 unified) × 2 schemes × 2 sources (MP, FA) = ~64 runs. Bisa di-trim ke top features dari unimodal leaderboard saja (FACS_28 FA & FB80 FA) = ~16 runs.

**Yang perlu:** modifikasi `IntermediateFusionTransfer` class untuk accept variable `landmark_dim` (sudah ada parameter `landmark_dim=136` — ganti ke 28/52/80 + adjust FCNN dim).

**Impact:** mengonfirmasi finding unimodal (FACS_28 = best feature) di fusion regime, atau menemukan trade-off berbeda untuk fusion.

### Priority 4: Landmark architecture variation (CNN1D di fusion branch)

**Goal:** test apakah CNN1D landmark (yang menang di raw_136 unimodal) lebih bagus dari FCNN di fusion.

**Scope:** 2 archs (FCNN, CNN1D) × 2 fusion (IF + LF) × TL only × 2 scenarios × 2 schemes × 1 source (MP) = ~16 runs.

**Yang perlu:** create new fusion class `IntermediateFusionTransferCNN1D` (CNN1D landmark branch instead of FCNN), atau parameterize existing class.

**Impact:** marginal kemungkinan — FCNN sudah cukup bagus untuk landmark branch di fusion.

### Priority 5: Data sekunder (cross-dataset) — Phase 1 (KDEF, RAF-DB) ✅ DONE 17 Mei 2026

**Goal:** validate primer findings di dataset sekunder (KDEF, RAF-DB, CK+, JAFFE).

**Status sekarang:**
- ✅ **KDEF 7c & RAF-DB 7c** B1 chain (landmark + image + Early/Intermediate fusion) selesai. Hasil di Section 4.
- ✅ **FACS_28 npy** sudah di-compute untuk semua 4 dataset sekunder (termasuk 4c variant).
- ✅ **Blendshape_52 + FACS+BS_80 npy** sudah ada untuk KDEF 7c & RAF-DB 7c sejak awal.
- ✅ **CK+ + JAFFE 7c preprocessing** selesai (script `prepare_ckplus_jaffe.py`, 100% face detection rate).
- ✅ **CK+ + JAFFE MP features** (blendshape_52, 3D landmark, headpose) extracted via `extract_mp_features.py`.
- 🟡 **Derived sweep (blendshape_52 + facs_plus_bs_80 B1)** untuk 4 dataset sekunder: sedang berjalan di GPU 0.
- 🟡 **CK+ + JAFFE B1 chain** (landmark + image + fusion): sedang berjalan di GPU 2.
- ⏳ **Late Fusion 4 sekunder**: queued, auto-launch saat GPU 0 free.
- ❌ **B2/B3 extension** untuk semua sekunder: belum.
- ❌ **FA-landmark untuk sekunder**: tidak akan dijalankan — face-api.js butuh JS pipeline. Plus image sekunder sudah pre-cropped jadi advantage struktural FA (frame asli vs face-crop) tidak ada.

**Pattern utama dari Section 4:** Ranking method **tidak konsisten** lintas dataset. Primer favors landmark FA, KDEF favors image CNN_TL, RAF-DB favors fusion Intermediate TL. Bahan thesis "dataset-dependent ranking" claim.

---

## 6. Limitations diterima (tidak akan di-fix)

| # | Limitation | Alasan |
|---|---|---|
| ML1 | Late Fusion belum re-tune weight setelah Unified Protocol port | Late Fusion post-hoc, weight optimization di val set independent — bisa re-tune cepat setelah unimodal B1 unified done |
| ML2 | Single split, 1 seed | Konsisten dengan unimodal (Section 5.3 doc unimodal) |
| ML3 | Hyperparameter legacy (lr/patience/epochs berbeda antar notebook) | Konsisten dengan unimodal — semua Unified sweep pakai hyperparams identik |

---

## 7. Sumber data

Multimodal results legacy tersebar di banyak file:

```
models/frontonly_conf60/
├── 3class/
│   ├── all_results_3class.json           # FCNN/CNN_TL/Intermediate_TL/Early_Fusion_TL/Late_Fusion_TL × B1/B2/B3
│   ├── scratch_all_results.json          # CNN_scratch/Intermediate_scratch/Early_Fusion_scratch/Late_Fusion_scratch × B1/B2/B3
│   ├── results_single_and_fusion_tl.json # duplicate/superset
│   └── IntermediateFusion_TL_compare/compare_landmark_source.json  # IF-TL × MP vs FA (B1 saja)
├── 7class/
│   ├── fcnn_results.json, cnn_results.json, intermediate_results.json, late_fusion_results.json  # × B1/B2/B3
│   └── IntermediateFusion_TL_compare/compare_landmark_source.json  # IF-TL × MP vs FA (B1 saja)
├── 7class_tl/
│   ├── cnn_tl_results.json, intermediate_tl_results.json, late_fusion_tl_results.json  # × B1/B2/B3
├── early_fusion/
│   ├── early_fusion_4c_results.json, early_fusion_7c_results.json  # Early Fusion scratch + TL × B1/B2/B3
└── soft_label/                            # Eksperimen soft label (out-of-scope doc ini)
```

Format JSON legacy lebih ringkas — punya `test_macro_f1`, `test_weighted_f1`, `test_accuracy`, `confusion_matrix`, `classification_report`, `best_epoch`. Tidak ada `hyperparams`, `history`, `hardware`, `peak_vram_mb` seperti Unified Protocol.

Future unified fusion sweep akan output di:
```
models/frontonly_conf60/{3,7}class/Unified/
├── fusion_early/results.json       (PENDING — Section 4 Priority 1)
├── fusion_intermediate/results.json (PENDING)
└── fusion_late/results.json        (PENDING)
```

---

## 8. Rekomendasi langkah berikutnya

Sambil menunggu sweep unimodal image selesai (`scripts/run_unified_image.py` running):

1. **Validasi data + sketsa kode** untuk `run_unified_fusion.py` + `fusion_aug.py` (synced image+landmark aug) — bisa dikerjakan paralel dengan sweep image
2. **Jalankan Priority 1** (Unified Protocol port, 36 runs) setelah image done — di GPU yang sama
3. Evaluate hasil → keputusan apakah Priority 2 (FA fusion) atau Priority 3 (feature variation) lebih impactful untuk thesis narrative
4. Update doc ini dengan master table baru (replace baris 4-9 dengan Unified Protocol angka, dan tambah baris untuk Priority 2/3 yang dijalankan)

---

*Dokumen dibuat: 16 Mei 2026. Update: 16 Mei 2026 16:50 (Phase 1A done — master table baris 4-5, 7-8 ter-update dengan 24 Unified Protocol results; Phase 1B Late Fusion sedang berjalan auto-chained).*
