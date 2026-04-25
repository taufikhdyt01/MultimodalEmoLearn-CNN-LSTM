# Data Request — Hasil Training 3-Class untuk Paper JITeCS

Kompilasi lengkap untuk Pass 29 paper update. Semua angka **val-based selection** (proper methodology, no test leakage).

Source files:
- `models/frontonly_conf60/3class/all_results_3class.json` (15 configs 3-class)
- `models/frontonly_conf60/7class/*.json` + `7class_tl/*.json` + `early_fusion/early_fusion_7c_results.json`
- `docs/figures/confusion_matrix.{png,pdf}` — 2-panel figure (a) 7c + (b) 3c

---

## 1. §2.1 Dataset — Revised Class Distribution

### 1.1 Class Scheme (3-class valence mapping, Russell 1980 circumplex)

| Original (7-class) | 3-class mapping |
|---|---|
| happy, surprised | **positive** (0) |
| neutral | **neutral** (1) |
| sad, angry, fearful, disgusted | **negative** (2) |

**Konfirmasi sesuai kesepakatan.** Tidak ada deviasi.

### 1.2 Per-Split Distribution (REMAP_3 ke dataset_frontonly_conf60)

| Split | positive | neutral | negative | **Total** |
|---|:---:|:---:|:---:|:---:|
| train | 432 | 4,526 | 329 | **5,287** |
| val | 72 | 477 | 30 | **579** |
| test | 186 | 688 | 55 | **929** |
| **Total** | **690** | **5,691** | **414** | **6,795** |

**Imbalance ratio (max/min di train):** 4526/329 = **13.8:1** (vs 4-class 62:1, vs 7-class 1138:1)

### 1.3 Augmented Train (Scenario B3 — weights + augmentation)

Augmentation script: `src/preprocessing/augment_conf60_3class.py --target-min 1500`. Teknik: horizontal flip, rotation ±5-15°, brightness adjust ±5-20%, kombinasi flip+rotasi.

| Split | positive | neutral | negative | **Total** |
|---|:---:|:---:|:---:|:---:|
| train (augmented) | 1,500 | 4,526 | 1,500 | **7,526** |
| val (no aug) | 72 | 477 | 30 | **579** |
| test (no aug) | 186 | 688 | 55 | **929** |

**Total samples = 6,795** — sama dengan 7-class run. Tidak ada drop. REMAP dilakukan di label saja, sample selection identik (confidence ≥ 60% Face API filter).

### 1.4 Konfirmasi Konsistensi vs Run Sebelumnya

- ✅ Train/val/test split identik dengan run 7-class & 4-class (stratified per-subject)
- ✅ Total samples identik (6,795)
- ✅ Confidence filter identik (≥ 60%)
- ✅ Image preprocessing identik (224×224×3, min-max [0,1])
- ✅ Landmark preprocessing identik (MediaPipe 68-point, normalized [0,1])
- ✅ Heatmap generation identik (Gaussian σ=3, 224×224 single channel)

---

## 2. Tables 2 & 3 — Overall Results

### 2.1 7-Class Results (baseline comparison column)

**Backbone:** ResNet-18 (pretrained ImageNet) untuk TL variants; scratch CNN (5-block) dan FCNN (5-layer MLP) untuk non-TL. Details di §2.3 Architecture.

| experiment_id | backbone | fusion_variant | class_scheme | scenario | val_macro_f1 | test_acc | test_macro_f1 | test_weighted_f1 |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 7c-cnn-b1 | ResNet-scratch (5-block) | CNN single | 7class | B1 | — | 0.811 | **0.277** | 0.809 |
| 7c-cnn-b2 | ResNet-scratch | CNN single | 7class | B2 | — | 0.774 | 0.240 | 0.767 |
| 7c-cnn-b3 | ResNet-scratch | CNN single | 7class | B3 | — | 0.785 | 0.253 | 0.782 |
| 7c-fcnn-b1 | MLP (5-layer) | FCNN single | 7class | B1 | — | 0.768 | 0.232 | 0.765 |
| 7c-fcnn-b2 | MLP | FCNN single | 7class | B2 | — | 0.765 | 0.244 | 0.767 |
| 7c-fcnn-b3 | MLP | FCNN single | 7class | B3 | — | 0.740 | 0.222 | 0.758 |
| 7c-interm-b1 | CNN+FCNN scratch | Intermediate (feature concat) | 7class | B1 | — | 0.792 | 0.261 | 0.791 |
| 7c-interm-b2 | CNN+FCNN scratch | Intermediate | 7class | B2 | — | 0.779 | 0.247 | 0.784 |
| 7c-interm-b3 | CNN+FCNN scratch | Intermediate | 7class | B3 | — | 0.775 | 0.229 | 0.754 |
| 7c-late-b1 | CNN+FCNN scratch | Late Fusion (val-tuned w) | 7class | B1 | — | 0.816 | 0.270 | 0.812 |
| 7c-late-b2 | CNN+FCNN scratch | Late Fusion | 7class | B2 | — | 0.777 | 0.248 | 0.778 |
| 7c-late-b3 | CNN+FCNN scratch | Late Fusion | 7class | B3 | — | 0.740 | 0.222 | 0.758 |
| 7c-cnn-tl-b1 | ResNet-18 ImageNet | CNN TL single | 7class | B1 | — | 0.793 | 0.273 | 0.782 |
| 7c-cnn-tl-b2 | ResNet-18 ImageNet | CNN TL | 7class | B2 | — | 0.750 | 0.243 | 0.746 |
| 7c-cnn-tl-b3 | ResNet-18 ImageNet | CNN TL | 7class | B3 | — | 0.807 | 0.241 | 0.797 |
| 7c-interm-tl-b1 | ResNet-18 + FCNN | Intermediate TL | 7class | B1 | — | 0.792 | 0.277 | 0.800 |
| 7c-interm-tl-b2 | ResNet-18 + FCNN | Intermediate TL | 7class | B2 | — | 0.825 | 0.283 | 0.825 |
| 7c-interm-tl-b3 | ResNet-18 + FCNN | Intermediate TL | 7class | B3 | — | 0.825 | 0.292 | 0.826 |
| 7c-late-tl-b1 | ResNet-18 + FCNN | Late Fusion TL (val-tuned w) | 7class | B1 | — | 0.790 | 0.238 | 0.784 |
| 7c-late-tl-b2 | ResNet-18 + FCNN | Late Fusion TL | 7class | B2 | — | 0.782 | 0.249 | 0.781 |
| 7c-late-tl-b3 | ResNet-18 + FCNN | Late Fusion TL | 7class | B3 | — | 0.762 | 0.232 | 0.780 |
| 7c-early-b1 | 4-channel CNN scratch | Early Fusion (RGB+heatmap) | 7class | B1 | — | 0.794 | 0.246 | 0.786 |
| 7c-early-b2 | 4-channel CNN | Early Fusion | 7class | B2 | — | 0.520 | 0.205 | 0.552 |
| 7c-early-b3 | 4-channel CNN | Early Fusion | 7class | B3 | — | 0.680 | 0.264 | 0.726 |
| 7c-early-tl-b1 | ResNet-18 (4-ch modified) | Early Fusion TL | 7class | B1 | — | 0.713 | 0.253 | 0.722 |
| 7c-early-tl-b2 | ResNet-18 (4-ch) | Early Fusion TL | 7class | B2 | — | 0.636 | 0.247 | 0.663 |
| **7c-early-tl-b3** ⭐ | **ResNet-18 (4-ch)** | **Early Fusion TL** | **7class** | **B3** | — | **0.753** | **0.333** | **0.773** |

**7-class juara (test macro):** Early Fusion TL B3 = **0.333** (best overall di 7-class setup).

> **Note:** Val macro not logged untuk 7-class run lama — bisa di-backfill via re-inference kalau dibutuhkan (`scripts/export_new_best_predictions.py` template).

### 2.2 3-Class Results (main experiment column, val-based selection)

**Hyperparam identik dengan run 4-class** (lihat §5). Selection by val macro F1.

> **Status coverage (per Apr 2026):**
> - ✅ **15 configs DONE** (nb 79): FCNN + 4 TL variants × B1/B2/B3
> - ⏳ **12 configs PENDING** (nb 82 prepared, belum di-run di VPS):
>   CNN scratch / Intermediate scratch / Late Fusion scratch / Early Fusion scratch × B1/B2/B3
>
> Tabel di bawah ini hanya yang sudah DONE. Setelah nb 82 selesai → master JSON `all_results_3class_full.json` akan punya full 27-config grid (mirror 4-class & 7-class earlier coverage).

| experiment_id | backbone | fusion_variant | class_scheme | scenario | val_macro_f1 | test_acc | test_macro_f1 | test_weighted_f1 | best_epoch | w_best |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 3c-fcnn-b1 | MLP (5-layer) | FCNN single | 3class | B1 | 0.6025 | 0.7406 | 0.5893 | 0.7589 | — | — |
| 3c-fcnn-b2 | MLP | FCNN single | 3class | B2 | 0.5641 | 0.7018 | 0.5750 | 0.7384 | — | — |
| 3c-fcnn-b3 | MLP | FCNN single | 3class | B3 | 0.6193 | 0.7492 | 0.6342 | 0.7702 | — | — |
| 3c-cnn-tl-b1 | ResNet-18 ImageNet | CNN TL single | 3class | B1 | 0.4927 | 0.7912 | 0.6340 | 0.7951 | — | — |
| 3c-cnn-tl-b2 | ResNet-18 | CNN TL | 3class | B2 | 0.5150 | 0.5813 | 0.5161 | 0.6432 | — | — |
| **3c-cnn-tl-b3** | **ResNet-18** | **CNN TL** | **3class** | **B3** | 0.4953 | **0.8396** | **0.7055** | **0.8373** | — | — |
| 3c-interm-tl-b1 | ResNet-18 + FCNN | Intermediate TL | 3class | B1 | 0.5537 | 0.7901 | 0.6856 | 0.8073 | — | — |
| 3c-interm-tl-b2 | ResNet-18 + FCNN | Intermediate TL | 3class | B2 | 0.5593 | 0.8149 | 0.6645 | 0.8217 | — | — |
| 3c-interm-tl-b3 | ResNet-18 + FCNN | Intermediate TL | 3class | B3 | 0.5005 | 0.8278 | 0.6891 | 0.8305 | — | — |
| 3c-late-tl-b1 | ResNet-18 + FCNN | Late Fusion TL (val-tuned w) | 3class | B1 | 0.6093 | 0.7966 | 0.6526 | 0.8080 | — | 0.25 |
| 3c-late-tl-b2 | ResNet-18 + FCNN | Late Fusion TL | 3class | B2 | 0.5914 | 0.7740 | 0.6456 | 0.7950 | — | 0.30 |
| **3c-late-tl-b3** ⭐val | **ResNet-18 + FCNN** | **Late Fusion TL** | **3class** | **B3** | **0.6229** | 0.7836 | 0.6370 | 0.7947 | — | 0.15 |
| 3c-early-tl-b1 | ResNet-18 (4-ch) | Early Fusion TL | 3class | B1 | 0.5371 | 0.7826 | 0.5872 | 0.7858 | — | — |
| 3c-early-tl-b2 | ResNet-18 (4-ch) | Early Fusion TL | 3class | B2 | 0.5039 | 0.6502 | 0.5840 | 0.6989 | — | — |
| 3c-early-tl-b3 | ResNet-18 (4-ch) | Early Fusion TL | 3class | B3 | 0.5088 | 0.8202 | 0.6988 | 0.8282 | — | — |

**Pending (nb 82 prepared, belum di-run):**

| experiment_id | backbone | fusion_variant | class_scheme | scenario | Status |
|---|---|---|:---:|:---:|:---:|
| 3c-cnn-b1, b2, b3 | 5-block CNN scratch | CNN single | 3class | B1/B2/B3 | ⏳ PENDING (nb 82) |
| 3c-interm-b1, b2, b3 | CNN+FCNN scratch | Intermediate (concat) | 3class | B1/B2/B3 | ⏳ PENDING (nb 82) |
| 3c-late-b1, b2, b3 | CNN+FCNN scratch | Late Fusion (val-tuned w) | 3class | B1/B2/B3 | ⏳ PENDING (nb 82) |
| 3c-early-b1, b2, b3 | 4-channel CNN scratch | Early Fusion (RGB+heatmap) | 3class | B1/B2/B3 | ⏳ PENDING (nb 82) |

**3-class juara (val-based, dari 15 config DONE):** Late Fusion TL B3 — val=0.6229, test=0.6370, acc=0.784. Mungkin shift kalau 12 scratch configs nb 82 punya juara baru (unlikely karena 4-class & 7-class pattern: TL > scratch konsisten).

**3-class juara (test-based sanity check):** CNN TL B3 — test=0.7055 (lebih tinggi dari val-based juara, tapi val CNN TL B3 = 0.4953 di bawah Late Fusion val 0.6229). Val-test mismatch ini worth mentioning di Discussion — confirm val-based selection tetap proper methodology meski test angka ranking beda.

> **Best epoch:** belum di-log per config di nb 79 output. Bisa di-backfill dari training logs kalau dibutuhkan.

### 2.3 Summary Best per Class Scheme (val-based)

| Class scheme | Juara config | Val F1 | Test F1 | Δ vs 7c |
|---|---|:---:|:---:|:---:|
| 7-class | Early Fusion TL B3 | — | 0.333 | — |
| 4-class (archived) | Intermediate TL B3 | 0.489 | 0.521 | +0.188 |
| **3-class** | **Late Fusion TL B3** | **0.623** | **0.637** | **+0.304** |

---

## 3. §3 Results & Discussion — Per-Class Analysis (Best 3-Class Model)

### 3.1 Per-Class Metrics (Late Fusion TL B3, 3-class val-based juara)

| class | precision | recall | F1 | support |
|---|:---:|:---:|:---:|:---:|
| positive | 0.607 | 0.930 | **0.735** | 186 |
| neutral | 0.935 | 0.776 | **0.848** | 688 |
| negative | 0.288 | 0.382 | **0.328** | 55 |
| **Macro avg** | **0.610** | **0.696** | **0.637** | 929 |
| Weighted avg | 0.844 | 0.784 | 0.795 | 929 |

### 3.2 Confusion Matrix (3×3)

```
                predicted
              positive  neutral  negative
   positive      173      10        3       (n=186)
true neutral    105      534       49       (n=688)
   negative       7       27       21       (n=55)
```

Markdown table format:

| | pred: positive | pred: neutral | pred: negative |
|---|:---:|:---:|:---:|
| **true: positive** (n=186) | **173** | 10 | 3 |
| **true: neutral** (n=688) | 105 | **534** | 49 |
| **true: negative** (n=55) | 7 | 27 | **21** |

CSV format:
```csv
,positive,neutral,negative
positive,173,10,3
neutral,105,534,49
negative,7,27,21
```

**Visualisasi:** `docs/figures/confusion_matrix.{pdf,png}` — 2-panel figure (a) 7-Class + (b) 3-Class side-by-side, generated via `scripts/make_confusion_matrix_figure.py`.

### 3.3 Misclassification Pattern

| Pattern | Count | % of class | Interpretasi |
|---|:---:|:---:|---|
| neutral → positive (false) | 105 | 15.3% of neutral | Dominant misclassification. Neutral expressions dengan sedikit smile/surprise sign sering di-classify positive. Typical in natural programming data where users have subtle positive reactions |
| neutral → negative (false) | 49 | 7.1% of neutral | Occasional. Neutral dengan frown/concentration mannerism kadang di-classify negative |
| negative → neutral (miss) | 27 | 49.1% of negative | Biggest negative recall issue. Frustrasi mild sering tidak distinguishable dari neutral deep concentration |
| positive → neutral (miss) | 10 | 5.4% of positive | Low — positive classes (smile, surprise) biasanya distinct |
| **Cross-valence confusion total** | 13 (3+7+10) | 1.4% of all | **Rendah** — model belajar valence distinction dengan baik |

**Key observation untuk Discussion:**
- Model menunjukkan **strong valence discrimination** — positive ↔ negative confusion hampir nol (13 kasus out of 929). Konsisten dengan Russell 1980 circumplex model: valence dimension paling prominent di facial expression.
- **Negative recall rendah (0.382)** karena boundary ambiguity ke neutral. Natural programming context: mild frustration sulit dipisahkan dari deep focus (keduanya appear neutral di Face API detection).
- **Positive recall tinggi (0.930)** karena smile/surprise = high-intensity facial action units yang distinct dari neutral baseline.

---

## 4. Table 4 — Best Config per Class (3-class)

Analysis lintas 15 configs 3-class, best F1 per kelas:

| class | best_backbone | best_fusion_variant | best_scenario | best_F1 | (config vs val-juara Late Fusion TL B3) |
|---|---|---|:---:|:---:|---|
| positive | ResNet-18 ImageNet | CNN TL single | B3 | **0.806** | +0.071 (Late Fusion TL B3 = 0.735) |
| neutral | ResNet-18 ImageNet | CNN TL single | B3 | **0.897** | +0.049 (Late Fusion TL B3 = 0.848) |
| negative | CNN+FCNN scratch | Intermediate TL | B1 | **0.513** | +0.185 (Late Fusion TL B3 = 0.328) |

**Observasi:**
- **CNN TL B3** juara di positive + neutral (test-based) — konfirmasi image stream cukup discriminative untuk dominant valence categories.
- **Intermediate TL B1** juara di negative — feature-level fusion + TL + baseline (tanpa class weights merusak learning) paling robust untuk minority class. Gain +0.185 vs val-juara.
- **Heterogenitas best config per kelas** mengindikasikan trade-off: single config val-juara (Late Fusion TL B3) tidak dominan di setiap kelas. Bisa dipertimbangkan ensemble per-class atau weighted averaging di deployment.

---

## 5. §2.4 Training and Experimental Design — Consistency Check

**Semua hyperparameter identik** dengan run 4-class sebelumnya (supaya results comparable).

### 5.1 Hyperparameters

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate (TL variants) | 5e-5 (ResNet-18 backbone fine-tune) |
| Learning rate (scratch variants) | 1e-4 (CNN/FCNN from scratch) |
| Learning rate (FCNN branch in Late Fusion) | 1e-4 |
| Weight decay | 1e-4 (all) |
| LR scheduler | ReduceLROnPlateau (mode='max' on val macro F1, factor=0.5, patience=8, min_lr=1e-7) |
| Batch size | 32 |
| Epochs (max) | 50 |
| Early stopping patience | 15 (on val macro F1) |
| Loss function (B1) | CrossEntropyLoss (unweighted) |
| Loss function (B2) | CrossEntropyLoss with class weights |
| Loss function (B3) | CrossEntropyLoss with class weights + augmented train set |
| Class weights computation | inverse frequency normalized — `w_i = N / (K × n_i)` then normalized to sum K |
| Selection criterion | best val macro F1 checkpoint → evaluated on test |
| Train/val/test split | 78% / 8.5% / 13.5% (stratified per-subject, identik dengan run 4-class/7-class) |

### 5.2 Class Weights per Scenario (3-class)

**B2 (unaugmented, weights from original 3-class train):**
```python
counts_train = [432, 4526, 329]    # positive, neutral, negative
w_B2_normalized = [2.55, 0.243, 3.34]  # inverse frequency × normalization
```

**B3 (augmented train):**
```python
counts_aug = [1500, 4526, 1500]    # positive, neutral, negative
w_B3_normalized = [1.05, 0.349, 1.05]  # near-uniform karena sudah balanced
```

### 5.3 Augmentation Strategy

**Same as 4-class run**: minority class upsampling via image transformations.

- Techniques: horizontal flip, rotation (±5-15°), brightness adjust (0.80-1.20 factor), flip+rotation combo
- Landmark: match image geometric transformation (flip: x = 1-x; rotation: around center (0.5, 0.5))
- Heatmap: match image geometric transformation (flip, rotation)
- Target: minimum 1,500 samples per minority class (positive, negative)
- Script: `src/preprocessing/augment_conf60_3class.py`
- Augmented train size: 7,526 (vs original 5,287)

### 5.4 Architecture Summary (unchanged from 4-class run)

| Fusion Variant | Image stream | Landmark stream | Fusion layer |
|---|---|---|---|
| CNN (scratch) | 5-block CNN (32→64→128→256→512) | — | — |
| FCNN | — | 5-layer MLP (256→512→512→256→128) | — |
| CNN TL | ResNet-18 ImageNet + FC(512→256) | — | — |
| Intermediate (scratch) | 5-block CNN → 256-d | FCNN → 128-d | concat 384 → 512 → 256 → K |
| Intermediate TL | ResNet-18 → FC(512→256) | FCNN → 128-d | concat 384 → 512 → 256 → K |
| Late Fusion (scratch) | 5-block CNN → softmax | FCNN → softmax | weighted avg `w·p_img + (1-w)·p_lmk` |
| Late Fusion TL | ResNet-18 TL → softmax | FCNN → softmax | weighted avg, w val-tuned grid [0.00:0.05:1.00] |
| Early Fusion (scratch) | CNN(4-channel input = RGB + heatmap) | — (merged) | — |
| Early Fusion TL | ResNet-18 (first Conv 3→4 channel extended) | — (merged) | — |

---

## 6. Reproducibility & Metadata

| Item | Value |
|---|---|
| Random seed | 42 (torch + numpy + cuda) |
| PyTorch version | 2.0+ (check `torch.__version__` at runtime) |
| torchvision | 0.15+ |
| CUDA version | 11.8 (T4 on VPS) |
| Hardware | NVIDIA Tesla T4 (16 GB VRAM) |
| Training time | nb 79 15 configs: ~7-9 hours total (including Late Fusion TL 2-branch training) |
| Checkpoint file size | CNN/FCNN ~1-5 MB; ResNet-18 ~45 MB per checkpoint |
| Parameter count | CNN scratch ~1.3M; FCNN ~0.6M; ResNet-18 TL ~11.2M; Intermediate TL ~11.8M |
| Environment | conda env `mothertrain` (per guide rename) |

---

## 7. Files Location Summary

```
models/frontonly_conf60/3class/
├── all_results_3class.json              # master results JSON (15 configs)
├── results_single_and_fusion_tl.json    # same data, pre-Late Fusion merge
├── FCNN/
│   ├── fcnn_b1.pth, fcnn_b2.pth, fcnn_b3.pth
├── CNN_TL/
│   └── cnn_tl_b{1,2,3}.pth
├── Intermediate_TL/
│   └── intermediate_tl_b{1,2,3}.pth
├── Early_Fusion_TL/
│   └── early_fusion_tl_b{1,2,3}.pth
└── Late_Fusion_TL/
    ├── cnn_tl_b{1,2,3}.pth     # CNN branch per scenario
    ├── fcnn_b{1,2,3}.pth        # FCNN branch per scenario
    └── results.json             # w_best + val_macro per scenario

data/dataset_frontonly_conf60_3class_augmented/   # B3 source
notebooks/79_threeclass_exploration.ipynb
notebooks/results/79_threeclass_exploration_executed.ipynb
docs/figures/confusion_matrix.{pdf,png}   # 2-panel (a) 7c + (b) 3c
```

---

## 8. Usulan Integrasi ke Paper JITeCS

1. **§2.1 Dataset** — replace 4-class distribution table dengan 3-class (§1.2 dokumen ini)
2. **§2.4 Training Design** — no change, hyperparameter identik
3. **§3 Table 2 (baseline per-scheme)** — kolom baru 3-class (§2.2 dokumen ini)
4. **§3 Table 3 (best fusion)** — Late Fusion TL B3 val=0.623 jadi primary claim; 7-class Early Fusion TL B3 jadi secondary
5. **§3 Figure 5/6 (confusion matrix)** — pakai `docs/figures/confusion_matrix.pdf` (2-panel)
6. **§3 Table 4 (per-class best)** — §4 dokumen ini
7. **§4 Discussion** — narasi shift fusion strategy per class granularity (7c Early Fusion, 3c Late Fusion) + val-test mismatch CNN TL B3 sebagai caveat methodology

Mapping `happy + surprised → positive`, `sad + angry + fearful + disgusted → negative` harus disclosed eksplisit di §2.1 dengan citation Russell 1980 sebagai justifikasi valence dimension grouping.
