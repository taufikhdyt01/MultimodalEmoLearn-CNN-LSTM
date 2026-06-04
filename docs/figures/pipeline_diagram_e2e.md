# End-to-End Pipeline Diagram — Penjelasan Figure

**File:** `docs/figures/pipeline_diagram_e2e.png` / `.pdf`
**Script:** `scripts/make_pipeline_diagram_e2e.py`

## Konteks

Figure overview untuk **Bab 3 Metodologi** — single-figure ringkasan seluruh sistem dari raw input sampai output klasifikasi emosi. Berfungsi sebagai roadmap visual yang menghubungkan komponen-komponen yang dibahas detail di figure-figure lain (preprocessing_pipeline, landmark68_per_class, feature_decomposition_overview, heatmap_landmark_overview).

## Layout — 4 section vertikal

### Section 1: Input + Preprocessing
- **Raw video frame** (1920×1080 webcam) ← detail di `preprocessing_pipeline.png` Stage 1
- **Preprocessing block** dengan 4 sub-step:
  1. Face detection (MediaPipe, confidence ≥ 0.6)
  2. Crop & resize 224×224
  3. Landmark extraction (MP + face-api.js paralel)
  4. Heatmap rendering (Gaussian per landmark)

### Section 2: Feature representations (6 stream)
Hasil preprocessing dipecah menjadi 6 representasi yang dipakai berbeda-beda oleh model paths:

| Feature | Dimensi | Sumber | Dipakai oleh |
|---|---|---|---|
| RGB image | 224×224×3 | crop | Image, Early, Intermediate, Late |
| raw_136 | 68×2 = 136 | MP/FA landmark | Landmark unimodal, Intermediate, Late |
| FACS_28 | 28 | derived dari raw_136 | Landmark unimodal, Intermediate, Late |
| Blendshape_52 | 52 | MP only | Landmark unimodal, Intermediate, Late |
| FACS+BS_80 | 80 | concat FACS+BS | Landmark unimodal, Intermediate, Late |
| Heatmap | 224×224 | dari landmark | Early Fusion |

Detail visual di `feature_decomposition_overview.png` + `heatmap_landmark_overview.png`.

### Section 3: 5 model family (kolom paralel)

#### Unimodal Image
- **Arch**: CNN scratch (4-block) / CNN_TL (ResNet-18 ImageNet)
- **Input**: RGB 224×224×3
- **Output**: softmax class

#### Unimodal Landmark
- **Arch**: FCNN (5-dense) / CNN1D (3-block Conv1d)
- **Input**: 1 dari {raw_136, FACS_28, Blendshape_52, FACS+BS_80}
- **Output**: softmax class

#### Early Fusion (input-level)
- **Input**: 4-channel tensor (RGB stack Heatmap) → CNN
- **Variant**: `concat` (channel-stack langsung) / `gated` (spatial gating sebelum conv)
- Detail: input ke channel ke-4 ditunjukkan di `heatmap_landmark_overview.png`

#### Intermediate Fusion (feature-level)
- **Input**: RGB (image branch) + landmark vector (landmark branch)
- **Image branch**: CNN scratch / CNN_TL → 256-dim
- **Landmark branch**: FCNN → 128-dim
- **Fusion**: concat (384-dim) → MLP head → softmax

#### Late Fusion (decision-level)
- Train 2 unimodal sub-model independent:
  - **CNN(image)** → softmax(P_img)
  - **FCNN(landmark)** → softmax(P_lm)
- **Fusion**: weighted average — `P_fused = w · P_img + (1−w) · P_lm` dengan w_image swept di val set

### Section 4: Output classifier

Single softmax classifier dengan 2 mode:
- **3-class** (valence): positive / neutral / negative
- **7-class** (basic): neutral / happy / sad / angry / fearful / disgusted / surprised

Setiap model family di Section 3 menghasilkan satu output yang masuk ke classifier ini.

## Color coding

| Section | Warna box | Edge color |
|---|---|---|
| Input (raw) | Gray (#dcdcdc) | Default |
| Preprocessing | Light blue (#cfe4f4) | Default |
| Feature representation | Light green (#b9e5b8) | Per-feature distinct |
| Model header strip | Light yellow (#fde4a0) | Per-model distinct |
| Model body | Cream (#fff8dc) | Per-model distinct |
| Output | Light red (#f7b8b8) | Dark red |

Border color tiap model family:
- Unimodal Image: red (#ee6666)
- Unimodal Landmark: blue (#5470c6)
- Early Fusion: dark red (#a4262c)
- Intermediate Fusion: green (#3ba272)
- Late Fusion: purple (#9b59b6)

## Arrows (dataflow)

- **Bracket-style arrow** dari preprocessing block → spread ke 6 feature boxes
- **Thin gray arrows** dari setiap feature ke model family yang pakai feature itu
- **Bold arrow** dari setiap model family → softmax classifier
- Catatan: arrows yang **bercabang banyak** (ke 5 model paths) intentionally dibikin tipis-tipis (lw=0.7) supaya tidak overwhelming visual

## Coverage eksperimen

Tiap model family di Section 3 dijalankan untuk:
- **2 skema kelas**: 3c + 7c
- **3 skenario imbalance**: B1 / B2 / B3
- **Feature variants** (kalau applicable): 4 untuk Unimodal Landmark, 4 untuk Intermediate, 4 untuk Late
- **Source variants** (kalau applicable): MP + FA untuk landmark

Total ~282 run di Primer + ratusan run per benchmark (KDEF, RAF-DB, CK+, JAFFE). Detail metrik lengkap di `docs/all_metrics_tables.md`.

## Penggunaan di tesis

Cocok untuk:
- **Bab 3.1 Overview Sistem** — figure utama yang menjelaskan arsitektur tinggi sebelum masuk detail per komponen
- **Bab 1 Pendahuluan** — kalau perlu visual ringkas di awal untuk reader yang baru masuk
- **Slide presentasi defense** — slide opening setelah problem statement

Rekomendasi caption (paraphrased dari footnote di figure):
> *End-to-end pipeline sistem multimodal emotion recognition. Raw frame webcam diproses menjadi 6 representasi feature, lalu di-feed ke 5 model family (2 unimodal + 3 fusion strategies), dan diakhiri dengan softmax classifier untuk 3 atau 7 kelas emosi. Setiap model family dijalankan pada 2 skema kelas × 3 skenario imbalance (B1/B2/B3) dengan feature & source variants yang relevan.*

## Cross-reference ke figure lain

| Section di figure ini | Figure detail terkait |
|---|---|
| Preprocessing block | `preprocessing_pipeline.png` (5 stage detail) |
| Feature representations | `feature_decomposition_overview.png` |
| Landmark visualization | `landmark68_per_class.png` |
| Heatmap (Early Fusion input) | `heatmap_landmark_overview.png` |
| RGB sample | `class_samples_all_datasets.png` |
| Per-class result | `unimodal/per_class/per_class_metrics_top_*.png` + multimodal equivalent |

## Re-generate

```bash
python scripts/make_pipeline_diagram_e2e.py
```

Diagram murni schematic (tidak ada parameter sample) — output deterministic.
