# Data Preprocessing Pipeline — Penjelasan Figure

**File:** `docs/figures/preprocessing_pipeline.png` / `.pdf`
**Script:** `scripts/make_preprocessing_pipeline_viz.py`

## Konteks

Figure ini menjelaskan **alur preprocessing data** dari rekaman video mentah hingga tensor input yang siap di-feed ke model. Cocok untuk Bab 3 Metodologi tesis di subbab "Data Preprocessing".

## Layout

Figure 1 baris × 5 stage panel + footer annotation berisi output yang disimpan.

```
(1) Raw frame → (2) Face detection → (3) Crop 224×224 → (4) Landmark → (5) Heatmap
                                                                                  ↓
                                                  Disimpan ke disk (per sample)
```

## Stage 1 — Raw video frame

Frame yang ditangkap dari rekaman **webcam Studio** saat sesi data collection. Resolusi asli bervariasi (HD/Full-HD tergantung setting); di figure ini disimulasikan dengan padding gelap di sekeliling face crop untuk memberikan gambaran "wider scene di mana wajah berada".

> **Note**: di repo, frame raw asli tidak disimpan setelah preprocessing (untuk efisiensi storage). Yang ada hanya hasil crop di `X_*_images.npy`.

## Stage 2 — Face detection

Wajah dideteksi menggunakan **MediaPipe FaceLandmarker v2** (model `face_landmarker_v2_with_blendshapes.task`). Output utama:
- Bounding box (xyxy) dari wajah pertama yang terdeteksi
- Confidence score deteksi

Hanya frame dengan **confidence ≥ 0.6** yang dipakai untuk dataset `frontonly_conf60` (sehingga subset ini lebih bersih dari false positive atau wajah berpaling/oklusi).

Bounding box di-overlay dengan warna merah pada figure untuk visualisasi.

## Stage 3 — Crop & resize 224×224

Region wajah di-crop sesuai bounding box, lalu di-resize ke **224×224×3** (float32 range [0,1]).

- **Resize**: bilinear interpolation, mempertahankan aspect ratio dengan letterbox padding bila perlu (warna padding = mean wajah)
- **Normalisasi**: pixel dibagi 255 → [0, 1] (sebelum ImageNet mean/std normalization dilakukan di model)
- Output: `X_{split}_images.npy` shape `(N, 224, 224, 3)` float32

## Stage 4 — Landmark extraction

> **Note**: Figure menampilkan **face-api.js (FA)** sebagai contoh visual (yang dominasi di top result eksperimen). Source MediaPipe (MP) **juga di-extract paralel** dan disimpan terpisah untuk perbandingan di eksperimen.

68 landmark wajah diekstrak dari **dua sumber paralel**:

| Source | Detail | File output |
|---|---|---|
| **MediaPipe (MP)** | 478 landmark 3D dari FaceLandmarker v2, di-mapping ke 68 dlib points via skrip mapping standar | `X_{split}_landmarks.npy` shape `(N, 136)` (flatten 68×2) |
| **face-api.js (FA)** | Native 68 dlib via TinyFaceDetector + Landmark68Net, dijalankan di Node.js pre-processing pipeline | `X_{split}_faceapi_landmarks.npy` shape `(N, 136)` |

Koordinat keduanya **dinormalisasi ke [0, 1] relatif terhadap crop 224×224** sehingga compatible langsung dengan tensor input model.

> Pada figure, landmark di-overlay color-coded by region: **Jaw** (biru), **Eyebrows** (hijau), **Nose** (ungu), **Eyes** (emas), **Mouth** (merah).

## Stage 5 — Heatmap rendering

Untuk Early Fusion, 68 landmark di-render menjadi **single-channel heatmap 224×224**:
1. Buat canvas kosong `(224, 224)` zeros
2. Untuk setiap titik landmark `(xi, yi)`, draw Gaussian blob dengan sigma kecil di canvas
3. Sum semua 68 Gaussian → heatmap final
4. Normalisasi ke [0, 1]

Output: `X_{split}_heatmaps.npy` shape `(N, 224, 224)` float32.

Heatmap divisualisasi dengan colormap **hot** (hitam → merah → kuning → putih) — tapi saat masuk ke model, di-feed sebagai 1-channel grayscale (channel ke-4 input Early Fusion).

## Derived features (tidak ditampilkan di stage 1-5 tapi dihasilkan secara paralel)

Setelah landmark di-extract, tiga feature derivasi otomatis dihitung:

| Feature | Dimensi | Sumber landmark | Tujuan |
|---|---|---|---|
| **FACS_28** | 28 | FA (biasanya) atau MP | Euclidean distance antar pasangan landmark berbasis FACS Action Units, dinormalisasi dengan interocular distance |
| **Blendshape_52** | 52 | MP only (output langsung dari FaceLandmarker blendshape head) | ARKit-style blendshape coefficients |
| **FACS+BS_80** | 80 | concat FACS_28 + Blendshape_52 | hand-crafted + deep-learned hybrid |

File: `X_{split}_facs.npy`, `X_{split}_mp_blendshapes.npy`. Detail visualisasi feature variants ada di `feature_decomposition_overview.png`.

## Output yang disimpan ke disk per sample

Tabel ringkas semua artifact preprocessing:

| File | Shape | dtype | Konten |
|---|---|---|---|
| `X_{split}_images.npy` | (N, 224, 224, 3) | float32 | RGB face crop |
| `X_{split}_landmarks.npy` | (N, 136) | float32 | MP 68-point coords (flatten) |
| `X_{split}_faceapi_landmarks.npy` | (N, 136) | float32 | FA 68-point coords (flatten) |
| `X_{split}_heatmaps.npy` | (N, 224, 224) | float32 | Landmark heatmap |
| `X_{split}_mp_blendshapes.npy` | (N, 52) | float32 | ARKit blendshape coefficients |
| `X_{split}_facs.npy` | (N, 28) | float32 | FACS Euclidean distances (pre-computed) |
| `y_{split}.npy` | (N,) | int64 | Hard label 7-class |
| `y_{split}_soft.npy` | (N, 7) | float32 | Soft label (multi-annotator agreement) |

`split` ∈ {`train`, `val`, `test`}. Total Primer: 5287 train / 579 val / 929 test = **6795 sample dari 37 user** (per-user split, no leakage).

## Filter & quality control

Sebelum sample masuk ke dataset final:
1. **Confidence MP ≥ 0.6** (filter face detection confidence)
2. **Frontonly**: hanya frame yang wajahnya menghadap kamera (filter via head pose estimation dari landmark)
3. **Per-user split**: train/val/test dipisah berdasarkan user ID, bukan sample-level. Memastikan tidak ada subject leakage.

## Penggunaan di tesis

Figure ini cocok untuk **Bab 3 Metodologi**, subbab pertama:

- **Section 3.1 "Data Collection"** — referensi singkat ke webcam recording (stage 1)
- **Section 3.2 "Preprocessing Pipeline"** — figure utama section ini, jelaskan stage 1–5
- **Section 3.3 "Feature Representation"** — referensi ke "Derived features" + cross-reference ke `feature_decomposition_overview.png`

## Re-generate

```bash
python scripts/make_preprocessing_pipeline_viz.py             # default sample (neutral confidence tertinggi)
python scripts/make_preprocessing_pipeline_viz.py --sample-idx 1234
python scripts/make_preprocessing_pipeline_viz.py --dpi 300   # higher resolution
```
