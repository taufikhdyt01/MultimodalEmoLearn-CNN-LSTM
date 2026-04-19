# Eksplorasi Lanjutan — Arahan Dosen

Catatan arahan dosen untuk eksplorasi lanjutan tesis (di luar scope paper JITeCS yang fokus Primer conf60). Semua untuk memperkuat BAB eksperimen & novelty tesis.

---

## Daftar Arahan Dosen

### 1. GradCAM Evaluation
**Tujuan:** Visualisasi bobot NN — mengetahui model fokus ke bagian citra apa (mata, mulut, alis, dll).

**Scope:**
- Target model: best Late Fusion TL 4c B3 (0.567) + CNN TL single (baseline)
- Optional: Early Fusion TL untuk bandingkan apakah heatmap channel ke-4 menggeser fokus
- Output: overlay heatmap per sampel per kelas (neutral/happy/sad/negative)

**Implementasi:**
- Library: `pytorch-grad-cam` (pip install)
- Notebook baru: `67_gradcam_analysis.ipynb`
- Target layer: last conv block ResNet18 (`layer4[-1]`)

**Estimasi:** 1-2 hari
**Priority:** ⭐⭐⭐ (cepat + visual impact kuat)

---

### 2. Space Alignment Problem
**Tujuan:** Evaluasi apakah feature space image (CNN features 256-dim) dan landmark (FCNN features 128-dim) **sejajar** di latent space.

**Motivasi:** Kalau space tidak aligned, Intermediate Fusion (concat) suboptimal → justifikasi kenapa Late Fusion (decision-level) unggul di data natural.

**Metode:**
- CCA (Canonical Correlation Analysis) antara CNN features vs FCNN features
- t-SNE / UMAP visualisasi joint features (cek apakah same-class cluster)
- Cosine similarity per-class antara paired features
- Cross-modal retrieval: given image feature, find matching landmark

**Output:**
- Visualisasi 2D embedding
- Skor alignment quantitative
- Per-class analysis

**Implementasi:** Notebook baru `68_space_alignment.ipynb`
**Estimasi:** 1 hari
**Priority:** ⭐⭐ (analytical, masuk Discussion)

---

### 3. Attention Module (CBAM / Ghost / Triplet) — 2-3 varian
**Tujuan:** Meningkatkan performa dengan attention mechanism, mengusulkan layer tambahan.

#### 3a. CBAM (Convolutional Block Attention Module)
- **Ref:** Woo et al. (ECCV 2018)
- Mekanisme: Channel attention → Spatial attention (sequential)
- Overhead: ringan (~2% FLOPs, ~25K params per block)
- Integrasi: setelah tiap residual block ResNet18
- Ref FER: Li et al. (2020) — FER with CBAM + ResNet

#### 3b. Ghost Module (GhostNet)
- **Ref:** Han et al. (CVPR 2020)
- Mekanisme: Generate feature map via cheap operations (depthwise linear). Bukan attention murni — efficient feature generation.
- Overhead: **mengurangi** FLOPs ~50%
- Integrasi: replace Conv2d di ResNet block → Ghost Bottleneck
- Ref FER: Zhang et al. (2024) — GhostNet Multimodal FER

#### 3c. Triplet Attention
- **Ref:** Misra et al. (WACV 2021)
- Mekanisme: 3 parallel branches cross-dimension (C×H, C×W, H×W), no bottleneck
- Lebih simple dari CBAM, captures cross-dim dependencies

**Matriks Eksperimen:**

| Model | Backbone | Attention | Target |
|-------|----------|-----------|--------|
| CNN_TL_CBAM | ResNet18 | CBAM × 4 blocks | beat CNN TL 0.507 |
| CNN_TL_Ghost | ResNet18 | Ghost replace Conv | efficient variant |
| CNN_TL_Triplet | ResNet18 | Triplet after blocks | beat CBAM |
| LateFusion_TL_CBAM | CNN_TL_CBAM + FCNN | — | **beat 0.567** ⭐ |

**Implementasi:**
- `src/training/models.py`: tambah `EmotionCNN_CBAM`, `EmotionCNN_Ghost`, `EmotionCNN_Triplet` (+ TL variants)
- Library: `timm` punya built-in CBAM, lainnya manual (~50 line per module)
- Notebook baru: `69_attention_modules.ipynb`
- Config: 3 attention × 2 class × scratch+TL × B1 only = 12 config

**Estimasi:** 4-6 hari
**Priority:** ⭐⭐⭐⭐ (novelty tertinggi)

---

### 4. Dekomposisi Sub-Fitur Facial Expression (Geometric Features)
**Tujuan:** Feature engineering pada landmark — turunkan sub-fitur geometrik semantik berdasarkan framework psikologi.

**Referensi utama:** Liliana et al. (2019) — *Fuzzy emotion: a natural approach to automatic facial expression recognition from psychological perspective using fuzzy system*, **Cognitive Processing (Springer)**. ⭐ Penulis pertama adalah dosen di penelitian ini — wajib cite di tesis.

#### 4.1. Formulasi Geometric Features (dari Liliana 2019)

**10 facial components dari 68 landmark:**

| Code | Component | Linguistic States | Membership |
|------|-----------|-------------------|-----------|
| gf1 | Left eyebrow | lower, normal, raised | Triangular |
| gf2 | Right eyebrow | lower, normal, raised | Triangular |
| gf3 | Inner eyebrow | closer, normal | Trapezoidal |
| gf4 | Left eye | narrow, normal, wide | Triangular |
| gf5 | Right eye | narrow, normal, wide | Triangular |
| gf6 | Nose | normal, wrinkled | Trapezoidal |
| gf7 | Upper lip | thin, normal, thick | Triangular |
| gf8 | Lower lip | thin, normal, thick | Triangular |
| gf9 | Inner mouth | close, narrow, normal, open, widely open | Triangular |
| gf10 | Outer mouth | close, narrow, normal, open, widely open | Triangular |

**2 metrik geometrik per komponen:**

**(1) Eccentricity** — untuk komponen elips (mata, alis, bibir, mouth — bentuk half/full ellipse):
```
a = (P2x - P1x) / 2       ← semi-major axis (horizontal)
b = (P3y - P4y) / 2       ← semi-minor axis (vertical)
e = sqrt(a² - b²) / a     ← eccentricity ∈ [0, 1]
```
- `e ≈ 0` → circular (misal mata terbuka lebar)
- `e → 1` → highly elongated (misal mulut tertutup rapat)

**(2) Distance ratio** — untuk komponen linear (nose, inner eyebrow):
```
d = (P3y - P4y) / (P2x - P1x)
```

**Normalisasi scale-invariant:**
- Bagi semua nilai absolut dengan **face height (Fh)** dan **face width (Fw)**
- Fh = |landmark_8.y - landmark_27.y| (chin to nose bridge)
- Fw = |landmark_16.x - landmark_0.x| (left to right ear jaw)

**Output: GF vector 20-dim** (10 components × 2 metrik ecc+dist)

#### 4.2. Strategi Eksperimen

| Setup | Input Feature | Dimensi | Tujuan |
|-------|--------------|:-------:|--------|
| A (baseline) | Raw landmark | 136 | existing FCNN (= Primer 4c 0.459) |
| B | Geometric features only (Liliana 2019) | 20 | test interpretability |
| C | Raw + Geometric concat | 136 + 20 | augmented vector |
| D | Late Fusion TL + B (replace FCNN) | — | apakah beat Late TL 4c B3 = 0.567? |
| E | Late Fusion TL + C | — | best case |

#### 4.3. Implementasi

- **Script baru**: `src/preprocessing/compute_geometric_features.py`
  - Input: landmark CSV existing (tidak perlu re-extract MediaPipe)
  - Output: `X_{split}_geometric.npy (N, 20)` per dataset
  - Mapping 68-point MediaPipe ke 10 components (Table 3 Liliana 2019)
- **Model variant**: `EmotionFCNN_geometric` (input 20-dim) + `EmotionFCNN_combined` (input 156-dim)
- **Notebook baru**: `70_geometric_features.ipynb`

#### 4.4. Novelty Opportunity

Paper Liliana 2019 pakai **fuzzy rule-based** (no training). Kontribusi tesis Anda bisa jadi:
- **Replace fuzzy rules dengan FCNN / attention** pada 20-dim geometric features
- Atau: **hybrid** → FCNN untuk 20-dim geometric + CNN untuk image → Late Fusion
- Argumen: deep learning bisa belajar mapping feature → emotion lebih fleksibel dari fuzzy rules, tetap manfaatkan semantic/interpretable features

**Hubungan ke ambiguity problem:**
> Liliana (2019): *"JAFFE image labeled sad has no significant difference with neutral except inner eyebrow slightly raised"* — observasi sama dengan Primer Anda (sad F1 rendah). Paper tsb argue fuzzy logic bisa handle ambiguity. Tesis Anda bisa explore: apakah geometric features (high-level) lebih robust terhadap ambiguity dibanding raw coords (low-level)?

**Estimasi:** 2-3 hari
**Priority:** ⭐⭐⭐⭐ (upgraded — strong novelty karena dosen = penulis Liliana 2019, direct extension dari work beliau)

---

## Eksplorasi Turunan dari Paper Referensi

Ide tambahan yang muncul setelah baca Pitaloka (2017) & Liliana (2019). Diposisikan sebagai **complementary explorations** di samping 4 arahan utama dosen.

### 5. Soft Label Training dengan Face API Confidence Scores ⭐⭐⭐⭐⭐
**Sumber ide:** Liliana (2019) — concept of *mixed emotion with intensities*
**Kenapa relevan:** Anda **sudah punya** confidence score per-emosi dari Face API (dasar filter conf60) — tapi selama ini hanya dipakai sebagai filter on/off. Data mentahnya adalah **soft probability** yang bisa jadi target training.

**Setup current (hard label):**
```
Face API output: {neutral: 0.10, happy: 0.05, sad: 0.65, angry: 0.10, ...}
Current usage:   y = argmax → "sad" → one-hot [0, 0, 1, 0, 0, 0, 0]
Loss:            Cross-Entropy dengan hard label
```

**Usulan baru (soft label):**
```
y_soft = [0.10, 0.05, 0.65, 0.10, 0.05, 0.03, 0.02]   (distribusi probability)
Loss:   KL-divergence atau soft Cross-Entropy dengan distribusi target
```

**Justifikasi ilmiah:**
- Liliana (2019): *"humans express multiple emotions simultaneously"*
- Primer adalah **natural data** — emosi memang fuzzy/ambiguous, bukan one-hot
- Confidence score Face API sudah mengandung informasi ambiguity ini, selama ini dibuang

**Strategi eksperimen:**

| Setup | Target | Loss | Expected |
|-------|--------|------|----------|
| Baseline | Hard label (argmax) | CE | 0.567 (Late Fusion TL 4c B3 existing) |
| Soft-KL | Face API distribution | KL-divergence | apakah naik? |
| Soft-CE | Face API distribution | Soft CE | alternative loss |
| Label Smoothing | Hard + ε=0.1 smoothing | smooth CE | baseline middle |

**Evaluasi:**
- Multi-class metrics tetap dihitung dari argmax prediction
- Tambahan: **multi-label metrics** (precision@k, top-2 accuracy) untuk validasi mixed emotion capture

**Implementasi:**
- Preprocess: simpan confidence score lengkap (7-dim) per sampel → `y_soft_train.npy` etc
- Model: tidak berubah, hanya loss function + target format
- Notebook: `71_soft_label_training.ipynb`

**Effort:** 1-2 hari
**Priority:** ⭐⭐⭐⭐⭐ (tertinggi — data sudah ada, novelty unik, connect langsung ke Primer)

---

### 6. Fuzzy Rule-Based (FEIS) sebagai Non-DL Baseline ⭐⭐⭐⭐
**Sumber ide:** Liliana (2019) — direct replication
**Tujuan:** bandingkan deep learning (Late Fusion TL = 0.567) vs fuzzy rule-based di Primer

**Kenapa menarik:**
- Paper Liliana menunjukkan 0.90 di CK+ dengan rule-based no-training
- Primer = natural + imbalanced → deep learning struggle (0.567)
- **Hipotesis:** fuzzy rule mungkin lebih robust karena tidak overfit ke distribusi training

**Implementasi:**
1. Replikasi FFCIS (10 subsystem, Mamdani inference) per facial component
2. Replikasi FEIS (6 emotion inference engines)
3. Rules dari Table 3 paper Liliana + sesuaikan ke 4-class Primer (gabungkan rules emosi minoritas)
4. Evaluate di Primer test set 929 imgs

**Library:** `scikit-fuzzy` (Python port dari MATLAB fuzzy toolbox paper asli)

**Strategi eksperimen:**

| Model | Input | Trained? | Target |
|-------|-------|:--------:|--------|
| FEIS (Liliana replication) | 20-dim geometric features | No | benchmark non-DL |
| FCNN_geometric | 20-dim geometric features | Yes | DL equivalent |
| Late Fusion TL (current) | Image + raw landmark | Yes | SOTA current |
| Hybrid: DL voting + FEIS voting | semuanya | Partial | ensemble |

**Value ke BAB Discussion:**
- Kalau FEIS > DL di Primer → argumen kuat bahwa rule-based lebih robust untuk data imbalanced natural
- Kalau FEIS < DL → quantify the advantage of DL + justifikasi pemilihan pendekatan
- Kalau FEIS ≈ DL → ensemble opportunity

**Effort:** 2-3 hari
**Priority:** ⭐⭐⭐⭐ (cite langsung ke paper dosen, strong for Discussion, close research loop)

---

### 7. Geometric Features + Soft Label Combined ⭐⭐⭐⭐⭐
**Kombinasi ide #4 (geometric 20-dim) + #5 (soft label)** — paling ambisius, paling novel.

**Hipotesis:** High-level interpretable features (20-dim geometric Liliana) + soft target distribution = double defense against ambiguity problem di natural data.

**Pipeline:**
```
Landmark 68 points
        ↓
[Geometric feature extraction] → 20-dim GF vector
        ↓
[FCNN_geometric]  →  soft output 7-dim
        ↓
Loss: KL-divergence dengan Face API soft target
```

**Strategi eksperimen (ablation ladder):**

| Setup | Features | Labels | Expected |
|-------|----------|--------|----------|
| A (existing) | Raw 136-dim | Hard | 0.459 (FCNN conf60 baseline) |
| B (soft only) | Raw 136-dim | Soft | +? |
| C (geo only) | Geometric 20-dim | Hard | baseline Liliana-style |
| D (geo + soft) | Geometric 20-dim | Soft | **target best** |
| E | Late Fusion TL + D | Mixed | beat 0.567? |

**Effort:** 4-5 hari (include #4 + #5 sebagai pre-work)
**Priority:** ⭐⭐⭐⭐⭐ (novelty tertinggi, direct double-extension dari paper Liliana)

---

### 8. GCN Preprocessing Ablation (Quick Win)
**Sumber ide:** Pitaloka (2017) — GCN terbaik di antara normalization methods
**Scope:** 1 ablation study pada best existing model.

**Eksperimen:**

| Setup | Preprocessing | Target |
|-------|--------------|--------|
| A (existing) | Min-max `[0, 1]` | 0.567 baseline |
| B | A + Global Contrast Normalization | +? |
| C | A + Histogram Equalization | comparison |

**Formula GCN (Pitaloka Eq. 1):**
```
X' = s · (X - mean) / max(ε, sqrt(λ + (1/3rc) · Σ(X - mean)²))
```
Parameters: s=1, λ=10, ε=1e-8 (standard)

**Implementasi:** 1 hari (tambah preprocessing function + rerun best config)
**Priority:** ⭐⭐ (low-hanging fruit, good ablation)

---

## Prioritas Kerja (Ranked by Impact/Effort)

### Arahan Utama Dosen (4 item)

| # | Eksplorasi | Effort | Novelty | Priority |
|---|-----------|:------:|:-------:|:--------:|
| 1 | GradCAM evaluation | 1-2 hari | Medium (visual) | ⭐⭐⭐ |
| 2 | **Geometric features (Liliana 2019)** | 2-3 hari | **High** (extend dosen's work) | ⭐⭐⭐⭐ |
| 3 | Space alignment (CCA/t-SNE) | 1 hari | Low-Medium | ⭐⭐ |
| 4 | **Attention (CBAM/Ghost/Triplet)** | **4-6 hari** | **High** | ⭐⭐⭐⭐ |

### Turunan dari Paper Referensi (4 ide tambahan)

| # | Eksplorasi | Effort | Novelty | Priority |
|---|-----------|:------:|:-------:|:--------:|
| 5 | **Soft Label Training (confidence as target)** | 1-2 hari | ⭐⭐⭐⭐⭐ unique | **Tertinggi** |
| 6 | Fuzzy Rule FEIS baseline (Liliana replication) | 2-3 hari | ⭐⭐⭐⭐ close loop | Tinggi |
| 7 | **Geometric + Soft Label (combo #2+#5)** | 4-5 hari | ⭐⭐⭐⭐⭐ double ext | Ambisius |
| 8 | GCN Preprocessing (Pitaloka) | 1 hari | ⭐⭐ ablation | Quick win |

### Saran Urutan Eksekusi

**Phase 1 — Quick wins (2-3 hari):**
1. GCN Preprocessing (#8) — 1 hari, ablation study
2. Soft Label Training (#5) — 1-2 hari, data sudah ada, potential big win

**Phase 2 — Core exploration (5-7 hari):**
3. GradCAM (#1) — 1-2 hari, inform desain attention
4. Geometric features (#2) — 2-3 hari, extend Liliana 2019
5. Fuzzy FEIS baseline (#6) — 2-3 hari (bisa paralel dengan #2)

**Phase 3 — Advanced (4-6 hari):**
6. Attention modules (#4) — 4-6 hari, novelty tertinggi
7. Geometric + Soft Label (#7) — combo setelah #2 & #5 selesai

**Phase 4 — Analytical (1-2 hari):**
8. Space alignment (#3) — untuk BAB Discussion

---

## Relasi ke Output Tesis

| Eksplorasi | Kontribusi ke BAB |
|-----------|-------------------|
| GradCAM | BAB 4 Hasil (Figures) + BAB 5 Discussion (model interpretability) |
| Space alignment | BAB 5 Discussion (justifikasi pemilihan fusion strategy) |
| Attention modules | BAB 3 Metodologi (novelty) + BAB 4 Hasil (ablation) |
| Geometric features | BAB 3 Metodologi (feature engineering) + BAB 4 Hasil + BAB 2 Related Work (cite Liliana 2019) |

---

## Relasi ke JITeCS Paper

Semua eksplorasi ini **di luar scope JITeCS paper** (sesuai arahan dosen sebelumnya: paper fokus pada 5 fusion strategy standar di Primer conf60, 54 configs).

Kalau hasilnya impressive, bisa:
- Jadi paper lanjutan (extension / journal version)
- Atau referenced di JITeCS sebagai "future work" di Section 6 Conclusion

---

## Referensi Paper yang Diberikan Dosen

### 1. Pitaloka et al. (2017) — Enhancing CNN with Preprocessing Stage
**File:** `docs/1-s2.0-S1877050917320860-main.pdf`
**Publikasi:** Procedia Computer Science 116 (ICCSCI 2017), UI

**Kontribusi:**
- Studi komparatif 7 skenario preprocessing: raw, face crop, GCN, local norm, histogram eq, noise, combined
- **Face detection + crop = +24% accuracy** (dominant factor)
- Best: crop + noise augmentation = **97.06% accuracy** di CK+/JAFFE/MUG
- CNN sederhana: 2 conv (5×5) + 2 maxpool + FC 25 neurons

**Relevansi ke tesis:**
- Justifikasi pipeline face crop via MediaPipe (konsisten dengan temuan)
- Support argumen B3 augmentation
- Temuan "Sadness paling sulit" — konsisten dengan Primer (sad-neutral ambiguity)

**Cite di tesis:** BAB 2 Related Work (preprocessing pipeline), BAB 3 Metodologi (augmentation justification)

---

### 2. Liliana et al. (2019) — Fuzzy Emotion
**File:** `docs/s10339-019-00923-0.pdf`
**Publikasi:** Cognitive Processing (Springer, Q2), UI — **penulis pertama = dosen Anda** ⭐

**Kontribusi:**
- Fuzzy rule-based emotion recognition (no training, psychology-driven)
- **Geometric feature decomposition**: 10 facial components → 20-dim feature vector
- Formula eccentricity + distance ratio (detail di section 4.1 di atas)
- High-level linguistic features (e.g., "eyebrow raised", "eyes narrow")
- 4 dataset: CK+ 0.90, JAFFE 0.82, DISFA 0.89, IMED 0.87 (avg 0.88)
- **Multiple simultaneous emotions** — mixed emotion framework

**Relevansi ke tesis:**
- **Template langsung untuk eksplorasi #4** (geometric features) — formulasi, mapping landmark, normalization
- **Observasi ambiguity** JAFFE sad~neutral = **same as Primer problem**
- Argumen untuk interpretable high-level features vs raw coordinates
- **Wajib cite** karena penulis = dosen pembimbing/penguji

**Cite di tesis:**
- BAB 2 Related Work (geometric features + psychology perspective)
- BAB 3 Metodologi (formulasi eccentricity/distance ratio)
- BAB 5 Discussion (ambiguity problem + justifikasi high-level features)

**Novelty opportunity untuk tesis:**
- Liliana 2019 pakai **fuzzy rule-based** (no training)
- Tesis Anda bisa **extend ke deep learning**:
  - FCNN/Attention pada 20-dim geometric features
  - Hybrid: geometric features + image CNN → Late Fusion
  - Comparison: fuzzy rules vs learned mapping untuk geometric features

---

## Status

*Belum ada eksplorasi yang dijalankan. Menunggu prioritas user & compute slot.*

**Progress tracking:**

*Arahan utama:*
- [ ] #1 GradCAM — notebook `67_gradcam_analysis.ipynb`
- [ ] #2 Geometric features — script `compute_geometric_features.py` + notebook `70_geometric_features.ipynb`
- [ ] #3 Space alignment — notebook `68_space_alignment.ipynb`
- [ ] #4 Attention modules — `models.py` update + notebook `69_attention_modules.ipynb`

*Turunan paper referensi:*
- [ ] #5 Soft Label Training — notebook `71_soft_label_training.ipynb` + preprocess confidence → soft target
- [ ] #6 Fuzzy FEIS baseline — script `src/baseline/fuzzy_feis.py` + notebook `72_fuzzy_baseline.ipynb`
- [ ] #7 Geometric + Soft Label combo — setelah #2 dan #5 selesai
- [ ] #8 GCN preprocessing — function di `src/preprocessing/` + rerun best config

**Last updated:** 2026-04-19 (konsultasi dengan dosen + 2 paper referensi dibaca + eksplorasi turunan)
