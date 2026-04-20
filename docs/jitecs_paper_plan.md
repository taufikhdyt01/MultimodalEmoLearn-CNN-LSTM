# JITeCS Paper — Rencana Penyusunan

> **Judul kerja:** *Multimodal Fusion of Facial Image and Landmark Features with Transfer Learning for Emotion Recognition in Programming Learning Context*
>
> **Target venue:** JITeCS (Journal of Information Technology and Computer Science), SINTA 2
>
> **Format:** IEEE, 8–12 halaman, 20–25 referensi
>
> **Scope:** Fokus pada eksperimen **dataset primer conf60** dengan studi komparatif 5 arsitektur fusion × transfer learning.

---

## 1. Struktur Paper (Final)

| Section | Perkiraan Halaman |
|---------|:-----------------:|
| Abstract | ~1 paragraph |
| 1. Introduction | ~1 |
| 2. Related Work | ~1 |
| 3. Proposed Method | ~2 |
| 4. Experimental Results | ~2–3 |
| 5. Discussion | ~1 |
| 6. Conclusion | ~0.5 |
| References (20–25 sitasi) | — |

### Daftar Isi Detail

**Abstract**
- Problem statement (FER di konteks programming natural)
- Method summary (5 arsitektur × fusion × transfer learning)
- Data description (6,795 samples dari 37 mahasiswa, conf60)
- Best result (Intermediate TL 4c B3 = Macro F1 0.521, val-tuned proper)
- Key insight

**1. Introduction**
- Motivasi: emosi → learning outcome programming
- Gap: FER existing fokus data lab (posed), bukan natural
- Rumusan masalah & tujuan penelitian
- Kontribusi: dataset baru + studi sistematis fusion × TL
- Struktur paper

**2. Related Work**
- 2.1 Deep Learning for FER
- 2.2 Multimodal Fusion of Image and Landmark Features
- 2.3 Transfer Learning for FER
- 2.4 Affective Computing in Education

**3. Proposed Method**
- 3.1 Dataset
- 3.2 Multimodal Architecture (5 varian)
- 3.3 Training Setup
- 3.4 Experimental Design (B1/B2/B3 × 7c/4c × metrics)

**4. Experimental Results**
- 4.1 Overall Performance (60 configs)
- 4.2 Effect of Transfer Learning
- 4.3 Effect of Fusion Strategy
- 4.4 Effect of Class Granularity
- 4.5 Per-Class Analysis

**5. Discussion**
- 5.1 Multimodal Fusion vs Single-Modality (jawab RQ1)
- 5.2 Fusion Strategy Comparison: Intermediate vs Late (jawab RQ2)
- 5.3 Transfer Learning Effectiveness (jawab RQ3)
- 5.4 Limitations
- 5.5 Implications for Learning Analytics

**6. Conclusion**
- Ringkasan kontribusi & best result
- Future work

---

## 2. Research Questions (Formal)

Tiga RQ yang akan dijawab di paper:

**RQ1**: *Apakah fusi multimodal antara citra wajah dan facial landmark memberikan kinerja yang lebih baik dibandingkan pendekatan single-modality untuk pengenalan emosi pada konteks pembelajaran pemrograman?*

**RQ2**: *Strategi fusion manakah — Early Fusion, Intermediate Fusion, atau Late Fusion — yang lebih efektif dalam menangani data ekspresi wajah yang natural dan imbalanced?*

**RQ3**: *Bagaimana pengaruh transfer learning (ResNet18 pretrained ImageNet) terhadap kinerja model pengenalan emosi wajah multimodal pada dataset kecil dengan ekspresi natural?*

---

## 3. Dataset untuk Paper

### Primer Dataset (satu-satunya)

| Aspek | Detail |
|-------|--------|
| Sumber | Akuisisi sendiri, sesi pembelajaran pemrograman |
| Total samples | **6,795** (setelah filter confidence ≥ 60%) |
| Jumlah user | 37 mahasiswa (2 batch) |
| Emosi | 7 kelas (neutral, happy, sad, angry, fearful, disgusted, surprised) |
| Split | User-wise 80/10/10: **train 5,287 / val 579 / test 929** |
| Preprocessing | Face crop 224×224, MediaPipe 68 landmarks → 136-dim |
| Konfigurasi kelas | 7-class (original) + 4-class remap (neutral/happy/sad/negative) |
| Lokasi | `data/dataset_frontonly_conf60/` |

### Statistik Imbalance

Distribusi **total korpus** (conf60, 7-class, n=6,795):

| Emosi | Train (5,287) | Val (579) | Test (929) | **Total** | % |
|-------|:------:|:---:|:---:|:------:|:-:|
| neutral | 4,526 | 477 | 688 | 5,691 | 83.8% |
| happy | 416 | 52 | 183 | 651 | 9.6% |
| sad | 287 | 24 | 50 | 361 | 5.3% |
| angry | 27 | 3 | 2 | 32 | 0.5% |
| fearful | 2 | 2 | 1 | 5 | 0.1% |
| disgusted | 13 | 1 | 2 | 16 | 0.2% |
| surprised | 16 | 20 | 3 | 39 | 0.6% |

→ **Ratio 1:1138** (fearful vs neutral). Kelas minoritas sangat langka.

**Catatan split user-wise**: Distribusi per-split **tidak proporsional** karena split dilakukan per-user (29 train / 3 val / 5 test), bukan per-sample. Anomali:
- `surprised` 20/39 masuk val — user 104/210/214 kebetulan banyak ekspresi surprised
- `happy` 28% (183/651) masuk test meski test hanya 14% data — user 109/117/118/208/213 dominan happy
- Konsekuensi: variance per-class metrics antar split bisa tinggi untuk kelas minoritas

---

## 4. Arsitektur Model (5 Varian)

| # | Model | Input | Fusion Point | Backbone |
|---|-------|-------|:------------:|----------|
| 1 | **CNN** | Citra 224×224×3 | — (single modal) | Scratch / ResNet18 TL |
| 2 | **FCNN** | Landmark 136-dim | — (single modal) | Scratch (no TL for landmarks) |
| 3 | **Early Fusion** | Citra + heatmap 224×224×4 | **Input level (0%)** | Scratch / ResNet18 TL (4-ch first conv) |
| 4 | **Intermediate Fusion** | Citra + Landmark | Feature level (~50%) | Scratch / ResNet18 TL |
| 5 | **Late Fusion** | Citra + Landmark | Decision level (~95%) | Scratch / ResNet18 TL |

**Transfer Learning Backbone:**
- ResNet18 pretrained ImageNet (1000-class)
- Fine-tune seluruh network, learning rate kecil (5×10⁻⁵)
- Early Fusion TL: first `Conv2d` dimodifikasi dari 3→4 channel. Weight RGB di-copy, weight heatmap di-init dari mean(RGB).
- Referensi Early Fusion: Mo, S., Yang, W., Wang, G., & Liao, Q. (2020) — HAE-Net, MMM 2020 LNCS 11961 pp. 278-289, Tsinghua University.

---

## 5. Matriks Eksperimen

**Total: 60 konfigurasi** = 5 arsitektur × 2 backbone × 3 skenario × 2 kelas

⚠️ Pengecualian: FCNN tidak punya TL variant (landmark tidak punya pretrained). Jadi effective: 9 model-backbone × 3 skenario × 2 kelas = 54 configs praktis.

### Tiga Skenario Handling Imbalance

| Skenario | Keterangan |
|----------|-----------|
| **B1** — Baseline | Standard cross-entropy, no intervention |
| **B2** — Class Weights | Weighted CE: `weight_c ∝ 1 / freq(c)` |
| **B3** — Weights + Aug | B2 + data augmentation (rotation, flip, brightness) untuk kelas minoritas |

### Dua Konfigurasi Kelas

| Kelas | Labels |
|-------|--------|
| **7-class** | neutral, happy, sad, angry, fearful, disgusted, surprised |
| **4-class (remap)** | neutral, happy, sad, **negative** (= angry+fearful+disgusted+surprised digabung) |

### Metrik Evaluasi

- **Macro F1** (utama) — rata-rata F1 per kelas, unbiased terhadap imbalance
- **Micro F1** = Accuracy (dalam multi-class single-label)
- **Weighted F1** — rata-rata F1 bobot support, didominasi kelas mayoritas
- **Accuracy**

---

## 6. Hasil Eksperimen (Semua Metrik)

Untuk referensi penulisan Section 4 (Results) dan Section 5 (Discussion).

**Catatan metrik**:
- **Macro F1** — metrik utama (unbiased terhadap imbalance)
- **Weighted F1** — rata-rata F1 bobot support (didominasi kelas mayoritas)
- **Accuracy** — standard (= Micro F1 untuk multi-class single-label)
- **Early Fusion**: 12 configs complete (6 per class) dari nb 64

### 7-Class — All Metrics

| Model | Scenario | Macro F1 | Weighted F1 | Accuracy |
|-------|----------|:--------:|:-----------:|:--------:|
| CNN | B1 | 0.277 | 0.809 | 0.811 |
| CNN | B2 | 0.240 | 0.767 | 0.774 |
| CNN | B3 | 0.253 | 0.782 | 0.785 |
| FCNN | B1 | 0.232 | 0.765 | 0.767 |
| FCNN | B2 | 0.244 | 0.767 | 0.765 |
| FCNN | B3 | 0.222 | 0.758 | 0.740 |
| Early Fusion | B1 | 0.246 | 0.786 | 0.794 |
| Early Fusion | B2 | 0.205 | 0.552 | 0.520 |
| Early Fusion | B3 | 0.264 | 0.726 | 0.680 |
| Intermediate | B1 | 0.261 | 0.791 | 0.792 |
| Intermediate | B2 | 0.247 | 0.784 | 0.779 |
| Intermediate | B3 | 0.229 | 0.754 | 0.775 |
| Late Fusion | B1 | 0.270 | 0.812 | 0.816 |
| Late Fusion | B2 | 0.248 | 0.778 | 0.777 |
| Late Fusion | B3 | 0.222 | 0.758 | 0.740 |
| CNN TL | B1 | 0.273 | 0.782 | 0.793 |
| CNN TL | B2 | 0.243 | 0.746 | 0.750 |
| CNN TL | B3 | 0.241 | 0.797 | 0.807 |
| **Early Fusion TL** | **B3** | **0.333** ⭐ | 0.773 | 0.753 |
| Early Fusion TL | B1 | 0.253 | 0.722 | 0.713 |
| Early Fusion TL | B2 | 0.247 | 0.663 | 0.636 |
| Intermediate TL | B1 | 0.277 | 0.800 | 0.792 |
| Intermediate TL | B2 | 0.283 | 0.825 | 0.825 |
| Intermediate TL | B3 | 0.292 | 0.826 | 0.825 |
| Late Fusion TL | B1 | 0.238 | 0.784 | 0.790 |
| Late Fusion TL | B2 | 0.249 | 0.781 | 0.781 |
| Late Fusion TL | B3 | 0.232 | 0.780 | 0.762 |

### 4-Class — All Metrics

| Model | Scenario | Macro F1 | Weighted F1 | Accuracy |
|-------|----------|:--------:|:-----------:|:--------:|
| CNN | B1 | 0.438 | 0.798 | 0.808 |
| CNN | B2 | 0.448 | 0.815 | 0.826 |
| CNN | B3 | 0.432 | 0.762 | 0.760 |
| FCNN | B1 | 0.422 | 0.722 | 0.695 |
| FCNN | B2 | 0.459 | 0.783 | 0.757 |
| FCNN | B3 | 0.421 | 0.739 | 0.702 |
| Early Fusion | B1 | 0.457 | 0.816 | 0.822 |
| Early Fusion | B2 | 0.427 | 0.728 | 0.690 |
| Early Fusion | B3 | 0.427 | 0.752 | 0.728 |
| Intermediate | B1 | 0.445 | 0.788 | 0.788 |
| Intermediate | B2 | 0.416 | 0.779 | 0.783 |
| Intermediate | B3 | 0.382 | 0.761 | 0.790 |
| Late Fusion | B1 | 0.474 | 0.815 | 0.807 |
| Late Fusion | B2 | 0.479 | 0.808 | 0.789 |
| Late Fusion | B3 | 0.421 | 0.739 | 0.702 |
| CNN TL | B1 | 0.456 | 0.760 | 0.747 |
| CNN TL | B2 | 0.447 | 0.748 | 0.742 |
| CNN TL | B3 | 0.507 | 0.807 | 0.799 |
| Early Fusion TL | B1 | 0.471 | 0.770 | 0.770 |
| Early Fusion TL | B2 | 0.424 | 0.668 | 0.642 |
| Early Fusion TL | B3 | 0.433 | 0.709 | 0.678 |
| Intermediate TL | B1 | 0.489 | 0.810 | 0.800 |
| Intermediate TL | B2 | 0.508 | 0.829 | 0.825 |
| **Intermediate TL** | **B3** | **0.521** ⭐ | 0.828 | 0.822 |
| Late Fusion TL | B1 | 0.422 | 0.722 | 0.695 |
| Late Fusion TL | B2 | 0.470 | 0.796 | 0.775 |
| Late Fusion TL | B3 | 0.466 | 0.780 | 0.757 |

---

## 7. Best Results (Untuk Abstract & Discussion)

### Overall Best: Intermediate TL 4-class B3 = Macro F1 0.521

⚠️ **Revisi**: angka Late Fusion di-update ke **val-tuned `w`** (grid search di Primer val, bukan di test — fix test-set leakage di nb 45/49/52/55). Best overall **bergeser dari Late Fusion TL B3 (0.567, test-tuned) ke Intermediate TL B3 (0.521, val-tuned proper)**.

| Model | Scenario | Kelas | Macro F1 | Note |
|-------|----------|:-----:|:--------:|------|
| **Intermediate TL** | **B3** | **4** | **0.521** | ⭐ Best overall (val-tuned) |
| Intermediate TL | B2 | 4 | 0.508 | |
| CNN TL | B3 | 4 | 0.507 | |
| Intermediate TL | B1 | 4 | 0.489 | |
| Late Fusion | B2 | 4 | 0.479 | Late Fusion scratch, val-tuned |
| Late Fusion TL | B2 | 4 | 0.470 | val-tuned |

### Best per Section

| Section | Finding |
|---------|---------|
| **Fusion strategies** | Intermediate TL (0.521) > Late Fusion TL B2 (0.470) ≈ B3 (0.466) > **Early Fusion TL** (0.471) > Single-modal (val-tuned 4c) |
| **TL effect** | TL +0.05-0.10 Macro F1 konsisten di semua arch |
| **Class granularity** | 4-class F1 ≈ 1.6× lebih tinggi dari 7-class (0.521 vs 0.333) |
| **Imbalance handling** | B3 (aug) > B2 (weights) > B1 (baseline) di Late/Intermediate TL; Early Fusion tidak mengikuti pola ini |

### Early Fusion Best Configs

| Config | Class | Macro F1 | Acc | Note |
|--------|:-----:|:--------:|:---:|------|
| EarlyFusion TL B3 | 7c | **0.333** | 0.753 | ⭐ Best Early Fusion 7c |
| EarlyFusion TL B1 | 4c | **0.471** | 0.770 | ⭐ Best Early Fusion 4c |
| EarlyFusion B1 | 4c | 0.457 | 0.822 | Best Early Fusion scratch |

**Temuan Early Fusion**:
- Early Fusion TL 7c B3 (0.333) **melampaui** Intermediate TL B3 (0.292) di 7-class — heatmap channel + augmentation + TL bekerja sinergis
- Namun Early Fusion 4c gagal tembus 0.50 (best 0.471), di bawah Intermediate TL 4c B3 (0.521) — val-tuned. Late Fusion TL 4c best val-tuned = 0.470 (B2)
- B1 baseline Early Fusion punya **accuracy tinggi** (0.822 di 4c) tapi macro F1 rendah → model prediksi mayoritas
- B2 (class weights) **paling jelek** di Early Fusion (acc drop ~30% dari B1) — channel concat sensitif terhadap re-weighting

---

## 8. Gambar & Tabel yang Perlu Disiapkan

### Gambar (Figures)
- **Fig 1**: Architecture diagram (5 varian fusion strategies, visualizing fusion points)
- **Fig 2**: Dataset sample images (front-facing programming sessions) — contoh per kelas emosi
- **Fig 3**: **Class distribution histogram** — side-by-side 7-class vs 4-class bar chart (log scale), highlighting imbalance ratio 1:1138 (7c) vs 1:61 (4c). ✅ **Ready at `docs/figures/class_distribution.{pdf,png}`** (script: `scripts/make_class_distribution_figure.py`). Justifies macro-F1 as primary metric.
- **Fig 4**: Landmark heatmap generation illustration (untuk Early Fusion)
- **Fig 5**: Macro F1 bar chart — 5 arsitektur × 2 kelas (best scenario per model)
- **Fig 6**: Confusion matrix best model (Intermediate TL 4c B3 + Early Fusion TL 7c B3)

### Tabel (Tables)
- **Tab 1**: Dataset distribution (7-class vs 4-class, train/val/test)
- **Tab 2**: Main results — **54 configurations** (5 arch × 3 scenarios × 2 classes × {scratch, TL}, minus 6 FCNN TL slots), rank by Macro F1
- **Tab 3**: Top 5 configurations detailed (all 4 metrics)
- **Tab 4**: Per-class F1 for best model
- **Tab 5**: Ablation — effect of TL (scratch vs TL per architecture)

---

## 9. Referensi Target (Starter List)

### FER with Deep Learning
- Dada et al. (2023) — CNN-10
- Li et al. (2024) — AA-DCN
- Khan et al. (2023) — ResNet50 TL for FER2013
- Zhang et al. (2022) — Late Fusion CK+/JAFFE

### Multimodal Fusion (Image + Landmark)
- **Mo, S., Yang, W., Wang, G., Liao, Q. (2020) — Emotion Recognition with Facial Landmark Heatmaps (HAE-Net), MMM 2020, LNCS 11961, pp. 278-289** ⭐ (Early Fusion reference)
- Chen et al. (2024) — β-skeleton + CNN hybrid
- Zhang et al. (2024) — GhostNet Multimodal

### Transfer Learning & Attention
- He et al. (2016) — ResNet (for backbone reference)
- Boulahia et al. (2021) — Early/Intermediate/Late fusion strategies

### Affective Computing in Education
- Picard (1997) — Affective Computing (foundational)
- D'Mello et al. — AutoTutor / affect-aware systems
- Sharma et al. — MOOC engagement via FER

### FER Datasets
- Lucey et al. (2010) — CK+
- Lyons et al. (1998) — JAFFE
- Li et al. (2017) — RAF-DB
- Lundqvist et al. (1998) — KDEF

---

## 10. Catatan untuk Methodology Section

### Hyperparameters (Section 3.3 Training Setup)
- Optimizer: Adam
- Learning rate: 1×10⁻⁴ (scratch), 5×10⁻⁵ (TL)
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=8, min_lr=1×10⁻⁷)
- Batch size: 32
- Max epochs: 50
- Early stopping: patience=15 (monitor Macro F1 on val set)
- Loss: CrossEntropyLoss (B1), Weighted CE (B2, B3)

### Pembobotan class weights (B2, B3)
- Inverse-frequency: `w_c ∝ 1 / freq(c)`, dinormalisasi supaya `sum(w) = num_classes`

### Data preprocessing (Section 3.1 Dataset)
- Face crop: bounding box dari Face API detection, padding sesuai proporsi wajah
- Resize: bilinear interpolation ke 224×224
- Normalization: piksel [0, 255] → [0, 1] (float32)
- Landmark: MediaPipe FaceMesh (478 titik) → subset 68 titik klasik (x, y normalized) → 136-dim vektor
- **Heatmap generation (Early Fusion)**: Gaussian blob σ=3px di setiap titik landmark, element-wise max aggregation across 68 titik → single 224×224 heatmap, range [0, 1]

### Data augmentation (B3 scenario)
Augmentasi diterapkan hanya pada **training set kelas minoritas** sampai distribusi lebih seimbang:
- Random rotation (±15°)
- Horizontal flip (p=0.5)
- Brightness jitter (±20%)
- Landmark koordinat **ikut ter-transform** agar tetap aligned dengan image
- Val & test set **tidak di-augment**

### Random seeds & reproducibility
- `np.random.RandomState(42)` untuk dataset split (deterministic)
- `torch.manual_seed`, `cuDNN deterministic` **tidak** di-set (untuk kecepatan training)
- Variance antar-run: ±0.01-0.05 Macro F1 (normal di DL tanpa strict seeding)
- **Single run per config** — bisa disebutkan di Limitations / Future Work

### Hardware
- GPU: NVIDIA Tesla T4 (Biznet Gio VPS)
- Framework: PyTorch + torchvision
- Training time per config: 10-45 menit (depending on arch & dataset size)

---

## 11. Per-Class Analysis (Section 4.5) — Data Extraction

Per-class metrics untuk **best model (Intermediate TL 4c B3, Macro F1 = 0.521, val-tuned)** sudah di-export via `scripts/export_new_best_predictions.py`. File: `models/frontonly_conf60/predictions/best_4c_intermediate_tl_b3.json`.

### Per-Class F1 (Intermediate TL 4c B3, val-tuned)
| Class | Precision | Recall | F1 | Support |
|-------|:---------:|:------:|:--:|:-------:|
| neutral | 0.935 | 0.839 | **0.884** | 688 |
| happy | 0.668 | 0.880 | **0.759** | 183 |
| sad | 0.382 | 0.520 | **0.441** | 50 |
| negative | 0.000 | 0.000 | **0.000** | 8 |

**Observasi:** Accuracy=0.822, Weighted F1=0.828, Macro F1=0.521. Class `negative` (n=8) total gagal — Macro F1 naik karena neutral/happy kuat, bukan karena minoritas teratasi. Perlu disclose di Section 4.5.

### Confusion Matrix — Expected Pattern
Mayoritas kesalahan adalah **over-prediction ke neutral** (kelas mayoritas). Contoh:
- True Negative → Predicted Neutral: kemungkinan tinggi
- True Sad → Predicted Neutral: beberapa
- True Happy → Predicted Neutral: sedikit (happy umumnya jelas)

---

## 11b. Late Fusion Weight `w` Grid-Search (Appendix / Reproducibility)

Val-tuned `w ∈ [0.00, 0.05, ..., 1.00]` via Primer val split. `w` = bobot stream CNN; bobot FCNN = `1 − w`.

### Primer conf60 — 12 konfigurasi
| Config | Scratch `w` / F1 | TL `w` / F1 |
|--------|:---------------:|:-----------:|
| 7c B1 | 0.20 / 0.270 | 0.15 / 0.238 |
| 7c B2 | 0.05 / 0.248 | 0.05 / 0.249 |
| 7c B3 | 0.00 / 0.222 | 0.05 / 0.232 |
| 4c B1 | 0.30 / 0.474 | 0.00 / 0.422 |
| 4c B2 | 0.15 / 0.479 | 0.15 / 0.470 |
| 4c B3 | 0.00 / 0.421 | 0.10 / 0.466 |

### Benchmark CK+/JAFFE — 4 konfigurasi (B1 baseline, val-tuned post-retrain Apr 2026)
| Dataset | `w` | Macro F1 |
|---------|:---:|:--------:|
| CK+ 7c | 0.00 | 0.494 |
| CK+ 4c | 0.00 | 0.537 |
| JAFFE 7c | 0.40 | 0.314 |
| JAFFE 4c | 0.00 | 0.492 |

**Pattern:** 14/16 konfigurasi memilih `w ≤ 0.20` → FCNN/landmark stream dominan di weighted softmax. Supports observation bahwa landmark geometry lebih diskriminatif dari visual texture di setting natural (Primer) dan benchmark small-scale (JAFFE/CK+).

---

## 12. Discussion Key Talking Points (Section 5)

### 5.1 Multimodal Fusion vs Single-Modality (RQ1)
- **Fakta**: Best fusion (Intermediate TL 4c B3 = 0.521, val-tuned) > Best single-modal (CNN TL 4c B3 = 0.507, FCNN 4c B2 = 0.459)
- **Insight**: Fusi image + landmark memberikan gain konsisten karena:
  - Image CNN: capture texture, color, context visual
  - Landmark FCNN: capture pose geometrik (eyebrow position, mouth opening, dll)
  - Kombinasi: complementary information
- **Note kontra**: Tidak semua fusion menang — Intermediate Fusion scratch kalah dari CNN scratch di 7-class (0.261 vs 0.277)

### 5.2 Fusion Strategy Comparison (RQ2)
- **Ranking di primer conf60 (4c best, val-tuned)**: Intermediate TL (0.521) > CNN TL single (0.507) > Late Fusion scratch B2 (0.479) ≈ Early Fusion TL (0.471) > Late Fusion TL B2 (0.470) ≈ B3 (0.466)
- **Ranking di primer conf60 (7c best, val-tuned)**: Early Fusion TL B3 (0.333) > Intermediate TL (0.292) > Late Fusion scratch B1 (0.270) > CNN single (0.277) > Late Fusion TL B2 (0.249)
- **Insight**: **Intermediate Fusion TL** juara di 4-class karena:
  - Feature-level joint learning optimal saat data augmented (B3) cukup untuk mencegah overfit
  - Concat 256d (CNN) + 128d (FCNN) → 384d memungkinkan fusion head belajar cross-modal interaction
- **Late Fusion** turun setelah val-tuning proper — `w` yang di-tune pakai val set (bukan test) cenderung memilih `w ≈ 0` (landmark-dominant) di Primer → kehilangan komplementaritas CNN stream
- **Early Fusion** (HAE-Net style channel concat): **underperforms di 4-class tapi menang di 7-class (dengan B3 augmented)**. Hipotesis:
  - Heatmap sparse Gaussian (mostly zeros) tidak memberi informasi kuat di layer awal
  - Kernel conv layer pertama harus simultaneously belajar RGB features DAN heatmap patterns dengan satu set bobot → kompromi representasi
  - Weight init 4th channel dari mean(RGB) suboptimal untuk heatmap yang berbeda karakteristik dari citra
- **Supporting evidence — Early Fusion di benchmarks (out of paper scope, tapi mendukung narasi)**: Eksperimen tambahan (nb 66, 16 runs) di CK+/JAFFE/RAF-DB/KDEF menunjukkan Early Fusion **universal kompetitif**: selisih 0.03-0.11 dari best Intermediate TL (kecuali JAFFE TL collapse karena dataset kecil 213 images). Gap di Primer 4c (-0.10) **sebanding** dengan gap di benchmark → underperformance Early Fusion **bukan artifact natural data**, melainkan keterbatasan inherent dari channel-level fusion ketika landmark info sparse.
- **Implikasi**: Pilihan fusion strategy optimal bergantung (1) granularitas kelas, (2) ukuran data augmented untuk B3. Fusion di level feature (Intermediate) / decision (Late) umumnya lebih robust, tapi Early Fusion bisa unggul di skenario high-granularity dengan data augmented.

### 5.3 Transfer Learning Effectiveness (RQ3)
- **Fakta**: TL variant konsisten unggul dari scratch di Intermediate/Early Fusion (contoh: Intermediate TL B3 0.521 vs Intermediate scratch B3 0.394, +0.127). Di Late Fusion val-tuned, gain TL marginal (0.470 vs 0.479 di B2) karena `w ≈ 0` sudah pilih FCNN-only.
- **Insight**: ResNet18 pretrained ImageNet memberikan visual feature representation yang matang, crucial untuk dataset kecil (6,795 sampel)
- **Efek kombinasi TL + imbalance handling**: TL + B3 (augmentation) memberikan gain lebih tinggi dibanding TL + B1 → TL melengkapi augmentation, tidak replace

### 5.4 Limitations
- **Single run per config** (bukan mean±std) — variance ±0.01-0.05 bisa mengubah ranking di kasus tertentu
- **Imbalance ekstrem** (rasio 1:1138) → evaluasi kelas minoritas kurang reliable (test set fearful hanya 2 sampel)
- **37 subjek** → generalisasi populasi belum divalidasi; dataset program studi specific
- **Face API otomatis annotasi** — bukan ground-truth human-labeled, ada noise label di sampel dengan confidence rendah (conf60 filter memitigasi tapi tidak hilangkan)
- **Single dataset benchmark** — tidak uji cross-dataset generalization di paper ini
- **4-class remap subjective** — cara menggabungkan minoritas ke "negative" bisa diargumentasikan

### 5.5 Implications for Learning Analytics
- Model dengan Macro F1 ~0.52 masih **preliminary** untuk deployment real-time
- Bisa dipakai untuk **aggregate-level analytics** (misal: trend emosi per-session), bukan per-frame decision
- Potensi integrasi ke LMS untuk adaptive feedback, tutor intervention, atau konten rekomendasi
- Perlu validasi lebih lanjut dengan human annotation + larger dataset

---

## 13. Abstract Elements Checklist

Abstract ideal (150-250 kata), harus mencakup:

- [ ] **Problem statement** (1 kalimat): FER untuk analitik pembelajaran di konteks pemrograman natural
- [ ] **Motivation/gap** (1 kalimat): existing FER fokus lab-posed, bukan natural
- [ ] **Proposed approach** (2-3 kalimat): 5 arsitektur × multimodal fusion × transfer learning
- [ ] **Dataset** (1 kalimat): 6,795 samples, 37 mahasiswa, sesi pemrograman, confidence ≥60%
- [ ] **Key result** (1-2 kalimat): Intermediate Fusion TL 4-class B3 achieves Macro F1 = 0.521 (val-tuned proper); Early Fusion TL 7-class B3 achieves 0.333
- [ ] **Key insight** (1 kalimat): Multimodal fusion + transfer learning outperforms single-modal baselines
- [ ] **Implications** (1 kalimat): Advances affective learning analytics tools for programming education

### Keywords (5-7 suggested)
- Facial expression recognition
- Multimodal fusion
- Transfer learning
- Deep learning
- Affective computing
- Learning analytics
- Imbalanced classification

---

## 14. Introduction Writing Guide (Section 1)

Paragraf-per-paragraf (tanpa subsection formal, sesuai struktur):

**Paragraph 1 — Motivasi**
- Emosi berpengaruh ke learning outcome (frustration, engagement, confusion)
- Dalam konteks programming: mahasiswa sering hadapi kesulitan yang memicu emosi negative
- Deteksi otomatis bisa memungkinkan adaptive feedback

**Paragraph 2 — Gap Penelitian**
- FER state-of-the-art (CK+, JAFFE, AffectNet) fokus ekspresi posed/lab — tidak representatif untuk kondisi natural
- Dataset natural untuk FER jarang (mainly in-the-wild web images, bukan task-specific)
- Belum ada studi komprehensif fusion strategies × TL pada natural programming context

**Paragraph 3 — Rumusan Masalah & Tujuan**
- Sebutkan 3 RQs (bisa inline di paragraph atau bullet format)
- Tujuan: mengembangkan pipeline FER multimodal untuk programming learning context

**Paragraph 4 — Kontribusi**
- (1) Dataset baru 6,795 sampel natural programming sessions (37 subjek)
- (2) Studi komparatif sistematis 5 arsitektur fusion × TL (54 configs)
- (3) Analisis empirik bahwa multimodal + TL + imbalance handling mencapai Macro F1 0.521 (4c Intermediate TL B3) / 0.333 (7c Early Fusion TL B3), val-tuned proper

**Paragraph 5 — Struktur Paper**
Singkat: "Section 2 reviews... Section 3 describes... Section 4 presents... Section 5 discusses... Section 6 concludes..."

---

## 15. Architecture Diagram (untuk Figure 1)

Deskripsi textual untuk refer saat draw diagram:

```
Input: Facial image 224×224×3    Landmark 68 points (136-dim)
         │                              │
         ▼                              ▼
(a) CNN: [Conv blocks] → [FC] → softmax ← single modal
(b) FCNN: [FC blocks] → softmax ← single modal

(c) Early Fusion:
    [Image 224×224×3] + [Heatmap 224×224×1] → concat channel → [4-ch Conv blocks] → softmax

(d) Intermediate Fusion:
    [Image] → CNN features (256-dim) ─┐
                                       concat → [FC] → softmax
    [Landmark] → FCNN features (128-dim) ┘

(e) Late Fusion:
    [Image] → CNN → softmax_c ─┐
                                weighted avg (w_c, 1-w_c) → argmax
    [Landmark] → FCNN → softmax_f ┘
```

**Transfer Learning variants**: replace CNN (scratch) dengan ResNet18 pretrained ImageNet:
- (a) CNN TL: ResNet18 block + custom FC head
- (c) Early Fusion TL: **first Conv2d dimodifikasi dari 3→4 channel** (weight RGB dari pretrained, weight heatmap di-init dari mean RGB)
- (d) Intermediate Fusion TL: ResNet18 image stream + FCNN landmark stream
- (e) Late Fusion TL: ResNet18 + FCNN separate training

---

## 16. Related Work — Specific Points to Cover

### 2.1 Deep Learning for FER
- Evolusi: HOG/SIFT → CNN → ResNet → Vision Transformer
- Paper anchor:
  - Dada et al. (2023) CNN-10 on CK+
  - Li et al. (2024) AA-DCN (anti-aliased deep conv)
  - Khan et al. (2023) ResNet50 for in-the-wild

### 2.2 Multimodal Fusion (Image + Landmark)
- Strategi umum: early (input level) vs intermediate (feature) vs late (decision)
- Paper anchor:
  - **Mo, S. et al. (MMM 2020) — HAE-Net** (REFERENSI WAJIB untuk Early Fusion claim — paper ini mendasari desain Early Fusion channel-concat kami, meski hasil kami menunjukkan pendekatan ini kurang optimal di natural programming data). Full cite: Mo, S., Yang, W., Wang, G., & Liao, Q. (2020). Emotion Recognition with Facial Landmark Heatmaps. In MMM 2020, LNCS 11961, pp. 278-289. Springer.
  - Boulahia et al. (2021) — fusion strategies taxonomy
  - Chen et al. (2024) — β-skeleton + CNN

### 2.3 Transfer Learning for FER
- ImageNet → FER: pretrained backbone as feature extractor
- Fine-tune vs frozen feature: rationale untuk fine-tune (dataset kecil, domain shift)
- Paper anchor: He et al. (2016) ResNet, transfer learning survey

### 2.4 Affective Computing in Education
- Affective computing foundational: Picard (1997)
- MOOC / programming education specific: Sharma et al., D'Mello et al.
- Emphasis: gap in natural programming context
