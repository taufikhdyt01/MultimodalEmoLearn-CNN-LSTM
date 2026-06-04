# Sampler Strategy B1/B2/B3 — Penjelasan Figure

**File:** `docs/figures/sampler_strategy_b1_b2_b3.png` / `.pdf`
**Script:** `scripts/make_sampler_strategy_viz.py`

## Konteks

Figure ini menjelaskan **3 skenario sampling+augmentation** yang dipakai di seluruh eksperimen (unimodal & multimodal) untuk menangani **class imbalance ekstrem** di Primer dataset (rasio max:min ≈ 1138:1 di 7-class).

Cocok untuk **Bab 3 Metodologi** subbab "Imbalance Handling" / "Training Scenarios".

## Mengapa 3 skenario

Distribusi kelas Primer **sangat tidak seimbang** (~5691 sample neutral vs 5 fearful), kalau train langsung pakai uniform sampling, model akan didominasi neutral. Tiga strategi yang dibandingkan:

| Skenario | Sampler | Augmentation | Rationale |
|---|---|---|---|
| **B1** | shuffle uniform | none | baseline tanpa intervensi — model "lihat" distribusi natural |
| **B2** | `WeightedRandomSampler` (prob ∝ 1/class_count) | none | balanced batch sehingga model treat semua class equal |
| **B3** | `WeightedRandomSampler` | synced per-batch aug | balanced + diversifikasi → harapan generalization lebih baik |

## Layout figure

### Row 1: Class distribution (atas, full width)
Bar chart full dataset Primer (frontonly_conf60) per kelas. Memperjelas seberapa ekstrem imbalance: neutral dominant (5691), fearful only 5 sample.

### Row 2-5: 3 kolom (B1 / B2 / B3) dengan 4 sub-row:

#### Sub-row A — Header
- Title chip dengan nama scenario (B1, B2, B3)
- 1-2 baris deskripsi singkat (sampler type + aug status)

#### Sub-row B — Sample probability per class
Bar chart 7 class menunjukkan **probabilitas tiap class dipilih** oleh sampler:
- **B1**: probability ∝ count → neutral ~84%, fearful ~0.07%
- **B2 & B3**: probability ∝ 1/count → semua class probability ~14% (uniform sekitar 1/7)

#### Sub-row C — Simulated batch (32 samples)
Grid 4×8 = 32 kotak berwarna sesuai class. Hasil simulasi satu batch dengan sampler masing-masing:
- **B1**: hampir semua kotak warna neutral (biru) — bias majority
- **B2**: kotak warna campuran semua class — balanced
- **B3**: sama dengan B2 (sampler identik), aug akan diaplikasikan setelah ini

#### Sub-row D — Augmentation
- **B1, B2**: ✗ No augmentation (raw sample langsung ke model)
- **B3**: ✓ Synced per-batch:
  - hflip + landmark swap + heatmap flip
  - rotate ±10° (semua channel synced)
  - brightness/contrast ±10% (image only, heatmap dibiarkan)

#### Sub-row E (footer)
Convergent box: "Training step (1 batch → model forward+backward)" — menunjukkan 3 path bertemu di forward pass yang sama.

## Implementasi teknis

### B1 — Uniform sampler
PyTorch `DataLoader(..., shuffle=True)` standar. Setiap epoch, dataset di-shuffle uniform tanpa balancing.

```python
loader = DataLoader(ds, batch_size=32, shuffle=True)
```

### B2 — WeightedRandomSampler (no aug)
```python
counts = np.bincount(y_train, minlength=7)
weights = 1.0 / counts[y_train]
sampler = WeightedRandomSampler(weights, num_samples=len(y_train), replacement=True)
loader = DataLoader(ds, batch_size=32, sampler=sampler)
```
Catatan: `replacement=True` agar minority class bisa di-resample dalam satu epoch.

### B3 — WeightedRandomSampler + augmentation
Sampler sama dengan B2. Augmentation dilakukan di `__getitem__` dataset:

**Synced augmentation** untuk image + landmark + heatmap (kalau aplicable):
- **hflip** (p=0.5): flip image horizontal + swap 68-point landmark left/right secara proper (lookup table dari `src/training/landmark_aug.py`) + flip heatmap horizontal
- **rotate ±10°**: rotasi affine, semua channel pakai matrix yang sama
- **photometric** (image only): brightness ±10%, contrast ×0.9-1.1
- **landmark noise** (B3 landmark unimodal): per-coord Gaussian σ=0.005

## Pengaruh ke hasil eksperimen

Insight dari `all_metrics_tables.md`:

| Modality | Best scenario (mf1) di 3c | Best scenario di 7c |
|---|---|---|
| Landmark FCNN | **B1** (FA features) | **B1** (FA features) |
| Landmark CNN1D | **B3** (raw_136 + aug helpful) | mixed |
| Image CNN_TL | **B2** (class weights) | **B2** |
| Intermediate Fusion | **B1** (FA, dominant majority weight bahkan tanpa balancing) | mixed |
| Late Fusion | **B1** (best mf1 0.7604 di 3c B1) | mixed |

**Observasi**:
- **B1 sering menang** untuk landmark FA-source karena top model tidak terlalu sensitif ke imbalance (FA-features informatif walau sampling biased)
- **B2 winning di image** mengindikasikan CNN benefit dari class-balanced batches
- **B3 winning untuk CNN1D + raw_136** karena sequential locality (positional features) + aug synergize baik

## Color coding

| Element | Warna |
|---|---|
| Header chip B1 | Light blue (#cfe4f4) |
| Header chip B2 | Light yellow (#fde4a0) |
| Header chip B3 | Light orange (#ffc8a0) |
| Aug box B1/B2 | Light gray (#f5f5f5) |
| Aug box B3 | Light green (#e8f5e9) — emphasize "ada aug" |
| Class colors | Konsisten dengan figure lain (neutral=biru, happy=kuning, dst) |

## Penggunaan di tesis

- **Bab 3 Metodologi** — subbab "Imbalance Handling": figure utama
- **Bab 4 Hasil & Pembahasan**: referensi balik saat membahas mengapa B1/B2/B3 menang di kondisi berbeda
- **Caption rekomendasi**:
  > *Tiga skenario sampling+augmentation untuk menangani imbalance ekstrem di Primer dataset (max:min ~1138:1). B1 = uniform sampler, B2 = weighted random sampler, B3 = weighted + synced per-batch augmentation. Setiap baris menunjukkan probability tiap class dipilih (Row 2), composition 1 batch hasil sampling (Row 3), dan augmentation yang diaplikasikan (Row 4).*

## Re-generate

```bash
python scripts/make_sampler_strategy_viz.py
```

Diagram murni schematic + simulasi batch (deterministic dengan seed=42). Bisa ganti seed di script kalau mau variasi sampel batch.
