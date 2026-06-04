# Augmentation Examples B3 — Penjelasan Figure

**File:** `docs/figures/augmentation_examples_b3.png` / `.pdf`
**Script:** `scripts/make_augmentation_examples_viz.py`

## Konteks

Figure ini menampilkan contoh visual **synced per-batch augmentation** yang dipakai di skenario **B3**. Augmentation diaplikasikan secara konsisten ke 3 channel: image (RGB), landmark, dan heatmap — kecuali photometric (brightness/contrast) yang hanya untuk image.

Cocok untuk **Bab 3 Metodologi** subbab "Augmentation Strategy" (subordinat dari "Imbalance Handling").

## Layout — 3 baris × 5 kolom

### Kolom (augmentation type)

| Col | Title | Apa yang berubah | Channel terimbas |
|---|---|---|---|
| **(a)** | Original | nothing — reference | none |
| **(b)** | Horizontal flip | flip horizontal (p=0.5 saat training) | image + landmark + heatmap |
| **(c)** | Rotate +10° | rotasi affine sekitar pusat (-10° s/d +10°) | image + landmark + heatmap |
| **(d)** | Brightness/Contrast | brightness ±0.10, contrast ×0.9-1.1 | **image only** (landmark & heatmap unchanged) |
| **(e)** | Combined (B3 example) | hflip + rotate -8° + photometric | semua (realistic B3 sample) |

### Baris (channel ditampilkan)

| Row | Channel | Visualisasi |
|---|---|---|
| **(1)** | RGB image | citra wajah 224×224×3 |
| **(2)** | Landmark overlay | 68 titik FA + connection lines di atas RGB |
| **(3)** | Heatmap (hot) | landmark heatmap 224×224, colormap hot |

## Highlight insight visual

### Spatial sync (kol b, c, e)

- **Hflip kolom (b)**: image mirror, landmark **proper left-right swap** (point 0↔16, 17↔26, dst — bukan sekadar mirror x karena 68-dlib semantically left-right paired), heatmap mirror — semua konsisten secara semantik
- **Rotate kolom (c)**: image rotate via cv2 affine, landmark coords rotated via matriks rotasi yang sama, heatmap rotate via cv2 affine matrix yang sama — semua align
- **Combined kolom (e)**: kombinasi spatial + photometric, contoh batch realistic B3

### Photometric isolation (kol d)
- Brightness +0.08, contrast ×1.10 — **hanya image yang berubah** (terlihat lebih terang & kontras)
- **Landmark (row 2) dan heatmap (row 3) di kolom (d) sama persis dengan original (kol a)** — sengaja, photometric tidak relevan untuk modality non-image

## Implementasi teknis

Augmentation function dari `src/training/fusion_aug.py`:

```python
# Spatial sync
img_hf  = _hflip_image(img)         # np.ascontiguousarray(img[:, ::-1, :])
lm_hf   = _hflip_landmark_136(lm)   # flip x → 1-x, then HFLIP_PERM swap
hm_hf   = _hflip_heatmap(hm)        # np.ascontiguousarray(hm[:, ::-1])

img_rot = _rotate_image(img, +10)   # cv2.warpAffine dengan reflect border
lm_rot  = _rotate_landmark_136(lm, +10)  # matrix rotation di normalized coords
hm_rot  = _rotate_heatmap(hm, +10)  # cv2.warpAffine sama matrix

# Photometric (image-only)
img_bc = (img - 0.5) * 1.10 + 0.5 + 0.08  # contrast first, then brightness
```

`HFLIP_PERM` defined di `src/training/landmark_aug.py` — pre-computed permutation 68-element array yang map kiri↔kanan untuk semua landmark (mis. mata kiri 36-41 ↔ mata kanan 42-47).

## Penggunaan di training (B3 only)

Di `EarlyFusionDataset.__getitem__` (di `src/training/fusion_aug.py`):
```python
if self.augment:
    if rng.random() < 0.5:               # hflip p=0.5
        img = _hflip_image(img); lm = _hflip_landmark_136(lm); hm = _hflip_heatmap(hm)
    angle = float(rng.uniform(-10, 10))  # rotate ±10° uniform
    if abs(angle) > 0.1:
        img = _rotate_image(img, angle); lm = _rotate_landmark_136(lm, angle); hm = _rotate_heatmap(hm, angle)
    img = _augment_image_photometric(img, rng,
                                       brightness=0.10, contrast=0.10)
```

**Per-sample**: setiap kali `__getitem__` dipanggil (mis. setiap iterasi training), aug parameters di-resample dengan `rng` deterministic per-sample. Sehingga 1 sample di-augmentasi berbeda tiap epoch — **infinite virtual sample size**.

## Pengaruh ke hasil

Insight `all_metrics_tables.md`:
- **B3 menang untuk CNN1D + raw_136**: lokality positional info benefit dari aug (rotation augment teaches model rotation invariance, hflip teaches left-right symmetry)
- **B3 tidak menang untuk landmark FCNN**: FCNN sudah translation-invariant secara desain (Linear layers tidak peduli spasial), aug tidak banyak menambah
- **B3 mixed untuk image CNN_TL**: kadang menang, kadang B2 — depends on scheme

## Penggunaan di tesis

- **Bab 3 Metodologi** subbab "Augmentation":
  - Tampilkan figure sebagai contoh visual dari B3 aug pipeline
  - Caption rekomendasi: *"Contoh visual augmentation B3 — synced per-batch. (a) Original, (b) hflip dengan proper landmark left-right swap, (c) rotasi affine sync ke 3 channel, (d) brightness/contrast hanya untuk image, (e) kombinasi realistic B3."*

- **Bab 4 Hasil & Pembahasan**:
  - Referensi figure saat membahas mengapa B3 bagus untuk CNN1D tapi tidak konsisten untuk FCNN
  - Highlight bahwa augment **synced** — bukan independent per channel — sehingga semantically valid

## Re-generate

```bash
python scripts/make_augmentation_examples_viz.py
```

Sample idx auto-selected (neutral confidence tertinggi). Aug parameters hardcoded supaya output deterministic & visually clear (rotate +10° untuk col c, kombinasi spesifik di col e).
