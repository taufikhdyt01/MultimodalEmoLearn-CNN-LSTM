# Heatmap Landmark Visualization — Penjelasan Figure

**File:** `docs/figures/heatmap_landmark_overview.png` / `.pdf`
**Script:** `scripts/make_heatmap_landmark_viz.py`

## Konteks

Figure ini menjelaskan **landmark heatmap** — input tambahan yang digunakan oleh model Early Fusion (input-level fusion) untuk menggabungkan informasi spasial wajah (citra) dengan informasi struktural (landmark).

## Apa yang model Early Fusion sebenarnya terima

Early Fusion menerima **1 tensor 4-channel** dengan dimensi `(4, 224, 224)`:

| Channel | Konten | Range nilai |
|---|---|---|
| 0 | R (red) — dari RGB image | [0, 1] |
| 1 | G (green) — dari RGB image | [0, 1] |
| 2 | B (blue) — dari RGB image | [0, 1] |
| 3 | **Landmark heatmap** (grayscale) | [0, 1] |

Heatmap channel di-bentuk dengan **rendering 68 landmark titik sebagai Gaussian blob** pada canvas 224×224, lalu ditambah pada tensor sebagai channel ke-4 (channel index 3).

## Layout figure (4 baris × 7 kolom)

Tiap kolom = 1 sample wajah untuk satu kelas emosi (neutral, happy, sad, angry, fearful, disgusted, surprised), dipilih dengan strategi yang sama dengan figure `class_samples_all_datasets` agar konsisten visual antar figure tesis.

### Row (a) — RGB image

Citra wajah asli (224×224×3 float32 [0,1]). Ini adalah **channel 0-2 dari tensor input** model Early Fusion.

### Row (b) — Landmark heatmap (hot colormap)

Heatmap 224×224 di-render dari 68 landmark titik (hasil deteksi face-api.js), divisualisasi dengan colormap `hot` (hitam → merah → kuning → putih) untuk menonjolkan area dengan intensitas landmark tinggi. **Isi data sama persis dengan row (d)**, hanya colormap-nya berbeda untuk readability.

> Catatan: warna hot di sini **hanya untuk explanation**. Saat masuk ke model, heatmap di-feed sebagai 1-channel grayscale (lihat row d).

### Row (c) — Heatmap overlay (alpha-blend)

Heatmap di-blend di atas RGB image dengan alpha=0.6 untuk pixel dimana intensitas heatmap > 0.05. Visualisasi ini menunjukkan **alignment antara landmark dan area wajah** — pembaca bisa langsung melihat bahwa landmark menutupi region anatomis penting (kontur wajah, alis, mata, hidung, mulut).

> Row (c) **bukan input ke model** — ini composite untuk human comprehension saja.

### Row (d) — Heatmap intensity (grayscale)

Heatmap dalam representasi **grayscale 1-channel** — inilah yang **secara teknis di-feed sebagai channel ke-4 (channel index 3)** ke model Early Fusion. Nilai pixel `[0, 1]` di mana 0 = jauh dari landmark, 1 = tepat di titik landmark.

## Ringkasan: mana yang dipakai model

| Row | Dipakai model? | Fungsi di figure |
|---|---|---|
| (a) RGB image | ✅ Channel 0-2 input | Tunjukkan source RGB |
| (b) Heatmap (hot) | ❌ Explanatory | Tunjukkan heatmap dengan colormap kontras |
| (c) Overlay | ❌ Explanatory | Tunjukkan alignment heatmap ke wajah |
| (d) Heatmap (grayscale) | ✅ Channel 3 input | Representasi exact channel ke-4 model |

**Kesimpulan**: Model Early Fusion hanya menerima dua sumber data — RGB image (row a) dan landmark heatmap grayscale (row d). Row (b) dan (c) adalah visualisasi pendukung untuk membantu pembaca memahami struktur input.

## Konstruksi heatmap (detail teknis)

Heatmap di-generate via:
1. Deteksi 68 landmark dengan **face-api.js** (TinyFaceDetector + Landmark68Net) pada frame asli
2. Untuk setiap titik landmark `(xi, yi)`, gambar Gaussian blob dengan sigma kecil di canvas kosong 224×224
3. Sum semua 68 Gaussian → heatmap final, normalisasi ke [0, 1]
4. Simpan sebagai `X_{split}_heatmaps.npy` dengan dtype float32

Total energy heatmap per sample (sum of pixel values) ≈ 2947, konsisten dengan 68 landmark × Gaussian footprint.

## Penggunaan di tesis

Figure ini cocok untuk **Bab 3 Metodologi** — section Early Fusion:

- Subbab "Input representation": tampilkan figure ini untuk menjelaskan bagaimana RGB dan landmark digabung di level input
- Subbab "Architecture": diagram Conv2D pertama menerima 4 input channel (dari 3 standar ImageNet); referensi ke figure
- Subbab "Justification": argumen mengapa landmark heatmap = spatial prior yang complement RGB texture

## Re-generate

```bash
python scripts/make_heatmap_landmark_viz.py             # default seed
python scripts/make_heatmap_landmark_viz.py --seed 7    # variasi sample
python scripts/make_heatmap_landmark_viz.py --dpi 300   # higher resolution
```
