# Feature & Fusion Design — Q&A

> Catatan desain yang menjawab pertanyaan tentang kombinasi feature × source × fusion
> yang muncul di `all_metrics_tables.md`. Disusun supaya jelas kenapa beberapa kombinasi
> tidak ada di tabel (bukan oversight, tapi by-design constraint).

---

## Q1. Kenapa landmark source FaceAPI tidak punya `blendshape_52`, tapi punya `facs_plus_bs_80`?

### Ringkasan

| Feature | MP | FA | Catatan |
|---|---|---|---|
| `raw_136` | ✅ | ✅ | 2D koordinat landmark |
| `facs_28` | ✅ | ✅ | 28 jarak Euclidean dari koordinat landmark |
| **`blendshape_52`** | ✅ | ❌ | **MP-only** — FA tidak punya output blendshape |
| **`facs_plus_bs_80`** | ✅ | ✅ | **Hybrid** — FACS dari source X, blendshape selalu MP |

### Alasan detail

**1. `blendshape_52` MP-only**

52-dim blendshape adalah koefisien ARKit (`mouthSmile`, `eyeBlink`, `browInnerUp`, dll.)
yang di-regress oleh **dedicated NN head di MediaPipe Face Landmarker** model. face-api.js
hanya mengeluarkan 68 landmark point (koordinat 2D), tidak ada output blendshape sama sekali.

Confirmed di `scripts/run_unified_derived.py:22`:
```
# blendshape selalu MP-source, FACS bisa MP/FA
```

**2. `facs_28` source-agnostic**

`facs_28` = 28 jarak Euclidean yang dihitung dari koordinat landmark (jarak antar-titik
wajah, misalnya `dist(mouth_left, mouth_right)`). Karena hanya butuh koordinat, bisa
dihitung dari MP (468 titik) atau FA (68 titik). Distance metric-nya source-dependent
tapi feature design-nya identik.

**3. `facs_plus_bs_80` hybrid construction**

Konstruksi: `facs_28 (28-dim)` ⊕ `blendshape_52 (52-dim)` = 80-dim concat.

Code (`scripts/run_unified_derived.py:231`):
```python
"facs_landmark_source": source if feature == "facs_plus_bs_80" else None,
"blendshape_source": "mediapipe",  # selalu MP
```

Jadi untuk **FA version FB80**:
- FACS-part (28-dim) dihitung dari **FA landmarks**
- Blendshape-part (52-dim) **selalu di-borrow dari MP**
- Hasilnya 80-dim hybrid feature dengan dua source berbeda di-stack

### Implikasi eksperimen

FB80-FA **tidak bisa di-interpret sebagai "pure FA feature"** — selalu ada komponen MP
di komponen blendshape-nya. Kalau ingin compare apple-to-apple "FA only", gunakan
`facs_28` FA atau `raw_136` FA.

---

## Q2. Kenapa Early Fusion tidak punya dekomposisi feature landmark (raw_136 only)?

### Ringkasan

Early Fusion `§X.3` di tabel **hanya punya `raw_136`**, sementara Intermediate Fusion
`§X.4` punya semua varian (`raw_136`, `facs_28`, `blendshape_52`, `facs_plus_bs_80`).

Bukan oversight — ini **konsekuensi arsitektural**.

### Alasan detail

**Early Fusion = channel-stacking RGB + heatmap di awal CNN backbone**

```
Input = [RGB (3ch), heatmap (1ch)]  →  4-channel tensor  →  CNN  →  classifier
```

Heatmap dihitung dari **rasterize 2D landmark coordinates ke pixel plane** (Gaussian
blob di setiap titik landmark). Confirmed di `scripts/run_unified_fusion.py`:

```python
# line 157
hm = np.load(DATA_DIR / f"X_{split}_heatmaps.npy")

# line 165
"Early Fusion ignores landmark_dim (input is RGB+heatmap channel, not coords)"

# line 192
return EarlyFusionDataset(X_img, X_hm, y, augment=augment, seed=seed)
```

`EarlyFusionDataset` cuma butuh `X_img` + `X_hm` (image + heatmap), tidak terima `X_lm`.

**Kenapa hanya `raw_136` yang bisa jadi heatmap?**

| Feature | Bisa rasterize ke heatmap? | Alasan |
|---|---|---|
| `raw_136` | ✅ | 2D koordinat → langsung plot ke pixel plane |
| `facs_28` | ❌ | 28 jarak Euclidean = scalar distances. Tidak punya posisi spatial |
| `blendshape_52` | ❌ | 52 koefisien abstract (`smile`, `eye_blink`, ...) = global scalars. Tidak ada "lokasi" |
| `facs_plus_bs_80` | ❌ | Concat dari dua scalar feature di atas |

`facs_28` / `blendshape_52` / `facs_plus_bs_80` adalah **abstract scalar features** —
tidak punya lokasi spatial di image plane, jadi tidak bisa dirasterize jadi heatmap
channel. Channel-stacking-nya tidak meaningful (broadcasting global scalar ke seluruh
pixel = no spatial signal).

### Bandingkan dengan Intermediate Fusion

```
RGB           →  CNN backbone  →  img_feat ∈ R^512
landmark_vec  →  FCNN          →  lm_feat ∈ R^128       (D ∈ {28, 52, 80, 136})
concat(img_feat, lm_feat)  →  classifier
```

Intermediate Fusion FCNN cuma butuh **fixed-size input vector**, jadi bisa terima fitur
abstract apapun. Itu sebabnya `§X.4` punya semua varian feature, sedangkan `§X.3` Early
Fusion cuma `raw_136`.

### Apakah bisa "dipaksakan" untuk feature lain?

Secara teori bisa, tapi tidak meaningful tanpa research tambahan:

- **Inject `facs_28` sebagai 28 channel global-broadcasted** (semua pixel = nilai sama):
  secara informasi setara dengan masukin ke FC layer di Intermediate Fusion. Tidak gain
  apa-apa dari spatial conv.
- **Bangun heatmap dari subset landmark yang dipakai oleh FACS** (e.g., titik mulut +
  mata): cuma sebagian raw_136, kehilangan informasi distance-nya.
- **Multi-scale heatmap weighted by blendshape**: research-grade, butuh justifikasi
  desain tersendiri.

Jadi keputusan **Early Fusion = `raw_136` only** itu by-design karena cocok dengan
paradigma "spatial channel stacking". Bukan kelalaian sweep design.

### Gated mode juga raw_136-only

Early Fusion gated mode (`--early-fusion-modes gated`) menerapkan spatial sigmoid
gating dari heatmap ke RGB feature map. Karena gating mechanism-nya juga spatial
(per-pixel sigmoid), tetap butuh heatmap dari raw_136. Constraint identik.

---

## Cross-reference

- Master tabel hasil eksperimen: [`all_metrics_tables.md`](./all_metrics_tables.md)
- Konfigurasi 54 primer: [`all_54_configs_metrics.md`](./all_54_configs_metrics.md)
- Code:
  - Feature extraction: `scripts/run_unified_derived.py` (blendshape/FB80)
  - Fusion training: `scripts/run_unified_fusion.py` (early/intermediate)
  - Models: `src/training/models.py`
