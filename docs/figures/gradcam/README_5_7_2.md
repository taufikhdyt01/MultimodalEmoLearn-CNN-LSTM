# Grad-CAM — Aset Gambar Sub-bab 5.7.2

Dokumen pendamping untuk figure Grad-CAM interpretability. Semua di-generate dari:
- `scripts/make_gradcam_5_7_2.py` (primer 3c & 7c)
- `scripts/make_gradcam_ckplus.py` (kontras CK+)

---

## ⚠️ Keterbatasan teknis (WAJIB ditulis di naskah)

**Grad-CAM hanya diterapkan pada cabang citra (CNN).** Grad-CAM membutuhkan peta
fitur konvolusi spasial untuk memproyeksikan gradien kelas kembali ke lokasi piksel.
Cabang landmark (FCNN / CNN1D) menerima **vektor fitur** (koordinat / FACS / blendshape),
bukan citra, sehingga **tidak memiliki dimensi spasial** dan tidak dapat di-Grad-CAM
secara langsung. Untuk interpretabilitas kontribusi landmark, diperlukan teknik lain
(mis. gradient saliency terhadap input vektor, atau occlusion per-grup landmark/AU) —
di luar lingkup sub-bab ini. Karena itu analisis Grad-CAM di sini **dibatasi pada
modalitas citra**, dan keterbatasan ini dinyatakan eksplisit.

---

## Model & konfigurasi

Tiga model dengan **backbone citra sama** (ResNet-18, transfer learning ImageNet)
agar perbandingan atensi adil. Target layer Grad-CAM = **blok konvolusi terakhir
(layer4 / `features[-2][-1]`)**.

| Model | Input | Peran |
|---|---|---|
| **CNN_TL (unimodal)** | RGB 3-channel | Baseline citra murni |
| **Early Fusion concat** | RGB + heatmap landmark (4-channel, setara) | Apakah landmark menggeser atensi? |
| **Early Fusion gated** | RGB di-*gate* oleh heatmap landmark | Gating spasial eksplisit |

Checkpoint: `models/frontonly_conf60/gradcam_ckpts{,_7c}/` dan
`7class/Unified/fusion_early_gated_tl/checkpoints/b1.pt` (gated 7c).
Skenario training: best-per-model (B1; CNN_TL 3c memakai B2/class-weight). Perbandingan
atensi bersifat **kualitatif**, jadi perbedaan skenario tidak memengaruhi argumen.

---

## Daftar figure

Tiap figure = **satu section**, kolom `[Citra asli | CNN_TL (unimodal) | Early concat | Early gated]`
(header diperbesar sekali di atas), label `→ pred (conf) ✓/✗` di bawah tiap heatmap, baris
diberi `True: <kelas>`. ✓/✗ dihitung per-model; sampel dipilih dengan CNN_TL sebagai acuan.

### Primer 3 kelas
- **`gradcam_5_7_2_3c_a_benar`** — (A) prediksi benar (*positive*, *neutral*): atensi di region ekspresif.
- **`gradcam_5_7_2_3c_b_misklasifikasi`** — (B) pasangan tertukar **neutral ↔ negative** (ekspresi halus); atensi "salah arah" (latar/oklusi/kacamata/identitas).
- **`gradcam_5_7_2_3c_c_subjek`** — (C) beberapa frame satu partisipan, sebagian benar/salah.

### Primer 7 kelas
- **`gradcam_5_7_2_7c_a_benar`** — (A) prediksi benar (*happy*, *neutral*).
- **`gradcam_5_7_2_7c_b_misklasifikasi`** — (B) **kelas minoritas** (takut/jijik/sedih) tertukar jadi netral/bahagia → atensi gagal menangkap AU pembeda.
- **`gradcam_5_7_2_7c_c_subjek`** — (C) frame multi satu partisipan.

### Kontras CK+
**`gradcam_5_7_2_ckplus.png/pdf`** — Kontras "ideal" CK+
3 sampel ekspresi posed intensitas tinggi (happy/surprised/disgusted), `[asli | CNN_TL CAM]`.
Atensi **jauh lebih tajam & terlokalisasi** pada region ekspresif dibanding kondisi webcam
primer. **Caveat jujur:** model CNN_TL dilatih di primer lalu diterapkan cross-dataset ke
CK+ — beberapa ekspresi (surprised, disgusted) tetap salah diklasifikasi menjadi *neutral*
(domain gap), walau peta atensinya jelas. Ini memperkuat dua poin sekaligus: atensi terbaca
pada ekspresi kuat, dan keterbatasan generalisasi cross-dataset.

---

## Poin insight untuk naskah (RQ fusi)

- **Atensi pada region ekspresif** (mulut/mata/alis) berkorelasi dengan prediksi benar;
  pada kasus salah, atensi bergeser ke latar/oklusi/identitas.
- **Early Fusion (concat/gated) menggeser atensi**: penambahan heatmap landmark cenderung
  menarik atensi ke kontur wajah/region AU. Bandingkan kolom CNN_TL vs concat vs gated pada
  baris yang sama — terutama gated yang men-*gate* atensi spasial.
- **Ekspresi halus (neutral↔negative) & kelas minoritas 7c** adalah sumber error utama,
  konsisten dengan confusion matrix (Gambar CM unimodal & Late Fusion).
- **CK+ vs primer**: kontras kondisi posed-ideal vs natural-webcam menjelaskan mengapa
  angka primer lebih rendah — ekspresi natural lebih halus & atensi lebih ambigu.

*Generated: 5 Juni 2026.*
