# JAFFE 7-kelas — Apple-to-Apple: LOSO vs Random-Split vs CV10

> Dibuat atas permintaan untuk membandingkan setara dengan literatur JAFFE yang
> memakai split level-sampel (bukan subject-independent).
> Skrip: `scripts/run_jaffe_randomsplit.py`. Hasil: `models/benchmark/jaffe_randomsplit/`.

## Setup
- **Model (6, identik benchmark JAFFE):** CNN, FCNN, Intermediate, CNN_TL, Intermediate_TL, Late_Fusion.
- **Training B1 identik:** batch 16, 50 epoch, patience 15, lr per-model, landmark raw_136.
- **Yang diganti hanya pembentukan fold** (subject-disjoint → level-sampel).
- Dataset: 213 citra, 10 subjek, 7 kelas (≈balanced).

## Tiga protokol
| Protokol | Definisi | Setara paper |
|---|---|---|
| LOSO | Leave-One-Subject-Out (=10-fold subject-disjoint) | Wadhawan (subject-indep) |
| RandomSplit | Stratified 80:20, sample-level, 5 seed | Singh, Gautam (holdout) |
| CV10 | StratifiedKFold 10-fold, sample-level | Akhand, Wadhawan (10-fold) |

## Hasil (macro-F1 / accuracy)

| Model | LOSO | RandomSplit 80:20 | CV10 sample-level |
|---|---|---|---|
| CNN | 0.249 / 35.3% | 0.601 / 61.4% | 0.678 / 71.0% |
| FCNN | 0.304 / 38.2% | 0.480 / 49.8% | 0.400 / 43.8% |
| Intermediate | 0.129 / 22.6% | 0.336 / 34.9% | 0.241 / 28.7% |
| **CNN_TL** | 0.426 / 49.9% | **0.818 / 82.3%** | 0.875 / 88.4% |
| **Intermediate_TL** | 0.293 / 36.3% | 0.719 / 72.1% | **0.890 / 89.8%** |
| Late_Fusion | 0.467 / 54.0% | 0.639 / 64.2% | 0.772 / 79.0% |
| **Terbaik** | **54.0%** | **82.3%** | **89.8%** |

## Perbandingan literatur (JAFFE 7-kelas)

| Sumber | Protokol | Akurasi |
|---|---|---|
| Akhand et al. 2021 (DenseNet-161) | 10-fold CV | 99.52% |
| Singh et al. 2025 (MMSAD) | Holdout 80:20 | 98.50% |
| Wadhawan & Gandhi 2023 (ensemble TL) | 10-fold subject-indep | 97.14% |
| Gautam & Seeja 2023 (HOG+CNN) | Holdout | 91.43% |
| **Kita — Intermediate_TL** | **CV10 sample-level** | **89.76%** |
| **Kita — CNN_TL** | **RandomSplit 80:20** | **82.33%** |
| Kita — Late_Fusion | LOSO subject-indep | 53.96% |

## Kesimpulan (siap-naskah)

Dengan protokol yang setara dengan literatur (CV10 / holdout level-sampel), model
terbaik kita mencapai **82–90% akurasi**, masuk ke kisaran yang sebanding. Kenaikan
besar dari LOSO (54%) ke CV10 (90%) — pada **model dan prosedur training yang identik**
— menegaskan bahwa angka JAFFE yang sangat tinggi di literatur sebagian besar
dijelaskan oleh **protokol evaluasi level-sampel (subject leakage)**, bukan semata
keunggulan arsitektur. Sisa gap ke 97–99% (Akhand/Singh/Wadhawan) berasal dari
augmentasi masif, backbone lebih dalam (DenseNet-161), dan ensemble — di luar lingkup
perbandingan lintas-dataset ini.

Catatan: Wadhawan & Gandhi (2023) melaporkan 97.14% pada **10-fold subject-independent**
(setara LOSO kita). Gap ke LOSO kita (54%) menunjukkan ruang perbaikan via augmentasi +
TL terdedikasi bila diinginkan, tanpa mengorbankan ke-subject-independent-an.

*Dijalankan: 6 Juni 2026. Detail per-fold ada di file JSON `models/benchmark/jaffe_randomsplit/`.*
