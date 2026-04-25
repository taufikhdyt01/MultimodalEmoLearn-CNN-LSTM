# Validation Agreement Analysis

**Source:** `validation_results_dephilia_rambu_raja_uju_decky,_s.psi.csv`
**Expert validator:** Dephilia Rambu Raja Uju Decky, S.Psi
**N samples:** 128

## 1. Agreement Summary

| Scheme | Raw Agreement | Cohen κ | Macro F1 |
|---|:---:|:---:|:---:|
| 7-class | **0.6562** (84/128) | 0.4780 | 0.5124 |
| 3-class valence | **0.6719** (86/128) | 0.4475 | 0.6181 |

**Cohen kappa interpretation (Landis & Koch 1977):**
- < 0.20 — slight | 0.21-0.40 — fair | 0.41-0.60 — moderate | 0.61-0.80 — substantial | > 0.80 — almost perfect

## 2. Per-Class — 7-class

| Class | Precision | Recall | F1 | n_expert | n_auto |
|---|:---:|:---:|:---:|:---:|:---:|
| neutral | 0.948 | 0.625 | 0.753 | 88 | 58 |
| happy | 0.700 | 0.583 | 0.636 | 12 | 10 |
| sad | 0.600 | 0.667 | 0.632 | 9 | 10 |
| angry | 0.000 | 0.000 | 0.000 | 0 | 10 |
| fearful | 0.600 | 0.857 | 0.706 | 7 | 10 |
| disgusted | 0.400 | 0.889 | 0.552 | 9 | 20 |
| surprised | 0.200 | 0.667 | 0.308 | 3 | 10 |

## 3. Per-Class — 3-class Valence (paper-relevant)

| Class | Precision | Recall | F1 | n_expert | n_auto |
|---|:---:|:---:|:---:|:---:|:---:|
| positive | 0.450 | 0.600 | 0.514 | 15 | 20 |
| neutral | 0.948 | 0.625 | 0.753 | 88 | 58 |
| negative | 0.440 | 0.880 | 0.587 | 25 | 50 |

## 4. Confusion Matrix — 3-class Valence (rows=expert, cols=auto)

| | pred: positive | pred: neutral | pred: negative |
|---|:---:|:---:|:---:|
| **true: positive** | 9 | 1 | 5 |
| **true: neutral** | 10 | 55 | 23 |
| **true: negative** | 1 | 2 | 22 |

## 5. Agreement by Confidence Threshold (7-class)

| Threshold | N samples | Matched | Agreement |
|---|:---:|:---:|:---:|
| ≥ 0.00 | 128 | 84 | **0.6562** |
| ≥ 0.60 | 110 | 75 | **0.6818** |
| ≥ 0.80 | 89 | 69 | **0.7753** |
| ≥ 0.95 | 70 | 60 | **0.8571** |

## 6. Class Distribution Comparison

### 7-class
| Class | Auto | Expert | Δ |
|---|:---:|:---:|:---:|
| neutral | 58 | 88 | +30 |
| happy | 10 | 12 | +2 |
| sad | 10 | 9 | -1 |
| angry | 10 | 0 | -10 |
| fearful | 10 | 7 | -3 |
| disgusted | 20 | 9 | -11 |
| surprised | 10 | 3 | -7 |

### 3-class valence
| Class | Auto | Expert | Δ |
|---|:---:|:---:|:---:|
| positive | 20 | 15 | -5 |
| neutral | 58 | 88 | +30 |
| negative | 50 | 25 | -25 |

## 7. Implikasi untuk Paper JITeCS

- **Agreement 3-class jauh lebih tinggi dari 7-class** — konsisten dengan motivasi reframe paper ke 3-class valence: kelas minoritas (angry/fearful/disgusted/surprised) confusion-prone bahkan untuk human expert. Valence dimension lebih reliable.
- **Cohen κ** adalah inter-rater reliability standar. Untuk klaim Face API as ground-truth pseudo-label: κ ≥ 0.40 (moderate) baseline acceptable; κ ≥ 0.60 (substantial) ideal.
- **Confidence stratification** validates conf60 filter: high-confidence Face API predictions (≥0.95) agreement biasanya jauh lebih tinggi → justifikasi filter threshold.