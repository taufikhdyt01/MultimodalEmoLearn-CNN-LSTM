# Test Subset Analysis by Face API Confidence

**Model:** Late Fusion TL B3 (3-class juara val-tuned, w_best = 0.15)
**Tujuan:** evaluasi efek label noise — apakah model performance scale dengan confidence input?

## Hasil

| Conf Threshold | N | % Retained | Macro F1 | Acc | pos F1 | neu F1 | neg F1 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ≥ 0.60 | 929 | 100.0% | **0.6370** | 0.7836 | 0.735 | 0.848 | 0.328 |
| ≥ 0.70 | 892 | 96.0% | **0.6422** | 0.7915 | 0.737 | 0.853 | 0.336 |
| ≥ 0.80 | 846 | 91.1% | **0.6388** | 0.8061 | 0.754 | 0.864 | 0.298 |
| ≥ 0.90 | 785 | 84.5% | **0.6446** | 0.8178 | 0.747 | 0.876 | 0.312 |
| ≥ 0.95 | 724 | 77.9% | **0.6188** | 0.8301 | 0.748 | 0.886 | 0.222 |
| ≥ 0.99 | 631 | 67.9% | **0.5808** | 0.8605 | 0.760 | 0.910 | 0.071 |

## Interpretasi

- Pattern (kalau Macro F1 naik dengan threshold) menunjukkan label noise adalah faktor pembatas, bukan model limitation.
- Argumen paper Discussion: model performance scale dengan label quality — gap dari Face API agreement (Cohen κ = 0.45) accountable for residual error pada full test set.
- Trade-off ini justify retain conf60 untuk training (preserve sample sufficiency + minority class viability) tapi acknowledge label noise impact saat report metric.

## Caveat

- Subset sizes shrinking saat threshold naik → metric variance naik (semakin sedikit sample minority).
- Per-class F1 negative class harus dilihat juga, bukan macro saja — at high threshold, negative count kecil.
- Comparison vs expert agreement: di high-conf subset, expert κ = 0.86 (n=70 dari validation CSV) → di test full subset conf95, model performance harus lebih dekat ke this ceiling.