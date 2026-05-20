# Per-class F1 Summary (auto-generated)

> Output dari `scripts/analyze_per_class.py`. Per-class F1 dari semua run terbaik.

## Best F1 per class (Primer 7c)

| Class | Best Landmark | Best Image | Best Fusion | Best Overall |
|---|:---:|:---:|:---:|:---:|
| neutral | 0.933 (facs_plus_bs_80) | 0.883 (cnn_tl) | 0.926 (fusion_late_scratch) | **0.933** (facs_plus_bs_80) |
| happy | 0.874 (raw_136) | 0.742 (cnn_tl) | 0.854 (fusion_late_scratch) | **0.874** (raw_136) |
| sad | 0.529 (raw_136) | 0.462 (cnn_tl) | 0.545 (fusion_late_scratch) | **0.545** (fusion_late_scratch) |
| angry | 0.000 () | 0.000 () | 0.000 () | **0.000** () |
| fearful | 0.000 () | 0.000 () | 0.000 () | **0.000** () |
| disgusted | 0.067 (facs_28) | 0.000 () | 0.000 () | **0.067** (facs_28) |
| surprised | 0.250 (facs_plus_bs_80) | 0.000 () | 0.222 (fusion_late_tl_facs_plus_bs_80) | **0.250** (facs_plus_bs_80) |

## Class size distribution (train) — Primer

| Class (7c) | Train count | Ratio vs largest |
|---|:---:|:---:|
| neutral | 4526 | 1.0000 |
| happy | 416 | 0.0919 |
| sad | 287 | 0.0634 |
| angry | 27 | 0.0060 |
| fearful | 2 | 0.0004 |
| disgusted | 13 | 0.0029 |
| surprised | 16 | 0.0035 |

---

*Regenerate: `python scripts/analyze_per_class.py`*