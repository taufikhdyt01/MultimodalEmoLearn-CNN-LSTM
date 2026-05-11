# Hand Occlusion Detection Report

Dataset: `data/dataset_frontonly_conf60/` — MediaPipe HandLandmarker (tasks API, min_conf=0.3, num_hands=2, RunningMode.IMAGE)

**Interpretasi:** Sample ditandai `occluded` jika MediaPipe mendeteksi ≥1 tangan pada crop wajah 224×224. Karena gambar input sudah merupakan face crop, deteksi tangan di crop = tangan menutupi area wajah.

## Ringkasan per Split

| Split | N | Occluded | % |
|:---|---:|---:|---:|
| train | 5287 | 940 | 17.78% |
| val | 579 | 176 | 30.40% |
| test | 929 | 197 | 21.21% |
| **TOTAL** | **6795** | **1313** | **19.32%** |

## Breakdown per Kelas (semua split digabung)

| Class ID | Emotion | N | Occluded | % |
|:---:|:---|---:|---:|---:|
| 0 | neutral | 5691 | 910 | 15.99% |
| 1 | happy | 651 | 160 | 24.58% |
| 2 | sad | 361 | 219 | 60.66% |
| 3 | angry | 32 | 7 | 21.88% |
| 4 | fearful | 5 | 5 | 100.00% |
| 5 | disgusted | 16 | 3 | 18.75% |
| 6 | surprised | 39 | 9 | 23.08% |

## File Output

```
deploy/data_cleaning/hand_occlusion_{train,val,test}.npz
  occluded_mask (N,) bool
  hand_count    (N,) int8
  max_score     (N,) float32
```

Cara filter training set:

```python
import numpy as np
mask = np.load('deploy/data_cleaning/hand_occlusion_train.npz')['occluded_mask']
keep = ~mask
X_clean = np.load('data/dataset_frontonly_conf60/X_train_images.npy')[keep]
y_clean = np.load('data/dataset_frontonly_conf60/y_train.npy')[keep]
```