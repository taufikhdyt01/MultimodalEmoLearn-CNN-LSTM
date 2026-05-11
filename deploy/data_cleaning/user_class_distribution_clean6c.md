# User × Class Distribution (Clean 6-class)

After filtering hand-occluded samples and dropping `fearful` (had 0 clean samples).
Total: **5482 samples** across **37 users**.

Per-class totals:

| neutral | happy | sad | angry | disgusted | surprised |
|---:|---:|---:|---:|---:|---:|
| 4781 | 491 | 142 | 25 | 13 | 30 |

## Minority-class holders

Users yang menyumbang sampel di kelas minoritas (angry/disgusted/surprised). Re-split harus pastikan val & test masing-masing dapat ≥1 user dari tiap kelas minoritas.

### `angry` (25 clean samples)

| user | orig_split | n |
|:---|:---:|---:|
| 116 | train | 10 |
| 107 | train | 4 |
| 113 | train | 3 |
| 103 | train | 2 |
| 202 | train | 2 |
| 210 | val | 2 |
| 102 | train | 1 |
| 214 | val | 1 |

### `disgusted` (13 clean samples)

| user | orig_split | n |
|:---|:---:|---:|
| 97 | train | 2 |
| 106 | train | 2 |
| 107 | train | 2 |
| 111 | train | 2 |
| 117 | test | 2 |
| 102 | train | 1 |
| 116 | train | 1 |
| 214 | val | 1 |

### `surprised` (30 clean samples)

| user | orig_split | n |
|:---|:---:|---:|
| 210 | val | 11 |
| 107 | train | 5 |
| 97 | train | 3 |
| 104 | val | 3 |
| 117 | test | 2 |
| 203 | train | 2 |
| 99 | train | 1 |
| 102 | train | 1 |
| 108 | train | 1 |
| 214 | val | 1 |

## Full user × class matrix

| user | orig | neutral | happy | sad | angry | disgusted | surprised | total |
|:---|:---:|---:|---:|---:|---:|---:|---:|---:|
| 97 | train | 118 | 32 | 18 | 0 | 2 | 3 | 173 |
| 99 | train | 107 | 14 | 25 | 0 | 0 | 1 | 147 |
| 100 | train | 171 | 7 | 3 | 0 | 0 | 0 | 181 |
| 101 | train | 26 | 0 | 2 | 0 | 0 | 0 | 28 |
| 102 | train | 77 | 26 | 1 | 1 | 1 | 1 | 107 |
| 103 | train | 127 | 10 | 1 | 2 | 0 | 0 | 140 |
| 104 | val | 129 | 35 | 2 | 0 | 0 | 3 | 169 |
| 106 | train | 41 | 39 | 13 | 0 | 2 | 0 | 95 |
| 107 | train | 195 | 76 | 7 | 4 | 2 | 5 | 289 |
| 108 | train | 180 | 11 | 7 | 0 | 0 | 1 | 199 |
| 109 | test | 209 | 14 | 0 | 0 | 0 | 0 | 223 |
| 110 | train | 75 | 6 | 1 | 0 | 0 | 0 | 82 |
| 111 | train | 214 | 8 | 0 | 0 | 2 | 0 | 224 |
| 112 | train | 165 | 3 | 0 | 0 | 0 | 0 | 168 |
| 113 | train | 188 | 1 | 0 | 3 | 0 | 0 | 192 |
| 114 | train | 123 | 8 | 12 | 0 | 0 | 0 | 143 |
| 115 | train | 112 | 0 | 0 | 0 | 0 | 0 | 112 |
| 116 | train | 88 | 2 | 2 | 10 | 1 | 0 | 103 |
| 117 | test | 36 | 128 | 0 | 0 | 2 | 2 | 168 |
| 118 | test | 130 | 7 | 1 | 0 | 0 | 0 | 138 |
| 197 | train | 37 | 0 | 0 | 0 | 0 | 0 | 37 |
| 200 | train | 123 | 0 | 2 | 0 | 0 | 0 | 125 |
| 201 | train | 226 | 1 | 1 | 0 | 0 | 0 | 228 |
| 202 | train | 69 | 0 | 12 | 2 | 0 | 0 | 83 |
| 203 | train | 79 | 0 | 0 | 0 | 0 | 2 | 81 |
| 205 | train | 177 | 26 | 3 | 0 | 0 | 0 | 206 |
| 206 | train | 371 | 0 | 0 | 0 | 0 | 0 | 371 |
| 207 | train | 236 | 3 | 1 | 0 | 0 | 0 | 240 |
| 208 | test | 136 | 8 | 5 | 0 | 0 | 0 | 149 |
| 209 | train | 157 | 0 | 0 | 0 | 0 | 0 | 157 |
| 210 | val | 68 | 0 | 0 | 2 | 0 | 11 | 81 |
| 211 | train | 129 | 2 | 17 | 0 | 0 | 0 | 148 |
| 212 | train | 118 | 4 | 0 | 0 | 0 | 0 | 122 |
| 213 | test | 52 | 2 | 0 | 0 | 0 | 0 | 54 |
| 214 | val | 144 | 4 | 2 | 1 | 1 | 1 | 153 |
| 215 | train | 22 | 10 | 2 | 0 | 0 | 0 | 34 |
| 216 | train | 126 | 4 | 2 | 0 | 0 | 0 | 132 |
| **TOTAL** | | **4781** | **491** | **142** | **25** | **13** | **30** | **5482** |