"""Generate nb 81 — Geometric Features (arahan dosen #4, Liliana 2019)."""
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / 'notebooks' / '81_geometric_features.ipynb'

NB_META = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.10'},
}


def code(src):
    return {'cell_type': 'code', 'metadata': {}, 'source': src,
            'outputs': [], 'execution_count': None}


def md(src):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': src}


CELLS = []

CELLS.append(md('''# 81 — Geometric Features (arahan dosen #4, Liliana 2019)

**Motivasi:** paper dosen pembimbing (Liliana et al. 2019, Cognitive Processing Springer) mendefinisikan 20-dim geometric features dari 68 landmarks — 10 facial components × 2 metrik (eccentricity + distance ratio). Direct extension work beliau.

**Strategi (3-class Primer conf60, val-based selection):**

| Setup | Input | Dim | Model | Tujuan |
|---|---|:---:|---|---|
| A (baseline) | Raw landmark | 136 | FCNN (existing) | baseline dari nb 79 FCNN B1/B3 |
| **B** | Geometric only | **20** | FCNN_geom | test interpretability (compact features) |
| **C** | Raw + Geometric | **156** | FCNN_combined | augmented input vector |
| **D** | Late Fusion TL + FCNN_combined | — | Late Fusion + combined branch | target beat 0.623 plain LF TL B3 |

**Total 5 configs** = B × {B1, B3} + C × {B1, B3} + D × B3.

**Prerequisite:** run `python src/preprocessing/compute_geometric_features.py` untuk generate `X_{split}_geometric.npy` di:
- `data/dataset_frontonly_conf60/` (B1 scenario)
- `data/dataset_frontonly_conf60_3class_augmented/` (B3 scenario, kalau sudah di-gen via nb 79 prereq)

**Baselines dari nb 79 (3-class val-based):**
- FCNN B1 val=0.603 test=0.589
- FCNN B3 val=0.619 test=0.634
- Late Fusion TL B3 ⭐ val=0.623 test=0.637 (w=0.15) — target

**Estimasi:** ~2-3 jam di T4 (5 configs, FCNN cepat).
'''))

CELLS.append(code('''import sys, os, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

PROJECT_ROOT = Path('..').resolve()
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from training.models import EmotionCNNTransfer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

DATA_CONF60 = PROJECT_ROOT / 'data' / 'dataset_frontonly_conf60'
DATA_AUG3   = PROJECT_ROOT / 'data' / 'dataset_frontonly_conf60_3class_augmented'
OUT_DIR     = PROJECT_ROOT / 'models' / 'frontonly_conf60' / '3class' / 'Geometric'
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 3
EMOTIONS = ['positive', 'neutral', 'negative']
REMAP_3 = np.array([1, 0, 2, 2, 2, 2, 0], dtype=np.int64)

BATCH = 32
EPOCHS = 50
PATIENCE = 15
LR = 1e-3
LR_TL = 5e-5
SEED = 42

torch.manual_seed(SEED); np.random.seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
'''))

CELLS.append(code('''# ── Prerequisite check: geometric features ──
for d in [DATA_CONF60, DATA_AUG3]:
    print(f'\\n  {d.name}:')
    for split in ['train', 'val', 'test']:
        p = d / f'X_{split}_geometric.npy'
        if not p.exists():
            print(f'    [MISSING] {p.name} — run: python src/preprocessing/compute_geometric_features.py --data-dir {d.relative_to(PROJECT_ROOT)}')
        else:
            arr = np.load(p)
            print(f'    {split}: {arr.shape}  mean={arr.mean():.3f}  std={arr.std():.3f}')
'''))

CELLS.append(code('''# ── Load data ──
def load_conf60(split):
    lm  = np.load(DATA_CONF60 / f'X_{split}_landmarks.npy').astype(np.float32)
    geo = np.load(DATA_CONF60 / f'X_{split}_geometric.npy').astype(np.float32)
    img = np.load(DATA_CONF60 / f'X_{split}_images.npy').astype(np.float32)
    y7  = np.load(DATA_CONF60 / f'y_{split}.npy')
    return img, lm, geo, REMAP_3[y7]

def load_aug(split):
    lm  = np.load(DATA_AUG3 / f'X_{split}_landmarks.npy').astype(np.float32)
    geo = np.load(DATA_AUG3 / f'X_{split}_geometric.npy').astype(np.float32)
    img = np.load(DATA_AUG3 / f'X_{split}_images.npy').astype(np.float32)
    y   = np.load(DATA_AUG3 / f'y_{split}.npy').astype(np.int64)
    return img, lm, geo, y

img_tr, lm_tr, geo_tr, y_tr = load_conf60('train')
img_va, lm_va, geo_va, y_va = load_conf60('val')
img_te, lm_te, geo_te, y_te = load_conf60('test')
img_tr_aug, lm_tr_aug, geo_tr_aug, y_tr_aug = load_aug('train')

print(f'Conf60 train: {len(y_tr)}  | aug train: {len(y_tr_aug)}')
print(f'Val: {len(y_va)}  | Test: {len(y_te)}')

W_B3 = None  # class weights from augmented distribution
counts = np.bincount(y_tr_aug, minlength=3).astype(np.float32)
w = counts.sum() / (3 * np.maximum(counts, 1))
W_B3 = torch.FloatTensor(w / w.sum() * 3).to(device)
print(f'B3 class weights: {W_B3.cpu().numpy().tolist()}')
'''))

CELLS.append(md('## Models'))

CELLS.append(code('''class FCNN_Geometric(nn.Module):
    """Setup B — 20-dim Liliana GF only (compact interpretable)."""
    def __init__(self, in_dim=20, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(in_dim, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.3),
        )
        self.head = nn.Linear(32, num_classes)

    def forward(self, x): return self.head(self.features(x))
    def extract_features(self, x): return self.features(x)   # 32-d


class FCNN_Combined(nn.Module):
    """Setup C — 136 raw + 20 geometric concat = 156-d."""
    def __init__(self, raw_dim=136, geo_dim=20, num_classes=3, feat_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(raw_dim + geo_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, feat_dim), nn.BatchNorm1d(feat_dim), nn.ReLU(), nn.Dropout(0.3),
        )
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, lm, geo):
        x = torch.cat([lm, geo], dim=1)
        return self.head(self.features(x))

    def extract_features(self, lm, geo):
        x = torch.cat([lm, geo], dim=1)
        return self.features(x)   # feat_dim
'''))

CELLS.append(code('''# ── Training helpers ──
def build_fcnn_loader(arrs, y, shuffle=False):
    """arrs: single array or (raw, geo) tuple."""
    y_t = torch.from_numpy(y).long()
    if isinstance(arrs, tuple):
        ts = [torch.from_numpy(a).float() for a in arrs]
        return DataLoader(TensorDataset(*ts, y_t), batch_size=BATCH, shuffle=shuffle,
                          num_workers=0, pin_memory=True)
    return DataLoader(TensorDataset(torch.from_numpy(arrs).float(), y_t),
                      batch_size=BATCH, shuffle=shuffle, num_workers=0, pin_memory=True)


def train_fcnn_variant(model, tr_loader, va_loader, criterion, is_combined, save_path):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8, min_lr=1e-7)
    best_val, best_ep, stale, best_state = 0.0, 0, 0, None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch in tr_loader:
            *xs, y = [b.to(device) for b in batch]
            out = model(*xs) if is_combined else model(xs[0])
            loss = criterion(out, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        # Val
        model.eval()
        yt, yp = [], []
        with torch.no_grad():
            for batch in va_loader:
                *xs, y = [b.to(device) for b in batch]
                out = model(*xs) if is_combined else model(xs[0])
                yt.append(y.cpu().numpy()); yp.append(out.argmax(1).cpu().numpy())
        vf1 = f1_score(np.concatenate(yt), np.concatenate(yp), average='macro', zero_division=0)
        scheduler.step(vf1)

        if vf1 > best_val:
            best_val, best_ep, stale = vf1, epoch, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, save_path)
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f'    early stop @ ep{epoch}')
                break
    return best_val, best_ep


def eval_fcnn_test(model, te_loader, is_combined):
    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for batch in te_loader:
            *xs, y = [b.to(device) for b in batch]
            out = model(*xs) if is_combined else model(xs[0])
            yt.append(y.cpu().numpy()); yp.append(out.argmax(1).cpu().numpy())
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    return {
        'test_macro_f1':    float(f1_score(yt, yp, average='macro', zero_division=0)),
        'test_micro_f1':    float(f1_score(yt, yp, average='micro', zero_division=0)),
        'test_weighted_f1': float(f1_score(yt, yp, average='weighted', zero_division=0)),
        'test_accuracy':    float(accuracy_score(yt, yp)),
        'confusion_matrix': confusion_matrix(yt, yp, labels=list(range(NUM_CLASSES))).tolist(),
        'classification_report': classification_report(yt, yp, target_names=EMOTIONS,
                                                       labels=list(range(NUM_CLASSES)),
                                                       zero_division=0, output_dict=True),
    }
'''))

CELLS.append(md('## Setup B — FCNN_geometric (20-d only) × B1 + B3'))

CELLS.append(code('''results = {}

# ── B1 ──
tr_data = (geo_tr, y_tr)
tr_loader = build_fcnn_loader(geo_tr, y_tr, shuffle=True)
va_loader = build_fcnn_loader(geo_va, y_va)
te_loader = build_fcnn_loader(geo_te, y_te)

print('== FCNN_Geometric B1 (20-d only, no weights, no aug) ==')
model = FCNN_Geometric(in_dim=20, num_classes=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
save_path = OUT_DIR / 'fcnn_geometric_b1.pth'
best_val, best_ep = train_fcnn_variant(model, tr_loader, va_loader, criterion, False, str(save_path))
model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
m = eval_fcnn_test(model, te_loader, False)
m['val_macro_f1'] = float(best_val); m['best_epoch'] = int(best_ep)
results['FCNN_Geometric_B1'] = m
print(f"  val={best_val:.4f}@ep{best_ep}  test_macro={m['test_macro_f1']:.4f}  acc={m['test_accuracy']:.4f}")

# ── B3 ──
tr_loader = build_fcnn_loader(geo_tr_aug, y_tr_aug, shuffle=True)
print('\\n== FCNN_Geometric B3 (20-d, weights+aug) ==')
model = FCNN_Geometric(in_dim=20, num_classes=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss(weight=W_B3)
save_path = OUT_DIR / 'fcnn_geometric_b3.pth'
best_val, best_ep = train_fcnn_variant(model, tr_loader, va_loader, criterion, False, str(save_path))
model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
m = eval_fcnn_test(model, te_loader, False)
m['val_macro_f1'] = float(best_val); m['best_epoch'] = int(best_ep)
results['FCNN_Geometric_B3'] = m
print(f"  val={best_val:.4f}@ep{best_ep}  test_macro={m['test_macro_f1']:.4f}  acc={m['test_accuracy']:.4f}")
'''))

CELLS.append(md('## Setup C — FCNN_combined (156-d raw+geo) × B1 + B3'))

CELLS.append(code('''# Combined train loaders
tr_c_b1 = build_fcnn_loader((lm_tr, geo_tr), y_tr, shuffle=True)
tr_c_b3 = build_fcnn_loader((lm_tr_aug, geo_tr_aug), y_tr_aug, shuffle=True)
va_c = build_fcnn_loader((lm_va, geo_va), y_va)
te_c = build_fcnn_loader((lm_te, geo_te), y_te)

# ── B1 ──
print('== FCNN_Combined B1 (156-d = 136 raw + 20 geo, no weights, no aug) ==')
model = FCNN_Combined(num_classes=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
save_path = OUT_DIR / 'fcnn_combined_b1.pth'
best_val, best_ep = train_fcnn_variant(model, tr_c_b1, va_c, criterion, True, str(save_path))
model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
m = eval_fcnn_test(model, te_c, True)
m['val_macro_f1'] = float(best_val); m['best_epoch'] = int(best_ep)
results['FCNN_Combined_B1'] = m
print(f"  val={best_val:.4f}@ep{best_ep}  test_macro={m['test_macro_f1']:.4f}  acc={m['test_accuracy']:.4f}")

# ── B3 ──
print('\\n== FCNN_Combined B3 (156-d, weights+aug) ==')
model = FCNN_Combined(num_classes=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss(weight=W_B3)
save_path = OUT_DIR / 'fcnn_combined_b3.pth'
best_val, best_ep = train_fcnn_variant(model, tr_c_b3, va_c, criterion, True, str(save_path))
model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
m = eval_fcnn_test(model, te_c, True)
m['val_macro_f1'] = float(best_val); m['best_epoch'] = int(best_ep)
results['FCNN_Combined_B3'] = m
print(f"  val={best_val:.4f}@ep{best_ep}  test_macro={m['test_macro_f1']:.4f}  acc={m['test_accuracy']:.4f}")
'''))

CELLS.append(md('''## Setup D — Late Fusion TL + FCNN_Combined (target beat 0.623)

CNN TL branch dari existing nb 79 checkpoint (reuse). FCNN branch = FCNN_Combined B3 (baru dari step C). Grid search `w` di val untuk val-tuned weight.'''))

CELLS.append(code('''def softmax_from_loader(model, loader, is_combined):
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in loader:
            *xs, y = [b.to(device) for b in batch]
            out = model(*xs) if is_combined else model(xs[0])
            probs.append(F.softmax(out, dim=1).cpu().numpy())
    return np.concatenate(probs)


def search_best_w(p_cnn, p_fcnn, y_val):
    best_f1, best_w = 0.0, 0.5
    for w in np.arange(0.0, 1.05, 0.05):
        fused = w * p_cnn + (1.0 - w) * p_fcnn
        f1 = f1_score(y_val, fused.argmax(1), average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1, best_w = f1, float(w)
    return best_w, float(best_f1)


# ── Setup D: Late Fusion TL + FCNN_Combined B3 ──
print('== Setup D: Late Fusion TL + FCNN_Combined B3 ==')

# CNN TL branch — reuse nb 79 CNN_TL_B3 checkpoint (Late Fusion TL B3 had CNN_TL trained fresh; let's train fresh here for clean methodology)
cnn_ckpt = PROJECT_ROOT / 'models' / 'frontonly_conf60' / '3class' / 'Late_Fusion_TL' / 'cnn_tl_b3.pth'
if cnn_ckpt.exists():
    cnn = EmotionCNNTransfer(num_classes=NUM_CLASSES).to(device)
    cnn.load_state_dict(torch.load(cnn_ckpt, map_location=device, weights_only=True))
    cnn.eval()
    print(f'  CNN TL B3: reuse checkpoint {cnn_ckpt.relative_to(PROJECT_ROOT)}')
else:
    raise FileNotFoundError(f'{cnn_ckpt} missing — run nb 79 first')

# FCNN_Combined B3 — reuse from step C above
fcnn_comb = FCNN_Combined(num_classes=NUM_CLASSES).to(device)
fcnn_comb.load_state_dict(torch.load(OUT_DIR / 'fcnn_combined_b3.pth', map_location=device, weights_only=True))
fcnn_comb.eval()

# Build image loaders for CNN
img_va_t = torch.from_numpy(img_va).permute(0, 3, 1, 2).float()
img_te_t = torch.from_numpy(img_te).permute(0, 3, 1, 2).float()
va_img = DataLoader(TensorDataset(img_va_t, torch.from_numpy(y_va).long()), batch_size=BATCH)
te_img = DataLoader(TensorDataset(img_te_t, torch.from_numpy(y_te).long()), batch_size=BATCH)

# Softmax per branch
p_val_cnn  = softmax_from_loader(cnn, va_img, False)
p_val_fcnn = softmax_from_loader(fcnn_comb, va_c, True)
p_te_cnn   = softmax_from_loader(cnn, te_img, False)
p_te_fcnn  = softmax_from_loader(fcnn_comb, te_c, True)

# Grid-search w di val
best_w, best_val_f1 = search_best_w(p_val_cnn, p_val_fcnn, y_va)
print(f'  w_best (val-tuned) = {best_w:.2f}  val_macro = {best_val_f1:.4f}')

# Apply di test
fused_te = best_w * p_te_cnn + (1.0 - best_w) * p_te_fcnn
pred_te = fused_te.argmax(1)
cm = confusion_matrix(y_te, pred_te, labels=list(range(NUM_CLASSES))).tolist()

m = {
    'val_macro_f1': best_val_f1,
    'best_cnn_weight': best_w,
    'test_macro_f1':    float(f1_score(y_te, pred_te, average='macro', zero_division=0)),
    'test_micro_f1':    float(f1_score(y_te, pred_te, average='micro', zero_division=0)),
    'test_weighted_f1': float(f1_score(y_te, pred_te, average='weighted', zero_division=0)),
    'test_accuracy':    float(accuracy_score(y_te, pred_te)),
    'confusion_matrix': cm,
}
results['Late_Fusion_TL_Combined_B3'] = m
print(f"  TEST: macro={m['test_macro_f1']:.4f}  acc={m['test_accuracy']:.4f}")
'''))

CELLS.append(md('## Summary & Comparison'))

CELLS.append(code('''# Save results
with open(OUT_DIR / 'geometric_3class_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'Saved: {OUT_DIR / "geometric_3class_results.json"}')

# Comparison
print(f"\\n{'='*90}")
print(f'  Geometric Features (Liliana 2019) — 3-class Primer conf60, val-based')
print(f"{'='*90}")
print(f"  {'Config':<34} {'Val':>8} {'Test Macro':>11} {'Test Acc':>10} {'w':>6}")
print(f"  {'-'*80}")

for cfg in sorted(results.keys()):
    r = results[cfg]
    w = r.get('best_cnn_weight')
    ws = f'{w:.2f}' if w is not None else '—'
    print(f"  {cfg:<34} {r['val_macro_f1']:>8.4f} {r['test_macro_f1']:>11.4f} "
          f"{r['test_accuracy']:>10.4f} {ws:>6}")

print('\\n  Baselines (nb 79, 3-class plain):')
print(f"  {'FCNN_B1 (raw 136)':<34} {'0.6025':>8} {'0.5893':>11} {'0.7406':>10} {'—':>6}")
print(f"  {'FCNN_B3 (raw 136)':<34} {'0.6193':>8} {'0.6342':>11} {'0.7492':>10} {'—':>6}")
print(f"  {'Late_Fusion_TL_B3 ⭐ (raw)':<34} {'0.6229':>8} {'0.6370':>11} {'0.7836':>10} {'0.15':>6}")
'''))

CELLS.append(md('''## Analysis & Interpretation

**Target beat:** Late Fusion TL B3 plain (raw landmark) val = **0.6229**.

**Expected outcomes:**

| Setup | vs Raw baseline | Interpretasi |
|---|---|---|
| B Geo-only 20-d | val ~0.50-0.60 | Kalau > 0.50, compact features interpretable — strong argument Liliana |
| C Combined 156-d | val ~0.62+ | Kalau ≈ raw baseline, geo redundant. Kalau > raw, geo adds complementary info |
| D Late Fusion + Combined | val > 0.623 | Novel SOTA 3-class — combined features + fusion sinergis |

**Next step kalau D > 0.62:** dokumentasikan sebagai Liliana 2019 extension finding, cite paper dosen eksplisit di paper JITeCS.

**Kalau flat (semua < raw baseline):** 20-d GF hilang information dari 136-d raw landmarks. Tapi interpretability value tetap valuable untuk BAB Discussion (what geometric features matter per-class).

Commit:
```bash
git add models/frontonly_conf60/3class/Geometric/ notebooks/results/81_*
git commit -m "Add Geometric Features results (nb 81, arahan dosen #4)"
```
'''))


def main():
    nb = {'cells': CELLS, 'metadata': NB_META, 'nbformat': 4, 'nbformat_minor': 5}
    with open(NB_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f'Wrote: {NB_PATH} ({len(CELLS)} cells)')


if __name__ == '__main__':
    main()
