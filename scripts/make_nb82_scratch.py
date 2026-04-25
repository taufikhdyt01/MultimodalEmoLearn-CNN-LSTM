"""Generate nb 82 — 3-class scratch variants (12 missing configs untuk paper JITeCS)."""
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / 'notebooks' / '82_threeclass_scratch.ipynb'

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

CELLS.append(md('''# 82 — 3-Class Scratch Variants (12 missing configs untuk paper JITeCS)

**Motivasi:** nb 79 cover 15 configs (5 arch × B1/B2/B3) tapi hanya **TL variants** + FCNN. Untuk paper JITeCS yang butuh full 27-config grid (mirror 7-class & 4-class earlier), 12 konfigurasi **scratch** belum ada:

| Arch | B1 | B2 | B3 |
|---|:---:|:---:|:---:|
| CNN scratch | ❌ | ❌ | ❌ |
| Intermediate scratch | ❌ | ❌ | ❌ |
| Late Fusion scratch | ❌ | ❌ | ❌ |
| Early Fusion scratch | ❌ | ❌ | ❌ |

**Total missing: 12 configs.**

**Already covered in nb 79 (15 configs, no need to re-run):**
- FCNN × B1/B2/B3 (FCNN inherently scratch, no TL variant)
- CNN TL × B1/B2/B3
- Intermediate TL × B1/B2/B3
- Late Fusion TL × B1/B2/B3
- Early Fusion TL × B1/B2/B3

**Methodology identik dengan nb 79:** hyperparam align (EPOCHS=50, BATCH=32, val-based selection). Use same data sources (conf60 + augmented_3class).

**Estimasi:** ~5-7 jam di T4 (12 configs, mix arch).
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

from training.models import (
    EmotionCNN,
    EmotionFCNN,
    IntermediateFusion,
    EmotionEarlyFusion,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

DATA_CONF60 = PROJECT_ROOT / 'data' / 'dataset_frontonly_conf60'
DATA_AUG3   = PROJECT_ROOT / 'data' / 'dataset_frontonly_conf60_3class_augmented'
OUT_DIR     = PROJECT_ROOT / 'models' / 'frontonly_conf60' / '3class'
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 3
EMOTIONS = ['positive', 'neutral', 'negative']
REMAP_3 = np.array([1, 0, 2, 2, 2, 2, 0], dtype=np.int64)

BATCH = 32
EPOCHS = 50
PATIENCE = 15
LR_SCRATCH = 1e-4
SEED = 42

torch.manual_seed(SEED); np.random.seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
'''))

CELLS.append(code('''# ── Data loading ──
def load_conf60(split):
    img = np.load(DATA_CONF60 / f'X_{split}_images.npy').astype(np.float32)
    lm  = np.load(DATA_CONF60 / f'X_{split}_landmarks.npy').astype(np.float32)
    hm  = np.load(DATA_CONF60 / f'X_{split}_heatmaps.npy').astype(np.float32)
    y7  = np.load(DATA_CONF60 / f'y_{split}.npy')
    return img, lm, hm, REMAP_3[y7]

def load_aug(split):
    img = np.load(DATA_AUG3 / f'X_{split}_images.npy').astype(np.float32)
    lm  = np.load(DATA_AUG3 / f'X_{split}_landmarks.npy').astype(np.float32)
    hm  = np.load(DATA_AUG3 / f'X_{split}_heatmaps.npy').astype(np.float32)
    y   = np.load(DATA_AUG3 / f'y_{split}.npy').astype(np.int64)
    return img, lm, hm, y

img_tr, lm_tr, hm_tr, y_tr = load_conf60('train')
img_va, lm_va, hm_va, y_va = load_conf60('val')
img_te, lm_te, hm_te, y_te = load_conf60('test')
img_tr_aug, lm_tr_aug, hm_tr_aug, y_tr_aug = load_aug('train')

print(f'Conf60 train: {len(y_tr)}  val: {len(y_va)}  test: {len(y_te)}')
print(f'Augmented train: {len(y_tr_aug)}')


def class_weights(y, num_classes=3):
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    w = counts.sum() / (num_classes * np.maximum(counts, 1))
    return torch.FloatTensor(w / w.sum() * num_classes).to(device)

W_B2 = class_weights(y_tr)
W_B3 = class_weights(y_tr_aug)


def stack_4ch(img, hm):
    if hm.ndim == 3:
        hm = hm[..., None]
    return np.concatenate([img, hm], axis=-1).astype(np.float32)


def get_data(sc):
    if sc == 'B1':
        return (img_tr, lm_tr, hm_tr, y_tr), None
    if sc == 'B2':
        return (img_tr, lm_tr, hm_tr, y_tr), W_B2
    if sc == 'B3':
        return (img_tr_aug, lm_tr_aug, hm_tr_aug, y_tr_aug), W_B3
    raise ValueError(sc)

val_data  = (img_va, lm_va, hm_va, y_va)
test_data = (img_te, lm_te, hm_te, y_te)
'''))

CELLS.append(code('''# ── Loaders + helpers (same pattern as nb 79) ──
def build_loader(arch, data, shuffle=False):
    img, lm, hm, y = data
    y_t = torch.from_numpy(y).long()
    if arch == 'fcnn':
        return DataLoader(TensorDataset(torch.from_numpy(lm).float(), y_t),
                          batch_size=BATCH, shuffle=shuffle, num_workers=0, pin_memory=True)
    if arch == 'cnn':
        t = torch.from_numpy(img).permute(0, 3, 1, 2).float()
        return DataLoader(TensorDataset(t, y_t), batch_size=BATCH, shuffle=shuffle,
                          num_workers=0, pin_memory=True)
    if arch == 'fusion':
        t_img = torch.from_numpy(img).permute(0, 3, 1, 2).float()
        t_lm  = torch.from_numpy(lm).float()
        return DataLoader(TensorDataset(t_img, t_lm, y_t), batch_size=BATCH, shuffle=shuffle,
                          num_workers=0, pin_memory=True)
    if arch == 'earlyfusion':
        x4 = stack_4ch(img, hm)
        t = torch.from_numpy(x4).permute(0, 3, 1, 2).float()
        return DataLoader(TensorDataset(t, y_t), batch_size=BATCH, shuffle=shuffle,
                          num_workers=0, pin_memory=True)
    raise ValueError(arch)


def eval_model(model, loader, arch):
    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for batch in loader:
            *x, y = [b.to(device) for b in batch]
            out = model(*x) if arch == 'fusion' else model(x[0])
            yt.append(y.cpu().numpy())
            yp.append(out.argmax(1).cpu().numpy())
    return np.concatenate(yt), np.concatenate(yp)


def train_single(model, arch, tr_loader, va_loader, criterion, lr, save_path):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8, min_lr=1e-7)

    best_val, best_ep, stale, best_state = 0.0, 0, 0, None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch in tr_loader:
            *x, y = [b.to(device) for b in batch]
            out = model(*x) if arch == 'fusion' else model(x[0])
            loss = criterion(out, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        yt, yp = eval_model(model, va_loader, arch)
        vf1 = f1_score(yt, yp, average='macro', zero_division=0)
        scheduler.step(vf1)

        if vf1 > best_val:
            best_val, best_ep, stale = vf1, epoch, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, save_path)
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    return best_val, best_ep


def eval_test(model, arch, te_loader):
    yt, yp = eval_model(model, te_loader, arch)
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

CELLS.append(md('## Run scratch arch (single-arch loop): CNN, Intermediate, Early Fusion × B1/B2/B3 = 9 configs'))

CELLS.append(code('''ARCH_REGISTRY = {
    'CNN_scratch':           (lambda: EmotionCNN(num_classes=NUM_CLASSES),         'cnn',         LR_SCRATCH),
    'Intermediate_scratch':  (lambda: IntermediateFusion(num_classes=NUM_CLASSES), 'fusion',      LR_SCRATCH),
    'Early_Fusion_scratch':  (lambda: EmotionEarlyFusion(num_classes=NUM_CLASSES), 'earlyfusion', LR_SCRATCH),
}

results = {}
for arch_name, (build_fn, arch_type, lr) in ARCH_REGISTRY.items():
    for sc in ['B1', 'B2', 'B3']:
        train_data, weights = get_data(sc)
        cfg = f'{arch_name}_{sc}'
        print(f"\\n{'='*70}\\n  {cfg}\\n{'='*70}")

        tr_loader = build_loader(arch_type, train_data, shuffle=True)
        va_loader = build_loader(arch_type, val_data)
        te_loader = build_loader(arch_type, test_data)

        model = build_fn().to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)

        save_dir = OUT_DIR / arch_name
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f'{arch_name.lower()}_{sc.lower()}.pth'

        best_val, best_ep = train_single(model, arch_type, tr_loader, va_loader,
                                          criterion, lr, str(save_path))
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
        m = eval_test(model, arch_type, te_loader)
        m['val_macro_f1'] = float(best_val)
        m['best_epoch'] = int(best_ep)
        results[cfg] = m
        print(f"  val={best_val:.4f}@ep{best_ep}  test_macro={m['test_macro_f1']:.4f}  "
              f"acc={m['test_accuracy']:.4f}")

with open(OUT_DIR / 'scratch_singlearch_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'\\nSaved: {OUT_DIR / "scratch_singlearch_results.json"}')
'''))

CELLS.append(md('## Run Late Fusion scratch × B1/B2/B3 (3 configs, 2-branch + grid w)'))

CELLS.append(code('''def softmax_from_loader(model, loader, arch):
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in loader:
            *x, y = [b.to(device) for b in batch]
            out = model(*x) if arch == 'fusion' else model(x[0])
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


LF_OUT = OUT_DIR / 'Late_Fusion_scratch'
LF_OUT.mkdir(parents=True, exist_ok=True)
lf_results = {}

for sc in ['B1', 'B2', 'B3']:
    train_data, weights = get_data(sc)
    cfg = f'Late_Fusion_scratch_{sc}'
    print(f"\\n{'='*70}\\n  {cfg} (2-branch + grid w)\\n{'='*70}")

    criterion = nn.CrossEntropyLoss(weight=weights)

    # CNN scratch branch
    cnn = EmotionCNN(num_classes=NUM_CLASSES).to(device)
    cnn_path = LF_OUT / f'cnn_{sc.lower()}.pth'
    print(f'  [CNN scratch] training...')
    cnn_val, cnn_ep = train_single(cnn, 'cnn',
                                    build_loader('cnn', train_data, shuffle=True),
                                    build_loader('cnn', val_data),
                                    criterion, LR_SCRATCH, str(cnn_path))
    print(f'    CNN val={cnn_val:.4f}@ep{cnn_ep}')

    # FCNN branch (scratch)
    fcnn = EmotionFCNN(num_classes=NUM_CLASSES).to(device)
    fcnn_path = LF_OUT / f'fcnn_{sc.lower()}.pth'
    print(f'  [FCNN] training...')
    fcnn_val, fcnn_ep = train_single(fcnn, 'fcnn',
                                      build_loader('fcnn', train_data, shuffle=True),
                                      build_loader('fcnn', val_data),
                                      criterion, LR_SCRATCH, str(fcnn_path))
    print(f'    FCNN val={fcnn_val:.4f}@ep{fcnn_ep}')

    cnn.load_state_dict(torch.load(cnn_path, map_location=device, weights_only=True))
    fcnn.load_state_dict(torch.load(fcnn_path, map_location=device, weights_only=True))

    p_val_cnn  = softmax_from_loader(cnn, build_loader('cnn', val_data), 'cnn')
    p_val_fcnn = softmax_from_loader(fcnn, build_loader('fcnn', val_data), 'fcnn')
    p_te_cnn   = softmax_from_loader(cnn, build_loader('cnn', test_data), 'cnn')
    p_te_fcnn  = softmax_from_loader(fcnn, build_loader('fcnn', test_data), 'fcnn')

    best_w, best_val_f1 = search_best_w(p_val_cnn, p_val_fcnn, y_va)
    print(f'  w_best (val-tuned) = {best_w:.2f}  val_macro = {best_val_f1:.4f}')

    fused_te = best_w * p_te_cnn + (1.0 - best_w) * p_te_fcnn
    pred_te = fused_te.argmax(1)
    cm = confusion_matrix(y_te, pred_te, labels=list(range(NUM_CLASSES))).tolist()

    lf_results[cfg] = {
        'val_macro_f1': best_val_f1,
        'best_cnn_weight': best_w,
        'cnn_val_macro_f1':  float(cnn_val),
        'fcnn_val_macro_f1': float(fcnn_val),
        'test_macro_f1':    float(f1_score(y_te, pred_te, average='macro', zero_division=0)),
        'test_micro_f1':    float(f1_score(y_te, pred_te, average='micro', zero_division=0)),
        'test_weighted_f1': float(f1_score(y_te, pred_te, average='weighted', zero_division=0)),
        'test_accuracy':    float(accuracy_score(y_te, pred_te)),
        'confusion_matrix': cm,
    }
    print(f"  TEST: macro={lf_results[cfg]['test_macro_f1']:.4f}  acc={lf_results[cfg]['test_accuracy']:.4f}")

with open(LF_OUT / 'results.json', 'w') as f:
    json.dump(lf_results, f, indent=2)
results.update(lf_results)
'''))

CELLS.append(md('## Save Combined Results + Summary'))

CELLS.append(code('''# Save combined scratch results (12 configs)
with open(OUT_DIR / 'scratch_all_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'Saved 12 scratch configs: {OUT_DIR / "scratch_all_results.json"}')

# Print summary
print(f"\\n{'='*85}")
print(f'  3-Class Scratch Variants (12 configs, val-based selection)')
print(f"{'='*85}")
print(f"  {'Config':<32} {'Val':>8} {'Test Macro':>11} {'Test Acc':>10} {'w':>6}")
print(f"  {'-'*80}")
for cfg in sorted(results.keys()):
    r = results[cfg]
    w = r.get('best_cnn_weight')
    ws = f'{w:.2f}' if w is not None else '—'
    print(f"  {cfg:<32} {r['val_macro_f1']:>8.4f} {r['test_macro_f1']:>11.4f} "
          f"{r['test_accuracy']:>10.4f} {ws:>6}")

# Best per scenario
print(f'\\n  Best per scenario (by val):')
for sc in ['B1', 'B2', 'B3']:
    sc_configs = {k: v for k, v in results.items() if k.endswith('_' + sc)}
    if not sc_configs: continue
    best = max(sc_configs, key=lambda k: sc_configs[k]['val_macro_f1'])
    r = sc_configs[best]
    print(f"    {sc}: {best}  val={r['val_macro_f1']:.4f}  test={r['test_macro_f1']:.4f}")

print('\\n  Reference (nb 79 TL variants juara):')
print(f"    Late Fusion TL B3 ⭐ val=0.6229 test=0.6370  (overall juara 3c)")
print(f"    CNN TL B3 val=0.4953 test=0.7055")
print(f"    Intermediate TL B3 val=0.5005 test=0.6891")
'''))

CELLS.append(md('''## Merge ke Master Results JSON

Setelah nb 82 selesai, merge `scratch_all_results.json` ke `all_results_3class.json` (jadi total 27 configs lengkap untuk paper JITeCS).

```python
import json
from pathlib import Path

OUT_DIR = Path('../models/frontonly_conf60/3class')
master = json.load(open(OUT_DIR / 'all_results_3class.json'))
scratch = json.load(open(OUT_DIR / 'scratch_all_results.json'))
master.update(scratch)
with open(OUT_DIR / 'all_results_3class_full.json', 'w') as f:
    json.dump(master, f, indent=2)
print(f'Merged: {len(master)} configs total')
```

Commit:
```bash
git add models/frontonly_conf60/3class/CNN_scratch/ \\
        models/frontonly_conf60/3class/Intermediate_scratch/ \\
        models/frontonly_conf60/3class/Late_Fusion_scratch/ \\
        models/frontonly_conf60/3class/Early_Fusion_scratch/ \\
        models/frontonly_conf60/3class/scratch_*.json \\
        models/frontonly_conf60/3class/all_results_3class_full.json \\
        notebooks/results/82_*
git commit -m "Add 3-class scratch variants results (nb 82, 12 configs) — close paper grid"
```
'''))


def main():
    nb = {'cells': CELLS, 'metadata': NB_META, 'nbformat': 4, 'nbformat_minor': 5}
    with open(NB_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f'Wrote: {NB_PATH} ({len(CELLS)} cells)')


if __name__ == '__main__':
    main()
