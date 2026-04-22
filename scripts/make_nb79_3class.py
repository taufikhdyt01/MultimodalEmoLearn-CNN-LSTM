"""Generate nb 79 — 3-class exploration (5 arch × 3 scenario = 15 configs)."""
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / 'notebooks' / '79_threeclass_exploration.ipynb'

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

CELLS.append(md('''# 79 — 3-Class Exploration (Positive / Neutral / Negative)

**Motivasi:** 4-class mapping (neutral/happy/sad/negative) tidak ada di literature —
3-class valence (positive/neutral/negative) adalah standard Russell 1980 circumplex.

**REMAP_3** (applied to 7-class labels):
- `happy, surprised` → **positive** (0)
- `neutral` → **neutral** (1)
- `sad, angry, fearful, disgusted` → **negative** (2)

**Imbalance improvement (vs 4-class):**

| Mapping | Kelas | Max/min ratio |
|---|---|:---:|
| 7-class | 7 (raw) | 1:1138 |
| 4-class | neutral/happy/sad/negative | 1:62 |
| **3-class** | positive/neutral/negative | **1:14** (4× lebih balanced) |

**Scope:** 5 arsitektur × 3 scenario = **15 configs**:
- FCNN, CNN TL, Intermediate TL, Late Fusion TL, Early Fusion TL
- × B1 (baseline) / B2 (class weights) / B3 (weights + augmentation)

**Prerequisite:** run `python src/preprocessing/augment_conf60_3class.py` dulu
untuk generate augmented dataset (scenario B3).

**Selection criterion:** val macro F1 (proper methodology).

**Estimasi:** 7-10 jam di T4.
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
    EmotionFCNN,
    EmotionCNNTransfer,
    IntermediateFusionTransfer,
    EmotionEarlyFusionTransfer,
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
LR_TL = 5e-5
LR_SCRATCH = 1e-4
SEED = 42

torch.manual_seed(SEED); np.random.seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
'''))

CELLS.append(code('''# ── Data loading ──
def load_conf60(split):
    """Original conf60 data → 3-class (REMAP_3)."""
    img = np.load(DATA_CONF60 / f'X_{split}_images.npy').astype(np.float32)
    lm  = np.load(DATA_CONF60 / f'X_{split}_landmarks.npy').astype(np.float32)
    hm  = np.load(DATA_CONF60 / f'X_{split}_heatmaps.npy').astype(np.float32)
    y7  = np.load(DATA_CONF60 / f'y_{split}.npy')
    return img, lm, hm, REMAP_3[y7]

def load_aug(split):
    """Augmented 3-class data (train augmented, val/test not)."""
    img = np.load(DATA_AUG3 / f'X_{split}_images.npy').astype(np.float32)
    lm  = np.load(DATA_AUG3 / f'X_{split}_landmarks.npy').astype(np.float32)
    hm  = np.load(DATA_AUG3 / f'X_{split}_heatmaps.npy').astype(np.float32)
    y   = np.load(DATA_AUG3 / f'y_{split}.npy').astype(np.int64)
    return img, lm, hm, y

# Load both variants
img_tr, lm_tr, hm_tr, y_tr     = load_conf60('train')
img_va, lm_va, hm_va, y_va     = load_conf60('val')
img_te, lm_te, hm_te, y_te     = load_conf60('test')

if DATA_AUG3.exists():
    img_tr_aug, lm_tr_aug, hm_tr_aug, y_tr_aug = load_aug('train')
    aug_available = True
else:
    print('[WARN] Augmented dataset missing — B3 scenarios will be skipped')
    aug_available = False

print(f'Conf60 train: {len(y_tr)}  val: {len(y_va)}  test: {len(y_te)}')
if aug_available:
    print(f'Augmented train: {len(y_tr_aug)}')

print(f'\\n3-class dist (train): {np.bincount(y_tr, minlength=3).tolist()}')
if aug_available:
    print(f'3-class dist (aug train): {np.bincount(y_tr_aug, minlength=3).tolist()}')
'''))

CELLS.append(code('''# ── Compute class weights per scenario ──
def class_weights(y, num_classes=3):
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    w = counts.sum() / (num_classes * np.maximum(counts, 1))
    w_norm = w / w.sum() * num_classes
    return torch.FloatTensor(w_norm).to(device)

W_B2 = class_weights(y_tr)           # weights dari original conf60 distribution
W_B3 = class_weights(y_tr_aug) if aug_available else None  # dari augmented
print(f'B2 weights (original): {W_B2.cpu().numpy().tolist()}')
if aug_available:
    print(f'B3 weights (augmented): {W_B3.cpu().numpy().tolist()}')

# Stack 4-channel for Early Fusion (image + heatmap)
def stack_4ch(img, hm):
    if hm.ndim == 3:
        hm = hm[..., None]
    return np.concatenate([img, hm], axis=-1).astype(np.float32)
'''))

CELLS.append(code('''# ── Unified training function ──
def build_loader(arch, split_data, shuffle=False):
    """arch: 'fcnn' | 'cnn' | 'fusion' | 'earlyfusion'."""
    img, lm, hm, y = split_data
    y_t = torch.from_numpy(y).long()
    if arch == 'fcnn':
        ds = TensorDataset(torch.from_numpy(lm).float(), y_t)
    elif arch == 'cnn':
        t = torch.from_numpy(img).permute(0, 3, 1, 2).float()
        ds = TensorDataset(t, y_t)
    elif arch == 'fusion':
        t_img = torch.from_numpy(img).permute(0, 3, 1, 2).float()
        t_lm  = torch.from_numpy(lm).float()
        ds = TensorDataset(t_img, t_lm, y_t)
    elif arch == 'earlyfusion':
        x4 = stack_4ch(img, hm)
        t = torch.from_numpy(x4).permute(0, 3, 1, 2).float()
        ds = TensorDataset(t, y_t)
    else:
        raise ValueError(arch)
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, num_workers=2, pin_memory=True)


def eval_model(model, loader, arch):
    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for batch in loader:
            *x, y = batch
            x = [xi.to(device) for xi in x]
            out = model(*x) if arch == 'fusion' else model(x[0])
            yt.append(y.numpy())
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
            *x, y = batch
            x = [xi.to(device) for xi in x]; y = y.to(device)
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

CELLS.append(md('## Run 15 Configs (5 arch × 3 scenarios)'))

CELLS.append(code('''# ── Build scenario data references ──
def get_data(scenario):
    """Returns (train_tuple, weights_tensor_or_None, suffix)."""
    if scenario == 'B1':
        return (img_tr, lm_tr, hm_tr, y_tr), None, 'B1'
    if scenario == 'B2':
        return (img_tr, lm_tr, hm_tr, y_tr), W_B2, 'B2'
    if scenario == 'B3':
        if not aug_available:
            return None, None, None
        return (img_tr_aug, lm_tr_aug, hm_tr_aug, y_tr_aug), W_B3, 'B3'

val_data  = (img_va, lm_va, hm_va, y_va)
test_data = (img_te, lm_te, hm_te, y_te)

# Architecture registry: name → (model_class, arch_type, lr)
ARCH_REGISTRY = {
    'FCNN':             (lambda: EmotionFCNN(num_classes=NUM_CLASSES), 'fcnn', LR_SCRATCH),
    'CNN_TL':           (lambda: EmotionCNNTransfer(num_classes=NUM_CLASSES), 'cnn', LR_TL),
    'Intermediate_TL':  (lambda: IntermediateFusionTransfer(num_classes=NUM_CLASSES), 'fusion', LR_TL),
    'Early_Fusion_TL':  (lambda: EmotionEarlyFusionTransfer(num_classes=NUM_CLASSES), 'earlyfusion', LR_TL),
}

results = {}

for arch_name, (build_fn, arch_type, lr) in ARCH_REGISTRY.items():
    for sc in ['B1', 'B2', 'B3']:
        train_data, weights, suf = get_data(sc)
        if train_data is None:
            print(f'\\n[SKIP] {arch_name} {sc} — augmented data missing')
            continue
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
        metrics = eval_test(model, arch_type, te_loader)
        metrics['val_macro_f1'] = float(best_val)
        metrics['best_epoch'] = int(best_ep)

        results[cfg] = metrics
        print(f"  val={best_val:.4f}@ep{best_ep}  test_macro={metrics['test_macro_f1']:.4f}  "
              f"acc={metrics['test_accuracy']:.4f}")

# Save intermediate results
with open(OUT_DIR / 'results_single_and_fusion_tl.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'\\nSaved: {OUT_DIR / "results_single_and_fusion_tl.json"}')
'''))

CELLS.append(md('''## Late Fusion TL 3-class (special: 2-branch training + val-tuned w)

Late Fusion TL butuh train CNN TL + FCNN branch terpisah, lalu grid-search `w ∈ [0.00, 0.05, ..., 1.00]` di val untuk optimal weight.'''))

CELLS.append(code('''# ── Late Fusion TL helpers ──
def softmax_from_loader(model, loader, arch):
    model.eval()
    probs_list = []
    with torch.no_grad():
        for batch in loader:
            *x, y = batch
            x = [xi.to(device) for xi in x]
            out = model(x[0]) if arch != 'fusion' else model(*x)
            probs_list.append(F.softmax(out, dim=1).cpu().numpy())
    return np.concatenate(probs_list)


def search_best_w(p_cnn, p_fcnn, y_val):
    best_f1, best_w = 0.0, 0.5
    for w in np.arange(0.0, 1.05, 0.05):
        fused = w * p_cnn + (1.0 - w) * p_fcnn
        pred = fused.argmax(1)
        f1 = f1_score(y_val, pred, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1; best_w = float(w)
    return best_w, float(best_f1)


LF_OUT = OUT_DIR / 'Late_Fusion_TL'
LF_OUT.mkdir(parents=True, exist_ok=True)
lf_results = {}

for sc in ['B1', 'B2', 'B3']:
    train_data, weights, suf = get_data(sc)
    if train_data is None:
        continue
    cfg = f'Late_Fusion_TL_{sc}'
    print(f"\\n{'='*70}\\n  {cfg} (train 2 branches + grid w)\\n{'='*70}")

    criterion = nn.CrossEntropyLoss(weight=weights)

    # CNN TL branch
    cnn = EmotionCNNTransfer(num_classes=NUM_CLASSES).to(device)
    cnn_path = LF_OUT / f'cnn_tl_{sc.lower()}.pth'
    tr_cnn = build_loader('cnn', train_data, shuffle=True)
    va_cnn = build_loader('cnn', val_data)
    print(f'  [CNN] training...')
    cnn_val, cnn_ep = train_single(cnn, 'cnn', tr_cnn, va_cnn, criterion, LR_TL, str(cnn_path))
    print(f'    val={cnn_val:.4f}@ep{cnn_ep}')

    # FCNN branch
    fcnn = EmotionFCNN(num_classes=NUM_CLASSES).to(device)
    fcnn_path = LF_OUT / f'fcnn_{sc.lower()}.pth'
    tr_fcnn = build_loader('fcnn', train_data, shuffle=True)
    va_fcnn = build_loader('fcnn', val_data)
    print(f'  [FCNN] training...')
    fcnn_val, fcnn_ep = train_single(fcnn, 'fcnn', tr_fcnn, va_fcnn, criterion, LR_SCRATCH, str(fcnn_path))
    print(f'    val={fcnn_val:.4f}@ep{fcnn_ep}')

    # Load best ckpts + compute softmax on val & test
    cnn.load_state_dict(torch.load(cnn_path, map_location=device, weights_only=True))
    fcnn.load_state_dict(torch.load(fcnn_path, map_location=device, weights_only=True))

    te_cnn  = build_loader('cnn', test_data)
    te_fcnn = build_loader('fcnn', test_data)

    # Softmax
    p_val_cnn  = softmax_from_loader(cnn, va_cnn, 'cnn')
    p_val_fcnn = softmax_from_loader(fcnn, va_fcnn, 'fcnn')
    p_te_cnn   = softmax_from_loader(cnn, te_cnn, 'cnn')
    p_te_fcnn  = softmax_from_loader(fcnn, te_fcnn, 'fcnn')

    # Grid-search w di val (proper)
    best_w, best_val_f1 = search_best_w(p_val_cnn, p_val_fcnn, y_va)
    print(f'  w_best (val-tuned) = {best_w:.2f}  val_macro_f1 = {best_val_f1:.4f}')

    # Apply best w di test
    fused_te = best_w * p_te_cnn + (1.0 - best_w) * p_te_fcnn
    pred_te = fused_te.argmax(1)
    test_macro = float(f1_score(y_te, pred_te, average='macro', zero_division=0))
    test_acc   = float(accuracy_score(y_te, pred_te))
    test_mic   = float(f1_score(y_te, pred_te, average='micro', zero_division=0))
    test_wf1   = float(f1_score(y_te, pred_te, average='weighted', zero_division=0))
    cm = confusion_matrix(y_te, pred_te, labels=list(range(NUM_CLASSES))).tolist()

    lf_results[cfg] = {
        'val_macro_f1': best_val_f1,
        'best_cnn_weight': best_w,
        'cnn_val_macro_f1': float(cnn_val),
        'fcnn_val_macro_f1': float(fcnn_val),
        'test_macro_f1': test_macro,
        'test_micro_f1': test_mic,
        'test_weighted_f1': test_wf1,
        'test_accuracy': test_acc,
        'confusion_matrix': cm,
    }
    print(f'  TEST: macro={test_macro:.4f}  acc={test_acc:.4f}')

with open(LF_OUT / 'results.json', 'w') as f:
    json.dump(lf_results, f, indent=2)

# Merge into main results
results.update(lf_results)
'''))

CELLS.append(md('## Summary & Comparison'))

CELLS.append(code('''# ── Save master results JSON ──
with open(OUT_DIR / 'all_results_3class.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'Master results saved: {OUT_DIR / "all_results_3class.json"}')

# ── Print summary table ──
print(f"\\n{'='*90}")
print(f'  3-Class Exploration Summary (Primer conf60, val-based selection)')
print(f"{'='*90}")
print(f"  {'Config':<32} {'Val Macro':>10} {'Test Macro':>11} {'Test Acc':>10} {'w_best':>8}")
print(f"  {'-'*85}")
for cfg in sorted(results.keys()):
    r = results[cfg]
    w = r.get('best_cnn_weight')
    w_s = f'{w:.2f}' if w is not None else '—'
    print(f"  {cfg:<32} {r['val_macro_f1']:>10.4f} {r['test_macro_f1']:>11.4f} "
          f"{r['test_accuracy']:>10.4f} {w_s:>8}")

# Best per scenario
print(f'\\n\\n  Best val-based per scenario:')
for sc in ['B1', 'B2', 'B3']:
    sc_configs = {k: v for k, v in results.items() if k.endswith('_' + sc)}
    if not sc_configs: continue
    best_cfg = max(sc_configs.keys(), key=lambda k: sc_configs[k]['val_macro_f1'])
    r = sc_configs[best_cfg]
    print(f"    {sc}: {best_cfg}  val={r['val_macro_f1']:.4f}  test={r['test_macro_f1']:.4f}")
'''))

CELLS.append(md('''## Comparison vs 4-class & 7-class Best

Reference baselines (val-tuned):
- **4-class Intermediate TL B3** = Macro F1 0.521 (juara val-tuned overall)
- **7-class Early Fusion TL B3** = Macro F1 0.333 (juara 7c)
- **4-class CNN TL B3** = Macro F1 0.507 (best single-modal 4c)

Kalau 3-class best > 0.60 → valence mapping memang lebih cocok untuk Primer natural data.
Kalau 3-class best 0.50-0.60 → comparable dengan 4-class, literature precedent jadi alasan preferensi.
Kalau 3-class best < 0.50 → remap terlalu lossy, 4-class/7-class tetap preferred.
'''))


def main():
    nb = {'cells': CELLS, 'metadata': NB_META, 'nbformat': 4, 'nbformat_minor': 5}
    with open(NB_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f'Wrote: {NB_PATH} ({len(CELLS)} cells)')


if __name__ == '__main__':
    main()
