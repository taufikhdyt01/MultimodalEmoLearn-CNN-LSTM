"""Generate nb 84 (Skema 1 3c benchmark) + nb 85 (Skema 2 3c cross-dataset)."""
import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent.parent / 'notebooks'
NB_META = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.10'},
}


def code(src):  return {'cell_type': 'code', 'metadata': {}, 'source': src, 'outputs': [], 'execution_count': None}
def md(src):    return {'cell_type': 'markdown', 'metadata': {}, 'source': src}
def write_nb(name, cells):
    nb = {'cells': cells, 'metadata': NB_META, 'nbformat': 4, 'nbformat_minor': 5}
    out = NB_DIR / name
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f'  Wrote: {out.name} ({len(cells)} cells)')


# ════════════════════════════════════════════════════════════════════
# nb 84 — Skema 1: Self-Train-Test 3-class di 4 Benchmark Datasets
# ════════════════════════════════════════════════════════════════════
NB_84 = []

NB_84.append(md('''# 84 — Skema 1 Benchmark 3-class (Self-Train-Test)

**Motivasi:** paper reframe ke 3-class. Untuk konsistensi, benchmark Skema 1 (CK+, JAFFE, RAF-DB, KDEF) perlu di-3-class-kan via REMAP_3.

**Scope:** 4 datasets × 9 archs × B1 baseline = **36 configs**.

**REMAP_3** (uniform across all benchmarks, label_map identik dengan Primer):
- happy(1), surprised(6) → positive(0)
- neutral(0) → neutral(1)
- sad(2), angry(3), fearful(4), disgusted(5) → negative(2)

**Split strategy per dataset:**
- CK+/JAFFE: subject-based hold-out (80% train / 10% val / 10% test by subject)
- RAF-DB: existing train/test + 15% train→val
- KDEF: existing train/val/test (sudah lengkap)

**Hyperparam:** match nb 79 (EPOCHS=50, BATCH=32, val-based selection).

**Estimasi:** ~10-12 jam di T4 (RAF-DB paling besar 11k samples ~2 jam per arch).
'''))

NB_84.append(code('''import sys, os, json
from pathlib import Path
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

PROJECT_ROOT = Path('..').resolve()
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from training.models import (
    EmotionCNN, EmotionFCNN, IntermediateFusion,
    EmotionCNNTransfer, IntermediateFusionTransfer,
    EmotionEarlyFusion, EmotionEarlyFusionTransfer,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

BENCH_DIR = PROJECT_ROOT / 'data' / 'benchmark'
OUT_BASE  = PROJECT_ROOT / 'models' / 'benchmark'

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

NB_84.append(code('''# ── Data loading helpers ──
def load_split_data(dataset):
    """Returns (img_tr, lm_tr, y_tr, img_va, lm_va, y_va, img_te, lm_te, y_te) all 3-class.

    Handles 4 dataset types:
      - ckplus/jaffe: combined data, subject-based hold-out
      - rafdb: pre-split train/test, derive val from train
      - kdef: pre-split train/val/test
    """
    rng = np.random.RandomState(SEED)
    src = BENCH_DIR / f'{dataset}_7class'

    if dataset in ('ckplus', 'jaffe'):
        X_img = np.load(src / 'X_images.npy').astype(np.float32)
        X_lm  = np.load(src / 'X_landmarks.npy').astype(np.float32)
        y     = REMAP_3[np.load(src / 'y_labels.npy')]
        subjects = np.load(src / 'subjects.npy')
        unique_subj = sorted(set(subjects.tolist()))
        # 80/10/10 subject split
        rng.shuffle(unique_subj)
        n = len(unique_subj)
        n_tr = int(0.8 * n)
        n_va = int(0.1 * n)
        tr_subj = set(unique_subj[:n_tr])
        va_subj = set(unique_subj[n_tr:n_tr + n_va])
        te_subj = set(unique_subj[n_tr + n_va:])
        tr_idx = np.array([i for i, s in enumerate(subjects) if s in tr_subj])
        va_idx = np.array([i for i, s in enumerate(subjects) if s in va_subj])
        te_idx = np.array([i for i, s in enumerate(subjects) if s in te_subj])
        return (X_img[tr_idx], X_lm[tr_idx], y[tr_idx],
                X_img[va_idx], X_lm[va_idx], y[va_idx],
                X_img[te_idx], X_lm[te_idx], y[te_idx])

    if dataset == 'rafdb':
        X_tr_img = np.load(src / 'X_train_images.npy').astype(np.float32)
        X_tr_lm  = np.load(src / 'X_train_landmarks.npy').astype(np.float32)
        y_tr     = REMAP_3[np.load(src / 'y_train.npy')]
        X_te_img = np.load(src / 'X_test_images.npy').astype(np.float32)
        X_te_lm  = np.load(src / 'X_test_landmarks.npy').astype(np.float32)
        y_te     = REMAP_3[np.load(src / 'y_test.npy')]
        # Derive val from train (15%) — random split (RAF-DB doesn't have subject info)
        n = len(y_tr); idx = rng.permutation(n)
        n_va = int(0.15 * n)
        va_idx = idx[:n_va]; tr_idx = idx[n_va:]
        return (X_tr_img[tr_idx], X_tr_lm[tr_idx], y_tr[tr_idx],
                X_tr_img[va_idx], X_tr_lm[va_idx], y_tr[va_idx],
                X_te_img, X_te_lm, y_te)

    if dataset == 'kdef':
        return (np.load(src / 'X_train_images.npy').astype(np.float32),
                np.load(src / 'X_train_landmarks.npy').astype(np.float32),
                REMAP_3[np.load(src / 'y_train.npy')],
                np.load(src / 'X_val_images.npy').astype(np.float32),
                np.load(src / 'X_val_landmarks.npy').astype(np.float32),
                REMAP_3[np.load(src / 'y_val.npy')],
                np.load(src / 'X_test_images.npy').astype(np.float32),
                np.load(src / 'X_test_landmarks.npy').astype(np.float32),
                REMAP_3[np.load(src / 'y_test.npy')])
    raise ValueError(dataset)


def make_heatmap(lm, img_size=224, sigma=3.0):
    """Generate Gaussian heatmap dari 136-d landmark."""
    h = np.zeros((img_size, img_size), dtype=np.float32)
    pts = lm.reshape(-1, 2) * img_size
    yg, xg = np.ogrid[:img_size, :img_size]
    denom = 2.0 * sigma * sigma
    for cx, cy in pts:
        g = np.exp(-((xg - cx) ** 2 + (yg - cy) ** 2) / denom)
        h = np.maximum(h, g.astype(np.float32))
    return h

def make_heatmaps_batch(lms, img_size=224, sigma=3.0):
    return np.stack([make_heatmap(lm, img_size, sigma) for lm in lms]).astype(np.float32)
'''))

NB_84.append(code('''# ── Training + evaluation helpers (mirror nb 79 pattern) ──
def stack_4ch(img, hm):
    if hm.ndim == 3: hm = hm[..., None]
    return np.concatenate([img, hm], axis=-1).astype(np.float32)


def build_loader(arch, img, lm, hm, y, shuffle=False):
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
        return DataLoader(TensorDataset(t_img, torch.from_numpy(lm).float(), y_t),
                          batch_size=BATCH, shuffle=shuffle, num_workers=0, pin_memory=True)
    if arch == 'earlyfusion':
        x4 = stack_4ch(img, hm)
        t = torch.from_numpy(x4).permute(0, 3, 1, 2).float()
        return DataLoader(TensorDataset(t, y_t), batch_size=BATCH, shuffle=shuffle,
                          num_workers=0, pin_memory=True)
    raise ValueError(arch)


def eval_loader(model, loader, arch):
    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for batch in loader:
            *x, y = [b.to(device) for b in batch]
            out = model(*x) if arch == 'fusion' else model(x[0])
            yt.append(y.cpu().numpy()); yp.append(out.argmax(1).cpu().numpy())
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
        yt, yp = eval_loader(model, va_loader, arch)
        vf1 = f1_score(yt, yp, average='macro', zero_division=0)
        scheduler.step(vf1)
        if vf1 > best_val:
            best_val, best_ep, stale = vf1, epoch, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, save_path)
        else:
            stale += 1
            if stale >= PATIENCE: break
    return best_val, best_ep


def eval_test(model, arch, te_loader):
    yt, yp = eval_loader(model, te_loader, arch)
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

NB_84.append(code('''# ── Late Fusion helper (2-branch + grid w on val) ──
def softmax_loader(model, loader, arch):
    model.eval()
    p = []
    with torch.no_grad():
        for batch in loader:
            *x, _ = [b.to(device) for b in batch]
            out = model(*x) if arch == 'fusion' else model(x[0])
            p.append(F.softmax(out, dim=1).cpu().numpy())
    return np.concatenate(p)


def grid_w(p_cnn, p_fcnn, y_val):
    best_f1, best_w = 0.0, 0.5
    for w in np.arange(0.0, 1.05, 0.05):
        f1 = f1_score(y_val, (w * p_cnn + (1 - w) * p_fcnn).argmax(1),
                      average='macro', zero_division=0)
        if f1 > best_f1: best_f1, best_w = f1, float(w)
    return best_w, float(best_f1)


# ── Single arch loop ──
ARCH_REGISTRY = {
    'CNN':              (lambda: EmotionCNN(num_classes=NUM_CLASSES),         'cnn',         LR_SCRATCH),
    'FCNN':             (lambda: EmotionFCNN(num_classes=NUM_CLASSES),        'fcnn',        LR_SCRATCH),
    'Intermediate':     (lambda: IntermediateFusion(num_classes=NUM_CLASSES), 'fusion',      LR_SCRATCH),
    'CNN_TL':           (lambda: EmotionCNNTransfer(num_classes=NUM_CLASSES), 'cnn',         LR_TL),
    'Intermediate_TL':  (lambda: IntermediateFusionTransfer(num_classes=NUM_CLASSES), 'fusion',  LR_TL),
    'EarlyFusion':      (lambda: EmotionEarlyFusion(num_classes=NUM_CLASSES), 'earlyfusion', LR_SCRATCH),
    'EarlyFusion_TL':   (lambda: EmotionEarlyFusionTransfer(num_classes=NUM_CLASSES), 'earlyfusion', LR_TL),
}
'''))

NB_84.append(md('## Loop Train: 4 datasets × 9 archs (single + Late Fusion 2-branch)'))

NB_84.append(code('''def run_dataset(dataset):
    print(f"\\n{'='*70}\\n  Dataset: {dataset.upper()} (3-class)\\n{'='*70}")
    img_tr, lm_tr, y_tr, img_va, lm_va, y_va, img_te, lm_te, y_te = load_split_data(dataset)
    print(f'  Train: {len(y_tr)}  Val: {len(y_va)}  Test: {len(y_te)}')
    print(f'  Train dist: {np.bincount(y_tr, minlength=3).tolist()}  '
          f'Val: {np.bincount(y_va, minlength=3).tolist()}  '
          f'Test: {np.bincount(y_te, minlength=3).tolist()}')

    # Heatmap untuk Early Fusion (lazy compute, cache once)
    print(f'  Generating heatmaps...', end=' ')
    hm_tr = make_heatmaps_batch(lm_tr)
    hm_va = make_heatmaps_batch(lm_va)
    hm_te = make_heatmaps_batch(lm_te)
    print(f'tr {hm_tr.shape}, va {hm_va.shape}, te {hm_te.shape}')

    out_dir = OUT_BASE / dataset / '3class'
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    criterion = nn.CrossEntropyLoss()

    # 7 single-arch configs
    for arch_name, (build_fn, arch_type, lr) in ARCH_REGISTRY.items():
        cfg = f'{arch_name}_B1'
        print(f"\\n  -- {cfg} --")
        tr = build_loader(arch_type, img_tr, lm_tr, hm_tr, y_tr, shuffle=True)
        va = build_loader(arch_type, img_va, lm_va, hm_va, y_va)
        te = build_loader(arch_type, img_te, lm_te, hm_te, y_te)
        model = build_fn().to(device)
        save_path = out_dir / f'{arch_name.lower()}_b1.pth'
        best_val, best_ep = train_single(model, arch_type, tr, va, criterion, lr, str(save_path))
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
        m = eval_test(model, arch_type, te)
        m['val_macro_f1'] = float(best_val); m['best_epoch'] = int(best_ep)
        results[cfg] = m
        print(f"    val={best_val:.4f}@{best_ep} test_macro={m['test_macro_f1']:.4f} acc={m['test_accuracy']:.4f}")

    # Late Fusion (scratch + TL): train CNN + FCNN + grid w
    for variant, (cnn_cls, lr_cnn) in [('Late_Fusion', (EmotionCNN, LR_SCRATCH)),
                                         ('Late_Fusion_TL', (EmotionCNNTransfer, LR_TL))]:
        cfg = f'{variant}_B1'
        print(f"\\n  -- {cfg} (2-branch + grid w) --")

        cnn = cnn_cls(num_classes=NUM_CLASSES).to(device)
        cnn_path = out_dir / f'{variant.lower()}_cnn.pth'
        tr_c = build_loader('cnn', img_tr, lm_tr, hm_tr, y_tr, shuffle=True)
        va_c = build_loader('cnn', img_va, lm_va, hm_va, y_va)
        cnn_val, cnn_ep = train_single(cnn, 'cnn', tr_c, va_c, criterion, lr_cnn, str(cnn_path))

        fcnn = EmotionFCNN(num_classes=NUM_CLASSES).to(device)
        fcnn_path = out_dir / f'{variant.lower()}_fcnn.pth'
        tr_f = build_loader('fcnn', img_tr, lm_tr, hm_tr, y_tr, shuffle=True)
        va_f = build_loader('fcnn', img_va, lm_va, hm_va, y_va)
        fcnn_val, fcnn_ep = train_single(fcnn, 'fcnn', tr_f, va_f, criterion, LR_SCRATCH, str(fcnn_path))

        cnn.load_state_dict(torch.load(cnn_path, map_location=device, weights_only=True))
        fcnn.load_state_dict(torch.load(fcnn_path, map_location=device, weights_only=True))

        te_c = build_loader('cnn', img_te, lm_te, hm_te, y_te)
        te_f = build_loader('fcnn', img_te, lm_te, hm_te, y_te)
        p_val_cnn  = softmax_loader(cnn, va_c, 'cnn')
        p_val_fcnn = softmax_loader(fcnn, va_f, 'fcnn')
        p_te_cnn   = softmax_loader(cnn, te_c, 'cnn')
        p_te_fcnn  = softmax_loader(fcnn, te_f, 'fcnn')

        best_w, best_val = grid_w(p_val_cnn, p_val_fcnn, y_va)
        fused_te = best_w * p_te_cnn + (1 - best_w) * p_te_fcnn
        pred_te = fused_te.argmax(1)
        cm = confusion_matrix(y_te, pred_te, labels=list(range(NUM_CLASSES))).tolist()

        results[cfg] = {
            'val_macro_f1': best_val,
            'best_cnn_weight': best_w,
            'cnn_val_macro_f1': float(cnn_val),
            'fcnn_val_macro_f1': float(fcnn_val),
            'test_macro_f1':    float(f1_score(y_te, pred_te, average='macro', zero_division=0)),
            'test_micro_f1':    float(f1_score(y_te, pred_te, average='micro', zero_division=0)),
            'test_weighted_f1': float(f1_score(y_te, pred_te, average='weighted', zero_division=0)),
            'test_accuracy':    float(accuracy_score(y_te, pred_te)),
            'confusion_matrix': cm,
        }
        print(f'    w_best={best_w:.2f} val={best_val:.4f} test_macro={results[cfg]["test_macro_f1"]:.4f}')

    out_json = OUT_BASE / dataset / f'{dataset}_3c_results.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\\n  Saved: {out_json.relative_to(PROJECT_ROOT)}")
    return results


# Run all 4 datasets
all_results = {}
for ds in ['ckplus', 'jaffe', 'rafdb', 'kdef']:
    all_results[ds] = run_dataset(ds)

# Master summary
with open(OUT_BASE / 'all_3c_skema1_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\\nMaster: {OUT_BASE / 'all_3c_skema1_results.json'}")
'''))

NB_84.append(md('''## Summary table

```bash
git add models/benchmark/ckplus/ckplus_3c_results.json \\
        models/benchmark/jaffe/jaffe_3c_results.json \\
        models/benchmark/rafdb/rafdb_3c_results.json \\
        models/benchmark/kdef/kdef_3c_results.json \\
        models/benchmark/all_3c_skema1_results.json \\
        models/benchmark/*/3class/ \\
        notebooks/results/84_*
git commit -m "Add 3-class Skema 1 benchmark (nb 84, 4 datasets x 9 archs)"
```
'''))


# ════════════════════════════════════════════════════════════════════
# nb 85 — Skema 2: Cross-Dataset 3-class → Primer 3-class
# ════════════════════════════════════════════════════════════════════
NB_85 = []

NB_85.append(md('''# 85 — Skema 2 Cross-Dataset 3-class → Primer 3-class

**Motivasi:** train di benchmark dataset (CK+/JAFFE/RAF-DB/KDEF) dengan label 3-class, lalu test di Primer 3-class test set. **Inference only** — pakai checkpoint dari nb 84.

**Scope:** 4 sources × 7 archs (single + 2 Late Fusion variants) = **36 configs**.

**Estimasi:** ~30 menit di T4 (inference saja, no training).

**Prerequisite:** nb 84 selesai dulu di VPS untuk generate checkpoint per dataset.
'''))

NB_85.append(code('''import sys, os, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

PROJECT_ROOT = Path('..').resolve()
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from training.models import (
    EmotionCNN, EmotionFCNN, IntermediateFusion,
    EmotionCNNTransfer, IntermediateFusionTransfer,
    EmotionEarlyFusion, EmotionEarlyFusionTransfer,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PRIMER_DIR = PROJECT_ROOT / 'data' / 'dataset_frontonly_conf60'
BENCH_BASE = PROJECT_ROOT / 'models' / 'benchmark'
NUM_CLASSES = 3
EMOTIONS = ['positive', 'neutral', 'negative']
REMAP_3 = np.array([1, 0, 2, 2, 2, 2, 0], dtype=np.int64)
BATCH = 32

# Load Primer test (3-class)
img_te = np.load(PRIMER_DIR / 'X_test_images.npy').astype(np.float32)
lm_te  = np.load(PRIMER_DIR / 'X_test_landmarks.npy').astype(np.float32)
hm_te  = np.load(PRIMER_DIR / 'X_test_heatmaps.npy').astype(np.float32)
y_te   = REMAP_3[np.load(PRIMER_DIR / 'y_test.npy')]
print(f'Primer test: {len(y_te)} samples, dist {np.bincount(y_te, minlength=3).tolist()}')
'''))

NB_85.append(code('''# ── Helpers ──
def stack_4ch(img, hm):
    if hm.ndim == 3: hm = hm[..., None]
    return np.concatenate([img, hm], axis=-1).astype(np.float32)

def load_te_loader(arch):
    y_t = torch.from_numpy(y_te).long()
    if arch == 'fcnn':
        return DataLoader(TensorDataset(torch.from_numpy(lm_te).float(), y_t),
                          batch_size=BATCH, num_workers=0)
    if arch == 'cnn':
        t = torch.from_numpy(img_te).permute(0, 3, 1, 2).float()
        return DataLoader(TensorDataset(t, y_t), batch_size=BATCH, num_workers=0)
    if arch == 'fusion':
        t = torch.from_numpy(img_te).permute(0, 3, 1, 2).float()
        return DataLoader(TensorDataset(t, torch.from_numpy(lm_te).float(), y_t),
                          batch_size=BATCH, num_workers=0)
    if arch == 'earlyfusion':
        x4 = stack_4ch(img_te, hm_te)
        t = torch.from_numpy(x4).permute(0, 3, 1, 2).float()
        return DataLoader(TensorDataset(t, y_t), batch_size=BATCH, num_workers=0)


def predict_arch(model, arch, loader):
    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for batch in loader:
            *x, y = [b.to(device) for b in batch]
            out = model(*x) if arch == 'fusion' else model(x[0])
            yt.append(y.cpu().numpy()); yp.append(out.argmax(1).cpu().numpy())
    return np.concatenate(yt), np.concatenate(yp)


def softmax_arch(model, arch, loader):
    model.eval()
    p = []
    with torch.no_grad():
        for batch in loader:
            *x, _ = [b.to(device) for b in batch]
            out = model(*x) if arch == 'fusion' else model(x[0])
            p.append(F.softmax(out, dim=1).cpu().numpy())
    return np.concatenate(p)


def metrics_pack(yt, yp):
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

NB_85.append(code('''# ── Cross-dataset inference loop ──
ARCH_BUILDERS = {
    'CNN':              (lambda: EmotionCNN(num_classes=NUM_CLASSES),                 'cnn'),
    'FCNN':             (lambda: EmotionFCNN(num_classes=NUM_CLASSES),                'fcnn'),
    'Intermediate':     (lambda: IntermediateFusion(num_classes=NUM_CLASSES),         'fusion'),
    'CNN_TL':           (lambda: EmotionCNNTransfer(num_classes=NUM_CLASSES),         'cnn'),
    'Intermediate_TL':  (lambda: IntermediateFusionTransfer(num_classes=NUM_CLASSES), 'fusion'),
    'EarlyFusion':      (lambda: EmotionEarlyFusion(num_classes=NUM_CLASSES),         'earlyfusion'),
    'EarlyFusion_TL':   (lambda: EmotionEarlyFusionTransfer(num_classes=NUM_CLASSES), 'earlyfusion'),
}

cross_results = {}
for ds in ['ckplus', 'jaffe', 'rafdb', 'kdef']:
    bench_dir = BENCH_BASE / ds / '3class'
    print(f"\\n{'='*70}\\n  {ds.upper()} → Primer (3-class)\\n{'='*70}")

    for arch_name, (build_fn, arch_type) in ARCH_BUILDERS.items():
        ckpt = bench_dir / f'{arch_name.lower()}_b1.pth'
        if not ckpt.exists():
            print(f'  [SKIP] {arch_name}: {ckpt} missing')
            continue
        model = build_fn().to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        loader = load_te_loader(arch_type)
        yt, yp = predict_arch(model, arch_type, loader)
        m = metrics_pack(yt, yp)
        cfg = f'{ds}_to_primer_{arch_name}'
        cross_results[cfg] = m
        print(f'  {arch_name:<20}: macro={m["test_macro_f1"]:.4f}  acc={m["test_accuracy"]:.4f}')

    # Late Fusion (2 variants)
    for variant, cnn_cls in [('Late_Fusion', EmotionCNN), ('Late_Fusion_TL', EmotionCNNTransfer)]:
        cnn_ckpt  = bench_dir / f'{variant.lower()}_cnn.pth'
        fcnn_ckpt = bench_dir / f'{variant.lower()}_fcnn.pth'
        if not (cnn_ckpt.exists() and fcnn_ckpt.exists()):
            print(f'  [SKIP] {variant}: checkpoints missing')
            continue
        cnn = cnn_cls(num_classes=NUM_CLASSES).to(device)
        cnn.load_state_dict(torch.load(cnn_ckpt, map_location=device, weights_only=True))
        fcnn = EmotionFCNN(num_classes=NUM_CLASSES).to(device)
        fcnn.load_state_dict(torch.load(fcnn_ckpt, map_location=device, weights_only=True))

        # Use w from benchmark trained val (load from results.json)
        bench_results = json.load(open(BENCH_BASE / ds / f'{ds}_3c_results.json'))
        best_w = bench_results.get(f'{variant}_B1', {}).get('best_cnn_weight', 0.5)

        p_cnn  = softmax_arch(cnn, 'cnn', load_te_loader('cnn'))
        p_fcnn = softmax_arch(fcnn, 'fcnn', load_te_loader('fcnn'))
        fused = best_w * p_cnn + (1 - best_w) * p_fcnn
        pred = fused.argmax(1)
        m = metrics_pack(y_te, pred)
        m['best_cnn_weight'] = best_w
        cfg = f'{ds}_to_primer_{variant}'
        cross_results[cfg] = m
        print(f'  {variant:<20}: w={best_w:.2f} macro={m["test_macro_f1"]:.4f} acc={m["test_accuracy"]:.4f}')

with open(BENCH_BASE / 'all_3c_skema2_cross_results.json', 'w') as f:
    json.dump(cross_results, f, indent=2)
print(f"\\nSaved master: {BENCH_BASE / 'all_3c_skema2_cross_results.json'}")
'''))

NB_85.append(md('''## Commit

```bash
git add models/benchmark/all_3c_skema2_cross_results.json notebooks/results/85_*
git commit -m "Add 3-class Skema 2 cross-dataset → Primer (nb 85, inference)"
```
'''))


def main():
    print(f'Output dir: {NB_DIR}')
    write_nb('84_threeclass_skema1_benchmark.ipynb', NB_84)
    write_nb('85_threeclass_skema2_crossdataset.ipynb', NB_85)


if __name__ == '__main__':
    main()
