import sys, os, json
from pathlib import Path
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
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

BATCH = 128
EPOCHS = 50
PATIENCE = 15
LR_TL = 5e-5
LR_SCRATCH = 1e-4
SEED = 42

torch.manual_seed(SEED); np.random.seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)


# ── Data loading helpers ──
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


# ── Training + evaluation helpers (mirror nb 79 pattern) ──
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


# ── Late Fusion helper (2-branch + grid w on val) ──
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


def run_dataset(dataset):
    print(f'\n{"="*60}\n  {dataset.upper()}\n{"="*60}')

    out_dir = OUT_BASE / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load img/lm/y splits ──
    img_tr, lm_tr, y_tr, img_va, lm_va, y_va, img_te, lm_te, y_te = load_split_data(dataset)
    print(f'  splits → tr={len(y_tr)}, va={len(y_va)}, te={len(y_te)}')

    # ── Load heatmaps from file (same split logic as load_split_data) ──
    src_dir = BENCH_DIR / f'{dataset}_7class'
    if dataset == 'rafdb':
        rng2 = np.random.RandomState(SEED)
        hm_all = np.load(src_dir / 'X_train_heatmaps.npy').astype(np.float32)
        n = len(np.load(src_dir / 'y_train.npy'))
        idx = rng2.permutation(n)
        n_va = int(0.15 * n)
        va_idx = idx[:n_va]; tr_idx = idx[n_va:]
        hm_tr = hm_all[tr_idx]; hm_va = hm_all[va_idx]
        hm_te = np.load(src_dir / 'X_test_heatmaps.npy').astype(np.float32)
    elif dataset == 'kdef':
        hm_tr = np.load(src_dir / 'X_train_heatmaps.npy').astype(np.float32)
        hm_va = np.load(src_dir / 'X_val_heatmaps.npy').astype(np.float32)
        hm_te = np.load(src_dir / 'X_test_heatmaps.npy').astype(np.float32)
    else:
        hm_tr = make_heatmaps_batch(lm_tr)
        hm_va = make_heatmaps_batch(lm_va)
        hm_te = make_heatmaps_batch(lm_te)

    criterion = nn.CrossEntropyLoss()
    results = {}
    softmax_preds = {}

    # ── Single arch loop ──
    for arch_name, (model_fn, arch_type, lr) in ARCH_REGISTRY.items():
        print(f'\n  [{arch_name}]')
        tr_loader = build_loader(arch_type, img_tr, lm_tr, hm_tr, y_tr, shuffle=True)
        va_loader = build_loader(arch_type, img_va, lm_va, hm_va, y_va)
        te_loader = build_loader(arch_type, img_te, lm_te, hm_te, y_te)

        run_results = []
        softmax_preds[arch_name] = {}

        for b in range(1, 4):
            torch.manual_seed(SEED + b); np.random.seed(SEED + b)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED + b)

            model = model_fn().to(device)
            save_path = out_dir / f'{arch_name}_b{b}.pth'
            if save_path.exists():
                print(f'    B{b}: [SKIP] .pth exists → load+eval')
                model.load_state_dict(torch.load(save_path, map_location=device))
                model.eval()
                best_val, best_ep = 0.0, 0
            else:
                best_val, best_ep = train_single(model, arch_type, tr_loader, va_loader,
                                                 criterion, lr, save_path)
                model.load_state_dict(torch.load(save_path, map_location=device))
                model.eval()

            test_res = eval_test(model, arch_type, te_loader)
            test_res['val_f1'] = float(best_val)
            test_res['best_ep'] = int(best_ep)
            run_results.append(test_res)
            print(f'    B{b}: val_f1={best_val:.4f} ep={best_ep} ' +
                  f'test_f1={test_res["test_macro_f1"]:.4f}')

            if arch_name in ('CNN', 'FCNN', 'CNN_TL'):
                softmax_preds[arch_name][b] = {
                    'val': softmax_loader(model, va_loader, arch_type),
                    'te':  softmax_loader(model, te_loader, arch_type),
                }

        results[arch_name] = {
            'runs': run_results,
            'test_macro_f1_mean': float(np.mean([r['test_macro_f1'] for r in run_results])),
            'test_macro_f1_std':  float(np.std([r['test_macro_f1'] for r in run_results])),
        }

    # ── Late Fusion ──
    for cnn_k, fcnn_k, lf_name in [('CNN', 'FCNN', 'LateFusion'), ('CNN_TL', 'FCNN', 'LateFusion_TL')]:
        if cnn_k not in softmax_preds or fcnn_k not in softmax_preds:
            continue
        print(f'\n  [{lf_name}]')
        lf_runs = []
        for b in range(1, 4):
            if b not in softmax_preds[cnn_k] or b not in softmax_preds[fcnn_k]:
                continue
            w, val_f1 = grid_w(softmax_preds[cnn_k][b]['val'],
                               softmax_preds[fcnn_k][b]['val'], y_va)
            fused = w * softmax_preds[cnn_k][b]['te'] + (1 - w) * softmax_preds[fcnn_k][b]['te']
            y_pred = fused.argmax(1)
            run_res = {
                'test_macro_f1':    float(f1_score(y_te, y_pred, average='macro', zero_division=0)),
                'test_micro_f1':    float(f1_score(y_te, y_pred, average='micro', zero_division=0)),
                'test_weighted_f1': float(f1_score(y_te, y_pred, average='weighted', zero_division=0)),
                'test_accuracy':    float(accuracy_score(y_te, y_pred)),
                'weight_cnn': float(w), 'val_f1': float(val_f1),
            }
            lf_runs.append(run_res)
            print(f'    B{b}: w={w:.2f} val_f1={val_f1:.4f} test_f1={run_res["test_macro_f1"]:.4f}')

        if lf_runs:
            results[lf_name] = {
                'runs': lf_runs,
                'test_macro_f1_mean': float(np.mean([r['test_macro_f1'] for r in lf_runs])),
                'test_macro_f1_std':  float(np.std([r['test_macro_f1'] for r in lf_runs])),
            }

    # ── Save per-dataset results ──
    out_json = out_dir / f'{dataset}_3c_results.json'
    with open(out_json, 'w') as fout:
        json.dump(results, fout, indent=2)
    print(f'\n  Saved → {out_json}')
    return results



# ── KDEF only runner (GPU 1) ──
out_json = OUT_BASE / 'kdef' / 'kdef_3c_results.json'
if out_json.exists():
    print('[RESUME] kdef_3c_results.json exists → skip')
else:
    result = run_dataset('kdef')
    master = OUT_BASE / 'all_3c_skema1_results.json'
    all_res = json.load(open(master)) if master.exists() else {}
    all_res['kdef'] = result
    with open(master, 'w') as f:
        json.dump(all_res, f, indent=2)
    print(f'Master updated: {master}')
