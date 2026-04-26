"""Generate nb 83 — partial re-train dengan history logging untuk lengkapi audit dosen."""
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / 'notebooks' / '83_history_logging_partial.ipynb'

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

CELLS.append(md('''# 83 — Partial Re-Train dengan History Logging (3-Class, untuk audit dosen)

**Motivasi:** nb 79 (15 configs 3-class) tidak save training history per epoch. Untuk audit dosen + analisis convergence, re-train **subset 5 key configs** dengan logging:
- train_loss, val_loss, train_acc, val_acc, val_macro_f1 per epoch

**5 configs dipilih (most relevant untuk paper analysis):**
1. **Late Fusion TL B3** — juara overall val-tuned (0.6229)
2. **Late Fusion TL B1** — runner-up (val 0.6093)
3. **Intermediate TL B3** — feature-level fusion best (val 0.5005)
4. **CNN TL B3** — single-modal best test (test 0.7055, val-test mismatch case)
5. **FCNN B3** — landmark-only best (val 0.6193)

**Hyperparam:** identik nb 79 (EPOCHS=50, LR_TL=5e-5, LR_FCNN=1e-4, BATCH=32, val-based selection).

**Output:**
- Per-config `training_history.json` — array epoch metrics
- Loss curves PNG (subplot 5 configs)
- Accuracy curves PNG
- Val Macro F1 curves PNG

**Estimasi:** ~3-4 jam di T4 (5 configs × ~30-60 min).
'''))

CELLS.append(code('''import sys, os, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score

PROJECT_ROOT = Path('..').resolve()
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from training.models import (
    EmotionFCNN, EmotionCNNTransfer, IntermediateFusionTransfer,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

DATA_CONF60 = PROJECT_ROOT / 'data' / 'dataset_frontonly_conf60'
DATA_AUG3   = PROJECT_ROOT / 'data' / 'dataset_frontonly_conf60_3class_augmented'
OUT_DIR     = PROJECT_ROOT / 'models' / 'frontonly_conf60' / '3class' / 'history'
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 3
REMAP_3 = np.array([1, 0, 2, 2, 2, 2, 0], dtype=np.int64)
BATCH = 32
EPOCHS = 50
PATIENCE = 15
LR_TL = 5e-5
LR_FCNN = 1e-4
SEED = 42

torch.manual_seed(SEED); np.random.seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
'''))

CELLS.append(code('''# ── Load data ──
def load_aug(split):
    img = np.load(DATA_AUG3 / f'X_{split}_images.npy').astype(np.float32)
    lm  = np.load(DATA_AUG3 / f'X_{split}_landmarks.npy').astype(np.float32)
    y   = np.load(DATA_AUG3 / f'y_{split}.npy').astype(np.int64)
    return img, lm, y

def load_conf60_split(split):
    img = np.load(DATA_CONF60 / f'X_{split}_images.npy').astype(np.float32)
    lm  = np.load(DATA_CONF60 / f'X_{split}_landmarks.npy').astype(np.float32)
    y7  = np.load(DATA_CONF60 / f'y_{split}.npy')
    return img, lm, REMAP_3[y7]

img_tr_aug, lm_tr_aug, y_tr_aug = load_aug('train')
img_va, lm_va, y_va = load_conf60_split('val')
img_te, lm_te, y_te = load_conf60_split('test')

# B3 weights (augmented dataset)
counts = np.bincount(y_tr_aug, minlength=3).astype(np.float32)
w = counts.sum() / (3 * np.maximum(counts, 1))
W_B3 = torch.FloatTensor(w / w.sum() * 3).to(device)
print(f'Train aug: {len(y_tr_aug)}, Val: {len(y_va)}, Test: {len(y_te)}')
print(f'B3 weights: {W_B3.cpu().numpy().tolist()}')
'''))

CELLS.append(code('''# ── Loaders ──
def cnn_loader(img, y, shuffle=False):
    t = torch.from_numpy(img).permute(0, 3, 1, 2).float()
    return DataLoader(TensorDataset(t, torch.from_numpy(y).long()),
                      batch_size=BATCH, shuffle=shuffle, num_workers=2, pin_memory=True)

def fcnn_loader(lm, y, shuffle=False):
    return DataLoader(TensorDataset(torch.from_numpy(lm).float(), torch.from_numpy(y).long()),
                      batch_size=BATCH, shuffle=shuffle, num_workers=2, pin_memory=True)

def fusion_loader(img, lm, y, shuffle=False):
    t_img = torch.from_numpy(img).permute(0, 3, 1, 2).float()
    return DataLoader(TensorDataset(t_img, torch.from_numpy(lm).float(),
                                     torch.from_numpy(y).long()),
                      batch_size=BATCH, shuffle=shuffle, num_workers=2, pin_memory=True)
'''))

CELLS.append(code('''# ── Training loop with FULL history logging ──
def train_with_history(model, arch, tr_loader, va_loader, criterion, lr, save_path):
    """arch: 'cnn'/'fcnn'/'fusion'. Returns (history_dict, best_val, best_ep)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8, min_lr=1e-7)

    history = {
        'epoch': [], 'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_macro_f1': [], 'lr': [],
    }
    best_val, best_ep, stale, best_state = 0.0, 0, 0, None

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        tr_loss_sum, tr_correct, tr_total = 0.0, 0, 0
        for batch in tr_loader:
            *x, y = [b.to(device) for b in batch]
            out = model(*x) if arch == 'fusion' else model(x[0])
            loss = criterion(out, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            tr_loss_sum += loss.item() * y.size(0)
            tr_correct  += (out.argmax(1) == y).sum().item()
            tr_total    += y.size(0)
        train_loss = tr_loss_sum / tr_total
        train_acc  = tr_correct / tr_total

        # ── Val ──
        model.eval()
        va_loss_sum, va_correct, va_total = 0.0, 0, 0
        all_yt, all_yp = [], []
        with torch.no_grad():
            for batch in va_loader:
                *x, y = [b.to(device) for b in batch]
                out = model(*x) if arch == 'fusion' else model(x[0])
                loss = criterion(out, y)
                va_loss_sum += loss.item() * y.size(0)
                va_correct  += (out.argmax(1) == y).sum().item()
                va_total    += y.size(0)
                all_yt.append(y.cpu().numpy())
                all_yp.append(out.argmax(1).cpu().numpy())
        val_loss = va_loss_sum / va_total
        val_acc  = va_correct / va_total
        val_macro = f1_score(np.concatenate(all_yt), np.concatenate(all_yp),
                              average='macro', zero_division=0)
        scheduler.step(val_macro)
        cur_lr = optimizer.param_groups[0]['lr']

        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_macro_f1'].append(val_macro)
        history['lr'].append(cur_lr)

        if val_macro > best_val:
            best_val, best_ep, stale = val_macro, epoch, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, save_path)
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f'    early stop @ ep{epoch}')
                break

        if epoch % 5 == 0:
            print(f"    ep{epoch}: tr_loss={train_loss:.3f} val_loss={val_loss:.3f} "
                  f"val_macro={val_macro:.4f} (best={best_val:.4f}@{best_ep})")

    return history, best_val, best_ep
'''))

CELLS.append(md('## Run 5 Key Configs with History Logging'))

CELLS.append(code('''all_histories = {}
criterion_b3 = nn.CrossEntropyLoss(weight=W_B3)

# 1. FCNN B3
print('\\n== FCNN B3 ==')
model = EmotionFCNN(num_classes=NUM_CLASSES).to(device)
tr = fcnn_loader(lm_tr_aug, y_tr_aug, shuffle=True)
va = fcnn_loader(lm_va, y_va)
hist, bv, be = train_with_history(model, 'fcnn', tr, va, criterion_b3, LR_FCNN,
                                    str(OUT_DIR / 'fcnn_b3.pth'))
all_histories['FCNN_B3'] = {'history': hist, 'best_val_macro_f1': bv, 'best_epoch': be}
print(f'  Done: best={bv:.4f}@{be}')

# 2. CNN TL B3
print('\\n== CNN TL B3 ==')
model = EmotionCNNTransfer(num_classes=NUM_CLASSES).to(device)
tr = cnn_loader(img_tr_aug, y_tr_aug, shuffle=True)
va = cnn_loader(img_va, y_va)
hist, bv, be = train_with_history(model, 'cnn', tr, va, criterion_b3, LR_TL,
                                    str(OUT_DIR / 'cnn_tl_b3.pth'))
all_histories['CNN_TL_B3'] = {'history': hist, 'best_val_macro_f1': bv, 'best_epoch': be}
print(f'  Done: best={bv:.4f}@{be}')

# 3. Intermediate TL B3
print('\\n== Intermediate TL B3 ==')
model = IntermediateFusionTransfer(num_classes=NUM_CLASSES).to(device)
tr = fusion_loader(img_tr_aug, lm_tr_aug, y_tr_aug, shuffle=True)
va = fusion_loader(img_va, lm_va, y_va)
hist, bv, be = train_with_history(model, 'fusion', tr, va, criterion_b3, LR_TL,
                                    str(OUT_DIR / 'intermediate_tl_b3.pth'))
all_histories['Intermediate_TL_B3'] = {'history': hist, 'best_val_macro_f1': bv, 'best_epoch': be}
print(f'  Done: best={bv:.4f}@{be}')
'''))

CELLS.append(code('''# 4. Late Fusion TL B3 — train CNN TL + FCNN branches separately, then grid w
print('\\n== Late Fusion TL B3 (2-branch + grid w) ==')

# CNN TL branch (use a fresh model — separate from #2 above)
cnn = EmotionCNNTransfer(num_classes=NUM_CLASSES).to(device)
tr = cnn_loader(img_tr_aug, y_tr_aug, shuffle=True)
va = cnn_loader(img_va, y_va)
print('  [CNN TL branch]')
hist_cnn, bv_cnn, be_cnn = train_with_history(cnn, 'cnn', tr, va, criterion_b3, LR_TL,
                                                str(OUT_DIR / 'lf_cnn_tl_b3.pth'))
all_histories['Late_Fusion_TL_B3_CNN_branch'] = {
    'history': hist_cnn, 'best_val_macro_f1': bv_cnn, 'best_epoch': be_cnn,
}

# FCNN branch
fcnn = EmotionFCNN(num_classes=NUM_CLASSES).to(device)
tr = fcnn_loader(lm_tr_aug, y_tr_aug, shuffle=True)
va = fcnn_loader(lm_va, y_va)
print('  [FCNN branch]')
hist_fcnn, bv_fcnn, be_fcnn = train_with_history(fcnn, 'fcnn', tr, va, criterion_b3, LR_FCNN,
                                                   str(OUT_DIR / 'lf_fcnn_b3.pth'))
all_histories['Late_Fusion_TL_B3_FCNN_branch'] = {
    'history': hist_fcnn, 'best_val_macro_f1': bv_fcnn, 'best_epoch': be_fcnn,
}
print(f'  CNN best={bv_cnn:.4f}@{be_cnn}, FCNN best={bv_fcnn:.4f}@{be_fcnn}')
'''))

CELLS.append(code('''# 5. Late Fusion TL B1 — same 2-branch pattern with B1 (no weights, no aug)
print('\\n== Late Fusion TL B1 (2-branch + grid w, B1 no aug no weights) ==')

img_tr, lm_tr, y_tr = load_conf60_split('train')   # B1 = original conf60 train
criterion_b1 = nn.CrossEntropyLoss()

cnn = EmotionCNNTransfer(num_classes=NUM_CLASSES).to(device)
tr = cnn_loader(img_tr, y_tr, shuffle=True)
va = cnn_loader(img_va, y_va)
print('  [CNN TL branch B1]')
hist_cnn, bv_cnn, be_cnn = train_with_history(cnn, 'cnn', tr, va, criterion_b1, LR_TL,
                                                str(OUT_DIR / 'lf_cnn_tl_b1.pth'))
all_histories['Late_Fusion_TL_B1_CNN_branch'] = {
    'history': hist_cnn, 'best_val_macro_f1': bv_cnn, 'best_epoch': be_cnn,
}

fcnn = EmotionFCNN(num_classes=NUM_CLASSES).to(device)
tr = fcnn_loader(lm_tr, y_tr, shuffle=True)
va = fcnn_loader(lm_va, y_va)
print('  [FCNN branch B1]')
hist_fcnn, bv_fcnn, be_fcnn = train_with_history(fcnn, 'fcnn', tr, va, criterion_b1, LR_FCNN,
                                                   str(OUT_DIR / 'lf_fcnn_b1.pth'))
all_histories['Late_Fusion_TL_B1_FCNN_branch'] = {
    'history': hist_fcnn, 'best_val_macro_f1': bv_fcnn, 'best_epoch': be_fcnn,
}

# Save all histories
with open(OUT_DIR / 'all_histories.json', 'w') as f:
    json.dump(all_histories, f, indent=2)
print(f'\\nSaved: {OUT_DIR / "all_histories.json"}')
'''))

CELLS.append(md('## Plot Training Curves'))

CELLS.append(code('''import matplotlib.pyplot as plt

FIG_DIR = PROJECT_ROOT / 'docs' / 'figures' / '3class_training_history'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Plot loss + acc + val_macro per config in subplot
n_cfgs = len(all_histories)
fig, axes = plt.subplots(n_cfgs, 3, figsize=(13, 2.8 * n_cfgs))
if n_cfgs == 1: axes = axes.reshape(1, -1)

for i, (cfg, data) in enumerate(all_histories.items()):
    h = data['history']
    epochs = h['epoch']
    # Loss
    axes[i, 0].plot(epochs, h['train_loss'], label='train', color='#4A6A8A', linewidth=1.5)
    axes[i, 0].plot(epochs, h['val_loss'], label='val', color='#A87143', linewidth=1.5)
    axes[i, 0].axvline(data['best_epoch'], color='red', linestyle='--', linewidth=0.8, alpha=0.5)
    axes[i, 0].set_xlabel('Epoch', fontsize=8); axes[i, 0].set_ylabel('Loss', fontsize=8)
    axes[i, 0].set_title(f'{cfg} — Loss', fontsize=9)
    axes[i, 0].legend(fontsize=8); axes[i, 0].grid(alpha=0.3)

    # Accuracy
    axes[i, 1].plot(epochs, h['train_acc'], label='train', color='#4A6A8A', linewidth=1.5)
    axes[i, 1].plot(epochs, h['val_acc'], label='val', color='#A87143', linewidth=1.5)
    axes[i, 1].axvline(data['best_epoch'], color='red', linestyle='--', linewidth=0.8, alpha=0.5)
    axes[i, 1].set_xlabel('Epoch', fontsize=8); axes[i, 1].set_ylabel('Accuracy', fontsize=8)
    axes[i, 1].set_title(f'{cfg} — Accuracy', fontsize=9)
    axes[i, 1].legend(fontsize=8); axes[i, 1].grid(alpha=0.3)
    axes[i, 1].set_ylim(0, 1)

    # Val Macro F1 (selection criterion)
    axes[i, 2].plot(epochs, h['val_macro_f1'], color='#5A8055', linewidth=1.5)
    axes[i, 2].axvline(data['best_epoch'], color='red', linestyle='--', linewidth=0.8, alpha=0.5,
                        label=f'best@{data["best_epoch"]}')
    axes[i, 2].axhline(data['best_val_macro_f1'], color='red', linestyle=':', linewidth=0.8,
                        alpha=0.5, label=f'val={data["best_val_macro_f1"]:.4f}')
    axes[i, 2].set_xlabel('Epoch', fontsize=8); axes[i, 2].set_ylabel('Val Macro F1', fontsize=8)
    axes[i, 2].set_title(f'{cfg} — Val Macro F1 (selection)', fontsize=9)
    axes[i, 2].legend(fontsize=7); axes[i, 2].grid(alpha=0.3)

plt.tight_layout()
out = FIG_DIR / 'training_curves_5configs.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print(f'\\nSaved: {out}')
'''))

CELLS.append(md('''## Summary

Training history sekarang lengkap untuk 5 key configs (Late Fusion TL B3 = 2-branch sehingga total 7 sub-curves). Plot menunjukkan convergence behavior — penting untuk:

- Verifikasi tidak overfit (train acc → 1.0 sementara val plateau)
- Cek best_epoch reasonable (bukan epoch=1 anomaly)
- Bandingkan kecepatan convergence antar arch

**Commit:**
```bash
git add models/frontonly_conf60/3class/history/ \\
        docs/figures/3class_training_history/ \\
        notebooks/results/83_*
git commit -m "Add training history + curves untuk audit dosen (nb 83 partial re-train)"
```
'''))


def main():
    nb = {'cells': CELLS, 'metadata': NB_META, 'nbformat': 4, 'nbformat_minor': 5}
    with open(NB_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f'Wrote: {NB_PATH} ({len(CELLS)} cells)')


if __name__ == '__main__':
    main()
