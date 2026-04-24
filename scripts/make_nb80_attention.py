"""Generate nb 80 — Attention Module CBAM (arahan dosen #3, 3-class B1+B3)."""
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / 'notebooks' / '80_attention_cbam.ipynb'

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

CELLS.append(md('''# 80 — CBAM Attention Module (arahan dosen #3)

**Motivasi:** tambah attention mechanism ke image stream — target beat 3-class Late Fusion TL B3 juara val-tuned (Macro F1 = 0.623).

**CBAM** (Convolutional Block Attention Module, Woo et al. ECCV 2018):
- Channel Attention (focus channel penting) → Spatial Attention (focus region penting), sequential
- Overhead ringan: ~2% FLOPs, ~25K params per block
- Applied after each of 4 residual stages di ResNet-18

**Scope:**
- Class scheme: **3-class** (juara overall, Russell 1980)
- 2 variants × 2 scenarios = **4 configs**:
  - CNN_TL_CBAM (single-modal) × B1/B3
  - Late_Fusion_TL_CBAM (CNN branch dengan CBAM, FCNN branch plain) × B1/B3
- Hyperparam match nb 79 (EPOCHS=50, LR_TL=5e-5, BATCH=32, PATIENCE=15)
- Selection by val macro F1

**Baselines (dari nb 79):**
- CNN_TL plain B3 (3c): val=0.495, test=0.706
- Late Fusion TL B3 (3c): val=**0.623**, test=0.637 — juara val-based
- CNN_TL plain B1 (3c): val=0.493, test=0.634
- Late Fusion TL B1 (3c): val=0.609, test=0.653

**Estimasi:** ~4-5 jam di T4 (4 configs, each 30-60 min).

**Next step opsional:** kalau CBAM promising → extend ke Ghost Module + Triplet Attention.
'''))

CELLS.append(code('''import sys, os, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

PROJECT_ROOT = Path('..').resolve()
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from training.models import EmotionFCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

DATA_CONF60 = PROJECT_ROOT / 'data' / 'dataset_frontonly_conf60'
DATA_AUG3   = PROJECT_ROOT / 'data' / 'dataset_frontonly_conf60_3class_augmented'
OUT_DIR     = PROJECT_ROOT / 'models' / 'frontonly_conf60' / '3class' / 'CBAM'
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 3
EMOTIONS = ['positive', 'neutral', 'negative']
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

CELLS.append(md('## CBAM Module Implementation'))

CELLS.append(code('''# ── Channel Attention (CBAM Part 1) ──
class ChannelAttention(nn.Module):
    """Per-channel attention via avg-pool + max-pool → shared MLP → sigmoid."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        avg_out = self.mlp(self.avg_pool(x).view(b, c))
        max_out = self.mlp(self.max_pool(x).view(b, c))
        scale = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * scale


# ── Spatial Attention (CBAM Part 2) ──
class SpatialAttention(nn.Module):
    """Per-location attention via channel avg+max → 7×7 conv → sigmoid."""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out, _ = x.max(dim=1, keepdim=True)
        scale = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * scale


# ── Full CBAM Block ──
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention(spatial_kernel)

    def forward(self, x):
        return self.spatial(self.channel(x))
'''))

CELLS.append(md('## Model Variants'))

CELLS.append(code('''# ── Variant 1: CNN TL + CBAM (single-modal) ──
class EmotionCNNTransferCBAM(nn.Module):
    """ResNet-18 ImageNet pretrained + CBAM after each of 4 residual stages.

    CBAM inserted between layer1-4 (after each stage output). Classifier head
    match EmotionCNNTransfer (Flatten → 512→256 → num_classes).
    """
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = tv_models.resnet18(weights=weights)

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1; self.cbam1 = CBAM(64)
        self.layer2 = resnet.layer2; self.cbam2 = CBAM(128)
        self.layer3 = resnet.layer3; self.cbam3 = CBAM(256)
        self.layer4 = resnet.layer4; self.cbam4 = CBAM(512)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
        )
        self.head = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.cbam1(self.layer1(x))
        x = self.cbam2(self.layer2(x))
        x = self.cbam3(self.layer3(x))
        x = self.cbam4(self.layer4(x))
        x = self.avgpool(x)
        x = self.classifier(x)
        return self.head(x)

    def extract_features(self, x):
        x = self.stem(x)
        x = self.cbam1(self.layer1(x))
        x = self.cbam2(self.layer2(x))
        x = self.cbam3(self.layer3(x))
        x = self.cbam4(self.layer4(x))
        x = self.avgpool(x)
        return self.classifier(x)


# Quick sanity check
m = EmotionCNNTransferCBAM(num_classes=NUM_CLASSES)
n_params = sum(p.numel() for p in m.parameters())
print(f'CNN TL + CBAM: {n_params/1e6:.2f}M params (baseline ResNet-18 TL ~11.2M)')
print(f'CBAM overhead: ~{(n_params/1e6 - 11.2):.2f}M extra')
del m
'''))

CELLS.append(md('''## Data Loading (3-class B1 + B3)'''))

CELLS.append(code('''# ── Load data ──
def load_conf60(split):
    img = np.load(DATA_CONF60 / f'X_{split}_images.npy').astype(np.float32)
    lm  = np.load(DATA_CONF60 / f'X_{split}_landmarks.npy').astype(np.float32)
    y7  = np.load(DATA_CONF60 / f'y_{split}.npy')
    return img, lm, REMAP_3[y7]

def load_aug(split):
    img = np.load(DATA_AUG3 / f'X_{split}_images.npy').astype(np.float32)
    lm  = np.load(DATA_AUG3 / f'X_{split}_landmarks.npy').astype(np.float32)
    y   = np.load(DATA_AUG3 / f'y_{split}.npy').astype(np.int64)
    return img, lm, y

img_tr, lm_tr, y_tr = load_conf60('train')
img_va, lm_va, y_va = load_conf60('val')
img_te, lm_te, y_te = load_conf60('test')

if DATA_AUG3.exists():
    img_tr_aug, lm_tr_aug, y_tr_aug = load_aug('train')
    aug_available = True
else:
    print('[WARN] Augmented dataset missing — B3 scenarios will be skipped')
    aug_available = False

print(f'Conf60 train: {len(y_tr)}  val: {len(y_va)}  test: {len(y_te)}')
if aug_available:
    print(f'Augmented train: {len(y_tr_aug)}  dist: {np.bincount(y_tr_aug, minlength=3).tolist()}')
'''))

CELLS.append(code('''# ── Class weights ──
def class_weights(y, num_classes=3):
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    w = counts.sum() / (num_classes * np.maximum(counts, 1))
    w_norm = w / w.sum() * num_classes
    return torch.FloatTensor(w_norm).to(device)

W_B3 = class_weights(y_tr_aug) if aug_available else None
print(f'B3 weights (augmented): {W_B3.cpu().numpy().tolist() if W_B3 is not None else None}')

def get_scenario_data(sc):
    if sc == 'B1':
        return (img_tr, lm_tr, y_tr), None
    if sc == 'B3':
        if not aug_available: return None, None
        return (img_tr_aug, lm_tr_aug, y_tr_aug), W_B3
    raise ValueError(sc)

val_data  = (img_va, lm_va, y_va)
test_data = (img_te, lm_te, y_te)
'''))

CELLS.append(code('''# ── Loaders + training helper ──
def build_loader(arch, data, shuffle=False):
    img, lm, y = data
    y_t = torch.from_numpy(y).long()
    if arch == 'cnn':
        t = torch.from_numpy(img).permute(0, 3, 1, 2).float()
        return DataLoader(TensorDataset(t, y_t), batch_size=BATCH, shuffle=shuffle,
                          num_workers=2, pin_memory=True)
    if arch == 'fcnn':
        return DataLoader(TensorDataset(torch.from_numpy(lm).float(), y_t),
                          batch_size=BATCH, shuffle=shuffle, num_workers=2, pin_memory=True)
    raise ValueError(arch)


def eval_arch(model, loader):
    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for x, y in loader:
            yt.append(y.numpy())
            yp.append(model(x.to(device)).argmax(1).cpu().numpy())
    return np.concatenate(yt), np.concatenate(yp)


def train_single(model, arch, train_data, criterion, lr, save_path):
    tr_loader = build_loader(arch, train_data, shuffle=True)
    va_loader = build_loader(arch, val_data)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8, min_lr=1e-7)

    best_val, best_ep, stale, best_state = 0.0, 0, 0, None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        yt, yp = eval_arch(model, va_loader)
        vf1 = f1_score(yt, yp, average='macro', zero_division=0)
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
        if epoch % 5 == 0:
            print(f'    ep{epoch}: val={vf1:.4f} best={best_val:.4f}@{best_ep}')
    return best_val, best_ep, best_state


def eval_test(model, arch):
    te_loader = build_loader(arch, test_data)
    yt, yp = eval_arch(model, te_loader)
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

CELLS.append(md('## Run: CNN TL + CBAM × B1/B3'))

CELLS.append(code('''results = {}

for sc in ['B1', 'B3']:
    train_data, weights = get_scenario_data(sc)
    if train_data is None:
        print(f'\\n[SKIP] CNN_TL_CBAM_{sc} — augmented data missing')
        continue
    cfg = f'CNN_TL_CBAM_{sc}'
    print(f"\\n{'='*70}\\n  {cfg}\\n{'='*70}")

    model = EmotionCNNTransferCBAM(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    save_path = OUT_DIR / f'cnn_tl_cbam_{sc.lower()}.pth'

    best_val, best_ep, _ = train_single(model, 'cnn', train_data, criterion, LR_TL, str(save_path))
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    metrics = eval_test(model, 'cnn')
    metrics['val_macro_f1'] = float(best_val)
    metrics['best_epoch'] = int(best_ep)

    results[cfg] = metrics
    print(f"  val={best_val:.4f}@ep{best_ep}  test_macro={metrics['test_macro_f1']:.4f}  acc={metrics['test_accuracy']:.4f}")
'''))

CELLS.append(md('''## Run: Late Fusion TL + CBAM × B1/B3

Train CNN_TL_CBAM branch (above) + FCNN branch terpisah, lalu grid-search `w` di val.'''))

CELLS.append(code('''def softmax_from_loader(model, loader):
    model.eval()
    probs = []
    with torch.no_grad():
        for x, _ in loader:
            probs.append(F.softmax(model(x.to(device)), dim=1).cpu().numpy())
    return np.concatenate(probs)


def search_best_w(p_cnn, p_fcnn, y_val):
    best_f1, best_w = 0.0, 0.5
    for w in np.arange(0.0, 1.05, 0.05):
        fused = w * p_cnn + (1.0 - w) * p_fcnn
        f1 = f1_score(y_val, fused.argmax(1), average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1, best_w = f1, float(w)
    return best_w, float(best_f1)


LF_OUT = OUT_DIR / 'Late_Fusion_CBAM'
LF_OUT.mkdir(parents=True, exist_ok=True)

for sc in ['B1', 'B3']:
    train_data, weights = get_scenario_data(sc)
    if train_data is None:
        continue
    cfg = f'Late_Fusion_TL_CBAM_{sc}'
    print(f"\\n{'='*70}\\n  {cfg} (2-branch train + grid w)\\n{'='*70}")

    criterion = nn.CrossEntropyLoss(weight=weights)

    # CNN_TL_CBAM branch — REUSE checkpoint dari CNN_TL_CBAM_{sc}
    cnn_ckpt_reuse = OUT_DIR / f'cnn_tl_cbam_{sc.lower()}.pth'
    cnn = EmotionCNNTransferCBAM(num_classes=NUM_CLASSES).to(device)
    if cnn_ckpt_reuse.exists():
        print(f'  [CNN_TL_CBAM] reuse checkpoint {cnn_ckpt_reuse.name}')
        cnn.load_state_dict(torch.load(cnn_ckpt_reuse, map_location=device, weights_only=True))
    else:
        print(f'  [CNN_TL_CBAM] training from scratch...')
        _, _, _ = train_single(cnn, 'cnn', train_data, criterion, LR_TL, str(cnn_ckpt_reuse))

    # FCNN branch — train fresh (consistency dengan nb 79 Late Fusion logic)
    fcnn = EmotionFCNN(num_classes=NUM_CLASSES).to(device)
    fcnn_path = LF_OUT / f'fcnn_{sc.lower()}.pth'
    print(f'  [FCNN] training...')
    fcnn_val, fcnn_ep, _ = train_single(fcnn, 'fcnn', train_data, criterion, LR_FCNN, str(fcnn_path))
    fcnn.load_state_dict(torch.load(fcnn_path, map_location=device, weights_only=True))
    print(f'    FCNN val={fcnn_val:.4f}@ep{fcnn_ep}')

    # Softmax di val + test
    va_cnn = build_loader('cnn', val_data)
    va_fcnn = build_loader('fcnn', val_data)
    te_cnn = build_loader('cnn', test_data)
    te_fcnn = build_loader('fcnn', test_data)

    p_val_cnn  = softmax_from_loader(cnn, va_cnn)
    p_val_fcnn = softmax_from_loader(fcnn, va_fcnn)
    p_te_cnn   = softmax_from_loader(cnn, te_cnn)
    p_te_fcnn  = softmax_from_loader(fcnn, te_fcnn)

    # Grid-search w di val
    best_w, best_val_f1 = search_best_w(p_val_cnn, p_val_fcnn, y_va)
    print(f'  w_best (val-tuned) = {best_w:.2f}  val_macro = {best_val_f1:.4f}')

    # Apply di test
    fused_te = best_w * p_te_cnn + (1.0 - best_w) * p_te_fcnn
    pred_te = fused_te.argmax(1)
    test_macro = float(f1_score(y_te, pred_te, average='macro', zero_division=0))
    test_acc   = float(accuracy_score(y_te, pred_te))
    cm = confusion_matrix(y_te, pred_te, labels=list(range(NUM_CLASSES))).tolist()

    results[cfg] = {
        'val_macro_f1': best_val_f1,
        'best_cnn_weight': best_w,
        'cnn_cbam_val_macro_f1': float(results.get(f'CNN_TL_CBAM_{sc}', {}).get('val_macro_f1', 0)),
        'fcnn_val_macro_f1': float(fcnn_val),
        'test_macro_f1': test_macro,
        'test_accuracy': test_acc,
        'test_micro_f1': float(f1_score(y_te, pred_te, average='micro', zero_division=0)),
        'test_weighted_f1': float(f1_score(y_te, pred_te, average='weighted', zero_division=0)),
        'confusion_matrix': cm,
    }
    print(f'  TEST: macro={test_macro:.4f}  acc={test_acc:.4f}')
'''))

CELLS.append(md('## Summary vs Baselines (from nb 79)'))

CELLS.append(code('''# Save results
with open(OUT_DIR / 'cbam_3class_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'Saved: {OUT_DIR / "cbam_3class_results.json"}')

# Comparison table
print(f"\\n{'='*85}")
print(f'  CBAM Attention Module — 3-class Primer conf60 (val-based selection)')
print(f"{'='*85}")
print(f"  {'Config':<32} {'Val F1':>8} {'Test F1':>8} {'Test Acc':>10} {'w_best':>8}")
print(f"  {'-'*80}")
for cfg in sorted(results.keys()):
    r = results[cfg]
    w = r.get('best_cnn_weight')
    w_s = f'{w:.2f}' if w is not None else '—'
    print(f"  {cfg:<32} {r['val_macro_f1']:>8.4f} {r['test_macro_f1']:>8.4f} "
          f"{r['test_accuracy']:>10.4f} {w_s:>8}")

print('\\n  Baselines (from nb 79 plain, val-based):')
print(f"  {'CNN_TL_B1 plain (nb 79)':<32} {'0.4927':>8} {'0.6340':>8} {'0.7912':>10} {'—':>8}")
print(f"  {'CNN_TL_B3 plain (nb 79)':<32} {'0.4953':>8} {'0.7055':>8} {'0.8396':>10} {'—':>8}")
print(f"  {'Late_Fusion_TL_B1 (nb 79)':<32} {'0.6093':>8} {'0.6526':>8} {'0.7966':>10} {'0.25':>8}")
print(f"  {'Late_Fusion_TL_B3 (nb 79 ⭐)':<32} {'0.6229':>8} {'0.6370':>8} {'0.7836':>10} {'0.15':>8}")
'''))

CELLS.append(md('''## Analysis & Interpretation

**Target beat:** Late Fusion TL B3 plain val = **0.6229** (3-class juara val-tuned overall).

**Expected outcomes:**

| Scenario | Interpretation |
|---|---|
| CNN_TL_CBAM val > 0.50 | Attention substantial boost vs plain CNN (0.493 / 0.495) — worth extending ke Ghost/Triplet |
| CNN_TL_CBAM val ≈ 0.49 | Attention tidak bantu CNN single — dataset Primer kecil + natural noise → attention learn spurious |
| Late_Fusion_CBAM val > 0.62 | CBAM image stream lift fusion — new SOTA 3-class |
| Late_Fusion_CBAM val ≈ 0.62 | CBAM tidak transfer ke fusion (mirror pattern soft label nb 72) — plain already optimal |

**Kalau hasil negative (semua < 0.50 val):** stop eksplorasi attention setelah CBAM, skip Ghost/Triplet. Document sebagai finding: attention module tidak menambah value di Primer conf60.

**Next step kalau promising:**
- Ghost Module (Han et al. CVPR 2020) — focus on efficient feature generation
- Triplet Attention (Misra et al. WACV 2021) — cross-dimension attention

Commit results:
```bash
git add models/frontonly_conf60/3class/CBAM/ notebooks/results/80_*
git commit -m "Add CBAM attention results (nb 80, arahan dosen #3)"
```
'''))


def main():
    nb = {'cells': CELLS, 'metadata': NB_META, 'nbformat': 4, 'nbformat_minor': 5}
    with open(NB_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f'Wrote: {NB_PATH} ({len(CELLS)} cells)')


if __name__ == '__main__':
    main()
