"""
Export predictions + confusion matrix dari NEW best models (val-tuned).

Post val-tuning fix, best model bergeser:
  - 4c: Intermediate TL B3 = 0.521  (was Late Fusion TL B3 = 0.567 inflated)
  - 7c: Early Fusion TL B3 = 0.333  (was Late Fusion TL B1 = 0.301 inflated)

Script ini load Intermediate TL 4c B3 + Early Fusion TL 7c B3 checkpoint,
inference di Primer test set, save predictions + CM + classification report.

Jalan di VPS (butuh checkpoint .pth).

Usage:
    python scripts/export_new_best_predictions.py

Output:
    models/frontonly_conf60/predictions/best_4c_intermediate_tl_b3.json
    models/frontonly_conf60/predictions/best_7c_early_fusion_tl_b3.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              classification_report, f1_score)
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from training.models import (  # noqa: E402
    IntermediateFusionTransfer, EmotionEarlyFusionTransfer,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

DATA_DIR = PROJECT_ROOT / 'data' / 'dataset_frontonly_conf60'
OUTPUT_DIR = PROJECT_ROOT / 'models' / 'frontonly_conf60' / 'predictions'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 64
REMAP_4 = np.array([0, 1, 2, 3, 3, 3, 3], dtype=np.int64)

EMOTIONS_7 = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised']
EMOTIONS_4 = ['neutral', 'happy', 'sad', 'negative']


def load_test():
    t_img = np.load(DATA_DIR / 'X_test_images.npy')
    t_lm = np.load(DATA_DIR / 'X_test_landmarks.npy')
    t_y7 = np.load(DATA_DIR / 'y_test.npy')
    return t_img, t_lm, t_y7


def stack_4ch(img, heat):
    if heat.ndim == 3:
        heat = heat[..., None]
    return np.concatenate([img, heat], axis=-1).astype(np.float32, copy=False)


def make_fusion_loader(img, lm, y):
    """Dataloader yang return (image_tensor, landmark_tensor, label)."""
    img_t = torch.from_numpy(img).permute(0, 3, 1, 2).contiguous()
    lm_t = torch.from_numpy(lm).float()
    y_t = torch.from_numpy(y).long()
    ds = TensorDataset(img_t, lm_t, y_t)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=0, pin_memory=True)


def make_4ch_loader(x4, y):
    """Dataloader untuk Early Fusion (4-channel input)."""
    t = torch.from_numpy(x4).permute(0, 3, 1, 2).contiguous()
    y_t = torch.from_numpy(y).long()
    return DataLoader(TensorDataset(t, y_t), batch_size=BATCH_SIZE,
                      shuffle=False, num_workers=0, pin_memory=True)


def metrics_pack(y_true, y_pred, emotions, num_classes):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    report = classification_report(
        y_true, y_pred, labels=list(range(num_classes)),
        target_names=emotions, zero_division=0, output_dict=True)
    return acc, macro_f1, micro_f1, weighted_f1, cm, report


def export_intermediate_tl_4c_b3():
    """4-class best: Intermediate TL B3 (Macro F1 ≈ 0.521 val-tuned)."""
    tag = 'Intermediate TL 4c B3 (best overall, val-tuned)'
    print(f"\n{'='*70}\n  {tag}\n{'='*70}")

    ckpt = PROJECT_ROOT / 'models' / 'frontonly_conf60' / '4class_tl' / 'intermediate_tl_b3.pth'
    if not ckpt.exists():
        print(f'  [ERROR] checkpoint missing: {ckpt}')
        return None

    t_img, t_lm, t_y7 = load_test()
    y_test = REMAP_4[t_y7]
    print(f'  Test: {len(y_test)} samples')

    model = IntermediateFusionTransfer(num_classes=4).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    loader = make_fusion_loader(t_img, t_lm, y_test)
    preds = []
    with torch.no_grad():
        for img, lm, _ in loader:
            img = img.to(device, non_blocking=True)
            lm = lm.to(device, non_blocking=True)
            preds.append(model(img, lm).argmax(dim=1).cpu().numpy())
    y_pred = np.concatenate(preds)

    acc, macro_f1, micro_f1, wf1, cm, report = metrics_pack(
        y_test, y_pred, EMOTIONS_4, 4)

    print(f'  Test Macro F1 = {macro_f1:.4f}  Acc = {acc:.4f}')
    print(f'\n  Confusion Matrix:')
    print(f'    {"":12s} ' + ' '.join(f'{e[:6]:>7s}' for e in EMOTIONS_4))
    for i, emo in enumerate(EMOTIONS_4):
        print(f'    {emo[:12]:12s} ' + ' '.join(f'{cm[i, j]:>7d}' for j in range(4)))

    data = {
        'tag': tag,
        'num_classes': 4,
        'scenario': 'b3',
        'architecture': 'Intermediate Fusion TL',
        'emotions': EMOTIONS_4,
        'test_metrics': {
            'accuracy': float(acc),
            'macro_f1': float(macro_f1),
            'micro_f1': float(micro_f1),
            'weighted_f1': float(wf1),
        },
        'y_true': y_test.tolist(),
        'y_pred': y_pred.tolist(),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'checkpoint': str(ckpt.relative_to(PROJECT_ROOT)),
    }
    out = OUTPUT_DIR / 'best_4c_intermediate_tl_b3.json'
    with open(out, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'\n  Saved: {out.relative_to(PROJECT_ROOT)}')
    return data


def export_early_fusion_tl_7c_b3():
    """7-class best: Early Fusion TL B3 (Macro F1 ≈ 0.333 val-tuned)."""
    tag = 'Early Fusion TL 7c B3 (best 7-class)'
    print(f"\n{'='*70}\n  {tag}\n{'='*70}")

    ckpt = (PROJECT_ROOT / 'models' / 'frontonly_conf60' / 'early_fusion' /
            '7c' / 'EarlyFusion_TL_B3' / 'model.pth')
    if not ckpt.exists():
        print(f'  [ERROR] checkpoint missing: {ckpt}')
        return None

    t_img, t_lm, t_y7 = load_test()
    # Early Fusion needs heatmap — load pre-generated
    heat_path = DATA_DIR / 'X_test_heatmaps.npy'
    if not heat_path.exists():
        print(f'  [ERROR] heatmap missing: {heat_path}')
        return None
    t_heat = np.load(heat_path)
    t_x4 = stack_4ch(t_img, t_heat)
    y_test = t_y7  # 7-class
    print(f'  Test: {len(y_test)} samples, 4-channel shape: {t_x4.shape}')

    model = EmotionEarlyFusionTransfer(num_classes=7).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    loader = make_4ch_loader(t_x4, y_test)
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            preds.append(model(xb).argmax(dim=1).cpu().numpy())
    y_pred = np.concatenate(preds)

    acc, macro_f1, micro_f1, wf1, cm, report = metrics_pack(
        y_test, y_pred, EMOTIONS_7, 7)

    print(f'  Test Macro F1 = {macro_f1:.4f}  Acc = {acc:.4f}')
    print(f'\n  Confusion Matrix:')
    print(f'    {"":12s} ' + ' '.join(f'{e[:6]:>7s}' for e in EMOTIONS_7))
    for i, emo in enumerate(EMOTIONS_7):
        print(f'    {emo[:12]:12s} ' + ' '.join(f'{cm[i, j]:>7d}' for j in range(7)))

    data = {
        'tag': tag,
        'num_classes': 7,
        'scenario': 'b3',
        'architecture': 'Early Fusion TL',
        'emotions': EMOTIONS_7,
        'test_metrics': {
            'accuracy': float(acc),
            'macro_f1': float(macro_f1),
            'micro_f1': float(micro_f1),
            'weighted_f1': float(wf1),
        },
        'y_true': y_test.tolist(),
        'y_pred': y_pred.tolist(),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'checkpoint': str(ckpt.relative_to(PROJECT_ROOT)),
    }
    out = OUTPUT_DIR / 'best_7c_early_fusion_tl_b3.json'
    with open(out, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'\n  Saved: {out.relative_to(PROJECT_ROOT)}')
    return data


def main():
    export_intermediate_tl_4c_b3()
    export_early_fusion_tl_7c_b3()

    print('\nDone. Commit files:')
    print('  git add models/frontonly_conf60/predictions/best_4c_intermediate_tl_b3.json \\')
    print('          models/frontonly_conf60/predictions/best_7c_early_fusion_tl_b3.json')
    print('  git commit -m "Export new best predictions (val-tuned): Intermediate TL 4c, Early Fusion TL 7c"')


if __name__ == '__main__':
    main()
