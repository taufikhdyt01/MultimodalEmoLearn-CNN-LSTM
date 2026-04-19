"""
Export Predictions + Confusion Matrix from Best Late Fusion TL Models
=====================================================================
Jalan di VPS (punya checkpoint). Output JSON kecil yang berisi:
  - y_true (list int)
  - y_pred (list int)
  - best_cnn_weight (float)
  - confusion_matrix (list of list)
  - classification_report (per-class precision/recall/F1)
  - summary metrics (macro_f1, micro_f1, weighted_f1, accuracy)

Best configs (dari jitecs_paper_plan.md):
  - 7-class: Late Fusion TL B1 (Macro F1 = 0.301)
  - 4-class: Late Fusion TL B3 (Macro F1 = 0.567) — overall best

Usage (di VPS):
    python scripts/export_best_predictions.py

Output:
    models/frontonly_conf60/predictions/best_7c_late_fusion_tl_b1.json
    models/frontonly_conf60/predictions/best_4c_late_fusion_tl_b3.json
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

from training.models import EmotionCNNTransfer, EmotionFCNN  # noqa: E402

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

DATA_DIR = PROJECT_ROOT / 'data' / 'dataset_frontonly_conf60'
OUTPUT_DIR = PROJECT_ROOT / 'models' / 'frontonly_conf60' / 'predictions'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 64
REMAP_4 = np.array([0, 1, 2, 3, 3, 3, 3], dtype=np.int64)

EMOTIONS_7 = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised']
EMOTIONS_4 = ['neutral', 'happy', 'sad', 'negative']


def load_test_val():
    """Load Primer val + test (same for all training scenarios)."""
    v_img = np.load(DATA_DIR / 'X_val_images.npy')
    v_lm  = np.load(DATA_DIR / 'X_val_landmarks.npy')
    v_y7  = np.load(DATA_DIR / 'y_val.npy')
    t_img = np.load(DATA_DIR / 'X_test_images.npy')
    t_lm  = np.load(DATA_DIR / 'X_test_landmarks.npy')
    t_y7  = np.load(DATA_DIR / 'y_test.npy')
    return v_img, v_lm, v_y7, t_img, t_lm, t_y7


def make_cnn_loader(img, y):
    t = torch.from_numpy(img).permute(0, 3, 1, 2).contiguous()
    ys = torch.from_numpy(y).long()
    return DataLoader(TensorDataset(t, ys), batch_size=BATCH_SIZE,
                      shuffle=False, num_workers=0, pin_memory=True)


def make_fcnn_loader(lm, y):
    t = torch.from_numpy(lm).float()
    ys = torch.from_numpy(y).long()
    return DataLoader(TensorDataset(t, ys), batch_size=BATCH_SIZE,
                      shuffle=False, num_workers=0, pin_memory=True)


@torch.no_grad()
def batched_softmax(model, loader):
    model.eval()
    probs = []
    for xb, _ in loader:
        xb = xb.to(device)
        probs.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(probs, axis=0)


def grid_search_w(cnn_p_val, fcnn_p_val, y_val):
    best_f1, best_w = 0.0, 0.5
    for w in np.arange(0.0, 1.05, 0.05):
        pr = (w * cnn_p_val + (1 - w) * fcnn_p_val).argmax(axis=1)
        f = f1_score(y_val, pr, average='macro', zero_division=0)
        if f > best_f1:
            best_f1, best_w = float(f), float(w)
    return best_w, best_f1


def export(num_classes, scenario, emotions,
           cnn_ckpt, fcnn_ckpt, tag):
    print(f"\n{'='*70}\n  {tag}\n{'='*70}")
    if not cnn_ckpt.exists() or not fcnn_ckpt.exists():
        print(f'  [ERROR] checkpoint missing:')
        print(f'    cnn:  {cnn_ckpt}  exists={cnn_ckpt.exists()}')
        print(f'    fcnn: {fcnn_ckpt}  exists={fcnn_ckpt.exists()}')
        return None

    v_img, v_lm, v_y7, t_img, t_lm, t_y7 = load_test_val()
    if num_classes == 4:
        y_val = REMAP_4[v_y7]
        y_test = REMAP_4[t_y7]
    else:
        y_val = v_y7
        y_test = t_y7
    print(f'  Val: {len(y_val)}  Test: {len(y_test)}')

    cnn = EmotionCNNTransfer(num_classes=num_classes).to(device)
    cnn.load_state_dict(torch.load(cnn_ckpt, map_location=device, weights_only=True))
    fcnn = EmotionFCNN(num_classes=num_classes).to(device)
    fcnn.load_state_dict(torch.load(fcnn_ckpt, map_location=device, weights_only=True))

    # Inference
    vc_loader = make_cnn_loader(v_img, y_val)
    vf_loader = make_fcnn_loader(v_lm, y_val)
    tc_loader = make_cnn_loader(t_img, y_test)
    tf_loader = make_fcnn_loader(t_lm, y_test)

    vc = batched_softmax(cnn, vc_loader)
    vf = batched_softmax(fcnn, vf_loader)
    tc = batched_softmax(cnn, tc_loader)
    tf = batched_softmax(fcnn, tf_loader)

    # Grid search w pada val
    best_w, best_val_f1 = grid_search_w(vc, vf, y_val)
    print(f'  Best w(CNN_TL) = {best_w:.2f}  val Macro F1 = {best_val_f1:.4f}')

    # Evaluate pada test
    test_preds = (best_w * tc + (1 - best_w) * tf).argmax(axis=1)
    acc = accuracy_score(y_test, test_preds)
    macro_f1 = f1_score(y_test, test_preds, average='macro', zero_division=0)
    micro_f1 = f1_score(y_test, test_preds, average='micro', zero_division=0)
    weighted_f1 = f1_score(y_test, test_preds, average='weighted', zero_division=0)

    cm = confusion_matrix(y_test, test_preds, labels=list(range(num_classes)))
    report = classification_report(
        y_test, test_preds, labels=list(range(num_classes)),
        target_names=emotions, zero_division=0, output_dict=True)

    print(f'  Test Macro F1 = {macro_f1:.4f}  Acc = {acc:.4f}')
    print(f'\n  Confusion Matrix:')
    print(f'    {"":12s} ' + ' '.join(f'{e[:6]:>7s}' for e in emotions))
    for i, emo in enumerate(emotions):
        print(f'    {emo[:12]:12s} ' + ' '.join(f'{cm[i, j]:>7d}' for j in range(num_classes)))

    data = {
        'tag': tag,
        'num_classes': num_classes,
        'scenario': scenario,
        'emotions': emotions,
        'best_cnn_tl_weight': best_w,
        'val_macro_f1': float(best_val_f1),
        'test_metrics': {
            'accuracy': float(acc),
            'macro_f1': float(macro_f1),
            'micro_f1': float(micro_f1),
            'weighted_f1': float(weighted_f1),
        },
        'y_true': y_test.tolist(),
        'y_pred': test_preds.tolist(),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'checkpoints': {
            'cnn_tl': str(cnn_ckpt.relative_to(PROJECT_ROOT)),
            'fcnn': str(fcnn_ckpt.relative_to(PROJECT_ROOT)),
        },
    }
    out_path = OUTPUT_DIR / f'best_{num_classes}c_late_fusion_tl_{scenario}.json'
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'\n  Saved: {out_path}')
    return data


def main():
    # 7-class best: Late Fusion TL B1
    export(
        num_classes=7, scenario='b1', emotions=EMOTIONS_7,
        cnn_ckpt=PROJECT_ROOT / 'models' / 'frontonly_conf60' / '7class_tl' / 'cnn_tl_b1.pth',
        fcnn_ckpt=PROJECT_ROOT / 'models' / 'frontonly_conf60' / '7class' / 'fcnn_b1.pth',
        tag='Late Fusion TL 7c B1 (best 7-class)',
    )

    # 4-class best: Late Fusion TL B3
    export(
        num_classes=4, scenario='b3', emotions=EMOTIONS_4,
        cnn_ckpt=PROJECT_ROOT / 'models' / 'frontonly_conf60' / '4class_tl' / 'cnn_tl_b3.pth',
        fcnn_ckpt=PROJECT_ROOT / 'models' / 'frontonly_conf60' / '4class' / 'fcnn_b3.pth',
        tag='Late Fusion TL 4c B3 (best overall)',
    )

    print('\nDone. Commit files:')
    print('  git add models/frontonly_conf60/predictions/')
    print('  git commit -m "Export predictions from best Late Fusion TL models"')


if __name__ == '__main__':
    main()
