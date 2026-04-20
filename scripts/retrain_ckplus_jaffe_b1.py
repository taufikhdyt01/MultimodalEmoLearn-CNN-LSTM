"""
Retrain CNN_B1 + FCNN_B1 untuk CK+/JAFFE (Skema 1 benchmark).

Checkpoint tsb terhapus saat VPS cleanup. Perlu di-train ulang supaya
rerun_late_fusion_proper.py bisa compute proper val-tuned Late Fusion
results untuk CK+/JAFFE Skema 1.

Scope: 4 combos × 2 models = 8 trainings total
  - CK+ 7c: CNN_B1, FCNN_B1
  - CK+ 4c: CNN_B1, FCNN_B1  (uses ckplus_4class_contempt)
  - JAFFE 7c: CNN_B1, FCNN_B1
  - JAFFE 4c: CNN_B1, FCNN_B1

Estimasi: ~30-60 menit di T4 (CK+/JAFFE small datasets, cepat).

Usage (di VPS):
    conda activate emotrain
    python scripts/retrain_ckplus_jaffe_b1.py

Output checkpoint path (mengikuti konvensi nb 65):
    models/benchmark/ckplus/ckplus_7c/CNN_B1/model.pth
    models/benchmark/ckplus/ckplus_7c/FCNN_B1/model.pth
    models/benchmark/ckplus/ckplus_4c/CNN_B1/model.pth
    models/benchmark/ckplus/ckplus_4c/FCNN_B1/model.pth
    models/benchmark/jaffe/jaffe_7c/CNN_B1/model.pth
    models/benchmark/jaffe/jaffe_7c/FCNN_B1/model.pth
    models/benchmark/jaffe/jaffe_4c/CNN_B1/model.pth
    models/benchmark/jaffe/jaffe_4c/FCNN_B1/model.pth

Setelah training selesai, jalankan `rerun_late_fusion_proper.py` lagi
untuk dapat val-tuned Late Fusion numbers untuk CK+/JAFFE.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from training.models import EmotionCNN, EmotionFCNN  # noqa: E402
from training.utils import train_model  # noqa: E402

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

BENCHMARK_DIR = PROJECT_ROOT / 'data' / 'benchmark'
MODELS_DIR = PROJECT_ROOT / 'models' / 'benchmark'
BATCH_SIZE = 32
EPOCHS = 50
PATIENCE = 15
LR = 0.0001


def _subject_split(subjects, seed=42, train_ratio=0.8, val_ratio=0.1):
    """Match nb 65 / retrain_ckplus_jaffe convention."""
    rng = np.random.RandomState(seed)
    uniq = np.array(sorted(set(subjects.tolist())))
    rng.shuffle(uniq)
    n = len(uniq)
    n_tr = int(n * train_ratio)
    n_v = int(n * val_ratio)
    return (set(uniq[:n_tr].tolist()),
            set(uniq[n_tr:n_tr + n_v].tolist()),
            set(uniq[n_tr + n_v:].tolist()))


def _dataset_dir(dataset_name, num_classes):
    if dataset_name == 'ckplus' and num_classes == 4:
        return BENCHMARK_DIR / 'ckplus_4class_contempt'
    return BENCHMARK_DIR / f'{dataset_name}_{num_classes}class'


def load_data(dataset_name, num_classes):
    """Load subject-wise split for CK+/JAFFE."""
    d = _dataset_dir(dataset_name, num_classes)
    X = np.load(d / 'X_images.npy')
    L = np.load(d / 'X_landmarks.npy')
    y = np.load(d / 'y_labels.npy')
    subjects = np.load(d / 'subjects.npy', allow_pickle=True)
    tr_subs, v_subs, te_subs = _subject_split(subjects)
    tr_idx = np.where(np.isin(subjects, list(tr_subs)))[0]
    v_idx = np.where(np.isin(subjects, list(v_subs)))[0]
    te_idx = np.where(np.isin(subjects, list(te_subs)))[0]
    return (X[tr_idx], L[tr_idx], y[tr_idx],
            X[v_idx], L[v_idx], y[v_idx],
            X[te_idx], L[te_idx], y[te_idx])


def make_cnn_loader(img, y, shuffle=True):
    t = torch.from_numpy(img).permute(0, 3, 1, 2).contiguous()
    return DataLoader(TensorDataset(t, torch.from_numpy(y).long()),
                      batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=0, pin_memory=True)


def make_fcnn_loader(lm, y, shuffle=True):
    return DataLoader(
        TensorDataset(torch.from_numpy(lm).float(), torch.from_numpy(y).long()),
        batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0, pin_memory=True)


def checkpoint_path(dataset_name, num_classes, model_key):
    return (MODELS_DIR / dataset_name / f'{dataset_name}_{num_classes}c' /
            model_key / 'model.pth')


def train_one(dataset_name, num_classes, model_key, model_class, model_type,
              tr_x, tr_y, v_x, v_y):
    """Train single model, save checkpoint."""
    save_path = checkpoint_path(dataset_name, num_classes, model_key)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if save_path.exists():
        print(f'    [SKIP] {model_key} already exists at {save_path}')
        return

    if model_type == 'cnn':
        tr_loader = make_cnn_loader(tr_x, tr_y, shuffle=True)
        v_loader = make_cnn_loader(v_x, v_y, shuffle=False)
    else:
        tr_loader = make_fcnn_loader(tr_x, tr_y, shuffle=True)
        v_loader = make_fcnn_loader(v_x, v_y, shuffle=False)

    model = model_class(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8, min_lr=1e-7)

    print(f'    [TRAIN] {model_key} → {save_path.relative_to(PROJECT_ROOT)}')
    train_model(model, tr_loader, v_loader, criterion, optimizer, scheduler,
                device, model_type, EPOCHS, PATIENCE, str(save_path))


def retrain_dataset(dataset_name, num_classes):
    print(f"\n{'='*70}\n  {dataset_name.upper()} {num_classes}c  B1 (baseline)\n{'='*70}")

    tr_img, tr_lm, tr_y, v_img, v_lm, v_y, _, _, _ = load_data(dataset_name, num_classes)
    print(f'  Train: {len(tr_y)}  Val: {len(v_y)}')

    # CNN (image only)
    train_one(dataset_name, num_classes, 'CNN_B1',
              EmotionCNN, 'cnn', tr_img, tr_y, v_img, v_y)

    # FCNN (landmark only)
    train_one(dataset_name, num_classes, 'FCNN_B1',
              EmotionFCNN, 'fcnn', tr_lm, tr_y, v_lm, v_y)


def main():
    for ds in ('ckplus', 'jaffe'):
        for nc in (7, 4):
            retrain_dataset(ds, nc)

    print('\n' + '=' * 70)
    print('Done. Next step:')
    print('  python scripts/rerun_late_fusion_proper.py')
    print('  git add models/benchmark/ckplus/ models/benchmark/jaffe/')
    print('  git commit -m "Retrain CK+/JAFFE CNN/FCNN B1 + val-tuned Late Fusion"')
    print('  git push')


if __name__ == '__main__':
    main()
