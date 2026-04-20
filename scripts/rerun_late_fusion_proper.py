"""
Re-evaluate Late Fusion configs with proper val-tuned w.

Scope (Opsi B — comprehensive):
  (A) Primer conf60: 12 configs (scratch × TL × B1/B2/B3 × 7c/4c)
      Source notebooks: nb 45/49 (scratch) + nb 52/55 (TL) → ALL had test-set leakage
  (B) Benchmark Skema 1 CK+/JAFFE Late Fusion (non-TL): 4 configs
      (CK+ 7c/4c, JAFFE 7c/4c, B1 baseline)
      Source notebooks: nb 36/37 → test-set leakage
      (RAF-DB, KDEF already used val-tuned — no fix needed)
      (Late Fusion TL Skema 1 from nb 65 already val-tuned — no fix needed)

Total: 16 configs re-evaluated with both val-tuned (proper, default)
+ test-tuned (paper-compat) numbers.

Original nb grid-searched fusion weight w on TEST set — leakage that
inflates Macro F1 by ~0.06-0.10. Script ini re-compute proper val-tuned
results and save both to JSON.

Jalan di VPS (butuh checkpoint). Output:
  Primer conf60 (4 JSON files):
    models/frontonly_conf60/4class/late_fusion_results.json         (updated)
    models/frontonly_conf60/4class_tl/late_fusion_tl_results.json   (updated)
    models/frontonly_conf60/7class/late_fusion_results.json         (updated)
    models/frontonly_conf60/7class_tl/late_fusion_tl_results.json   (updated)
  Benchmark (2 JSON files, selected keys only):
    models/benchmark/ckplus/ckplus_7c_results.json  (update Late_Fusion_B1)
    models/benchmark/ckplus/ckplus_4c_results.json  (update Late_Fusion_B1)
    models/benchmark/jaffe/jaffe_7c_results.json    (update Late_Fusion_B1)
    models/benchmark/jaffe/jaffe_4c_results.json    (update Late_Fusion_B1)

Setiap JSON entry punya struktur:
  "B1 Baseline": {
      "accuracy": ..., "macro_f1": ..., "weighted_f1": ...,    # val-tuned (proper)
      "best_cnn_weight": ...,
      "test_tuned": {                                          # paper-compat
          "accuracy": ..., "macro_f1": ..., "weighted_f1": ...,
          "best_cnn_weight": ...,
      }
  }

Usage:
    python scripts/rerun_late_fusion_proper.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from training.models import EmotionCNN, EmotionCNNTransfer, EmotionFCNN  # noqa: E402

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

DATA_DIR = PROJECT_ROOT / 'data' / 'dataset_frontonly_conf60'
MODELS_ROOT = PROJECT_ROOT / 'models' / 'frontonly_conf60'
BATCH_SIZE = 64
REMAP_4 = np.array([0, 1, 2, 3, 3, 3, 3], dtype=np.int64)


def load_val_test():
    v_img = np.load(DATA_DIR / 'X_val_images.npy')
    v_lm = np.load(DATA_DIR / 'X_val_landmarks.npy')
    v_y7 = np.load(DATA_DIR / 'y_val.npy')
    t_img = np.load(DATA_DIR / 'X_test_images.npy')
    t_lm = np.load(DATA_DIR / 'X_test_landmarks.npy')
    t_y7 = np.load(DATA_DIR / 'y_test.npy')
    return v_img, v_lm, v_y7, t_img, t_lm, t_y7


def make_cnn_loader(img, y):
    t = torch.from_numpy(img).permute(0, 3, 1, 2).contiguous()
    return DataLoader(TensorDataset(t, torch.from_numpy(y).long()),
                      batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)


def make_fcnn_loader(lm, y):
    return DataLoader(TensorDataset(torch.from_numpy(lm).float(), torch.from_numpy(y).long()),
                      batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)


@torch.no_grad()
def batched_softmax(model, loader):
    model.eval()
    probs = []
    for xb, _ in loader:
        xb = xb.to(device)
        probs.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(probs, axis=0)


def grid_search_w(c_probs, f_probs, y):
    """Return (best_w, best_macro_f1) from w grid 0.0-1.0 step 0.05."""
    best_f1, best_w = 0.0, 0.5
    for w in np.arange(0.0, 1.05, 0.05):
        preds = (w * c_probs + (1 - w) * f_probs).argmax(axis=1)
        f = f1_score(y, preds, average='macro', zero_division=0)
        if f > best_f1:
            best_f1, best_w = float(f), float(w)
    return best_w, best_f1


def eval_with_w(c_probs, f_probs, y, w):
    preds = (w * c_probs + (1 - w) * f_probs).argmax(axis=1)
    return {
        'accuracy': float(accuracy_score(y, preds)),
        'macro_f1': float(f1_score(y, preds, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y, preds, average='weighted', zero_division=0)),
        'best_cnn_weight': float(w),
    }


def eval_config(cnn_class, cnn_ckpt, fcnn_ckpt, num_classes, y_val, y_test,
                v_img, t_img, v_lm, t_lm):
    """Return dict with val-tuned + test-tuned results."""
    if not cnn_ckpt.exists():
        return {'error': f'cnn checkpoint missing: {cnn_ckpt}'}
    if not fcnn_ckpt.exists():
        return {'error': f'fcnn checkpoint missing: {fcnn_ckpt}'}

    cnn = cnn_class(num_classes=num_classes).to(device)
    cnn.load_state_dict(torch.load(cnn_ckpt, map_location=device, weights_only=True))
    fcnn = EmotionFCNN(num_classes=num_classes).to(device)
    fcnn.load_state_dict(torch.load(fcnn_ckpt, map_location=device, weights_only=True))

    vc = batched_softmax(cnn, make_cnn_loader(v_img, y_val))
    vf = batched_softmax(fcnn, make_fcnn_loader(v_lm, y_val))
    tc = batched_softmax(cnn, make_cnn_loader(t_img, y_test))
    tf = batched_softmax(fcnn, make_fcnn_loader(t_lm, y_test))

    # Val-tuned (proper)
    w_val, val_f1 = grid_search_w(vc, vf, y_val)
    r_val = eval_with_w(tc, tf, y_test, w_val)
    r_val['val_macro_f1'] = float(val_f1)

    # Test-tuned (paper-compat)
    w_test, test_f1 = grid_search_w(tc, tf, y_test)
    r_test = eval_with_w(tc, tf, y_test, w_test)

    return {**r_val, 'test_tuned': r_test}


def rerun_dataset(num_classes, is_tl):
    """Re-run all 3 scenarios (B1/B2/B3) for one dataset×backbone combination."""
    tag = f'{num_classes}-class {"TL" if is_tl else "scratch"}'
    print(f"\n{'='*70}\n  Re-evaluating Late Fusion {tag}\n{'='*70}")

    v_img, v_lm, v_y7, t_img, t_lm, t_y7 = load_val_test()
    if num_classes == 4:
        y_val, y_test = REMAP_4[v_y7], REMAP_4[t_y7]
    else:
        y_val, y_test = v_y7, t_y7

    # Checkpoint paths (match nb convention)
    cnn_dir_name = f'{num_classes}class_tl' if is_tl else f'{num_classes}class'
    fcnn_dir_name = f'{num_classes}class'  # FCNN selalu di dir non-tl
    cnn_prefix = 'cnn_tl' if is_tl else 'cnn'
    cnn_dir = MODELS_ROOT / cnn_dir_name
    fcnn_dir = MODELS_ROOT / fcnn_dir_name

    cnn_class = EmotionCNNTransfer if is_tl else EmotionCNN

    scenarios = [('b1', 'B1 Baseline'), ('b2', 'B2 Class Weights'), ('b3', 'B3 Weights+Aug')]
    results = {}

    for sc_key, sc_label in scenarios:
        print(f'\n  Scenario: {sc_label}')
        cnn_ckpt = cnn_dir / f'{cnn_prefix}_{sc_key}.pth'
        fcnn_ckpt = fcnn_dir / f'fcnn_{sc_key}.pth'
        res = eval_config(cnn_class, cnn_ckpt, fcnn_ckpt, num_classes,
                           y_val, y_test, v_img, t_img, v_lm, t_lm)
        if 'error' in res:
            print(f'    [SKIP] {res["error"]}')
            continue

        results[sc_label] = res
        print(f'    val-tuned:  w={res["best_cnn_weight"]:.2f}  '
              f'Macro F1={res["macro_f1"]:.4f}  Acc={res["accuracy"]:.4f}')
        t = res['test_tuned']
        print(f'    test-tuned: w={t["best_cnn_weight"]:.2f}  '
              f'Macro F1={t["macro_f1"]:.4f}  Acc={t["accuracy"]:.4f}')

    # Save to the appropriate JSON (same path as original nb 45/49/52/55)
    out_json_name = 'late_fusion_tl_results.json' if is_tl else 'late_fusion_results.json'
    out_path = cnn_dir / out_json_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  Saved: {out_path.relative_to(PROJECT_ROOT)}')
    return results


# ──────────────────────────────────────────────────────────────
# Benchmark CK+/JAFFE Skema 1 Late Fusion re-evaluation
# ──────────────────────────────────────────────────────────────

BENCHMARK_DIR = PROJECT_ROOT / 'data' / 'benchmark'
BENCHMARK_MODELS_DIR = PROJECT_ROOT / 'models' / 'benchmark'


def _subject_split(subjects, seed=42, train_ratio=0.8, val_ratio=0.1):
    """Match nb 36/37/65 convention."""
    rng = np.random.RandomState(seed)
    uniq = np.array(sorted(set(subjects.tolist())))
    rng.shuffle(uniq)
    n = len(uniq)
    n_tr = int(n * train_ratio)
    n_v = int(n * val_ratio)
    return (set(uniq[:n_tr].tolist()),
            set(uniq[n_tr:n_tr+n_v].tolist()),
            set(uniq[n_tr+n_v:].tolist()))


def _benchmark_dir(dataset_name, num_classes):
    """Data dir — match nb 65 convention (ckplus 4c uses contempt variant)."""
    if dataset_name == 'ckplus' and num_classes == 4:
        return BENCHMARK_DIR / 'ckplus_4class_contempt'
    return BENCHMARK_DIR / f'{dataset_name}_{num_classes}class'


def rerun_benchmark_late_fusion(dataset_name, num_classes):
    """Re-eval Late_Fusion_B1 for CK+/JAFFE benchmark Skema 1."""
    tag = f'Benchmark {dataset_name.upper()} {num_classes}c (Skema 1)'
    print(f"\n{'='*70}\n  {tag}\n{'='*70}")

    d = _benchmark_dir(dataset_name, num_classes)
    if not d.exists():
        print(f'  [SKIP] dataset dir missing: {d}')
        return None

    X = np.load(d / 'X_images.npy')
    L = np.load(d / 'X_landmarks.npy')
    y = np.load(d / 'y_labels.npy')
    subjects = np.load(d / 'subjects.npy', allow_pickle=True)

    tr_subs, v_subs, te_subs = _subject_split(subjects)
    v_idx = np.where(np.isin(subjects, list(v_subs)))[0]
    te_idx = np.where(np.isin(subjects, list(te_subs)))[0]

    v_img, v_lm, v_y = X[v_idx], L[v_idx], y[v_idx]
    te_img, te_lm, te_y = X[te_idx], L[te_idx], y[te_idx]
    print(f'  Val: {len(v_y)}  Test: {len(te_y)}')

    # Checkpoint path (nb 65 convention)
    ckpt_root = BENCHMARK_MODELS_DIR / dataset_name / f'{dataset_name}_{num_classes}c'
    cnn_ckpt = ckpt_root / 'CNN_B1' / 'model.pth'
    fcnn_ckpt = ckpt_root / 'FCNN_B1' / 'model.pth'

    if not cnn_ckpt.exists() or not fcnn_ckpt.exists():
        print(f'  [SKIP] checkpoint missing: cnn={cnn_ckpt.exists()}  fcnn={fcnn_ckpt.exists()}')
        return None

    cnn = EmotionCNN(num_classes=num_classes).to(device)
    cnn.load_state_dict(torch.load(cnn_ckpt, map_location=device, weights_only=True))
    fcnn = EmotionFCNN(num_classes=num_classes).to(device)
    fcnn.load_state_dict(torch.load(fcnn_ckpt, map_location=device, weights_only=True))

    vc = batched_softmax(cnn, make_cnn_loader(v_img, v_y))
    vf = batched_softmax(fcnn, make_fcnn_loader(v_lm, v_y))
    tc = batched_softmax(cnn, make_cnn_loader(te_img, te_y))
    tf = batched_softmax(fcnn, make_fcnn_loader(te_lm, te_y))

    w_val, val_f1 = grid_search_w(vc, vf, v_y)
    r_val = eval_with_w(tc, tf, te_y, w_val)
    r_val['val_macro_f1'] = float(val_f1)

    w_test, _ = grid_search_w(tc, tf, te_y)
    r_test = eval_with_w(tc, tf, te_y, w_test)

    print(f'  val-tuned:  w={w_val:.2f}  Macro F1={r_val["macro_f1"]:.4f}  Acc={r_val["accuracy"]:.4f}')
    print(f'  test-tuned: w={w_test:.2f}  Macro F1={r_test["macro_f1"]:.4f}  Acc={r_test["accuracy"]:.4f}')

    # Update existing benchmark JSON (keep other models' entries intact)
    results_file = BENCHMARK_MODELS_DIR / dataset_name / f'{dataset_name}_{num_classes}c_results.json'
    existing = {}
    if results_file.exists():
        with open(results_file) as f:
            existing = json.load(f)

    # Late_Fusion_B1 convention in benchmark JSON uses snake_case keys:
    # macro_f1, micro_f1, weighted_f1, accuracy, best_cnn_weight
    existing['Late_Fusion_B1'] = {
        'accuracy': r_val['accuracy'],
        'macro_f1': r_val['macro_f1'],
        'micro_f1': r_val['accuracy'],  # micro == acc for multi-class single-label
        'weighted_f1': r_val['weighted_f1'],
        'best_cnn_weight': r_val['best_cnn_weight'],
        'test_tuned': {
            'accuracy': r_test['accuracy'],
            'macro_f1': r_test['macro_f1'],
            'micro_f1': r_test['accuracy'],
            'weighted_f1': r_test['weighted_f1'],
            'best_cnn_weight': r_test['best_cnn_weight'],
        },
    }
    with open(results_file, 'w') as f:
        json.dump(existing, f, indent=2)
    print(f'  Updated: {results_file.relative_to(PROJECT_ROOT)}')
    return {'val_tuned': r_val, 'test_tuned': r_test}


def main():
    all_results = {}

    # ── (A) Primer conf60 — 12 configs ──
    print('\n' + '#' * 70)
    print('# (A) PRIMER CONF60 Late Fusion re-evaluation')
    print('#' * 70)
    all_results['primer_4c_tl'] = rerun_dataset(4, is_tl=True)
    all_results['primer_4c_scratch'] = rerun_dataset(4, is_tl=False)
    all_results['primer_7c_tl'] = rerun_dataset(7, is_tl=True)
    all_results['primer_7c_scratch'] = rerun_dataset(7, is_tl=False)

    # ── (B) Benchmark CK+/JAFFE Skema 1 — 4 configs ──
    print('\n' + '#' * 70)
    print('# (B) BENCHMARK CK+/JAFFE Skema 1 Late Fusion re-evaluation')
    print('#' * 70)
    benchmark_results = {}
    for ds in ('ckplus', 'jaffe'):
        for nc in (7, 4):
            key = f'{ds}_{nc}c'
            benchmark_results[key] = rerun_benchmark_late_fusion(ds, nc)

    # Summary comparison — Primer
    print(f"\n{'='*82}")
    print('  SUMMARY (A): Primer conf60 — val-tuned vs test-tuned Macro F1')
    print(f"{'='*82}")
    print(f"  {'Combo':<20} {'Scenario':<20} {'Val-tuned':>10} {'Test-tuned':>11} {'Δ':>8}")
    print(f"  {'-'*72}")
    for combo, res in all_results.items():
        if res is None:
            continue
        for sc, r in res.items():
            if 'error' in r:
                continue
            v = r['macro_f1']
            t = r['test_tuned']['macro_f1']
            print(f"  {combo:<20} {sc:<20} {v:>10.4f} {t:>11.4f} {t-v:>+8.4f}")

    # Summary comparison — Benchmark
    print(f"\n{'='*82}")
    print('  SUMMARY (B): Benchmark Skema 1 CK+/JAFFE — val-tuned vs test-tuned')
    print(f"{'='*82}")
    print(f"  {'Dataset':<15} {'Val-tuned':>10} {'Test-tuned':>11} {'Δ':>8}")
    print(f"  {'-'*50}")
    for key, res in benchmark_results.items():
        if res is None:
            continue
        v = res['val_tuned']['macro_f1']
        t = res['test_tuned']['macro_f1']
        print(f"  {key:<15} {v:>10.4f} {t:>11.4f} {t-v:>+8.4f}")

    print(f"\nCommit via:")
    print('  git add models/frontonly_conf60/*class*/late_fusion*.json \\')
    print('          models/benchmark/ckplus/ckplus_*c_results.json \\')
    print('          models/benchmark/jaffe/jaffe_*c_results.json')
    print('  git commit -m "Re-evaluate Late Fusion with val-tuned w (fix test-set leakage)"')
    print('  git push')


if __name__ == '__main__':
    main()
