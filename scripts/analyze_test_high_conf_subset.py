"""
Post-hoc analysis: evaluate Late Fusion TL B3 (3-class juara val-tuned) di subset
test set dengan confidence threshold berbeda.

Tujuan: argument paper bahwa gap performance largely karena label noise di
low-conf samples, bukan model limitation.

Workflow (no retrain):
  1. Load CNN TL + FCNN B3 checkpoints (both already trained at nb 79 / nb 82)
  2. Inference di full test set → softmax both branches
  3. Apply w_best=0.15 → fused predictions
  4. Compute Face API confidence per sample (max y_test_soft.npy 7-dim)
  5. Filter test indices per threshold {0.60, 0.70, 0.80, 0.90, 0.95, 0.99}
  6. Re-compute metrics di each subset

Output:
  models/frontonly_conf60/3class/test_subset_by_confidence.json
  docs/figures/test_macro_f1_by_confidence.png
  docs/test_subset_analysis.md (paper-ready)

Run di VPS (butuh checkpoint .pth):
    python scripts/analyze_test_high_conf_subset.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score)
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from training.models import EmotionCNNTransfer, EmotionFCNN  # noqa: E402

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_DIR = PROJECT_ROOT / 'data' / 'dataset_frontonly_conf60'
CKPT_DIR = PROJECT_ROOT / 'models' / 'frontonly_conf60' / '3class' / 'Late_Fusion_TL'
OUT_DIR  = PROJECT_ROOT / 'models' / 'frontonly_conf60' / '3class'
FIG_DIR  = PROJECT_ROOT / 'docs' / 'figures'
MD_OUT   = PROJECT_ROOT / 'docs' / 'test_subset_analysis.md'

NUM_CLASSES = 3
EMOTIONS_3 = ['positive', 'neutral', 'negative']
REMAP_3 = np.array([1, 0, 2, 2, 2, 2, 0], dtype=np.int64)
W_BEST = 0.15  # val-tuned dari nb 79 Late Fusion TL B3

THRESHOLDS = [0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
BATCH = 64


def softmax_from_loader(model, loader, arch):
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in loader:
            x, _ = batch
            x = x.to(device)
            out = model(x)
            probs.append(F.softmax(out, dim=1).cpu().numpy())
    return np.concatenate(probs)


def metrics_subset(y_true, y_pred, mask, n_classes=3):
    yt = y_true[mask]; yp = y_pred[mask]
    if len(yt) == 0:
        return None
    return {
        'n_samples':        int(mask.sum()),
        'macro_f1':         float(f1_score(yt, yp, average='macro', zero_division=0)),
        'micro_f1':         float(f1_score(yt, yp, average='micro', zero_division=0)),
        'weighted_f1':      float(f1_score(yt, yp, average='weighted', zero_division=0)),
        'accuracy':         float(accuracy_score(yt, yp)),
        'confusion_matrix': confusion_matrix(yt, yp, labels=list(range(n_classes))).tolist(),
        'classification_report': classification_report(
            yt, yp, target_names=EMOTIONS_3, labels=list(range(n_classes)),
            zero_division=0, output_dict=True),
        'class_distribution': np.bincount(yt, minlength=n_classes).tolist(),
    }


def main():
    print(f'Device: {device}')

    # ── Load test data ──
    img = np.load(DATA_DIR / 'X_test_images.npy').astype(np.float32)
    lm  = np.load(DATA_DIR / 'X_test_landmarks.npy').astype(np.float32)
    y7  = np.load(DATA_DIR / 'y_test.npy')
    y3  = REMAP_3[y7]

    # Confidence dari Face API soft labels (7-dim, max = original Face API confidence)
    y_soft7 = np.load(DATA_DIR / 'y_test_soft.npy')
    confs = y_soft7.max(axis=1)
    print(f'Test: {len(y3)} samples, conf range [{confs.min():.3f}, {confs.max():.3f}], '
          f'mean={confs.mean():.3f}')

    # ── Load checkpoints with fallback ──
    cnn_path  = CKPT_DIR / 'cnn_tl_b3.pth'
    fcnn_path = CKPT_DIR / 'fcnn_b3.pth'

    use_fallback = False
    if not cnn_path.exists() or not fcnn_path.exists():
        # Fallback: try single-modal checkpoints (different folders, dari nb 79 ARCH_REGISTRY loop)
        fallback_cnn  = OUT_DIR / 'CNN_TL' / 'cnn_tl_b3.pth'
        fallback_fcnn = OUT_DIR / 'FCNN'   / 'fcnn_b3.pth'

        print('[WARN] Late Fusion TL B3 dedicated checkpoints missing:')
        print(f'  {cnn_path}  exists={cnn_path.exists()}')
        print(f'  {fcnn_path}  exists={fcnn_path.exists()}')
        print(f'\nTrying fallback to single-modal checkpoints (different training trajectory):')
        print(f'  {fallback_cnn}  exists={fallback_cnn.exists()}')
        print(f'  {fallback_fcnn}  exists={fallback_fcnn.exists()}')

        if fallback_cnn.exists() and fallback_fcnn.exists():
            cnn_path = fallback_cnn
            fcnn_path = fallback_fcnn
            use_fallback = True
            print('\n[INFO] Using fallback single-modal checkpoints. Will re-grid-search w on val.')
        else:
            raise FileNotFoundError(
                'Both Late Fusion TL and single-modal CNN_TL/FCNN B3 checkpoints missing.\n'
                'Available alternatives:\n'
                '  1. Re-run nb 79 di VPS (~7-10 jam) untuk regenerate semua checkpoint\n'
                '  2. Re-run hanya Late Fusion TL B3 config (~60-90 min) — perlu modify nb 79\n'
                '  3. Use any other arch checkpoint yang tersedia (modify script)\n\n'
                f'Cek path:\n'
                f'  ls -la {OUT_DIR}/*/cnn*.pth {OUT_DIR}/*/fcnn*.pth')

    print(f'\nLoading: CNN={cnn_path.name}, FCNN={fcnn_path.name}')
    if use_fallback:
        print('  [FALLBACK MODE] approximate analysis with single-modal checkpoints')

    cnn = EmotionCNNTransfer(num_classes=NUM_CLASSES).to(device)
    cnn.load_state_dict(torch.load(cnn_path, map_location=device, weights_only=True))
    cnn.eval()

    fcnn = EmotionFCNN(num_classes=NUM_CLASSES).to(device)
    fcnn.load_state_dict(torch.load(fcnn_path, map_location=device, weights_only=True))
    fcnn.eval()

    # ── Build loaders ──
    img_t = torch.from_numpy(img).permute(0, 3, 1, 2).float()
    lm_t  = torch.from_numpy(lm).float()
    y_t   = torch.from_numpy(y3).long()
    cnn_loader  = DataLoader(TensorDataset(img_t, y_t), batch_size=BATCH)
    fcnn_loader = DataLoader(TensorDataset(lm_t,  y_t), batch_size=BATCH)

    # ── Inference both branches ──
    p_cnn  = softmax_from_loader(cnn,  cnn_loader,  'cnn')
    p_fcnn = softmax_from_loader(fcnn, fcnn_loader, 'fcnn')

    # ── Fused prediction with val-tuned w ──
    if use_fallback:
        # Re-grid-search w on val set (since single-modal trained differently)
        from sklearn.metrics import f1_score as _f1
        val_img = np.load(DATA_DIR / 'X_val_images.npy').astype(np.float32)
        val_lm  = np.load(DATA_DIR / 'X_val_landmarks.npy').astype(np.float32)
        val_y   = REMAP_3[np.load(DATA_DIR / 'y_val.npy')]
        val_img_t = torch.from_numpy(val_img).permute(0, 3, 1, 2).float()
        val_lm_t  = torch.from_numpy(val_lm).float()
        val_y_t   = torch.from_numpy(val_y).long()
        val_cnn_loader  = DataLoader(TensorDataset(val_img_t, val_y_t), batch_size=BATCH)
        val_fcnn_loader = DataLoader(TensorDataset(val_lm_t,  val_y_t), batch_size=BATCH)
        p_val_cnn  = softmax_from_loader(cnn,  val_cnn_loader,  'cnn')
        p_val_fcnn = softmax_from_loader(fcnn, val_fcnn_loader, 'fcnn')
        best_w, best_vf1 = 0.5, 0.0
        for w in np.arange(0.0, 1.05, 0.05):
            fused_val = w * p_val_cnn + (1.0 - w) * p_val_fcnn
            vf = _f1(val_y, fused_val.argmax(1), average='macro', zero_division=0)
            if vf > best_vf1: best_vf1, best_w = vf, float(w)
        w_used = best_w
        print(f'  Fallback grid-search: w_best = {w_used:.2f}  val_macro = {best_vf1:.4f}')
    else:
        w_used = W_BEST  # 0.15 dari nb 79 Late Fusion TL B3
        print(f'  Using val-tuned w = {w_used:.2f} (dari nb 79 Late Fusion TL B3)')

    fused = w_used * p_cnn + (1.0 - w_used) * p_fcnn
    y_pred = fused.argmax(1)

    # ── Per threshold metrics ──
    results = {
        'model':           'Late Fusion TL B3 (3-class, val-tuned)' + (' [FALLBACK single-modal]' if use_fallback else ''),
        'w_best':          float(w_used),
        'fallback_mode':   bool(use_fallback),
        'cnn_checkpoint':  str(cnn_path.relative_to(PROJECT_ROOT)),
        'fcnn_checkpoint': str(fcnn_path.relative_to(PROJECT_ROOT)),
        'n_total':         int(len(y3)),
        'thresholds':      [],
    }

    print(f"\n{'='*82}")
    print(f"  Test Subset Analysis by Face API Confidence")
    print(f"{'='*82}")
    print(f"  {'Threshold':<11} {'N':>6} {'%':>5}  {'Macro F1':>9} {'Micro F1':>9} "
          f"{'W-F1':>8} {'Acc':>7}  per-class F1")

    for th in THRESHOLDS:
        mask = confs >= th
        m = metrics_subset(y3, y_pred, mask)
        if m is None:
            continue
        # Per-class F1
        cr = m['classification_report']
        pcs = [cr.get(c, {}).get('f1-score', 0.0) for c in EMOTIONS_3]
        m['threshold'] = th
        m['fraction_retained'] = m['n_samples'] / len(y3)
        results['thresholds'].append(m)

        print(f"  ≥ {th:.2f}      {m['n_samples']:>6} {m['fraction_retained']*100:>4.1f}%  "
              f"{m['macro_f1']:>9.4f} {m['micro_f1']:>9.4f} {m['weighted_f1']:>8.4f} "
              f"{m['accuracy']:>7.4f}  "
              f"pos={pcs[0]:.3f}/neu={pcs[1]:.3f}/neg={pcs[2]:.3f}")

    # ── Save JSON ──
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / 'test_subset_by_confidence.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: {out_json.relative_to(PROJECT_ROOT)}')

    # ── Save figure: Macro F1 vs threshold ──
    try:
        import matplotlib.pyplot as plt
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        ths = [m['threshold'] for m in results['thresholds']]
        macro = [m['macro_f1'] for m in results['thresholds']]
        acc = [m['accuracy'] for m in results['thresholds']]
        ns = [m['n_samples'] for m in results['thresholds']]

        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(ths, macro, 'o-', color='#4A6A8A', label='Macro F1', linewidth=2)
        ax1.plot(ths, acc, 's-', color='#5A8055', label='Accuracy', linewidth=2)
        ax1.set_xlabel('Face API confidence threshold', fontsize=10)
        ax1.set_ylabel('Test metric', fontsize=10)
        ax1.set_ylim(0, 1)
        ax1.grid(alpha=0.3)
        ax1.legend(loc='lower right', fontsize=9)

        # Right axis: N samples retained
        ax2 = ax1.twinx()
        ax2.bar(ths, ns, width=0.025, color='#A87143', alpha=0.3, label='N retained')
        ax2.set_ylabel('N samples', fontsize=10, color='#A87143')
        ax2.tick_params(axis='y', labelcolor='#A87143')

        plt.title('Test Performance vs Face API Confidence Threshold\n'
                  '(Late Fusion TL B3, 3-class, val-tuned w=0.15)', fontsize=10)
        plt.tight_layout()
        png_path = FIG_DIR / 'test_macro_f1_by_confidence.png'
        pdf_path = FIG_DIR / 'test_macro_f1_by_confidence.pdf'
        plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
        print(f'Saved figure: {png_path.relative_to(PROJECT_ROOT)}')
    except ImportError:
        print('matplotlib not available — skipping figure')

    # ── Save markdown summary ──
    md_lines = [
        '# Test Subset Analysis by Face API Confidence',
        '',
        '**Model:** Late Fusion TL B3 (3-class juara val-tuned, w_best = 0.15)',
        '**Tujuan:** evaluasi efek label noise — apakah model performance scale dengan confidence input?',
        '',
        '## Hasil',
        '',
        '| Conf Threshold | N | % Retained | Macro F1 | Acc | pos F1 | neu F1 | neg F1 |',
        '|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|',
    ]
    for m in results['thresholds']:
        cr = m['classification_report']
        pcs = [cr.get(c, {}).get('f1-score', 0.0) for c in EMOTIONS_3]
        md_lines.append(
            f"| ≥ {m['threshold']:.2f} | {m['n_samples']} | {m['fraction_retained']*100:.1f}% | "
            f"**{m['macro_f1']:.4f}** | {m['accuracy']:.4f} | "
            f"{pcs[0]:.3f} | {pcs[1]:.3f} | {pcs[2]:.3f} |"
        )
    md_lines += [
        '',
        '## Interpretasi',
        '',
        '- Pattern (kalau Macro F1 naik dengan threshold) menunjukkan label noise adalah faktor pembatas, bukan model limitation.',
        '- Argumen paper Discussion: model performance scale dengan label quality — gap dari Face API agreement (Cohen κ = 0.45) accountable for residual error pada full test set.',
        '- Trade-off ini justify retain conf60 untuk training (preserve sample sufficiency + minority class viability) tapi acknowledge label noise impact saat report metric.',
        '',
        '## Caveat',
        '',
        '- Subset sizes shrinking saat threshold naik → metric variance naik (semakin sedikit sample minority).',
        '- Per-class F1 negative class harus dilihat juga, bukan macro saja — at high threshold, negative count kecil.',
        '- Comparison vs expert agreement: di high-conf subset, expert κ = 0.86 (n=70 dari validation CSV) → di test full subset conf95, model performance harus lebih dekat ke this ceiling.',
    ]
    MD_OUT.write_text('\n'.join(md_lines), encoding='utf-8')
    print(f'Saved MD: {MD_OUT.relative_to(PROJECT_ROOT)}')


if __name__ == '__main__':
    main()
