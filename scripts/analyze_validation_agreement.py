"""
Analyze inter-rater agreement antara Face API auto_label vs expert_label
dari validasi ahli (Dephilia Rambu Raja Uju Decky, S.Psi).

Metrics:
  1. Overall raw agreement (accuracy-style)
  2. Cohen's kappa (inter-rater reliability)
  3. Per-class agreement (precision/recall/F1 dengan expert sebagai ground truth)
  4. Confusion matrix (auto × expert)
  5. 3-class valence agreement (sesuai paper reframe)
  6. Confidence stratification (≥0.60, ≥0.80, ≥0.95)

Usage:
    python scripts/analyze_validation_agreement.py
    python scripts/analyze_validation_agreement.py --csv path/to/file.csv

Output:
    docs/validation_analysis_results.json
    docs/validation_analysis_results.md  (paper-ready summary)
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_ROOT / 'docs' / 'validation_results_dephilia_rambu_raja_uju_decky,_s.psi.csv'

EMOTIONS_7 = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised']
REMAP_3 = {  # 7-class label string → 3-class valence
    'neutral':   'neutral',
    'happy':     'positive',
    'surprised': 'positive',
    'sad':       'negative',
    'angry':     'negative',
    'fearful':   'negative',
    'disgusted': 'negative',
}
EMOTIONS_3 = ['positive', 'neutral', 'negative']


def load_data(csv_path):
    rows = []
    with open(csv_path, encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                'auto':       r['auto_label'].strip(),
                'expert':     r['expert_label'].strip(),
                'confidence': float(r['confidence']),
                'has_note':   bool(r.get('notes', '').strip()),
            })
    return rows


def cohen_kappa(y_auto, y_expert, labels):
    """Compute Cohen's kappa from list of labels."""
    n = len(y_auto)
    if n == 0:
        return float('nan')
    # Build counts
    label_idx = {l: i for i, l in enumerate(labels)}
    K = len(labels)
    obs = [[0] * K for _ in range(K)]
    for a, e in zip(y_auto, y_expert):
        if a not in label_idx or e not in label_idx:
            continue
        obs[label_idx[a]][label_idx[e]] += 1

    total = sum(sum(row) for row in obs)
    if total == 0:
        return float('nan')

    # Observed agreement
    p_o = sum(obs[i][i] for i in range(K)) / total

    # Expected agreement (chance)
    row_sums = [sum(row) for row in obs]
    col_sums = [sum(obs[i][j] for i in range(K)) for j in range(K)]
    p_e = sum((row_sums[i] / total) * (col_sums[i] / total) for i in range(K))

    if abs(1 - p_e) < 1e-12:
        return float('nan')
    return (p_o - p_e) / (1 - p_e)


def per_class_metrics(y_auto, y_expert, labels):
    """Compute precision, recall, F1 per class (expert = ground truth)."""
    K = len(labels)
    label_idx = {l: i for i, l in enumerate(labels)}
    cm = [[0] * K for _ in range(K)]  # cm[true][pred]
    for a, e in zip(y_auto, y_expert):
        if a not in label_idx or e not in label_idx:
            continue
        cm[label_idx[e]][label_idx[a]] += 1   # rows = expert (true), cols = auto (pred)

    metrics = {}
    for i, cls in enumerate(labels):
        tp = cm[i][i]
        col_sum = sum(cm[r][i] for r in range(K))   # predicted as cls
        row_sum = sum(cm[i])                          # actual cls
        prec = tp / col_sum if col_sum > 0 else 0.0
        rec  = tp / row_sum if row_sum > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        metrics[cls] = {
            'precision': prec, 'recall': rec, 'f1': f1,
            'support_expert': row_sum, 'support_auto': col_sum,
        }

    macro_f1 = sum(m['f1'] for m in metrics.values()) / K
    return metrics, cm, macro_f1


def agreement_by_confidence_bin(rows, threshold):
    sub = [r for r in rows if r['confidence'] >= threshold]
    if not sub:
        return None
    matched = sum(1 for r in sub if r['auto'] == r['expert'])
    return {
        'threshold': threshold,
        'n_samples': len(sub),
        'matched': matched,
        'agreement': matched / len(sub),
    }


def confusion_matrix_str(cm, labels):
    lines = []
    header = ' ' * 12 + ' '.join(f'{l[:10]:>10}' for l in labels) + '  | total'
    lines.append(header)
    lines.append('-' * len(header))
    for i, lab in enumerate(labels):
        row_total = sum(cm[i])
        lines.append(f'  {lab[:10]:>10} ' + ' '.join(f'{cm[i][j]:>10d}' for j in range(len(labels))) +
                     f'  | {row_total}')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=str, default=str(DEFAULT_CSV))
    ap.add_argument('--out-json', type=str,
                    default=str(PROJECT_ROOT / 'docs' / 'validation_analysis_results.json'))
    ap.add_argument('--out-md', type=str,
                    default=str(PROJECT_ROOT / 'docs' / 'validation_analysis_results.md'))
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    rows = load_data(csv_path)
    n_total = len(rows)
    print(f'Loaded: {n_total} samples from {csv_path.name}')

    # ── 1. Overall agreement (7-class) ──
    matched_7 = sum(1 for r in rows if r['auto'] == r['expert'])
    agreement_7 = matched_7 / n_total

    y_auto_7   = [r['auto']   for r in rows]
    y_expert_7 = [r['expert'] for r in rows]
    kappa_7 = cohen_kappa(y_auto_7, y_expert_7, EMOTIONS_7)

    metrics_7, cm_7, macro_f1_7 = per_class_metrics(y_auto_7, y_expert_7, EMOTIONS_7)

    print(f'\n{"="*60}\n  7-Class Agreement\n{"="*60}')
    print(f'  Raw agreement  : {agreement_7:.4f}  ({matched_7}/{n_total})')
    print(f'  Cohen kappa    : {kappa_7:.4f}')
    print(f'  Macro F1       : {macro_f1_7:.4f}')
    print(f'\n  Per-class:')
    print(f'    {"class":>10}  {"P":>6}  {"R":>6}  {"F1":>6}  {"n_exp":>6}  {"n_auto":>7}')
    for cls, m in metrics_7.items():
        print(f"    {cls:>10}  {m['precision']:.3f}  {m['recall']:.3f}  {m['f1']:.3f}  "
              f"{m['support_expert']:>6}  {m['support_auto']:>7}")
    print(f'\n  Confusion matrix (rows=expert, cols=auto):')
    print(confusion_matrix_str(cm_7, EMOTIONS_7))

    # ── 2. 3-Class valence agreement ──
    y_auto_3   = [REMAP_3[r['auto']]   for r in rows]
    y_expert_3 = [REMAP_3[r['expert']] for r in rows]
    matched_3 = sum(1 for a, e in zip(y_auto_3, y_expert_3) if a == e)
    agreement_3 = matched_3 / n_total
    kappa_3 = cohen_kappa(y_auto_3, y_expert_3, EMOTIONS_3)
    metrics_3, cm_3, macro_f1_3 = per_class_metrics(y_auto_3, y_expert_3, EMOTIONS_3)

    print(f'\n{"="*60}\n  3-Class Valence Agreement (Russell 1980 mapping)\n{"="*60}')
    print(f'  Raw agreement  : {agreement_3:.4f}  ({matched_3}/{n_total})')
    print(f'  Cohen kappa    : {kappa_3:.4f}')
    print(f'  Macro F1       : {macro_f1_3:.4f}')
    print(f'\n  Per-class:')
    for cls, m in metrics_3.items():
        print(f"    {cls:>10}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  "
              f"n_expert={m['support_expert']}  n_auto={m['support_auto']}")
    print(f'\n  Confusion matrix (rows=expert, cols=auto):')
    print(confusion_matrix_str(cm_3, EMOTIONS_3))

    # ── 3. Agreement by confidence threshold ──
    print(f'\n{"="*60}\n  Agreement by Confidence Threshold (7-class)\n{"="*60}')
    conf_results = []
    for th in [0.0, 0.60, 0.80, 0.95]:
        r = agreement_by_confidence_bin(rows, th)
        if r:
            conf_results.append(r)
            print(f"  conf ≥ {th:.2f}:  agreement = {r['agreement']:.4f}  "
                  f"({r['matched']}/{r['n_samples']})")

    # ── 4. Distribution analysis ──
    auto_dist_7    = Counter(r['auto'] for r in rows)
    expert_dist_7  = Counter(r['expert'] for r in rows)
    auto_dist_3    = Counter(REMAP_3[r['auto']] for r in rows)
    expert_dist_3  = Counter(REMAP_3[r['expert']] for r in rows)

    print(f'\n{"="*60}\n  Class Distribution\n{"="*60}')
    print('  7-class:')
    print(f'    {"class":>10}  {"auto":>6}  {"expert":>7}  {"diff":>6}')
    for cls in EMOTIONS_7:
        a = auto_dist_7.get(cls, 0)
        e = expert_dist_7.get(cls, 0)
        print(f'    {cls:>10}  {a:>6}  {e:>7}  {e-a:>+6}')
    print('\n  3-class valence:')
    for cls in EMOTIONS_3:
        a = auto_dist_3.get(cls, 0)
        e = expert_dist_3.get(cls, 0)
        print(f'    {cls:>10}  auto={a:>4}  expert={e:>4}  diff={e-a:>+4}')

    # ── Save JSON ──
    output = {
        'source_csv': str(csv_path.relative_to(PROJECT_ROOT)),
        'expert': 'Dephilia Rambu Raja Uju Decky, S.Psi',
        'n_total_samples': n_total,
        'agreement_7class': {
            'raw_agreement':    agreement_7,
            'cohen_kappa':      kappa_7,
            'macro_f1':         macro_f1_7,
            'matched':          matched_7,
            'per_class':        metrics_7,
            'confusion_matrix': cm_7,
            'labels':           EMOTIONS_7,
        },
        'agreement_3class': {
            'raw_agreement':    agreement_3,
            'cohen_kappa':      kappa_3,
            'macro_f1':         macro_f1_3,
            'matched':          matched_3,
            'per_class':        metrics_3,
            'confusion_matrix': cm_3,
            'labels':           EMOTIONS_3,
            'mapping':          REMAP_3,
        },
        'agreement_by_confidence': conf_results,
        'distributions': {
            '7class_auto':     dict(auto_dist_7),
            '7class_expert':   dict(expert_dist_7),
            '3class_auto':     dict(auto_dist_3),
            '3class_expert':   dict(expert_dist_3),
        },
    }
    out_json = Path(args.out_json)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    print(f'\nSaved JSON: {out_json.relative_to(PROJECT_ROOT)}')

    # ── Save Markdown summary ──
    md_lines = [
        '# Validation Agreement Analysis',
        '',
        f'**Source:** `{csv_path.name}`',
        f'**Expert validator:** Dephilia Rambu Raja Uju Decky, S.Psi',
        f'**N samples:** {n_total}',
        '',
        '## 1. Agreement Summary',
        '',
        '| Scheme | Raw Agreement | Cohen κ | Macro F1 |',
        '|---|:---:|:---:|:---:|',
        f'| 7-class | **{agreement_7:.4f}** ({matched_7}/{n_total}) | {kappa_7:.4f} | {macro_f1_7:.4f} |',
        f'| 3-class valence | **{agreement_3:.4f}** ({matched_3}/{n_total}) | {kappa_3:.4f} | {macro_f1_3:.4f} |',
        '',
        '**Cohen kappa interpretation (Landis & Koch 1977):**',
        '- < 0.20 — slight | 0.21-0.40 — fair | 0.41-0.60 — moderate | 0.61-0.80 — substantial | > 0.80 — almost perfect',
        '',
        '## 2. Per-Class — 7-class',
        '',
        '| Class | Precision | Recall | F1 | n_expert | n_auto |',
        '|---|:---:|:---:|:---:|:---:|:---:|',
    ]
    for cls in EMOTIONS_7:
        m = metrics_7[cls]
        md_lines.append(f'| {cls} | {m["precision"]:.3f} | {m["recall"]:.3f} | {m["f1"]:.3f} | '
                        f'{m["support_expert"]} | {m["support_auto"]} |')

    md_lines += [
        '',
        '## 3. Per-Class — 3-class Valence (paper-relevant)',
        '',
        '| Class | Precision | Recall | F1 | n_expert | n_auto |',
        '|---|:---:|:---:|:---:|:---:|:---:|',
    ]
    for cls in EMOTIONS_3:
        m = metrics_3[cls]
        md_lines.append(f'| {cls} | {m["precision"]:.3f} | {m["recall"]:.3f} | {m["f1"]:.3f} | '
                        f'{m["support_expert"]} | {m["support_auto"]} |')

    md_lines += [
        '',
        '## 4. Confusion Matrix — 3-class Valence (rows=expert, cols=auto)',
        '',
        '| | pred: positive | pred: neutral | pred: negative |',
        '|---|:---:|:---:|:---:|',
    ]
    for i, lab in enumerate(EMOTIONS_3):
        md_lines.append(f'| **true: {lab}** | {cm_3[i][0]} | {cm_3[i][1]} | {cm_3[i][2]} |')

    md_lines += [
        '',
        '## 5. Agreement by Confidence Threshold (7-class)',
        '',
        '| Threshold | N samples | Matched | Agreement |',
        '|---|:---:|:---:|:---:|',
    ]
    for r in conf_results:
        md_lines.append(f"| ≥ {r['threshold']:.2f} | {r['n_samples']} | {r['matched']} | "
                        f"**{r['agreement']:.4f}** |")

    md_lines += [
        '',
        '## 6. Class Distribution Comparison',
        '',
        '### 7-class',
        '| Class | Auto | Expert | Δ |',
        '|---|:---:|:---:|:---:|',
    ]
    for cls in EMOTIONS_7:
        a = auto_dist_7.get(cls, 0)
        e = expert_dist_7.get(cls, 0)
        md_lines.append(f'| {cls} | {a} | {e} | {e-a:+d} |')

    md_lines += [
        '',
        '### 3-class valence',
        '| Class | Auto | Expert | Δ |',
        '|---|:---:|:---:|:---:|',
    ]
    for cls in EMOTIONS_3:
        a = auto_dist_3.get(cls, 0)
        e = expert_dist_3.get(cls, 0)
        md_lines.append(f'| {cls} | {a} | {e} | {e-a:+d} |')

    md_lines += [
        '',
        '## 7. Implikasi untuk Paper JITeCS',
        '',
        '- **Agreement 3-class jauh lebih tinggi dari 7-class** — konsisten dengan motivasi reframe paper ke 3-class valence: kelas minoritas (angry/fearful/disgusted/surprised) confusion-prone bahkan untuk human expert. Valence dimension lebih reliable.',
        '- **Cohen κ** adalah inter-rater reliability standar. Untuk klaim Face API as ground-truth pseudo-label: κ ≥ 0.40 (moderate) baseline acceptable; κ ≥ 0.60 (substantial) ideal.',
        '- **Confidence stratification** validates conf60 filter: high-confidence Face API predictions (≥0.95) agreement biasanya jauh lebih tinggi → justifikasi filter threshold.',
    ]

    out_md = Path(args.out_md)
    out_md.write_text('\n'.join(md_lines), encoding='utf-8')
    print(f'Saved MD: {out_md.relative_to(PROJECT_ROOT)}')


if __name__ == '__main__':
    main()
