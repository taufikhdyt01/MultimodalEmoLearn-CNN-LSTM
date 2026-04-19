"""
Generate Sample Images Figure for JITeCS Paper (Fig 2).

Pilih 1 sampel representatif per kelas emosi dari dataset primer conf60,
tampilkan dalam grid dengan label emosi. Pilih sampel dengan confidence
tertinggi (paling clean) supaya ilustrasi fair.

Output:
  docs/figures/class_samples.pdf   (IEEE paper insert)
  docs/figures/class_samples.png   (preview)

Data source:
  data/dataset_frontonly_conf60/X_train_images.npy + y_train.npy + y_train_soft.npy

Usage:
    python scripts/make_sample_images_figure.py                 # 7-class strip 1x7
    python scripts/make_sample_images_figure.py --layout 2x4    # 2 baris
    python scripts/make_sample_images_figure.py --seed 123      # pilih sampel lain
    python scripts/make_sample_images_figure.py --anonymize     # blur Gaussian wajah

PRIVACY NOTE:
    Sampel adalah wajah asli mahasiswa. Untuk paper publish,
    pastikan sudah dapat consent form. Pakai --anonymize kalau
    butuh blur sebagai safety.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EMOTIONS_7 = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised']

# Warna label per kelas (match class_distribution.py)
COLOR_MAJOR = '#1f77b4'
COLOR_MINOR = '#ff7f0e'
COLOR_SEVERE = '#d62728'

# Threshold severitas (mirror class_distribution.py)
TOTAL_SAMPLES = 6795


def get_label_color(count):
    pct = count / TOTAL_SAMPLES * 100
    if pct < 1:
        return COLOR_SEVERE
    if pct < 5:
        return COLOR_MINOR
    return COLOR_MAJOR


def pick_representative_sample(y_hard, y_soft, class_idx, seed=42):
    """Pick sampel dengan confidence tertinggi untuk kelas tsb.
    Fallback ke random kalau soft tidak tersedia.
    """
    idx_class = np.where(y_hard == class_idx)[0]
    if len(idx_class) == 0:
        return None
    if y_soft is not None:
        # Ambil top-5 tertinggi confidence, lalu random pick 1 (variasi)
        confs = y_soft[idx_class, class_idx]
        top_k = min(5, len(idx_class))
        top_idx_local = np.argsort(-confs)[:top_k]
        rng = np.random.RandomState(seed + class_idx)
        pick = top_idx_local[rng.randint(top_k)]
        return idx_class[pick]
    rng = np.random.RandomState(seed + class_idx)
    return idx_class[rng.randint(len(idx_class))]


def gaussian_blur_face(img, sigma=8):
    """Blur Gaussian sederhana pakai scipy (kalau ada) atau simple box filter fallback."""
    try:
        from scipy.ndimage import gaussian_filter
        # img (H,W,3) float32 [0,1]
        return np.stack([gaussian_filter(img[..., c], sigma=sigma) for c in range(3)], axis=-1)
    except ImportError:
        # Box filter fallback — manual convolve dengan kernel uniform
        k = max(3, int(sigma * 2))
        if k % 2 == 0:
            k += 1
        from numpy.lib.stride_tricks import sliding_window_view
        pad = k // 2
        padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
        # Separable box: horizontal then vertical
        h_kernel = np.ones(k) / k
        blurred = np.zeros_like(img)
        for c in range(3):
            v = padded[..., c]
            # Horizontal blur
            v = np.apply_along_axis(lambda m: np.convolve(m, h_kernel, mode='valid'), 1, v)
            # Vertical blur
            v = np.apply_along_axis(lambda m: np.convolve(m, h_kernel, mode='valid'), 0, v)
            blurred[..., c] = v
        return blurred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dpi', type=int, default=300)
    ap.add_argument('--seed', type=int, default=42, help='Random seed untuk pilihan sampel')
    ap.add_argument('--layout', choices=['1x7', '2x4', '4x2'], default='1x7')
    ap.add_argument('--label-pos', choices=['top', 'bottom'], default='bottom',
                    help='Posisi label emosi relative terhadap image')
    ap.add_argument('--anonymize', action='store_true',
                    help='Apply Gaussian blur untuk privacy')
    ap.add_argument('--split', choices=['train', 'val', 'test'], default='train')
    ap.add_argument('--no-pdf', action='store_true')
    args = ap.parse_args()

    data_dir = PROJECT_ROOT / 'data' / 'dataset_frontonly_conf60'
    X = np.load(data_dir / f'X_{args.split}_images.npy')  # (N, 224, 224, 3) float32 [0,1]
    y = np.load(data_dir / f'y_{args.split}.npy')
    y_soft_path = data_dir / f'y_{args.split}_soft.npy'
    y_soft = np.load(y_soft_path) if y_soft_path.exists() else None

    # Full dataset counts untuk label "n=" (konsisten dengan class_distribution.py)
    import json
    with open(data_dir / 'dataset_info.json') as f:
        info = json.load(f)
    dist = info['emotion_distribution']
    full_counts = [dist[e] for e in EMOTIONS_7]

    split_counts = [(y == c).sum() for c in range(len(EMOTIONS_7))]
    print(f'Split: {args.split}  n={len(y)}  (full dataset n={info["total_samples"]})')
    for emo, s_cnt, f_cnt in zip(EMOTIONS_7, split_counts, full_counts):
        print(f'  {emo:>10}: split={s_cnt:>4d}  full={f_cnt:>4d}')

    # Pick 1 sample per class
    selected_indices = []
    for c in range(len(EMOTIONS_7)):
        idx = pick_representative_sample(y, y_soft, c, args.seed)
        selected_indices.append(idx)

    # Layout
    if args.layout == '1x7':
        nrows, ncols = 1, 7
        figsize = (7.16, 1.5)  # IEEE double column strip
    elif args.layout == '4x2':
        nrows, ncols = 4, 2
        figsize = (3.5, 7.0)  # portrait single column
    else:  # 2x4 (default)
        nrows, ncols = 2, 4
        figsize = (7.16, 3.8)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for i, (emo, f_cnt) in enumerate(zip(EMOTIONS_7, full_counts)):
        ax = axes_flat[i]
        idx = selected_indices[i]
        if idx is None:
            ax.text(0.5, 0.5, f'{emo}\n(no samples)', ha='center', va='center',
                    transform=ax.transAxes, fontsize=8, color=COLOR_SEVERE)
            ax.axis('off')
            continue

        img = X[idx]  # (224, 224, 3) float32 [0,1]
        if args.anonymize:
            img = gaussian_blur_face(img, sigma=8)
        img = np.clip(img, 0, 1)

        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])

        # Simple label — nama kelas saja (atas atau bawah)
        if args.label_pos == 'top':
            ax.set_title(emo, fontsize=8, pad=3)
        else:  # bottom
            ax.set_xlabel(emo, fontsize=8, labelpad=3)

        # Thin neutral border
        for spine in ax.spines.values():
            spine.set_edgecolor('#888888')
            spine.set_linewidth(0.6)

    # Hide unused subplot (2x4 layout punya 1 slot kosong)
    for j in range(len(EMOTIONS_7), len(axes_flat)):
        axes_flat[j].axis('off')

    plt.tight_layout(pad=0.5)

    out_dir = PROJECT_ROOT / 'docs' / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = '_blur' if args.anonymize else ''
    png_path = out_dir / f'class_samples{suffix}.png'
    plt.savefig(png_path, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    print(f'\nSaved: {png_path}')

    if not args.no_pdf:
        pdf_path = out_dir / f'class_samples{suffix}.pdf'
        plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
        print(f'Saved: {pdf_path}')

    print(f'\nSelected indices: {selected_indices}')
    print(f'Seed: {args.seed}  Layout: {args.layout}  Anonymize: {args.anonymize}')


if __name__ == '__main__':
    main()
