"""
Generate Sample Images Figure untuk semua dataset (Primer + 4 benchmark).

Layout: 5 baris × 7 kolom.
Tiap baris = 1 dataset, dengan label (a) Primer, (b) KDEF, (c) RAF-DB, (d) CK+, (e) JAFFE.
Tiap kolom = 1 kelas emosi (neutral, happy, sad, angry, fearful, disgusted, surprised).

Output:
  docs/figures/class_samples_all_datasets.pdf
  docs/figures/class_samples_all_datasets.png

Usage:
    python scripts/make_sample_images_all_datasets.py
    python scripts/make_sample_images_all_datasets.py --seed 7
    python scripts/make_sample_images_all_datasets.py --anonymize
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EMOTIONS_7 = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised']

# (label_caption, data_dir, image_filename_prefix, label_filename, soft_label_filename_or_None)
DATASETS = [
    ('(a) Primer', PROJECT_ROOT / 'data' / 'dataset_frontonly_conf60', 'X_train_images.npy', 'y_train.npy', 'y_train_soft.npy'),
    ('(b) KDEF',   PROJECT_ROOT / 'data' / 'benchmark' / 'kdef_7class',  'X_train_images.npy', 'y_train.npy', None),
    ('(c) RAF-DB', PROJECT_ROOT / 'data' / 'benchmark' / 'rafdb_7class', 'X_train_images.npy', 'y_train.npy', None),
    ('(d) CK+',    PROJECT_ROOT / 'data' / 'benchmark' / 'ckplus_7class', 'X_train_images.npy', 'y_train.npy', None),
    ('(e) JAFFE',  PROJECT_ROOT / 'data' / 'benchmark' / 'jaffe_7class',  'X_train_images.npy', 'y_train.npy', None),
]


def pick_representative_sample(y_hard, y_soft, class_idx, seed, rank=None):
    """Pick 1 sample dari kelas class_idx.

    rank=None → default: random pick dari top-5 confidence (seeded).
    rank=N (int) → deterministic: pick sample dengan rank-N confidence
                   (0=tertinggi, 1=kedua, dst). Berguna untuk override per-class
                   biar dapat sample yang benar-benar berbeda dari default.
    """
    idx_class = np.where(y_hard == class_idx)[0]
    if len(idx_class) == 0:
        return None
    if rank is not None and y_soft is not None:
        # Deterministic rank-based pick
        confs = y_soft[idx_class, class_idx]
        sorted_local = np.argsort(-confs)
        rank_eff = min(rank, len(sorted_local) - 1)
        return idx_class[sorted_local[rank_eff]]
    if rank is not None and y_soft is None:
        # No soft labels: deterministic index pick (offset within hard list)
        rng = np.random.RandomState(seed + class_idx)
        shuffled = rng.permutation(len(idx_class))
        return idx_class[shuffled[min(rank, len(shuffled) - 1)]]
    if y_soft is not None:
        # Pilih top-5 confidence tertinggi, lalu random pick 1
        confs = y_soft[idx_class, class_idx]
        top_k = min(5, len(idx_class))
        top_idx_local = np.argsort(-confs)[:top_k]
        rng = np.random.RandomState(seed + class_idx)
        pick = top_idx_local[rng.randint(top_k)]
        return idx_class[pick]
    rng = np.random.RandomState(seed + class_idx)
    return idx_class[rng.randint(len(idx_class))]


# Per-dataset per-class RANK override: pilih sample dengan confidence
# rank-N (0=tertinggi). Pakai ini untuk dapat sample yang benar-benar
# berbeda dari default random-top-5. Dataset lain tidak terpengaruh.
# Map: dataset_caption -> {class_idx: rank}
SAMPLE_OVERRIDES = {
    '(a) Primer': {
        2: 7,    # sad — pakai sample dengan confidence rank-7
        3: 7,    # angry
        4: 7,    # fearful
    },
}


def gaussian_blur_face(img, sigma=8):
    from scipy.ndimage import gaussian_filter
    return np.stack([gaussian_filter(img[..., c], sigma=sigma) for c in range(3)], axis=-1)


def load_dataset(data_dir, img_fname, y_fname, soft_fname):
    X = np.load(data_dir / img_fname)  # (N, H, W, 3) float32 [0,1]
    y = np.load(data_dir / y_fname)
    y_soft = None
    if soft_fname is not None:
        soft_path = data_dir / soft_fname
        if soft_path.exists():
            y_soft = np.load(soft_path)
    return X, y, y_soft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--dpi', type=int, default=200)
    ap.add_argument('--anonymize', action='store_true',
                    help='Apply Gaussian blur to Primer dataset only (others are public).')
    ap.add_argument('--no-pdf', action='store_true')
    args = ap.parse_args()

    n_datasets = len(DATASETS)
    n_classes = len(EMOTIONS_7)

    # Figure ~13in wide, ~10in tall for 5 rows × 7 cols
    fig, axes = plt.subplots(n_datasets, n_classes,
                             figsize=(2.0 * n_classes, 2.0 * n_datasets),
                             gridspec_kw={'wspace': 0.05, 'hspace': 0.18,
                                          'left': 0.08, 'right': 0.99,
                                          'top': 0.95, 'bottom': 0.03})

    for di, (caption, data_dir, img_fname, y_fname, soft_fname) in enumerate(DATASETS):
        if not data_dir.exists():
            print(f'  SKIP {caption}: dir not found ({data_dir})')
            for ax in axes[di]:
                ax.axis('off')
            continue
        print(f'  Loading {caption} from {data_dir}...')
        X, y, y_soft = load_dataset(data_dir, img_fname, y_fname, soft_fname)
        print(f'    images={X.shape}  classes_present={sorted(set(y.tolist()))}')

        overrides = SAMPLE_OVERRIDES.get(caption, {})
        for ci, emo in enumerate(EMOTIONS_7):
            ax = axes[di, ci]
            idx = pick_representative_sample(y, y_soft, ci, args.seed,
                                              rank=overrides.get(ci))
            if idx is None:
                ax.text(0.5, 0.5, '—', ha='center', va='center',
                        transform=ax.transAxes, fontsize=14, color='#888888')
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_edgecolor('#cccccc'); spine.set_linewidth(0.5)
            else:
                img = X[idx]
                if args.anonymize and di == 0:  # only Primer
                    img = gaussian_blur_face(img, sigma=8)
                img = np.clip(img, 0, 1)
                ax.imshow(img)
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_edgecolor('#888888'); spine.set_linewidth(0.6)

            # Class label on the top row only
            if di == 0:
                ax.set_title(emo, fontsize=10, pad=4)

        # Row label di kiri figure (di luar axes), left-aligned, supaya
        # tidak menimpa image. Posisi diambil dari pusat-vertikal axes baris ini.
        ax_pos = axes[di, 0].get_position()
        y_center = (ax_pos.y0 + ax_pos.y1) / 2
        fig.text(0.005, y_center, caption, fontsize=11,
                 ha='left', va='center', fontweight='bold')

    out_dir = PROJECT_ROOT / 'docs' / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = '_blur' if args.anonymize else ''

    png_path = out_dir / f'class_samples_all_datasets{suffix}.png'
    plt.savefig(png_path, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    print(f'\nSaved: {png_path}')

    if not args.no_pdf:
        pdf_path = out_dir / f'class_samples_all_datasets{suffix}.pdf'
        plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
        print(f'Saved: {pdf_path}')

    print(f'\nSeed: {args.seed}  Anonymize Primer: {args.anonymize}')


if __name__ == '__main__':
    main()
