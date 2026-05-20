"""
Visualize spatial gate maps dari Gated Early Fusion.

Load trained checkpoint, run forward dengan `return_gates=True`, plot:
1. Per-sample gate map (RGB+heatmap + g_rgb mask + g_heatmap mask + prediction)
2. Per-class average gate map (mean across test samples per true label)
3. Aggregate statistics (mean/std gate values per modality)

Output: docs/figures/multimodal/gates/

Usage:
    python scripts/visualize_gates.py \
        --checkpoint models/.../fusion_early_gated_tl/model.pt \
        --num-samples 8
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))
from training.models import EmotionEarlyFusionGated, EmotionEarlyFusionTransferGated  # noqa: E402

CLASSES_3C = ["positive", "neutral", "negative"]
CLASSES_7C = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]
REMAP_3 = np.array([1, 0, 2, 2, 2, 2, 0], dtype=np.int64)


def load_model(checkpoint_path, variant, num_classes):
    """Load gated fusion model. variant ∈ {scratch, tl}."""
    cls = EmotionEarlyFusionTransferGated if variant == "tl" else EmotionEarlyFusionGated
    model = cls(num_classes=num_classes,
                pretrained=False if variant == "tl" else None)  # weights from checkpoint, not pretrained
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def load_data(data_dir, split, num_classes):
    img = np.load(Path(data_dir) / f"X_{split}_images.npy")
    hm = np.load(Path(data_dir) / f"X_{split}_heatmaps.npy")
    y = np.load(Path(data_dir) / f"y_{split}.npy").astype(np.int64)
    if num_classes == 3:
        y = REMAP_3[y]
    return img, hm, y


def stack_input(img, hm):
    """(N, H, W, 3) + (N, H, W) → (N, 4, H, W)"""
    img_chw = np.transpose(img, (0, 3, 1, 2))
    hm_c = hm[:, None, :, :]
    return np.concatenate([img_chw, hm_c], axis=1)


# ---------- Visualization ----------
def fig_sample_gates(model, img, hm, y, classes, out_dir, n_samples=8, seed=42):
    rng = np.random.default_rng(seed)
    # Pick balanced samples (one per class kalau bisa)
    by_class = {c: np.where(y == c)[0] for c in range(len(classes))}
    indices = []
    for c, idxs in by_class.items():
        if len(idxs):
            indices.append(rng.choice(idxs))
    while len(indices) < n_samples and len(indices) < len(y):
        i = rng.integers(0, len(y))
        if i not in indices:
            indices.append(i)
    indices = indices[:n_samples]

    x_input = stack_input(img[indices], hm[indices])
    x_tensor = torch.from_numpy(x_input).float()
    with torch.no_grad():
        logits, gates = model(x_tensor, return_gates=True)
    preds = logits.argmax(1).numpy()
    gates_np = gates.numpy()  # (N, 2, H, W)

    n = len(indices)
    fig, axes = plt.subplots(n, 4, figsize=(13, 3 * n))
    if n == 1:
        axes = axes[None, :]
    for i, idx in enumerate(indices):
        # Col 1: RGB image
        axes[i, 0].imshow(img[idx])
        axes[i, 0].set_title(f"RGB (true={classes[y[idx]]})\npred={classes[preds[i]]}",
                              fontsize=9)
        axes[i, 0].axis("off")
        # Col 2: Heatmap
        axes[i, 1].imshow(hm[idx], cmap="hot")
        axes[i, 1].set_title("Landmark heatmap", fontsize=9)
        axes[i, 1].axis("off")
        # Col 3: g_rgb gate
        g_rgb = gates_np[i, 0]
        im_rgb = axes[i, 2].imshow(g_rgb, cmap="RdYlGn", vmin=0, vmax=1)
        axes[i, 2].set_title(f"g_rgb (mean={g_rgb.mean():.3f})", fontsize=9)
        axes[i, 2].axis("off")
        plt.colorbar(im_rgb, ax=axes[i, 2], fraction=0.04)
        # Col 4: g_heatmap gate
        g_heat = gates_np[i, 1]
        im_heat = axes[i, 3].imshow(g_heat, cmap="RdYlGn", vmin=0, vmax=1)
        axes[i, 3].set_title(f"g_heatmap (mean={g_heat.mean():.3f})", fontsize=9)
        axes[i, 3].axis("off")
        plt.colorbar(im_heat, ax=axes[i, 3], fraction=0.04)
    plt.suptitle(f"Gated Early Fusion — Per-sample Gate Maps (n={n})", y=1.005, fontsize=11)
    plt.tight_layout()
    out = out_dir / "gates_per_sample.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out}")


def fig_class_average_gates(model, img, hm, y, classes, out_dir):
    """Compute mean gate map per true class — visualize what regions model attends to per emotion."""
    # Batch the input
    x_input = stack_input(img, hm)
    x_tensor = torch.from_numpy(x_input).float()
    # Process in batches to avoid OOM
    batch = 32
    gates_list = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(x_tensor), batch):
            _, g = model(x_tensor[i:i+batch], return_gates=True)
            gates_list.append(g.numpy())
    gates_all = np.concatenate(gates_list, axis=0)  # (N, 2, H, W)

    # Average per class
    n_classes = len(classes)
    fig, axes = plt.subplots(n_classes, 2, figsize=(7, 3 * n_classes))
    if n_classes == 1:
        axes = axes[None, :]
    for ci, cls in enumerate(classes):
        mask = (y == ci)
        if not mask.any():
            axes[ci, 0].text(0.5, 0.5, "no samples", ha="center", va="center")
            axes[ci, 1].text(0.5, 0.5, "no samples", ha="center", va="center")
            axes[ci, 0].set_title(f"{cls} g_rgb (n=0)", fontsize=9)
            axes[ci, 1].set_title(f"{cls} g_heatmap (n=0)", fontsize=9)
            axes[ci, 0].axis("off"); axes[ci, 1].axis("off")
            continue
        mean_gate = gates_all[mask].mean(axis=0)  # (2, H, W)
        im_rgb = axes[ci, 0].imshow(mean_gate[0], cmap="RdYlGn", vmin=0, vmax=1)
        axes[ci, 0].set_title(f"{cls} g_rgb (n={mask.sum()}, mean={mean_gate[0].mean():.3f})",
                              fontsize=9)
        axes[ci, 0].axis("off")
        plt.colorbar(im_rgb, ax=axes[ci, 0], fraction=0.04)
        im_heat = axes[ci, 1].imshow(mean_gate[1], cmap="RdYlGn", vmin=0, vmax=1)
        axes[ci, 1].set_title(f"{cls} g_heatmap (mean={mean_gate[1].mean():.3f})", fontsize=9)
        axes[ci, 1].axis("off")
        plt.colorbar(im_heat, ax=axes[ci, 1], fraction=0.04)
    plt.suptitle("Per-class Average Gate Maps (test set)", y=1.001, fontsize=11)
    plt.tight_layout()
    out = out_dir / "gates_per_class_average.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out}")


def fig_gate_stats(model, img, hm, y, out_dir):
    """Aggregate gate value distribution (histogram)."""
    x_input = stack_input(img, hm)
    x_tensor = torch.from_numpy(x_input).float()
    gates_list = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(x_tensor), 32):
            _, g = model(x_tensor[i:i+32], return_gates=True)
            gates_list.append(g.numpy())
    gates_all = np.concatenate(gates_list, axis=0)

    g_rgb = gates_all[:, 0].flatten()
    g_heat = gates_all[:, 1].flatten()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bins = np.linspace(0, 1, 41)
    ax.hist(g_rgb, bins=bins, alpha=0.6, label=f"g_rgb (mean={g_rgb.mean():.3f}, std={g_rgb.std():.3f})",
            color="#3b7dd8")
    ax.hist(g_heat, bins=bins, alpha=0.6, label=f"g_heatmap (mean={g_heat.mean():.3f}, std={g_heat.std():.3f})",
            color="#e07b00")
    ax.set_xlabel("gate value")
    ax.set_ylabel("frequency (pixels)")
    ax.set_title("Gate value distribution — semua pixel di test set")
    ax.legend(loc="upper right")
    plt.tight_layout()
    out = out_dir / "gates_value_distribution.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path ke .pt checkpoint")
    ap.add_argument("--variant", choices=["scratch", "tl"], required=True)
    ap.add_argument("--num-classes", type=int, required=True, choices=[3, 7])
    ap.add_argument("--data-dir", default=str(PROJECT / "data" / "dataset_frontonly_conf60"))
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--output-dir", default=str(PROJECT / "docs" / "figures" / "multimodal" / "gates"))
    args = ap.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, args.variant, args.num_classes)
    print(f"Loading {args.split} split from {args.data_dir}")
    img, hm, y = load_data(args.data_dir, args.split, args.num_classes)
    print(f"  N={len(y)}, image shape={img.shape}, heatmap shape={hm.shape}")
    classes = CLASSES_3C if args.num_classes == 3 else CLASSES_7C

    fig_sample_gates(model, img, hm, y, classes, out_dir, n_samples=args.num_samples)
    fig_class_average_gates(model, img, hm, y, classes, out_dir)
    fig_gate_stats(model, img, hm, y, out_dir)
    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()
