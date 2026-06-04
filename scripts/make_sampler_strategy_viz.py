"""
Visualisasi strategi sampling B1/B2/B3 untuk Bab 3 Metodologi tesis.

  B1: shuffle uniform sampler — no augmentation
  B2: WeightedRandomSampler (prob ∝ 1/class_count) — no augmentation
  B3: WeightedRandomSampler + synced per-batch augmentation

Output:
  docs/figures/sampler_strategy_b1_b2_b3.png
  docs/figures/sampler_strategy_b1_b2_b3.pdf

Usage:
    python scripts/make_sampler_strategy_viz.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.gridspec import GridSpec
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "dataset_frontonly_conf60"
OUT_DIR = PROJECT_ROOT / "docs" / "figures"

EMOTIONS_7 = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]
CLASS_COLORS = ["#5470c6", "#fac858", "#91cc75", "#ee6666",
                 "#73c0de", "#9b59b6", "#fd6c6c"]


def load_class_counts():
    p = DATA_DIR / "dataset_info.json"
    if not p.exists():
        y = np.load(DATA_DIR / "y_train.npy")
        return np.bincount(y, minlength=7)
    info = json.load(open(p))
    dist = info.get("emotion_distribution", {})
    return np.array([dist.get(e, 0) for e in EMOTIONS_7], dtype=int)


def simulate_batch(counts, sampler_type, batch_size=32, seed=42):
    rng = np.random.RandomState(seed)
    if sampler_type == "uniform":
        probs = counts / counts.sum()
    else:
        w = 1.0 / np.maximum(counts, 1)
        probs = w / w.sum()
    return rng.choice(len(counts), size=batch_size, p=probs)


def main():
    counts = load_class_counts()
    print(f"Class counts: {dict(zip(EMOTIONS_7, counts.tolist()))}")
    print(f"Imbalance max:min = {counts.max() // max(counts.min(), 1)}:1")

    fig = plt.figure(figsize=(15, 15))
    # Gridspec: 5 rows × 3 cols. Top row spans all 3 cols for class distribution.
    # Heights tuned dengan ruang yang cukup, hspace lebih besar untuk hindari overlap.
    gs = GridSpec(5, 3, figure=fig,
                   height_ratios=[1.4, 0.7, 1.2, 1.4, 0.9],
                   hspace=0.65, wspace=0.30,
                   left=0.06, right=0.97, top=0.93, bottom=0.07)

    # ── Row 0: Class distribution (full width) ────────────────────────────────
    ax_dist = fig.add_subplot(gs[0, :])
    bars = ax_dist.bar(range(7), counts, color=CLASS_COLORS,
                       edgecolor="black", linewidth=0.5)
    for b, c in zip(bars, counts):
        ax_dist.text(b.get_x() + b.get_width() / 2, b.get_height() + counts.max() * 0.018,
                     f"{c}", ha="center", fontsize=9, fontweight="bold")
    ax_dist.set_xticks(range(7))
    ax_dist.set_xticklabels(EMOTIONS_7, fontsize=10)
    ax_dist.set_ylabel("samples", fontsize=10)
    ax_dist.set_title(
        f"Class distribution Primer — max:min ratio = "
        f"{counts.max() // max(counts.min(), 1)}:1 (severe imbalance)",
        fontsize=12, fontweight="bold", pad=10)
    ax_dist.set_axisbelow(True)
    ax_dist.grid(axis="y", linestyle=":", alpha=0.4)
    ax_dist.set_ylim(0, counts.max() * 1.18)

    # ── Row 1: Scenario headers ──────────────────────────────────────────────
    titles = ["B1", "B2", "B3"]
    subtitles = [
        "Shuffle uniform sampler\n+ no augmentation",
        "WeightedRandomSampler\n(prob ∝ 1/class_count)\n+ no augmentation",
        "WeightedRandomSampler\n+ synced per-batch aug\n(hflip + rotate + photo)",
    ]
    header_colors = ["#cfe4f4", "#fde4a0", "#ffc8a0"]
    for i in range(3):
        ax = fig.add_subplot(gs[1, i])
        ax.axis("off")
        ax.add_patch(FancyBboxPatch((0.04, 0.10), 0.92, 0.80,
                                      boxstyle="round,pad=0.04",
                                      facecolor=header_colors[i],
                                      edgecolor="#555555", linewidth=1.2))
        ax.text(0.5, 0.74, titles[i], ha="center", va="center",
                fontsize=22, fontweight="bold", color="#222222")
        ax.text(0.5, 0.28, subtitles[i], ha="center", va="center",
                fontsize=9, color="#333333")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # ── Row 2: Sample probability per class ──────────────────────────────────
    for i, sampler in enumerate(["uniform", "weighted_random", "weighted_random"]):
        ax = fig.add_subplot(gs[2, i])
        if sampler == "uniform":
            probs = counts / counts.sum()
            title = "Sample probability\n(proportional to count)"
        else:
            w = 1.0 / np.maximum(counts, 1)
            probs = w / w.sum()
            title = "Sample probability\n(inverse class freq)"
        ax.bar(range(7), probs * 100, color=CLASS_COLORS,
               edgecolor="black", linewidth=0.4)
        ax.set_xticks(range(7))
        ax.set_xticklabels([e[:3] for e in EMOTIONS_7], fontsize=8)
        ax.set_ylabel("prob (%)", fontsize=9)
        ax.set_title(title, fontsize=10, pad=6)
        ax.set_axisbelow(True)
        ax.grid(axis="y", linestyle=":", alpha=0.3)

    # ── Row 3: Simulated batch (32 samples grid 4×8) ─────────────────────────
    for i, sampler in enumerate(["uniform", "weighted_random", "weighted_random"]):
        ax = fig.add_subplot(gs[3, i])
        batch_classes = simulate_batch(counts, sampler, 32, seed=42 + i)
        for k in range(32):
            r, c = divmod(k, 8)
            ax.add_patch(mpatches.Rectangle((c, -r), 0.85, 0.85,
                                              facecolor=CLASS_COLORS[batch_classes[k]],
                                              edgecolor="white", linewidth=0.7))
        ax.set_xlim(-0.4, 8.4); ax.set_ylim(-3.6, 1.1)
        ax.set_aspect("equal"); ax.axis("off")
        bc = np.bincount(batch_classes, minlength=7)
        comp = "  ".join(f"{EMOTIONS_7[c][:3]}={bc[c]}" for c in range(7) if bc[c] > 0)
        ax.set_title(f"Simulated batch (32 samples)\n{comp}",
                     fontsize=9, pad=8)

    # ── Row 4: Augmentation status ───────────────────────────────────────────
    aug_descs = [
        "✗ No augmentation\n(raw sample direct)",
        "✗ No augmentation\n(raw sample direct)",
        "✓ Synced per-batch aug:\n"
        "• hflip + landmark swap + heatmap flip\n"
        "• rotate ±10°\n"
        "• brightness/contrast ±10%",
    ]
    aug_colors = ["#f5f5f5", "#f5f5f5", "#e8f5e9"]
    for i in range(3):
        ax = fig.add_subplot(gs[4, i])
        ax.axis("off")
        ax.add_patch(FancyBboxPatch((0.04, 0.08), 0.92, 0.84,
                                      boxstyle="round,pad=0.05",
                                      facecolor=aug_colors[i],
                                      edgecolor="#999999", linewidth=0.9))
        ax.text(0.5, 0.5, aug_descs[i], ha="center", va="center",
                fontsize=9.5, color="#222222")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # ── Main title ───────────────────────────────────────────────────────────
    fig.suptitle("Sampler Strategy — B1 / B2 / B3 (3 imbalance scenarios)",
                 fontsize=15, fontweight="bold", y=0.97)

    # ── Class color legend (footer, full width) ──────────────────────────────
    handles = [mpatches.Patch(color=CLASS_COLORS[i], label=EMOTIONS_7[i])
                for i in range(7)]
    fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, 0.005))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "sampler_strategy_b1_b2_b3.png"
    out_pdf = OUT_DIR / "sampler_strategy_b1_b2_b3.pdf"
    plt.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.20,
                facecolor="white")
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.20, facecolor="white")
    plt.close(fig)
    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
