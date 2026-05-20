"""
Per-class F1 + confusion analysis.

Reads classification_report dari semua results.json, output:
- Per-class F1 across method categories (landmark / image / fusion)
- Identify class minoritas yang paling kritikal (low F1 di best models)
- Comparison unimodal vs fusion per class — which classes benefit most from fusion?

Output: docs/figures/per_class_analysis/*.png + docs/per_class_summary.md

Usage:
    python scripts/analyze_per_class.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
FIG = PROJECT / "docs" / "figures" / "per_class_analysis"
FIG.mkdir(parents=True, exist_ok=True)
OUT_MD = PROJECT / "docs" / "per_class_summary.md"

PRIMER = PROJECT / "models" / "frontonly_conf60"
SECONDARY = {
    "KDEF": PROJECT / "models/benchmark/kdef_7class",
    "RAF-DB": PROJECT / "models/benchmark/rafdb_7class",
    "CK+": PROJECT / "models/benchmark/ckplus_7class",
    "JAFFE": PROJECT / "models/benchmark/jaffe_7class",
}

CLASSES_3C = ["positive", "neutral", "negative"]
CLASSES_7C = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]

LANDMARK_DIRS = {"raw_136", "facs_28", "blendshape_52", "facs_plus_bs_80"}
IMAGE_DIRS = {"cnn_scratch", "cnn_tl"}


# ---------- Loaders ----------
def load_all(scheme_dir: Path) -> dict:
    out = {}
    if not scheme_dir.exists():
        return out
    for f in scheme_dir.glob("*/results.json"):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for k, v in d.get("runs", {}).items():
            v["_method_dir"] = f.parent.name
            v["_run_key"] = k
            out[k] = v
    return out


def category_of(run):
    md = run.get("_method_dir", "")
    if md in LANDMARK_DIRS: return "landmark"
    if md in IMAGE_DIRS: return "image"
    if md.startswith("fusion_"): return "fusion"
    return "other"


def per_class_f1(run, classes):
    rep = run.get("test", {}).get("classification_report", {})
    return [rep.get(c, {}).get("f1-score", np.nan) for c in classes]


# ============================================================
# 1. Per-class F1 distribution — best run per category
# ============================================================
def fig_per_class_best_per_category(dataset_label, dataset_path, scheme):
    runs = load_all(dataset_path / f"{scheme[0]}class" / "Unified")
    classes = CLASSES_3C if scheme == "3c" else CLASSES_7C
    cats = ["landmark", "image", "fusion"]
    best_per_cat = {}
    for cat in cats:
        cand = [r for r in runs.values() if category_of(r) == cat
                and r.get("test", {}).get("macro_f1") is not None]
        if not cand:
            continue
        best = max(cand, key=lambda r: r["test"]["macro_f1"])
        best_per_cat[cat] = best
    if not best_per_cat:
        return

    x = np.arange(len(classes))
    width = 0.27
    fig, ax = plt.subplots(figsize=(0.9 * len(classes) + 4, 4.5))
    colors = {"landmark": "#3b7dd8", "image": "#e07b00", "fusion": "#2a9d8f"}
    for i, cat in enumerate(cats):
        if cat not in best_per_cat:
            continue
        r = best_per_cat[cat]
        f1s = per_class_f1(r, classes)
        offset = (i - 1) * width
        method = r.get("_method_dir", "?")
        mf1 = r["test"]["macro_f1"]
        bars = ax.bar(x + offset, np.nan_to_num(f1s), width,
                      label=f"{cat}: {method} mf1={mf1:.3f}", color=colors[cat])
        for rect, v in zip(bars, f1s):
            if not np.isnan(v):
                ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.01,
                        f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylabel("F1-score")
    ax.set_title(f"Per-class F1 — Best per Category ({dataset_label}, {scheme})")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    out = FIG / f"best_per_cat_{dataset_label.lower().replace('-','')}_{scheme}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  wrote {out.relative_to(PROJECT)}")


# ============================================================
# 2. Class imbalance vs F1 (scatter)
# ============================================================
def fig_class_imbalance_vs_f1(scheme):
    """Scatter: x = class size in train, y = best F1 across all runs. Per dataset."""
    if scheme == "3c":
        classes = CLASSES_3C
    else:
        classes = CLASSES_7C
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), sharey=True)
    datasets = [("Primer", PRIMER, "data/dataset_frontonly_conf60")] + [
        (n, p, f"data/benchmark/{p.name}") for n, p in SECONDARY.items()
    ]
    for ax, (ds_name, ds_path, data_path) in zip(axes, datasets):
        runs = load_all(ds_path / f"{scheme[0]}class" / "Unified")
        ydata_path = PROJECT / data_path / "y_train.npy"
        if not ydata_path.exists():
            ax.set_title(f"{ds_name} (no train y)"); continue
        y = np.load(ydata_path)
        # Remap to 3c if needed
        if scheme == "3c":
            REMAP_3 = np.array([1, 0, 2, 2, 2, 2, 0])
            y = REMAP_3[y]
        from collections import Counter
        counts = Counter(y.tolist())
        best_per_class = []
        for ci, _ in enumerate(classes):
            best = 0
            for r in runs.values():
                f1s = per_class_f1(r, classes)
                if not np.isnan(f1s[ci]) and f1s[ci] > best:
                    best = f1s[ci]
            best_per_class.append(best)
        sizes = [counts.get(ci, 0) for ci in range(len(classes))]
        ax.scatter(sizes, best_per_class, s=50, alpha=0.7)
        for ci, (sz, f1) in enumerate(zip(sizes, best_per_class)):
            ax.annotate(classes[ci][:6], (sz, f1), fontsize=7, alpha=0.8)
        ax.set_xlabel("class size (train)")
        ax.set_ylabel("best F1 across runs")
        ax.set_title(f"{ds_name}")
        ax.grid(alpha=0.3)
        ax.set_xscale("log" if max(sizes) > 100 * max(min(sizes), 1) else "linear")
    plt.suptitle(f"Class imbalance vs Best F1 — {scheme}")
    plt.tight_layout()
    out = FIG / f"class_size_vs_f1_{scheme}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  wrote {out.relative_to(PROJECT)}")


# ============================================================
# 3. Fusion vs unimodal gain per class (primer)
# ============================================================
def fig_fusion_gain_per_class(scheme):
    runs = load_all(PRIMER / f"{scheme[0]}class" / "Unified")
    classes = CLASSES_3C if scheme == "3c" else CLASSES_7C
    # Best per class for unimodal vs fusion
    cats_runs = {"unimodal": [], "fusion": []}
    for r in runs.values():
        c = category_of(r)
        if c in ("landmark", "image"):
            cats_runs["unimodal"].append(r)
        elif c == "fusion":
            cats_runs["fusion"].append(r)
    best_per_class = {"unimodal": [], "fusion": []}
    for cat, cat_runs in cats_runs.items():
        for ci in range(len(classes)):
            best = 0
            for r in cat_runs:
                f1s = per_class_f1(r, classes)
                if not np.isnan(f1s[ci]) and f1s[ci] > best:
                    best = f1s[ci]
            best_per_class[cat].append(best)
    x = np.arange(len(classes))
    width = 0.4
    fig, ax = plt.subplots(figsize=(0.8 * len(classes) + 3, 4.5))
    ax.bar(x - width/2, best_per_class["unimodal"], width, label="Best Unimodal",
           color="#5470c6")
    ax.bar(x + width/2, best_per_class["fusion"], width, label="Best Fusion",
           color="#ee6666")
    for ci in range(len(classes)):
        u, f = best_per_class["unimodal"][ci], best_per_class["fusion"][ci]
        ax.text(x[ci] - width/2, u + 0.005, f"{u:.2f}", ha="center", fontsize=7)
        ax.text(x[ci] + width/2, f + 0.005, f"{f:.2f}", ha="center", fontsize=7)
        # Delta
        delta = f - u
        col = "green" if delta > 0 else "red"
        ax.text(x[ci], max(u, f) + 0.05, f"Δ={delta:+.2f}", ha="center",
                fontsize=7, color=col, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylabel("Best F1-score (across all runs)")
    ax.set_title(f"Per-class: Unimodal vs Fusion (Best run per category) — Primer {scheme}")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower left")
    plt.tight_layout()
    out = FIG / f"fusion_gain_per_class_primer_{scheme}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  wrote {out.relative_to(PROJECT)}")


# ============================================================
# 4. Markdown summary
# ============================================================
def build_md_summary():
    lines = [
        "# Per-class F1 Summary (auto-generated)",
        "",
        "> Output dari `scripts/analyze_per_class.py`. Per-class F1 dari semua run terbaik.",
        "",
        "## Best F1 per class (Primer 7c)",
        "",
        "| Class | Best Landmark | Best Image | Best Fusion | Best Overall |",
        "|---|:---:|:---:|:---:|:---:|",
    ]
    runs = load_all(PRIMER / "7class" / "Unified")
    classes = CLASSES_7C
    for ci, cls in enumerate(classes):
        best = {"landmark": (0, ""), "image": (0, ""), "fusion": (0, "")}
        for r in runs.values():
            c = category_of(r)
            if c not in best:
                continue
            f1s = per_class_f1(r, classes)
            if np.isnan(f1s[ci]):
                continue
            if f1s[ci] > best[c][0]:
                best[c] = (f1s[ci], r.get("_method_dir", ""))
        overall = max(best.values(), key=lambda x: x[0])
        lines.append(
            f"| {cls} "
            f"| {best['landmark'][0]:.3f} ({best['landmark'][1]}) "
            f"| {best['image'][0]:.3f} ({best['image'][1]}) "
            f"| {best['fusion'][0]:.3f} ({best['fusion'][1]}) "
            f"| **{overall[0]:.3f}** ({overall[1]}) |"
        )

    lines.append("")
    lines.append("## Class size distribution (train) — Primer")
    lines.append("")
    y = np.load(PROJECT / "data/dataset_frontonly_conf60/y_train.npy")
    from collections import Counter
    counts = Counter(y.tolist())
    lines.append("| Class (7c) | Train count | Ratio vs largest |")
    lines.append("|---|:---:|:---:|")
    max_c = max(counts.values())
    for ci, cls in enumerate(classes):
        c = counts.get(ci, 0)
        lines.append(f"| {cls} | {c} | {c/max_c:.4f} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Regenerate: `python scripts/analyze_per_class.py`*")

    OUT_MD.write_text("\n".join(lines))
    print(f"\nWrote {OUT_MD.relative_to(PROJECT)}")


# ============================================================
if __name__ == "__main__":
    for scheme in ("3c", "7c"):
        fig_per_class_best_per_category("Primer", PRIMER, scheme)
        for ds_name, ds_path in SECONDARY.items():
            fig_per_class_best_per_category(ds_name, ds_path, scheme)
        fig_class_imbalance_vs_f1(scheme)
        fig_fusion_gain_per_class(scheme)
    build_md_summary()
    print(f"\nDone. Figures: {FIG.relative_to(PROJECT)}")
