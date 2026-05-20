"""
Build visualizations untuk multimodal fusion + cross-dataset comparison.
Output ke docs/figures/multimodal/.

Re-run kapan saja untuk refresh figures saat hasil baru selesai:
    python scripts/build_multimodal_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
FIG_ROOT = PROJECT / "docs" / "figures" / "multimodal"
for sub in ("comparisons", "confusion_matrices", "training_curves",
            "leaderboards"):
    (FIG_ROOT / sub).mkdir(parents=True, exist_ok=True)

PRIMER = PROJECT / "models" / "frontonly_conf60"

SECONDARY_DATASETS = [
    ("KDEF", PROJECT / "models/benchmark/kdef_7class"),
    ("RAF-DB", PROJECT / "models/benchmark/rafdb_7class"),
    ("CK+", PROJECT / "models/benchmark/ckplus_7class"),
    ("JAFFE", PROJECT / "models/benchmark/jaffe_7class"),
]

CLASSES_3C = ["positive", "neutral", "negative"]
CLASSES_7C = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]


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


def to_macro(r):
    return r.get("test", {}).get("macro_f1") if r else None


def fusion_runs(runs):
    """Filter for fusion only."""
    return {k: v for k, v in runs.items() if v.get("_method_dir", "").startswith("fusion_")}


# ============================================================
# 1. Fusion master heatmap (per scheme, fusion × variant × source × scenario)
# ============================================================
def fig_fusion_master_heatmap(scheme):
    runs = load_all(PRIMER / f"{scheme[0]}class" / "Unified")
    fus = fusion_runs(runs)

    # Rows: (fusion, variant, source) ; Cols: scenarios
    methods = []
    for fusion in ["early", "intermediate", "late"]:
        for variant in ["scratch", "tl"]:
            for src in ["MP", "FA"]:
                src_tag = "" if src == "MP" else "_faceapi"
                # raw_136 default key
                methods.append((fusion, variant, src, src_tag))
    scn_order = ["B1", "B2", "B3"]
    row_labels = []
    matrix = []
    for fusion, variant, src, src_tag in methods:
        vals = [None, None, None]
        for i, scn in enumerate(scn_order):
            key = f"fusion_{fusion}_{variant}{src_tag}_{scn.lower()}_{scheme}"
            v = to_macro(fus.get(key))
            vals[i] = v
        if any(v is not None for v in vals):
            row_labels.append(f"{fusion[:8]} {variant.upper()} {src}")
            matrix.append(vals)
    if not matrix:
        return

    M = np.array([[(v if v is not None else np.nan) for v in row] for row in matrix])
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(row_labels) + 1.5))
    im = ax.imshow(M, cmap="viridis", aspect="auto",
                   vmin=np.nanmin(M), vmax=np.nanmax(M))
    ax.set_xticks(range(3)); ax.set_xticklabels(scn_order)
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels, fontsize=8)
    mid = (np.nanmax(M) + np.nanmin(M)) / 2
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        color="white" if v < mid else "black", fontsize=7)
    ax.set_title(f"Multimodal Fusion Master Table — Primer {scheme} (raw_136 only)")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    plt.tight_layout()
    out = FIG_ROOT / "comparisons" / f"heatmap_fusion_primer_{scheme}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  wrote {out.relative_to(PROJECT)}")


# ============================================================
# 2. Fusion vs Unimodal per dataset
# ============================================================
def fig_fusion_vs_unimodal_per_dataset(scheme):
    """Compare best unimodal vs best fusion (B1) per dataset."""
    datasets = [("Primer", PRIMER)] + SECONDARY_DATASETS
    method_names, ds_names, mf1_matrix = [], [], []
    method_labels = ["Best Landmark", "CNN scratch", "CNN_TL", "Best Fusion"]

    for ds_name, ds_path in datasets:
        runs = load_all(ds_path / f"{scheme[0]}class" / "Unified")
        ds_names.append(ds_name)
        row = []
        # Best landmark (any feature/source/arch/scenario)
        lm_runs = [v for k, v in runs.items() if v.get("_method_dir", "") in
                   {"raw_136", "facs_28", "blendshape_52", "facs_plus_bs_80"}]
        best_lm = max((to_macro(r) for r in lm_runs if to_macro(r) is not None), default=None)
        row.append(best_lm)
        # CNN scratch — pick B1 by default (sekunder cuma B1)
        for arch in ("cnn_scratch", "cnn_tl"):
            arch_runs = [v for k, v in runs.items() if v.get("_method_dir") == arch]
            best = max((to_macro(r) for r in arch_runs if to_macro(r) is not None), default=None)
            row.append(best)
        # Best fusion (any kind)
        fus = fusion_runs(runs)
        best_fus = max((to_macro(r) for r in fus.values() if to_macro(r) is not None), default=None)
        row.append(best_fus)
        mf1_matrix.append(row)

    M = np.array([[v if v is not None else np.nan for v in row] for row in mf1_matrix])
    x = np.arange(len(ds_names))
    width = 0.2
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["#5470c6", "#91cc75", "#fac858", "#ee6666"]
    for i, lbl in enumerate(method_labels):
        vals = M[:, i]
        bars = ax.bar(x + (i - 1.5) * width, np.nan_to_num(vals, nan=0), width,
                      label=lbl, color=colors[i])
        for j, (rect, v) in enumerate(zip(bars, vals)):
            if not np.isnan(v):
                ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.005,
                        f"{v:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(ds_names, fontsize=10)
    ax.set_ylabel("test macro_f1 (best across runs)")
    ax.set_title(f"Best per Method Category × Dataset — {scheme}")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, max(0.05 + np.nanmax(M), 0.5))
    plt.tight_layout()
    out = FIG_ROOT / "comparisons" / f"fusion_vs_unimodal_cross_dataset_{scheme}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  wrote {out.relative_to(PROJECT)}")


# ============================================================
# 3. Cross-dataset best fusion comparison
# ============================================================
def fig_cross_dataset_best_fusion():
    """For each dataset × scheme, show best fusion method + its mf1."""
    datasets = [("Primer", PRIMER)] + SECONDARY_DATASETS
    schemes = ["3c", "7c"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, scheme in zip(axes, schemes):
        ds_labels, mf1_best, method_text = [], [], []
        for ds_name, ds_path in datasets:
            runs = load_all(ds_path / f"{scheme[0]}class" / "Unified")
            fus = fusion_runs(runs)
            if not fus:
                ds_labels.append(ds_name); mf1_best.append(0); method_text.append("—")
                continue
            best_k = max(fus, key=lambda k: to_macro(fus[k]) or -1)
            best_v = to_macro(fus[best_k])
            ds_labels.append(ds_name); mf1_best.append(best_v or 0)
            # Shorten method name
            method_text.append(fus[best_k].get("_method_dir", "?")
                               .replace("fusion_", "").replace("_", " "))
        bars = ax.bar(ds_labels, mf1_best, color="#5470c6")
        for rect, v, t in zip(bars, mf1_best, method_text):
            ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.01,
                    f"{v:.3f}\n{t}", ha="center", fontsize=7)
        ax.set_ylabel("best fusion test macro_f1")
        ax.set_title(f"Best Fusion Method per Dataset — {scheme}")
        ax.set_ylim(0, max(mf1_best) * 1.2 if mf1_best else 1)
    plt.tight_layout()
    out = FIG_ROOT / "comparisons" / "best_fusion_per_dataset.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  wrote {out.relative_to(PROJECT)}")


# ============================================================
# 4. Top-10 multimodal leaderboard (primer)
# ============================================================
def fig_top10_fusion(scheme):
    runs = load_all(PRIMER / f"{scheme[0]}class" / "Unified")
    fus = fusion_runs(runs)
    items = []
    for k, v in fus.items():
        mf1 = to_macro(v)
        if mf1 is None:
            continue
        label = v.get("_method_dir", k).replace("fusion_", "").replace("_", " ") + \
                f" · {v.get('hyperparams', {}).get('scenario', '?')}"
        items.append((label, mf1))
    items.sort(key=lambda x: -x[1])
    top = items[:10]
    if not top:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    y = np.arange(len(top))
    ax.barh(y, [it[1] for it in top], color="#5470c6")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{i+1}. {it[0]}" for i, it in enumerate(top)], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("test macro_f1")
    ax.set_title(f"Top-10 Multimodal Fusion — Primer {scheme}")
    for i, it in enumerate(top):
        ax.text(it[1] + 0.005, i, f"{it[1]:.4f}", va="center", fontsize=8)
    ax.set_xlim(0, max(it[1] for it in top) * 1.15)
    plt.tight_layout()
    out = FIG_ROOT / "leaderboards" / f"top10_fusion_primer_{scheme}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  wrote {out.relative_to(PROJECT)}")


# ============================================================
# 5. Confusion matrices for top fusion per scheme
# ============================================================
def plot_cm(cm, classes, title, out_path):
    cm = np.array(cm, dtype=float)
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sum > 0, cm / np.maximum(row_sum, 1), 0)
    fig, ax = plt.subplots(figsize=(0.8*len(classes)+2.5, 0.8*len(classes)+2))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            t = f"{cm_norm[i,j]:.2f}\n({int(cm[i,j])})"
            ax.text(j, i, t, ha="center", va="center",
                    color="white" if cm_norm[i,j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.04)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150); plt.close(fig)


def fig_top_fusion_cm(scheme):
    runs = load_all(PRIMER / f"{scheme[0]}class" / "Unified")
    fus = fusion_runs(runs)
    cand = [(k, v, to_macro(v)) for k, v in fus.items() if to_macro(v) is not None]
    if not cand:
        return
    cand.sort(key=lambda x: -x[2])
    classes = CLASSES_3C if scheme == "3c" else CLASSES_7C
    for k, v, mf1 in cand[:3]:
        cm = v.get("test", {}).get("confusion_matrix")
        if cm is None:
            continue
        method = v.get("_method_dir", k)
        title = f"Top fusion {scheme}: {method}\nmacro_f1={mf1:.4f}"
        out = FIG_ROOT / "confusion_matrices" / f"top_{method}_{scheme}.png"
        plot_cm(cm, classes, title, out)
        print(f"  wrote {out.relative_to(PROJECT)}")


# ============================================================
# 6. Cross-dataset fusion training time
# ============================================================
def fig_training_time_comparison():
    datasets = [("Primer", PRIMER)] + SECONDARY_DATASETS
    method_groups = {
        "early scratch": "fusion_early_scratch",
        "early TL": "fusion_early_tl",
        "intermediate scratch": "fusion_intermediate_scratch",
        "intermediate TL": "fusion_intermediate_tl",
        "late scratch": "fusion_late_scratch",
        "late TL": "fusion_late_tl",
    }
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(datasets))
    width = 0.13
    colors = plt.cm.tab10(np.linspace(0, 1, len(method_groups)))
    for j, (label, method_dir) in enumerate(method_groups.items()):
        times = []
        for ds_name, ds_path in datasets:
            t_total = 0; n = 0
            for nc in ("3class", "7class"):
                runs = load_all(ds_path / nc / "Unified")
                for k, v in runs.items():
                    if v.get("_method_dir") == method_dir:
                        elapsed = v.get("training", {}).get("elapsed_sec")
                        if isinstance(elapsed, (int, float)):
                            t_total += elapsed; n += 1
            times.append(t_total / max(n, 1) if n > 0 else 0)
        offset = (j - len(method_groups)/2 + 0.5) * width
        ax.bar(x + offset, times, width, label=label, color=colors[j])
    ax.set_xticks(x); ax.set_xticklabels([d[0] for d in datasets], fontsize=10)
    ax.set_ylabel("mean training time per run (sec)")
    ax.set_title("Training Time: Fusion Methods × Datasets (mean across schemes)")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    plt.tight_layout()
    out = FIG_ROOT / "comparisons" / "training_time_comparison.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  wrote {out.relative_to(PROJECT)}")


# ============================================================
# 7. Late Fusion weight (alpha) heatmap — primer + sekunder
# ============================================================
def fig_late_fusion_weights():
    """Show optimal w_image picked per scenario × scheme × dataset."""
    datasets = [("Primer", PRIMER)] + SECONDARY_DATASETS
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, scheme in zip(axes, ["3c", "7c"]):
        ds_labels = []
        rows = []
        for ds_name, ds_path in datasets:
            runs = load_all(ds_path / f"{scheme[0]}class" / "Unified")
            late = {k: v for k, v in runs.items()
                    if v.get("_method_dir", "").startswith("fusion_late")}
            # 2 variants × 3 scenarios
            row = []
            for variant in ("scratch", "tl"):
                for scn in ("b1", "b2", "b3"):
                    key = f"fusion_late_{variant}_{scn}_{scheme[0]}c"
                    v = late.get(key)
                    if v:
                        w = v.get("best_image_weight")
                        row.append(w if w is not None else np.nan)
                    else:
                        row.append(np.nan)
            ds_labels.append(ds_name)
            rows.append(row)
        M = np.array(rows)
        col_labels = [f"{v} {s.upper()}" for v in ("scr", "tl") for s in ("b1", "b2", "b3")]
        im = ax.imshow(M, cmap="RdBu_r", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels, rotation=45, ha="right")
        ax.set_yticks(range(len(ds_labels))); ax.set_yticklabels(ds_labels)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=8)
        ax.set_title(f"Late Fusion best w_image — {scheme}\n(0=landmark-only, 1=image-only)")
        plt.colorbar(im, ax=ax, fraction=0.04)
    plt.tight_layout()
    out = FIG_ROOT / "comparisons" / "late_fusion_weights.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  wrote {out.relative_to(PROJECT)}")


# ============================================================
# Run all
# ============================================================
if __name__ == "__main__":
    for scheme in ("3c", "7c"):
        fig_fusion_master_heatmap(scheme)
        fig_fusion_vs_unimodal_per_dataset(scheme)
        fig_top10_fusion(scheme)
        fig_top_fusion_cm(scheme)
    fig_cross_dataset_best_fusion()
    fig_training_time_comparison()
    fig_late_fusion_weights()
    print(f"\nAll multimodal figures written to {FIG_ROOT.relative_to(PROJECT)}")
