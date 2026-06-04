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
# RQ3 helpers — parse fusion key dari run_key
# ============================================================
def parse_rq3_key(run_key: str):
    """Decompose run_key 'fusion_<type>_<...>_<scenario>_<scheme>'.

    Returns dict with: ftype, mode (concat/gated, only for early), variant,
    feature, source (MP/FA), scenario, scheme. None if unparseable.
    """
    parts = run_key.split("_")
    if len(parts) < 4 or parts[0] != "fusion":
        return None
    scheme = parts[-1]
    scenario = parts[-2].upper()
    mid = parts[1:-2]
    if not mid:
        return None
    ftype = mid[0]
    body = mid[1:]
    if body and body[-1] == "faceapi":
        source = "FA"; body = body[:-1]
    else:
        source = "MP"
    if ftype == "early":
        mode = body[0] if body else "concat"
        variant = body[1] if len(body) > 1 else "scratch"
        feature = "raw_136"
    else:
        mode = ""
        variant = body[0] if body else "scratch"
        feature = "_".join(body[1:]) if len(body) > 1 else "raw_136"
    return {"ftype": ftype, "mode": mode, "variant": variant,
            "feature": feature, "source": source,
            "scenario": scenario, "scheme": scheme}


def collect_rq3_rows(runs_dict, scheme):
    rows = []
    for k, r in runs_dict.items():
        info = parse_rq3_key(k)
        if info is None or info["scheme"] != scheme:
            continue
        mf1 = to_macro(r)
        if mf1 is None:
            continue
        info["mf1"] = mf1
        rows.append(info)
    return rows


# ============================================================
# RQ3 (a): Early concat vs gated — paired comparison
# ============================================================
def fig_rq3_early_concat_vs_gated(scheme):
    runs = fusion_runs(load_all(PRIMER / f"{scheme[0]}class" / "Unified"))
    rows = [r for r in collect_rq3_rows(runs, scheme) if r["ftype"] == "early"]
    # Group key = (variant, source, scenario) — pair concat vs gated
    keys = sorted({(r["variant"], r["source"], r["scenario"]) for r in rows})
    if not keys:
        return
    concat_vals, gated_vals, labels = [], [], []
    for variant, source, scn in keys:
        cv = next((r["mf1"] for r in rows if r["mode"] == "concat"
                   and r["variant"] == variant and r["source"] == source
                   and r["scenario"] == scn), None)
        gv = next((r["mf1"] for r in rows if r["mode"] == "gated"
                   and r["variant"] == variant and r["source"] == source
                   and r["scenario"] == scn), None)
        concat_vals.append(cv); gated_vals.append(gv)
        labels.append(f"{variant.upper()}\n{source} · {scn}")
    fig, ax = plt.subplots(figsize=(max(8, 0.7*len(labels)), 5.2))
    x = np.arange(len(labels)); width = 0.4
    ax.bar(x - width/2, [v or 0 for v in concat_vals], width,
           label="Early-concat", color="#3b7dd8")
    ax.bar(x + width/2, [v or 0 for v in gated_vals], width,
           label="Early-gated", color="#fac858")
    for i, (cv, gv) in enumerate(zip(concat_vals, gated_vals)):
        if cv is not None:
            ax.text(i - width/2, cv + 0.004, f"{cv:.3f}", ha="center", fontsize=7)
        if gv is not None:
            ax.text(i + width/2, gv + 0.004, f"{gv:.3f}", ha="center", fontsize=7)
        if cv is not None and gv is not None:
            d = gv - cv
            color = "#1a7f37" if d > 0 else "#cf222e"
            ax.text(i, max(cv, gv) + 0.025, f"Δ={d:+.4f}",
                    ha="center", fontsize=7, color=color)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("test macro_f1")
    ax.set_title(f"RQ3: Early Fusion — concat vs gated — Primer {scheme}")
    ax.legend()
    plt.tight_layout()
    out = FIG_ROOT / "comparisons" / f"rq3_early_concat_vs_gated_{scheme}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  wrote {out.relative_to(PROJECT)}")


# ============================================================
# RQ3 (b): Fusion strategy comparison — Early / Intermediate / Late
# ============================================================
def fig_rq3_fusion_strategy_comparison(scheme):
    runs = fusion_runs(load_all(PRIMER / f"{scheme[0]}class" / "Unified"))
    rows = collect_rq3_rows(runs, scheme)
    if not rows:
        return
    # Group ke 4 kategori: Early-concat, Early-gated, Intermediate, Late
    groups = {"Early-concat": [], "Early-gated": [], "Intermediate": [], "Late": []}
    for r in rows:
        if r["ftype"] == "early":
            key = "Early-concat" if r["mode"] == "concat" else "Early-gated"
        elif r["ftype"] == "intermediate":
            key = "Intermediate"
        elif r["ftype"] == "late":
            key = "Late"
        else:
            continue
        groups[key].append(r["mf1"])
    cats = [k for k in groups if groups[k]]
    data = [groups[k] for k in cats]
    means = [np.mean(v) for v in data]
    stds = [np.std(v) for v in data]
    maxs = [max(v) for v in data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    # Left: boxplot distribusi
    bp = ax1.boxplot(data, labels=cats, patch_artist=True)
    palette = ["#3b7dd8", "#fac858", "#91cc75", "#ee6666"]
    for patch, color in zip(bp["boxes"], palette[:len(cats)]):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    # Overlay max points
    for i, m in enumerate(maxs, start=1):
        ax1.scatter(i, m, color="black", marker="*", s=80, zorder=5)
        ax1.text(i, m + 0.005, f"max={m:.3f}", ha="center", fontsize=7)
    ax1.set_ylabel("test macro_f1")
    ax1.set_title(f"Distribusi mf1 per fusion strategy — {scheme}")
    ax1.set_xticklabels(cats, rotation=15)

    # Right: bar mean ± std
    x = np.arange(len(cats))
    ax2.bar(x, means, yerr=stds, capsize=5,
            color=palette[:len(cats)], alpha=0.85,
            edgecolor="black", linewidth=0.5)
    for i, (m, s, mx) in enumerate(zip(means, stds, maxs)):
        ax2.text(i, m + s + 0.005,
                 f"μ={m:.3f}\nσ={s:.3f}\nmax={mx:.3f}\nn={len(data[i])}",
                 ha="center", fontsize=7)
    ax2.set_xticks(x); ax2.set_xticklabels(cats, rotation=15)
    ax2.set_ylabel("test macro_f1 (mean ± std)")
    ax2.set_title(f"Ringkasan per strategy — {scheme}")

    plt.suptitle(f"RQ3: Perbandingan 3 Strategi Fusion (+ Early modes) — Primer {scheme}",
                 fontsize=12)
    plt.tight_layout()
    out = FIG_ROOT / "comparisons" / f"rq3_fusion_strategy_comparison_{scheme}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  wrote {out.relative_to(PROJECT)}")


# ============================================================
# RQ3 (c): Fusion × feature decomposition heatmap
# Note: Early Fusion hanya menerima raw_136 (heatmap = MP landmark
# rendering); facs/blendshape/fb80 hanya untuk Intermediate & Late.
# ============================================================
def fig_rq3_fusion_feature_decomposition(scheme):
    runs = fusion_runs(load_all(PRIMER / f"{scheme[0]}class" / "Unified"))
    rows = collect_rq3_rows(runs, scheme)
    if not rows:
        return
    features = ["raw_136", "facs_28", "blendshape_52", "facs_plus_bs_80"]
    # Build strategy_variant_source rows (FA preferred; MP also shown if FA absent)
    strat_variants = [
        ("Intermediate scratch FA", "intermediate", "scratch", "FA"),
        ("Intermediate scratch MP", "intermediate", "scratch", "MP"),
        ("Intermediate tl FA",      "intermediate", "tl",      "FA"),
        ("Intermediate tl MP",      "intermediate", "tl",      "MP"),
        ("Late scratch FA",         "late",         "scratch", "FA"),
        ("Late scratch MP",         "late",         "scratch", "MP"),
        ("Late tl FA",              "late",         "tl",      "FA"),
        ("Late tl MP",              "late",         "tl",      "MP"),
        ("Early concat scratch MP", "early-concat", "scratch", "MP"),
        ("Early concat scratch FA", "early-concat", "scratch", "FA"),
        ("Early concat tl MP",      "early-concat", "tl",      "MP"),
        ("Early concat tl FA",      "early-concat", "tl",      "FA"),
        ("Early gated scratch MP",  "early-gated",  "scratch", "MP"),
        ("Early gated scratch FA",  "early-gated",  "scratch", "FA"),
        ("Early gated tl MP",       "early-gated",  "tl",      "MP"),
        ("Early gated tl FA",       "early-gated",  "tl",      "FA"),
    ]
    M = np.full((len(strat_variants), len(features)), np.nan)
    for ri, (label, ftype_key, variant, source) in enumerate(strat_variants):
        for ci, feat in enumerate(features):
            cands = []
            for r in rows:
                # Match strategy
                if ftype_key.startswith("early-"):
                    if r["ftype"] != "early": continue
                    mode = ftype_key.split("-", 1)[1]
                    if r["mode"] != mode: continue
                else:
                    if r["ftype"] != ftype_key: continue
                if r["variant"] != variant: continue
                if r["source"] != source: continue
                if r["feature"] != feat: continue
                cands.append(r["mf1"])
            if cands:
                M[ri, ci] = max(cands)  # best across scenarios

    # Drop fully NaN rows
    keep = [i for i in range(len(strat_variants)) if not np.all(np.isnan(M[i]))]
    M2 = M[keep]
    labels = [strat_variants[i][0] for i in keep]

    fig, ax = plt.subplots(figsize=(8, 0.35 * len(labels) + 1.5))
    im = ax.imshow(M2, cmap="viridis", aspect="auto",
                   vmin=np.nanmin(M2), vmax=np.nanmax(M2))
    ax.set_xticks(range(len(features))); ax.set_xticklabels(features, rotation=15)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
    mid = (np.nanmax(M2) + np.nanmin(M2)) / 2
    for i in range(M2.shape[0]):
        for j in range(M2.shape[1]):
            v = M2[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        color="white" if v < mid else "black", fontsize=7)
    ax.set_title(f"RQ3: Fusion × feature decomposition (best mf1 across scenarios) — Primer {scheme}\n"
                 f"Early Fusion hanya raw_136; Intermediate & Late mendukung semua feature variant")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    plt.tight_layout()
    out = FIG_ROOT / "comparisons" / f"rq3_fusion_feature_decomposition_{scheme}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  wrote {out.relative_to(PROJECT)}")


# ============================================================
# Multi-metric, inference throughput, per-class for fusion
# ============================================================
def fig_fusion_multi_metric(scheme):
    """Best run per fusion strategy × 3 metrik (acc / mf1 / wf1)."""
    runs = fusion_runs(load_all(PRIMER / f"{scheme[0]}class" / "Unified"))
    rows = collect_rq3_rows(runs, scheme)
    # Group: strategy_variant
    groups = {}
    for r in rows:
        if r["ftype"] == "early":
            key = f"Early-{r['mode']} {r['variant'].upper()}"
        else:
            key = f"{r['ftype'].capitalize()} {r['variant'].upper()}"
        groups.setdefault(key, []).append(r)
    items = []
    for key, rs in groups.items():
        best = max(rs, key=lambda r: r["mf1"])
        # Map back to actual run for full metric extraction
        rk_match = None
        for raw_k, raw_v in runs.items():
            info = parse_rq3_key(raw_k)
            if (info and info["scheme"] == scheme
                    and info["ftype"] == best["ftype"]
                    and info["mode"] == best["mode"]
                    and info["variant"] == best["variant"]
                    and info["feature"] == best["feature"]
                    and info["source"] == best["source"]
                    and info["scenario"] == best["scenario"]):
                rk_match = raw_v; break
        if rk_match is None: continue
        t = rk_match.get("test", {})
        items.append({"name": key,
                      "subtag": f"{best['feature']}/{best['source']}/B{best['scenario'][-1]}",
                      "accuracy": t.get("accuracy"),
                      "macro_f1": t.get("macro_f1"),
                      "weighted_f1": t.get("weighted_f1")})
    if not items: return
    items.sort(key=lambda d: -(d["macro_f1"] or 0))

    metric_keys = ["accuracy", "macro_f1", "weighted_f1"]
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(items)); width = 0.27
    colors = ["#5470c6", "#91cc75", "#fac858"]
    for i, mk in enumerate(metric_keys):
        vals = [d[mk] if d[mk] is not None else 0 for d in items]
        ax.bar(x + (i - 1) * width, vals, width, label=mk, color=colors[i])
        for j, v in enumerate(vals):
            if v:
                ax.text(x[j] + (i - 1) * width, v + 0.005, f"{v:.3f}",
                        ha="center", fontsize=6, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d['name']}\n{d['subtag']}" for d in items],
                       rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("metric value")
    ax.set_title(f"Multi-metric (accuracy + macro_f1 + weighted_f1) — best per fusion strategy × variant — {scheme}")
    ax.legend()
    plt.tight_layout()
    out = FIG_ROOT / "comparisons" / f"multi_metric_fusion_{scheme}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  wrote {out.relative_to(PROJECT)}")


def fig_fusion_inference_throughput(scheme):
    """Inference throughput (samples/sec) per fusion strategy."""
    runs = fusion_runs(load_all(PRIMER / f"{scheme[0]}class" / "Unified"))
    rows = collect_rq3_rows(runs, scheme)
    groups = {}
    for r in rows:
        if r["ftype"] == "early":
            key = f"Early-{r['mode']} {r['variant'].upper()}"
        else:
            key = f"{r['ftype'].capitalize()} {r['variant'].upper()}"
        # find run object
        for raw_k, raw_v in runs.items():
            info = parse_rq3_key(raw_k)
            if (info and info["scheme"] == scheme
                    and info["ftype"] == r["ftype"]
                    and info["mode"] == r["mode"]
                    and info["variant"] == r["variant"]
                    and info["feature"] == r["feature"]
                    and info["source"] == r["source"]
                    and info["scenario"] == r["scenario"]):
                t = raw_v.get("test", {})
                thr = t.get("inference_throughput_samples_per_sec")
                if thr is not None:
                    groups.setdefault(key, []).append(thr)
                break
    if not groups: return
    items = sorted([(k, float(np.mean(v)), len(v)) for k, v in groups.items()],
                   key=lambda t: -t[1])
    fig, ax = plt.subplots(figsize=(11, 0.4*len(items) + 1.5))
    ys = np.arange(len(items))[::-1]
    labels = [t[0] for t in items]; vals = [t[1] for t in items]
    ax.barh(ys, vals, color=["#91cc75" if "Late" in lbl
                              else ("#fac858" if "Intermediate" in lbl else "#5470c6")
                              for lbl in labels])
    for y, v, (_, _, n) in zip(ys, vals, items):
        ax.text(v + max(vals)*0.005, y, f"{v:,.0f} (n={n})", va="center", fontsize=7)
    ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("inference throughput (samples/sec) — mean across runs")
    ax.set_xlim(0, max(vals) * 1.18)
    ax.set_title(f"Inference throughput per fusion strategy — Primer {scheme}")
    plt.tight_layout()
    out = FIG_ROOT / "comparisons" / f"inference_throughput_fusion_{scheme}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  wrote {out.relative_to(PROJECT)}")


def fig_fusion_per_class_metrics(scheme):
    """Heatmap per-class precision/recall/f1 untuk top fusion per strategy."""
    runs = fusion_runs(load_all(PRIMER / f"{scheme[0]}class" / "Unified"))
    rows = collect_rq3_rows(runs, scheme)
    # Per strategy_variant, ambil best by mf1 lalu cari run object
    groups = {}
    for r in rows:
        if r["ftype"] == "early":
            key = f"Early-{r['mode']} {r['variant'].upper()}"
        else:
            key = f"{r['ftype'].capitalize()} {r['variant'].upper()}"
        prev = groups.get(key)
        if prev is None or r["mf1"] > prev["mf1"]:
            groups[key] = r
    # Get run objects
    targets = []
    for key, r in groups.items():
        for raw_k, raw_v in runs.items():
            info = parse_rq3_key(raw_k)
            if (info and info["scheme"] == scheme
                    and info["ftype"] == r["ftype"]
                    and info["mode"] == r["mode"]
                    and info["variant"] == r["variant"]
                    and info["feature"] == r["feature"]
                    and info["source"] == r["source"]
                    and info["scenario"] == r["scenario"]):
                lbl = f"{key} {r['feature']}/{r['source']}/B{r['scenario'][-1]}"
                targets.append((lbl, raw_v)); break
    if not targets: return
    # Class names
    drop = {"accuracy", "macro avg", "weighted avg"}
    all_classes = []
    for _, run in targets:
        for c in run.get("test", {}).get("classification_report", {}):
            if c not in drop and c not in all_classes:
                all_classes.append(c)
    rows_lbl = []; M = []
    for lbl, run in targets:
        rep = run.get("test", {}).get("classification_report", {})
        for mk, pretty in [("precision","P"),("recall","R"),("f1-score","F1")]:
            rows_lbl.append(f"{lbl} · {pretty}")
            M.append([rep.get(c, {}).get(mk, np.nan) if c not in drop else np.nan
                      for c in all_classes])
    M = np.array(M, dtype=float)
    fig, ax = plt.subplots(figsize=(max(8, 0.9*len(all_classes)+3),
                                    0.30*len(rows_lbl)+2))
    im = ax.imshow(M, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(all_classes))); ax.set_xticklabels(all_classes, rotation=25, ha="right")
    ax.set_yticks(range(len(rows_lbl))); ax.set_yticklabels(rows_lbl, fontsize=6.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="black" if v > 0.5 else "white")
    ax.set_title(f"Per-class Precision/Recall/F1 — top per fusion strategy — Primer {scheme}")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    plt.tight_layout()
    out = FIG_ROOT / "confusion_matrices" / f"per_class_metrics_fusion_{scheme}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  wrote {out.relative_to(PROJECT)}")


def rq3_summary_table():
    """Return summary: optimal fusion combination per scheme × scenario."""
    out = {"scheme": [], "scenario": [], "top_strategy": [],
           "top_variant": [], "top_feature": [], "top_source": [],
           "top_mf1": []}
    for scheme in ("3c", "7c"):
        runs = fusion_runs(load_all(PRIMER / f"{scheme[0]}class" / "Unified"))
        rows = collect_rq3_rows(runs, scheme)
        for scn in ("B1", "B2", "B3"):
            cands = [r for r in rows if r["scenario"] == scn]
            if not cands: continue
            top = max(cands, key=lambda r: r["mf1"])
            strat = "Early-" + top["mode"] if top["ftype"] == "early" else top["ftype"].capitalize()
            out["scheme"].append(scheme); out["scenario"].append(scn)
            out["top_strategy"].append(strat)
            out["top_variant"].append(top["variant"])
            out["top_feature"].append(top["feature"]); out["top_source"].append(top["source"])
            out["top_mf1"].append(top["mf1"])
    return out


# ============================================================
# Run all
# ============================================================
if __name__ == "__main__":
    for scheme in ("3c", "7c"):
        fig_fusion_master_heatmap(scheme)
        fig_fusion_vs_unimodal_per_dataset(scheme)
        fig_top10_fusion(scheme)
        fig_top_fusion_cm(scheme)
        fig_rq3_early_concat_vs_gated(scheme)
        fig_rq3_fusion_strategy_comparison(scheme)
        fig_rq3_fusion_feature_decomposition(scheme)
        fig_fusion_multi_metric(scheme)
        fig_fusion_inference_throughput(scheme)
        fig_fusion_per_class_metrics(scheme)
    fig_cross_dataset_best_fusion()
    fig_training_time_comparison()
    fig_late_fusion_weights()
    print(f"\nAll multimodal figures written to {FIG_ROOT.relative_to(PROJECT)}")
