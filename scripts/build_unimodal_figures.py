"""
Build visualizations for unimodal experiments.
Output to docs/figures/unimodal/.

Re-run kapan saja untuk refresh figures saat hasil baru selesai.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
FIG_ROOT = PROJECT / "docs" / "figures" / "unimodal"
for sub in ("comparisons", "confusion_matrices", "training_curves",
            "leaderboards", "dataset", "resources", "per_class"):
    (FIG_ROOT / sub).mkdir(parents=True, exist_ok=True)

PRIMER = PROJECT / "models" / "frontonly_conf60"

# Cross-dataset benchmarks (name → models dir → data dir for class distribution)
BENCHMARKS = [
    ("KDEF",   PROJECT / "models" / "benchmark" / "kdef_7class",
                PROJECT / "data"   / "benchmark" / "kdef_7class",   "#91cc75"),
    ("RAF-DB", PROJECT / "models" / "benchmark" / "rafdb_7class",
                PROJECT / "data"   / "benchmark" / "rafdb_7class",  "#fac858"),
    ("CK+",    PROJECT / "models" / "benchmark" / "ckplus_7class",
                PROJECT / "data"   / "benchmark" / "ckplus_7class", "#ee6666"),
    ("JAFFE",  PROJECT / "models" / "benchmark" / "jaffe_7class",
                PROJECT / "data"   / "benchmark" / "jaffe_7class",  "#73c0de"),
]

UNIMODAL_DIRS = {"raw_136", "facs_28", "blendshape_52", "facs_plus_bs_80",
                 "cnn_scratch", "cnn_tl"}


# ---------- Loaders ----------
def load_results_dir(scheme_dir: Path):
    out = []
    if not scheme_dir.exists():
        return out
    for results_file in scheme_dir.glob("*/results.json"):
        method = results_file.parent.name
        if method not in UNIMODAL_DIRS:
            continue
        try:
            d = json.load(open(results_file))
        except Exception as e:
            print(f"  skip {results_file}: {e}")
            continue
        for run_key, run in d.get("runs", {}).items():
            run["_method_dir"] = method
            run["_run_key"] = run_key
            out.append(run)
    return out


primer = {}
for scheme, sk in [("3class", "3c"), ("7class", "7c")]:
    primer[sk] = load_results_dir(PRIMER / scheme / "Unified")
    print(f"Primer {sk}: {len(primer[sk])} unimodal runs")

benchmark_runs = {}  # {bench_name: {sk: [runs]}}
for bname, bdir, _, _ in BENCHMARKS:
    benchmark_runs[bname] = {}
    for scheme, sk in [("3class", "3c"), ("7class", "7c")]:
        benchmark_runs[bname][sk] = load_results_dir(bdir / scheme / "Unified")
        print(f"{bname} {sk}: {len(benchmark_runs[bname][sk])} unimodal runs")


def to_macro(r):
    return r.get("test", {}).get("macro_f1")


def collect_landmark_rows(runs):
    rows = []
    for r in runs:
        rk = r["_run_key"]
        parts = rk.split("_")
        if parts[0] not in ("mediapipe", "faceapi"):
            continue
        source = "MP" if parts[0] == "mediapipe" else "FA"
        arch = parts[-3]
        scn = parts[-2].upper()
        scheme = parts[-1]
        feature = "_".join(parts[1:-3])
        rows.append({"source": source, "feature": feature, "arch": arch,
                     "scenario": scn, "scheme": scheme,
                     "mf1": to_macro(r), "key": rk, "run": r})
    return rows


def collect_image_rows(runs):
    rows = []
    for r in runs:
        if r["_method_dir"] not in ("cnn_scratch", "cnn_tl"):
            continue
        rk = r["_run_key"]
        parts = rk.split("_")
        rows.append({"arch": r["_method_dir"],
                     "scenario": parts[-2].upper(),
                     "scheme": parts[-1],
                     "mf1": to_macro(r), "key": rk, "run": r})
    return rows


# ============================================================
# 1. Master table heatmap (per scheme)
# ============================================================
def fig_master_heatmap(scheme):
    lm = collect_landmark_rows(primer[scheme])
    img = collect_image_rows(primer[scheme])

    feat_order = ["raw_136", "facs_28", "blendshape_52", "facs_plus_bs_80"]
    src_order = ["MP", "FA"]
    arch_order = ["fcnn", "cnn1d"]
    scn_order = ["B1", "B2", "B3"]

    row_labels, matrix = [], []
    for feat in feat_order:
        for src in src_order:
            for arch in arch_order:
                if feat == "blendshape_52" and src == "FA":
                    continue
                vals = [None, None, None]
                for r in lm:
                    if (r["feature"] == feat and r["source"] == src
                        and r["arch"] == arch and r["scheme"] == scheme):
                        vals[scn_order.index(r["scenario"])] = r["mf1"]
                if any(v is not None for v in vals):
                    row_labels.append(f"{feat[:14]} • {src} • {arch.upper()}")
                    matrix.append(vals)
    for arch in ["cnn_scratch", "cnn_tl"]:
        vals = [None, None, None]
        for r in img:
            if r["arch"] == arch and r["scheme"] == scheme:
                vals[scn_order.index(r["scenario"])] = r["mf1"]
        if any(v is not None for v in vals):
            row_labels.append(f"image • — • {arch.upper()}")
            matrix.append(vals)

    M = np.array([[(v if v is not None else np.nan) for v in row]
                  for row in matrix])
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
    ax.set_title(f"Unimodal Master Table — {scheme} (test macro_f1)")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    plt.tight_layout()
    out = FIG_ROOT / "comparisons" / f"heatmap_master_{scheme}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  wrote {out.relative_to(PROJECT)}")


# ============================================================
# 2. Top-10 leaderboard (landmark + image)
# ============================================================
def fig_top10(scheme):
    lm = collect_landmark_rows(primer[scheme])
    img = collect_image_rows(primer[scheme])
    items = []
    for r in lm:
        if r["mf1"] is None: continue
        items.append((f"{r['feature']} • {r['source']} • {r['arch'].upper()} • {r['scenario']}",
                      r["mf1"], "landmark"))
    for r in img:
        if r["mf1"] is None: continue
        items.append((f"image • {r['arch'].upper()} • {r['scenario']}", r["mf1"], "image"))
    items.sort(key=lambda x: -x[1])
    top = items[:10]
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"landmark": "#3b7dd8", "image": "#e07b00"}
    y = np.arange(len(top))
    ax.barh(y, [it[1] for it in top], color=[colors[it[2]] for it in top])
    ax.set_yticks(y)
    ax.set_yticklabels([f"{i+1}. {it[0]}" for i, it in enumerate(top)], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("test macro_f1")
    ax.set_title(f"Top-10 Unimodal — {scheme}")
    for i, it in enumerate(top):
        ax.text(it[1] + 0.005, i, f"{it[1]:.4f}", va="center", fontsize=8)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=n) for n, c in colors.items()],
              loc="lower right")
    ax.set_xlim(0, max(it[1] for it in top) * 1.15)
    plt.tight_layout()
    out = FIG_ROOT / "leaderboards" / f"top10_{scheme}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  wrote {out.relative_to(PROJECT)}")


# ============================================================
# 3. B1 / B2 / B3 scenario comparison
# ============================================================
def fig_scenario_compare(scheme):
    lm = collect_landmark_rows(primer[scheme])
    img = collect_image_rows(primer[scheme])

    methods = [
        ("FACS_28 FA FCNN", "lm", dict(feature="facs_28", source="FA", arch="fcnn")),
        ("Raw_136 FA CNN1D", "lm", dict(feature="raw_136", source="FA", arch="cnn1d")),
        ("FB80 FA FCNN", "lm", dict(feature="facs_plus_bs_80", source="FA", arch="fcnn")),
        ("Raw_136 MP FCNN", "lm", dict(feature="raw_136", source="MP", arch="fcnn")),
        ("Raw_136 MP CNN1D", "lm", dict(feature="raw_136", source="MP", arch="cnn1d")),
        ("FACS_28 MP FCNN", "lm", dict(feature="facs_28", source="MP", arch="fcnn")),
        ("Blendshape MP FCNN", "lm", dict(feature="blendshape_52", source="MP", arch="fcnn")),
        ("CNN scratch (image)", "img", dict(arch="cnn_scratch")),
        ("CNN_TL (image)", "img", dict(arch="cnn_tl")),
    ]
    scn_order = ["B1", "B2", "B3"]
    bars, labels = [], []
    for name, kind, spec in methods:
        vals = [None, None, None]
        src = lm if kind == "lm" else img
        for r in src:
            if all(r.get(k) == v for k, v in spec.items()) and r["scheme"] == scheme:
                vals[scn_order.index(r["scenario"])] = r["mf1"]
        if any(v is not None for v in vals):
            bars.append(vals); labels.append(name)
    M = np.array([[v if v is not None else 0 for v in b] for b in bars])
    x = np.arange(len(labels)); width = 0.27
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["#5470c6", "#91cc75", "#fac858"]
    for i, scn in enumerate(scn_order):
        bs = ax.bar(x + (i - 1) * width, M[:, i], width, label=scn, color=colors[i])
        for rect, v in zip(bs, M[:, i]):
            if v > 0:
                ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8.5)
    ax.set_ylabel("test macro_f1")
    ax.set_title(f"B1 / B2 / B3 Comparison — {scheme}")
    ax.legend(title="Scenario")
    ax.set_ylim(0, M.max() * 1.15)
    plt.tight_layout()
    out = FIG_ROOT / "comparisons" / f"scenario_comparison_{scheme}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  wrote {out.relative_to(PROJECT)}")


# ============================================================
# 4. FA vs MP comparison (landmark)
# ============================================================
def fig_fa_vs_mp(scheme):
    lm = collect_landmark_rows(primer[scheme])
    keys = sorted({(r["feature"], r["arch"], r["scenario"]) for r in lm
                   if r["feature"] != "blendshape_52"})
    fa, mp, labels = [], [], []
    for feat, arch, scn in keys:
        v_fa = next((r["mf1"] for r in lm if r["feature"]==feat and r["arch"]==arch
                     and r["scenario"]==scn and r["source"]=="FA" and r["scheme"]==scheme), None)
        v_mp = next((r["mf1"] for r in lm if r["feature"]==feat and r["arch"]==arch
                     and r["scenario"]==scn and r["source"]=="MP" and r["scheme"]==scheme), None)
        if v_fa is not None and v_mp is not None:
            fa.append(v_fa); mp.append(v_mp)
            labels.append(f"{feat[:8]} {arch.upper()} {scn}")
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(labels)); width = 0.4
    ax.bar(x - width/2, mp, width, label="MediaPipe", color="#e07b00")
    ax.bar(x + width/2, fa, width, label="face-api.js", color="#3b7dd8")
    for i in range(len(labels)):
        ax.text(i - width/2, mp[i] + 0.005, f"{mp[i]:.3f}", ha="center", fontsize=6.5)
        ax.text(i + width/2, fa[i] + 0.005, f"{fa[i]:.3f}", ha="center", fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("test macro_f1")
    ax.set_title(f"Landmark Source: MediaPipe vs face-api.js — {scheme}")
    ax.legend()
    plt.tight_layout()
    out = FIG_ROOT / "comparisons" / f"fa_vs_mp_{scheme}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  wrote {out.relative_to(PROJECT)}")


# ============================================================
# 5. Feature comparison (raw / facs / blendshape / fb80) — at best scenario per
# ============================================================
def fig_feature_compare(scheme):
    lm = collect_landmark_rows(primer[scheme])
    feat_order = ["raw_136", "facs_28", "blendshape_52", "facs_plus_bs_80"]
    src_order = ["MP", "FA"]
    # Average over arch + scenario per (feature, source)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(feat_order)); width = 0.4
    for j, src in enumerate(src_order):
        means, errs, exists = [], [], []
        for feat in feat_order:
            vals = [r["mf1"] for r in lm
                    if r["feature"] == feat and r["source"] == src
                    and r["scheme"] == scheme and r["mf1"] is not None]
            if vals:
                means.append(np.mean(vals)); errs.append(np.max(vals) - np.min(vals))
                exists.append(True)
            else:
                means.append(0); errs.append(0); exists.append(False)
        offset = (j - 0.5) * width
        color = "#e07b00" if src == "MP" else "#3b7dd8"
        ax.bar(x + offset, means, width, yerr=errs, capsize=4,
               label=src, color=color, alpha=0.9)
        for i, (m, e, ok) in enumerate(zip(means, errs, exists)):
            if ok:
                ax.text(i + offset, m + e + 0.005, f"{m:.3f}",
                        ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(feat_order, fontsize=9)
    ax.set_ylabel("mean test macro_f1 (over arch × scenario)")
    ax.set_title(f"Feature Comparison — {scheme} (error bar = range across arch × scenario)")
    ax.legend(title="Source")
    plt.tight_layout()
    out = FIG_ROOT / "comparisons" / f"feature_comparison_{scheme}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  wrote {out.relative_to(PROJECT)}")


# ============================================================
# 6. Architecture comparison: FCNN vs CNN1D per feature × source
# ============================================================
def fig_arch_compare(scheme):
    lm = collect_landmark_rows(primer[scheme])
    pairs = []
    for feat in ["raw_136", "facs_28", "blendshape_52", "facs_plus_bs_80"]:
        for src in ["MP", "FA"]:
            if feat == "blendshape_52" and src == "FA":
                continue
            fcnn_vals = [r["mf1"] for r in lm if r["feature"]==feat and r["source"]==src
                         and r["arch"]=="fcnn" and r["scheme"]==scheme and r["mf1"] is not None]
            cnn1d_vals = [r["mf1"] for r in lm if r["feature"]==feat and r["source"]==src
                          and r["arch"]=="cnn1d" and r["scheme"]==scheme and r["mf1"] is not None]
            if fcnn_vals and cnn1d_vals:
                pairs.append((f"{feat[:8]} {src}", np.mean(fcnn_vals), np.mean(cnn1d_vals)))
    if not pairs:
        return
    labels = [p[0] for p in pairs]
    fcnn = [p[1] for p in pairs]; cnn1d = [p[2] for p in pairs]
    x = np.arange(len(labels)); width = 0.4
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, fcnn, width, label="FCNN", color="#9b59b6")
    ax.bar(x + width/2, cnn1d, width, label="CNN1D", color="#16a085")
    for i, (f, c) in enumerate(zip(fcnn, cnn1d)):
        ax.text(i - width/2, f + 0.005, f"{f:.3f}", ha="center", fontsize=7)
        ax.text(i + width/2, c + 0.005, f"{c:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("mean test macro_f1 (over scenarios)")
    ax.set_title(f"Architecture Comparison: FCNN vs CNN1D — {scheme}")
    ax.legend()
    plt.tight_layout()
    out = FIG_ROOT / "comparisons" / f"arch_comparison_{scheme}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  wrote {out.relative_to(PROJECT)}")


# ============================================================
# 7. Confusion matrices
# ============================================================
CLASSES_3C = ["positive", "neutral", "negative"]
CLASSES_7C = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]


def plot_cm(cm, classes, title, out_path, normalize=True):
    cm = np.array(cm, dtype=float)
    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        cm_norm = np.where(row_sum > 0, cm / np.maximum(row_sum, 1), 0)
    else:
        cm_norm = cm
    fig, ax = plt.subplots(figsize=(0.8*len(classes)+2.5, 0.8*len(classes)+2))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1 if normalize else cm.max())
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            t = f"{cm_norm[i,j]:.2f}\n({int(cm[i,j])})" if normalize else f"{int(cm[i,j])}"
            ax.text(j, i, t, ha="center", va="center",
                    color="white" if cm_norm[i,j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.04)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150); plt.close(fig)


def fig_confusion_top(scheme):
    classes = CLASSES_3C if scheme == "3c" else CLASSES_7C
    runs = primer[scheme]
    landmark_dirs = {"raw_136", "facs_28", "blendshape_52", "facs_plus_bs_80"}
    image_dirs = {"cnn_scratch", "cnn_tl"}
    cats = [("best_landmark", landmark_dirs), ("best_image", image_dirs)]
    for name, dirs in cats:
        cand = [r for r in runs if r["_method_dir"] in dirs and to_macro(r) is not None]
        if not cand: continue
        best = max(cand, key=to_macro)
        cm = best.get("test", {}).get("confusion_matrix")
        if cm is None: continue
        title = f"{name} {scheme}: {best['_method_dir']}\n{best['_run_key']}  macro_f1={best['test']['macro_f1']:.4f}"
        out = FIG_ROOT / "confusion_matrices" / f"{name}_{scheme}.png"
        plot_cm(cm, classes, title, out, normalize=True)
        print(f"  wrote {out.relative_to(PROJECT)}")


# ============================================================
# 8. Per-class F1 for top-3 models
# ============================================================
def fig_per_class_f1(scheme):
    classes = CLASSES_3C if scheme == "3c" else CLASSES_7C
    runs = primer[scheme]
    cand = [r for r in runs if to_macro(r) is not None]
    cand.sort(key=to_macro, reverse=True)
    top = cand[:3]
    if not top: return
    x = np.arange(len(classes)); width = 0.27
    fig, ax = plt.subplots(figsize=(0.9*len(classes)+5, 4.5))
    colors = ["#3b7dd8", "#e07b00", "#16a085"]
    for i, r in enumerate(top):
        rep = r["test"].get("classification_report", {})
        f1s = [rep.get(c, {}).get("f1-score", 0) for c in classes]
        bs = ax.bar(x + (i-1)*width, f1s, width,
                    color=colors[i], label=f"{r['_method_dir']} {r['_run_key'].split('_')[-2].upper()}: {to_macro(r):.3f}")
        for rect, v in zip(bs, f1s):
            ax.text(rect.get_x()+rect.get_width()/2, rect.get_height()+0.01,
                    f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylabel("f1-score")
    ax.set_title(f"Per-class F1 — Top-3 unimodal {scheme}")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    out = FIG_ROOT / "per_class" / f"top3_per_class_f1_{scheme}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  wrote {out.relative_to(PROJECT)}")


# ============================================================
# 9. Training curves (top-3 landmark + top-3 image)
# ============================================================
def fig_training_curves(scheme):
    runs = primer[scheme]
    cats = [
        ("landmark", {"raw_136","facs_28","blendshape_52","facs_plus_bs_80"}),
        ("image", {"cnn_scratch","cnn_tl"}),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, (name, dirs) in zip(axes, cats):
        cand = [r for r in runs if r["_method_dir"] in dirs and to_macro(r) is not None]
        cand.sort(key=to_macro, reverse=True)
        for r in cand[:3]:
            hist = r.get("training", {}).get("history", [])
            if not hist: continue
            ep = [h["epoch"] for h in hist]
            val = [h.get("val_macro_f1") for h in hist]
            label = f"{r['_method_dir']} / {r['hyperparams']['scenario']}: {to_macro(r):.3f}"
            ax.plot(ep, val, marker="o", markersize=3, linewidth=1.5, label=label)
            be = r.get("training",{}).get("best_epoch")
            if be is not None and 1 <= be <= len(hist):
                ax.scatter([be], [val[be-1]], marker="*", s=140,
                           edgecolor="k", zorder=10)
        ax.set_xlabel("epoch")
        ax.set_title(f"Top-3 {name} ({scheme})")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("val_macro_f1")
    plt.suptitle(f"Training curves — top-3 unimodal per category, {scheme}")
    plt.tight_layout()
    out = FIG_ROOT / "training_curves" / f"top3_{scheme}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  wrote {out.relative_to(PROJECT)}")


# ============================================================
# 10. Resource comparison: training time / params / VRAM (3c)
# ============================================================
def fig_resources(scheme):
    runs = primer[scheme]
    rows = []
    for r in runs:
        mf1 = to_macro(r)
        if mf1 is None: continue
        params = r.get("model", {}).get("n_params")
        train_t = r.get("training", {}).get("elapsed_sec")
        vram = r.get("training", {}).get("peak_vram_mb")
        rows.append({
            "label": f"{r['_method_dir']}",
            "mf1": mf1, "params": params, "train_t": train_t, "vram": vram,
        })
    # Aggregate per method_dir
    by_method = {}
    for r in rows:
        by_method.setdefault(r["label"], []).append(r)
    methods = sorted(by_method.keys())
    mean_mf1 = [np.mean([r["mf1"] for r in by_method[m]]) for m in methods]
    mean_t = [np.mean([r["train_t"] for r in by_method[m] if r["train_t"] is not None]) for m in methods]
    mean_p = [np.mean([r["params"] for r in by_method[m] if r["params"] is not None])/1e6 for m in methods]
    mean_v = [np.mean([r["vram"] for r in by_method[m] if r["vram"] is not None]) for m in methods]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    x = np.arange(len(methods))
    axes[0].bar(x, mean_t, color="#5470c6")
    axes[0].set_xticks(x); axes[0].set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("seconds"); axes[0].set_title("Mean training time")
    for i, v in enumerate(mean_t):
        axes[0].text(i, v+max(mean_t)*0.01, f"{v:.0f}s", ha="center", fontsize=7)
    axes[1].bar(x, mean_p, color="#91cc75")
    axes[1].set_xticks(x); axes[1].set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
    axes[1].set_ylabel("M params"); axes[1].set_title("Model size (parameters)")
    for i, v in enumerate(mean_p):
        axes[1].text(i, v+max(mean_p)*0.01, f"{v:.1f}M", ha="center", fontsize=7)
    axes[2].bar(x, mean_v, color="#fac858")
    axes[2].set_xticks(x); axes[2].set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
    axes[2].set_ylabel("MB"); axes[2].set_title("Peak VRAM")
    for i, v in enumerate(mean_v):
        axes[2].text(i, v+max(mean_v)*0.01, f"{v:.0f}", ha="center", fontsize=7)
    plt.suptitle(f"Resource Usage Comparison — unimodal {scheme} (mean across scenarios)")
    plt.tight_layout()
    out = FIG_ROOT / "resources" / f"resource_compare_{scheme}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  wrote {out.relative_to(PROJECT)}")


# ============================================================
# 11. Class distribution
# ============================================================
def fig_class_distribution():
    pairs = [
        ("Primer 7c (train)", PROJECT/"data/dataset_frontonly_conf60/y_train.npy",
         PROJECT/"data/dataset_frontonly_conf60/label_map.json"),
    ]
    for bname, _, ddir, _ in BENCHMARKS:
        pairs.append((f"{bname} 7c (train)", ddir/"y_train.npy", ddir/"label_map.json"))
    n = len(pairs)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4.5*nrows), squeeze=False)
    axes_flat = axes.flatten()
    for ax, (title, ypath, lm_file) in zip(axes_flat, pairs):
        if not ypath.exists():
            ax.set_title(f"{title} (missing)"); continue
        y = np.load(ypath)
        try:
            label_map = json.load(open(lm_file))
            if isinstance(label_map, dict) and all(isinstance(v, int) for v in label_map.values()):
                inv = {v: k for k, v in label_map.items()}
                classes = [inv.get(i, str(i)) for i in range(int(y.max())+1)]
            elif isinstance(label_map, dict):
                classes = list(label_map.keys())
            else:
                classes = [str(i) for i in range(int(y.max())+1)]
        except Exception:
            classes = [str(i) for i in range(int(y.max())+1)]
        counts = np.bincount(y, minlength=len(classes))
        ax.bar(range(len(counts)), counts, color="#5470c6")
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(classes, rotation=30, ha="right")
        ax.set_ylabel("samples")
        ratio = counts.max()/max(counts.min(), 1)
        ax.set_title(f"{title} — N={len(y)}, max:min = {ratio:.0f}:1")
        for i, c in enumerate(counts):
            ax.text(i, c+max(counts)*0.01, str(c), ha="center", fontsize=8)
    # Hide unused axes
    for j in range(len(pairs), len(axes_flat)):
        axes_flat[j].axis("off")
    plt.tight_layout()
    out = FIG_ROOT / "dataset" / "class_distribution.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  wrote {out.relative_to(PROJECT)}")


# ============================================================
# 12. Primer vs benchmark (B1, MP source) — per benchmark
# ============================================================
METHOD_SPECS = [
    ("FCNN raw_136 MP", "lm", dict(feature="raw_136", source="MP", arch="fcnn")),
    ("CNN1D raw_136 MP", "lm", dict(feature="raw_136", source="MP", arch="cnn1d")),
    ("FCNN facs_28 MP", "lm", dict(feature="facs_28", source="MP", arch="fcnn")),
    ("CNN1D facs_28 MP", "lm", dict(feature="facs_28", source="MP", arch="cnn1d")),
    ("CNN scratch", "img", dict(arch="cnn_scratch")),
    ("CNN_TL", "img", dict(arch="cnn_tl")),
]


def _pick_mf1(rows_lm, rows_img, kind, spec, scheme):
    src = rows_lm if kind == "lm" else rows_img
    return next((r["mf1"] for r in src
                 if all(r.get(k) == v for k, v in spec.items())
                 and r["scenario"] == "B1" and r["scheme"] == scheme), None)


def fig_primer_vs_benchmark(scheme, bname, bcolor):
    runs = benchmark_runs[bname][scheme]
    if not runs:
        return
    p_lm = collect_landmark_rows(primer[scheme])
    p_img = collect_image_rows(primer[scheme])
    b_lm = collect_landmark_rows(runs)
    b_img = collect_image_rows(runs)
    p_vals, b_vals, labels = [], [], []
    for name, kind, spec in METHOD_SPECS:
        pv = _pick_mf1(p_lm, p_img, kind, spec, scheme)
        bv = _pick_mf1(b_lm, b_img, kind, spec, scheme)
        if pv is not None or bv is not None:
            p_vals.append(pv); b_vals.append(bv); labels.append(name)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels)); width = 0.4
    ax.bar(x - width/2, [v if v is not None else 0 for v in p_vals], width,
           label="Primer", color="#3b7dd8")
    ax.bar(x + width/2, [v if v is not None else 0 for v in b_vals], width,
           label=bname, color=bcolor)
    for i, (pv, bv) in enumerate(zip(p_vals, b_vals)):
        if pv is not None:
            ax.text(i - width/2, pv + 0.005, f"{pv:.3f}", ha="center", fontsize=7)
        if bv is not None:
            ax.text(i + width/2, bv + 0.005, f"{bv:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("test macro_f1 (B1, MP source)")
    ax.set_title(f"Cross-dataset: Primer vs {bname} — {scheme}, B1")
    ax.legend()
    plt.tight_layout()
    slug = bname.lower().replace("-", "").replace("+", "plus")
    out = FIG_ROOT / "comparisons" / f"primer_vs_{slug}_{scheme}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  wrote {out.relative_to(PROJECT)}")


def fig_primer_vs_all_benchmarks(scheme):
    """Combined plot: Primer vs ALL benchmarks side-by-side."""
    p_lm = collect_landmark_rows(primer[scheme])
    p_img = collect_image_rows(primer[scheme])
    available = [(n, c) for n, _, _, c in BENCHMARKS if benchmark_runs[n][scheme]]
    if not available:
        return
    n_groups = 1 + len(available)
    p_vals_all, b_vals_all, labels = [], {n: [] for n, _ in available}, []
    for name, kind, spec in METHOD_SPECS:
        pv = _pick_mf1(p_lm, p_img, kind, spec, scheme)
        bvs = {}
        for bname, _ in available:
            runs = benchmark_runs[bname][scheme]
            bvs[bname] = _pick_mf1(collect_landmark_rows(runs), collect_image_rows(runs),
                                    kind, spec, scheme)
        if pv is not None or any(v is not None for v in bvs.values()):
            p_vals_all.append(pv); labels.append(name)
            for bname, _ in available:
                b_vals_all[bname].append(bvs[bname])
    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(len(labels)); width = 0.8 / n_groups
    ax.bar(x - (n_groups - 1) * width / 2, [v if v is not None else 0 for v in p_vals_all],
           width, label="Primer", color="#3b7dd8")
    for i, (bname, bcolor) in enumerate(available):
        offset = (i + 1 - (n_groups - 1) / 2) * width
        ax.bar(x + offset, [v if v is not None else 0 for v in b_vals_all[bname]],
               width, label=bname, color=bcolor)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("test macro_f1 (B1, MP source)")
    ax.set_title(f"Cross-dataset Comparison: Primer vs all benchmarks — {scheme}, B1")
    ax.legend(ncol=min(n_groups, 5), fontsize=8)
    plt.tight_layout()
    out = FIG_ROOT / "comparisons" / f"primer_vs_all_benchmarks_{scheme}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  wrote {out.relative_to(PROJECT)}")


def fig_benchmark_leaderboard(scheme, top_n=5):
    """Best unimodal model per benchmark + Primer (top-N by macro_f1, B1)."""
    datasets = [("Primer", primer[scheme])]
    for bname, _, _, _ in BENCHMARKS:
        if benchmark_runs[bname][scheme]:
            datasets.append((bname, benchmark_runs[bname][scheme]))
    if len(datasets) < 2:
        return
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.2*len(datasets), 5), squeeze=False)
    axes = axes[0]
    for ax, (dname, runs) in zip(axes, datasets):
        lm = collect_landmark_rows(runs)
        img = collect_image_rows(runs)
        rows = []
        for r in lm:
            if r["scenario"] == "B1" and r["scheme"] == scheme and r["mf1"] is not None:
                label = f"{r['arch'].upper()} {r['feature']}/{r['source']}"
                rows.append((label, r["mf1"]))
        for r in img:
            if r["scenario"] == "B1" and r["scheme"] == scheme and r["mf1"] is not None:
                rows.append((r["arch"].upper(), r["mf1"]))
        rows.sort(key=lambda t: -t[1])
        rows = rows[:top_n]
        if not rows:
            ax.set_title(f"{dname} (no data)"); continue
        ys = np.arange(len(rows))[::-1]
        labels = [t[0] for t in rows]
        vals = [t[1] for t in rows]
        ax.barh(ys, vals, color="#3b7dd8" if dname == "Primer" else "#73c0de")
        ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=8)
        for y, v in zip(ys, vals):
            ax.text(v + 0.005, y, f"{v:.3f}", va="center", fontsize=7)
        ax.set_xlim(0, max(vals) * 1.18)
        ax.set_title(f"{dname} — top-{len(rows)} ({scheme}, B1)")
        ax.set_xlabel("test macro_f1")
    plt.tight_layout()
    out = FIG_ROOT / "leaderboards" / f"benchmark_top{top_n}_{scheme}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  wrote {out.relative_to(PROJECT)}")


# ============================================================
# RQ1: Modality contribution — CNN-image vs FCNN-landmark (raw_136)
# Catatan: RQ1 hanya membandingkan modality dasar (citra vs landmark
# raw coordinates). Dekomposisi fitur (FACS / Blendshape) dibahas di RQ2.
# ============================================================
def fig_rq1_modality_contribution(scheme):
    """RQ1 — Unimodal raw feature: image (CNN_SCRATCH + CNN_TL) vs landmark
    raw_136 (FCNN + CNN1D × MP + FA). 6 kategori × 3 scenario.
    """
    lm = collect_landmark_rows(primer[scheme])
    img = collect_image_rows(primer[scheme])
    scenarios = ["B1", "B2", "B3"]

    # 6 categories
    categories = [
        ("CNN_SCRATCH (image)",       "img", dict(arch="cnn_scratch"),               "#ee6666"),
        ("CNN_TL (image)",            "img", dict(arch="cnn_tl"),                    "#a4262c"),
        ("FCNN raw_136 (MP)",         "lm",  dict(feature="raw_136", arch="fcnn",  source="MP"),  "#91cc75"),
        ("CNN1D raw_136 (MP)",        "lm",  dict(feature="raw_136", arch="cnn1d", source="MP"),  "#3ba272"),
        ("FCNN raw_136 (FA)",         "lm",  dict(feature="raw_136", arch="fcnn",  source="FA"),  "#5470c6"),
        ("CNN1D raw_136 (FA)",        "lm",  dict(feature="raw_136", arch="cnn1d", source="FA"),  "#3b7dd8"),
    ]
    n_cat = len(categories)
    width = 0.8 / n_cat
    x = np.arange(len(scenarios))

    fig, ax = plt.subplots(figsize=(13, 5.6))
    all_vals = {}
    for ci, (name, kind, spec, color) in enumerate(categories):
        vals = []
        for scn in scenarios:
            src = lm if kind == "lm" else img
            v = next((r["mf1"] for r in src
                      if all(r.get(k) == v for k, v in spec.items())
                      and r["scenario"] == scn and r["scheme"] == scheme
                      and r["mf1"] is not None), None)
            vals.append(v)
        offset = (ci - (n_cat - 1) / 2) * width
        ax.bar(x + offset, [v or 0 for v in vals], width, label=name, color=color)
        for i, v in enumerate(vals):
            if v is not None:
                ax.text(x[i] + offset, v + 0.003, f"{v:.3f}",
                        ha="center", fontsize=6, rotation=90)
        all_vals[name] = vals

    ax.set_xticks(x); ax.set_xticklabels(scenarios)
    ax.set_xlabel("Scenario")
    ax.set_ylabel("test macro_f1")
    ax.set_title(f"RQ1: Unimodal raw feature — Image (CNN/CNN_TL) vs Landmark raw_136 (FCNN/CNN1D × MP/FA) — {scheme}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=3, fontsize=8)
    plt.tight_layout()
    out = FIG_ROOT / "comparisons" / f"rq1_modality_contribution_{scheme}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  wrote {out.relative_to(PROJECT)}")


def rq1_summary_table():
    """Mean + max mf1 untuk image branch vs landmark raw_136 branch (semua arch × source)."""
    out = {"scheme": [], "modality": [], "best_subgroup": [],
           "mean_mf1": [], "max_mf1": [], "n_runs": []}
    for scheme in ("3c", "7c"):
        # Landmark raw_136 — all arch × source
        lm_rows = [r for r in collect_landmark_rows(primer[scheme])
                   if r["feature"] == "raw_136" and r["scheme"] == scheme
                   and r["mf1"] is not None]
        img_rows = [r for r in collect_image_rows(primer[scheme])
                    if r["scheme"] == scheme and r["mf1"] is not None]
        if lm_rows:
            best = max(lm_rows, key=lambda r: r["mf1"])
            sub = f"{best['arch'].upper()} {best['source']}/{best['scenario']}"
            out["scheme"].append(scheme); out["modality"].append("Landmark raw_136 (FCNN+CNN1D × MP+FA)")
            out["best_subgroup"].append(sub)
            out["mean_mf1"].append(float(np.mean([r["mf1"] for r in lm_rows])))
            out["max_mf1"].append(float(best["mf1"])); out["n_runs"].append(len(lm_rows))
        if img_rows:
            best = max(img_rows, key=lambda r: r["mf1"])
            sub = f"{best['arch'].upper()}/{best['scenario']}"
            out["scheme"].append(scheme); out["modality"].append("Image (CNN_SCRATCH+CNN_TL)")
            out["best_subgroup"].append(sub)
            out["mean_mf1"].append(float(np.mean([r["mf1"] for r in img_rows])))
            out["max_mf1"].append(float(best["mf1"])); out["n_runs"].append(len(img_rows))
    return out


# ============================================================
# RQ2: Feature decomposition — Δ vs raw_136 baseline (per source)
# ============================================================
def fig_rq2_feature_decomposition_delta(scheme):
    """Bar chart: absolute Δ macro_f1 facs_28/blendshape_52/fb80 vs raw_136 (per source).

    Mean over arch & scenario, dipisah per source. Positive = improvement.
    """
    lm = collect_landmark_rows(primer[scheme])
    features = [("facs_28", "FACS_28"), ("blendshape_52", "Blendshape_52"),
                ("facs_plus_bs_80", "FACS+BS_80")]
    sources = [("MP", "MediaPipe"), ("FA", "face-api.js")]

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.35
    x = np.arange(len(features))
    for si, (src, src_lbl) in enumerate(sources):
        baseline_vals = [r["mf1"] for r in lm
                         if r["feature"] == "raw_136" and r["source"] == src
                         and r["scheme"] == scheme and r["mf1"] is not None]
        if not baseline_vals:
            continue
        baseline_mean = np.mean(baseline_vals)
        deltas, abs_vals = [], []
        for feat, _ in features:
            vals = [r["mf1"] for r in lm
                    if r["feature"] == feat and r["source"] == src
                    and r["scheme"] == scheme and r["mf1"] is not None]
            if vals:
                m = np.mean(vals)
                deltas.append(m - baseline_mean); abs_vals.append(m)
            else:
                deltas.append(None); abs_vals.append(None)
        bars = ax.bar(x + (si - 0.5) * width,
                      [d if d is not None else 0 for d in deltas],
                      width,
                      label=f"{src_lbl} (raw_136 baseline = {baseline_mean:.4f})",
                      color="#3b7dd8" if src == "FA" else "#91cc75")
        for i, (d, a) in enumerate(zip(deltas, abs_vals)):
            if d is None: continue
            color = "#1a7f37" if d > 0 else "#cf222e"
            ax.text(i + (si - 0.5) * width, d + (0.002 if d >= 0 else -0.005),
                    f"Δ={d:+.4f}\n({a:.4f})", ha="center",
                    va="bottom" if d >= 0 else "top", fontsize=7, color=color)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f[1] for f in features])
    ax.set_ylabel("Δ macro_f1 vs raw_136 (mean over arch × scenario)")
    ax.set_title(f"RQ2: Feature decomposition — Δ vs raw_136 baseline — {scheme}")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    out = FIG_ROOT / "comparisons" / f"rq2_feature_decomposition_delta_{scheme}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  wrote {out.relative_to(PROJECT)}")


def rq2_summary_table():
    """Return mean mf1 per (feature, source) and Δ vs raw_136 baseline."""
    out = {"scheme": [], "feature": [], "source": [],
           "mean_mf1": [], "delta_vs_raw": [], "n_runs": []}
    for scheme in ("3c", "7c"):
        lm = collect_landmark_rows(primer[scheme])
        for src in ("MP", "FA"):
            base_vals = [r["mf1"] for r in lm
                         if r["feature"] == "raw_136" and r["source"] == src
                         and r["scheme"] == scheme and r["mf1"] is not None]
            base = np.mean(base_vals) if base_vals else None
            for feat in ("raw_136", "facs_28", "blendshape_52", "facs_plus_bs_80"):
                vals = [r["mf1"] for r in lm
                        if r["feature"] == feat and r["source"] == src
                        and r["scheme"] == scheme and r["mf1"] is not None]
                if not vals: continue
                m = np.mean(vals)
                out["scheme"].append(scheme)
                out["feature"].append(feat); out["source"].append(src)
                out["mean_mf1"].append(m)
                out["delta_vs_raw"].append(m - base if base is not None else None)
                out["n_runs"].append(len(vals))
    return out


# ============================================================
# Multi-metric helpers (accuracy + macro_f1 + weighted_f1 + micro_f1)
# ============================================================
def get_test_metrics(run):
    t = run.get("test", {})
    return {"accuracy": t.get("accuracy"), "macro_f1": t.get("macro_f1"),
            "weighted_f1": t.get("weighted_f1"), "micro_f1": t.get("micro_f1")}


def get_inference(run):
    t = run.get("test", {})
    return {"time_sec": t.get("inference_time_sec"),
            "throughput": t.get("inference_throughput_samples_per_sec")}


def get_per_class_report(run):
    """Return dict class_name -> {precision, recall, f1, support}, dropping aggregates."""
    rep = run.get("test", {}).get("classification_report", {})
    drop = {"accuracy", "macro avg", "weighted avg"}
    out = {}
    for k, v in rep.items():
        if k in drop or not isinstance(v, dict):
            continue
        out[k] = {"precision": v.get("precision"), "recall": v.get("recall"),
                  "f1": v.get("f1-score"), "support": v.get("support")}
    return out


# ============================================================
# Multi-metric summary — accuracy / macro_f1 / weighted_f1
# (RQ1: best run per kategori; RQ2/RQ3 sebagai pelengkap)
# ============================================================
def fig_multi_metric_unimodal(scheme):
    """Best run per kategori unimodal × 3 metrik (acc, mf1, wf1)."""
    lm = collect_landmark_rows(primer[scheme])
    img = collect_image_rows(primer[scheme])
    categories = [
        ("CNN_SCRATCH (image)", "img", dict(arch="cnn_scratch")),
        ("CNN_TL (image)",      "img", dict(arch="cnn_tl")),
        ("FCNN raw_136 MP",     "lm",  dict(feature="raw_136", arch="fcnn",  source="MP")),
        ("CNN1D raw_136 MP",    "lm",  dict(feature="raw_136", arch="cnn1d", source="MP")),
        ("FCNN raw_136 FA",     "lm",  dict(feature="raw_136", arch="fcnn",  source="FA")),
        ("CNN1D raw_136 FA",    "lm",  dict(feature="raw_136", arch="cnn1d", source="FA")),
        ("FCNN facs_28 FA",     "lm",  dict(feature="facs_28", arch="fcnn",  source="FA")),
        ("FCNN fb80 FA",        "lm",  dict(feature="facs_plus_bs_80", arch="fcnn", source="FA")),
    ]
    metrics_keys = ["accuracy", "macro_f1", "weighted_f1"]
    rows = []
    for name, kind, spec in categories:
        src_rows = lm if kind == "lm" else img
        # Best run by macro_f1 across all scenarios (for this category)
        cands = [r for r in src_rows if all(r.get(k) == v for k, v in spec.items())
                 and r["scheme"] == scheme and r["mf1"] is not None]
        if not cands: continue
        best = max(cands, key=lambda r: r["mf1"])
        m = get_test_metrics(best["run"])
        rows.append({"name": name, "scenario": best["scenario"],
                     **{k: m.get(k) for k in metrics_keys}})
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(rows))
    width = 0.27
    colors = ["#5470c6", "#91cc75", "#fac858"]
    for i, mk in enumerate(metrics_keys):
        vals = [r[mk] if r[mk] is not None else 0 for r in rows]
        ax.bar(x + (i - 1) * width, vals, width, label=mk, color=colors[i])
        for j, v in enumerate(vals):
            if v:
                ax.text(x[j] + (i - 1) * width, v + 0.005, f"{v:.3f}",
                        ha="center", fontsize=6, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['name']}\n(best@{r['scenario']})" for r in rows],
                       rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("metric value")
    ax.set_title(f"Multi-metric (accuracy + macro_f1 + weighted_f1) — best per kategori unimodal — {scheme}")
    ax.legend()
    plt.tight_layout()
    out = FIG_ROOT / "comparisons" / f"multi_metric_unimodal_{scheme}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  wrote {out.relative_to(PROJECT)}")


# ============================================================
# Inference throughput per arch (samples/sec)
# ============================================================
def fig_inference_throughput(scheme):
    lm = collect_landmark_rows(primer[scheme])
    img = collect_image_rows(primer[scheme])
    items = []
    # Image archs
    for arch in ("cnn_scratch", "cnn_tl"):
        cands = [r for r in img if r["arch"] == arch and r["scheme"] == scheme
                 and r["mf1"] is not None]
        ths = [get_inference(r["run"])["throughput"] for r in cands]
        ths = [t for t in ths if t is not None]
        if ths:
            items.append((arch.upper().replace("_", " "), float(np.mean(ths)), len(ths)))
    # Landmark: per arch × source × feature
    seen = set()
    for r in lm:
        if r["mf1"] is None or r["scheme"] != scheme: continue
        key = (r["feature"], r["source"], r["arch"])
        seen.add(key)
    for feat, src, arch in sorted(seen):
        cands = [r for r in lm if r["feature"] == feat and r["source"] == src
                 and r["arch"] == arch and r["scheme"] == scheme and r["mf1"] is not None]
        ths = [get_inference(r["run"])["throughput"] for r in cands]
        ths = [t for t in ths if t is not None]
        if ths:
            items.append((f"{arch.upper()} {feat} {src}", float(np.mean(ths)), len(ths)))
    items.sort(key=lambda t: -t[1])

    fig, ax = plt.subplots(figsize=(11, 0.32 * len(items) + 1.5))
    labels = [t[0] for t in items]
    vals = [t[1] for t in items]
    ys = np.arange(len(labels))[::-1]
    bars = ax.barh(ys, vals, color=["#ee6666" if "CNN" in lbl and "1D" not in lbl else "#5470c6"
                                     for lbl in labels])
    for y, v, (_, _, n) in zip(ys, vals, items):
        ax.text(v + max(vals)*0.005, y, f"{v:,.0f} (n={n})", va="center", fontsize=7)
    ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("inference throughput (samples/sec) — mean across scenarios")
    ax.set_xlim(0, max(vals) * 1.18)
    ax.set_title(f"Inference throughput per arch (unimodal) — Primer {scheme}")
    plt.tight_layout()
    out = FIG_ROOT / "resources" / f"inference_throughput_{scheme}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  wrote {out.relative_to(PROJECT)}")


# ============================================================
# Per-class precision/recall/f1 — top model per RQ kategori
# ============================================================
def fig_per_class_metrics_top(scheme):
    """Heatmap per-class (precision, recall, f1) untuk model terbaik per RQ kategori.
    Kategori (top-1 by mf1 di B1):
      - RQ1 best image arch
      - RQ1 best landmark raw_136 (across arch × source)
      - RQ2 best landmark FACS_28 / Blendshape_52 / FB80
    """
    lm = collect_landmark_rows(primer[scheme])
    img = collect_image_rows(primer[scheme])
    targets = []

    img_cands = [r for r in img if r["scheme"] == scheme and r["mf1"] is not None]
    if img_cands:
        b = max(img_cands, key=lambda r: r["mf1"])
        targets.append((f"[RQ1] Image {b['arch'].upper()} (B{b['scenario'][-1]})", b))
    lm_raw = [r for r in lm if r["feature"] == "raw_136"
              and r["scheme"] == scheme and r["mf1"] is not None]
    if lm_raw:
        b = max(lm_raw, key=lambda r: r["mf1"])
        targets.append((f"[RQ1] LM raw_136 {b['arch'].upper()} {b['source']} (B{b['scenario'][-1]})", b))
    for feat in ("facs_28", "blendshape_52", "facs_plus_bs_80"):
        cands = [r for r in lm if r["feature"] == feat
                 and r["scheme"] == scheme and r["mf1"] is not None]
        if cands:
            b = max(cands, key=lambda r: r["mf1"])
            tag = "RQ2"
            targets.append((f"[{tag}] LM {feat} {b['arch'].upper()} {b['source']} (B{b['scenario'][-1]})", b))
    if not targets:
        return
    # Build matrix: rows = (model, metric) — 3 metrics per model; cols = classes
    all_classes = []
    for label, rec in targets:
        rep = get_per_class_report(rec["run"])
        for cls in rep:
            if cls not in all_classes: all_classes.append(cls)
    rows_lbl = []
    M = []
    for label, rec in targets:
        rep = get_per_class_report(rec["run"])
        for mk, mk_pretty in [("precision", "P"), ("recall", "R"), ("f1", "F1")]:
            rows_lbl.append(f"{label} · {mk_pretty}")
            M.append([rep.get(c, {}).get(mk, np.nan) for c in all_classes])
    M = np.array(M, dtype=float)

    fig, ax = plt.subplots(figsize=(max(8, 0.9*len(all_classes)+3),
                                    0.32*len(rows_lbl)+2))
    im = ax.imshow(M, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(all_classes))); ax.set_xticklabels(all_classes, rotation=25, ha="right")
    ax.set_yticks(range(len(rows_lbl))); ax.set_yticklabels(rows_lbl, fontsize=7)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="black" if v > 0.5 else "white")
    ax.set_title(f"Per-class Precision / Recall / F1 — top models per RQ kategori — {scheme}")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    plt.tight_layout()
    out = FIG_ROOT / "per_class" / f"per_class_metrics_top_{scheme}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  wrote {out.relative_to(PROJECT)}")


# ============================================================
# Run all
# ============================================================
for scheme in ("3c", "7c"):
    fig_master_heatmap(scheme)
    fig_top10(scheme)
    fig_scenario_compare(scheme)
    fig_fa_vs_mp(scheme)
    fig_feature_compare(scheme)
    fig_arch_compare(scheme)
    fig_confusion_top(scheme)
    fig_per_class_f1(scheme)
    fig_training_curves(scheme)
    fig_resources(scheme)
    fig_rq1_modality_contribution(scheme)
    fig_rq2_feature_decomposition_delta(scheme)
    fig_multi_metric_unimodal(scheme)
    fig_inference_throughput(scheme)
    fig_per_class_metrics_top(scheme)
    for bname, _, _, bcolor in BENCHMARKS:
        fig_primer_vs_benchmark(scheme, bname, bcolor)
    fig_primer_vs_all_benchmarks(scheme)
    fig_benchmark_leaderboard(scheme, top_n=5)

fig_class_distribution()

print(f"\nAll figures written to {FIG_ROOT.relative_to(PROJECT)}")
