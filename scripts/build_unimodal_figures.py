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
KDEF = PROJECT / "models" / "benchmark" / "kdef_7class"

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

kdef = {}
for scheme, sk in [("3class", "3c"), ("7class", "7c")]:
    kdef[sk] = load_results_dir(KDEF / scheme / "Unified")
    print(f"KDEF {sk}: {len(kdef[sk])} unimodal runs")


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
        ("KDEF 7c (train)", PROJECT/"data/benchmark/kdef_7class/y_train.npy",
         PROJECT/"data/benchmark/kdef_7class/label_map.json"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax, (title, ypath, lm_file) in zip(axes, pairs):
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
    plt.tight_layout()
    out = FIG_ROOT / "dataset" / "class_distribution.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  wrote {out.relative_to(PROJECT)}")


# ============================================================
# 12. Primer vs KDEF (B1, MP source)
# ============================================================
def fig_primer_vs_kdef(scheme):
    if not kdef[scheme]:
        return
    p_lm = collect_landmark_rows(primer[scheme])
    p_img = collect_image_rows(primer[scheme])
    k_lm = collect_landmark_rows(kdef[scheme])
    k_img = collect_image_rows(kdef[scheme])
    method_specs = [
        ("FCNN raw_136 MP", "lm", dict(feature="raw_136", source="MP", arch="fcnn")),
        ("CNN1D raw_136 MP", "lm", dict(feature="raw_136", source="MP", arch="cnn1d")),
        ("FCNN facs_28 MP", "lm", dict(feature="facs_28", source="MP", arch="fcnn")),
        ("CNN1D facs_28 MP", "lm", dict(feature="facs_28", source="MP", arch="cnn1d")),
        ("CNN scratch", "img", dict(arch="cnn_scratch")),
        ("CNN_TL", "img", dict(arch="cnn_tl")),
    ]
    p_vals, k_vals, labels = [], [], []
    for name, kind, spec in method_specs:
        p_src = p_lm if kind == "lm" else p_img
        k_src = k_lm if kind == "lm" else k_img
        pv = next((r["mf1"] for r in p_src
                   if all(r.get(k) == v for k, v in spec.items())
                   and r["scenario"] == "B1" and r["scheme"] == scheme), None)
        kv = next((r["mf1"] for r in k_src
                   if all(r.get(k) == v for k, v in spec.items())
                   and r["scenario"] == "B1" and r["scheme"] == scheme), None)
        if pv is not None or kv is not None:
            p_vals.append(pv); k_vals.append(kv); labels.append(name)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels)); width = 0.4
    ax.bar(x - width/2, [v if v is not None else 0 for v in p_vals], width,
           label="Primer", color="#3b7dd8")
    ax.bar(x + width/2, [v if v is not None else 0 for v in k_vals], width,
           label="KDEF", color="#91cc75")
    for i, (pv, kv) in enumerate(zip(p_vals, k_vals)):
        if pv is not None:
            ax.text(i - width/2, pv + 0.005, f"{pv:.3f}", ha="center", fontsize=7)
        if kv is not None:
            ax.text(i + width/2, kv + 0.005, f"{kv:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("test macro_f1 (B1, MP source)")
    ax.set_title(f"Cross-dataset Comparison: Primer vs KDEF — {scheme}, B1")
    ax.legend()
    plt.tight_layout()
    out = FIG_ROOT / "comparisons" / f"primer_vs_kdef_{scheme}.png"
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
    fig_primer_vs_kdef(scheme)

fig_class_distribution()

print(f"\nAll figures written to {FIG_ROOT.relative_to(PROJECT)}")
