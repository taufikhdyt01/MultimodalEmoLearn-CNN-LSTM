#!/usr/bin/env python3
"""Render figur robustness (PNG + PDF) untuk laporan tesis.

Output ke docs/figures/robustness/:
  - macro_f1_per_config.{png,pdf}      bar mean±std macro_f1 per config × strategi
  - loso_boxplot_per_subject.{png,pdf} distribusi macro_f1 antar 37 subjek (LOSO)
  - strategy_comparison_per_scheme.{png,pdf}
  - stability_std_per_config.{png,pdf}
  - multimetric_grid.{png,pdf}         macro/weighted/accuracy × 3 strategi (grid)

Semua angka dari models/frontonly_conf60/robustness/{loso,cv5,randomsplit}/*.json.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
ROBUST = PROJECT / "models" / "frontonly_conf60" / "robustness"
OUT = PROJECT / "docs" / "figures" / "robustness"
OUT.mkdir(parents=True, exist_ok=True)

STRATS = [
    ("loso", "LOSO", "subjek"),
    ("cv5", "5-Fold CV", "fold"),
    ("randomsplit", "Random Split", "seed"),
]
COLORS = {"loso": "#4C72B0", "cv5": "#55A868", "randomsplit": "#C44E52"}
CONFIG_ORDER = list("ABCDEF")

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})


def load(strat):
    d = ROBUST / strat
    summ, perfold = {}, {}
    for sf in sorted(d.glob("?_*_summary.json")):
        s = json.load(open(sf))
        summ[s["config_key"]] = s
        perfold[s["config_key"]] = json.load(
            open(str(sf).replace("_summary.json", "_per_fold.json")))
    return summ, perfold


DATA = {st: load(st) for st, _, _ in STRATS}
SCHEME = {c: DATA["loso"][0][c]["config"]["scheme"] for c in CONFIG_ORDER}
LABELS = [f"{c}\n({SCHEME[c]})" for c in CONFIG_ORDER]
x = np.arange(len(CONFIG_ORDER))
w = 0.26


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("  wrote", name + ".{png,pdf}")


def arr(strat, metric):
    s = DATA[strat][0]
    mean = np.array([s[c][f"test_{metric}_mean"] for c in CONFIG_ORDER])
    std = np.array([s[c][f"test_{metric}_std"] for c in CONFIG_ORDER])
    return mean, std


# 1 — macro_f1 per config × strategi
fig, ax = plt.subplots(figsize=(11, 5))
for i, (st, title, _) in enumerate(STRATS):
    m, s = arr(st, "macro_f1")
    ax.bar(x + (i - 1) * w, m, w, yerr=s, capsize=3, label=title,
           color=COLORS[st], alpha=0.88)
ax.set_xticks(x); ax.set_xticklabels(LABELS)
ax.set_ylabel("macro_f1 (mean ± std)"); ax.set_xlabel("Config (scheme)")
ax.set_title("Robustness: macro_f1 per config across validation strategies")
ax.legend(fontsize=9)
save(fig, "macro_f1_per_config")

# 2 — LOSO boxplot per subjek
fig, ax = plt.subplots(figsize=(10, 5))
_, pf = DATA["loso"]
dist = [[r["macro_f1"] for r in pf[c] if "error" not in r] for c in CONFIG_ORDER]
ax.boxplot(dist, tick_labels=CONFIG_ORDER, showmeans=True)
ax.set_ylabel("macro_f1 (per subjek)"); ax.set_xlabel("Config")
ax.set_title("LOSO — distribusi macro_f1 antar 37 subjek")
save(fig, "loso_boxplot_per_subject")

# 3 — strategy comparison per scheme
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, sch in zip(axes, ["3c", "7c"]):
    vals, errs, names = [], [], []
    for st, title, _ in STRATS:
        m, _ = arr(st, "macro_f1")
        sub = np.array([m[i] for i, c in enumerate(CONFIG_ORDER) if SCHEME[c] == sch])
        vals.append(sub.mean()); errs.append(sub.std()); names.append(title)
    bars = ax.bar(names, vals, yerr=errs, capsize=4,
                  color=[COLORS[s] for s, _, _ in STRATS], alpha=0.88)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", fontsize=9)
    ax.set_title(f"{sch} — mean macro_f1 antar config")
axes[0].set_ylabel("macro_f1")
save(fig, "strategy_comparison_per_scheme")

# 4 — stability (std) per config
fig, ax = plt.subplots(figsize=(11, 4.5))
for i, (st, title, _) in enumerate(STRATS):
    _, s = arr(st, "macro_f1")
    ax.bar(x + (i - 1) * w, s, w, label=title, color=COLORS[st], alpha=0.88)
ax.set_xticks(x); ax.set_xticklabels(LABELS)
ax.set_ylabel("std macro_f1 (lower = more stable)"); ax.set_xlabel("Config (scheme)")
ax.set_title("Robustness: variansi macro_f1 lintas fold")
ax.legend(fontsize=9)
save(fig, "stability_std_per_config")

# 5 — multi-metric grid (3 metrik × 3 strategi)
metrics = [("macro_f1", "macro_f1"), ("weighted_f1", "weighted_f1"),
           ("accuracy", "accuracy")]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
for ax, (mk, mlabel) in zip(axes, metrics):
    for i, (st, title, _) in enumerate(STRATS):
        m, s = arr(st, mk)
        ax.bar(x + (i - 1) * w, m, w, yerr=s, capsize=2, label=title,
               color=COLORS[st], alpha=0.88)
    ax.set_xticks(x); ax.set_xticklabels(LABELS)
    ax.set_title(mlabel); ax.set_xlabel("Config")
axes[0].set_ylabel("score (mean ± std)")
axes[0].legend(fontsize=8)
fig.suptitle("Robustness multi-metric — macro_f1 / weighted_f1 / accuracy", y=1.02)
save(fig, "multimetric_grid")

print("DONE ->", OUT)
