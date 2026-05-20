"""
Statistical Significance Tests — Late Fusion vs Best Unimodal
==============================================================

Post-hoc analysis menggunakan softmax prediction cache yang sudah ada di
`models/{frontonly_conf60,benchmark/*}/late_fusion_cache/*_test.npy` + late
fusion result records (yang menyimpan `best_image_weight`, `best_landmark_weight`,
`image_branch`, `landmark_branch`).

Tidak perlu retrain — Late Fusion test predictions di-rekonstruksi dari unimodal
softmax + bobot optimal. Lalu hitung:
  1. McNemar's test (paired) — exact / continuity-corrected
  2. Bootstrap 95% CI untuk Δ-macro-F1 antara dua model

Output:
    docs/significance_tests.md
    docs/significance_tests.json

Usage:
    python scripts/compute_significance_tests.py
    python scripts/compute_significance_tests.py --dataset kdef_7class
    python scripts/compute_significance_tests.py --n-boot 2000
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import binom, chi2
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Label remap: 7-class index → 3-class (positive/neutral/negative)
# REMAP_3[7c_label] = 3c_label
#   7c: 0 neutral, 1 happy, 2 sad, 3 angry, 4 fearful, 5 disgusted, 6 surprised
#   3c: 0 positive, 1 neutral, 2 negative
REMAP_3 = np.array([1, 0, 2, 2, 2, 2, 0], dtype=np.int64)


def remap_labels(y_raw: np.ndarray, scheme: str) -> np.ndarray:
    """Apply REMAP_3 if scheme=3c, else return as-is."""
    if scheme == "3c":
        return REMAP_3[y_raw]
    return y_raw

# Cache locations: (dataset_label, cache_dir, y_test_path, late_fusion_glob)
DATASETS = [
    ("primer_conf60",
     PROJECT_ROOT / "models" / "frontonly_conf60" / "late_fusion_cache",
     PROJECT_ROOT / "data" / "dataset_frontonly_conf60" / "y_test.npy",
     PROJECT_ROOT / "models" / "frontonly_conf60"),
    ("kdef_7class",
     PROJECT_ROOT / "models" / "benchmark" / "kdef_7class" / "late_fusion_cache",
     PROJECT_ROOT / "data" / "benchmark" / "kdef_7class" / "y_test.npy",
     PROJECT_ROOT / "models" / "benchmark" / "kdef_7class"),
    ("rafdb_7class",
     PROJECT_ROOT / "models" / "benchmark" / "rafdb_7class" / "late_fusion_cache",
     PROJECT_ROOT / "data" / "benchmark" / "rafdb_7class" / "y_test.npy",
     PROJECT_ROOT / "models" / "benchmark" / "rafdb_7class"),
    ("ckplus_7class",
     PROJECT_ROOT / "models" / "benchmark" / "ckplus_7class" / "late_fusion_cache",
     PROJECT_ROOT / "data" / "benchmark" / "ckplus_7class" / "y_test.npy",
     PROJECT_ROOT / "models" / "benchmark" / "ckplus_7class"),
    ("jaffe_7class",
     PROJECT_ROOT / "models" / "benchmark" / "jaffe_7class" / "late_fusion_cache",
     PROJECT_ROOT / "data" / "benchmark" / "jaffe_7class" / "y_test.npy",
     PROJECT_ROOT / "models" / "benchmark" / "jaffe_7class"),
]


# ── Statistical tests ─────────────────────────────────────

def mcnemar_test(y_true, pred_a, pred_b):
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)
    b = int(np.sum(correct_a & ~correct_b))   # a right, b wrong
    c = int(np.sum(~correct_a & correct_b))   # a wrong, b right
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n": 0, "statistic": float("nan"),
                "p_value": 1.0, "test": "none (identical errors)"}
    if n < 25:
        k = min(b, c)
        p = min(2 * binom.cdf(k, n, 0.5), 1.0)
        return {"b": b, "c": c, "n": n, "statistic": int(k),
                "p_value": float(p), "test": "exact_binomial"}
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p = 1 - chi2.cdf(stat, df=1)
    return {"b": b, "c": c, "n": n, "statistic": float(stat),
            "p_value": float(p), "test": "chi2_continuity"}


def bootstrap_delta_f1(y_true, pred_a, pred_b, n_boot=1000, seed=42, ci=95):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m_a = f1_score(y_true[idx], pred_a[idx], average="macro", zero_division=0)
        m_b = f1_score(y_true[idx], pred_b[idx], average="macro", zero_division=0)
        deltas[i] = m_b - m_a
    alpha = (100 - ci) / 2
    lo = float(np.percentile(deltas, alpha))
    hi = float(np.percentile(deltas, 100 - alpha))
    p = min(2 * min(float(np.mean(deltas <= 0)), float(np.mean(deltas >= 0))), 1.0)
    return {"mean": float(np.mean(deltas)), "ci_low": lo, "ci_high": hi,
            "p_bootstrap": p, "n_boot": n_boot}


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


# ── Cache + result discovery ──────────────────────────────

def load_unimodal_caches(cache_dir: Path, scenario: str, ns: str):
    """Return list of (key, softmax_array) for unimodal caches matching scenario+scheme."""
    out = []
    if not cache_dir.exists():
        return out
    for f in sorted(cache_dir.glob("*_test.npy")):
        key = f.stem.replace("_test", "")
        if f"_{scenario}_{ns}" in key and (key.startswith("image_") or key.startswith("landmark_")):
            try:
                arr = np.load(f).astype(np.float64)
                out.append((key, arr))
            except Exception:
                pass
    return out


def find_late_fusion_records(model_root: Path):
    """Yield (record_key, run_dict) for every fusion_late_* results.json under model_root."""
    for p in model_root.rglob("fusion_late_*/results.json"):
        try:
            j = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        runs = j.get("runs", {})
        for run_key, run in runs.items():
            run["_record_path"] = str(p)
            yield run_key, run


def reconstruct_late_fusion_preds(run: dict, cache_dir: Path):
    """Reconstruct late fusion softmax from cached unimodals + best weights.

    Cache key convention:
      image:    image_{arch}_{scenario}_{ns}_test.npy        (arch e.g. cnn_scratch, cnn_tl)
      landmark: landmark_{arch}_{scenario}_{ns}_test.npy     (arch e.g. fcnn_facs_28_fa, cnn1d_raw_136_mp)

    Returns (softmax_arr, image_branch_key, landmark_branch_key) or None if missing.
    """
    img_branch = run.get("image_branch") or {}
    lm_branch = run.get("landmark_branch") or {}
    w_img = run.get("best_image_weight")
    w_lm = run.get("best_landmark_weight")
    if w_img is None or w_lm is None:
        return None

    ns = f"{run.get('num_classes', '?')}c"
    sc = str(run.get("scenario", "?")).lower()  # B1 → b1

    img_arch = img_branch.get("arch")
    lm_arch = lm_branch.get("arch")
    if not img_arch or not lm_arch:
        return None

    img_key = f"image_{img_arch}_{sc}_{ns}"
    lm_key = f"landmark_{lm_arch}_{sc}_{ns}"

    img_path = cache_dir / f"{img_key}_test.npy"
    lm_path = cache_dir / f"{lm_key}_test.npy"
    if not (img_path.exists() and lm_path.exists()):
        return None
    img_soft = np.load(img_path).astype(np.float64)
    lm_soft = np.load(lm_path).astype(np.float64)
    if img_soft.shape != lm_soft.shape:
        return None
    fused = w_img * img_soft + w_lm * lm_soft
    return fused, img_key, lm_key


def parse_scenario_scheme(key: str):
    m = re.search(r"_(b[123])_([347]c)\b", key)
    return (m.group(1), m.group(2)) if m else (None, None)


# ── Main analysis ─────────────────────────────────────────

def analyze_dataset(label, cache_dir, y_test_path, model_root, n_boot=1000):
    if not y_test_path.exists():
        return {"error": f"y_test missing: {y_test_path}"}
    if not cache_dir.exists():
        return {"error": f"cache_dir missing: {cache_dir}"}

    y_raw = np.load(y_test_path)
    result = {"dataset": label, "y_test_n": int(len(y_raw)), "groups": {}}

    # Collect all late fusion candidates per (scenario, scheme)
    fusion_by_group = {}
    for run_key, run in find_late_fusion_records(model_root):
        sc, ns = parse_scenario_scheme(run_key)
        if sc is None:
            continue
        rec = reconstruct_late_fusion_preds(run, cache_dir)
        if rec is None:
            continue
        soft, img_k, lm_k = rec
        pred = np.argmax(soft, axis=1)
        y_true_g = remap_labels(y_raw, ns)
        if len(pred) != len(y_true_g):
            continue
        mf1 = macro_f1(y_true_g, pred)
        fusion_by_group.setdefault((sc, ns), []).append({
            "key": run_key, "macro_f1": float(mf1), "pred": pred,
            "w_image": run["best_image_weight"], "w_landmark": run["best_landmark_weight"],
            "image_branch": img_k, "landmark_branch": lm_k,
        })

    # For each group: find best unimodal + best fusion, run tests
    for (sc, ns), fusions in sorted(fusion_by_group.items()):
        y_true_g = remap_labels(y_raw, ns)
        unis = load_unimodal_caches(cache_dir, sc, ns)
        if not unis:
            continue
        uni_preds = []
        for k, soft in unis:
            p = np.argmax(soft, axis=1)
            if len(p) != len(y_true_g):
                continue
            uni_preds.append({"key": k, "macro_f1": float(macro_f1(y_true_g, p)), "pred": p})
        if not uni_preds:
            continue

        best_uni = max(uni_preds, key=lambda x: x["macro_f1"])
        best_fus = max(fusions, key=lambda x: x["macro_f1"])

        mcn = mcnemar_test(y_true_g, best_uni["pred"], best_fus["pred"])
        boot = bootstrap_delta_f1(y_true_g, best_uni["pred"], best_fus["pred"], n_boot=n_boot)

        result["groups"][f"{sc}_{ns}"] = {
            "best_unimodal": best_uni["key"],
            "best_unimodal_mf1": best_uni["macro_f1"],
            "best_late_fusion": best_fus["key"],
            "best_late_fusion_mf1": best_fus["macro_f1"],
            "best_late_fusion_weights": {
                "w_image": best_fus["w_image"],
                "w_landmark": best_fus["w_landmark"],
                "image_branch": best_fus["image_branch"],
                "landmark_branch": best_fus["landmark_branch"],
            },
            "delta_mf1_point": best_fus["macro_f1"] - best_uni["macro_f1"],
            "mcnemar": mcn,
            "bootstrap": boot,
        }
    return result


def format_markdown(results):
    lines = ["# Statistical Significance Tests — Late Fusion vs Best Unimodal\n"]
    lines.append("> Auto-generated by `scripts/compute_significance_tests.py`.")
    lines.append("> Late Fusion predictions di-rekonstruksi dari cached unimodal softmax + bobot optimal "
                 "(`best_image_weight`, `best_landmark_weight`) yang tersimpan di `fusion_late_*/results.json`.")
    lines.append("> Tests: **McNemar** (paired classifier agreement, exact / continuity-corrected) + "
                 "**paired bootstrap 95% CI** untuk Δ-macro-F1.\n")
    lines.append("**Interpretasi:**")
    lines.append("- p_McNemar < 0.05 → kedua model berbeda secara signifikan dalam pola error.")
    lines.append("- Bootstrap CI tidak crossing 0 → Δ-macro-F1 konsisten ke salah satu arah.")
    lines.append("- ✅ = signifikan, ❌ = tidak signifikan.\n")
    lines.append("Catatan: cuma membandingkan LATE FUSION dengan unimodal karena fusion early/intermediate "
                 "tidak menyimpan softmax cache (cuma summary metrics). Untuk early/intermediate "
                 "perlu rerun inference dari checkpoint `.pt`.\n")

    for r in results:
        ds = r["dataset"]
        if "error" in r:
            lines.append(f"## {ds}\n\n⚠️ {r['error']}\n")
            continue
        lines.append(f"## {ds} (n_test={r['y_test_n']})\n")
        if not r["groups"]:
            lines.append("_No reconstructible late fusion ↔ unimodal pairs found._\n")
            continue
        lines.append("| Group | Best Unimodal (mf1) | Best Late Fusion (mf1) | Δmf1 | "
                     "McNemar p | Boot Δmf1 mean [95% CI] | Boot p | Verdict |")
        lines.append("|---|---|---|---:|---:|---|---:|---|")
        for key, g in sorted(r["groups"].items()):
            mcn_p = g["mcnemar"]["p_value"]
            boot = g["bootstrap"]
            sig_mcn = "✅" if mcn_p < 0.05 else "❌"
            sig_boot = "✅" if (boot["ci_low"] > 0 or boot["ci_high"] < 0) else "❌"
            if g["delta_mf1_point"] > 0 and (sig_mcn == "✅" or sig_boot == "✅"):
                verdict = "**Fusion > Unimodal**"
            elif g["delta_mf1_point"] < 0 and (sig_mcn == "✅" or sig_boot == "✅"):
                verdict = "**Unimodal > Fusion**"
            else:
                verdict = "n.s."
            lines.append(
                f"| **{key}** | `{g['best_unimodal'][:40]}` ({g['best_unimodal_mf1']:.4f}) | "
                f"`{g['best_late_fusion'][:40]}` ({g['best_late_fusion_mf1']:.4f}) | "
                f"{g['delta_mf1_point']:+.4f} | "
                f"{mcn_p:.3g} {sig_mcn} | "
                f"{boot['mean']:+.4f} [{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}] | "
                f"{boot['p_bootstrap']:.3g} {sig_boot} | "
                f"{verdict} |")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None,
                    help="Specific dataset label, e.g. kdef_7class. Default = all.")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--output", default=str(PROJECT_ROOT / "docs" / "significance_tests.md"))
    ap.add_argument("--json-output", default=str(PROJECT_ROOT / "docs" / "significance_tests.json"))
    args = ap.parse_args()

    targets = DATASETS if args.dataset is None else [d for d in DATASETS if d[0] == args.dataset]
    if not targets:
        print(f"Unknown dataset: {args.dataset}")
        return

    results = []
    for label, cache_dir, y_test, model_root in targets:
        print(f"[*] Analyzing {label} ...")
        r = analyze_dataset(label, cache_dir, y_test, model_root, n_boot=args.n_boot)
        results.append(r)

    md_path = Path(args.output)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(format_markdown(results), encoding="utf-8")
    print(f"[✓] Wrote {md_path}")

    json_path = Path(args.json_output)

    def scrub(obj):
        if isinstance(obj, dict):
            return {k: scrub(v) for k, v in obj.items() if not k.startswith("_") and k != "pred"}
        if isinstance(obj, list):
            return [scrub(x) for x in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    json_path.write_text(json.dumps(scrub(results), indent=2), encoding="utf-8")
    print(f"[✓] Wrote {json_path}")


if __name__ == "__main__":
    main()
