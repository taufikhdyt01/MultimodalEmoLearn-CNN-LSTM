"""
Generate markdown tables dengan SEMUA metrik (macro_f1, weighted_f1, accuracy) +
placeholder ⏳ untuk eksperimen yang belum dijalankan.

Output: docs/all_metrics_tables.md (auto-generated, regenerable)

Re-run kapan saja untuk refresh tabel saat hasil baru selesai.

Usage:
    python scripts/build_results_tables.py
"""
import json
from itertools import product
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
PRIMER = PROJECT / "models" / "frontonly_conf60"
OUT = PROJECT / "docs" / "all_metrics_tables.md"


# ---------- Loaders ----------
def load_all(scheme_dir: Path) -> dict:
    """Return {run_key: run_record} merged from all results.json under scheme_dir."""
    out = {}
    if not scheme_dir.exists():
        return out
    for f in scheme_dir.glob("*/results.json"):
        try:
            d = json.load(open(f))
        except Exception as e:
            print(f"  WARN: failed to load {f}: {e}")
            continue
        for k, v in d.get("runs", {}).items():
            out[k] = v
    return out


def cell(rec, metric, placeholder="⏳"):
    if rec is None:
        return placeholder
    v = rec.get("test", {}).get(metric)
    if v is None:
        return "?"
    return f"{v:.4f}"


# ---------- Spec definitions ----------
LANDMARK_FEATURES = ["raw_136", "facs_28", "blendshape_52", "facs_plus_bs_80"]
LANDMARK_ARCHS = ["fcnn", "cnn1d"]
LANDMARK_SOURCES = ["mediapipe", "faceapi"]
IMAGE_ARCHS = ["cnn_scratch", "cnn_tl"]
FUSIONS = ["early", "intermediate", "late"]
FUSION_VARIANTS = ["scratch", "tl"]
SCENARIOS = ["b1", "b2", "b3"]
SCHEMES = [3, 7]
METRICS = ["macro_f1", "weighted_f1", "accuracy"]


def landmark_combos(features=LANDMARK_FEATURES, archs=LANDMARK_ARCHS,
                    sources=LANDMARK_SOURCES, scenarios=SCENARIOS, schemes=SCHEMES):
    """Note: run_unified_landmark.py uses key format <source>_<feature>_<arch>_<scen>_<scheme>
    (raw_136, facs_28). run_unified_derived.py uses <feature>_<source>_<arch>_<scen>_<scheme>
    (blendshape_52, facs_plus_bs_80). We yield both formats so caller can try each."""
    DERIVED = {"blendshape_52", "facs_plus_bs_80"}
    for feat in features:
        for src in sources:
            # blendshape_52 only MP (no FA source available)
            if feat == "blendshape_52" and src == "faceapi":
                continue
            for arch in archs:
                for scen in scenarios:
                    for nc in schemes:
                        if feat in DERIVED:
                            key = f"{feat}_{src}_{arch}_{scen}_{nc}c"
                        else:
                            key = f"{src}_{feat}_{arch}_{scen}_{nc}c"
                        yield (feat, src, arch, scen, nc, key)


def image_combos(archs=IMAGE_ARCHS, scenarios=SCENARIOS, schemes=SCHEMES):
    for arch in archs:
        for scen in scenarios:
            for nc in schemes:
                yield (arch, scen, nc, f"{arch}_{scen}_{nc}c")


def fusion_combos(fusions=FUSIONS, variants=FUSION_VARIANTS, sources=("mediapipe", "faceapi"),
                  scenarios=SCENARIOS, schemes=SCHEMES, features=("raw_136",),
                  early_fusion_modes=("concat",)):
    """Iterate fusion combos. By default features=("raw_136",) untuk backward-compat.

    Early Fusion punya 2 mode: concat (default, channel-stack) atau gated (spatial
    sigmoid gating sebelum CNN). Pass early_fusion_modes=("concat","gated") untuk
    include kedua mode. Intermediate & Late Fusion tidak punya mode (selalu "concat").

    Skip invalid combos:
    - Early Fusion only supports raw_136 (input is RGB+heatmap channel)
    - blendshape_52 only available from MP source (no FA-blendshape pipeline)
    """
    for fusion in fusions:
        # Iterate fusion modes only for early fusion; intermediate/late selalu single mode
        modes_for_fusion = early_fusion_modes if fusion == "early" else ("concat",)
        for mode in modes_for_fusion:
            mode_tag = "_gated" if (fusion == "early" and mode == "gated") else ""
            for variant in variants:
                for src in sources:
                    src_tag = "" if src == "mediapipe" else f"_{src}"
                    for feat in features:
                        if fusion == "early" and feat != "raw_136":
                            continue
                        if feat == "blendshape_52" and src != "mediapipe":
                            continue
                        feat_tag = "" if feat == "raw_136" else f"_{feat}"
                        for scen in scenarios:
                            for nc in schemes:
                                key = f"fusion_{fusion}{mode_tag}_{variant}{feat_tag}{src_tag}_{scen}_{nc}c"
                                yield (fusion, mode, variant, src, feat, scen, nc, key)


# ---------- Table builders ----------
def landmark_table(runs, title, scenarios=SCENARIOS, schemes=SCHEMES, sources=LANDMARK_SOURCES,
                   features=LANDMARK_FEATURES, archs=LANDMARK_ARCHS):
    lines = [f"### {title}\n"]
    lines.append("| Feature | Source | Arch | Scheme | Scenario | macro_f1 | weighted_f1 | accuracy |")
    lines.append("|---|---|---|:---:|:---:|:---:|:---:|:---:|")
    for feat, src, arch, scen, nc, key in landmark_combos(
            features=features, archs=archs, sources=sources, scenarios=scenarios, schemes=schemes):
        rec = runs.get(key)
        src_lbl = "MP" if src == "mediapipe" else "FA"
        lines.append(
            f"| {feat} | {src_lbl} | {arch.upper()} | {nc}c | {scen.upper()} "
            f"| {cell(rec, 'macro_f1')} | {cell(rec, 'weighted_f1')} | {cell(rec, 'accuracy')} |"
        )
    return "\n".join(lines) + "\n"


def image_table(runs, title, scenarios=SCENARIOS, schemes=SCHEMES, archs=IMAGE_ARCHS):
    lines = [f"### {title}\n"]
    lines.append("| Arch | Scheme | Scenario | macro_f1 | weighted_f1 | accuracy |")
    lines.append("|---|:---:|:---:|:---:|:---:|:---:|")
    for arch, scen, nc, key in image_combos(archs=archs, scenarios=scenarios, schemes=schemes):
        rec = runs.get(key)
        lines.append(
            f"| {arch.upper()} | {nc}c | {scen.upper()} "
            f"| {cell(rec, 'macro_f1')} | {cell(rec, 'weighted_f1')} | {cell(rec, 'accuracy')} |"
        )
    return "\n".join(lines) + "\n"


def fusion_table(runs, title, scenarios=SCENARIOS, schemes=SCHEMES,
                 sources=("mediapipe", "faceapi"), fusions=FUSIONS, variants=FUSION_VARIANTS,
                 features=("raw_136",), early_fusion_modes=("concat",)):
    lines = [f"### {title}\n"]
    # Show "Mode" column only kalau ada early fusion + >1 mode
    show_mode = any(f == "early" for f in fusions) and len(early_fusion_modes) > 1
    if show_mode:
        lines.append("| Fusion | Mode | Variant | Feature | Source | Scheme | Scenario | macro_f1 | weighted_f1 | accuracy |")
        lines.append("|---|---|---|---|---|:---:|:---:|:---:|:---:|:---:|")
    else:
        lines.append("| Fusion | Variant | Feature | Source | Scheme | Scenario | macro_f1 | weighted_f1 | accuracy |")
        lines.append("|---|---|---|---|:---:|:---:|:---:|:---:|:---:|")
    for fusion, mode, variant, src, feat, scen, nc, key in fusion_combos(
            fusions=fusions, variants=variants, sources=sources,
            scenarios=scenarios, schemes=schemes, features=features,
            early_fusion_modes=early_fusion_modes):
        rec = runs.get(key)
        src_lbl = "MP" if src == "mediapipe" else "FA"
        # For Intermediate/Late, mode irrelevant — only show "concat"
        if show_mode:
            mode_str = mode if fusion == "early" else "—"
            lines.append(
                f"| {fusion} | {mode_str} | {variant} | {feat} | {src_lbl} | {nc}c | {scen.upper()} "
                f"| {cell(rec, 'macro_f1')} | {cell(rec, 'weighted_f1')} | {cell(rec, 'accuracy')} |"
            )
        else:
            lines.append(
                f"| {fusion} | {variant} | {feat} | {src_lbl} | {nc}c | {scen.upper()} "
                f"| {cell(rec, 'macro_f1')} | {cell(rec, 'weighted_f1')} | {cell(rec, 'accuracy')} |"
            )
    return "\n".join(lines) + "\n"


def summary_table(runs, dataset_name, expected_total):
    """Show count of done vs missing per category."""
    done = sum(1 for k, v in runs.items()
               if isinstance(v.get("test", {}).get("macro_f1"), (int, float)))
    return f"**{dataset_name}**: {done}/{expected_total} runs done."


# ---------- Main ----------
def main():
    lines = [
        "# All Metrics: Lengkap Hasil Eksperimen (auto-generated)",
        "",
        "> 📋 **Auto-generated** dari `scripts/build_results_tables.py`. Regenerate kapan saja saat sweep baru selesai.",
        ">",
        "> ⏳ = combination belum dijalankan / sedang berjalan. Cell ?  = file ada tapi metric tidak ada.",
        ">",
        "> Setiap baris tabel = satu run unik. Metrik: macro_f1, weighted_f1, accuracy.",
        "",
        "---",
        "",
    ]

    # 1. Primer Unimodal
    runs = load_all(PRIMER / "3class" / "Unified")
    runs.update(load_all(PRIMER / "7class" / "Unified"))
    lines.append("## 1. Primer Unimodal (Landmark + Image)\n")
    lines.append("Source data: `models/frontonly_conf60/{3,7}class/Unified/`.\n")
    lines.append(landmark_table(runs, "1.1 Landmark (raw_136 / facs_28 / blendshape_52 / facs_plus_bs_80)"))
    lines.append(image_table(runs, "1.2 Image (CNN scratch / CNN_TL)"))
    lines.append("---\n")

    # 2. Primer Multimodal (Fusion) — split per fusion type
    lines.append("## 2. Primer Multimodal (Fusion)\n")
    lines.append("Source data: `models/frontonly_conf60/{3,7}class/Unified/fusion_*/`. Tabel dipecah per jenis fusion (Early / Intermediate / Late) supaya feature × variant × source × scenario × scheme bisa di-compare dalam satu tabel.\n")
    ALL_FEATURES = ("raw_136", "facs_28", "blendshape_52", "facs_plus_bs_80")
    lines.append(fusion_table(runs,
        "2.1 Early Fusion (raw_136 only) — concat vs gated mode",
        fusions=("early",), features=("raw_136",),
        early_fusion_modes=("concat", "gated")))
    lines.append(fusion_table(runs,
        "2.2 Intermediate Fusion (semua feature × variant × source)",
        fusions=("intermediate",), features=ALL_FEATURES))
    lines.append(fusion_table(runs,
        "2.3 Late Fusion (semua feature × variant × source — saat ini hanya raw_136 MP yang implemented)",
        fusions=("late",), features=ALL_FEATURES))
    lines.append("---\n")

    # 3-6. Secondary datasets (KDEF, RAF-DB, CK+, JAFFE) — only B1, only MP source
    secondary = [
        ("KDEF 7c", PROJECT / "models/benchmark/kdef_7class"),
        ("RAF-DB 7c", PROJECT / "models/benchmark/rafdb_7class"),
        ("CK+ 7c", PROJECT / "models/benchmark/ckplus_7class"),
        ("JAFFE 7c", PROJECT / "models/benchmark/jaffe_7class"),
    ]
    for i, (name, path) in enumerate(secondary, start=3):
        lines.append(f"## {i}. {name} (Cross-dataset Benchmark)\n")
        lines.append(f"Source data: `{path.relative_to(PROJECT)}/{{3,7}}class/Unified/`.\n")
        r = load_all(path / "3class" / "Unified")
        r.update(load_all(path / "7class" / "Unified"))
        # B1 only, MP only for secondary (FA tidak tersedia)
        lines.append(landmark_table(r, f"{i}.1 Landmark (MP only)",
                                     scenarios=["b1", "b2", "b3"], sources=["mediapipe"]))
        lines.append(image_table(r, f"{i}.2 Image", scenarios=["b1", "b2", "b3"]))
        ALL_FEATURES_SEC = ("raw_136", "facs_28", "blendshape_52", "facs_plus_bs_80")
        lines.append(fusion_table(r, f"{i}.3 Early Fusion (raw_136, MP source) — concat vs gated",
                                  scenarios=["b1", "b2", "b3"], sources=("mediapipe",),
                                  fusions=("early",), features=("raw_136",),
                                  early_fusion_modes=("concat", "gated")))
        lines.append(fusion_table(r, f"{i}.4 Intermediate Fusion (semua feature × variant, MP source)",
                                  scenarios=["b1", "b2", "b3"], sources=("mediapipe",),
                                  fusions=("intermediate",), features=ALL_FEATURES_SEC))
        lines.append(fusion_table(r, f"{i}.5 Late Fusion (semua feature × variant, MP source)",
                                  scenarios=["b1", "b2", "b3"], sources=("mediapipe",),
                                  fusions=("late",), features=ALL_FEATURES_SEC))
        lines.append("---\n")

    # 7. Summary count per dataset
    lines.append("## 7. Summary: Runs DONE vs Expected\n")
    lines.append("| Dataset | Landmark expected | Landmark done | Image expected | Image done | Fusion expected | Fusion done |")
    lines.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|")

    datasets_for_summary = [
        ("Primer", PRIMER, (2, 2, 3, 2)),  # FA+MP for raw/facs/fb80, MP only for blend
        ("KDEF 7c", PROJECT / "models/benchmark/kdef_7class", (1, 2, 3, 2)),
        ("RAF-DB 7c", PROJECT / "models/benchmark/rafdb_7class", (1, 2, 3, 2)),
        ("CK+ 7c", PROJECT / "models/benchmark/ckplus_7class", (1, 2, 3, 2)),
        ("JAFFE 7c", PROJECT / "models/benchmark/jaffe_7class", (1, 2, 3, 2)),
    ]
    all_features = ("raw_136", "facs_28", "blendshape_52", "facs_plus_bs_80")
    all_early_modes = ("concat", "gated")
    for name, path, _ in datasets_for_summary:
        r = load_all(path / "3class" / "Unified")
        r.update(load_all(path / "7class" / "Unified"))
        is_primer = (name == "Primer")
        lm_keys = [k for f, s, a, sc, nc, k in landmark_combos()
                   if is_primer or s == "mediapipe"]
        img_keys = [k for *_, k in image_combos()]
        # Fusion: include all feature variants + early fusion modes (concat & gated)
        fus_keys = [k for f, mo, v, s, fe, sc, nc, k in fusion_combos(
                       features=all_features, early_fusion_modes=all_early_modes)
                    if is_primer or s == "mediapipe"]
        lm_done = sum(1 for k in lm_keys if r.get(k, {}).get("test", {}).get("macro_f1") is not None)
        img_done = sum(1 for k in img_keys if r.get(k, {}).get("test", {}).get("macro_f1") is not None)
        fus_done = sum(1 for k in fus_keys if r.get(k, {}).get("test", {}).get("macro_f1") is not None)
        lines.append(
            f"| {name} | {len(lm_keys)} | {lm_done} | {len(img_keys)} | {img_done} | {len(fus_keys)} | {fus_done} |"
        )
    lines.append("")
    lines.append("> Catatan: Primer expected includes MP+FA sources. Sekunder hanya MP "
                 "(FA tidak tersedia karena image sudah pre-cropped, butuh JS pipeline).")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Regenerate dengan: `python scripts/build_results_tables.py`*")

    OUT.write_text("\n".join(lines))
    print(f"Wrote {OUT.relative_to(PROJECT)} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
