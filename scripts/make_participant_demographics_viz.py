"""
Visualisasi komposisi 37 partisipan untuk Bab 3 Dataset tesis.

Dua panel:
  (a) Gender (Perempuan / Laki-Laki) — pie chart
  (b) Distribusi usia — histogram

Sumber: data/participant_demographics_anon.csv (mapping user_id ↔ demografi
dari formulir persetujuan, sudah anonim — tanpa nama/email/NIM).

Output:
  docs/figures/participant_demographics.png
  docs/figures/participant_demographics.pdf

Usage:
    python scripts/make_participant_demographics_viz.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "participant_demographics_anon.csv"
OUT_DIR = PROJECT_ROOT / "docs" / "figures"


def main():
    df = pd.read_csv(CSV_PATH)
    print(f"Total participants in CSV: {len(df)}")
    matched = df[df["gender"] != "UNKNOWN"].copy()
    print(f"Matched (gender known): {len(matched)}")
    print(f"Unmatched user_ids: {df[df['gender']=='UNKNOWN']['user_id'].tolist()}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5),
                              gridspec_kw={"wspace": 0.30, "width_ratios": [1.0, 1.4]})

    # ── Panel (a): Gender pie ────────────────────────────────────────────────
    gender_counts = matched["gender"].value_counts()
    colors_gender = {"Perempuan": "#ee6666", "Laki-Laki": "#3b7dd8"}
    pie_colors = [colors_gender.get(g, "#888") for g in gender_counts.index]
    wedges, texts, autotexts = axes[0].pie(
        gender_counts.values, labels=gender_counts.index,
        colors=pie_colors, autopct=lambda p: f"{p:.0f}%\n(n={int(p*len(matched)/100)})",
        startangle=90, textprops={"fontsize": 10})
    for at in autotexts:
        at.set_color("white"); at.set_fontweight("bold")
    axes[0].set_title("(a) Gender",
                       fontsize=12, fontweight="bold", pad=10)

    # ── Panel (b): Usia histogram ────────────────────────────────────────────
    ages = matched["usia"].dropna().astype(int)
    bins = np.arange(ages.min() - 0.5, ages.max() + 1.5, 1)
    counts, _, patches = axes[1].hist(ages, bins=bins, color="#91cc75",
                                       edgecolor="black", linewidth=0.8)
    for cnt, patch in zip(counts, patches):
        if cnt > 0:
            axes[1].text(patch.get_x() + patch.get_width() / 2,
                          patch.get_height() + 0.15,
                          f"{int(cnt)}", ha="center", fontsize=9, fontweight="bold")
    axes[1].set_xticks(range(int(ages.min()), int(ages.max()) + 1))
    axes[1].set_xlabel("Usia (tahun)", fontsize=10)
    axes[1].set_ylabel("Jumlah partisipan", fontsize=10)
    axes[1].set_title(f"(b) Distribusi Usia\n(rerata={ages.mean():.1f}, "
                       f"range={ages.min()}–{ages.max()})",
                       fontsize=12, fontweight="bold", pad=10)
    axes[1].set_ylim(0, max(counts) * 1.20)
    axes[1].grid(axis="y", linestyle=":", alpha=0.4)
    axes[1].set_axisbelow(True)

    # Main title
    n_unknown = (df["gender"] == "UNKNOWN").sum()
    suptitle = (f"Komposisi {len(df)} Partisipan Pengumpulan Data Primer"
                f"  ({len(matched)} dengan demografi tersedia"
                f"{', ' + str(n_unknown) + ' unmatched' if n_unknown > 0 else ''})")
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.02)

    # Footnote
    fig.text(0.5, -0.02,
             "Per-user split (train/val/test) memastikan tidak ada subject leakage. "
             "Data demografi diambil dari Formulir Persetujuan Partisipasi Penelitian (Google Forms).",
             ha="center", fontsize=9, style="italic", color="#555555")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "participant_demographics.png"
    out_pdf = OUT_DIR / "participant_demographics.pdf"
    plt.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.20,
                facecolor="white")
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.20, facecolor="white")
    plt.close(fig)
    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")

    # Print summary
    print("\n=== Summary ===")
    print(f"Gender: {dict(gender_counts)}")
    print(f"Age: mean={ages.mean():.1f}, range={ages.min()}-{ages.max()}, std={ages.std():.1f}")


if __name__ == "__main__":
    main()
