"""Analyze per-user class distribution AFTER hand-occlusion cleaning, dropping fearful.

Purpose: inform a strategic user-level re-split for 6-class so val/test each contain
at least one user that holds samples of every minority class (angry/disgusted/surprised).

Output:
- deploy/data_cleaning/user_class_distribution_clean6c.csv  (user × class matrix)
- deploy/data_cleaning/user_class_distribution_clean6c.md   (human-readable summary)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "dataset_frontonly_conf60"
CLEAN_DIR = ROOT / "deploy" / "data_cleaning"

# Original label IDs: 0=neutral 1=happy 2=sad 3=angry 4=fearful 5=disgusted 6=surprised
DROP_LABEL = 4  # fearful
ORIG_EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]
KEEP_LABELS = [0, 1, 2, 3, 5, 6]
NEW_EMOTIONS = [ORIG_EMOTIONS[c] for c in KEEP_LABELS]  # 6 classes

# Original splits (for reference)
ORIG_INFO = json.loads((DATA_DIR / "dataset_info.json").read_text(encoding="utf-8"))


def load_all_cleaned():
    """Concatenate train+val+test labels, user_ids, and occlusion masks. Apply cleaning."""
    ys, uids, occ = [], [], []
    for sp in ("train", "val", "test"):
        ys.append(np.load(DATA_DIR / f"y_{sp}.npy"))
        uids.append(np.load(DATA_DIR / f"user_ids_{sp}.npy"))
        occ.append(np.load(CLEAN_DIR / f"hand_occlusion_{sp}.npz")["occluded_mask"])
    y = np.concatenate(ys)
    u = np.concatenate(uids).astype(str)
    o = np.concatenate(occ)
    keep = (~o) & (y != DROP_LABEL)
    return y[keep], u[keep]


def build_matrix(y: np.ndarray, u: np.ndarray):
    users = sorted(np.unique(u).tolist(), key=lambda s: int(s) if s.isdigit() else s)
    mat = np.zeros((len(users), len(KEEP_LABELS)), dtype=np.int32)
    for i, usr in enumerate(users):
        sel = u == usr
        for j, c in enumerate(KEEP_LABELS):
            mat[i, j] = int(((y == c) & sel).sum())
    return users, mat


def orig_split_of(user: str) -> str:
    for sp in ("train", "val", "test"):
        if user in [str(x) for x in ORIG_INFO[sp]["users"]]:
            return sp
    return "?"


def main():
    y, u = load_all_cleaned()
    users, mat = build_matrix(y, u)
    totals = mat.sum(axis=0)
    print(f"Total clean samples (6-class): {len(y)}  | users: {len(users)}")
    print("Per-class totals:", dict(zip(NEW_EMOTIONS, totals.tolist())))

    # Identify minority-class holders
    minority_classes = ["angry", "disgusted", "surprised"]
    minority_idx = [NEW_EMOTIONS.index(c) for c in minority_classes]
    holders = {c: [] for c in minority_classes}
    for i, usr in enumerate(users):
        for c, j in zip(minority_classes, minority_idx):
            if mat[i, j] > 0:
                holders[c].append((usr, int(mat[i, j]), orig_split_of(usr)))

    # Write CSV
    csv_path = CLEAN_DIR / "user_class_distribution_clean6c.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("user,orig_split," + ",".join(NEW_EMOTIONS) + ",total\n")
        for i, usr in enumerate(users):
            row = [usr, orig_split_of(usr)] + [str(int(x)) for x in mat[i]] + [str(int(mat[i].sum()))]
            f.write(",".join(row) + "\n")
        f.write("TOTAL,," + ",".join(str(int(x)) for x in totals) + f",{int(mat.sum())}\n")
    print(f"CSV -> {csv_path}")

    # Markdown summary
    md = [
        "# User × Class Distribution (Clean 6-class)",
        "",
        f"After filtering hand-occluded samples and dropping `fearful` (had 0 clean samples).",
        f"Total: **{len(y)} samples** across **{len(users)} users**.",
        "",
        "Per-class totals:",
        "",
        "| " + " | ".join(NEW_EMOTIONS) + " |",
        "|" + "---:|" * len(NEW_EMOTIONS),
        "| " + " | ".join(str(int(x)) for x in totals) + " |",
        "",
        "## Minority-class holders",
        "",
        "Users yang menyumbang sampel di kelas minoritas (angry/disgusted/surprised). Re-split harus pastikan val & test masing-masing dapat ≥1 user dari tiap kelas minoritas.",
        "",
    ]
    for c in minority_classes:
        md.append(f"### `{c}` ({totals[NEW_EMOTIONS.index(c)]} clean samples)")
        md.append("")
        md.append("| user | orig_split | n |")
        md.append("|:---|:---:|---:|")
        for usr, n, sp in sorted(holders[c], key=lambda x: -x[1]):
            md.append(f"| {usr} | {sp} | {n} |")
        md.append("")

    md += [
        "## Full user × class matrix",
        "",
        "| user | orig | " + " | ".join(NEW_EMOTIONS) + " | total |",
        "|:---|:---:|" + "---:|" * (len(NEW_EMOTIONS) + 1),
    ]
    for i, usr in enumerate(users):
        row = [usr, orig_split_of(usr)] + [str(int(x)) for x in mat[i]] + [str(int(mat[i].sum()))]
        md.append("| " + " | ".join(row) + " |")
    md.append(f"| **TOTAL** | | " + " | ".join(f"**{int(x)}**" for x in totals) + f" | **{int(mat.sum())}** |")

    md_path = CLEAN_DIR / "user_class_distribution_clean6c.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"MD  -> {md_path}")

    print("\n=== Minority holders summary ===")
    for c in minority_classes:
        print(f"  {c}: {len(holders[c])} users hold this class")
        for usr, n, sp in sorted(holders[c], key=lambda x: -x[1]):
            print(f"    user {usr} (orig={sp}): {n} samples")


if __name__ == "__main__":
    main()
