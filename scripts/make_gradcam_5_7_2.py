#!/usr/bin/env python3
"""Grad-CAM figure untuk tesis sub-bab 5.7.2.

Membandingkan atensi cabang citra (ResNet-18 TL, layer4) antar 3 model:
  (1) CNN_TL          — unimodal citra
  (2) Early Fusion concat — RGB + channel heatmap landmark (diperlakukan setara)
  (3) Early Fusion gated  — heatmap landmark men-gate atensi spasial

Tujuan: menunjukkan apakah penambahan informasi landmark menggeser atensi ke
region ekspresif (mulut/mata/alis). Grad-CAM HANYA pada cabang citra (peta fitur
konvolusi spasial); cabang landmark FCNN/CNN1D tidak bisa di-Grad-CAM (input vektor).

Tiap baris panel: [Original | CNN_TL | EF-concat | EF-gated], judul "→ pred (conf) ✓/✗".
Bagian: (A) Prediksi benar, (B) Misklasifikasi (pasangan tertukar), (C) Subjek sama.

Output:
  docs/figures/gradcam/gradcam_5_7_2_3c.{png,pdf}
  docs/figures/gradcam/gradcam_5_7_2_7c.{png,pdf}
  docs/figures/gradcam/gradcam_5_7_2_ckplus.{png,pdf}

Usage:
  CUDA_VISIBLE_DEVICES=1 python scripts/make_gradcam_5_7_2.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from training.models import (
    EmotionCNNTransfer, EmotionEarlyFusionTransfer, EmotionEarlyFusionTransferGated,
)

DATA_DIR = PROJECT_ROOT / "data" / "dataset_frontonly_conf60"
CK3 = PROJECT_ROOT / "models" / "frontonly_conf60" / "gradcam_ckpts"
CK7 = PROJECT_ROOT / "models" / "frontonly_conf60" / "gradcam_ckpts_7c"
UNI7 = PROJECT_ROOT / "models" / "frontonly_conf60" / "7class" / "Unified"
OUT = PROJECT_ROOT / "docs" / "figures" / "gradcam"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RNG = np.random.RandomState(42)
BATCH = 64
REMAP_3 = np.array([1, 0, 2, 2, 2, 2, 0], dtype=np.int64)
NAMES_3 = ["positive", "neutral", "negative"]
NAMES_7 = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]

plt.rcParams.update({"font.size": 9})


# ─── data ───────────────────────────────────────────────────────────────────
def load_test(num_classes):
    img = np.load(DATA_DIR / "X_test_images.npy").astype(np.float32)
    hm = np.load(DATA_DIR / "X_test_heatmaps.npy").astype(np.float32)
    y7 = np.load(DATA_DIR / "y_test.npy").astype(np.int64)
    uid = np.load(DATA_DIR / "user_ids_test.npy", allow_pickle=True)
    y = REMAP_3[y7] if num_classes == 3 else y7
    return img, hm, y, uid


def load_ckpt(model, path):
    sd = torch.load(path, map_location=DEVICE, weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd)
    return model.to(DEVICE).eval()


# ─── tensors ────────────────────────────────────────────────────────────────
def t_img(img, idx):
    return torch.from_numpy(img[idx]).permute(0, 3, 1, 2).float().to(DEVICE)


def t_ef(img, hm, idx):
    a = torch.from_numpy(img[idx]).permute(0, 3, 1, 2).float()
    b = torch.from_numpy(hm[idx]).unsqueeze(1).float()
    return torch.cat([a, b], dim=1).to(DEVICE)


def predict(model, arch, img, hm):
    n = len(img); out = []
    with torch.no_grad():
        for i in range(0, n, BATCH):
            idx = list(range(i, min(i + BATCH, n)))
            x = t_ef(img, hm, idx) if arch == "ef" else t_img(img, idx)
            out.append(F.softmax(model(x), dim=1).cpu().numpy())
    return np.concatenate(out)


def target_layer(model):
    return model.features[-2][-1]


def gradcam_overlay(model, arch, img, hm, idx):
    rgb = np.clip(img[idx], 0, 1).astype(np.float32)
    cam = GradCAM(model=model, target_layers=[target_layer(model)])
    x = t_ef(img, hm, [idx]) if arch == "ef" else t_img(img, [idx])
    gray = cam(input_tensor=x, targets=None)[0]
    return rgb, show_cam_on_image(rgb, gray, use_rgb=True)


# ─── figure builder (satu section = satu figure) ────────────────────────────
HEADER_BOX = dict(boxstyle="round,pad=0.35", facecolor="#eef2f7", edgecolor="#7a8aa0")


def build_figure(models, img, hm, y, names, rows, ref_probs, title, subtitle, out_name):
    """Render satu section (beberapa baris) jadi satu figure dgn header kolom besar."""
    n_rows = len(rows)
    n_col = 1 + len(models)
    fig, axes = plt.subplots(n_rows, n_col,
                             figsize=(3.05 * n_col, 3.05 * n_rows + 0.5))
    if n_rows == 1:
        axes = axes[None, :]
    # header kolom — sekali di atas, diperbesar + kotak latar
    col_titles = ["Citra asli"] + [m[0] for m in models]
    for c, ct in enumerate(col_titles):
        axes[0, c].set_title(ct, fontsize=13.5, fontweight="bold", pad=12,
                             bbox=HEADER_BOX)
    for r, row in enumerate(rows):
        idx, true_cls = row["idx"], row["true"]
        rgb = np.clip(img[idx], 0, 1)
        axes[r, 0].imshow(rgb)
        axes[r, 0].set_ylabel(f"True: {names[true_cls]}", fontsize=11, fontweight="bold")
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        if row.get("frame_lbl"):
            axes[r, 0].set_xlabel(row["frame_lbl"], fontsize=8.5, color="#333")
        for c, (label, model, arch) in enumerate(models, start=1):
            probs = ref_probs[label][idx]
            pred = int(probs.argmax()); conf = float(probs[pred])
            _, ov = gradcam_overlay(model, arch, img, hm, idx)
            ax = axes[r, c]
            ax.imshow(ov); ax.set_xticks([]); ax.set_yticks([])
            ok = pred == true_cls
            # label prediksi di BAWAH sel (supaya tidak menimpa header kolom)
            ax.text(0.5, -0.045, f"→ {names[pred]} ({conf:.2f}) {'✓' if ok else '✗'}",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=10, fontweight="bold",
                    color=("#1a7f37" if ok else "#c0392b"))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.015)
    fig.text(0.5, 0.94, subtitle, ha="center", va="top", fontsize=10.5, color="#333")
    fig.text(0.5, 0.004,
             "Grad-CAM cabang citra (ResNet-18 TL, layer4). Cabang landmark tidak "
             "di-Grad-CAM (input vektor, bukan citra).",
             ha="center", va="bottom", fontsize=8.5, style="italic", color="#444")
    plt.tight_layout(rect=[0, 0.018, 1, 0.915], h_pad=3.0)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{out_name}.{ext}", dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_name}.{{png,pdf}}  ({n_rows} baris)")


SECTIONS = [
    ("(A) Prediksi benar", "a_benar", "Atensi pada region ekspresif (mulut/mata/alis)"),
    ("(B) Misklasifikasi", "b_misklasifikasi", "Atensi 'salah arah': latar, oklusi, atau ciri identitas"),
    ("(C) Subjek sama", "c_subjek", "Beberapa frame satu partisipan — sebagian benar, sebagian salah"),
]


def emit_sections(scheme_tag, models, img, hm, y, names, rows, ref_probs, scheme_title):
    """Pisah rows berdasarkan section → satu figure per section."""
    for sec_label, sec_slug, sec_sub in SECTIONS:
        sec_rows = [r for r in rows if r["section"] == sec_label
                    or (sec_label.startswith("(C)") and r["section"] == "")]
        # baris subjek-sama: baris pertama bertanda "(C)…", lanjutannya section=""
        if sec_label.startswith("(C)"):
            sec_rows = [r for r in rows if r["section"].startswith("(C)") or r["section"] == ""]
        if not sec_rows:
            continue
        title = f"Grad-CAM Cabang Citra — {scheme_title}\n{sec_label}"
        build_figure(models, img, hm, y, names, sec_rows, ref_probs,
                     title, sec_sub, f"gradcam_5_7_2_{scheme_tag}_{sec_slug}")


# ─── sample selection ───────────────────────────────────────────────────────
def pick(preds, y, true_cls, pred_cls, n=1, exclude=()):
    mask = (y == true_cls) & (preds == pred_cls)
    idx = [i for i in np.where(mask)[0] if i not in exclude]
    if not idx:
        return []
    return RNG.choice(idx, size=min(n, len(idx)), replace=False).tolist()


def same_subject_rows(cnn_preds, y, uid, names, n_frames=3):
    """Pilih 1 subjek dgn campuran benar/salah (CNN_TL), kembalikan beberapa frame."""
    users = {}
    for u in set(uid.tolist()):
        m = np.where(uid == u)[0]
        correct = (cnn_preds[m] == y[m])
        if 0 < correct.sum() < len(m):  # ada benar & ada salah
            users[u] = (m, correct)
    if not users:
        return []
    # pilih subjek dgn frame terbanyak
    u = max(users, key=lambda k: len(users[k][0]))
    m, correct = users[u]
    # ambil sebagian benar + sebagian salah
    cor = m[correct].tolist(); wro = m[~correct].tolist()
    chosen = (cor[:1] + wro[: n_frames - 1]) if len(cor) else wro[:n_frames]
    chosen = chosen[:n_frames]
    rows = []
    for k, i in enumerate(chosen):
        rows.append({"section": "(C) Subjek sama" if k == 0 else "",
                     "true": int(y[i]), "idx": int(i),
                     "frame_lbl": f"subj {u} · frame {k+1}"})
    return rows


def main():
    print(f"Device: {DEVICE}")

    # ===== 3-class =====
    print("\n[3c] building...")
    img, hm, y, uid = load_test(3)
    m_cnn = load_ckpt(EmotionCNNTransfer(num_classes=3), CK3 / "cnn_tl.pth")
    m_con = load_ckpt(EmotionEarlyFusionTransfer(num_classes=3), CK3 / "early_fusion_concat_tl.pth")
    m_gat = load_ckpt(EmotionEarlyFusionTransferGated(num_classes=3), CK3 / "early_fusion_tl.pth")
    models3 = [("CNN_TL (unimodal)", m_cnn, "cnn"),
               ("Early concat", m_con, "ef"),
               ("Early gated", m_gat, "ef")]
    ref3 = {lbl: predict(mdl, arch, img, hm) for lbl, mdl, arch in models3}
    pc = ref3["CNN_TL (unimodal)"].argmax(1)
    # NAMES_3 = positive(0), neutral(1), negative(2)
    rows3 = []
    # (A) benar: positive & neutral
    for cls in [0, 1]:
        s = pick(pc, y, cls, cls, 1)
        if s: rows3.append({"section": "(A) Prediksi benar", "true": cls, "idx": s[0]})
    # (B) misklasifikasi: neutral<->negative (ekspresi halus)
    used = {r["idx"] for r in rows3}
    for tc, pcl in [(1, 2), (2, 1), (2, 0)]:
        s = pick(pc, y, tc, pcl, 1, exclude=used)
        if s:
            rows3.append({"section": "(B) Misklasifikasi", "true": tc, "idx": s[0]})
            used.add(s[0])
    # (C) subjek sama
    rows3 += same_subject_rows(pc, y, uid, NAMES_3, n_frames=2)
    emit_sections("3c", models3, img, hm, y, NAMES_3, rows3, ref3, "3 Kelas (Primer)")

    # ===== 7-class =====
    print("\n[7c] building...")
    img, hm, y, uid = load_test(7)
    m_cnn = load_ckpt(EmotionCNNTransfer(num_classes=7), CK7 / "cnn_tl.pth")
    m_con = load_ckpt(EmotionEarlyFusionTransfer(num_classes=7), CK7 / "early_fusion_tl.pth")
    m_gat = load_ckpt(EmotionEarlyFusionTransferGated(num_classes=7),
                      UNI7 / "fusion_early_gated_tl" / "checkpoints" / "b1.pt")
    models7 = [("CNN_TL (unimodal)", m_cnn, "cnn"),
               ("Early concat", m_con, "ef"),
               ("Early gated", m_gat, "ef")]
    ref7 = {lbl: predict(mdl, arch, img, hm) for lbl, mdl, arch in models7}
    pc = ref7["CNN_TL (unimodal)"].argmax(1)
    # names idx: 0 neutral,1 happy,2 sad,3 angry,4 fearful,5 disgusted,6 surprised
    rows7 = []
    for cls in [1, 0]:  # happy, neutral benar
        s = pick(pc, y, cls, cls, 1)
        if s: rows7.append({"section": "(A) Prediksi benar", "true": cls, "idx": s[0]})
    used = {r["idx"] for r in rows7}
    # minoritas tertukar jadi netral/bahagia: fearful->neutral, disgusted->neutral/happy, sad->neutral
    for tc, pcl in [(4, 0), (5, 0), (5, 1), (2, 0), (3, 0)]:
        s = pick(pc, y, tc, pcl, 1, exclude=used)
        if s:
            rows7.append({"section": "(B) Misklasifikasi", "true": tc, "idx": s[0]})
            used.add(s[0])
        if sum(r["section"] == "(B) Misklasifikasi" for r in rows7) >= 3:
            break
    rows7 += same_subject_rows(pc, y, uid, NAMES_7, n_frames=2)
    emit_sections("7c", models7, img, hm, y, NAMES_7, rows7, ref7, "7 Kelas (Primer)")

    print("\nDone.")


if __name__ == "__main__":
    main()
