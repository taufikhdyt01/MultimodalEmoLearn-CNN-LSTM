#!/usr/bin/env python3
"""5-fold user-level CV on the hand-occlusion-cleaned 6-class dataset.

Pipeline:
  1. Load original conf60 arrays (train/val/test concatenated).
  2. Apply clean_index from `data/dataset_frontonly_conf60_clean6c/clean_index.npz`
     -> drop occluded samples + drop fearful class (6 classes remain).
  3. Use precomputed folds in `data/dataset_frontonly_conf60_clean6c/folds.json`
     for user-level stratified 5-fold CV.
  4. For each (config, fold): carve 10% stratified inner-val from train fold for
     early stopping, train, evaluate on test fold, save metrics.
  5. Aggregate macro-F1 mean ± std across 5 folds per config.

Configs (pilot, top-3 from 7-class baseline):
  - intermediate_tl   IntermediateFusionTransfer
  - late_fusion_tl    CNN_TL + FCNN weighted ensemble (w tuned on inner val)
  - early_fusion_tl   EmotionEarlyFusionTransfer (RGB+heatmap 4-channel)

All configs trained with B2 scenario (class weights, no augmentation).

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/run_cv5_clean6c.py
  CUDA_VISIBLE_DEVICES=0 python scripts/run_cv5_clean6c.py --configs intermediate_tl
  CUDA_VISIBLE_DEVICES=0 python scripts/run_cv5_clean6c.py --folds 0 1
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from training.models import (  # noqa: E402
    EmotionCNNTransfer,
    EmotionEarlyFusionTransfer,
    EmotionFCNN,
    IntermediateFusionTransfer,
)

# ─── Constants ───────────────────────────────────────────────────────────────
SRC_DIR = PROJECT_ROOT / "data" / "dataset_frontonly_conf60"
CLEAN_DIR = PROJECT_ROOT / "data" / "dataset_frontonly_conf60_clean6c"
OUT_BASE = PROJECT_ROOT / "models" / "frontonly_conf60_clean6c" / "cv5"

NUM_CLASSES = 6
EMOTIONS = ["neutral", "happy", "sad", "angry", "disgusted", "surprised"]
BATCH = 32
EPOCHS = 50
PATIENCE = 15
LR_TL = 5e-5
LR_SCRATCH = 1e-4
INNER_VAL_RATIO = 0.10
SEED = 42

CONFIGS = {
    "intermediate_tl": {"arch": "fusion", "lr": LR_TL},
    "late_fusion_tl":  {"arch": "late_fusion", "lr_cnn": LR_TL, "lr_fcnn": LR_SCRATCH},
    "early_fusion_tl": {"arch": "early_fusion", "lr": LR_TL},
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Data loading ────────────────────────────────────────────────────────────
def load_clean_data():
    """Concat original (train+val+test) arrays then apply clean_index filtering.
    Returns dict: img (N,224,224,3), lm (N,136), hm (N,224,224), y (N,) int64, uids (N,) str."""
    print(f"Loading original conf60 arrays from {SRC_DIR}...")
    img = np.concatenate([np.load(SRC_DIR / f"X_{s}_images.npy")    for s in ("train","val","test")])
    lm  = np.concatenate([np.load(SRC_DIR / f"X_{s}_landmarks.npy") for s in ("train","val","test")])
    hm  = np.concatenate([np.load(SRC_DIR / f"X_{s}_heatmaps.npy")  for s in ("train","val","test")])
    uids_all = np.concatenate([np.load(SRC_DIR / f"user_ids_{s}.npy") for s in ("train","val","test")]).astype(str)

    # Index of each sample in its original split (0..N_split-1)
    sizes = {s: len(np.load(SRC_DIR / f"y_{s}.npy")) for s in ("train","val","test")}
    orig_split = np.array(["train"]*sizes["train"] + ["val"]*sizes["val"] + ["test"]*sizes["test"])
    orig_idx_in_split = np.concatenate([np.arange(sizes[s]) for s in ("train","val","test")])

    # Load clean index
    print(f"Loading clean index from {CLEAN_DIR / 'clean_index.npz'}...")
    ci = np.load(CLEAN_DIR / "clean_index.npz", allow_pickle=True)
    ci_split = ci["orig_split"]
    ci_idx = ci["orig_index"]
    ci_uid = ci["user_id"].astype(str)
    ci_y6 = ci["label_6c"].astype(np.int64)
    n_clean = len(ci_y6)

    # For each clean sample, find flat index in concatenated array
    cumulative = {"train": 0, "val": sizes["train"], "test": sizes["train"]+sizes["val"]}
    flat_idx = np.array([cumulative[s] + int(i) for s, i in zip(ci_split, ci_idx)], dtype=np.int64)

    img_c = img[flat_idx].astype(np.float32)
    lm_c  = lm[flat_idx].astype(np.float32)
    hm_c  = hm[flat_idx].astype(np.float32)
    uids_c = uids_all[flat_idx]

    # Sanity check: uids from concat must match clean index user_ids
    assert (uids_c == ci_uid).all(), "user_id mismatch between concat and clean_index"

    print(f"Clean dataset: {n_clean} samples, "
          f"img {img_c.shape}, lm {lm_c.shape}, hm {hm_c.shape}")
    print(f"Class distribution: {[int((ci_y6==c).sum()) for c in range(NUM_CLASSES)]} ({EMOTIONS})")
    return {"img": img_c, "lm": lm_c, "hm": hm_c, "y": ci_y6, "uids": uids_c}


def load_folds():
    with open(CLEAN_DIR / "folds.json") as f:
        return json.load(f)


# ─── Tensors / loaders ───────────────────────────────────────────────────────
def class_weights_from_y(y, num_classes=NUM_CLASSES):
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    w = counts.sum() / (num_classes * np.maximum(counts, 1))
    w = w / w.sum() * num_classes
    return torch.FloatTensor(w).to(device)


def build_loader(arch, img, lm, hm, y, shuffle=False):
    y_t = torch.from_numpy(y).long()
    if arch == "fcnn":
        ds = TensorDataset(torch.from_numpy(lm).float(), y_t)
    elif arch == "cnn":
        t = torch.from_numpy(img).permute(0,3,1,2).float()
        ds = TensorDataset(t, y_t)
    elif arch == "fusion":
        t_img = torch.from_numpy(img).permute(0,3,1,2).float()
        t_lm = torch.from_numpy(lm).float()
        ds = TensorDataset(t_img, t_lm, y_t)
    elif arch == "early_fusion":
        # 4-channel: RGB + heatmap
        t_img = torch.from_numpy(img).permute(0,3,1,2).float()        # (N,3,H,W)
        t_hm  = torch.from_numpy(hm).unsqueeze(1).float()              # (N,1,H,W)
        t4 = torch.cat([t_img, t_hm], dim=1)                            # (N,4,H,W)
        ds = TensorDataset(t4, y_t)
    else:
        raise ValueError(arch)
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle,
                      num_workers=2, pin_memory=True, drop_last=shuffle)


# ─── Generic train/eval ──────────────────────────────────────────────────────
def _forward(model, arch, x_list):
    if arch == "fusion":
        return model(*x_list)
    return model(x_list[0])


def train_single(model, arch, tr_loader, va_loader, criterion, lr, save_path):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                           factor=0.5, patience=8, min_lr=1e-7)
    best_val, best_ep, stale = 0.0, 0, 0
    for epoch in range(1, EPOCHS+1):
        model.train()
        for batch in tr_loader:
            *x, y = batch
            x = [xi.to(device) for xi in x]
            y = y.to(device)
            out = _forward(model, arch, x)
            loss = criterion(out, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        # eval
        model.eval()
        yt, yp = [], []
        with torch.no_grad():
            for batch in va_loader:
                *x, y = batch
                x = [xi.to(device) for xi in x]
                out = _forward(model, arch, x)
                yt.append(y.numpy()); yp.append(out.argmax(1).cpu().numpy())
        yt, yp = np.concatenate(yt), np.concatenate(yp)
        vf1 = f1_score(yt, yp, average="macro", zero_division=0)
        scheduler.step(vf1)
        if vf1 > best_val:
            best_val, best_ep, stale = vf1, epoch, 0
            torch.save(model.state_dict(), save_path)
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    return best_val, best_ep


def eval_loader_probs(model, arch, loader):
    model.eval()
    yt, probs = [], []
    with torch.no_grad():
        for batch in loader:
            *x, y = batch
            x = [xi.to(device) for xi in x]
            out = _forward(model, arch, x)
            yt.append(y.numpy()); probs.append(F.softmax(out, dim=1).cpu().numpy())
    return np.concatenate(yt), np.concatenate(probs)


def metrics_from_pred(yt, yp):
    return {
        "test_macro_f1":    float(f1_score(yt, yp, average="macro",    zero_division=0)),
        "test_weighted_f1": float(f1_score(yt, yp, average="weighted", zero_division=0)),
        "test_accuracy":    float(accuracy_score(yt, yp)),
        "confusion_matrix": confusion_matrix(yt, yp, labels=list(range(NUM_CLASSES))).tolist(),
    }


# ─── Inner train/val split (stratified, sample-level inside train fold) ──────
def split_inner(idx, y, ratio=INNER_VAL_RATIO, seed=SEED):
    """Stratified inner split; for very rare classes (n<2), fallback to random."""
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=ratio, random_state=seed)
        tr, va = next(sss.split(idx, y))
        return idx[tr], idx[va]
    except ValueError:
        # Some classes have <2 samples in this fold's train — random split
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(idx))
        n_va = max(1, int(round(len(idx)*ratio)))
        return idx[perm[n_va:]], idx[perm[:n_va]]


# ─── Per-config training ─────────────────────────────────────────────────────
def fit_intermediate_tl(D, tr_idx, iv_idx, te_idx, ckpt_dir):
    arch = "fusion"
    cw = class_weights_from_y(D["y"][tr_idx])
    crit = nn.CrossEntropyLoss(weight=cw)
    model = IntermediateFusionTransfer(num_classes=NUM_CLASSES).to(device)
    tr_loader = build_loader(arch, D["img"][tr_idx], D["lm"][tr_idx], D["hm"][tr_idx], D["y"][tr_idx], shuffle=True)
    iv_loader = build_loader(arch, D["img"][iv_idx], D["lm"][iv_idx], D["hm"][iv_idx], D["y"][iv_idx])
    te_loader = build_loader(arch, D["img"][te_idx], D["lm"][te_idx], D["hm"][te_idx], D["y"][te_idx])
    save_path = ckpt_dir / "intermediate_tl.pth"
    val_f1, best_ep = train_single(model, arch, tr_loader, iv_loader, crit, LR_TL, save_path)
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    yt, p = eval_loader_probs(model, arch, te_loader)
    out = metrics_from_pred(yt, p.argmax(1))
    out.update({"inner_val_macro_f1": float(val_f1), "best_epoch": int(best_ep)})
    save_path.unlink(missing_ok=True)
    return out


def fit_early_fusion_tl(D, tr_idx, iv_idx, te_idx, ckpt_dir):
    arch = "early_fusion"
    cw = class_weights_from_y(D["y"][tr_idx])
    crit = nn.CrossEntropyLoss(weight=cw)
    model = EmotionEarlyFusionTransfer(num_classes=NUM_CLASSES, pretrained=True).to(device)
    tr_loader = build_loader(arch, D["img"][tr_idx], D["lm"][tr_idx], D["hm"][tr_idx], D["y"][tr_idx], shuffle=True)
    iv_loader = build_loader(arch, D["img"][iv_idx], D["lm"][iv_idx], D["hm"][iv_idx], D["y"][iv_idx])
    te_loader = build_loader(arch, D["img"][te_idx], D["lm"][te_idx], D["hm"][te_idx], D["y"][te_idx])
    save_path = ckpt_dir / "early_fusion_tl.pth"
    val_f1, best_ep = train_single(model, arch, tr_loader, iv_loader, crit, LR_TL, save_path)
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    yt, p = eval_loader_probs(model, arch, te_loader)
    out = metrics_from_pred(yt, p.argmax(1))
    out.update({"inner_val_macro_f1": float(val_f1), "best_epoch": int(best_ep)})
    save_path.unlink(missing_ok=True)
    return out


def fit_late_fusion_tl(D, tr_idx, iv_idx, te_idx, ckpt_dir):
    cw = class_weights_from_y(D["y"][tr_idx])
    crit = nn.CrossEntropyLoss(weight=cw)

    # CNN_TL branch
    cnn = EmotionCNNTransfer(num_classes=NUM_CLASSES).to(device)
    cnn_path = ckpt_dir / "lf_cnn_tl.pth"
    cnn_val, cnn_ep = train_single(
        cnn, "cnn",
        build_loader("cnn", D["img"][tr_idx], D["lm"][tr_idx], D["hm"][tr_idx], D["y"][tr_idx], shuffle=True),
        build_loader("cnn", D["img"][iv_idx], D["lm"][iv_idx], D["hm"][iv_idx], D["y"][iv_idx]),
        crit, LR_TL, cnn_path,
    )

    # FCNN branch
    fcnn = EmotionFCNN(num_classes=NUM_CLASSES).to(device)
    fcnn_path = ckpt_dir / "lf_fcnn.pth"
    fcnn_val, fcnn_ep = train_single(
        fcnn, "fcnn",
        build_loader("fcnn", D["img"][tr_idx], D["lm"][tr_idx], D["hm"][tr_idx], D["y"][tr_idx], shuffle=True),
        build_loader("fcnn", D["img"][iv_idx], D["lm"][iv_idx], D["hm"][iv_idx], D["y"][iv_idx]),
        crit, LR_SCRATCH, fcnn_path,
    )

    cnn.load_state_dict(torch.load(cnn_path, map_location=device, weights_only=True))
    fcnn.load_state_dict(torch.load(fcnn_path, map_location=device, weights_only=True))

    # Probs on inner-val + test
    yt_iv, p_iv_cnn  = eval_loader_probs(cnn,  "cnn",
        build_loader("cnn", D["img"][iv_idx], D["lm"][iv_idx], D["hm"][iv_idx], D["y"][iv_idx]))
    _, p_iv_fcnn = eval_loader_probs(fcnn, "fcnn",
        build_loader("fcnn", D["img"][iv_idx], D["lm"][iv_idx], D["hm"][iv_idx], D["y"][iv_idx]))
    yt_te, p_te_cnn  = eval_loader_probs(cnn,  "cnn",
        build_loader("cnn", D["img"][te_idx], D["lm"][te_idx], D["hm"][te_idx], D["y"][te_idx]))
    _, p_te_fcnn = eval_loader_probs(fcnn, "fcnn",
        build_loader("fcnn", D["img"][te_idx], D["lm"][te_idx], D["hm"][te_idx], D["y"][te_idx]))

    # Grid search w on inner val
    best_w, best_val_f1 = 0.5, 0.0
    for w in np.arange(0.0, 1.05, 0.05):
        fused = w * p_iv_cnn + (1-w) * p_iv_fcnn
        f1 = f1_score(yt_iv, fused.argmax(1), average="macro", zero_division=0)
        if f1 > best_val_f1:
            best_val_f1, best_w = float(f1), float(w)

    fused_te = best_w * p_te_cnn + (1-best_w) * p_te_fcnn
    out = metrics_from_pred(yt_te, fused_te.argmax(1))
    out.update({
        "inner_val_macro_f1": float(best_val_f1),
        "best_w_cnn": best_w,
        "cnn_val_macro_f1":  float(cnn_val),
        "fcnn_val_macro_f1": float(fcnn_val),
        "cnn_best_epoch":  int(cnn_ep),
        "fcnn_best_epoch": int(fcnn_ep),
    })
    cnn_path.unlink(missing_ok=True)
    fcnn_path.unlink(missing_ok=True)
    return out


FIT_FN = {
    "intermediate_tl": fit_intermediate_tl,
    "late_fusion_tl":  fit_late_fusion_tl,
    "early_fusion_tl": fit_early_fusion_tl,
}


# ─── Main CV runner ──────────────────────────────────────────────────────────
def run(configs, fold_ids):
    D = load_clean_data()
    folds_def = load_folds()
    OUT_BASE.mkdir(parents=True, exist_ok=True)

    summary = {}
    for cfg_key in configs:
        out_path = OUT_BASE / f"cv5_{cfg_key}.json"
        if out_path.exists():
            existing = json.load(open(out_path))
            done = {r["fold"] for r in existing.get("per_fold", [])}
            print(f"\n[{cfg_key}] resume: {sorted(done)} done")
        else:
            existing = {"config": cfg_key, "strategy": "cv5_user_clean6c", "per_fold": []}
            done = set()

        for k in fold_ids:
            if k in done:
                continue
            fold_users = folds_def[f"fold_{k}"]
            test_users = set(fold_users["test_users"])
            te_mask = np.array([u in test_users for u in D["uids"]])
            tr_mask = ~te_mask
            tr_all_idx = np.where(tr_mask)[0]
            te_idx = np.where(te_mask)[0]

            # Inner split for early stopping
            tr_idx, iv_idx = split_inner(tr_all_idx, D["y"][tr_all_idx])

            print(f"\n[{cfg_key}] fold {k}: train={len(tr_idx)} inner_val={len(iv_idx)} test={len(te_idx)} "
                  f"(test_users={len(test_users)})")

            ckpt_dir = OUT_BASE / "_tmp"
            ckpt_dir.mkdir(exist_ok=True)

            t0 = time.time()
            res = FIT_FN[cfg_key](D, tr_idx, iv_idx, te_idx, ckpt_dir)
            dt = time.time() - t0
            res.update({"fold": int(k), "n_train": int(len(tr_idx)),
                        "n_inner_val": int(len(iv_idx)), "n_test": int(len(te_idx)),
                        "elapsed_sec": dt})
            existing["per_fold"].append(res)
            json.dump(existing, open(out_path, "w"), indent=2)
            print(f"  -> test_macro_f1={res['test_macro_f1']:.4f}  weighted_f1={res['test_weighted_f1']:.4f}  "
                  f"acc={res['test_accuracy']:.4f}  ({dt:.1f}s)")

        # Aggregate
        if existing["per_fold"]:
            m = [r["test_macro_f1"] for r in existing["per_fold"]]
            w = [r["test_weighted_f1"] for r in existing["per_fold"]]
            existing["macro_f1_mean"] = float(np.mean(m))
            existing["macro_f1_std"]  = float(np.std(m))
            existing["weighted_f1_mean"] = float(np.mean(w))
            existing["weighted_f1_std"]  = float(np.std(w))
            json.dump(existing, open(out_path, "w"), indent=2)
            summary[cfg_key] = (existing["macro_f1_mean"], existing["macro_f1_std"], len(existing["per_fold"]))

    print("\n=== CV5 clean-6c summary ===")
    for k, (m, s, n) in summary.items():
        print(f"  {k:<20s}  macro_f1 = {m:.4f} ± {s:.4f}  (n={n})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()),
                    choices=list(CONFIGS.keys()))
    ap.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    args = ap.parse_args()
    run(args.configs, args.folds)


if __name__ == "__main__":
    main()
