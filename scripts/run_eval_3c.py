#!/usr/bin/env python3
"""
Robustness evaluation untuk Primer conf60 3-class.

Top-3 model (val-based selection from 27-config grid):
  1. Late_Fusion_TL  (best fusion+TL, val_macro_f1 0.6229 di B3)
  2. FCNN            (landmark-only, val_macro_f1 0.6193 di B3)
  3. Intermediate_TL (best intermediate fusion, val_macro_f1 0.5593 di B2)

3 evaluation strategies:
  --strategy random : 5 random seeds, sample-level shuffle (sengaja leakage baseline)
  --strategy cv5    : 5-fold CV subject-wise (37 users -> 5 groups)
  --strategy loso   : LOSO 37-fold (1 user per fold sebagai test)

Catatan:
  - Class weights dihitung per-fold dari training data (B2 scenario).
  - Augmentation DILEWATI karena augmented dataset tidak bisa di-regenerate per-fold
    secara konsisten. Konsekuensi: angka absolut sedikit lebih rendah dari B3
    champion, tapi relative comparison antar strategy tetap valid.
  - Resume-aware: skip kalau JSON output untuk (model, fold) sudah ada.

Usage:
  CUDA_VISIBLE_DEVICES=1 python scripts/run_eval_3c.py --strategy random --seeds 5
  CUDA_VISIBLE_DEVICES=1 python scripts/run_eval_3c.py --strategy cv5
  CUDA_VISIBLE_DEVICES=1 python scripts/run_eval_3c.py --strategy loso
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from training.models import (  # noqa: E402
    EmotionCNNTransfer,
    EmotionFCNN,
    IntermediateFusionTransfer,
)

# ─── Constants ───────────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data" / "dataset_frontonly_conf60"
OUT_BASE = PROJECT_ROOT / "models" / "frontonly_conf60" / "3class"

NUM_CLASSES = 3
EMOTIONS = ["positive", "neutral", "negative"]
REMAP_3 = np.array([1, 0, 2, 2, 2, 2, 0], dtype=np.int64)

BATCH = 32
EPOCHS = 50
PATIENCE = 15
LR_TL = 5e-5
LR_SCRATCH = 1e-4
SEED = 42

# Top-3 models — (key, builder, arch, lr)
TOP3 = {
    "late_fusion_tl": {
        "arch": "late_fusion",  # special case: 2 branches
        "lr_cnn": LR_TL,
        "lr_fcnn": LR_SCRATCH,
    },
    "fcnn": {
        "builder": lambda: EmotionFCNN(num_classes=NUM_CLASSES),
        "arch": "fcnn",
        "lr": LR_SCRATCH,
    },
    "intermediate_tl": {
        "builder": lambda: IntermediateFusionTransfer(num_classes=NUM_CLASSES),
        "arch": "fusion",
        "lr": LR_TL,
    },
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Data loading ────────────────────────────────────────────────────────────
def load_all():
    """Load all conf60 data + user_ids, combined across splits.
    Returns: dict with keys img, lm, hm, y (3-class), uids."""
    img = np.concatenate([
        np.load(DATA_DIR / f"X_{s}_images.npy").astype(np.float32)
        for s in ("train", "val", "test")
    ])
    lm = np.concatenate([
        np.load(DATA_DIR / f"X_{s}_landmarks.npy").astype(np.float32)
        for s in ("train", "val", "test")
    ])
    hm = np.concatenate([
        np.load(DATA_DIR / f"X_{s}_heatmaps.npy").astype(np.float32)
        for s in ("train", "val", "test")
    ])
    y7 = np.concatenate([
        np.load(DATA_DIR / f"y_{s}.npy") for s in ("train", "val", "test")
    ])
    uids = np.load(DATA_DIR / "user_ids_all.npy", allow_pickle=True)
    y = REMAP_3[y7].astype(np.int64)
    assert len(img) == len(lm) == len(hm) == len(y) == len(uids), (
        f"length mismatch: img={len(img)} lm={len(lm)} hm={len(hm)} y={len(y)} uids={len(uids)}"
    )
    return {"img": img, "lm": lm, "hm": hm, "y": y, "uids": uids}


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
        t = torch.from_numpy(img).permute(0, 3, 1, 2).float()
        ds = TensorDataset(t, y_t)
    elif arch == "fusion":
        t_img = torch.from_numpy(img).permute(0, 3, 1, 2).float()
        t_lm = torch.from_numpy(lm).float()
        ds = TensorDataset(t_img, t_lm, y_t)
    else:
        raise ValueError(arch)
    return DataLoader(
        ds, batch_size=BATCH, shuffle=shuffle, num_workers=2, pin_memory=True,
        drop_last=shuffle,
    )


# ─── Training loop ───────────────────────────────────────────────────────────
def train_single(model, arch, tr_loader, va_loader, criterion, lr, save_path):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=8, min_lr=1e-7
    )
    best_val, best_ep, stale = 0.0, 0, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch in tr_loader:
            *x, y = batch
            x = [xi.to(device) for xi in x]
            y = y.to(device)
            out = model(*x) if arch == "fusion" else model(x[0])
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # eval val
        model.eval()
        yt, yp = [], []
        with torch.no_grad():
            for batch in va_loader:
                *x, y = batch
                x = [xi.to(device) for xi in x]
                out = model(*x) if arch == "fusion" else model(x[0])
                yt.append(y.numpy())
                yp.append(out.argmax(1).cpu().numpy())
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


def eval_loader(model, arch, loader):
    model.eval()
    yt, yp, probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            *x, y = batch
            x = [xi.to(device) for xi in x]
            out = model(*x) if arch == "fusion" else model(x[0])
            yt.append(y.numpy())
            yp.append(out.argmax(1).cpu().numpy())
            probs.append(F.softmax(out, dim=1).cpu().numpy())
    return np.concatenate(yt), np.concatenate(yp), np.concatenate(probs)


def metrics_from_pred(yt, yp):
    return {
        "test_macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "test_weighted_f1": float(
            f1_score(yt, yp, average="weighted", zero_division=0)
        ),
        "test_accuracy": float(accuracy_score(yt, yp)),
        "confusion_matrix": confusion_matrix(
            yt, yp, labels=list(range(NUM_CLASSES))
        ).tolist(),
    }


# ─── Per-fold training: each model ───────────────────────────────────────────
def train_eval_single(model_key, train_data, val_data, test_data, ckpt_dir):
    """Train 1 of: fcnn, intermediate_tl. Returns metrics dict."""
    cfg = TOP3[model_key]
    arch = cfg["arch"]
    img_tr, lm_tr, hm_tr, y_tr = train_data
    img_va, lm_va, hm_va, y_va = val_data
    img_te, lm_te, hm_te, y_te = test_data

    weights = class_weights_from_y(y_tr)
    criterion = nn.CrossEntropyLoss(weight=weights)
    model = cfg["builder"]().to(device)

    tr_loader = build_loader(arch, img_tr, lm_tr, hm_tr, y_tr, shuffle=True)
    va_loader = build_loader(arch, img_va, lm_va, hm_va, y_va)
    te_loader = build_loader(arch, img_te, lm_te, hm_te, y_te)

    save_path = ckpt_dir / f"{model_key}.pth"
    best_val, best_ep = train_single(
        model, arch, tr_loader, va_loader, criterion, cfg["lr"], save_path
    )
    model.load_state_dict(
        torch.load(save_path, map_location=device, weights_only=True)
    )
    yt, yp, _ = eval_loader(model, arch, te_loader)
    out = metrics_from_pred(yt, yp)
    out["val_macro_f1"] = float(best_val)
    out["best_epoch"] = int(best_ep)
    save_path.unlink(missing_ok=True)  # cleanup
    return out


def train_eval_late_fusion(train_data, val_data, test_data, ckpt_dir):
    """Train CNN_TL + FCNN, then grid search w di val."""
    img_tr, lm_tr, hm_tr, y_tr = train_data
    img_va, lm_va, hm_va, y_va = val_data
    img_te, lm_te, hm_te, y_te = test_data

    weights = class_weights_from_y(y_tr)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # CNN_TL branch
    cnn = EmotionCNNTransfer(num_classes=NUM_CLASSES).to(device)
    cnn_path = ckpt_dir / "cnn_tl.pth"
    cnn_val, cnn_ep = train_single(
        cnn,
        "cnn",
        build_loader("cnn", img_tr, lm_tr, hm_tr, y_tr, shuffle=True),
        build_loader("cnn", img_va, lm_va, hm_va, y_va),
        criterion,
        LR_TL,
        cnn_path,
    )

    # FCNN branch
    fcnn = EmotionFCNN(num_classes=NUM_CLASSES).to(device)
    fcnn_path = ckpt_dir / "fcnn_lf.pth"
    fcnn_val, fcnn_ep = train_single(
        fcnn,
        "fcnn",
        build_loader("fcnn", img_tr, lm_tr, hm_tr, y_tr, shuffle=True),
        build_loader("fcnn", img_va, lm_va, hm_va, y_va),
        criterion,
        LR_SCRATCH,
        fcnn_path,
    )

    cnn.load_state_dict(torch.load(cnn_path, map_location=device, weights_only=True))
    fcnn.load_state_dict(torch.load(fcnn_path, map_location=device, weights_only=True))

    # Softmax val + test
    _, _, p_val_cnn = eval_loader(cnn, "cnn", build_loader("cnn", img_va, lm_va, hm_va, y_va))
    _, _, p_val_fcnn = eval_loader(fcnn, "fcnn", build_loader("fcnn", img_va, lm_va, hm_va, y_va))
    _, _, p_te_cnn = eval_loader(cnn, "cnn", build_loader("cnn", img_te, lm_te, hm_te, y_te))
    _, _, p_te_fcnn = eval_loader(fcnn, "fcnn", build_loader("fcnn", img_te, lm_te, hm_te, y_te))

    # Grid search w di val
    best_w, best_val_f1 = 0.5, 0.0
    for w in np.arange(0.0, 1.05, 0.05):
        fused = w * p_val_cnn + (1 - w) * p_val_fcnn
        f1 = f1_score(y_va, fused.argmax(1), average="macro", zero_division=0)
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_w = float(w)

    # Apply best w di test
    fused_te = best_w * p_te_cnn + (1 - best_w) * p_te_fcnn
    yp = fused_te.argmax(1)
    out = metrics_from_pred(y_te, yp)
    out["val_macro_f1"] = float(best_val_f1)
    out["best_w_cnn"] = best_w
    out["cnn_val_macro_f1"] = float(cnn_val)
    out["fcnn_val_macro_f1"] = float(fcnn_val)
    out["cnn_best_epoch"] = int(cnn_ep)
    out["fcnn_best_epoch"] = int(fcnn_ep)

    cnn_path.unlink(missing_ok=True)
    fcnn_path.unlink(missing_ok=True)
    return out


def fit_eval(model_key, train_data, val_data, test_data, ckpt_dir):
    if model_key == "late_fusion_tl":
        return train_eval_late_fusion(train_data, val_data, test_data, ckpt_dir)
    return train_eval_single(model_key, train_data, val_data, test_data, ckpt_dir)


# ─── Strategy: Random Split ──────────────────────────────────────────────────
def run_random(D, n_seeds=5):
    out_dir = OUT_BASE / "randomsplit"
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(D["y"])
    for model_key in TOP3.keys():
        out_path = out_dir / f"random_{model_key}.json"
        if out_path.exists():
            existing = json.load(open(out_path))
            done_seeds = {s["seed"] for s in existing.get("per_seed", [])}
            print(f"\n[{model_key}] resume: {len(done_seeds)} seeds done")
        else:
            existing = {"model": model_key, "strategy": "random", "per_seed": []}
            done_seeds = set()

        for seed in range(n_seeds):
            if seed in done_seeds:
                continue
            print(f"\n  → {model_key} seed={seed}")
            t0 = time.time()
            rng = np.random.RandomState(seed)
            idx = rng.permutation(n)
            n_te = int(0.10 * n)
            n_va = int(0.10 * n)
            te_idx = idx[:n_te]
            va_idx = idx[n_te : n_te + n_va]
            tr_idx = idx[n_te + n_va :]

            tr = (D["img"][tr_idx], D["lm"][tr_idx], D["hm"][tr_idx], D["y"][tr_idx])
            va = (D["img"][va_idx], D["lm"][va_idx], D["hm"][va_idx], D["y"][va_idx])
            te = (D["img"][te_idx], D["lm"][te_idx], D["hm"][te_idx], D["y"][te_idx])

            r = fit_eval(model_key, tr, va, te, out_dir)
            r["seed"] = seed
            r["n_train"] = int(len(tr_idx))
            r["n_val"] = int(len(va_idx))
            r["n_test"] = int(len(te_idx))
            r["elapsed_sec"] = round(time.time() - t0, 1)
            existing["per_seed"].append(r)

            # Update summary stats
            f1s = [s["test_macro_f1"] for s in existing["per_seed"]]
            existing["macro_f1_mean"] = float(np.mean(f1s))
            existing["macro_f1_std"] = float(np.std(f1s))
            existing["num_seeds"] = len(f1s)
            json.dump(existing, open(out_path, "w"), indent=2)
            print(
                f"    test_f1={r['test_macro_f1']:.4f}  ({r['elapsed_sec']}s)  "
                f"running mean={existing['macro_f1_mean']:.4f}±{existing['macro_f1_std']:.4f}"
            )


# ─── Strategy: 5-Fold CV (subject-wise) ──────────────────────────────────────
def run_cv5(D):
    out_dir = OUT_BASE / "crossval"
    out_dir.mkdir(parents=True, exist_ok=True)

    users = sorted(set(D["uids"].tolist()))
    rng = np.random.RandomState(SEED)
    rng.shuffle(users)
    # Bagi 37 users ke 5 grup (~7-8 per fold)
    fold_users = np.array_split(users, 5)

    for model_key in TOP3.keys():
        out_path = out_dir / f"cv5_{model_key}.json"
        if out_path.exists():
            existing = json.load(open(out_path))
            done_folds = {s["fold"] for s in existing.get("per_fold", [])}
            print(f"\n[{model_key}] resume: {len(done_folds)} folds done")
        else:
            existing = {
                "model": model_key,
                "strategy": "cv5_subject_wise",
                "per_fold": [],
            }
            done_folds = set()

        for fold_i, te_users in enumerate(fold_users):
            if fold_i in done_folds:
                continue
            te_users_set = set(te_users.tolist())
            te_mask = np.array([u in te_users_set for u in D["uids"]])
            # Val = sample 10% dari training (random, by user)
            train_users = [u for u in users if u not in te_users_set]
            n_val_users = max(2, int(0.15 * len(train_users)))
            rng2 = np.random.RandomState(SEED + fold_i)
            shuffled = list(train_users)
            rng2.shuffle(shuffled)
            va_users_set = set(shuffled[:n_val_users])
            va_mask = np.array([u in va_users_set for u in D["uids"]])
            tr_mask = ~te_mask & ~va_mask

            print(
                f"\n  → {model_key} fold={fold_i+1}/5  "
                f"test_users={len(te_users_set)}  val_users={len(va_users_set)}  "
                f"train={tr_mask.sum()}  val={va_mask.sum()}  test={te_mask.sum()}"
            )
            t0 = time.time()
            tr = tuple(D[k][tr_mask] for k in ("img", "lm", "hm", "y"))
            va = tuple(D[k][va_mask] for k in ("img", "lm", "hm", "y"))
            te = tuple(D[k][te_mask] for k in ("img", "lm", "hm", "y"))
            r = fit_eval(model_key, tr, va, te, out_dir)
            r["fold"] = fold_i
            r["test_users"] = sorted(list(te_users_set))
            r["val_users"] = sorted(list(va_users_set))
            r["n_train"] = int(tr_mask.sum())
            r["n_val"] = int(va_mask.sum())
            r["n_test"] = int(te_mask.sum())
            r["elapsed_sec"] = round(time.time() - t0, 1)
            existing["per_fold"].append(r)

            f1s = [s["test_macro_f1"] for s in existing["per_fold"]]
            existing["macro_f1_mean"] = float(np.mean(f1s))
            existing["macro_f1_std"] = float(np.std(f1s))
            existing["num_folds"] = len(f1s)
            json.dump(existing, open(out_path, "w"), indent=2)
            print(
                f"    test_f1={r['test_macro_f1']:.4f}  ({r['elapsed_sec']}s)  "
                f"running mean={existing['macro_f1_mean']:.4f}±{existing['macro_f1_std']:.4f}"
            )


# ─── Strategy: LOSO 37-fold ──────────────────────────────────────────────────
def run_loso(D):
    out_dir = OUT_BASE / "loso"
    out_dir.mkdir(parents=True, exist_ok=True)

    users = sorted(set(D["uids"].tolist()))

    for model_key in TOP3.keys():
        out_path = out_dir / f"loso_{model_key}.json"
        if out_path.exists():
            existing = json.load(open(out_path))
            done_users = {s["test_user"] for s in existing.get("per_fold", [])}
            print(f"\n[{model_key}] resume: {len(done_users)}/{len(users)} folds done")
        else:
            existing = {"model": model_key, "strategy": "loso", "per_fold": []}
            done_users = set()

        for i, te_user in enumerate(users):
            if te_user in done_users:
                continue
            te_mask = D["uids"] == te_user
            te_y = D["y"][te_mask]
            if len(set(te_y.tolist())) < 2:
                print(
                    f"\n  → {model_key} fold={i+1}/{len(users)} user_{te_user} "
                    f"SKIP (1 class only)"
                )
                continue

            # val = 2 random users from train
            train_users = [u for u in users if u != te_user]
            rng2 = np.random.RandomState(SEED + i)
            shuffled = list(train_users)
            rng2.shuffle(shuffled)
            va_users_set = set(shuffled[:2])
            va_mask = np.array([u in va_users_set for u in D["uids"]])
            tr_mask = (~te_mask) & (~va_mask)

            print(
                f"\n  → {model_key} fold={i+1}/{len(users)} user_{te_user}  "
                f"train={tr_mask.sum()} val={va_mask.sum()} test={te_mask.sum()}"
            )
            t0 = time.time()
            tr = tuple(D[k][tr_mask] for k in ("img", "lm", "hm", "y"))
            va = tuple(D[k][va_mask] for k in ("img", "lm", "hm", "y"))
            te = tuple(D[k][te_mask] for k in ("img", "lm", "hm", "y"))
            r = fit_eval(model_key, tr, va, te, out_dir)
            r["test_user"] = te_user
            r["val_users"] = sorted(list(va_users_set))
            r["n_train"] = int(tr_mask.sum())
            r["n_val"] = int(va_mask.sum())
            r["n_test"] = int(te_mask.sum())
            r["elapsed_sec"] = round(time.time() - t0, 1)
            existing["per_fold"].append(r)

            f1s = [s["test_macro_f1"] for s in existing["per_fold"]]
            existing["macro_f1_mean"] = float(np.mean(f1s))
            existing["macro_f1_std"] = float(np.std(f1s))
            existing["num_folds"] = len(f1s)
            json.dump(existing, open(out_path, "w"), indent=2)
            print(
                f"    test_f1={r['test_macro_f1']:.4f}  ({r['elapsed_sec']}s)  "
                f"running mean={existing['macro_f1_mean']:.4f}±{existing['macro_f1_std']:.4f}"
            )


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy", required=True, choices=["random", "cv5", "loso"]
    )
    parser.add_argument("--seeds", type=int, default=5, help="for --strategy random")
    args = parser.parse_args()

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Strategy: {args.strategy}")
    print(f"Output base: {OUT_BASE}")

    print("\nLoading data...")
    D = load_all()
    print(
        f"Total: {len(D['y'])} samples, "
        f"{len(set(D['uids'].tolist()))} users, "
        f"class dist: {np.bincount(D['y'], minlength=NUM_CLASSES).tolist()}"
    )

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    if args.strategy == "random":
        run_random(D, n_seeds=args.seeds)
    elif args.strategy == "cv5":
        run_cv5(D)
    elif args.strategy == "loso":
        run_loso(D)


if __name__ == "__main__":
    main()
