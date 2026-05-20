"""
Robustness Evaluation — Unified Top-3 (Primer conf60)
======================================================

LOSO + 5-Fold CV (subject-wise) untuk top-3 baru per scheme dari unified sweep:

3-class (positive / neutral / negative), scenario B1:
  - A. Landmark FCNN (facs_28, FA)
  - B. Intermediate Fusion scratch (facs_28, FA)
  - C. Late Fusion TL (facs_28, FA)  ← image_branch=cnn_tl, landmark_branch=fcnn_facs_28_fa

7-class (neutral/happy/sad/angry/fearful/disgusted/surprised), scenario B1:
  - D. Landmark FCNN (facs_plus_bs_80, FA-hybrid)
  - E. Intermediate Fusion scratch (facs_28, FA)
  - F. Late Fusion TL (facs_plus_bs_80, FA-hybrid)

Usage:
  python scripts/run_robustness_unified.py --strategy loso
  python scripts/run_robustness_unified.py --strategy cv5
  python scripts/run_robustness_unified.py --strategy loso --configs A B C --gpu 0

Output:
  models/frontonly_conf60/robustness/{loso,cv5}/{config}_per_fold.json
  models/frontonly_conf60/robustness/{loso,cv5}/{config}_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.models import (  # noqa: E402
    EmotionCNN, EmotionFCNN, IntermediateFusion,
    EmotionCNNTransfer, IntermediateFusionTransfer,
)

# ── DATA & CONFIG ────────────────────────────────────────

DATA_DIR = PROJECT_ROOT / "data" / "dataset_frontonly_conf60"
OUT_ROOT = PROJECT_ROOT / "models" / "frontonly_conf60" / "robustness"

BATCH = 32
EPOCHS = 50
PATIENCE = 15
SEED = 42

# 7c label → 3c valence: positive=happy/surprised, neutral, negative=others
REMAP_3 = np.array([1, 0, 2, 2, 2, 2, 0], dtype=np.int64)

# FACS pair definitions (mirror scripts/run_unified_landmark.py)
FACS_PAIRS = [
    ("AU1_left",  17, 21), ("AU1_right", 22, 26),
    ("AU2_left",  20, 39), ("AU2_right", 23, 42),
    ("AU4_left",  21, 39), ("AU4_right", 22, 42),
    ("AU5_left",  37, 41), ("AU5_right", 43, 47),
    ("AU6_left",   1, 41), ("AU6_right", 15, 47),
    ("AU7_left",  37, 40), ("AU7_right", 43, 46),
    ("AU9_left",  31, 39), ("AU9_right", 35, 42),
    ("AU10_top",   33, 51), ("AU10_left",  31, 48), ("AU10_right", 35, 54),
    ("AU12_left",  48, 36), ("AU12_right", 54, 45),
    ("AU15_left",  48,  4), ("AU15_right", 54, 12),
    ("AU17_left",   8, 57), ("AU17_right",  8, 51),
    ("AU20_left",  48, 60), ("AU20_right", 54, 64),
    ("AU23_outer", 48, 54), ("AU24_top",   51, 57),
    ("AU25",       62, 66),
]
INTEROCULAR_PAIR = (36, 45)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Feature computation ──────────────────────────────────

def compute_facs_distances(landmarks_flat):
    """(N, 136) → (N, 28) FACS Euclidean distances normalized by interocular."""
    pts = landmarks_flat.reshape(-1, 68, 2)
    a, b = INTEROCULAR_PAIR
    iod = np.maximum(np.linalg.norm(pts[:, a] - pts[:, b], axis=1), 1e-6)
    out = np.zeros((len(pts), len(FACS_PAIRS)), dtype=np.float32)
    for i, (_, p1, p2) in enumerate(FACS_PAIRS):
        d = np.linalg.norm(pts[:, p1] - pts[:, p2], axis=1)
        out[:, i] = (d / iod).astype(np.float32)
    return out


def load_all():
    """Load all primer data (concatenated across train/val/test) for fold splitting."""
    parts_img, parts_fa, parts_bs, parts_y = [], [], [], []
    for sp in ("train", "val", "test"):
        parts_img.append(np.load(DATA_DIR / f"X_{sp}_images.npy"))
        parts_fa.append(np.load(DATA_DIR / f"X_{sp}_faceapi_landmarks.npy").astype(np.float32))
        parts_bs.append(np.load(DATA_DIR / f"X_{sp}_mp_blendshapes.npy").astype(np.float32))
        parts_y.append(np.load(DATA_DIR / f"y_{sp}.npy").astype(np.int64))

    images = np.concatenate(parts_img, axis=0)
    fa_lm = np.concatenate(parts_fa, axis=0)        # (N, 136) FA landmarks
    bs = np.concatenate(parts_bs, axis=0)           # (N, 52) blendshapes (MP)
    y7 = np.concatenate(parts_y, axis=0)            # (N,) 7-class labels
    uids = np.load(DATA_DIR / "user_ids_all.npy", allow_pickle=True)

    # Derived features
    facs28 = compute_facs_distances(fa_lm)          # (N, 28) FA-derived FACS distances
    fb80 = np.concatenate([facs28, bs], axis=1)     # (N, 80) FACS + blendshape

    assert len(images) == len(uids) == len(y7) == len(facs28) == len(fb80)
    return {
        "images": images, "fa_lm": fa_lm, "facs28": facs28, "bs": bs, "fb80": fb80,
        "y7": y7, "uids": uids,
    }


# ── Config catalog ───────────────────────────────────────

CONFIGS = {
    # 3-class
    "A": {"scheme": "3c", "arch": "fcnn",         "feature": "facs_28", "source": "FA",
          "input_dim": 28, "input_kind": "facs28", "label": "Landmark FCNN (facs_28, FA)"},
    "B": {"scheme": "3c", "arch": "intermediate", "feature": "facs_28", "source": "FA",
          "input_dim": 28, "input_kind": "facs28", "label": "Intermediate scratch (facs_28, FA)"},
    "C": {"scheme": "3c", "arch": "late_tl",      "feature": "facs_28", "source": "FA",
          "input_dim": 28, "input_kind": "facs28", "label": "Late Fusion TL (facs_28, FA)"},
    # 7-class
    "D": {"scheme": "7c", "arch": "fcnn",         "feature": "facs_plus_bs_80", "source": "FA-hybrid",
          "input_dim": 80, "input_kind": "fb80",   "label": "Landmark FCNN (FB80, FA-hybrid)"},
    "E": {"scheme": "7c", "arch": "intermediate", "feature": "facs_28", "source": "FA",
          "input_dim": 28, "input_kind": "facs28", "label": "Intermediate scratch (facs_28, FA)"},
    "F": {"scheme": "7c", "arch": "late_tl",      "feature": "facs_plus_bs_80", "source": "FA-hybrid",
          "input_dim": 80, "input_kind": "fb80",   "label": "Late Fusion TL (FB80, FA-hybrid)"},
}


# ── Training core ────────────────────────────────────────

def make_loader(*tensors, batch_size=BATCH, shuffle=False):
    ds = TensorDataset(*tensors)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=True, drop_last=False)


def train_one(model, train_loader, val_loader, criterion, optimizer,
              device, epochs=EPOCHS, patience=PATIENCE,
              input_keys=("x",)):
    """Generic train loop. input_keys = ('x',) for unimodal or ('img','lm') for fusion."""
    best_val_mf1 = -1.0
    best_state = None
    no_imp = 0
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for batch in train_loader:
            *inputs, y = batch
            inputs = [t.to(device, non_blocking=True) for t in inputs]
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            out = model(*inputs)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
            n += len(y)

        model.eval()
        all_p, all_y = [], []
        with torch.no_grad():
            for batch in val_loader:
                *inputs, y = batch
                inputs = [t.to(device, non_blocking=True) for t in inputs]
                p = model(*inputs).argmax(1).cpu().numpy()
                all_p.append(p); all_y.append(y.numpy())
        all_p = np.concatenate(all_p); all_y = np.concatenate(all_y)
        val_mf1 = f1_score(all_y, all_p, average="macro", zero_division=0)

        if val_mf1 > best_val_mf1:
            best_val_mf1 = val_mf1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_mf1, ep


def evaluate(model, loader, device):
    model.eval()
    all_p, all_y, all_soft = [], [], []
    with torch.no_grad():
        for batch in loader:
            *inputs, y = batch
            inputs = [t.to(device, non_blocking=True) for t in inputs]
            logits = model(*inputs)
            soft = torch.softmax(logits, dim=1).cpu().numpy()
            all_p.append(logits.argmax(1).cpu().numpy())
            all_y.append(y.numpy())
            all_soft.append(soft)
    y = np.concatenate(all_y)
    p = np.concatenate(all_p)
    soft = np.concatenate(all_soft, axis=0)
    return {
        "acc": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, p, average="weighted", zero_division=0)),
        "soft": soft, "y_true": y, "y_pred": p,
    }


# ── Model builder for each config ────────────────────────

def build_landmark_model(input_dim, num_classes):
    return EmotionFCNN(input_dim=input_dim, num_classes=num_classes)


def build_intermediate_model(input_dim, num_classes):
    return IntermediateFusion(num_classes=num_classes, landmark_dim=input_dim)


def fit_config_fold(cfg, data, tr_idx, val_idx, te_idx, num_classes, device):
    """Fit a single config on a single fold. Returns metrics dict."""
    arch = cfg["arch"]
    input_kind = cfg["input_kind"]   # facs28 or fb80
    in_dim = cfg["input_dim"]

    # Prepare label
    if num_classes == 3:
        y_all = REMAP_3[data["y7"]]
    else:
        y_all = data["y7"]

    if input_kind == "facs28":
        lm_feat = data["facs28"]
    elif input_kind == "fb80":
        lm_feat = data["fb80"]
    else:
        raise ValueError(input_kind)

    lm_t = torch.from_numpy(lm_feat)
    y_t = torch.from_numpy(y_all)

    if arch in ("fcnn", "intermediate") or arch == "late_tl":
        img_t = torch.from_numpy(data["images"]).permute(0, 3, 1, 2).float()

    def slice_idx(idx):
        return idx

    set_seed(SEED)

    if arch == "fcnn":
        tr_loader = make_loader(lm_t[tr_idx], y_t[tr_idx], shuffle=True)
        va_loader = make_loader(lm_t[val_idx], y_t[val_idx])
        te_loader = make_loader(lm_t[te_idx], y_t[te_idx])
        model = build_landmark_model(in_dim, num_classes).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=0.0001)
        crit = nn.CrossEntropyLoss()
        model, best_val, _ = train_one(model, tr_loader, va_loader, crit, opt, device)
        result = evaluate(model, te_loader, device)
        return _strip(result, best_val)

    if arch == "intermediate":
        tr_loader = make_loader(img_t[tr_idx], lm_t[tr_idx], y_t[tr_idx], shuffle=True)
        va_loader = make_loader(img_t[val_idx], lm_t[val_idx], y_t[val_idx])
        te_loader = make_loader(img_t[te_idx], lm_t[te_idx], y_t[te_idx])
        model = build_intermediate_model(in_dim, num_classes).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=0.0001)
        crit = nn.CrossEntropyLoss()
        model, best_val, _ = train_one(model, tr_loader, va_loader, crit, opt, device)
        result = evaluate(model, te_loader, device)
        return _strip(result, best_val)

    if arch == "late_tl":
        # Train CNN TL + FCNN, then weighted softmax
        tr_img = make_loader(img_t[tr_idx], y_t[tr_idx], shuffle=True)
        va_img = make_loader(img_t[val_idx], y_t[val_idx])
        cnn = EmotionCNNTransfer(num_classes=num_classes, pretrained=True).to(device)
        opt = torch.optim.Adam(cnn.parameters(), lr=0.00005)
        crit = nn.CrossEntropyLoss()
        cnn, _, _ = train_one(cnn, tr_img, va_img, crit, opt, device)

        tr_lm = make_loader(lm_t[tr_idx], y_t[tr_idx], shuffle=True)
        va_lm = make_loader(lm_t[val_idx], y_t[val_idx])
        fcnn = EmotionFCNN(input_dim=in_dim, num_classes=num_classes).to(device)
        opt = torch.optim.Adam(fcnn.parameters(), lr=0.0001)
        fcnn, _, _ = train_one(fcnn, tr_lm, va_lm, crit, opt, device)

        # Weight sweep on val to pick w_image
        va_img_full = make_loader(img_t[val_idx], y_t[val_idx])
        va_lm_full = make_loader(lm_t[val_idx], y_t[val_idx])
        cnn_soft_val = _softmax_only(cnn, va_img_full, device)
        fcnn_soft_val = _softmax_only(fcnn, va_lm_full, device)
        y_val = y_t[val_idx].numpy()
        best_w, best_v_mf1 = 0.5, -1.0
        for w in np.arange(0.0, 1.05, 0.05):
            fused = w * cnn_soft_val + (1 - w) * fcnn_soft_val
            v = f1_score(y_val, fused.argmax(1), average="macro", zero_division=0)
            if v > best_v_mf1:
                best_v_mf1, best_w = v, float(w)

        te_img_full = make_loader(img_t[te_idx], y_t[te_idx])
        te_lm_full = make_loader(lm_t[te_idx], y_t[te_idx])
        cnn_soft_te = _softmax_only(cnn, te_img_full, device)
        fcnn_soft_te = _softmax_only(fcnn, te_lm_full, device)
        fused_te = best_w * cnn_soft_te + (1 - best_w) * fcnn_soft_te
        y_te = y_t[te_idx].numpy()
        p_te = fused_te.argmax(1)
        return {
            "acc": float(accuracy_score(y_te, p_te)),
            "macro_f1": float(f1_score(y_te, p_te, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_te, p_te, average="weighted", zero_division=0)),
            "best_w_image": best_w, "val_mf1_at_best_w": float(best_v_mf1),
        }

    raise ValueError(arch)


def _strip(result, best_val):
    return {
        "acc": result["acc"], "macro_f1": result["macro_f1"], "weighted_f1": result["weighted_f1"],
        "val_mf1_best": float(best_val),
    }


def _softmax_only(model, loader, device):
    model.eval()
    outs = []
    with torch.no_grad():
        for batch in loader:
            *inputs, y = batch
            inputs = [t.to(device) for t in inputs]
            outs.append(torch.softmax(model(*inputs), dim=1).cpu().numpy())
    return np.concatenate(outs, axis=0)


# ── Fold strategies ──────────────────────────────────────

def loso_folds(uids):
    """Yield (fold_idx, fold_label, tr_idx, val_idx, te_idx). 1 user test, 2 user val, rest train."""
    uniq = np.array(sorted(set(uids)))
    rng = np.random.default_rng(SEED)
    for i, u_test in enumerate(uniq):
        rest = [u for u in uniq if u != u_test]
        rest = list(rest)
        rng.shuffle(rest)
        u_val = rest[:2]
        u_tr = rest[2:]
        te_idx = np.where(uids == u_test)[0]
        val_idx = np.where(np.isin(uids, u_val))[0]
        tr_idx = np.where(np.isin(uids, u_tr))[0]
        yield i, str(u_test), tr_idx, val_idx, te_idx


def cv5_folds(uids, n_splits=5):
    """5-fold CV, subject-wise."""
    uniq = np.array(sorted(set(uids)))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for i, (tr_user_i, te_user_i) in enumerate(kf.split(uniq)):
        u_test = uniq[te_user_i]
        u_train_all = uniq[tr_user_i]
        # take 20% of train users as val
        rng = np.random.default_rng(SEED + i)
        u_train_all = list(u_train_all)
        rng.shuffle(u_train_all)
        n_val = max(1, len(u_train_all) // 5)
        u_val = np.array(u_train_all[:n_val])
        u_tr = np.array(u_train_all[n_val:])
        te_idx = np.where(np.isin(uids, u_test))[0]
        val_idx = np.where(np.isin(uids, u_val))[0]
        tr_idx = np.where(np.isin(uids, u_tr))[0]
        yield i, ",".join(map(str, u_test)), tr_idx, val_idx, te_idx


def randomsplit_folds(uids, y7, n_seeds=5, ratios=(0.80, 0.10, 0.10)):
    """Random Split per-SAMPLE stratified (NOT subject-wise) — upper bound.

    Yields (seed_idx, label, tr_idx, val_idx, te_idx) for each of n_seeds runs.
    Stratification key = 7-class label (consistent across schemes; mapping to 3c
    happens later in fit_config_fold).
    """
    n = len(uids)
    assert sum(ratios) == 1.0
    for s in range(n_seeds):
        rng = np.random.default_rng(SEED + s)
        # Stratified by y7
        tr_idx_list, val_idx_list, te_idx_list = [], [], []
        for c in np.unique(y7):
            cls_idx = np.where(y7 == c)[0]
            cls_idx = cls_idx.copy()
            rng.shuffle(cls_idx)
            n_c = len(cls_idx)
            n_tr = int(n_c * ratios[0])
            n_val = int(n_c * ratios[1])
            tr_idx_list.append(cls_idx[:n_tr])
            val_idx_list.append(cls_idx[n_tr:n_tr + n_val])
            te_idx_list.append(cls_idx[n_tr + n_val:])
        tr = np.concatenate(tr_idx_list); rng.shuffle(tr)
        va = np.concatenate(val_idx_list); rng.shuffle(va)
        te = np.concatenate(te_idx_list); rng.shuffle(te)
        yield s, f"seed{s}", tr, va, te


# ── Main driver ──────────────────────────────────────────

def run_config(config_key, strategy, data, device, out_dir):
    cfg = CONFIGS[config_key]
    num_classes = 3 if cfg["scheme"] == "3c" else 7
    if strategy == "loso":
        fold_gen = loso_folds(data["uids"])
    elif strategy == "cv5":
        fold_gen = cv5_folds(data["uids"])
    elif strategy == "randomsplit":
        fold_gen = randomsplit_folds(data["uids"], data["y7"])
    else:
        raise ValueError(strategy)
    folds = list(fold_gen)

    print(f"\n{'='*72}")
    print(f"  CONFIG {config_key}: {cfg['label']} | scheme={cfg['scheme']} | strategy={strategy}")
    print(f"  Total folds: {len(folds)}")
    print('=' * 72)

    per_fold = []
    t0 = time.time()
    for i, fold_label, tr_idx, val_idx, te_idx in folds:
        print(f"\n[Fold {i+1}/{len(folds)}] test_user={fold_label}  "
              f"n_tr={len(tr_idx)} n_val={len(val_idx)} n_te={len(te_idx)}")
        t_fold = time.time()
        try:
            metrics = fit_config_fold(cfg, data, tr_idx, val_idx, te_idx, num_classes, device)
            metrics["fold_idx"] = i
            metrics["fold_label"] = fold_label
            metrics["n_tr"] = int(len(tr_idx))
            metrics["n_val"] = int(len(val_idx))
            metrics["n_te"] = int(len(te_idx))
            metrics["elapsed_sec"] = float(time.time() - t_fold)
            per_fold.append(metrics)
            print(f"  → mf1={metrics['macro_f1']:.4f}  wf1={metrics['weighted_f1']:.4f}  "
                  f"acc={metrics['acc']:.4f}  ({metrics['elapsed_sec']:.0f}s)")
        except Exception as e:
            print(f"  ✗ Fold {i} FAILED: {type(e).__name__}: {e}")
            per_fold.append({"fold_idx": i, "fold_label": fold_label,
                              "error": f"{type(e).__name__}: {e}"})

    # Aggregate
    valid = [r for r in per_fold if "macro_f1" in r]
    if valid:
        mf1 = np.array([r["macro_f1"] for r in valid])
        wf1 = np.array([r["weighted_f1"] for r in valid])
        acc = np.array([r["acc"] for r in valid])
        summary = {
            "config_key": config_key, "config": cfg, "strategy": strategy,
            "n_folds_done": len(valid), "n_folds_total": len(per_fold),
            "test_macro_f1_mean": float(mf1.mean()), "test_macro_f1_std": float(mf1.std()),
            "test_weighted_f1_mean": float(wf1.mean()), "test_weighted_f1_std": float(wf1.std()),
            "test_accuracy_mean": float(acc.mean()), "test_accuracy_std": float(acc.std()),
            "elapsed_sec_total": float(time.time() - t0),
        }
    else:
        summary = {"config_key": config_key, "config": cfg, "strategy": strategy,
                   "error": "all folds failed", "n_folds_total": len(per_fold)}

    base = f"{config_key}_{cfg['arch']}_{cfg['feature']}_{cfg['source']}_{cfg['scheme']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{base}_per_fold.json").write_text(json.dumps(per_fold, indent=2))
    (out_dir / f"{base}_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n[✓] {strategy} {config_key} done: "
          f"mf1={summary.get('test_macro_f1_mean', float('nan')):.4f} ± "
          f"{summary.get('test_macro_f1_std', float('nan')):.4f} "
          f"(elapsed {summary.get('elapsed_sec_total', 0):.0f}s)")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=["loso", "cv5", "randomsplit"], required=True)
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()),
                    help="Subset of config keys (default: all A–F)")
    ap.add_argument("--gpu", type=int, default=None,
                    help="CUDA_VISIBLE_DEVICES override (set to int)")
    args = ap.parse_args()

    if args.gpu is not None:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')})"
          if args.gpu is not None else f"Device: {device}")

    print(f"Loading primer conf60 data ...")
    t0 = time.time()
    data = load_all()
    print(f"  Loaded N={len(data['uids'])} samples, {len(set(data['uids']))} users  "
          f"({time.time() - t0:.1f}s)")

    out_dir = OUT_ROOT / args.strategy

    summaries = []
    for k in args.configs:
        if k not in CONFIGS:
            print(f"[!] Skip unknown config key: {k}")
            continue
        s = run_config(k, args.strategy, data, device, out_dir)
        summaries.append(s)

    # Master summary
    master = {
        "strategy": args.strategy, "n_configs": len(summaries),
        "configs_run": [s["config_key"] for s in summaries],
        "summaries": summaries,
    }
    (out_dir / "master_summary.json").write_text(json.dumps(master, indent=2))
    print(f"\n[✓] {args.strategy} complete. Master summary: {out_dir / 'master_summary.json'}")


if __name__ == "__main__":
    import os  # for CUDA_VISIBLE_DEVICES print
    main()
