#!/usr/bin/env python3
"""
Unified protocol untuk feature yang punya komponen non-landmark
(Blendshape, FACS+Blendshape). Melengkapi `run_unified_landmark.py`
yang fokus di raw_136 dan FACS_28.

UNIFIED SCENARIO DEFINITION (consistent dengan run_unified_landmark.py):
  B1: no aug, shuffle uniform                                          (baseline)
  B2: WeightedRandomSampler (class-balanced)                           (handle imbalance)
  B3: WeightedRandomSampler + per-batch random aug                     (balance + diversity)

Aug strategy per feature:
  - Blendshape 52: B3 = B2 + Gaussian noise σ=0.02 di koefisien blendshape
    (no geometric aug — blendshape sudah high-level, geometric transform
    di space ARKit coefficient tidak meaningful)
  - FACS+BS 80: B3 = B2 + landmark aug → recompute FACS distance + noise σ=0.02
    di komponen blendshape

Coverage:
  Features:  {blendshape_52, facs_plus_bs_80}
  Sources:   {mediapipe, faceapi}
    (blendshape selalu MP-source, FACS bisa MP/FA)
  Archs:     {fcnn, cnn1d}
  Scenarios: {B1, B2, B3}
  Schemes:   {3c, 7c}

Usage:
  CUDA_VISIBLE_DEVICES=2 python scripts/run_unified_derived.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from training.models import EmotionCNN1D_FACS, EmotionFCNN  # noqa: E402
from training.exp_utils import class_counts, evaluate_full, make_run_record  # noqa: E402
from training.landmark_aug import augment_landmark_136, make_balanced_sampler  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "dataset_frontonly_conf60"
REMAP_3 = np.array([1, 0, 2, 2, 2, 2, 0], dtype=np.int64)
CLASS_NAMES_7 = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]
CLASS_NAMES_3 = ["positive", "neutral", "negative"]

FACS_PAIRS = [
    ("AU1_inner_brow_R", 21, 39), ("AU1_inner_brow_L", 22, 42),
    ("AU2_outer_brow_R", 17, 36), ("AU2_outer_brow_L", 26, 45),
    ("AU4_frown_inner_brows", 21, 22), ("AU4_brow_to_eye_R", 19, 37),
    ("AU4_brow_to_eye_L", 24, 44),
    ("AU5_eye_open_R_1", 37, 41), ("AU5_eye_open_R_2", 38, 40),
    ("AU5_eye_open_L_1", 43, 47), ("AU5_eye_open_L_2", 44, 46),
    ("AU6_cheek_R", 36, 31), ("AU6_cheek_L", 45, 35),
    ("AU9_nose_width", 31, 35), ("AU9_nose_bridge", 27, 30),
    ("AU10_nose_to_upperlip", 33, 51), ("AU10_lip_top_to_nose_R", 50, 33),
    ("AU10_lip_top_to_nose_L", 52, 33),
    ("AU12_corner_R_to_eye", 48, 36), ("AU12_corner_L_to_eye", 54, 45),
    ("AU15_corner_R_to_chin", 48, 8), ("AU15_corner_L_to_chin", 54, 8),
    ("AU20_mouth_width_outer", 48, 54), ("AU23_mouth_width_inner", 60, 64),
    ("AU25_lip_open_outer", 51, 57), ("AU25_lip_open_inner", 62, 66),
    ("AU26_nose_to_chin", 33, 8), ("face_height", 8, 27),
]
INTEROCULAR_PAIR = (36, 45)

BATCH = 32
EPOCHS = 50
PATIENCE = 15
LR = 1e-3
SEED = 42
BLENDSHAPE_NOISE_SIGMA = 0.02

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def compute_facs_distances(landmarks_flat):
    pts = landmarks_flat.reshape(-1, 68, 2)
    a, b = INTEROCULAR_PAIR
    iod = np.maximum(np.linalg.norm(pts[:, a] - pts[:, b], axis=1), 1e-6)
    out = np.zeros((len(pts), len(FACS_PAIRS)), dtype=np.float32)
    for i, (_, p1, p2) in enumerate(FACS_PAIRS):
        d = np.linalg.norm(pts[:, p1] - pts[:, p2], axis=1)
        out[:, i] = (d / iod).astype(np.float32)
    return out


def load_blendshape_with_impute(split: str):
    bs = np.load(DATA_DIR / f"X_{split}_mp_blendshapes.npy").astype(np.float32)
    return bs


def load_landmark(source: str, split: str):
    fname = "X_{}_landmarks.npy" if source == "mediapipe" else "X_{}_faceapi_landmarks.npy"
    return np.load(DATA_DIR / fname.format(split)).astype(np.float32)


class BlendshapeDataset(Dataset):
    """52-dim blendshape only. Per-batch noise aug for B3."""

    def __init__(self, X_bs, y, *, augment=False, seed=42):
        self.X = X_bs.astype(np.float32)
        self.y = y.astype(np.int64)
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        bs = self.X[i].copy()
        if self.augment:
            bs = bs + self.rng.normal(0, BLENDSHAPE_NOISE_SIGMA, size=bs.shape).astype(np.float32)
            bs = np.clip(bs, 0.0, 1.0)
        return torch.from_numpy(bs), torch.tensor(self.y[i])


class FACSPlusBSDataset(Dataset):
    """80-dim: 28 FACS distance (recompute per call after landmark aug) + 52 blendshape."""

    def __init__(self, X_lm, X_bs, y, *, augment=False, seed=42):
        self.X_lm = X_lm.astype(np.float32)
        self.X_bs = X_bs.astype(np.float32)
        self.y = y.astype(np.int64)
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        lm = self.X_lm[i]
        bs = self.X_bs[i].copy()
        if self.augment:
            lm = augment_landmark_136(lm, self.rng)
            bs = bs + self.rng.normal(0, BLENDSHAPE_NOISE_SIGMA, size=bs.shape).astype(np.float32)
            bs = np.clip(bs, 0.0, 1.0)
        facs = compute_facs_distances(lm[None, :])[0]
        feat = np.concatenate([facs, bs])
        return torch.from_numpy(feat.astype(np.float32)), torch.tensor(self.y[i])


def build_data(feature, source, num_classes):
    out = {}
    for split in ("train", "val", "test"):
        y = np.load(DATA_DIR / f"y_{split}.npy").astype(np.int64)
        if num_classes == 3:
            y = REMAP_3[y]
        bs = load_blendshape_with_impute(split)
        # Impute NaN with training median
        if split == "train":
            bs_train = bs
            med = np.nanmedian(bs_train, axis=0)
            med = np.where(np.isnan(med), 0.0, med)
        bs = np.where(np.isnan(bs), med, bs).astype(np.float32)

        if feature == "blendshape_52":
            out[split] = (bs, y)
        elif feature == "facs_plus_bs_80":
            lm = load_landmark(source, split)
            assert not np.isnan(lm).any()
            out[split] = (lm, bs, y)
    return out


def make_dataset(feature, data_split, augment):
    if feature == "blendshape_52":
        X, y = data_split
        return BlendshapeDataset(X, y, augment=augment, seed=SEED), y, X.shape[1]
    elif feature == "facs_plus_bs_80":
        lm, bs, y = data_split
        ds = FACSPlusBSDataset(lm, bs, y, augment=augment, seed=SEED)
        return ds, y, 80


def build_model(arch, in_dim, num_classes):
    if arch == "fcnn":
        return EmotionFCNN(input_dim=in_dim, num_classes=num_classes)
    if arch == "cnn1d":
        return EmotionCNN1D_FACS(num_classes=num_classes, in_dim=in_dim)
    raise ValueError(arch)


def train_run(feature, source, arch, scenario, num_classes, data):
    cls_names = CLASS_NAMES_3 if num_classes == 3 else CLASS_NAMES_7
    augment_train = scenario == "b3"

    train_ds, ytr, in_dim = make_dataset(feature, data["train"], augment=augment_train)
    val_ds, yva, _ = make_dataset(feature, data["val"], augment=False)
    test_ds, yte, _ = make_dataset(feature, data["test"], augment=False)

    print(f"\n==== {feature} × {source} × {arch} × {num_classes}c × {scenario.upper()} ====")
    print(f"  Train N={len(ytr)}  in_dim={in_dim}  augment={augment_train}")

    if scenario in ("b2", "b3"):
        sampler = make_balanced_sampler(ytr, num_classes)
        tl = DataLoader(train_ds, batch_size=BATCH, sampler=sampler,
                        drop_last=True, num_workers=0, pin_memory=True)
    else:
        tl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                        drop_last=True, num_workers=0, pin_memory=True)
    vl = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=True)
    el = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=True)

    set_seed(SEED)
    model = build_model(arch, in_dim, num_classes).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()

    record = make_run_record(
        config=f"unified_{feature}_{source}_{arch}_{num_classes}c_{scenario}",
        notes=("Unified protocol for derived features. "
               "Blendshape: B3 = B2 + Gaussian noise σ=0.02 on coefficient. "
               "FACS+BS: B3 = B2 + landmark aug (recompute FACS) + noise on BS."),
        hyperparams={"batch_size": BATCH, "epochs_max": EPOCHS, "patience": PATIENCE,
                     "lr": LR, "optimizer": "Adam", "loss": "CrossEntropyLoss",
                     "seed": SEED, "scenario": scenario.upper(),
                     "uses_weighted_sampler": scenario in ("b2", "b3"),
                     "uses_per_batch_aug": scenario == "b3",
                     "blendshape_noise_sigma": (BLENDSHAPE_NOISE_SIGMA
                                                 if scenario == "b3" else 0.0)},
        dataset={"data_dir": str(DATA_DIR.relative_to(PROJECT_ROOT)),
                 "feature_kind": feature, "feature_dim": in_dim,
                 "facs_landmark_source": source if feature == "facs_plus_bs_80" else None,
                 "blendshape_source": "mediapipe", "num_classes": num_classes,
                 "class_names": cls_names,
                 "n_train": int(len(ytr)), "n_val": int(len(yva)), "n_test": int(len(yte)),
                 "class_counts_train": class_counts(ytr, num_classes),
                 "class_counts_val": class_counts(yva, num_classes),
                 "class_counts_test": class_counts(yte, num_classes)},
        model=model,
    )
    record["arch_name"] = arch
    record["source"] = source

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    best_val = -1.0; best_state = None; best_epoch = -1; no_imp = 0
    history = []; t0 = time.time(); early_stopped = False; epochs_done = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        et0 = time.time()
        total = 0.0; correct = 0; n = 0
        for xb, yb in tl:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            optim.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward(); optim.step()
            total += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            n += xb.size(0)
        vm = evaluate_full(model, vl, num_classes, cls_names, device=device)
        history.append({"epoch": epoch, "epoch_time_sec": time.time() - et0,
                        "train_loss": total / max(n, 1), "train_accuracy": correct / max(n, 1),
                        "val_macro_f1": vm["macro_f1"], "val_weighted_f1": vm["weighted_f1"],
                        "val_accuracy": vm["accuracy"]})
        epochs_done = epoch
        if epoch == 1 or epoch % 10 == 0:
            print(f"  ep {epoch:3d}  loss={total/max(n,1):.4f}  val_mf1={vm['macro_f1']:.4f}")
        if vm["macro_f1"] > best_val:
            best_val = vm["macro_f1"]; best_epoch = epoch; no_imp = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                early_stopped = True
                break
    elapsed = time.time() - t0
    peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)
                    if device.type == "cuda" else 0.0)

    model.load_state_dict(best_state)
    test_m = evaluate_full(model, el, num_classes, cls_names, device=device)
    val_m_best = evaluate_full(model, vl, num_classes, cls_names, device=device)

    print(f"  best ep {best_epoch}  val_mf1={best_val:.4f}  ({elapsed:.0f}s)  "
          f"TEST mf1={test_m['macro_f1']:.4f}  wf1={test_m['weighted_f1']:.4f}  acc={test_m['accuracy']:.4f}")

    record["training"] = {"elapsed_sec": elapsed, "epochs_completed": epochs_done,
                          "best_epoch": best_epoch, "early_stopped": early_stopped,
                          "peak_vram_mb": float(peak_vram_mb), "history": history}
    record["test"] = test_m
    record["val_at_best"] = val_m_best
    return record


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", default=["blendshape_52", "facs_plus_bs_80"])
    ap.add_argument("--sources", nargs="+", default=None,
                    help="FACS source for facs_plus_bs_80 (default: mediapipe + faceapi). Blendshape selalu MP-only.")
    ap.add_argument("--scenarios", nargs="+", default=["b1", "b2", "b3"])
    ap.add_argument("--classes", nargs="+", type=int, default=[3, 7])
    ap.add_argument("--archs", nargs="+", default=["fcnn", "cnn1d"])
    ap.add_argument("--data-dir", default=None,
                    help="Override DATA_DIR (default: data/dataset_frontonly_conf60)")
    ap.add_argument("--output-prefix", default=None,
                    help="Output root (default: models/frontonly_conf60)")
    ap.add_argument("--force-retrain", action="store_true",
                    help="Retrain even if results.json already has the run (default: skip existing).")
    args = ap.parse_args()

    global DATA_DIR
    if args.data_dir is not None:
        DATA_DIR = Path(args.data_dir).resolve()
    output_prefix = Path(args.output_prefix).resolve() if args.output_prefix else (PROJECT_ROOT / "models" / "frontonly_conf60")
    fb80_sources = args.sources if args.sources is not None else ["mediapipe", "faceapi"]

    print(f"Device: {device}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"Output: {output_prefix}")
    sweep = []
    if "blendshape_52" in args.features:
        # Blendshape: source dummy "mediapipe" (selalu MP)
        for nc in args.classes:
            for arch in args.archs:
                for scen in args.scenarios:
                    sweep.append(("blendshape_52", "mediapipe", arch, scen, nc))
    if "facs_plus_bs_80" in args.features:
        # FACS+BS: configurable sources
        for nc in args.classes:
            for src in fb80_sources:
                for arch in args.archs:
                    for scen in args.scenarios:
                        sweep.append(("facs_plus_bs_80", src, arch, scen, nc))
    print(f"Total runs: {len(sweep)}")

    # Cache loaded data per (feature, source, nc) to avoid reload
    data_cache = {}
    all_recs = {}
    if not args.force_retrain:
        candidate_files = set()
        for nc in args.classes:
            for feat in args.features:
                candidate_files.add(output_prefix / f"{nc}class" / "Unified" / feat / "results.json")
        for f in candidate_files:
            if f.exists():
                try:
                    data = json.load(open(f))
                    for k, v in data.get("runs", {}).items():
                        all_recs[k] = v
                except Exception as e:
                    print(f"  warn: failed to load {f}: {e}")
        if all_recs:
            print(f"Pre-loaded {len(all_recs)} existing runs (use --force-retrain to ignore)")

    n_skip = n_train = 0
    for i, (feat, src, arch, scen, nc) in enumerate(sweep, 1):
        key = f"{feat}_{src}_{arch}_{scen}_{nc}c"
        if key in all_recs and not args.force_retrain:
            mf1 = all_recs[key].get("test", {}).get("macro_f1")
            mf1_s = f"{mf1:.4f}" if isinstance(mf1, (int, float)) else "?"
            print(f"\n[{i}/{len(sweep)}] SKIP {key} (existing, mf1={mf1_s})")
            n_skip += 1
            continue
        cache_key = (feat, src, nc)
        if cache_key not in data_cache:
            data_cache[cache_key] = build_data(feat, src, nc)
        print(f"\n[{i}/{len(sweep)}]")
        all_recs[key] = train_run(feat, src, arch, scen, nc, data_cache[cache_key])
        n_train += 1
    print(f"\nSweep done: trained={n_train}, skipped_existing={n_skip}")

    for nc in args.classes:
        for feat in args.features:
            out_dir = output_prefix / f"{nc}class" / "Unified" / feat
            out_dir.mkdir(parents=True, exist_ok=True)
            subset = {k: v for k, v in all_recs.items()
                      if k.startswith(feat) and k.endswith(f"_{nc}c")}
            with open(out_dir / "results.json", "w") as f:
                json.dump({"config": f"unified_{nc}c_{feat}", "runs": subset}, f, indent=2)
            print(f"Saved: {out_dir/'results.json'}")

    print("\n" + "=" * 95)
    print("SUMMARY: derived features macro_f1")
    print("=" * 95)
    print(f"  {'scheme':<6s}  {'feat':<18s}  {'src':<10s}  {'arch':<6s}  {'B1':>8s}  {'B2':>8s}  {'B3':>8s}")
    for nc in (3, 7):
        for feat in ("blendshape_52", "facs_plus_bs_80"):
            for src in ("mediapipe", "faceapi") if feat == "facs_plus_bs_80" else ("mediapipe",):
                for arch in ("fcnn", "cnn1d"):
                    vals = []
                    for scen in ("b1", "b2", "b3"):
                        key = f"{feat}_{src}_{arch}_{scen}_{nc}c"
                        v = all_recs.get(key, {}).get("test", {}).get("macro_f1")
                        vals.append(f"{v:.4f}" if v is not None else "  -   ")
                    print(f"  {nc}c     {feat:<18s}  {src:<10s}  {arch:<6s}  {vals[0]:>8s}  {vals[1]:>8s}  {vals[2]:>8s}")


if __name__ == "__main__":
    main()
