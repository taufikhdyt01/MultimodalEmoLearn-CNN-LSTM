#!/usr/bin/env python3
"""
Unified protocol: train top landmark configs dengan B1/B2/B3 yang consistent.

UNIFIED SCENARIO DEFINITION:
  B1: no aug, no class weight, no sampler         (baseline)
  B2: WeightedRandomSampler (class-balanced)      (handle imbalance, no aug)
  B3: WeightedRandomSampler + per-batch random aug (handle imbalance + diversity)

Aug pipeline (per __getitem__ random):
  - hflip with proper left/right index swap (p=0.5)
  - rotate ±10° around face center
  - scale 0.95-1.05
  - translate ±2%
  - per-coord Gaussian noise σ=0.005

Coverage runs (top contenders berdasarkan leaderboard):
  Features: {raw_136, facs_28}
  Sources:  {mediapipe, faceapi}
  Archs:    {fcnn, cnn1d}
  Scenarios:{B1, B2, B3}
  Schemes:  {3c, 7c}
  Total: 2 × 2 × 2 × 3 × 2 = 48 runs (~75 menit di GPU shared)

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/run_unified_landmark.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from training.models import EmotionCNN1D, EmotionCNN1D_FACS, EmotionFCNN  # noqa: E402
from training.exp_utils import class_counts, evaluate_full, make_run_record  # noqa: E402
from training.landmark_aug import (  # noqa: E402
    AugmentingLandmarkDataset, make_balanced_sampler,
)

DATA_DIR = PROJECT_ROOT / "data" / "dataset_frontonly_conf60"
REMAP_3 = np.array([1, 0, 2, 2, 2, 2, 0], dtype=np.int64)
CLASS_NAMES_7 = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]
CLASS_NAMES_3 = ["positive", "neutral", "negative"]

# FACS pair definition (same as run_cnn1d_facs.py)
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


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


def load_landmark(source: str, split: str):
    fname = "X_{}_landmarks.npy" if source == "mediapipe" else "X_{}_faceapi_landmarks.npy"
    return np.load(DATA_DIR / fname.format(split)).astype(np.float32)


def get_feature_fn(feature: str):
    if feature == "raw_136":
        return None  # passthrough
    elif feature == "facs_28":
        return compute_facs_distances
    raise ValueError(feature)


def get_feature_dim(feature: str):
    return {"raw_136": 136, "facs_28": 28}[feature]


def build_model(arch: str, in_dim: int, num_classes: int):
    if arch == "fcnn":
        return EmotionFCNN(input_dim=in_dim, num_classes=num_classes)
    if arch == "cnn1d" and in_dim == 136:
        return EmotionCNN1D(num_classes=num_classes)
    if arch == "cnn1d" and in_dim == 28:
        return EmotionCNN1D_FACS(num_classes=num_classes, in_dim=in_dim)
    raise ValueError(f"{arch} / {in_dim}")


def train_run(source, feature, arch, scenario, num_classes):
    """Single training run with unified protocol."""
    cls_names = CLASS_NAMES_3 if num_classes == 3 else CLASS_NAMES_7
    Xtr = load_landmark(source, "train")
    Xva = load_landmark(source, "val")
    Xte = load_landmark(source, "test")
    ytr = np.load(DATA_DIR / "y_train.npy").astype(np.int64)
    yva = np.load(DATA_DIR / "y_val.npy").astype(np.int64)
    yte = np.load(DATA_DIR / "y_test.npy").astype(np.int64)
    if num_classes == 3:
        ytr = REMAP_3[ytr]; yva = REMAP_3[yva]; yte = REMAP_3[yte]

    feature_fn = get_feature_fn(feature)
    in_dim = get_feature_dim(feature)
    augment_train = scenario == "b3"

    print(f"\n==== {source} × {feature} × {arch} × {num_classes}c × {scenario.upper()} ====")
    print(f"  Train N={len(Xtr)}  in_dim={in_dim}  augment={augment_train}")
    print(f"  class counts train: {class_counts(ytr, num_classes)}")

    train_ds = AugmentingLandmarkDataset(Xtr, ytr, augment=augment_train, feature_fn=feature_fn, seed=SEED)
    val_ds   = AugmentingLandmarkDataset(Xva, yva, augment=False, feature_fn=feature_fn)
    test_ds  = AugmentingLandmarkDataset(Xte, yte, augment=False, feature_fn=feature_fn)

    # Sampler / balancing
    if scenario in ("b2", "b3"):
        sampler = make_balanced_sampler(ytr, num_classes)
        train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler,
                                  drop_last=True, num_workers=0, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                                  drop_last=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=True)

    set_seed(SEED)
    model = build_model(arch, in_dim, num_classes).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()  # no weight — handled by sampler

    record = make_run_record(
        config=f"unified_{source}_{feature}_{arch}_{num_classes}c_{scenario}",
        notes=("Unified protocol: B1=no aug/no balance, B2=WeightedRandomSampler, "
               "B3=Sampler + per-batch random aug (hflip+swap, rotate±10°, scale 0.95-1.05, "
               "translate ±2%, noise σ=0.005)."),
        hyperparams={"batch_size": BATCH, "epochs_max": EPOCHS, "patience": PATIENCE,
                     "lr": LR, "optimizer": "Adam", "loss": "CrossEntropyLoss",
                     "seed": SEED, "scenario": scenario.upper(),
                     "uses_weighted_sampler": scenario in ("b2", "b3"),
                     "uses_per_batch_aug": scenario == "b3"},
        dataset={"data_dir": str(DATA_DIR.relative_to(PROJECT_ROOT)),
                 "feature_kind": feature, "feature_dim": in_dim,
                 "landmark_source": source, "num_classes": num_classes,
                 "class_names": cls_names,
                 "n_train": int(len(Xtr)), "n_val": int(len(Xva)), "n_test": int(len(Xte)),
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
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            optim.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward(); optim.step()
            total += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            n += xb.size(0)
        vm = evaluate_full(model, val_loader, num_classes, cls_names, device=device)
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
    test_m = evaluate_full(model, test_loader, num_classes, cls_names, device=device)
    val_m_best = evaluate_full(model, val_loader, num_classes, cls_names, device=device)

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
    ap.add_argument("--features", nargs="+", default=["raw_136", "facs_28"])
    ap.add_argument("--archs", nargs="+", default=["fcnn", "cnn1d"])
    ap.add_argument("--sources", nargs="+", default=["mediapipe", "faceapi"])
    ap.add_argument("--scenarios", nargs="+", default=["b1", "b2", "b3"])
    ap.add_argument("--classes", nargs="+", type=int, default=[3, 7])
    ap.add_argument("--data-dir", default=None,
                    help="Override DATA_DIR (default: data/dataset_frontonly_conf60 = primer)")
    ap.add_argument("--output-prefix", default=None,
                    help="Output root, e.g. models/benchmark/kdef_7class (default: models/frontonly_conf60)")
    ap.add_argument("--force-retrain", action="store_true",
                    help="Retrain even if results.json already has the run (default: skip existing).")
    args = ap.parse_args()

    # Override global DATA_DIR if specified
    global DATA_DIR
    if args.data_dir is not None:
        DATA_DIR = Path(args.data_dir).resolve()
    output_prefix = Path(args.output_prefix).resolve() if args.output_prefix else (PROJECT_ROOT / "models" / "frontonly_conf60")
    print(f"Device: {device}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"Output: {output_prefix}")
    sweep = []
    for nc in args.classes:
        for feat in args.features:
            for src in args.sources:
                for arch in args.archs:
                    for scen in args.scenarios:
                        sweep.append((src, feat, arch, scen, nc))
    print(f"Total runs: {len(sweep)}")

    # Pre-load existing runs from candidate output files (preserve untouched keys
    # in same file + enable skip-if-exists). Disable via --force-retrain.
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
    for i, (src, feat, arch, scen, nc) in enumerate(sweep, 1):
        key = f"{src}_{feat}_{arch}_{scen}_{nc}c"
        if key in all_recs and not args.force_retrain:
            mf1 = all_recs[key].get("test", {}).get("macro_f1")
            mf1_s = f"{mf1:.4f}" if isinstance(mf1, (int, float)) else "?"
            print(f"\n[{i}/{len(sweep)}] SKIP {key} (existing, mf1={mf1_s})")
            n_skip += 1
            continue
        print(f"\n[{i}/{len(sweep)}]")
        all_recs[key] = train_run(src, feat, arch, scen, nc)
        n_train += 1
    print(f"\nSweep done: trained={n_train}, skipped_existing={n_skip}")

    # Save per (nc, feat) for easy lookup
    for nc in args.classes:
        for feat in args.features:
            out_dir = output_prefix / f"{nc}class" / "Unified" / feat
            out_dir.mkdir(parents=True, exist_ok=True)
            subset = {k: v for k, v in all_recs.items() if k.endswith(f"_{nc}c") and f"_{feat}_" in k}
            with open(out_dir / "results.json", "w") as f:
                json.dump({"config": f"unified_{nc}c_{feat}", "runs": subset}, f, indent=2)
            print(f"Saved: {out_dir/'results.json'}")

    # Summary
    print("\n" + "=" * 95)
    print("SUMMARY: Unified protocol macro_f1")
    print("=" * 95)
    print(f"  {'scheme':<6s}  {'feat':<8s}  {'src':<10s}  {'arch':<6s}  {'B1':>8s}  {'B2':>8s}  {'B3':>8s}")
    for nc in args.classes:
        for feat in args.features:
            for src in args.sources:
                for arch in args.archs:
                    vals = []
                    for scen in ("b1", "b2", "b3"):
                        key = f"{src}_{feat}_{arch}_{scen}_{nc}c"
                        v = all_recs.get(key, {}).get("test", {}).get("macro_f1")
                        vals.append(f"{v:.4f}" if v is not None else "  -   ")
                    print(f"  {nc}c     {feat:<8s}  {src:<10s}  {arch:<6s}  {vals[0]:>8s}  {vals[1]:>8s}  {vals[2]:>8s}")


if __name__ == "__main__":
    main()
