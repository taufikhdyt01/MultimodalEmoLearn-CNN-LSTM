#!/usr/bin/env python3
"""Unified protocol: train image-based unimodal (CNN scratch + CNN_TL).

UNIFIED SCENARIO DEFINITION (identik dengan landmark sweep):
  B1: no aug, no class weight, no sampler           (baseline)
  B2: WeightedRandomSampler (class-balanced)        (handle imbalance, no aug)
  B3: WeightedRandomSampler + per-batch random aug  (handle imbalance + diversity)

Aug pipeline (per __getitem__ random):
  - hflip horizontal (p=0.5)
  - rotate ±10° around image center, reflect padding
  - brightness ±10%
  - contrast ×0.9–1.1

Coverage:
  Archs:    {cnn_scratch (EmotionCNN), cnn_tl (EmotionCNNTransfer / ResNet-18)}
  Scenarios:{B1, B2, B3}
  Schemes:  {3c, 7c}
  Total: 2 × 3 × 2 = 12 runs (~30 menit di L40 / RTX 4090, lebih lama di GPU shared)

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/run_unified_image.py

Output:
  models/frontonly_conf60/{3,7}class/Unified/cnn_scratch/results.json
  models/frontonly_conf60/{3,7}class/Unified/cnn_tl/results.json
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
from training.models import EmotionCNN, EmotionCNNTransfer  # noqa: E402
from training.exp_utils import class_counts, evaluate_full, make_run_record  # noqa: E402
from training.image_aug import AugmentingImageDataset, make_balanced_sampler  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "dataset_frontonly_conf60"
REMAP_3 = np.array([1, 0, 2, 2, 2, 2, 0], dtype=np.int64)
CLASS_NAMES_7 = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]
CLASS_NAMES_3 = ["positive", "neutral", "negative"]

BATCH = 32
EPOCHS = 50
PATIENCE = 15
LR = 1e-3
LR_TL = 1e-4  # transfer learning pakai lr lebih kecil supaya pretrained weights tidak rusak
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load_images(split: str) -> np.ndarray:
    """Load (N, H, W, C) float32 in [0, 1]."""
    return np.load(DATA_DIR / f"X_{split}_images.npy")


def build_model(arch: str, num_classes: int):
    if arch == "cnn_scratch":
        return EmotionCNN(num_classes=num_classes)
    if arch == "cnn_tl":
        return EmotionCNNTransfer(num_classes=num_classes, pretrained=True)
    raise ValueError(arch)


def get_lr(arch: str) -> float:
    return LR_TL if arch == "cnn_tl" else LR


def train_run(arch: str, scenario: str, num_classes: int):
    """Single training run with unified protocol."""
    cls_names = CLASS_NAMES_3 if num_classes == 3 else CLASS_NAMES_7
    Xtr = load_images("train")
    Xva = load_images("val")
    Xte = load_images("test")
    ytr = np.load(DATA_DIR / "y_train.npy").astype(np.int64)
    yva = np.load(DATA_DIR / "y_val.npy").astype(np.int64)
    yte = np.load(DATA_DIR / "y_test.npy").astype(np.int64)
    if num_classes == 3:
        ytr = REMAP_3[ytr]; yva = REMAP_3[yva]; yte = REMAP_3[yte]

    augment_train = scenario == "b3"
    lr = get_lr(arch)

    print(f"\n==== {arch} × {num_classes}c × {scenario.upper()} ====")
    print(f"  Train N={len(Xtr)}  img_shape={Xtr.shape[1:]}  augment={augment_train}  lr={lr}")
    print(f"  class counts train: {class_counts(ytr, num_classes)}")

    train_ds = AugmentingImageDataset(Xtr, ytr, augment=augment_train, seed=SEED)
    val_ds   = AugmentingImageDataset(Xva, yva, augment=False)
    test_ds  = AugmentingImageDataset(Xte, yte, augment=False)

    # Sampler / balancing
    if scenario in ("b2", "b3"):
        sampler = make_balanced_sampler(ytr, num_classes)
        train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler,
                                  drop_last=True, num_workers=2, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                                  drop_last=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)

    set_seed(SEED)
    model = build_model(arch, num_classes).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()  # no weight — handled by sampler

    record = make_run_record(
        config=f"unified_{arch}_{num_classes}c_{scenario}",
        notes=("Unified protocol image-based: B1=no aug/no balance, B2=WeightedRandomSampler, "
               "B3=Sampler + per-batch random aug (hflip p=0.5, rotate±10°, brightness±10%, contrast×0.9-1.1)."),
        hyperparams={"batch_size": BATCH, "epochs_max": EPOCHS, "patience": PATIENCE,
                     "lr": lr, "optimizer": "Adam", "loss": "CrossEntropyLoss",
                     "seed": SEED, "scenario": scenario.upper(),
                     "uses_weighted_sampler": scenario in ("b2", "b3"),
                     "uses_per_batch_aug": scenario == "b3"},
        dataset={"data_dir": str(DATA_DIR.relative_to(PROJECT_ROOT)),
                 "feature_kind": "image_224x224x3", "num_classes": num_classes,
                 "class_names": cls_names,
                 "n_train": int(len(Xtr)), "n_val": int(len(Xva)), "n_test": int(len(Xte)),
                 "class_counts_train": class_counts(ytr, num_classes),
                 "class_counts_val": class_counts(yva, num_classes),
                 "class_counts_test": class_counts(yte, num_classes)},
        model=model,
    )
    record["arch_name"] = arch
    record["source"] = "image_mp_crop"

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
        if epoch == 1 or epoch % 5 == 0:
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
    ap.add_argument("--archs", nargs="+", default=["cnn_scratch", "cnn_tl"])
    ap.add_argument("--scenarios", nargs="+", default=["b1", "b2", "b3"])
    ap.add_argument("--classes", nargs="+", type=int, default=[3, 7])
    ap.add_argument("--data-dir", default=None,
                    help="Override DATA_DIR (default: data/dataset_frontonly_conf60 = primer)")
    ap.add_argument("--output-prefix", default=None,
                    help="Output root, e.g. models/benchmark/kdef_7class (default: models/frontonly_conf60)")
    ap.add_argument("--force-retrain", action="store_true",
                    help="Retrain even if results.json already has the run (default: skip existing).")
    args = ap.parse_args()

    global DATA_DIR
    if args.data_dir is not None:
        DATA_DIR = Path(args.data_dir).resolve()
    output_prefix = Path(args.output_prefix).resolve() if args.output_prefix else (PROJECT_ROOT / "models" / "frontonly_conf60")
    print(f"Device: {device}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"Output: {output_prefix}")
    sweep = []
    for nc in args.classes:
        for arch in args.archs:
            for scen in args.scenarios:
                sweep.append((arch, scen, nc))
    print(f"Total runs: {len(sweep)}")

    all_recs = {}
    if not args.force_retrain:
        candidate_files = set()
        for nc in args.classes:
            for arch in args.archs:
                candidate_files.add(output_prefix / f"{nc}class" / "Unified" / arch / "results.json")
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
    for i, (arch, scen, nc) in enumerate(sweep, 1):
        key = f"{arch}_{scen}_{nc}c"
        if key in all_recs and not args.force_retrain:
            mf1 = all_recs[key].get("test", {}).get("macro_f1")
            mf1_s = f"{mf1:.4f}" if isinstance(mf1, (int, float)) else "?"
            print(f"\n[{i}/{len(sweep)}] SKIP {key} (existing, mf1={mf1_s})")
            n_skip += 1
            continue
        print(f"\n[{i}/{len(sweep)}]")
        all_recs[key] = train_run(arch, scen, nc)
        n_train += 1
    print(f"\nSweep done: trained={n_train}, skipped_existing={n_skip}")

    # Save per (nc, arch) for easy lookup
    for nc in args.classes:
        for arch in args.archs:
            out_dir = output_prefix / f"{nc}class" / "Unified" / arch
            out_dir.mkdir(parents=True, exist_ok=True)
            subset = {k: v for k, v in all_recs.items() if k.endswith(f"_{nc}c") and k.startswith(f"{arch}_")}
            with open(out_dir / "results.json", "w") as f:
                json.dump({"config": f"unified_{nc}c_{arch}", "runs": subset}, f, indent=2)
            print(f"Saved: {out_dir/'results.json'}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Unified protocol image-based macro_f1")
    print("=" * 80)
    print(f"  {'scheme':<6s}  {'arch':<12s}  {'B1':>8s}  {'B2':>8s}  {'B3':>8s}")
    for nc in args.classes:
        for arch in args.archs:
            mfs = {}
            for scen in args.scenarios:
                key = f"{arch}_{scen}_{nc}c"
                if key in all_recs:
                    mfs[scen] = all_recs[key]["test"]["macro_f1"]
            print(f"  {nc}c     {arch:<12s}  "
                  f"{mfs.get('b1', float('nan')):>8.4f}  "
                  f"{mfs.get('b2', float('nan')):>8.4f}  "
                  f"{mfs.get('b3', float('nan')):>8.4f}")


if __name__ == "__main__":
    main()
