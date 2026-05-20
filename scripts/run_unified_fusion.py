#!/usr/bin/env python3
"""Unified protocol: train Early & Intermediate Fusion (scratch + TL).

UNIFIED SCENARIO DEFINITION (identik dengan landmark + image sweep):
  B1: no aug, no class weight, no sampler           (baseline)
  B2: WeightedRandomSampler (class-balanced)        (handle imbalance, no aug)
  B3: WeightedRandomSampler + per-batch SYNCED aug  (handle imbalance + diversity)

Synced aug pipeline (per __getitem__):
  Geometric (synced image + landmark/heatmap):
    - hflip (p=0.5) + HFLIP_PERM swap untuk landmark
    - rotate ±10° around image center
  Photometric (image only):
    - brightness ±10%
    - contrast ×0.9-1.1

Coverage:
  Fusion: {early (RGB+heatmap), intermediate (RGB+landmark coord)}
  Variants: {scratch, TL (ResNet-18)}
  Scenarios: {B1, B2, B3}
  Schemes: {3c, 7c}
  Landmark source: {mediapipe (default), faceapi}
  Total: 2 × 2 × 3 × 2 × |sources| = 24 runs per source (~3-4 jam di GPU dedicated)

Late Fusion belum dicover di script ini — post-hoc, butuh prediksi image-only +
landmark-only. Dihandle di `compute_late_fusion_unified.py`.

Usage:
  # Default: MediaPipe landmark, 24 runs
  CUDA_VISIBLE_DEVICES=0 python scripts/run_unified_fusion.py
  # Priority 2: face-api.js landmark fusion, 24 runs tambahan
  CUDA_VISIBLE_DEVICES=0 python scripts/run_unified_fusion.py --landmark-sources faceapi
  # Both: 48 runs total
  CUDA_VISIBLE_DEVICES=0 python scripts/run_unified_fusion.py --landmark-sources mediapipe faceapi

Output (MP-only, default):
  models/frontonly_conf60/{3,7}class/Unified/fusion_{early,intermediate}_{scratch,tl}/results.json
Output (FA-only):
  models/frontonly_conf60/{3,7}class/Unified/fusion_{early,intermediate}_{scratch,tl}_faceapi/results.json
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
from training.models import (  # noqa: E402
    EmotionEarlyFusion, EmotionEarlyFusionTransfer,
    EmotionEarlyFusionGated, EmotionEarlyFusionTransferGated,
    IntermediateFusion, IntermediateFusionTransfer,
)
from training.exp_utils import class_counts, evaluate_full, make_run_record  # noqa: E402
from training.fusion_aug import (  # noqa: E402
    EarlyFusionDataset, IntermediateFusionDataset, make_balanced_sampler,
)

DATA_DIR = PROJECT_ROOT / "data" / "dataset_frontonly_conf60"
REMAP_3 = np.array([1, 0, 2, 2, 2, 2, 0], dtype=np.int64)
CLASS_NAMES_7 = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]
CLASS_NAMES_3 = ["positive", "neutral", "negative"]

BATCH = 32
EPOCHS = 50
PATIENCE = 15
LR = 1e-3
LR_TL = 1e-4  # ResNet-18 finetune
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---- Landmark feature definitions ----
LANDMARK_FEATURE_DIM = {
    "raw_136": 136,
    "facs_28": 28,
    "blendshape_52": 52,
    "facs_plus_bs_80": 80,
}


def feature_suffix(feature: str) -> str:
    """Suffix for arch_label & key. Empty for raw_136 (backward compat)."""
    return "" if feature == "raw_136" else f"_{feature}"


def feature_is_coord(feature: str) -> bool:
    return feature == "raw_136"


# Subset of FACS_PAIRS used to compute facs_28 from raw landmarks (identical to
# run_unified_landmark.py / run_unified_derived.py).
_FACS_PAIRS = [
    (21, 39), (22, 42), (17, 36), (26, 45), (21, 22), (19, 37), (24, 44),
    (37, 41), (38, 40), (43, 47), (44, 46), (36, 31), (45, 35),
    (31, 35), (27, 30), (33, 51), (50, 33), (52, 33), (48, 36), (54, 45),
    (48, 8), (54, 8), (48, 54), (60, 64), (51, 57), (62, 66), (33, 8), (8, 27),
]
_INTEROCULAR = (36, 45)


def _compute_facs_distances(lm_136: np.ndarray) -> np.ndarray:
    """(N, 136) → (N, 28) FACS Euclidean distances normalized by interocular distance."""
    pts = lm_136.reshape(-1, 68, 2)
    a, b = _INTEROCULAR
    iod = np.maximum(np.linalg.norm(pts[:, a] - pts[:, b], axis=1), 1e-6)
    out = np.zeros((len(pts), len(_FACS_PAIRS)), dtype=np.float32)
    for i, (p1, p2) in enumerate(_FACS_PAIRS):
        d = np.linalg.norm(pts[:, p1] - pts[:, p2], axis=1)
        out[:, i] = (d / iod).astype(np.float32)
    return out


def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load_arrays(split: str, landmark_source: str = "mediapipe",
                landmark_feature: str = "raw_136"):
    """Load image, landmark_feature, heatmap, label for a split.

    landmark_source: 'mediapipe' → `X_*_landmarks.npy`, 'faceapi' → `X_*_faceapi_landmarks.npy`.
    landmark_feature controls what's in the returned `lm` tensor:
      - raw_136: 2D coords langsung (136-dim), augmentable
      - facs_28: 28-dim Euclidean distances dari raw_136
      - blendshape_52: 52-dim ARKit blendshape (MP only, dari `X_*_mp_blendshapes.npy`)
      - facs_plus_bs_80: concat(FACS_28, blendshape_52) = 80-dim
    Image branch tetap dari MP face-crop (face-api.js tidak punya data citra).
    """
    img = np.load(DATA_DIR / f"X_{split}_images.npy")
    lm_file = "X_{}_landmarks.npy" if landmark_source == "mediapipe" else "X_{}_faceapi_landmarks.npy"
    raw_lm = np.load(DATA_DIR / lm_file.format(split)).astype(np.float32)

    if landmark_feature == "raw_136":
        lm = raw_lm
    elif landmark_feature == "facs_28":
        lm = _compute_facs_distances(raw_lm)
    elif landmark_feature == "blendshape_52":
        if landmark_source != "mediapipe":
            raise ValueError("blendshape_52 hanya tersedia dari MediaPipe source")
        bs = np.load(DATA_DIR / f"X_{split}_mp_blendshapes.npy").astype(np.float32)
        bs = np.nan_to_num(bs, nan=0.0)  # impute NaN with 0 for robustness
        lm = bs
    elif landmark_feature == "facs_plus_bs_80":
        facs = _compute_facs_distances(raw_lm)  # (N, 28)
        bs = np.load(DATA_DIR / f"X_{split}_mp_blendshapes.npy").astype(np.float32)
        bs = np.nan_to_num(bs, nan=0.0)
        lm = np.concatenate([facs, bs], axis=1).astype(np.float32)  # (N, 80)
    else:
        raise ValueError(f"Unknown landmark_feature: {landmark_feature}")

    hm = np.load(DATA_DIR / f"X_{split}_heatmaps.npy")
    y = np.load(DATA_DIR / f"y_{split}.npy").astype(np.int64)
    return img, lm, hm, y


def build_model(fusion: str, variant: str, num_classes: int, landmark_dim: int = 136,
                early_fusion_mode: str = "concat"):
    """fusion ∈ {early, intermediate}; variant ∈ {scratch, tl}.
    Early Fusion ignores landmark_dim (input is RGB+heatmap channel, not coords).
    early_fusion_mode ∈ {concat (default), gated} — only relevant for fusion=early."""
    if fusion == "early":
        gated = early_fusion_mode == "gated"
        if variant == "scratch":
            return EmotionEarlyFusionGated(num_classes=num_classes) if gated \
                else EmotionEarlyFusion(num_classes=num_classes)
        if variant == "tl":
            return EmotionEarlyFusionTransferGated(num_classes=num_classes, pretrained=True) if gated \
                else EmotionEarlyFusionTransfer(num_classes=num_classes, pretrained=True)
    if fusion == "intermediate" and variant == "scratch":
        return IntermediateFusion(num_classes=num_classes, landmark_dim=landmark_dim)
    if fusion == "intermediate" and variant == "tl":
        return IntermediateFusionTransfer(num_classes=num_classes, landmark_dim=landmark_dim, pretrained=True)
    raise ValueError(f"{fusion} / {variant}")


def get_lr(variant: str) -> float:
    return LR_TL if variant == "tl" else LR


def make_dataset(fusion: str, X_img, X_lm, X_hm, y, *, augment: bool,
                 seed: int = SEED, landmark_is_coord: bool = True):
    """Pick dataset class based on fusion type. landmark_is_coord controls aug behavior
    di IntermediateFusionDataset (False = skip geometric image aug supaya tidak mismatch
    dengan landmark feature yang non-coord)."""
    if fusion == "early":
        return EarlyFusionDataset(X_img, X_hm, y, augment=augment, seed=seed)
    if fusion == "intermediate":
        return IntermediateFusionDataset(X_img, X_lm, y, augment=augment, seed=seed,
                                          landmark_is_coord=landmark_is_coord)
    raise ValueError(fusion)


def forward_for(fusion: str):
    """Return forward_fn for evaluate_full — handles different fusion signatures."""
    if fusion == "early":
        def fwd(model, batch):
            xb, yb = batch
            xb = xb.to(device, non_blocking=True)
            return model(xb), yb
        return fwd
    if fusion == "intermediate":
        def fwd(model, batch):
            img, lm, yb = batch
            img = img.to(device, non_blocking=True)
            lm = lm.to(device, non_blocking=True)
            return model(img, lm), yb
        return fwd
    raise ValueError(fusion)


def train_run(fusion: str, variant: str, scenario: str, num_classes: int,
              landmark_source: str = "mediapipe", landmark_feature: str = "raw_136",
              early_fusion_mode: str = "concat", output_prefix: Path = None):
    """Single training run with unified protocol."""
    cls_names = CLASS_NAMES_3 if num_classes == 3 else CLASS_NAMES_7
    X_img_tr, X_lm_tr, X_hm_tr, ytr = load_arrays("train", landmark_source, landmark_feature)
    X_img_va, X_lm_va, X_hm_va, yva = load_arrays("val", landmark_source, landmark_feature)
    X_img_te, X_lm_te, X_hm_te, yte = load_arrays("test", landmark_source, landmark_feature)
    if num_classes == 3:
        ytr = REMAP_3[ytr]; yva = REMAP_3[yva]; yte = REMAP_3[yte]

    augment_train = scenario == "b3"
    is_coord = feature_is_coord(landmark_feature)
    lr = get_lr(variant)
    landmark_dim = LANDMARK_FEATURE_DIM[landmark_feature]

    mode_tag = "_gated" if (fusion == "early" and early_fusion_mode == "gated") else ""
    print(f"\n==== fusion_{fusion}{mode_tag}_{variant} × {num_classes}c × {scenario.upper()} "
          f"× lm={landmark_source}/{landmark_feature}(dim={landmark_dim}) ====")
    print(f"  Train N={len(X_img_tr)}  augment={augment_train}  is_coord={is_coord}  lr={lr}"
          + (f"  early_fusion_mode={early_fusion_mode}" if fusion == "early" else ""))
    print(f"  class counts train: {class_counts(ytr, num_classes)}")

    train_ds = make_dataset(fusion, X_img_tr, X_lm_tr, X_hm_tr, ytr,
                            augment=augment_train, landmark_is_coord=is_coord)
    val_ds   = make_dataset(fusion, X_img_va, X_lm_va, X_hm_va, yva,
                            augment=False, landmark_is_coord=is_coord)
    test_ds  = make_dataset(fusion, X_img_te, X_lm_te, X_hm_te, yte,
                            augment=False, landmark_is_coord=is_coord)

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
    model = build_model(fusion, variant, num_classes, landmark_dim=landmark_dim,
                        early_fusion_mode=early_fusion_mode).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    fwd = forward_for(fusion)

    arch_label = f"fusion_{fusion}{mode_tag}_{variant}{feature_suffix(landmark_feature)}"
    record = make_run_record(
        config=f"unified_{arch_label}_{num_classes}c_{scenario}",
        notes=("Unified protocol fusion: B1=no aug/no balance, B2=WeightedRandomSampler, "
               "B3=Sampler + synced per-batch aug (hflip+landmark_swap+heatmap_flip, rotate±10°, "
               "brightness±10%, contrast×0.9-1.1)."),
        hyperparams={"batch_size": BATCH, "epochs_max": EPOCHS, "patience": PATIENCE,
                     "lr": lr, "optimizer": "Adam", "loss": "CrossEntropyLoss",
                     "seed": SEED, "scenario": scenario.upper(),
                     "uses_weighted_sampler": scenario in ("b2", "b3"),
                     "uses_per_batch_aug": scenario == "b3"},
        dataset={"data_dir": str(DATA_DIR.relative_to(PROJECT_ROOT)),
                 "modalities": (["image_224x224x3", "heatmap_224x224"] if fusion == "early"
                                else ["image_224x224x3", "landmark_136"]),
                 "fusion_type": fusion, "num_classes": num_classes,
                 "class_names": cls_names,
                 "n_train": int(len(X_img_tr)), "n_val": int(len(X_img_va)), "n_test": int(len(X_img_te)),
                 "class_counts_train": class_counts(ytr, num_classes),
                 "class_counts_val": class_counts(yva, num_classes),
                 "class_counts_test": class_counts(yte, num_classes)},
        model=model,
    )
    record["arch_name"] = arch_label
    record["source"] = f"image_mp_crop + landmark_{landmark_source}_{landmark_feature}"
    record["landmark_source"] = landmark_source
    record["landmark_feature"] = landmark_feature
    record["landmark_dim"] = landmark_dim
    record["fusion_type"] = fusion
    record["variant"] = variant

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    best_val = -1.0; best_state = None; best_epoch = -1; no_imp = 0
    history = []; t0 = time.time(); early_stopped = False; epochs_done = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        et0 = time.time()
        total = 0.0; correct = 0; n = 0
        for batch in train_loader:
            logits, yb = fwd(model, batch)
            yb = yb.to(device, non_blocking=True)
            optim.zero_grad()
            loss = crit(logits, yb)
            loss.backward(); optim.step()
            bs = yb.size(0)
            total += loss.item() * bs
            correct += (logits.argmax(1) == yb).sum().item()
            n += bs
        vm = evaluate_full(model, val_loader, num_classes, cls_names,
                           forward_fn=fwd, device=device)
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
    test_m = evaluate_full(model, test_loader, num_classes, cls_names,
                           forward_fn=fwd, device=device)
    val_m_best = evaluate_full(model, val_loader, num_classes, cls_names,
                               forward_fn=fwd, device=device)

    print(f"  best ep {best_epoch}  val_mf1={best_val:.4f}  ({elapsed:.0f}s)  "
          f"TEST mf1={test_m['macro_f1']:.4f}  wf1={test_m['weighted_f1']:.4f}  acc={test_m['accuracy']:.4f}")

    # Save checkpoint untuk gated Early Fusion (untuk visualize_gates.py).
    # Skip untuk concat (existing) supaya disk usage tidak meledak.
    if fusion == "early" and early_fusion_mode == "gated" and output_prefix is not None:
        ckpt_dir = output_prefix / f"{num_classes}class" / "Unified" / arch_label / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"{scenario}.pt"
        torch.save({"model_state_dict": best_state,
                    "config": {"fusion": fusion, "variant": variant,
                               "early_fusion_mode": early_fusion_mode,
                               "num_classes": num_classes, "scenario": scenario,
                               "landmark_source": landmark_source,
                               "landmark_feature": landmark_feature,
                               "best_epoch": best_epoch, "best_val_macro_f1": best_val}},
                   ckpt_path)
        print(f"  Saved gated checkpoint: {ckpt_path}")

    record["training"] = {"elapsed_sec": elapsed, "epochs_completed": epochs_done,
                          "best_epoch": best_epoch, "early_stopped": early_stopped,
                          "peak_vram_mb": float(peak_vram_mb), "history": history}
    record["test"] = test_m
    record["val_at_best"] = val_m_best
    return record


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fusions", nargs="+", default=["early", "intermediate"])
    ap.add_argument("--variants", nargs="+", default=["scratch", "tl"])
    ap.add_argument("--scenarios", nargs="+", default=["b1", "b2", "b3"])
    ap.add_argument("--classes", nargs="+", type=int, default=[3, 7])
    ap.add_argument("--landmark-sources", nargs="+", default=["mediapipe"],
                    help="Landmark source(s): 'mediapipe' or 'faceapi'. Image branch always from MP face-crop.")
    ap.add_argument("--landmark-features", nargs="+", default=["raw_136"],
                    choices=list(LANDMARK_FEATURE_DIM.keys()),
                    help="Landmark feature(s): raw_136 (2D coords), facs_28 (Euclidean distances), "
                         "blendshape_52 (ARKit, MP-only), facs_plus_bs_80 (FACS+BS concat). "
                         "Default: raw_136 (backward compat). Early Fusion hanya valid untuk raw_136.")
    ap.add_argument("--early-fusion-modes", nargs="+", default=["concat"],
                    choices=["concat", "gated"],
                    help="Early Fusion mode(s): 'concat' (default, channel-stack RGB+heatmap) "
                         "atau 'gated' (spatial sigmoid gating sebelum CNN backbone). Hanya berlaku "
                         "untuk fusion=early. Intermediate Fusion ignore flag ini.")
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
    def _mode_tag(fusion, mode):
        """Append _gated suffix only for Early Fusion with gated mode (backward compat)."""
        return "_gated" if (fusion == "early" and mode == "gated") else ""

    sweep = []
    for nc in args.classes:
        for feat in args.landmark_features:
            for src in args.landmark_sources:
                if feat == "blendshape_52" and src != "mediapipe":
                    print(f"  SKIP combo: feature={feat} × source={src} (blendshape MP-only)")
                    continue
                for fusion in args.fusions:
                    if fusion == "early" and feat != "raw_136":
                        print(f"  SKIP combo: fusion=early × feature={feat} (Early Fusion only supports raw_136)")
                        continue
                    # Iterate fusion modes (only meaningful for Early Fusion; Intermediate ignores mode).
                    fusion_modes = args.early_fusion_modes if fusion == "early" else ["concat"]
                    for mode in fusion_modes:
                        for variant in args.variants:
                            for scen in args.scenarios:
                                sweep.append((fusion, variant, scen, nc, src, feat, mode))
    print(f"Total runs: {len(sweep)}")

    def make_key(fusion, variant, src, feat, scen, nc, mode="concat"):
        src_tag = "" if src == "mediapipe" else f"_{src}"
        return f"fusion_{fusion}{_mode_tag(fusion, mode)}_{variant}{feature_suffix(feat)}{src_tag}_{scen}_{nc}c"

    def make_arch_label(fusion, variant, src, feat, mode="concat"):
        src_tag = "" if src == "mediapipe" else f"_{src}"
        return f"fusion_{fusion}{_mode_tag(fusion, mode)}_{variant}{feature_suffix(feat)}{src_tag}"

    all_recs = {}
    if not args.force_retrain:
        candidate_files = set()
        for nc in args.classes:
            for feat in args.landmark_features:
                for src in args.landmark_sources:
                    if feat == "blendshape_52" and src != "mediapipe":
                        continue
                    for fusion in args.fusions:
                        if fusion == "early" and feat != "raw_136":
                            continue
                        fusion_modes = args.early_fusion_modes if fusion == "early" else ["concat"]
                        for mode in fusion_modes:
                            for variant in args.variants:
                                arch_label = make_arch_label(fusion, variant, src, feat, mode)
                                candidate_files.add(output_prefix / f"{nc}class" / "Unified" / arch_label / "results.json")
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
    for i, (fusion, variant, scen, nc, src, feat, mode) in enumerate(sweep, 1):
        key = make_key(fusion, variant, src, feat, scen, nc, mode)
        if key in all_recs and not args.force_retrain:
            mf1 = all_recs[key].get("test", {}).get("macro_f1")
            mf1_s = f"{mf1:.4f}" if isinstance(mf1, (int, float)) else "?"
            print(f"\n[{i}/{len(sweep)}] SKIP {key} (existing, mf1={mf1_s})")
            n_skip += 1
            continue
        print(f"\n[{i}/{len(sweep)}]")
        all_recs[key] = train_run(fusion, variant, scen, nc,
                                   landmark_source=src, landmark_feature=feat,
                                   early_fusion_mode=mode, output_prefix=output_prefix)
        n_train += 1
    print(f"\nSweep done: trained={n_train}, skipped_existing={n_skip}")

    # Save per (nc, fusion+variant, feature, source) for easy lookup
    for nc in args.classes:
        for feat in args.landmark_features:
            for src in args.landmark_sources:
                if feat == "blendshape_52" and src != "mediapipe":
                    continue
                for fusion in args.fusions:
                    if fusion == "early" and feat != "raw_136":
                        continue
                    fusion_modes = args.early_fusion_modes if fusion == "early" else ["concat"]
                    for mode in fusion_modes:
                        for variant in args.variants:
                            arch_label = make_arch_label(fusion, variant, src, feat, mode)
                            out_dir = output_prefix / f"{nc}class" / "Unified" / arch_label
                            out_dir.mkdir(parents=True, exist_ok=True)
                            prefix = f"fusion_{fusion}{_mode_tag(fusion, mode)}_{variant}{feature_suffix(feat)}"
                            if src != "mediapipe":
                                prefix += f"_{src}"
                            prefix += "_"
                            subset = {k: v for k, v in all_recs.items()
                                      if k.endswith(f"_{nc}c") and k.startswith(prefix)}
                            with open(out_dir / "results.json", "w") as f:
                                json.dump({"config": f"unified_{nc}c_{arch_label}", "runs": subset}, f, indent=2)
                            print(f"Saved: {out_dir/'results.json'}")

    # Summary
    print("\n" + "=" * 120)
    print("SUMMARY: Unified protocol fusion macro_f1")
    print("=" * 120)
    print(f"  {'scheme':<6s}  {'fusion':<13s}  {'mode':<6s}  {'variant':<8s}  {'feature':<16s}  "
          f"{'lm_src':<10s}  {'B1':>8s}  {'B2':>8s}  {'B3':>8s}")
    for nc in args.classes:
        for feat in args.landmark_features:
            for src in args.landmark_sources:
                if feat == "blendshape_52" and src != "mediapipe":
                    continue
                for fusion in args.fusions:
                    if fusion == "early" and feat != "raw_136":
                        continue
                    fusion_modes = args.early_fusion_modes if fusion == "early" else ["concat"]
                    for mode in fusion_modes:
                        for variant in args.variants:
                            mfs = {}
                            for scen in args.scenarios:
                                key = make_key(fusion, variant, src, feat, scen, nc, mode)
                                if key in all_recs:
                                    mfs[scen] = all_recs[key]["test"]["macro_f1"]
                            print(f"  {nc}c     {fusion:<13s}  {mode:<6s}  {variant:<8s}  {feat:<16s}  {src:<10s}  "
                                  f"{mfs.get('b1', float('nan')):>8.4f}  "
                                  f"{mfs.get('b2', float('nan')):>8.4f}  "
                                  f"{mfs.get('b3', float('nan')):>8.4f}")


if __name__ == "__main__":
    main()
