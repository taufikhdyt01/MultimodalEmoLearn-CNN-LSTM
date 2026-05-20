#!/usr/bin/env python3
"""Phase 1B: Late Fusion under Unified Protocol (post-hoc weighted average).

Cara kerja:
  1. Train image-only model (CNN scratch / CNN_TL) di {B1, B2, B3} × {3c, 7c} → save val+test soft predictions
  2. Train landmark-only FCNN (raw 136-dim MP) di {B1, B2, B3} × {3c, 7c} → save val+test soft predictions
  3. Untuk setiap (variant, scenario, scheme): sweep weight w ∈ [0, 1] step 0.05
     combined_pred = w * image_softmax + (1-w) * landmark_softmax
     Pick w* yang maksimalkan val macro_f1, evaluate di test
  4. Save 12 records (2 variants × 3 scenarios × 2 schemes) di Unified format

Caching: predictions disimpan ke `models/frontonly_conf60/late_fusion_cache/`. Rerun
tidak akan retrain kalau cache npy sudah ada.

Output:
  models/frontonly_conf60/{3,7}class/Unified/fusion_late_{scratch,tl}/results.json
  models/frontonly_conf60/late_fusion_cache/*.npy (intermediate soft predictions)

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/compute_late_fusion_unified.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from training.models import EmotionCNN, EmotionCNNTransfer, EmotionFCNN  # noqa: E402
from training.exp_utils import class_counts, evaluate_full, make_run_record  # noqa: E402
from training.image_aug import AugmentingImageDataset, make_balanced_sampler as image_sampler  # noqa: E402
from training.landmark_aug import AugmentingLandmarkDataset, make_balanced_sampler as lm_sampler  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "dataset_frontonly_conf60"
OUTPUT_PREFIX = PROJECT_ROOT / "models" / "frontonly_conf60"
CACHE_DIR = OUTPUT_PREFIX / "late_fusion_cache"
REMAP_3 = np.array([1, 0, 2, 2, 2, 2, 0], dtype=np.int64)
CLASS_NAMES_7 = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]
CLASS_NAMES_3 = ["positive", "neutral", "negative"]

LANDMARK_FEATURE_DIM = {
    "raw_136": 136, "facs_28": 28, "blendshape_52": 52, "facs_plus_bs_80": 80,
}


def _feature_suffix(feature: str) -> str:
    return "" if feature == "raw_136" else f"_{feature}"


# FACS pair definitions (same as run_unified_landmark.py)
_FACS_PAIRS = [
    (21, 39), (22, 42), (17, 36), (26, 45), (21, 22), (19, 37), (24, 44),
    (37, 41), (38, 40), (43, 47), (44, 46), (36, 31), (45, 35),
    (31, 35), (27, 30), (33, 51), (50, 33), (52, 33), (48, 36), (54, 45),
    (48, 8), (54, 8), (48, 54), (60, 64), (51, 57), (62, 66), (33, 8), (8, 27),
]
_INTEROCULAR = (36, 45)


def _compute_facs(lm_136):
    pts = lm_136.reshape(-1, 68, 2)
    a, b = _INTEROCULAR
    iod = np.maximum(np.linalg.norm(pts[:, a] - pts[:, b], axis=1), 1e-6)
    out = np.zeros((len(pts), len(_FACS_PAIRS)), dtype=np.float32)
    for i, (p1, p2) in enumerate(_FACS_PAIRS):
        d = np.linalg.norm(pts[:, p1] - pts[:, p2], axis=1)
        out[:, i] = (d / iod).astype(np.float32)
    return out


def load_landmark_feature(split, source, feature):
    """Load landmark feature tensor. source ∈ {mediapipe, faceapi}; feature ∈ keys."""
    lm_file = "X_{}_landmarks.npy" if source == "mediapipe" else "X_{}_faceapi_landmarks.npy"
    raw = np.load(DATA_DIR / lm_file.format(split)).astype(np.float32)
    if feature == "raw_136":
        return raw
    if feature == "facs_28":
        return _compute_facs(raw)
    if feature == "blendshape_52":
        if source != "mediapipe":
            raise ValueError("blendshape_52 hanya tersedia dari MediaPipe source")
        bs = np.load(DATA_DIR / f"X_{split}_mp_blendshapes.npy").astype(np.float32)
        return np.nan_to_num(bs, nan=0.0)
    if feature == "facs_plus_bs_80":
        facs = _compute_facs(raw)
        bs = np.load(DATA_DIR / f"X_{split}_mp_blendshapes.npy").astype(np.float32)
        bs = np.nan_to_num(bs, nan=0.0)
        return np.concatenate([facs, bs], axis=1).astype(np.float32)
    raise ValueError(f"Unknown feature: {feature}")

BATCH = 32
EPOCHS = 50
PATIENCE = 15
LR = 1e-3
LR_TL = 1e-4
SEED = 42
WEIGHT_GRID = np.arange(0.0, 1.001, 0.05)  # 0, 0.05, ..., 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ---- Data loaders ----

def load_image_split(split: str) -> np.ndarray:
    return np.load(DATA_DIR / f"X_{split}_images.npy")


def load_landmark_split(split: str) -> np.ndarray:
    return np.load(DATA_DIR / f"X_{split}_landmarks.npy").astype(np.float32)


def load_y(split: str, num_classes: int) -> np.ndarray:
    y = np.load(DATA_DIR / f"y_{split}.npy").astype(np.int64)
    if num_classes == 3:
        y = REMAP_3[y]
    return y


# ---- Training core (mirror Unified Protocol) ----

def train_image_branch(arch: str, scenario: str, num_classes: int):
    """Returns (val_soft, test_soft, train_record). Soft = post-softmax probability."""
    cache_pattern = f"{arch}_{scenario}_{num_classes}c"
    val_cache = CACHE_DIR / f"image_{cache_pattern}_val.npy"
    test_cache = CACHE_DIR / f"image_{cache_pattern}_test.npy"
    record_cache = CACHE_DIR / f"image_{cache_pattern}_record.json"
    if val_cache.exists() and test_cache.exists() and record_cache.exists():
        print(f"  [cache hit] {cache_pattern}")
        return (np.load(val_cache), np.load(test_cache), json.load(open(record_cache)))

    Xtr, Xva, Xte = load_image_split("train"), load_image_split("val"), load_image_split("test")
    ytr, yva, yte = (load_y(s, num_classes) for s in ["train", "val", "test"])
    augment_train = scenario == "b3"
    lr = LR_TL if arch == "cnn_tl" else LR
    cls_names = CLASS_NAMES_3 if num_classes == 3 else CLASS_NAMES_7

    print(f"  Training image_{arch} × {num_classes}c × {scenario.upper()} (lr={lr}, aug={augment_train})")
    train_ds = AugmentingImageDataset(Xtr, ytr, augment=augment_train, seed=SEED)
    val_ds = AugmentingImageDataset(Xva, yva, augment=False)
    test_ds = AugmentingImageDataset(Xte, yte, augment=False)
    if scenario in ("b2", "b3"):
        sampler = image_sampler(ytr, num_classes)
        train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler,
                                  drop_last=True, num_workers=2, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                                  drop_last=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)

    set_seed(SEED)
    model = (EmotionCNNTransfer(num_classes=num_classes, pretrained=True)
             if arch == "cnn_tl" else EmotionCNN(num_classes=num_classes)).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    best_val = -1.0; best_state = None; best_epoch = -1; no_imp = 0
    t0 = time.time(); epochs_done = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            optim.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward(); optim.step()
        vm = evaluate_full(model, val_loader, num_classes, cls_names, device=device)
        epochs_done = epoch
        if vm["macro_f1"] > best_val:
            best_val = vm["macro_f1"]; best_epoch = epoch; no_imp = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                break
    elapsed = time.time() - t0
    model.load_state_dict(best_state)

    # Get soft predictions
    val_soft = _collect_soft(model, val_loader, _image_forward)
    test_soft = _collect_soft(model, test_loader, _image_forward)

    rec = {"arch": arch, "scenario": scenario, "num_classes": num_classes,
           "best_epoch": best_epoch, "best_val_macro_f1": best_val,
           "elapsed_sec": elapsed, "epochs_done": epochs_done, "lr": lr}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(val_cache, val_soft); np.save(test_cache, test_soft)
    json.dump(rec, open(record_cache, "w"), indent=2)
    print(f"    best ep {best_epoch}  val_mf1={best_val:.4f}  ({elapsed:.0f}s)")
    return val_soft, test_soft, rec


def train_landmark_branch(scenario: str, num_classes: int,
                           source: str = "mediapipe", feature: str = "raw_136"):
    """Returns (val_soft, test_soft, train_record). FCNN, configurable feature/source.

    For raw_136: per-batch landmark aug applied jika augment=True (coords-friendly).
    For FACS_28 / Blendshape_52 / FB80: aug skipped (B3 ≈ B2 untuk landmark branch).
    """
    src_tag = "mp" if source == "mediapipe" else "fa"
    cache_pattern = f"fcnn_{feature}_{src_tag}_{scenario}_{num_classes}c"
    val_cache = CACHE_DIR / f"landmark_{cache_pattern}_val.npy"
    test_cache = CACHE_DIR / f"landmark_{cache_pattern}_test.npy"
    record_cache = CACHE_DIR / f"landmark_{cache_pattern}_record.json"
    if val_cache.exists() and test_cache.exists() and record_cache.exists():
        print(f"  [cache hit] {cache_pattern}")
        return (np.load(val_cache), np.load(test_cache), json.load(open(record_cache)))

    Xtr = load_landmark_feature("train", source, feature)
    Xva = load_landmark_feature("val", source, feature)
    Xte = load_landmark_feature("test", source, feature)
    ytr, yva, yte = (load_y(s, num_classes) for s in ["train", "val", "test"])
    is_coord = feature == "raw_136"
    augment_train = scenario == "b3" and is_coord  # only raw coords support landmark aug
    cls_names = CLASS_NAMES_3 if num_classes == 3 else CLASS_NAMES_7
    input_dim = LANDMARK_FEATURE_DIM[feature]

    print(f"  Training landmark_fcnn × {feature}/{source}(dim={input_dim}) × {num_classes}c × "
          f"{scenario.upper()} (aug={augment_train})")
    train_ds = AugmentingLandmarkDataset(Xtr, ytr, augment=augment_train, feature_fn=None, seed=SEED)
    val_ds = AugmentingLandmarkDataset(Xva, yva, augment=False, feature_fn=None)
    test_ds = AugmentingLandmarkDataset(Xte, yte, augment=False, feature_fn=None)
    if scenario in ("b2", "b3"):
        sampler = lm_sampler(ytr, num_classes)
        train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler,
                                  drop_last=True, num_workers=0, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                                  drop_last=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=True)

    set_seed(SEED)
    model = EmotionFCNN(input_dim=input_dim, num_classes=num_classes).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()

    best_val = -1.0; best_state = None; best_epoch = -1; no_imp = 0
    t0 = time.time(); epochs_done = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            optim.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward(); optim.step()
        vm = evaluate_full(model, val_loader, num_classes, cls_names, device=device)
        epochs_done = epoch
        if vm["macro_f1"] > best_val:
            best_val = vm["macro_f1"]; best_epoch = epoch; no_imp = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                break
    elapsed = time.time() - t0
    model.load_state_dict(best_state)

    val_soft = _collect_soft(model, val_loader, _lm_forward)
    test_soft = _collect_soft(model, test_loader, _lm_forward)

    rec = {"arch": f"fcnn_{feature}_{src_tag}", "scenario": scenario, "num_classes": num_classes,
           "feature": feature, "source": source, "input_dim": input_dim,
           "best_epoch": best_epoch, "best_val_macro_f1": best_val,
           "elapsed_sec": elapsed, "epochs_done": epochs_done, "lr": LR}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(val_cache, val_soft); np.save(test_cache, test_soft)
    json.dump(rec, open(record_cache, "w"), indent=2)
    print(f"    best ep {best_epoch}  val_mf1={best_val:.4f}  ({elapsed:.0f}s)")
    return val_soft, test_soft, rec


def _image_forward(model, batch):
    xb, yb = batch
    xb = xb.to(device, non_blocking=True)
    return model(xb), yb


def _lm_forward(model, batch):
    xb, yb = batch
    xb = xb.to(device, non_blocking=True)
    return model(xb), yb


def _collect_soft(model, loader, forward_fn):
    """Run forward + softmax, return (N, C) array."""
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            logits, _ = forward_fn(model, batch)
            preds.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(preds, axis=0)


# ---- Late fusion combine ----

def late_fusion_combine(img_val, img_test, lm_val, lm_test, yval, ytest, num_classes, cls_names):
    """Sweep w, pick best on val, evaluate at test."""
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

    best_w = 0.5; best_val_mf1 = -1.0
    for w in WEIGHT_GRID:
        combined = w * img_val + (1.0 - w) * lm_val
        preds_v = combined.argmax(axis=1)
        mf1 = f1_score(yval, preds_v, average="macro", zero_division=0)
        if mf1 > best_val_mf1:
            best_val_mf1 = mf1
            best_w = float(w)

    # Apply best_w to test
    combined_test = best_w * img_test + (1.0 - best_w) * lm_test
    preds_t = combined_test.argmax(axis=1)
    test_mf1 = f1_score(ytest, preds_t, average="macro", zero_division=0)
    test_wf1 = f1_score(ytest, preds_t, average="weighted", zero_division=0)
    test_acc = accuracy_score(ytest, preds_t)
    cm = confusion_matrix(ytest, preds_t, labels=list(range(num_classes))).tolist()
    cls_report = classification_report(ytest, preds_t,
                                       labels=list(range(num_classes)),
                                       target_names=list(cls_names),
                                       output_dict=True, zero_division=0)

    return {
        "best_image_weight": best_w,
        "best_landmark_weight": 1.0 - best_w,
        "best_val_macro_f1": float(best_val_mf1),
        "test": {
            "accuracy": float(test_acc),
            "macro_f1": float(test_mf1),
            "weighted_f1": float(test_wf1),
            "confusion_matrix": cm,
            "classification_report": cls_report,
            "n_samples": int(len(ytest)),
        },
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None,
                    help="Override DATA_DIR (default: data/dataset_frontonly_conf60 = primer)")
    ap.add_argument("--output-prefix", default=None,
                    help="Output root, e.g. models/benchmark/kdef_7class (default: models/frontonly_conf60)")
    ap.add_argument("--scenarios", nargs="+", default=["b1", "b2", "b3"])
    ap.add_argument("--classes", nargs="+", type=int, default=[3, 7])
    ap.add_argument("--variants", nargs="+", default=["scratch", "tl"])
    ap.add_argument("--landmark-features", nargs="+", default=["raw_136"],
                    choices=list(LANDMARK_FEATURE_DIM.keys()),
                    help="Landmark branch feature(s) untuk Late Fusion. Default raw_136. "
                         "Bisa juga facs_28, blendshape_52, facs_plus_bs_80.")
    ap.add_argument("--landmark-sources", nargs="+", default=["mediapipe"],
                    help="Landmark source(s): mediapipe atau faceapi.")
    args = ap.parse_args()

    global DATA_DIR, OUTPUT_PREFIX, CACHE_DIR
    if args.data_dir is not None:
        DATA_DIR = Path(args.data_dir).resolve()
    if args.output_prefix is not None:
        OUTPUT_PREFIX = Path(args.output_prefix).resolve()
    CACHE_DIR = OUTPUT_PREFIX / "late_fusion_cache"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"Output prefix: {OUTPUT_PREFIX}")
    print(f"Cache dir: {CACHE_DIR}")

    all_records = {}
    schemes = list(args.classes)
    variants = list(args.variants)
    scenarios = list(args.scenarios)
    features = list(args.landmark_features)
    sources = list(args.landmark_sources)

    for nc in schemes:
        yval = load_y("val", nc)
        ytest = load_y("test", nc)
        cls_names = CLASS_NAMES_3 if nc == 3 else CLASS_NAMES_7

        # Image branch shared across (feature, source) — only depends on (variant, scen, nc)
        img_results = {}
        for variant in variants:
            arch_image = "cnn_scratch" if variant == "scratch" else "cnn_tl"
            for scen in scenarios:
                print(f"\n[{nc}c × {variant} × {scen.upper()}] Image branch")
                img_val, img_test, img_rec = train_image_branch(arch_image, scen, nc)
                img_results[(variant, scen)] = (img_val, img_test, img_rec)

        # Iterate landmark feature × source × scen (each combination needs its own training)
        for feature in features:
            for source in sources:
                if feature == "blendshape_52" and source != "mediapipe":
                    print(f"  SKIP combo: feature={feature} × source={source} (blendshape MP-only)")
                    continue
                # Train landmark branch per (feature, source, scen, nc)
                lm_results = {}
                for scen in scenarios:
                    print(f"\n[{nc}c × {feature}/{source} × {scen.upper()}] Landmark branch")
                    v, t, rec = train_landmark_branch(scen, nc, source=source, feature=feature)
                    lm_results[scen] = (v, t, rec)

                feat_tag = _feature_suffix(feature)
                src_tag = "" if source == "mediapipe" else f"_{source}"
                for variant in variants:
                    for scen in scenarios:
                        img_val, img_test, img_rec = img_results[(variant, scen)]
                        lm_val, lm_test, lm_rec = lm_results[scen]
                        print(f"\n  [{nc}c × {variant} × {feature}/{source} × {scen.upper()}] "
                              f"Computing late fusion weight sweep...")
                        lf = late_fusion_combine(img_val, img_test, lm_val, lm_test,
                                                 yval, ytest, nc, cls_names)
                        print(f"  best_w_image={lf['best_image_weight']:.2f}  "
                              f"val_mf1={lf['best_val_macro_f1']:.4f}  "
                              f"TEST mf1={lf['test']['macro_f1']:.4f}  "
                              f"wf1={lf['test']['weighted_f1']:.4f}  "
                              f"acc={lf['test']['accuracy']:.4f}")

                        key = f"fusion_late_{variant}{feat_tag}{src_tag}_{scen}_{nc}c"
                        record = {
                            "config": f"unified_{key}",
                            "notes": ("Late Fusion (Unified Protocol): weighted avg image_softmax + "
                                      "landmark_softmax. Image branch = CNN scratch/TL; landmark "
                                      f"branch = FCNN {feature} {source}. Both branches Unified "
                                      "Protocol. Weight w_image dari val macro_f1 sweep "
                                      "w ∈ [0,1] step 0.05."),
                            "fusion_type": "late",
                            "variant": variant,
                            "landmark_feature": feature,
                            "landmark_source": source,
                            "source": f"image_mp_crop + landmark_{source}_{feature}",
                            "scenario": scen.upper(),
                            "num_classes": nc,
                            "best_image_weight": lf["best_image_weight"],
                            "best_landmark_weight": lf["best_landmark_weight"],
                            "best_val_macro_f1": lf["best_val_macro_f1"],
                            "image_branch": img_rec,
                            "landmark_branch": lm_rec,
                            "test": lf["test"],
                        }
                        all_records[key] = record

    # Save grouped by (nc, variant, feature, source)
    for nc in schemes:
        for feature in features:
            for source in sources:
                if feature == "blendshape_52" and source != "mediapipe":
                    continue
                feat_tag = _feature_suffix(feature)
                src_tag = "" if source == "mediapipe" else f"_{source}"
                for variant in variants:
                    arch_label = f"fusion_late_{variant}{feat_tag}{src_tag}"
                    out_dir = OUTPUT_PREFIX / f"{nc}class" / "Unified" / arch_label
                    out_dir.mkdir(parents=True, exist_ok=True)
                    prefix = f"fusion_late_{variant}{feat_tag}{src_tag}_"
                    subset = {k: v for k, v in all_records.items()
                              if k.endswith(f"_{nc}c") and k.startswith(prefix)}
                    with open(out_dir / "results.json", "w") as f:
                        json.dump({"config": f"unified_{nc}c_{arch_label}", "runs": subset}, f, indent=2)
                    print(f"Saved: {out_dir/'results.json'}")
    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY: Late Fusion Unified Protocol macro_f1")
    print("=" * 100)
    print(f"  {'scheme':<6s}  {'variant':<8s}  {'feature':<18s}  {'source':<10s}  "
          f"{'B1':>8s}  {'B2':>8s}  {'B3':>8s}")
    for nc in schemes:
        for feature in features:
            for source in sources:
                if feature == "blendshape_52" and source != "mediapipe":
                    continue
                feat_tag = _feature_suffix(feature)
                src_tag = "" if source == "mediapipe" else f"_{source}"
                for variant in variants:
                    mfs = {}
                    for scen in scenarios:
                        key = f"fusion_late_{variant}{feat_tag}{src_tag}_{scen}_{nc}c"
                        if key in all_records:
                            mfs[scen] = all_records[key]["test"]["macro_f1"]
                    print(f"  {nc}c     {variant:<8s}  {feature:<18s}  {source:<10s}  "
                          f"{mfs.get('b1', float('nan')):>8.4f}  "
                          f"{mfs.get('b2', float('nan')):>8.4f}  "
                          f"{mfs.get('b3', float('nan')):>8.4f}")


if __name__ == "__main__":
    main()
