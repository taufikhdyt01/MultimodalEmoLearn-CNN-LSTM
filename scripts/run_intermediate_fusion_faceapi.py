#!/usr/bin/env python3
"""
Image-based pipeline + face-api.js: apakah fusion model dapat lift?

Train IntermediateFusionTransfer (ResNet18 image + FCNN landmark) di:
  landmark: {mediapipe, faceapi} × class: {3, 7} = 4 runs.
Image stream tetap sama (face-crop 224×224 RGB). Hanya landmark stream swap.

Hipotesis: FCNN+face-api.js solo dapat +0.20 macro_f1. Apakah lift bertahan
saat di-fuse dengan image features?
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
from training.models import IntermediateFusionTransfer  # noqa: E402
from training.exp_utils import (  # noqa: E402
    class_counts, evaluate_full, make_run_record,
)

DATA_DIR = PROJECT_ROOT / "data" / "dataset_frontonly_conf60"

REMAP_3 = np.array([1, 0, 2, 2, 2, 2, 0], dtype=np.int64)

CLASS_NAMES_7 = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]
CLASS_NAMES_3 = ["positive", "neutral", "negative"]

BATCH = 32
EPOCHS = 30
PATIENCE = 8
LR = 5e-5
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class ImageLandmarkDataset(Dataset):
    def __init__(self, X_img, X_lm, y):
        self.X_img = torch.from_numpy(X_img).permute(0, 3, 1, 2).contiguous()
        self.X_lm = torch.from_numpy(X_lm)
        self.y = torch.from_numpy(y)

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        return self.X_img[i], self.X_lm[i], self.y[i]


def load_data(lm_source: str, num_classes: int):
    out = {}
    lm_fname = ("X_{}_landmarks.npy" if lm_source == "mediapipe"
                else "X_{}_faceapi_landmarks.npy")
    for split in ("train", "val", "test"):
        mask = np.load(DATA_DIR / f"mask_{split}_faceapi.npy")
        assert mask.all()
        X_img = np.load(DATA_DIR / f"X_{split}_images.npy").astype(np.float32)
        X_lm = np.load(DATA_DIR / lm_fname.format(split)).astype(np.float32)
        assert not np.isnan(X_lm).any()
        y = np.load(DATA_DIR / f"y_{split}.npy").astype(np.int64)
        if num_classes == 3:
            y = REMAP_3[y]
        out[split] = (ImageLandmarkDataset(X_img, X_lm, y), y)  # also keep y for stats
    return out


def make_loader(ds, shuffle):
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, drop_last=shuffle,
                      num_workers=0, pin_memory=True)


def train_run(data, lm_source: str, num_classes: int):
    cls_names = CLASS_NAMES_3 if num_classes == 3 else CLASS_NAMES_7
    train_ds, ytr = data["train"]; val_ds, yva = data["val"]; test_ds, yte = data["test"]
    print(f"\n========== IF-TL  landmark={lm_source}  {num_classes}c ==========")

    set_seed(SEED)
    model = IntermediateFusionTransfer(num_classes=num_classes, landmark_dim=136).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()  # B1

    tl = make_loader(train_ds, True)
    vl = make_loader(val_ds, False)
    el = make_loader(test_ds, False)

    record = make_run_record(
        config=f"if_tl_b1_{lm_source}_{num_classes}c",
        notes="Intermediate Fusion (ResNet18 + FCNN landmark) with face-api.js vs MediaPipe landmark.",
        hyperparams={
            "batch_size": BATCH, "epochs_max": EPOCHS, "patience": PATIENCE,
            "lr": LR, "optimizer": "Adam", "loss": "CrossEntropyLoss",
            "seed": SEED, "scenario": "B1", "pretrained_resnet18": True,
        },
        dataset={
            "data_dir": str(DATA_DIR.relative_to(PROJECT_ROOT)),
            "landmark_source": lm_source,
            "image_source": "mediapipe_face_crop_224",
            "num_classes": num_classes,
            "class_names": cls_names,
            "n_train": int(len(train_ds)), "n_val": int(len(val_ds)), "n_test": int(len(test_ds)),
            "class_counts_train": class_counts(ytr, num_classes),
            "class_counts_val": class_counts(yva, num_classes),
            "class_counts_test": class_counts(yte, num_classes),
        },
        model=model,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    best_val = -1.0; best_state = None; best_epoch = -1; no_imp = 0
    history = []
    t0 = time.time()
    early_stopped = False
    epochs_done = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        et0 = time.time()
        total = 0.0; correct = 0; n = 0
        for img, lm, yb in tl:
            img = img.to(device, non_blocking=True)
            lm = lm.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optim.zero_grad()
            logits = model(img, lm)
            loss = crit(logits, yb)
            loss.backward(); optim.step()
            total += loss.item() * img.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            n += img.size(0)
        epoch_time = time.time() - et0
        train_loss = total / max(n, 1)
        train_acc = correct / max(n, 1)
        vm = evaluate_full(model, vl, num_classes, cls_names, device=device)
        history.append({
            "epoch": epoch,
            "epoch_time_sec": epoch_time,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_macro_f1": vm["macro_f1"],
            "val_weighted_f1": vm["weighted_f1"],
            "val_accuracy": vm["accuracy"],
        })
        epochs_done = epoch
        print(f"  ep {epoch:2d}/{EPOCHS}  ({epoch_time:.0f}s)  loss={train_loss:.4f}  tr_acc={train_acc:.4f}  val_mf1={vm['macro_f1']:.4f}  val_acc={vm['accuracy']:.4f}")
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

    print(f"  best epoch {best_epoch}  val_mf1={best_val:.4f}  ({elapsed:.0f}s)  peak_vram={peak_vram_mb:.0f}MB")
    print(f"  TEST: macro_f1={test_m['macro_f1']:.4f}  wf1={test_m['weighted_f1']:.4f}  acc={test_m['accuracy']:.4f}")

    record["training"] = {
        "elapsed_sec": elapsed,
        "epochs_completed": epochs_done,
        "best_epoch": best_epoch,
        "early_stopped": early_stopped,
        "peak_vram_mb": float(peak_vram_mb),
        "history": history,
    }
    record["test"] = test_m
    record["val_at_best"] = val_m_best
    return record


def main():
    print(f"Device: {device}")
    all_records = {}
    for nc in (3, 7):
        data = {}
        for src in ("mediapipe", "faceapi"):
            data[src] = load_data(src, nc)
        for src in ("mediapipe", "faceapi"):
            all_records[f"{src}_{nc}c"] = train_run(data[src], src, nc)
        del data
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for nc in (3, 7):
        out_dir = PROJECT_ROOT / "models" / "frontonly_conf60" / f"{nc}class" / "IntermediateFusion_TL_compare"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_json = out_dir / "compare_landmark_source.json"
        with open(out_json, "w") as f:
            json.dump({
                "config": f"if_tl_b1_{nc}c_landmark_source_compare",
                "mediapipe": all_records[f"mediapipe_{nc}c"],
                "faceapi": all_records[f"faceapi_{nc}c"],
            }, f, indent=2)
        print(f"\nSaved: {out_json}")

    print("\n" + "=" * 70)
    print("Summary: IntermediateFusion_TL B1 (image+landmark)")
    print("=" * 70)
    for nc in (3, 7):
        mp = all_records[f"mediapipe_{nc}c"]["test"]
        fa = all_records[f"faceapi_{nc}c"]["test"]
        print(f"\n  --- {nc}-class ---")
        print(f"  {'metric':<20s}  {'MediaPipe':>10s}  {'face-api.js':>12s}  {'delta':>8s}")
        for k in ("macro_f1", "weighted_f1", "accuracy"):
            print(f"  test_{k:<14s}  {mp[k]:>10.4f}  {fa[k]:>12.4f}  {fa[k]-mp[k]:>+8.4f}")


if __name__ == "__main__":
    main()
