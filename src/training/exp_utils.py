"""
Experiment logging helpers — capture semua metrik per run untuk laporan/analisis.

Fields yang di-capture:
  - timestamp, git_commit, hardware (GPU/torch/cuda versions)
  - model: name, n_params, n_params_trainable, architecture (str repr)
  - hyperparams: batch, epochs, patience, lr, optimizer, loss, seed, dll
  - dataset: source, num_classes, split sizes, class counts per split
  - training: elapsed_sec, best_epoch, epochs_completed, early_stopped,
              peak_vram_mb, history (train_loss + train_acc + val metrics
              + epoch_time per epoch)
  - test: accuracy, macro_f1, weighted_f1, micro_f1, confusion_matrix,
          per-class classification_report (precision/recall/f1/support),
          inference_time_sec, inference_throughput samples/sec
"""
from __future__ import annotations

import datetime
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def git_commit_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def hardware_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        info["gpu_name"] = torch.cuda.get_device_name(idx)
        props = torch.cuda.get_device_properties(idx)
        info["gpu_total_vram_mb"] = int(props.total_memory / (1024 ** 2))
        info["cuda_version"] = torch.version.cuda
    return info


def model_info(model: nn.Module) -> Dict[str, Any]:
    n_all = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "name": type(model).__name__,
        "n_params": int(n_all),
        "n_params_trainable": int(n_trainable),
        "architecture": str(model),
    }


@torch.no_grad()
def evaluate_full(
    model: nn.Module,
    loader,
    num_classes: int,
    class_names: Sequence[str],
    forward_fn: Callable | None = None,
    device: torch.device | str | None = None,
) -> Dict[str, Any]:
    """Full evaluation: metrics + per-class report + timing.

    forward_fn(model, batch) → logits. Default: handle (x, y) or (x1, x2, y).
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    def _default_forward(model, batch):
        # Try common signatures
        if len(batch) == 2:
            xb, yb = batch
            xb = xb.to(device, non_blocking=True)
            return model(xb), yb
        # Multi-input (e.g., image + landmark + y)
        *inputs, yb = batch
        inputs = [t.to(device, non_blocking=True) for t in inputs]
        return model(*inputs), yb

    forward_fn = forward_fn or _default_forward

    preds, targets = [], []
    n_samples = 0
    t0 = time.time()
    for batch in loader:
        logits, yb = forward_fn(model, batch)
        preds.append(logits.argmax(1).cpu().numpy())
        targets.append(yb.numpy())
        n_samples += yb.size(0)
    inf_time = time.time() - t0
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    cls_report = classification_report(
        targets, preds,
        labels=list(range(num_classes)),
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(targets, preds)),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(targets, preds, average="weighted", zero_division=0)),
        "micro_f1": float(f1_score(targets, preds, average="micro", zero_division=0)),
        "confusion_matrix": confusion_matrix(
            targets, preds, labels=list(range(num_classes))
        ).tolist(),
        "classification_report": cls_report,
        "inference_time_sec": float(inf_time),
        "inference_throughput_samples_per_sec": float(n_samples / max(inf_time, 1e-9)),
        "n_samples": int(n_samples),
    }


def make_run_record(
    *,
    config: str,
    notes: str = "",
    hyperparams: Dict[str, Any],
    dataset: Dict[str, Any],
    model: nn.Module,
) -> Dict[str, Any]:
    """Initialize a run record with metadata. Append training+test results later."""
    return {
        "config": config,
        "notes": notes,
        "timestamp": datetime.datetime.now().isoformat(),
        "git_commit": git_commit_short(),
        "hardware": hardware_info(),
        "model": model_info(model),
        "hyperparams": hyperparams,
        "dataset": dataset,
    }


def class_counts(y: np.ndarray, num_classes: int) -> List[int]:
    return np.bincount(y, minlength=num_classes).astype(int).tolist()
