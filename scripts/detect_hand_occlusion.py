"""Detect hand occlusion in cropped face dataset using MediaPipe HandLandmarker (tasks API).

Input  : data/dataset_frontonly_conf60/X_{split}_images.npy (N, 224, 224, 3) float32 [0,1] or uint8
Output : deploy/data_cleaning/hand_occlusion_{split}.npz
            occluded_mask (N,) bool  -- True = hand detected on face crop
            hand_count    (N,) int8
            max_score     (N,) float32 (handedness score)
         deploy/data_cleaning/hand_occlusion_report.md

Usage:
    python scripts/detect_hand_occlusion.py
    python scripts/detect_hand_occlusion.py --min-conf 0.3 --splits train val test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "dataset_frontonly_conf60"
OUT_DIR = ROOT / "deploy" / "data_cleaning"
MODEL_PATH = OUT_DIR / "hand_landmarker.task"


def to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    """Dataset images stored as float32 in [0,1] or uint8 — normalize to uint8 RGB."""
    if img.dtype == np.uint8:
        return np.ascontiguousarray(img)
    arr = img.astype(np.float32)
    if arr.max() <= 1.5:
        arr = arr * 255.0
    return np.ascontiguousarray(np.clip(arr, 0, 255).astype(np.uint8))


def make_landmarker(min_conf: float) -> mp_vision.HandLandmarker:
    base_opts = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
    opts = mp_vision.HandLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=min_conf,
        min_hand_presence_confidence=min_conf,
        min_tracking_confidence=min_conf,
    )
    return mp_vision.HandLandmarker.create_from_options(opts)


def detect_split(split: str, min_conf: float) -> dict:
    images_path = DATA_DIR / f"X_{split}_images.npy"
    labels_path = DATA_DIR / f"y_{split}.npy"
    if not images_path.exists():
        raise FileNotFoundError(images_path)

    images = np.load(images_path)
    labels = np.load(labels_path) if labels_path.exists() else None
    n = len(images)
    print(f"[{split}] {n} samples, shape={images.shape}, dtype={images.dtype}")

    occluded = np.zeros(n, dtype=bool)
    hand_count = np.zeros(n, dtype=np.int8)
    max_score = np.zeros(n, dtype=np.float32)

    landmarker = make_landmarker(min_conf)
    try:
        for i in range(n):
            img_u8 = to_uint8_rgb(images[i])
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_u8)
            result = landmarker.detect(mp_img)
            if result.handedness:
                k = len(result.handedness)
                hand_count[i] = k
                occluded[i] = True
                scores = [h[0].score for h in result.handedness]
                max_score[i] = float(max(scores))
            if (i + 1) % 500 == 0 or i == n - 1:
                pct = 100 * occluded[: i + 1].mean()
                print(f"  [{split}] {i+1}/{n}  occluded so far: {occluded[: i+1].sum()} ({pct:.1f}%)")
    finally:
        landmarker.close()

    out_path = OUT_DIR / f"hand_occlusion_{split}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, occluded_mask=occluded, hand_count=hand_count, max_score=max_score)
    print(f"[{split}] saved -> {out_path}")

    summary = {
        "split": split,
        "n": int(n),
        "occluded": int(occluded.sum()),
        "occluded_pct": float(100 * occluded.mean()),
    }
    if labels is not None:
        per_class = {}
        for c in np.unique(labels):
            mask = labels == c
            per_class[int(c)] = {
                "n": int(mask.sum()),
                "occluded": int(occluded[mask].sum()),
                "occluded_pct": float(100 * occluded[mask].mean()) if mask.any() else 0.0,
            }
        summary["per_class"] = per_class
    return summary


def write_report(summaries: list[dict], min_conf: float) -> Path:
    info_path = DATA_DIR / "dataset_info.json"
    emotions = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        emotions = info.get("emotions", emotions)

    lines = [
        "# Hand Occlusion Detection Report",
        "",
        f"Dataset: `data/dataset_frontonly_conf60/` — MediaPipe HandLandmarker (tasks API, min_conf={min_conf}, num_hands=2, RunningMode.IMAGE)",
        "",
        "**Interpretasi:** Sample ditandai `occluded` jika MediaPipe mendeteksi ≥1 tangan pada crop wajah 224×224. Karena gambar input sudah merupakan face crop, deteksi tangan di crop = tangan menutupi area wajah.",
        "",
        "## Ringkasan per Split",
        "",
        "| Split | N | Occluded | % |",
        "|:---|---:|---:|---:|",
    ]
    total_n = total_occ = 0
    for s in summaries:
        lines.append(f"| {s['split']} | {s['n']} | {s['occluded']} | {s['occluded_pct']:.2f}% |")
        total_n += s["n"]
        total_occ += s["occluded"]
    lines.append(f"| **TOTAL** | **{total_n}** | **{total_occ}** | **{100*total_occ/total_n:.2f}%** |")

    lines += ["", "## Breakdown per Kelas (semua split digabung)", ""]
    agg: dict[int, dict] = {}
    for s in summaries:
        for c, d in s.get("per_class", {}).items():
            a = agg.setdefault(int(c), {"n": 0, "occluded": 0})
            a["n"] += d["n"]
            a["occluded"] += d["occluded"]
    lines.append("| Class ID | Emotion | N | Occluded | % |")
    lines.append("|:---:|:---|---:|---:|---:|")
    for c in sorted(agg):
        d = agg[c]
        name = emotions[c] if c < len(emotions) else str(c)
        pct = 100 * d["occluded"] / d["n"] if d["n"] else 0.0
        lines.append(f"| {c} | {name} | {d['n']} | {d['occluded']} | {pct:.2f}% |")

    lines += [
        "",
        "## File Output",
        "",
        "```",
        "deploy/data_cleaning/hand_occlusion_{train,val,test}.npz",
        "  occluded_mask (N,) bool",
        "  hand_count    (N,) int8",
        "  max_score     (N,) float32",
        "```",
        "",
        "Cara filter training set:",
        "",
        "```python",
        "import numpy as np",
        "mask = np.load('deploy/data_cleaning/hand_occlusion_train.npz')['occluded_mask']",
        "keep = ~mask",
        "X_clean = np.load('data/dataset_frontonly_conf60/X_train_images.npy')[keep]",
        "y_clean = np.load('data/dataset_frontonly_conf60/y_train.npy')[keep]",
        "```",
    ]

    report_path = OUT_DIR / "hand_occlusion_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--min-conf", type=float, default=0.3)
    args = ap.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit(f"Model not found: {MODEL_PATH}\nDownload: curl -L -o {MODEL_PATH} "
                         "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                         "hand_landmarker/float16/latest/hand_landmarker.task")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = [detect_split(s, args.min_conf) for s in args.splits]
    report = write_report(summaries, args.min_conf)
    print(f"\nReport: {report}")
    for s in summaries:
        print(f"  {s['split']:5s}: {s['occluded']}/{s['n']} ({s['occluded_pct']:.2f}%) occluded")


if __name__ == "__main__":
    main()
