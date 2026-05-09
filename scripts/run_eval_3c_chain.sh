#!/bin/bash
# Chain robustness eval 3-class: random → cv5 → loso
# Usage: CUDA_VISIBLE_DEVICES=1 nohup bash scripts/run_eval_3c_chain.sh > logs/eval_3c_chain.log 2>&1 &

set -e
cd "$(dirname "$0")/.."

PYTHON=/mnt/extended-home/fitra_dosen/2025_iris_fer_taufik/miniconda3/envs/2025_iris_fer_taufik/bin/python3.12

echo "====================================================="
echo "  ROBUSTNESS EVAL 3-CLASS — CHAIN"
echo "  Started: $(date)"
echo "====================================================="

echo ""
echo "[1/3] Random Split (5 seeds × 3 models = 15 runs, ~2.5h)..."
$PYTHON scripts/run_eval_3c.py --strategy random --seeds 5

echo ""
echo "[2/3] 5-Fold CV subject-wise (5 folds × 3 models = 15 runs, ~7h)..."
$PYTHON scripts/run_eval_3c.py --strategy cv5

echo ""
echo "[3/3] LOSO 37-fold (37 folds × 3 models = 111 runs, ~30h)..."
$PYTHON scripts/run_eval_3c.py --strategy loso

echo ""
echo "====================================================="
echo "  ALL DONE: $(date)"
echo "====================================================="
