#!/bin/bash
# Jalankan 3 strategi robustness eval 7-class secara berurutan.
# Usage: nohup bash scripts/run_eval_7c_chain.sh > logs/eval_7c_chain.log 2>&1 &

set -e
cd "$(dirname "$0")/.."

PYTHON=/mnt/extended-home/fitra_dosen/2025_iris_fer_taufik/miniconda3/envs/2025_iris_fer_taufik/bin/python3.12
export CUDA_VISIBLE_DEVICES=1

echo "=== eval_7c_chain start: $(date) ==="

echo "--- [1/3] Random Split ---"
$PYTHON scripts/run_eval_7c.py --strategy random --seeds 5

echo "--- [2/3] CV5 ---"
$PYTHON scripts/run_eval_7c.py --strategy cv5

echo "--- [3/3] LOSO ---"
$PYTHON scripts/run_eval_7c.py --strategy loso

echo "=== eval_7c_chain DONE: $(date) ==="
