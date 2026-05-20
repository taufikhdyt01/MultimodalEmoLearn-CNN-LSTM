#!/bin/bash
# retry_run.sh — wrap any command with auto-retry on failure.
#
# Usage: ./retry_run.sh <max_retries> <wait_sec> <command...>
# Example: ./retry_run.sh 3 300 python scripts/run_unified_fusion.py --scenarios b1
#
# Retries up to max_retries times if command exits non-zero, waiting wait_sec
# seconds between attempts. Returns 0 if any attempt succeeds, last exit code
# if all attempts fail.

set -u
MAX_RETRIES=$1
WAIT_SEC=$2
shift 2
CMD="$@"

for attempt in $(seq 1 $MAX_RETRIES); do
    echo "[retry_run] Attempt $attempt/$MAX_RETRIES: $CMD"
    eval "$CMD"
    ret=$?
    if [ $ret -eq 0 ]; then
        echo "[retry_run] ✅ Succeeded on attempt $attempt"
        exit 0
    fi
    echo "[retry_run] ❌ Attempt $attempt exited with code $ret"
    if [ $attempt -lt $MAX_RETRIES ]; then
        echo "[retry_run] Waiting ${WAIT_SEC}s before next attempt..."
        sleep $WAIT_SEC
    fi
done

echo "[retry_run] All $MAX_RETRIES attempts failed. Last exit code: $ret"
exit $ret
