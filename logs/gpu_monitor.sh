#!/bin/bash
LOG=/mnt/extended-home/fitra_dosen/2025_iris_fer_taufik/MultimodalEmoLearn/logs/gpu_monitor.log
NB_LOG=/mnt/extended-home/fitra_dosen/2025_iris_fer_taufik/MultimodalEmoLearn/logs/nb84_run.log

echo "[$(date '+%H:%M:%S')] GPU monitor started (nb84 PID parent=1801745)" >> $LOG

while true; do
    TS=$(date '+%H:%M:%S')

    # Cek proses masih hidup
    if ! ps -p 1801745 > /dev/null 2>&1; then
        echo "[$TS] ⚠️  PROSES MATI (PID 1801745 tidak ditemukan)" >> $LOG
        break
    fi

    # Cek OOM di log
    if grep -q "out of memory\|CUDA error\|RuntimeError" $NB_LOG 2>/dev/null; then
        echo "[$TS] 🔴 OOM / ERROR TERDETEKSI di nb84_run.log!" >> $LOG
        grep -i "out of memory\|CUDA error\|RuntimeError" $NB_LOG | tail -3 >> $LOG
        break
    fi

    # Snapshot GPU 2
    GPU2_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 2)
    GPU2_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2)
    GPU2_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 2)
    OUR_MEM=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep 1803983 | awk -F',' '{print $2}' | tr -d ' ')

    echo "[$TS] GPU2: util=${GPU2_UTIL}% mem=${GPU2_MEM}MiB free=${GPU2_FREE}MiB | nb84=${OUR_MEM:-N/A}" >> $LOG

    # Alert kalau free VRAM < 3 GB
    if [ "$GPU2_FREE" -lt 3000 ] 2>/dev/null; then
        echo "[$TS] ⚠️  WARNING: Free VRAM < 3GB (${GPU2_FREE}MiB) — risiko OOM!" >> $LOG
    fi

    # Alert kalau util naik tinggi (training mulai)
    if [ "$GPU2_UTIL" -gt 40 ] 2>/dev/null; then
        echo "[$TS] 🟢 Training aktif (util=${GPU2_UTIL}%)" >> $LOG
    fi

    sleep 60
done

echo "[$(date '+%H:%M:%S')] Monitor selesai." >> $LOG
