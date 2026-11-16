#!/usr/bin/env bash
# run_smoke.sh
# Fast Debug / Smoke Configuration
# Uses 1 epoch and small batch size (16) with num_workers=0 to verify pipeline execution
set -euo pipefail

PYTHON="${PYTHON:-python}"

echo "========================================================================"
echo "BiSSA Smoke Test (Debug Configuration)"
echo "Batch size: 16 | Epochs: 1 | Num Workers: 0"
echo "========================================================================"

echo -e "\n>>> Running Davis Smoke Run (1 epoch) <<<"
$PYTHON main.py --dataset davis --split random --batch_size 16 --max_epochs 1 --num_workers 0 --skip_pre_check
