# run_smoke.ps1
# Fast Debug / Smoke Configuration
# Uses 1 epoch and small batch size (16) with num_workers=0 to verify pipeline execution
$ErrorActionPreference = "Stop"
$python = "python"

Write-Output "========================================================================"
Write-Output "BiSSA Smoke Test (Debug Configuration)"
Write-Output "Batch size: 16 | Epochs: 1 | Num Workers: 0"
Write-Output "========================================================================"

Write-Output "`n>>> Running Davis Smoke Run (1 epoch) <<<"
& $python main.py --dataset davis --split random --batch_size 16 --max_epochs 1 --num_workers 0 --skip_pre_check
