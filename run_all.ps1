# run_all.ps1
# Full Paper Benchmark Training
# Paper: "Bidirectional Sequence–Structure Attention for Drug–Target Interaction and Affinity Prediction"
# Hyperparameters: batch_size=128, max_epochs=200, lr=1e-3, seed=0
$ErrorActionPreference = "Stop"
$python = "python"

Write-Output "========================================================================"
Write-Output "BiSSA Benchmark Replication Suite (Full Paper Protocol)"
Write-Output "Batch size: 128 | Epochs: 200 | LR: 0.001 | Seed: 0"
Write-Output "========================================================================"

# --- 1. Davis DTA Benchmark (Table I & Table II) ---
Write-Output "`n>>> [1/7] Running Davis DTA (Random Split, 80/10/10) <<<"
& $python main.py --dataset davis --task dta --split random --batch_size 128 --max_epochs 200 --seed 0

Write-Output "`n>>> [2/7] Running Davis DTA (Cold Target Split) <<<"
& $python main.py --dataset davis --task dta --split cold_target --batch_size 128 --max_epochs 200 --seed 0

Write-Output "`n>>> [3/7] Running Davis DTA (Cold Drug Split) <<<"
& $python main.py --dataset davis --task dta --split cold_drug --batch_size 128 --max_epochs 200 --seed 0

Write-Output "`n>>> [4/7] Running Davis DTA (Cold Pair Split) <<<"
& $python main.py --dataset davis --task dta --split cold_pair --batch_size 128 --max_epochs 200 --seed 0

# --- 2. PDBbind DTA Benchmark (Table I) ---
Write-Output "`n>>> [5/7] Running PDBbind DTA <<<"
& $python main.py --dataset pdbbind --task dta --batch_size 128 --max_epochs 200 --seed 0

# --- 3. TDC-DG DTA Benchmark (Table I) ---
Write-Output "`n>>> [6/7] Running TDC-DG DTA <<<"
& $python main.py --dataset tdc_dg --task dta --batch_size 128 --max_epochs 200 --seed 0

# --- 4. BindingDB DTI Benchmark (Table IV) ---
Write-Output "`n>>> [7/7] Running BindingDB DTI (Classification) <<<"
& $python main.py --dataset bindingdb --task dti --batch_size 128 --max_epochs 200 --seed 0

Write-Output "`nAll benchmark runs completed. Check runs/ directory for logs and final metrics."
