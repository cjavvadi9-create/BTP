#!/usr/bin/env bash
# run_all.sh
# Full Paper Benchmark Training
# Paper: "Bidirectional Sequence–Structure Attention for Drug–Target Interaction and Affinity Prediction"
# Hyperparameters: batch_size=128, max_epochs=200, lr=1e-3, seed=0
set -euo pipefail

PYTHON="${PYTHON:-python}"

echo "========================================================================"
echo "BiSSA Benchmark Replication Suite (Full Paper Protocol)"
echo "Batch size: 128 | Epochs: 200 | LR: 0.001 | Seed: 0"
echo "========================================================================"

# --- 1. Davis DTA Benchmark (Table I & Table II) ---
echo -e "\n>>> [1/7] Running Davis DTA (Random Split, 80/10/10) <<<"
$PYTHON main.py --dataset davis --task dta --split random --batch_size 128 --max_epochs 200 --seed 0

echo -e "\n>>> [2/7] Running Davis DTA (Cold Target Split) <<<"
$PYTHON main.py --dataset davis --task dta --split cold_target --batch_size 128 --max_epochs 200 --seed 0

echo -e "\n>>> [3/7] Running Davis DTA (Cold Drug Split) <<<"
$PYTHON main.py --dataset davis --task dta --split cold_drug --batch_size 128 --max_epochs 200 --seed 0

echo -e "\n>>> [4/7] Running Davis DTA (Cold Pair Split) <<<"
$PYTHON main.py --dataset davis --task dta --split cold_pair --batch_size 128 --max_epochs 200 --seed 0

# --- 2. PDBbind DTA Benchmark (Table I) ---
echo -e "\n>>> [5/7] Running PDBbind DTA <<<"
$PYTHON main.py --dataset pdbbind --task dta --batch_size 128 --max_epochs 200 --seed 0

# --- 3. TDC-DG DTA Benchmark (Table I) ---
echo -e "\n>>> [6/7] Running TDC-DG DTA <<<"
$PYTHON main.py --dataset tdc_dg --task dta --batch_size 128 --max_epochs 200 --seed 0

# --- 4. BindingDB DTI Benchmark (Table IV) ---
echo -e "\n>>> [7/7] Running BindingDB DTI (Classification) <<<"
$PYTHON main.py --dataset bindingdb --task dti --batch_size 128 --max_epochs 200 --seed 0

echo -e "\nAll benchmark runs completed. Check runs/ directory for logs and final metrics."
