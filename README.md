# BiSSA: Bidirectional Sequence–Structure Attention for Drug–Target Interaction and Affinity Prediction

[![Paper](https://img.shields.io/badge/Paper-PDF-blue)](#)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-green)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#)

> Official implementation of **"Bidirectional Sequence–Structure Attention for Drug–Target Interaction and Affinity Prediction"**
>
> **Sai Koushik Chowdary Thati** and **Charan Subhash Javvadi**  
> Department of Computer Science and Engineering, IIIT Sri City, India

---

## Overview

We propose a symmetric multimodal deep learning framework that integrates **pretrained sequence embeddings** (ChemBERTa-2 for drugs, ESM-2 for proteins) with **explicit graph representations** (RDKit molecular graphs, AlphaFold-derived protein residue contact graphs) for Drug–Target Interaction (DTI) and Drug–Target Affinity (DTA) prediction.

### Key Innovation: Bidirectional Cross-Attention (S ↔ G)

Within both the drug and protein decoders, two parallel multi-head cross-attention pathways execute **reciprocal information exchange**:

- **Sequence → Graph** path: sequence features attend to graph features (*S* ← *G*)
- **Graph → Sequence** path: graph features attend to sequence features (*G* ← *S*)

Each path is followed by 1D convolutional refinement and residual connections per Eq. (6). Four multi-head linear attention readout modules aggregate the resulting representations into a **512-dimensional joint embedding** for binary interaction classification or continuous affinity regression.

```
                    ┌──────────────────────────────────────┐
                    │         Drug Branch Decoder          │
  ChemBERTa-2 ──▶  │  S_d ← G_d  (sequence-queried)      │
  Drug Graph   ──▶  │  G_d ← S_d  (graph-queried)         │
                    │  + 1D Conv refinement + residual     │
                    └────────────┬─────────────────────────┘
                                 │
                    ┌────────────▼─────────────────────────┐
                    │    4× Linear Attention Readout       │
                    │    → 512-dim joint embedding z       │
                    │    → FC MLP → ŷ (DTA / DTI)          │
                    └────────────▲─────────────────────────┘
                                 │
                    ┌────────────┴─────────────────────────┐
                    │       Protein Branch Decoder         │
  ESM-2       ──▶   │  S_p ← G_p  (sequence-queried)      │
  AlphaFold   ──▶   │  G_p ← S_p  (graph-queried)         │
  Contact Graph     │  + 1D Conv refinement + residual     │
                    └──────────────────────────────────────┘
```

---

## Environment & Requirements

- **Paper Specification:** Python 3.10
- Core dependencies:
  - `torch >= 2.0.0`
  - `transformers >= 4.35.2` (DeepChem/ChemBERTa-77M-MLM)
  - `fair-esm >= 2.0.0` (esm2_t12_35M_UR50D)
  - `rdkit >= 2023.09.2`
  - `biopython >= 1.79`
  - `requests >= 2.28.0`
  - `scikit-learn >= 1.0.0`
  - `pandas >= 1.3.0`, `numpy >= 1.21.0`, `scipy >= 1.7.0`

Install dependencies via:
```bash
pip install -r requirements.txt
```

---

## Preprocessing Workflow

Preprocessing extracts frozen embeddings and constructs contact graphs offline.

### 1. Drug Sequence Embeddings (ChemBERTa-2, 384-d)
Extracts offline embeddings with [CLS]/[SEP] stripped and saves to `data/<dataset>/compound/mol_dict.pkl`:
```bash
python preprocessing/compound_pretrain.py --root_data_path data --dataset davis
python preprocessing/compound_pretrain.py --root_data_path data --dataset pdb
python preprocessing/compound_pretrain.py --root_data_path data --dataset tdc_dg
python preprocessing/compound_pretrain.py --root_data_path data --dataset bindingdb
```

### 2. Protein Sequence Embeddings (ESM-2 Layer 12, 480-d)
Extracts per-residue representations with BOS/EOS stripped, truncated to $N_p = \min(L, 1022)$, and saves `.npy` files to `data/<dataset>/protein/`:
```bash
# Regression benchmarks (Davis, PDBbind, TDC-DG)
python preprocessing/protein_pretrain/protein_dta.py --root_data_path data --dataset davis
python preprocessing/protein_pretrain/protein_dta.py --root_data_path data --dataset pdb
python preprocessing/protein_pretrain/protein_dta.py --root_data_path data --dataset tdc_dg

# Classification benchmark (BindingDB)
python preprocessing/protein_pretrain/protein_dti.py --root_data_path data --dataset bindingdb
```

### 3. Protein Contact Graphs (AlphaFold v4 / RCSB)
Fetches coordinates from AlphaFold DB v4 (UniProt) or RCSB PDB (4-letter code), extracts $C\alpha$ coordinates, and computes the binary contact adjacency at $< 8.0$ Å threshold with self-loops. Generates 22-d node features (21-d AA one-hot + normalized confidence/pLDDT) and writes `data/<dataset>/graph_coverage.json`:
```bash
python preprocessing/protein_pretrain/protein_structure.py --root_data_path data --dataset davis
python preprocessing/protein_pretrain/protein_structure.py --root_data_path data --dataset pdb
python preprocessing/protein_pretrain/protein_structure.py --root_data_path data --dataset tdc_dg
python preprocessing/protein_pretrain/protein_structure.py --root_data_path data --dataset bindingdb
```

---

## Reproducing the Paper

All runs use global seed 0, Adam optimizer ($10^{-3}$), batch size 128, dropout 0.2, and maximum 200 epochs with `ReduceLROnPlateau` (patience 30, factor 0.1).

Runs write strictly into timestamped directories under `runs/`:
```
runs/<dataset>_<split>_<variant>_<timestamp>/
├── config.json               # Complete effective arguments + git commit hash
├── split_stats.json          # Exact dataset split counts and entity isolation stats
├── train.log                 # Per-epoch training loss, validation metrics, and LR
├── best_model.pth            # Checkpoint selected strictly on best validation metric
└── final_test_metrics.json   # Final evaluation on test partition (evaluated once)
```

> **Note on Published Artifacts:** `benchmark_results.csv` and `results_report.md` preserve the published historical reference numbers and are never hand-written or overwritten by training runs.

### Table I — DTA Benchmarks (Random Splits)

| Benchmark | Table Cell | Exact Command |
|:---|:---|:---|
| Davis (DTA) | Full Proposed (BiSSA) | `python main.py --dataset davis --task dta --split random --variant full --batch_size 128 --max_epochs 200 --seed 0` |
| Davis (DTA) | Ablation A1 (Graph-Only) | `python main.py --dataset davis --task dta --split random --variant A1 --batch_size 128 --max_epochs 200 --seed 0` |
| Davis (DTA) | Ablation A2 (Seq-Only) | `python main.py --dataset davis --task dta --split random --variant A2 --batch_size 128 --max_epochs 200 --seed 0` |
| Davis (DTA) | Ablation A3 (w/o Prot Graph) | `python main.py --dataset davis --task dta --split random --variant A3 --batch_size 128 --max_epochs 200 --seed 0` |
| Davis (DTA) | Ablation A4 (Concat) | `python main.py --dataset davis --task dta --split random --variant A4 --batch_size 128 --max_epochs 200 --seed 0` |
| Davis (DTA) | Ablation A5 ($S \leftarrow G$) | `python main.py --dataset davis --task dta --split random --variant A5 --batch_size 128 --max_epochs 200 --seed 0` |
| Davis (DTA) | Ablation A6 ($G \leftarrow S$) | `python main.py --dataset davis --task dta --split random --variant A6 --batch_size 128 --max_epochs 200 --seed 0` |
| PDBbind (DTA) | Full Proposed | `python main.py --dataset pdbbind --task dta --variant full --batch_size 128 --max_epochs 200 --seed 0` |
| TDC-DG (DTA) | Full Proposed | `python main.py --dataset tdc_dg --task dta --variant full --batch_size 128 --max_epochs 200 --seed 0` |

### Table II — Davis Generalization (Cold Splits)

| Setting | Table Cell | Exact Command |
|:---|:---|:---|
| Davis Cold Target | Full Proposed | `python main.py --dataset davis --task dta --split cold_target --variant full --batch_size 128 --max_epochs 200 --seed 0` |
| Davis Cold Target | Ablations A1–A6 | `python main.py --dataset davis --task dta --split cold_target --variant {A1,A2,A3,A4,A5,A6} --batch_size 128 --max_epochs 200 --seed 0` |
| Davis Cold Drug | Full Proposed | `python main.py --dataset davis --task dta --split cold_drug --variant full --batch_size 128 --max_epochs 200 --seed 0` |
| Davis Cold Pair | Full Proposed | `python main.py --dataset davis --task dta --split cold_pair --variant full --batch_size 128 --max_epochs 200 --seed 0` |

### Table IV — BindingDB DTI Classification

| Setting | Table Cell | Exact Command |
|:---|:---|:---|
| BindingDB DTI | Full Proposed | `python main.py --dataset bindingdb --task dti --variant full --batch_size 128 --max_epochs 200 --seed 0` |
| BindingDB DTI | Ablation A1 (Graph-Only) | `python main.py --dataset bindingdb --task dti --variant A1 --batch_size 128 --max_epochs 200 --seed 0` |
| BindingDB DTI | Ablation A2 (Seq-Only) | `python main.py --dataset bindingdb --task dti --variant A2 --batch_size 128 --max_epochs 200 --seed 0` |
| BindingDB DTI | Ablation A3 (w/o Prot Graph) | `python main.py --dataset bindingdb --task dti --variant A3 --batch_size 128 --max_epochs 200 --seed 0` |
| BindingDB DTI | Ablation A4 (Concat) | `python main.py --dataset bindingdb --task dti --variant A4 --batch_size 128 --max_epochs 200 --seed 0` |
| BindingDB DTI | Ablation A5 ($S \leftarrow G$) | `python main.py --dataset bindingdb --task dti --variant A5 --batch_size 128 --max_epochs 200 --seed 0` |
| BindingDB DTI | Ablation A6 ($G \leftarrow S$) | `python main.py --dataset bindingdb --task dti --variant A6 --batch_size 128 --max_epochs 200 --seed 0` |

---

## Batch Scripts

- **Full Benchmark Suite:**
  - PowerShell (Windows): `.\run_all.ps1`
  - Bash (Linux/macOS): `./run_all.sh`
- **Fast Smoke / Verification Suite:**
  - PowerShell (Windows): `.\run_smoke.ps1`
  - Bash (Linux/macOS): `./run_smoke.sh`

---

## Multi-GPU & Hardware Acceleration

The model supports multi-GPU training via PyTorch `DataParallel`.
```bash
# Use all visible GPUs
python main.py --dataset davis --gpus all

# Use specific GPUs
python main.py --dataset davis --gpus 0,1,2,3

# Force CPU
python main.py --dataset davis --gpus cpu
```

Checkpoints automatically unwrap the module prefix so saved models load seamlessly across single-GPU, multi-GPU, and CPU environments.

---

## Verification & Audit Tools

- **Architecture Audit:** `python verify_architecture.py` verifies all tensor dimensions, parameter counts (2.42M DTI / 2.95M DTA), post-LN encoders, and Eq. (6) projected query residuals.
- **Protocol Audit:** `python verify_protocol.py` verifies split isolation, test-set locking, metrics unit tests, and smoke execution.
- **Leakage Audit:** `python leakage_report.py` audits duplicate pairs and label conflicts across datasets and generates `leakage_report.json`.

---

## Citation

```bibtex
@article{thati2026bissa,
  title={Bidirectional Sequence--Structure Attention for Drug--Target Interaction and Affinity Prediction},
  author={Thati, Sai Koushik Chowdary and Javvadi, Charan Subhash},
  year={2026},
  institution={Indian Institute of Information Technology, Sri City}
}
```
