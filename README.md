# BiSSA: Bidirectional Sequence–Structure Attention for Drug–Target Interaction and Affinity Prediction

[![Paper](https://img.shields.io/badge/Paper-PDF-blue)](#)
[![Python 3.7+](https://img.shields.io/badge/Python-3.7%2B-green)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.4%2B-red)](https://pytorch.org/)
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

Each path is followed by 1D convolutional refinement and residual connections. Four multi-head linear attention readout modules aggregate the resulting representations into a **512-dimensional joint embedding** for binary interaction classification or continuous affinity regression.

```
                    ┌──────────────────────────────────────┐
                    │         Drug Branch Decoder           │
  ChemBERTa-2 ──▶  │  S_d ← G_d  (sequence-queried)      │
  Drug Graph   ──▶  │  G_d ← S_d  (graph-queried)         │
                    │  + 1D Conv refinement + residual      │
                    └────────────┬─────────────────────────┘
                                 │
                    ┌────────────▼─────────────────────────┐
                    │    4× Linear Attention Readout        │
                    │    → 512-dim joint embedding z        │
                    │    → FC MLP → ŷ (DTA / DTI)          │
                    └────────────▲─────────────────────────┘
                                 │
                    ┌────────────┴─────────────────────────┐
                    │       Protein Branch Decoder          │
  ESM-2       ──▶   │  S_p ← G_p  (sequence-queried)      │
  AlphaFold   ──▶   │  G_p ← S_p  (graph-queried)         │
  Contact Graph     │  + 1D Conv refinement + residual      │
                    └──────────────────────────────────────┘
```

---

## Results

### DTA Regression (Davis, Random 80/10/10 Split)

| Model | MSE ↓ | RMSE ↓ | CI ↑ | Pearson *r* ↑ |
|:---|:---:|:---:|:---:|:---:|
| DeepDTA | 0.261 | 0.511 | 0.878 | 0.806 |
| GraphDTA | 0.229 | 0.478 | 0.893 | 0.824 |
| DGraphDTA | 0.204 | 0.452 | 0.904 | 0.838 |
| MGraphDTA | 0.207 | 0.455 | 0.900 | 0.841 |
| **BiSSA (Ours)** | **0.181** | **0.426** | **0.912** | **0.862** |

### DTA Regression (PDBbind v2016)

| Model | MSE ↓ | RMSE ↓ | CI ↑ | Pearson *r* ↑ |
|:---|:---:|:---:|:---:|:---:|
| GraphDTA | 1.855 | 1.362 | 0.782 | 0.754 |
| DGraphDTA | 1.727 | 1.314 | 0.794 | 0.772 |
| **BiSSA (Ours)** | **1.508** | **1.228** | **0.821** | **0.811** |

### Cold-Target Generalization (Davis)

| Model | MSE ↓ | CI ↑ | Pearson *r* ↑ |
|:---|:---:|:---:|:---:|
| GraphDTA | 0.456 | 0.785 | 0.624 |
| **BiSSA (Ours)** | **0.338** | **0.848** | **0.738** |

### DTI Classification (BindingDB)

| Model | AUROC ↑ | AUPR ↑ | F1 ↑ | Accuracy ↑ |
|:---|:---:|:---:|:---:|:---:|
| DrugBAN | 0.946 | 0.932 | 0.874 | 0.882 |
| **BiSSA (Ours)** | **0.963** | **0.956** | **0.902** | **0.908** |

> Complete benchmark tables including TDC-DG temporal splits, all cold-start protocols, and ablation studies are available in [`results_report.md`](results_report.md) and [`benchmark_results.csv`](benchmark_results.csv).

---

## Repository Structure

```
BiSSA/
├── models/
│   ├── core.py              # Main BiSSA model architecture
│   ├── decoder.py           # Bidirectional cross-attention decoders
│   └── gnn.py               # GCN encoders for drug and protein graphs
├── preprocessing/
│   └── protein_pretrain/     # ESM-2 protein feature extraction
│   └── compound_pretrain.py  # ChemBERTa-2 drug feature extraction
├── data/
│   ├── davis/                # Davis kinase benchmark
│   ├── pdb/                  # PDBbind v2016 benchmark
│   └── bindingdb/            # BindingDB DTI classification benchmark
├── data.py                   # Dataset loading and graph construction
├── main.py                   # Training & evaluation entry point
├── utils.py                  # Metrics (CI, MSE, RMSE, Pearson, etc.)
├── benchmark_results.csv     # All paper benchmark results (machine-readable)
└── results_report.md         # Detailed results report with tables
```

---

## Requirements

```
python >= 3.7
pytorch >= 1.4.0
cudatoolkit >= 10.1
numpy >= 1.16.4
scikit-learn >= 0.21.2
pandas >= 0.24.2
scipy >= 1.3.0
rdkit >= 2023.09.2
esm == 2.0.0
transformers >= 4.35.2
```

---

## Data Preprocessing

### Protein Features (ESM-2)

For **regression** benchmarks (Davis, PDBbind):
```bash
cd preprocessing/protein_pretrain
python protein_dta.py --dataset davis   # or --dataset pdb
```

For **classification** benchmarks (BindingDB):
```bash
cd preprocessing/protein_pretrain
python protein_dti.py --dataset bindingdb
```

### Drug Features (ChemBERTa-2)
```bash
cd preprocessing
python compound_pretrain.py --dataset davis   # or pdb, bindingdb
```

### Protein Contact Graphs (AlphaFold / RCSB)

Residue contact graphs are constructed using a **Cα distance threshold of 8.0 Å** from AlphaFold v4 predicted structures (Davis, TDC-DG, BindingDB) or RCSB PDB experimental coordinates (PDBbind v2016). Each residue node carries a 22-dimensional feature vector (20 amino acid one-hot + normalized position + pLDDT confidence).

---

## Training & Evaluation

### DTA Regression
```bash
# Davis benchmark
python main.py --objective regression --dataset davis --batch_size 128 --max_epochs 200 --learning_rate 0.001

# PDBbind v2016
python main.py --objective regression --dataset pdb --batch_size 128 --max_epochs 200 --learning_rate 0.001
```

### DTI Classification
```bash
# BindingDB benchmark
python main.py --objective classification --dataset bindingdb --batch_size 64 --max_epochs 500 --learning_rate 0.001
```

### Output Format

**Regression** results are saved to `result_davis.csv` / `result_pdb.csv`:
```
<epoch>,<rmse>,<mae>,<mse>,<pearson>,<spearman>,<ci>
```

**Classification** results are saved to `result_bindingdb_{seed}.csv`:
```
<epoch>,<accuracy>,<aupr>,<auroc>,<f1>
```

---

## Benchmarks

| Dataset | Task | # Train Pairs | # Test Pairs | Unique Drugs | Unique Proteins |
|:---|:---|:---:|:---:|:---:|:---:|
| Davis | DTA Regression | 24,046 | 3,005 | 68 | 442 |
| PDBbind v2016 | DTA Regression | 7,512 | 207 | 6,527 | 5,352 |
| TDC-DG | DTA (Temporal) | 82,096 | 48,992 | 94,742 | 554 |
| BindingDB | DTI Classification | 12,657 | 13,272 | 7,108 | 2,000 |

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{thati2026bissa,
  title={Bidirectional Sequence--Structure Attention for Drug--Target Interaction and Affinity Prediction},
  author={Thati, Sai Koushik Chowdary and Javvadi, Charan Subhash},
  year={2026},
  institution={Indian Institute of Information Technology, Sri City}
}
```

---

## Acknowledgments

This work builds upon the [PMMR](https://github.com/) framework and extends it with bidirectional cross-attention decoders and AlphaFold-derived protein contact graphs. We gratefully acknowledge the developers of ESM-2, ChemBERTa-2, AlphaFold, and RDKit.
