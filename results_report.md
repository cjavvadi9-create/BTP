---
## BiSSA — Full Experimental Results & Benchmark Report

> **Paper:** *Bidirectional Sequence–Structure Attention for Drug–Target Interaction and Affinity Prediction*
> **Authors:** Sai Koushik Chowdary Thati, Charan Subhash Javvadi — IIIT Sri City

---

## 1. Executive Summary

## 1. Benchmark Datasets & Preprocessing Statistics

### Key Results
1. **Davis DTA Regression (Random Split):** BiSSA achieves **MSE 0.181**, **RMSE 0.426**, **CI 0.912**, outperforming DeepDTA (MSE 0.261), GraphDTA (MSE 0.229), and the unidirectional baseline (MSE 0.195).
2. **PDBbind v2016:** RMSE decreases to **1.228** with Pearson *r* = **0.811** (vs. GraphDTA RMSE 1.362).
3. **TDC-DG (Temporal OOD Split):** CI improves to **0.754**, MSE drops to **1.680**, demonstrating temporal generalization.
4. **BindingDB DTI Classification:** **0.963 AUROC**, **0.956 AUPR**, **0.902 F1**, **0.908 Accuracy**.
5. **Cold-Target Generalization:** On Davis Cold-Target splits, BiSSA achieves **CI = 0.848** (+2.2% over unidirectional, +6.3% over GraphDTA).

---

## 2. Benchmark Datasets & Preprocessing Statistics

| Dataset | Learning Task | Target Metric | Train Pairs | Val / Test Pairs | Unique Drugs | Unique Proteins | Structure Source |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Davis** | DTA Regression | $\text{pK}_d$ | 24,046 | 3,005 / 3,005 | 68 | 442 | AlphaFold v4 ($\tau = 8.0\text{ Å}$) |
| **PDBbind v2016** | DTA Regression | $-\log(K_d/K_i/\text{IC}_{50})$ | 7,512 | 207 (Test) | 6,527 | 5,352 | RCSB PDB Experimental Coordinates |
| **TDC-DG** | DTA Domain Gen. | Temporal Patent Split | 82,096 | 48,992 (Test) | 94,742 | 554 | AlphaFold v4 ($\tau = 8.0\text{ Å}$) |
| **BindingDB** | DTI Classification | Binary $y \in \{0, 1\}$ | 12,657 | 13,272 (Test) | 7,108 | 2,000 | AlphaFold v4 ($\tau = 8.0\text{ Å}$) |

---

## 2. Experimental Performance Tables

### Table 2: DTA Regression Performance on Standard and Temporal Benchmarks

| Dataset | Model | RMSE $\downarrow$ | MAE $\downarrow$ | MSE $\downarrow$ | Pearson ($r$) $\uparrow$ | Spearman ($\rho$) $\uparrow$ | CI $\uparrow$ | $r_m^2 \uparrow$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Davis** (Random 80/10/10) | DeepDTA | 0.511 | 0.342 | 0.261 | 0.806 | 0.775 | 0.878 | 0.630 |
| | GraphDTA | 0.478 | 0.318 | 0.229 | 0.824 | 0.794 | 0.893 | 0.685 |
| | DGraphDTA | 0.452 | 0.301 | 0.204 | 0.838 | 0.809 | 0.904 | 0.700 |
| | MGraphDTA | 0.455 | 0.298 | 0.207 | 0.841 | 0.812 | 0.900 | 0.710 |
| | PMMR Unidirectional ($S \leftarrow G$) | 0.442 | 0.294 | 0.195 | 0.848 | 0.819 | 0.905 | 0.722 |
| | **BiSSA (Ours)** | **0.426** | **0.278** | **0.181** | **0.862** | **0.836** | **0.912** | **0.741** |
| **PDBbind v2016** | GraphDTA | 1.362 | 1.084 | 1.855 | 0.754 | 0.741 | 0.782 | — |
| | DGraphDTA | 1.314 | 1.042 | 1.727 | 0.772 | 0.758 | 0.794 | — |
| | PMMR Unidirectional ($S \leftarrow G$) | 1.284 | 1.018 | 1.649 | 0.788 | 0.774 | 0.804 | — |
| | **BiSSA (Ours)** | **1.228** | **0.962** | **1.508** | **0.811** | **0.798** | **0.821** | — |
| **TDC-DG** (Temporal) | GraphDTA | 1.425 | 1.118 | 2.031 | 0.552 | 0.538 | 0.708 | — |
| | PMMR Unidirectional ($S \leftarrow G$) | 1.358 | 1.054 | 1.844 | 0.581 | 0.566 | 0.729 | — |
| | **BiSSA (Ours)** | **1.296** | **0.994** | **1.680** | **0.618** | **0.602** | **0.754** | — |

---

### Table 3: Out-of-Distribution Generalization on Davis Cold-Start Protocols

| Protocol | Model | RMSE $\downarrow$ | MAE $\downarrow$ | MSE $\downarrow$ | Pearson ($r$) $\uparrow$ | Spearman ($\rho$) $\uparrow$ | CI $\uparrow$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cold Drug Split** | GraphDTA | 0.802 | 0.612 | 0.643 | 0.482 | 0.451 | 0.728 |
| | PMMR Unidirectional ($S \leftarrow G$) | 0.724 | 0.544 | 0.524 | 0.552 | 0.518 | 0.766 |
| | **BiSSA (Ours)** | **0.686** | **0.511** | **0.471** | **0.594** | **0.561** | **0.788** |
| **Cold Target Split** | GraphDTA | 0.675 | 0.498 | 0.456 | 0.624 | 0.592 | 0.785 |
| | PMMR Unidirectional ($S \leftarrow G$) | 0.614 | 0.446 | 0.377 | 0.698 | 0.664 | 0.826 |
| | **BiSSA (Ours)** | **0.581** | **0.414** | **0.338** | **0.738** | **0.703** | **0.848** |
| **Cold Pair (Both) Split** | GraphDTA | 0.895 | 0.710 | 0.801 | 0.334 | 0.308 | 0.645 |
| | PMMR Unidirectional ($S \leftarrow G$) | 0.828 | 0.654 | 0.686 | 0.402 | 0.372 | 0.694 |
| | **BiSSA (Ours)** | **0.789** | **0.618** | **0.623** | **0.456** | **0.424** | **0.725** |

---

### Table 4: DTI Binary Classification on BindingDB

| Model | Accuracy $\uparrow$ | AUPR $\uparrow$ | AUROC $\uparrow$ | F1-Score $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: |
| MolTrans | 0.852 | 0.898 | 0.915 | 0.840 |
| HyperAttentionDTI | 0.868 | 0.915 | 0.931 | 0.858 |
| DrugBAN | 0.882 | 0.932 | 0.946 | 0.874 |
| PMMR Unidirectional ($S \leftarrow G$) | 0.893 | 0.942 | 0.951 | 0.886 |
| **BiSSA (Ours)** | **0.908** | **0.956** | **0.963** | **0.902** |

---

### Table 5: Architectural Ablation Study (Davis & BindingDB)

| Variant | Description | Drug Graph | Prot Graph | PLMs | Attn. Direction | Davis MSE (Rand) $\downarrow$ | Davis CI (Rand) $\uparrow$ | Davis CI (Cold Prot) $\uparrow$ | BindingDB AUROC $\uparrow$ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A1** | Graph-Only (GCNs) | $\checkmark$ | $\checkmark$ | $\times$ | None | 0.236 | 0.886 | 0.774 | 0.912 |
| **A2** | Sequence-Only (PLMs) | $\times$ | $\times$ | $\checkmark$ | None | 0.218 | 0.895 | 0.804 | 0.934 |
| **A3** | w/o AlphaFold Graph | $\checkmark$ | $\times$ | $\checkmark$ | $S_d \leftrightarrow G_d$ | 0.203 | 0.901 | 0.814 | 0.944 |
| **A4** | w/o Cross-Attention | $\checkmark$ | $\checkmark$ | $\checkmark$ | Concat | 0.199 | 0.903 | 0.820 | 0.947 |
| **A5** | Unidirectional ($S \leftarrow G$) | $\checkmark$ | $\checkmark$ | $\checkmark$ | $S \leftarrow G$ | 0.195 | 0.905 | 0.826 | 0.951 |
| **A6** | Unidirectional ($G \leftarrow S$) | $\checkmark$ | $\checkmark$ | $\checkmark$ | $G \leftarrow S$ | 0.197 | 0.904 | 0.822 | 0.949 |
| **Full** | **BiSSA (Ours)** | $\checkmark$ | $\checkmark$ | $\checkmark$ | **$S \leftrightarrow G$** | **0.181** | **0.912** | **0.848** | **0.963** |

---

## 4. Output Format Specification

When running the model using `main.py`, the training pipeline writes evaluation results to CSV files:

### A. DTA Regression Output (`result_davis.csv`, `result_pdb.csv`)
Each row corresponds to an evaluation epoch:
```
<epoch>,<rmse>,<mae>,<mse>,<pearson>,<spearman>,<ci>
```
*Paper-reported best epoch on Davis:*
```
199,0.426,0.278,0.181,0.862,0.836,0.912
```

### B. DTI Classification Output (`result_bindingdb_{seed}.csv`)
Each row corresponds to an evaluation epoch:
```
<epoch>,<acc>,<aupr>,<auroc>,<f1>
```
*Paper-reported best epoch on BindingDB:*
```
499,0.908,0.956,0.963,0.902
```
