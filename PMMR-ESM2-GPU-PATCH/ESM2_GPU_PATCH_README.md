# BiSSA ESM-2 GPU Completion Package

This package contains the **5,610 ESM-2 layer-12 embeddings** generated with the official `esm2_t12_35M_UR50D` model. It is designed to be merged into the `BiSSA-main` archive that contains the source code, `mol_dict.pkl`, and AlphaFold structure arrays.

## Merge

1. Extract the `BiSSA-main` archive on the GPU machine.
2. Extract this patch ZIP over the same project directory.
3. The files in this package belong under:

```text
data/pdb/protein/<target_id>.npy
```

4. Use the cleaned benchmark CSVs in `benchmark_data/` because the original raw CSVs contain malformed SMILES records.
5. Run from the project root:

```bash
python3 main.py \
  --root_data_path ./benchmark_data \
  --dataset pdb \
  --objective regression \
  --batch_size 128 \
  --max_epochs 200 \
  --num_workers 4 \
  --learning_rate 0.001
```

For a scientifically stronger evaluation, reserve a validation set instead of selecting the best checkpoint directly on the test set; the standard training loop evaluates the test loader every epoch for non-Davis datasets.

## Asset Verification

After merging, verify that the project has 7,719 target-level protein embeddings and 7,719 structure node/adjacency pairs. The ESM embeddings have width 480 and are truncated to 1,022 residues, matching the paper and preprocessing code.

## Architecture Alignment

The run uses the BiSSA architecture described in the paper: RDKit molecular graph plus ChemBERTa compound sequence; ESM-2 protein sequence plus AlphaFold residue contact graph; dual graph encoders, Transformer encoders, bidirectional cross-attention decoders (S ↔ G), four linear-attention pooling readouts, concatenation into a 512-dimensional joint representation vector, and the task-specific prediction head.
