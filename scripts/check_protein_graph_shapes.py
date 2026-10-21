"""
scripts/check_protein_graph_shapes.py
--------------------------------------
Smoke-test for the full PMMR feature pipeline.

Run from the repo root:
    python scripts/check_protein_graph_shapes.py [--dataset pdb]

Checks performed
  1. CPIDataset single-item shapes for all keys
  2. collate_fn batch shapes for all keys
  3. PMMRNet forward() on that batch (catches wiring / dimension mismatches)

Exit code 0 = all shapes look correct and the forward pass completed.
Exit code 1 = a shape mismatch or runtime error was caught.
"""

import sys
import os
import traceback

# ── repo root on the path so imports match main.py ─────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import argparse
import torch
from torch.utils.data import DataLoader

from data import CPIDataset
from models.core import PMMRNet

# ── helpers ─────────────────────────────────────────────────────────────────

SEP  = "-" * 60
PASS = "[PASS]"
FAIL = "[FAIL]"


def section(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


def check_key(label: str, tensor, expected_ndim: int | None = None) -> bool:
    """Print shape and optionally validate number of dimensions."""
    ok = True
    if expected_ndim is not None and tensor.ndim != expected_ndim:
        print(f"  {FAIL}  {label:40s} shape={tuple(tensor.shape)}  "
              f"[expected {expected_ndim}D, got {tensor.ndim}D]")
        ok = False
    else:
        print(f"  {PASS}  {label:40s} shape={tuple(tensor.shape)}")
    return ok


def build_args(dataset='pdb') -> argparse.Namespace:
    """Return a Namespace with the same defaults as main.py's parse_args()."""
    args = argparse.Namespace(
        # architecture
        compound_gnn_dim       = 78,
        compound_structure_dim = 78,
        protein_structure_dim  = 22,
        compound_text_dim      = 128,
        compound_pretrained_dim= 384,
        protein_pretrained_dim = 480,
        protein_dim            = 128,
        decoder_dim            = 128,
        decoder_heads          = 4,
        decoder_layers         = 2,
        encoder_heads          = 4,
        encoder_layers         = 1,
        pf_dim                 = 1024,
        linear_heads           = 10,
        linear_hidden_dim      = 32,
        dropout                = 0.2,
        # task
        objective              = 'regression',
    )
    return args


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="PMMR pipeline smoke test")
    parser.add_argument('--dataset', type=str, default='pdb',
                        help='Dataset to test (pdb, davis, bindingdb, etc.)')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Data root directory (default: <repo>/data)')
    cli_args = parser.parse_args()

    dataset = cli_args.dataset
    data_root = cli_args.data_root or os.path.join(REPO_ROOT, 'data')

    # Resolve paths
    path = os.path.join(data_root, dataset)
    if dataset == 'davis':
        csv_file = os.path.join(data_root, f'{dataset}.csv')
    else:
        csv_file = os.path.join(data_root, f'{dataset}_train.csv')
    compound_dir = os.path.join(path, 'compound')
    protein_dir = os.path.join(path, 'protein')

    print(f"Dataset:      {dataset}")
    print(f"CSV:          {csv_file}")
    print(f"Compound dir: {compound_dir}")
    print(f"Protein dir:  {protein_dir}")

    all_ok = True

    # ------------------------------------------------------------------
    # 0. Check prerequisites
    # ------------------------------------------------------------------
    section("0 · Prerequisite checks")
    prereqs_ok = True
    if not os.path.isfile(csv_file):
        print(f"  {FAIL}  CSV file missing: {csv_file}")
        prereqs_ok = False
    else:
        print(f"  {PASS}  CSV file exists")

    mol_dict = os.path.join(compound_dir, 'mol_dict.pkl')
    if not os.path.isfile(mol_dict):
        print(f"  {FAIL}  ChemBERTa embeddings missing: {mol_dict}")
        prereqs_ok = False
    else:
        print(f"  {PASS}  ChemBERTa embeddings exist")

    if not os.path.isdir(protein_dir):
        print(f"  {FAIL}  Protein dir missing: {protein_dir}")
        prereqs_ok = False
    else:
        npy_count = len([f for f in os.listdir(protein_dir) if f.endswith('.npy')])
        print(f"  {PASS}  Protein dir exists ({npy_count} .npy files)")

    struct_dir = os.path.join(protein_dir, 'structure')
    if not os.path.isdir(struct_dir):
        print(f"  {FAIL}  Structure dir missing: {struct_dir}")
        prereqs_ok = False
    else:
        node_count = len([f for f in os.listdir(struct_dir) if f.endswith('_node.npy')])
        print(f"  {PASS}  Structure dir exists ({node_count} node files)")

    if not prereqs_ok:
        print(f"\n  {FAIL}  Prerequisites not met. Run preprocessing first.")
        return 1

    # ------------------------------------------------------------------
    # 1. Dataset instantiation
    # ------------------------------------------------------------------
    section("1 · CPIDataset instantiation")
    try:
        dataset_obj = CPIDataset(csv_file, compound_dir, protein_dir)
        print(f"  {PASS}  Loaded {len(dataset_obj)} samples from {csv_file}")
    except Exception as exc:
        print(f"  {FAIL}  CPIDataset failed: {exc}")
        traceback.print_exc()
        return 1

    # ------------------------------------------------------------------
    # 2. Single-item shapes
    # ------------------------------------------------------------------
    section("2 · Single-item shapes  (dataset[0])")
    try:
        item = dataset_obj[0]
    except Exception as exc:
        print(f"  {FAIL}  dataset[0] raised: {exc}")
        traceback.print_exc()
        return 1

    import numpy as np

    def to_tensor(v):
        """Accept numpy arrays or tensors."""
        if isinstance(v, torch.Tensor):
            return v
        return torch.from_numpy(np.asarray(v))

    all_ok &= check_key("PROTEIN_EMBEDDING",  to_tensor(item['PROTEIN_EMBEDDING']),  expected_ndim=2)
    all_ok &= check_key("PROTEIN_NODE_FEAT",  to_tensor(item['PROTEIN_NODE_FEAT']),  expected_ndim=2)
    all_ok &= check_key("PROTEIN_ADJ",        to_tensor(item['PROTEIN_ADJ']),        expected_ndim=2)
    all_ok &= check_key("COMPOUND_NODE_FEAT", to_tensor(item['COMPOUND_NODE_FEAT']), expected_ndim=2)
    all_ok &= check_key("COMPOUND_ADJ",       to_tensor(item['COMPOUND_ADJ']),       expected_ndim=2)
    all_ok &= check_key("COMPOUND_EMBEDDING", to_tensor(item['COMPOUND_EMBEDDING']), expected_ndim=2)

    # Sanity: PROTEIN_NODE_FEAT last dim should be 22
    pnf = to_tensor(item['PROTEIN_NODE_FEAT'])
    if pnf.shape[-1] != 22:
        print(f"  {FAIL}  PROTEIN_NODE_FEAT feature dim = {pnf.shape[-1]}, expected 22")
        all_ok = False
    else:
        print(f"  {PASS}  PROTEIN_NODE_FEAT feature dim = 22")

    # Sanity: PROTEIN_ADJ should be square and match residue count
    padj = to_tensor(item['PROTEIN_ADJ'])
    if padj.shape[0] != padj.shape[1]:
        print(f"  {FAIL}  PROTEIN_ADJ is not square: {tuple(padj.shape)}")
        all_ok = False
    elif padj.shape[0] != pnf.shape[0]:
        print(f"  {FAIL}  PROTEIN_ADJ size {padj.shape[0]} != "
              f"PROTEIN_NODE_FEAT length {pnf.shape[0]}")
        all_ok = False
    else:
        print(f"  {PASS}  PROTEIN_ADJ is square and matches residue count")

    # ------------------------------------------------------------------
    # 3. Batch shapes via collate_fn
    # ------------------------------------------------------------------
    section("3 · Batch shapes  (batch_size=2)")
    try:
        loader = DataLoader(
            dataset_obj,
            batch_size=2,
            collate_fn=dataset_obj.collate_fn,
            shuffle=False,
            num_workers=0,       # 0 = in-process, avoids multiprocessing issues
        )
        batch = next(iter(loader))
    except Exception as exc:
        print(f"  {FAIL}  collate_fn / DataLoader raised: {exc}")
        traceback.print_exc()
        return 1

    # Protein structure branch
    all_ok &= check_key("PROTEIN_NODE_FEAT  [B, L_struct, 22]",
                         batch['PROTEIN_NODE_FEAT'], expected_ndim=3)
    all_ok &= check_key("PROTEIN_ADJ        [B, L_struct, L_struct]",
                         batch['PROTEIN_ADJ'],       expected_ndim=3)
    all_ok &= check_key("PROTEIN_STRUCT_NODE_NUM [B]",
                         batch['PROTEIN_STRUCT_NODE_NUM'], expected_ndim=1)
    # Protein sequence branch (unchanged)
    all_ok &= check_key("PROTEIN_EMBEDDING  [B, L_seq, 480]",
                         batch['PROTEIN_EMBEDDING'], expected_ndim=3)
    all_ok &= check_key("PROTEIN_NODE_NUM   [B]",
                         batch['PROTEIN_NODE_NUM'],  expected_ndim=1)
    # Compound branch
    all_ok &= check_key("COMPOUND_NODE_FEAT [B, L_atom, 78]",
                         batch['COMPOUND_NODE_FEAT'], expected_ndim=3)
    all_ok &= check_key("COMPOUND_ADJ       [B, L_atom, L_atom]",
                         batch['COMPOUND_ADJ'],       expected_ndim=3)
    all_ok &= check_key("COMPOUND_EMBEDDING [B, L_smiles, 384]",
                         batch['COMPOUND_EMBEDDING'], expected_ndim=3)

    # Dimension consistency checks
    B = batch['PROTEIN_NODE_FEAT'].shape[0]
    L_struct = batch['PROTEIN_NODE_FEAT'].shape[1]
    adj_L    = batch['PROTEIN_ADJ'].shape[1]
    adj_L2   = batch['PROTEIN_ADJ'].shape[2]

    if batch['PROTEIN_NODE_FEAT'].shape[-1] != 22:
        print(f"  {FAIL}  PROTEIN_NODE_FEAT batch feature dim = "
              f"{batch['PROTEIN_NODE_FEAT'].shape[-1]}, expected 22")
        all_ok = False
    else:
        print(f"  {PASS}  PROTEIN_NODE_FEAT batch feature dim = 22")

    if adj_L != L_struct or adj_L2 != L_struct:
        print(f"  {FAIL}  PROTEIN_ADJ [{B},{adj_L},{adj_L2}] not square "
              f"or does not match L_struct={L_struct}")
        all_ok = False
    else:
        print(f"  {PASS}  PROTEIN_ADJ is [B, L_struct, L_struct]")

    # ------------------------------------------------------------------
    # 4. PMMRNet forward pass
    # ------------------------------------------------------------------
    section("4 · PMMRNet forward pass")

    if not torch.cuda.is_available():
        print(f"  ⚠  CUDA not available — skipping forward pass "
              f"(model uses .cuda() internally)")
        print(f"\n{'─'*60}")
        print("  Overall:", f"{PASS} all shape checks passed" if all_ok
                            else f"{FAIL} some shape checks FAILED")
        return 0 if all_ok else 1

    try:
        args  = build_args()
        model = PMMRNet(args).cuda()
        model.eval()
        print(f"  {PASS}  PMMRNet instantiated")
    except Exception as exc:
        print(f"  {FAIL}  PMMRNet init failed: {exc}")
        traceback.print_exc()
        return 1

    try:
        with torch.no_grad():
            output = model(batch)
        all_ok &= check_key("model output", output, expected_ndim=2)
        print(f"  {PASS}  Forward pass completed without errors")
    except RuntimeError as exc:
        # Pretty-print dimension mismatch errors — the most common wiring bug
        msg = str(exc)
        print(f"\n  {FAIL}  RuntimeError during forward():\n")
        for line in msg.splitlines():
            print(f"         {line}")
        print()
        traceback.print_exc()
        all_ok = False
    except Exception as exc:
        print(f"  {FAIL}  Unexpected error during forward(): {exc}")
        traceback.print_exc()
        all_ok = False

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{SEP}")
    if all_ok:
        print(f"  {PASS}  All checks passed — pipeline wiring looks correct.")
    else:
        print(f"  {FAIL}  One or more checks FAILED — see details above.")
    print(SEP)

    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
