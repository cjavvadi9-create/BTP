"""
Lightweight CPU-only dummy-tensor smoke test for the refactored PMMRNet.
batch_size=2, sequence/node lengths between 4 and 8.
Runs in under 1 second on any laptop — no data files needed.
"""
import sys, os
import argparse
import torch

# Ensure the project root is on sys.path so `models` can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.core import PMMRNet

def make_dummy_args():
    """Return a namespace that mirrors parse_args() defaults."""
    return argparse.Namespace(
        compound_gnn_dim=78,
        dropout=0.2,
        decoder_dim=128,
        decoder_heads=4,
        decoder_layers=2,
        compound_text_dim=128,
        compound_structure_dim=78,
        protein_dim=128,
        linear_heads=10,
        linear_hidden_dim=32,
        pf_dim=1024,
        encoder_heads=4,
        encoder_layers=1,
        protein_pretrained_dim=480,
        compound_pretrained_dim=384,
        protein_structure_dim=22,
        objective='regression',        # will test both
    )


def build_dummy_batch(B=2,
                      L_smiles=6, L_atoms=5,
                      L_seq=8, L_struct=4,
                      compound_feat_dim=78, compound_emb_dim=384,
                      protein_feat_dim=22, protein_emb_dim=480):
    """Synthesise a minimal data dict that mimics a real collated batch."""
    data = {
        # Drug
        'COMPOUND_NODE_FEAT': torch.randn(B, L_atoms, compound_feat_dim),
        'COMPOUND_ADJ':       torch.eye(L_atoms).unsqueeze(0).expand(B, -1, -1),
        'COMPOUND_EMBEDDING': torch.randn(B, L_smiles, compound_emb_dim),
        'COMPOUND_NODE_NUM':  torch.tensor([L_atoms] * B),
        'COMPOUND_SMILES_LENGTH': torch.tensor([L_smiles] * B),

        # Protein
        'PROTEIN_NODE_FEAT':  torch.randn(B, L_struct, protein_feat_dim),
        'PROTEIN_ADJ':        torch.eye(L_struct).unsqueeze(0).expand(B, -1, -1),
        'PROTEIN_EMBEDDING':  torch.randn(B, L_seq, protein_emb_dim),
        'PROTEIN_NODE_NUM':   torch.tensor([L_seq] * B),
        'PROTEIN_STRUCT_NODE_NUM': torch.tensor([L_struct] * B),

        # Label
        'LABEL': torch.tensor([0.5, 1.2]),
    }
    return data


def test_regression():
    print("=" * 60)
    print("TEST: Regression (DTA) forward pass")
    print("=" * 60)
    args = make_dummy_args()
    args.objective = 'regression'
    model = PMMRNet(args)
    model.eval()

    data = build_dummy_batch()
    with torch.no_grad():
        out = model(data)

    assert out.shape == (2, 1), f"Expected (2,1), got {out.shape}"
    print(f"  Output shape: {out.shape}  OK")
    print(f"  Output values: {out.squeeze().tolist()}")
    print()


def test_classification():
    print("=" * 60)
    print("TEST: Classification (DTI) forward pass")
    print("=" * 60)
    args = make_dummy_args()
    args.objective = 'classification'
    model = PMMRNet(args)
    model.eval()

    data = build_dummy_batch()
    data['LABEL'] = torch.tensor([0, 1])  # class labels

    with torch.no_grad():
        out = model(data)

    assert out.shape == (2, 2), f"Expected (2,2), got {out.shape}"
    print(f"  Output shape: {out.shape}  OK")
    print(f"  Output values: {out.tolist()}")
    print()


def test_asymmetric_lengths():
    """Verify that the bidirectional paths handle different seq/struct lengths
    within the same batch (via padding and masking)."""
    print("=" * 60)
    print("TEST: Asymmetric lengths in same batch")
    print("=" * 60)
    args = make_dummy_args()
    args.objective = 'regression'
    model = PMMRNet(args)
    model.eval()

    # Pad to max length but set per-sample lengths differently
    data = build_dummy_batch(B=2, L_smiles=7, L_atoms=5, L_seq=8, L_struct=6)
    data['COMPOUND_SMILES_LENGTH'] = torch.tensor([5, 7])   # sample 0 shorter
    data['COMPOUND_NODE_NUM']      = torch.tensor([4, 5])
    data['PROTEIN_NODE_NUM']       = torch.tensor([6, 8])
    data['PROTEIN_STRUCT_NODE_NUM']= torch.tensor([4, 6])

    with torch.no_grad():
        out = model(data)

    assert out.shape == (2, 1), f"Expected (2,1), got {out.shape}"
    print(f"  Output shape: {out.shape}  OK")
    print(f"  Output values: {out.squeeze().tolist()}")
    print()


def test_gradient_flow():
    """Quick backward pass to ensure gradients flow through all 4 decoder paths."""
    print("=" * 60)
    print("TEST: Gradient flow through all decoder paths")
    print("=" * 60)
    args = make_dummy_args()
    args.objective = 'regression'
    model = PMMRNet(args)
    model.train()

    data = build_dummy_batch()
    out = model(data)
    loss = out.sum()
    loss.backward()

    # Spot-check a few key parameters for non-None gradients
    checks = [
        ('drug_seq_cross_atten[0].ea.w_q.weight',  model.drug_seq_cross_atten[0].ea.w_q.weight),
        ('drug_graph_cross_atten[0].ea.w_q.weight', model.drug_graph_cross_atten[0].ea.w_q.weight),
        ('protein_seq_cross_atten[0].ea.w_q.weight', model.protein_seq_cross_atten[0].ea.w_q.weight),
        ('protein_graph_cross_atten[0].ea.w_q.weight', model.protein_graph_cross_atten[0].ea.w_q.weight),
        ('drug_seq_attn.linear_first.weight',  model.drug_seq_attn.linear_first.weight),
        ('drug_graph_attn.linear_first.weight', model.drug_graph_attn.linear_first.weight),
        ('protein_seq_attn.linear_first.weight', model.protein_seq_attn.linear_first.weight),
        ('protein_graph_attn.linear_first.weight', model.protein_graph_attn.linear_first.weight),
    ]
    for name, param in checks:
        assert param.grad is not None, f"No gradient for {name}"
        assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"
        print(f"  {name}: grad norm = {param.grad.norm().item():.6f}  OK")

    print()


if __name__ == '__main__':
    test_regression()
    test_classification()
    test_asymmetric_lengths()
    test_gradient_flow()
    print("=" * 60)
    print("ALL TESTS PASSED OK")
    print("=" * 60)
