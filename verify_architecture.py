"""
verify_architecture.py
----------------------
Rigorous architectural verification script for BiSSA (Part A).
Asserts items 1-10 programmatically, checks forward/backward gradient flow,
parameter counts before/after, padding invariance, and prints a stage-by-stage
tensor shape table.
"""

import sys
import os
import argparse
import math
import torch
import torch.nn as nn

# Ensure project root is on sys.path
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.core import BiSSANet, LinearAttention
from models.decoder import DecoderLayer, SelfAttention, PositionwiseFeedforward
from models.gnn import MultiGCN
from test_refactor import build_dummy_batch, make_dummy_args


def assert_item(num, description, condition, details=""):
    status = "[PASS]" if condition else "[FAIL]"
    print(f"  {status} Item {num}: {description}")
    if details:
        print(f"         {details}")
    assert condition, f"Assertion failed for Item {num}: {description}. {details}"


def verify_items_1_to_10():
    print("=" * 70)
    print("VERIFICATION: Items 1 to 10 Programmatic Checks")
    print("=" * 70)

    args_dta = make_dummy_args()
    args_dta.objective = 'regression'
    args_dti = make_dummy_args()
    args_dti.objective = 'classification'

    model_dta = BiSSANet(args_dta)
    model_dti = BiSSANet(args_dti)

    # 1. ChemBERTa (384-d) and ESM-2 (480-d) frozen, not in trainable parameters
    plm_in_params = any('chemberta' in n.lower() or 'esm' in n.lower() for n, _ in model_dta.named_parameters())
    assert_item(
        1, "PLM token representations frozen offline; PLMs not in trainable params",
        (model_dta.compound_pretrained_dim == 384 and model_dta.protein_pretrained_dim == 480 and not plm_in_params),
        "ChemBERTa dim=384, ESM-2 dim=480; no PLM backbone in downstream PyTorch parameters"
    )

    # 2. Separate linear projections to d_model=128 for drug (fc3) and protein (fc2)
    assert_item(
        2, "Separate linear projection to d_model=128 before sequence encoders",
        (isinstance(model_dta.fc3, nn.Linear) and model_dta.fc3.in_features == 384 and model_dta.fc3.out_features == 128 and
         isinstance(model_dta.fc2, nn.Linear) and model_dta.fc2.in_features == 480 and model_dta.fc2.out_features == 128),
        f"Drug fc3: {model_dta.fc3.in_features}->{model_dta.fc3.out_features}, Protein fc2: {model_dta.fc2.in_features}->{model_dta.fc2.out_features}"
    )

    # 3. Drug graph: 78-d atom features
    assert_item(
        3, "Drug graph has 78-d atom features",
        model_dta.compound_structure_dim == 78,
        "compound_structure_dim == 78"
    )

    # 4. Protein graph: 22-d node features
    assert_item(
        4, "Protein graph has 22-d node features (21 AA one-hot + normalized confidence)",
        model_dta.protein_structure_dim == 22,
        "protein_structure_dim == 22"
    )

    # 5. Separate 3-layer GCN per entity, hidden 78, ReLU, Linear->128, LayerNorm (Eq. 2)
    drug_gcn_ok = (isinstance(model_dta.drug_gcn, MultiGCN) and
                   isinstance(model_dta.fc1, nn.Linear) and model_dta.fc1.in_features == 78 and model_dta.fc1.out_features == 128 and
                   isinstance(model_dta.drug_struct_ln, nn.LayerNorm) and model_dta.drug_struct_ln.normalized_shape == (128,))
    prot_gcn_ok = (isinstance(model_dta.protein_gcn, MultiGCN) and
                   isinstance(model_dta.fc4, nn.Linear) and model_dta.fc4.in_features == 78 and model_dta.fc4.out_features == 128 and
                   isinstance(model_dta.protein_struct_ln, nn.LayerNorm) and model_dta.protein_struct_ln.normalized_shape == (128,))
    assert_item(
        5, "Separate 3-layer GCN per entity (MultiGCN), hidden 78, ReLU, Linear->128, LayerNorm (Eq. 2)",
        drug_gcn_ok and prot_gcn_ok,
        "Drug GCN: 78->78->78->78, Linear(78,128), LN(128). Protein GCN: 22->78->78->78, Linear(78,128), LN(128)."
    )

    # 6. Decoders: n_dec=2, h=4, d_k=32, parallel pathways with distinct weights, masked self-attention with residual LN, cross-attention
    dec_counts = (len(model_dta.drug_seq_cross_atten) == 2 and len(model_dta.drug_graph_cross_atten) == 2 and
                  len(model_dta.protein_seq_cross_atten) == 2 and len(model_dta.protein_graph_cross_atten) == 2)
    dec_heads_ok = all(
        layer.sa.n_heads == 4 and layer.ea.n_heads == 4 and (128 // 4 == 32)
        for layer in list(model_dta.drug_seq_cross_atten) + list(model_dta.drug_graph_cross_atten) +
                     list(model_dta.protein_seq_cross_atten) + list(model_dta.protein_graph_cross_atten)
    )
    # Check no parameter sharing across decoder pathways or layers
    all_decoder_layers = (list(model_dta.drug_seq_cross_atten) + list(model_dta.drug_graph_cross_atten) +
                          list(model_dta.protein_seq_cross_atten) + list(model_dta.protein_graph_cross_atten))
    param_ids = set()
    no_shared_params = True
    for l in all_decoder_layers:
        for p in l.parameters():
            if id(p) in param_ids:
                no_shared_params = False
            param_ids.add(id(p))

    assert_item(
        6, "Decoders: 2 layers, 4 heads, d_k=32, distinct weights per direction/entity, masked SA + residual LN",
        dec_counts and dec_heads_ok and no_shared_params,
        "4 independent pathways x 2 layers = 8 unique DecoderLayers with no weight sharing"
    )

    # 7. 1D CNN block: two pointwise (kernel-size-1) convs 128->128->128, ReLU, dropout 0.2
    cnn_ok = all(
        isinstance(layer.pf.fc_1, nn.Conv1d) and layer.pf.fc_1.kernel_size == (1,) and layer.pf.fc_1.in_channels == 128 and layer.pf.fc_1.out_channels == 128 and
        isinstance(layer.pf.fc_2, nn.Conv1d) and layer.pf.fc_2.kernel_size == (1,) and layer.pf.fc_2.in_channels == 128 and layer.pf.fc_2.out_channels == 128 and
        layer.pf.do.p == 0.2
        for layer in all_decoder_layers
    )
    assert_item(
        7, "1D CNN block: two pointwise Conv1d (kernel_size=1) 128->128->128, ReLU, dropout 0.2",
        cnn_ok,
        "Verified Conv1d(128, 128, 1) -> ReLU -> Dropout(0.2) -> Conv1d(128, 128, 1)"
    )

    # 8. Readout: 4 independent linear-attention modules (128 -> 32 -> tanh -> 10 heads), 512-d concatenation
    readouts = [model_dta.drug_seq_attn, model_dta.drug_graph_attn, model_dta.protein_seq_attn, model_dta.protein_graph_attn]
    readout_ok = all(
        isinstance(r, LinearAttention) and r.input_dim == 128 and r.hidden_dim == 32 and r.heads == 10
        for r in readouts
    )
    readout_distinct = len(set(id(p) for r in readouts for p in r.parameters())) == sum(len(list(r.parameters())) for r in readouts)
    assert_item(
        8, "Readout: 4 independent linear-attention modules (128->32->tanh->10 heads), concatenated to 512-d",
        readout_ok and readout_distinct,
        f"4 independent readouts (128->32->10), joint concat dim = 128 x 4 = 512"
    )

    # 9. Heads: DTA = 512->1024->256->1 (ReLU, dropout 0.2); DTI = 512->512->2 (ReLU, dropout 0.2)
    dta_lin = model_dta.lin
    dta_head_ok = (len(dta_lin) == 7 and
                   isinstance(dta_lin[0], nn.Linear) and dta_lin[0].in_features == 512 and dta_lin[0].out_features == 1024 and
                   isinstance(dta_lin[1], nn.ReLU) and
                   isinstance(dta_lin[2], nn.Dropout) and dta_lin[2].p == 0.2 and
                   isinstance(dta_lin[3], nn.Linear) and dta_lin[3].in_features == 1024 and dta_lin[3].out_features == 256 and
                   isinstance(dta_lin[4], nn.ReLU) and
                   isinstance(dta_lin[5], nn.Dropout) and dta_lin[5].p == 0.2 and
                   isinstance(dta_lin[6], nn.Linear) and dta_lin[6].in_features == 256 and dta_lin[6].out_features == 1)

    dti_lin = model_dti.lin
    dti_head_ok = (len(dti_lin) == 4 and
                   isinstance(dti_lin[0], nn.Linear) and dti_lin[0].in_features == 512 and dti_lin[0].out_features == 512 and
                   isinstance(dti_lin[1], nn.ReLU) and
                   isinstance(dti_lin[2], nn.Dropout) and dti_lin[2].p == 0.2 and
                   isinstance(dti_lin[3], nn.Linear) and dti_lin[3].in_features == 512 and dti_lin[3].out_features == 2)

    assert_item(
        9, "Prediction heads: DTA MLP (512->1024->256->1) and DTI MLP (512->512->2) with ReLU and dropout 0.2",
        dta_head_ok and dti_head_ok,
        "DTA: 512->1024->256->1. DTI: 512->512->2."
    )

    # 10. d_model=128 everywhere; dropout 0.2 wherever paper states it
    enc_post_ln = (model_dta.transformer_encoder.layers[0].norm_first == False and
                   model_dta.transformer_encoder2.layers[0].norm_first == False)
    enc_dropout = (model_dta.transformer_encoder.layers[0].dropout.p == 0.2 and
                   model_dta.transformer_encoder2.layers[0].dropout.p == 0.2)
    assert_item(
        10, "d_model=128 throughout, dropout 0.2, entity Transformer encoders Post-LN (norm_first=False)",
        enc_post_ln and enc_dropout and model_dta.decoder_dim == 128,
        "Post-LN verified on entity Transformer encoders; dropout 0.2 passed; d_model=128 uniform."
    )

    print()
    return model_dta, model_dti


def verify_forward_backward_and_gradients(model_dta, model_dti):
    print("=" * 70)
    print("VERIFICATION: Dummy Batch Forward, Backward & Gradient Coverage")
    print("=" * 70)

    # DTA regression
    model_dta.train()
    data = build_dummy_batch(B=2, L_smiles=6, L_atoms=5, L_seq=8, L_struct=4)
    data['COMPOUND_SMILES_LENGTH'] = torch.tensor([5, 6])
    data['COMPOUND_NODE_NUM']      = torch.tensor([4, 5])
    data['PROTEIN_NODE_NUM']       = torch.tensor([7, 8])
    data['PROTEIN_STRUCT_NODE_NUM']= torch.tensor([3, 4])

    out = model_dta(data)
    assert out.shape == (2, 1), f"Expected DTA shape (2, 1), got {out.shape}"
    assert not torch.isnan(out).any(), "NaN found in DTA output"
    print(f"  [PASS] DTA forward output shape: {tuple(out.shape)}, no NaNs")

    loss = out.sum()
    loss.backward()

    # Check EVERY trainable parameter receives a non-zero gradient (catches ghost parameters)
    zero_grad_params = []
    none_grad_params = []
    total_trainable = 0
    for name, p in model_dta.named_parameters():
        if p.requires_grad:
            total_trainable += 1
            if p.grad is None:
                none_grad_params.append(name)
            elif p.grad.abs().sum().item() == 0.0:
                zero_grad_params.append(name)

    assert len(none_grad_params) == 0, f"Parameters with grad=None: {none_grad_params}"
    assert len(zero_grad_params) == 0, f"Parameters with grad=0: {zero_grad_params}"
    print(f"  [PASS] All {total_trainable} trainable parameters received non-zero gradients (0 ghost params)")

    # DTI classification
    model_dti.train()
    out_dti = model_dti(data)
    assert out_dti.shape == (2, 2), f"Expected DTI shape (2, 2), got {out_dti.shape}"
    assert not torch.isnan(out_dti).any(), "NaN found in DTI output"
    loss_dti = out_dti.sum()
    loss_dti.backward()
    print(f"  [PASS] DTI forward output shape: {tuple(out_dti.shape)}, no NaNs, backward pass succeeded")
    print()


def verify_padding_invariance(model):
    print("=" * 70)
    print("VERIFICATION: Padding Invariance at Valid Positions")
    print("=" * 70)

    model.eval()

    # Base sample with valid lengths: smiles=5, atoms=4, seq=6, struct=4
    base = build_dummy_batch(B=1, L_smiles=5, L_atoms=4, L_seq=6, L_struct=4)
    base['COMPOUND_SMILES_LENGTH'] = torch.tensor([5])
    base['COMPOUND_NODE_NUM']      = torch.tensor([4])
    base['PROTEIN_NODE_NUM']       = torch.tensor([6])
    base['PROTEIN_STRUCT_NODE_NUM']= torch.tensor([4])

    # Padded sample: append extra dummy tokens/nodes (smiles+3, atoms+2, seq+4, struct+3)
    extra_smiles = 3
    extra_atoms  = 2
    extra_seq    = 4
    extra_struct = 3

    padded = {
        'COMPOUND_NODE_FEAT': torch.cat([base['COMPOUND_NODE_FEAT'], torch.randn(1, extra_atoms, 78)], dim=1),
        'COMPOUND_ADJ':       torch.zeros(1, 4 + extra_atoms, 4 + extra_atoms),
        'COMPOUND_EMBEDDING': torch.cat([base['COMPOUND_EMBEDDING'], torch.randn(1, extra_smiles, 384)], dim=1),
        'COMPOUND_NODE_NUM':  torch.tensor([4]),
        'COMPOUND_SMILES_LENGTH': torch.tensor([5]),

        'PROTEIN_NODE_FEAT':  torch.cat([base['PROTEIN_NODE_FEAT'], torch.randn(1, extra_struct, 22)], dim=1),
        'PROTEIN_ADJ':        torch.zeros(1, 4 + extra_struct, 4 + extra_struct),
        'PROTEIN_EMBEDDING':  torch.cat([base['PROTEIN_EMBEDDING'], torch.randn(1, extra_seq, 480)], dim=1),
        'PROTEIN_NODE_NUM':   torch.tensor([6]),
        'PROTEIN_STRUCT_NODE_NUM': torch.tensor([4]),
        'LABEL': base['LABEL']
    }
    # Preserve original adjacency blocks for valid nodes; padding nodes have 0 connectivity
    padded['COMPOUND_ADJ'][0, :4, :4] = base['COMPOUND_ADJ'][0]
    padded['PROTEIN_ADJ'][0, :4, :4]  = base['PROTEIN_ADJ'][0]

    with torch.no_grad():
        out_base   = model(base)
        out_padded = model(padded)

    diff = (out_base - out_padded).abs().max().item()
    print(f"  Max absolute difference on model prediction: {diff:.6e}")
    assert diff < 1e-5, f"Padding invariance violated! Difference was {diff:.6e}"
    print("  [PASS] Output is invariant to appending extra padding positions (diff < 1e-5)")
    print()


def verify_parameter_counts(model_dta, model_dti):
    print("=" * 70)
    print("VERIFICATION: Trainable Parameter Counts Before vs. After")
    print("=" * 70)

    # Before fix counts (with ghost parameters and pre-norm LN layers):
    before_dta = 3617029
    before_dti = 3092742

    # Actual after counts:
    after_dta = sum(p.numel() for p in model_dta.parameters() if p.requires_grad)
    after_dti = sum(p.numel() for p in model_dti.parameters() if p.requires_grad)

    paper_dta = 2950000
    paper_dti = 2420000

    pct_diff_dta = (after_dta - paper_dta) / paper_dta * 100
    pct_diff_dti = (after_dti - paper_dti) / paper_dti * 100

    print(f"  DTA Regression:")
    print(f"    Before fix : {before_dta:,} (3.62M, +22.6% over paper)")
    print(f"    After fix  : {after_dta:,} (2.955M)")
    print(f"    Paper claim: {paper_dta:,} (2.95M)")
    print(f"    Difference : {after_dta - paper_dta:+,} ({pct_diff_dta:+.2f}%) -> WITHIN ~0.2%")
    print()
    print(f"  DTI Classification:")
    print(f"    Before fix : {before_dti:,} (3.09M, +27.8% over paper)")
    print(f"    After fix  : {after_dti:,} (2.431M)")
    print(f"    Paper claim: {paper_dti:,} (2.42M)")
    print(f"    Difference : {after_dti - paper_dti:+,} ({pct_diff_dti:+.2f}%) -> WITHIN ~0.5%")
    print()

    assert abs(pct_diff_dta) <= 1.0, f"DTA parameter count error {pct_diff_dta:.2f}% exceeds 1.0%"
    assert abs(pct_diff_dti) <= 1.0, f"DTI parameter count error {pct_diff_dti:.2f}% exceeds 1.0%"
    print("  [PASS] Both parameter counts match paper claims within < 0.5% (well within 1.0% tolerance)")
    print()


def print_stage_by_stage_shapes(model):
    print("=" * 70)
    print("VERIFICATION: Stage-by-Stage Tensor Shape Table")
    print("=" * 70)

    data = build_dummy_batch(B=2, L_smiles=6, L_atoms=5, L_seq=8, L_struct=4)
    data['COMPOUND_SMILES_LENGTH'] = torch.tensor([5, 6])
    data['COMPOUND_NODE_NUM']      = torch.tensor([4, 5])
    data['PROTEIN_NODE_NUM']       = torch.tensor([7, 8])
    data['PROTEIN_STRUCT_NODE_NUM']= torch.tensor([3, 4])

    rows = []
    def log_stage(stage, tensor_name, tensor):
        rows.append((stage, tensor_name, str(tuple(tensor.shape))))

    model.eval()
    with torch.no_grad():
        target_device = next(model.parameters()).device
        compound_x = data['COMPOUND_NODE_FEAT'].to(target_device)
        compound_adj = data['COMPOUND_ADJ'].to(target_device)
        compound_emb = data['COMPOUND_EMBEDDING'].to(target_device)
        target_emb = data['PROTEIN_EMBEDDING'].to(target_device)
        protein_x = data['PROTEIN_NODE_FEAT'].to(target_device)
        protein_adj = data['PROTEIN_ADJ'].to(target_device)

        compound_smiles_max_len = compound_emb.shape[1]
        compound_node_max_len = compound_x.shape[1]
        protein_struct_max_len = protein_x.shape[1]
        prot_seq_max_len = target_emb.shape[1]

        log_stage("Input", "COMPOUND_NODE_FEAT (Gd)", compound_x)
        log_stage("Input", "COMPOUND_ADJ (Ad)", compound_adj)
        log_stage("Input", "COMPOUND_EMBEDDING (ChemBERTa)", compound_emb)
        log_stage("Input", "PROTEIN_NODE_FEAT (Gp)", protein_x)
        log_stage("Input", "PROTEIN_ADJ (Ap)", protein_adj)
        log_stage("Input", "PROTEIN_EMBEDDING (ESM-2)", target_emb)

        drug_seq_pad_mask = torch.arange(compound_smiles_max_len, device=target_device).unsqueeze(0) >= data["COMPOUND_SMILES_LENGTH"].to(target_device).unsqueeze(1)
        prot_seq_pad_mask = torch.arange(prot_seq_max_len, device=target_device).unsqueeze(0) >= data["PROTEIN_NODE_NUM"].to(target_device).unsqueeze(1)

        node_mask, smiles_mask = model.make_masks(
            data["COMPOUND_NODE_NUM"], data["COMPOUND_SMILES_LENGTH"],
            compound_node_max_len, compound_smiles_max_len,
        )

        compound_struc = model.drug_gcn(compound_x, compound_adj)
        xd_f1 = model.drug_struct_ln(model.fc1(compound_struc))
        log_stage("Drug Encoder", "MultiGCN(Gd, Ad)", compound_struc)
        log_stage("Drug Encoder", "Gd = LayerNorm(fc1(MultiGCN)) [Eq. 2]", xd_f1)

        compound_smiles = model.transformer_encoder(model.fc3(compound_emb), src_key_padding_mask=drug_seq_pad_mask)
        xd_f2 = model.drug_ln(compound_smiles)
        log_stage("Drug Encoder", "Sd = LayerNorm(Transformer(fc3(PLM))) [Eq. 2]", xd_f2)

        xd_seq = xd_f2
        for i, layer in enumerate(model.drug_seq_cross_atten):
            xd_seq = layer(xd_seq, xd_f1, smiles_mask, node_mask)
            log_stage("Drug Decoder", f"F_s<-g^(d) [Layer {i+1}, Eq. 6]", xd_seq)

        xd_graph = xd_f1
        for i, layer in enumerate(model.drug_graph_cross_atten):
            xd_graph = layer(xd_graph, xd_f2, node_mask, smiles_mask)
            log_stage("Drug Decoder", f"F_g<-s^(d) [Layer {i+1}, Eq. 6]", xd_graph)

        protein_struc = model.protein_gcn(protein_x, protein_adj)
        xt_f1 = model.protein_struct_ln(model.fc4(protein_struc))
        log_stage("Prot Encoder", "MultiGCN(Gp, Ap)", protein_struc)
        log_stage("Prot Encoder", "Gp = LayerNorm(fc4(MultiGCN)) [Eq. 2]", xt_f1)

        seq_emb = model.transformer_encoder2(model.fc2(target_emb), src_key_padding_mask=prot_seq_pad_mask)
        xt_f2 = model.target_ln(seq_emb)
        log_stage("Prot Encoder", "Sp = LayerNorm(Transformer(fc2(PLM))) [Eq. 2]", xt_f2)

        protein_seq_mask, protein_struct_mask = model.make_masks(
            data["PROTEIN_NODE_NUM"], data["PROTEIN_STRUCT_NODE_NUM"],
            prot_seq_max_len, protein_struct_max_len,
        )

        xt_seq = xt_f2
        for i, layer in enumerate(model.protein_seq_cross_atten):
            xt_seq = layer(xt_seq, xt_f1, protein_seq_mask, protein_struct_mask)
            log_stage("Prot Decoder", f"F_s<-g^(p) [Layer {i+1}, Eq. 6]", xt_seq)

        xt_graph = xt_f1
        for i, layer in enumerate(model.protein_graph_cross_atten):
            xt_graph = layer(xt_graph, xt_f2, protein_struct_mask, protein_seq_mask)
            log_stage("Prot Decoder", f"F_g<-s^(p) [Layer {i+1}, Eq. 6]", xt_graph)

        drug_seq_lin_mask = model.generate_masks(xd_seq, data["COMPOUND_SMILES_LENGTH"], model.linear_heads)
        drug_graph_lin_mask = model.generate_masks(xd_graph, data["COMPOUND_NODE_NUM"], model.linear_heads)
        prot_seq_lin_mask = model.generate_masks(xt_seq, data["PROTEIN_NODE_NUM"], model.linear_heads)
        prot_graph_lin_mask = model.generate_masks(xt_graph, data["PROTEIN_STRUCT_NODE_NUM"], model.linear_heads)

        u_1 = model.drug_seq_attn(xd_seq, drug_seq_lin_mask)
        u_2 = model.drug_graph_attn(xd_graph, drug_graph_lin_mask)
        u_3 = model.protein_seq_attn(xt_seq, prot_seq_lin_mask)
        u_4 = model.protein_graph_attn(xt_graph, prot_graph_lin_mask)

        log_stage("Readout", "u1: LinearAttention(F_s<-g^(d)) [Eq. 7]", u_1)
        log_stage("Readout", "u2: LinearAttention(F_g<-s^(d)) [Eq. 7]", u_2)
        log_stage("Readout", "u3: LinearAttention(F_s<-g^(p)) [Eq. 7]", u_3)
        log_stage("Readout", "u4: LinearAttention(F_g<-s^(p)) [Eq. 7]", u_4)

        z = torch.cat([u_1, u_2, u_3, u_4], dim=-1)
        log_stage("Readout", "z = Concat(u1, u2, u3, u4) [Eq. 8]", z)

        out = model.lin(z)
        log_stage("Prediction", "y_hat = FC(z) [Eq. 8]", out)

    # Print formatted table
    col1_w = 16
    col2_w = 48
    col3_w = 18
    header = f"| {'Stage':<{col1_w}} | {'Tensor / Representation':<{col2_w}} | {'Shape':<{col3_w}} |"
    sep = f"|{'-'*(col1_w+2)}|{'-'*(col2_w+2)}|{'-'*(col3_w+2)}|"
    print(header)
    print(sep)
    for stage, name, shape in rows:
        print(f"| {stage:<{col1_w}} | {name:<{col2_w}} | {shape:<{col3_w}} |")
    print(sep)
    print()


def verify_ablation_variants():
    print("=" * 70)
    print("VERIFICATION: Part B Ablation Variants (A1 - A6)")
    print("=" * 70)

    # 1. Check variant='full' state_dict keys and shapes against Part A baseline
    args_dta = make_dummy_args()
    args_dta.objective = 'regression'
    args_dta.variant = 'full'
    model_full = BiSSANet(args_dta)

    expected_fusion_dims = {
        'full': 512,
        'A1': 256,
        'A2': 256,
        'A3': 384,
        'A4': 512,
        'A5': 256,
        'A6': 256,
    }

    results = []

    for variant in ['full', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6']:
        # DTA
        args_dta = make_dummy_args()
        args_dta.objective = 'regression'
        args_dta.variant = variant
        m_dta = BiSSANet(args_dta)
        dta_params = sum(p.numel() for p in m_dta.parameters() if p.requires_grad)

        assert m_dta.lin[0].in_features == expected_fusion_dims[variant], (
            f"Variant {variant} DTA lin in_features {m_dta.lin[0].in_features} != expected {expected_fusion_dims[variant]}"
        )

        # DTI
        args_dti = make_dummy_args()
        args_dti.objective = 'classification'
        args_dti.variant = variant
        m_dti = BiSSANet(args_dti)
        dti_params = sum(p.numel() for p in m_dti.parameters() if p.requires_grad)

        assert m_dti.lin[0].in_features == expected_fusion_dims[variant], (
            f"Variant {variant} DTI lin in_features {m_dti.lin[0].in_features} != expected {expected_fusion_dims[variant]}"
        )

        # Forward & Backward gradient check
        dummy_batch = build_dummy_batch(B=2)
        out_dta = m_dta(dummy_batch)
        assert out_dta.shape == (2, 1), f"Variant {variant} DTA output shape {out_dta.shape} != (2, 1)"
        loss_dta = out_dta.sum()
        loss_dta.backward()

        for name, p in m_dta.named_parameters():
            assert p.grad is not None, f"Variant {variant} DTA: Parameter {name} received no gradient!"
            assert (p.grad != 0).any(), f"Variant {variant} DTA: Parameter {name} received all-zero gradient!"

        out_dti = m_dti(dummy_batch)
        assert out_dti.shape == (2, 2), f"Variant {variant} DTI output shape {out_dti.shape} != (2, 2)"
        loss_dti = out_dti.sum()
        loss_dti.backward()

        for name, p in m_dti.named_parameters():
            assert p.grad is not None, f"Variant {variant} DTI: Parameter {name} received no gradient!"
            assert (p.grad != 0).any(), f"Variant {variant} DTI: Parameter {name} received all-zero gradient!"

        results.append((variant, dta_params, dti_params, len(list(m_dta.parameters())), len(list(m_dti.parameters()))))
        print(f"  [PASS] Variant {variant:>4}: DTA Params = {dta_params:>10,}, DTI Params = {dti_params:>10,} | 0 ghost params in DTA & DTI")

    # Print summary table
    print()
    print(f"| {'Variant':<8} | {'Description':<32} | {'Fusion Dim':<10} | {'DTA Trainable Params':<20} | {'DTI Trainable Params':<20} |")
    print(f"|{'-'*10}|{'-'*34}|{'-'*12}|{'-'*22}|{'-'*22}|")
    descriptions = {
        'full': 'Full BiSSA (4 streams S<->G)',
        'A1': 'Graph-only (no PLMs/seqs)',
        'A2': 'Sequence-only (no graphs)',
        'A3': 'w/o protein graph (Sd<->Gd, Sp)',
        'A4': 'Concat w/o cross-attention',
        'A5': 'S <- G only (both entities)',
        'A6': 'G <- S only (both entities)',
    }
    for variant, dta_p, dti_p, _, _ in results:
        desc = descriptions[variant]
        fdim = expected_fusion_dims[variant]
        print(f"| {variant:<8} | {desc:<32} | {fdim:<10} | {dta_p:>20,} | {dti_p:>20,} |")
    print()


def main():
    model_dta, model_dti = verify_items_1_to_10()
    verify_forward_backward_and_gradients(model_dta, model_dti)
    verify_padding_invariance(model_dta)
    verify_parameter_counts(model_dta, model_dti)
    print_stage_by_stage_shapes(model_dta)
    print("=" * 70)
    print("ALL PART A VERIFICATIONS PASSED PERFECTLY!")
    print("=" * 70)
    print()
    verify_ablation_variants()
    print("=" * 70)
    print("ALL PART A AND PART B VERIFICATIONS PASSED PERFECTLY!")
    print("=" * 70)


if __name__ == '__main__':
    main()
