import math
import torch
from models.gnn import MultiGCN
from torch import nn
import torch.nn.functional as F
from models.decoder import DecoderLayer
from torch.nn.utils.weight_norm import weight_norm
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LinearAttention(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=32, heads=10):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.heads = heads

        self.linear_first = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.linear_second = torch.nn.Linear(self.hidden_dim, self.heads)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, masks):
        sentence_att = torch.tanh(self.linear_first(x))
        sentence_att = self.linear_second(sentence_att)
        sentence_att = sentence_att.transpose(1, 2)
        minus_inf = -9e15 * torch.ones_like(sentence_att)
        e = torch.where(masks > 0.5, sentence_att, minus_inf)  # (B,heads,seq_len)
        att = self.softmax(e)
        sentence_embed = att @ x
        avg_sentence_embed = torch.sum(sentence_embed, 1) / self.heads

        return avg_sentence_embed


class BiSSANet(torch.nn.Module):
    """Bidirectional Sequence-Structure Attention Network (BiSSA).

    Official architecture for:
    'Bidirectional Sequence-Structure Attention for Drug-Target Interaction
     and Affinity Prediction' (Thati & Javvadi, 2026).

    Features:
      - Symmetric drug & protein branches
      - Pretrained sequence embeddings (ChemBERTa-2, ESM-2)
      - Topological & 3D contact graphs (RDKit, AlphaFold/RCSB)
      - Reciprocal cross-attention pathways (S <- G and G <- S)
      - Multi-head Linear Attention readouts -> joint embedding
    
    Ablation variants supported via `args.variant`:
      - 'full': Full BiSSA (4 streams: S<-G, G<-S for drug and protein; 512-d fusion)
      - 'A1': Graph-only (no PLMs/sequences; drug GCN + protein GCN; 256-d fusion)
      - 'A2': Sequence-only (no graphs; drug Transformer + protein Transformer; 256-d fusion)
      - 'A3': w/o protein graph (drug bidirectional S<->G, protein sequence only; 384-d fusion)
      - 'A4': Concat w/o cross-attention (all 4 encoders pooled directly; 512-d fusion)
      - 'A5': S <- G only (unidirectional seq queried by graph; 256-d fusion)
      - 'A6': G <- S only (unidirectional graph queried by seq; 256-d fusion)
    """

    def __init__(self, args):
        super(BiSSANet, self).__init__()

        self.gnn_dim = args.compound_gnn_dim
        self.dropout = args.dropout
        self.decoder_dim = args.decoder_dim
        self.decoder_heads = args.decoder_heads
        self.decoder_layers = args.decoder_layers
        self.compound_text_dim = args.compound_text_dim
        self.compound_structure_dim = args.compound_structure_dim
        self.protein_dim = args.protein_dim
        self.linear_heads = args.linear_heads
        self.linears_hidden_dim = args.linear_hidden_dim
        self.feedforward_dim = args.pf_dim
        self.encoder_heads = args.encoder_heads
        self.encoder_layers = args.encoder_layers
        self.protein_pretrained_dim = args.protein_pretrained_dim
        self.compound_pretrained_dim = args.compound_pretrained_dim
        self.protein_structure_dim = args.protein_structure_dim
        self.objective = args.objective
        self.variant = getattr(args, 'variant', 'full')

        valid_variants = ['full', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6']
        if self.variant not in valid_variants:
            raise ValueError(f"Unknown variant '{self.variant}'. Valid options: {valid_variants}")

        has_drug_graph = self.variant in ['full', 'A1', 'A3', 'A4', 'A5', 'A6']
        has_drug_seq = self.variant in ['full', 'A2', 'A3', 'A4', 'A5', 'A6']
        has_drug_seq_dec = self.variant in ['full', 'A3', 'A5']
        has_drug_graph_dec = self.variant in ['full', 'A3', 'A6']

        has_prot_graph = self.variant in ['full', 'A1', 'A4', 'A5', 'A6']
        has_prot_seq = self.variant in ['full', 'A2', 'A3', 'A4', 'A5', 'A6']
        has_prot_seq_dec = self.variant in ['full', 'A5']
        has_prot_graph_dec = self.variant in ['full', 'A6']

        has_drug_seq_attn = self.variant in ['full', 'A2', 'A3', 'A4', 'A5']
        has_drug_graph_attn = self.variant in ['full', 'A1', 'A3', 'A4', 'A6']
        has_prot_seq_attn = self.variant in ['full', 'A2', 'A3', 'A4', 'A5']
        has_prot_graph_attn = self.variant in ['full', 'A1', 'A4', 'A6']

        # ── Drug graph encoder ──────────────────────────────────────────
        if has_drug_graph:
            self.drug_gcn = MultiGCN(self.compound_structure_dim, self.gnn_dim)

        # ── Drug decoder: bidirectional cross-attention (Double QKV) ────
        if has_drug_seq_dec:
            # Path 1 (S_d <- G_d): SMILES queries graph
            self.drug_seq_cross_atten = nn.ModuleList([
                DecoderLayer(self.decoder_dim, self.decoder_heads, self.dropout)
                for _ in range(self.decoder_layers)
            ])
        if has_drug_graph_dec:
            # Path 2 (G_d <- S_d): Graph queries SMILES
            self.drug_graph_cross_atten = nn.ModuleList([
                DecoderLayer(self.decoder_dim, self.decoder_heads, self.dropout)
                for _ in range(self.decoder_layers)
            ])

        # ── Protein graph encoder ───────────────────────────────────────
        if has_prot_graph:
            self.protein_gcn = MultiGCN(self.protein_structure_dim, self.gnn_dim)
            self.fc4 = nn.Linear(self.gnn_dim, self.protein_dim)
            self.protein_struct_ln = nn.LayerNorm(self.protein_dim)

        # ── Protein decoder: bidirectional cross-attention (Double QKV) ─
        if has_prot_seq_dec:
            # Path 1 (S_p <- G_p): Sequence queries structure
            self.protein_seq_cross_atten = nn.ModuleList([
                DecoderLayer(self.protein_dim, self.decoder_heads, self.dropout)
                for _ in range(self.decoder_layers)
            ])
        if has_prot_graph_dec:
            # Path 2 (G_p <- S_p): Structure queries sequence
            self.protein_graph_cross_atten = nn.ModuleList([
                DecoderLayer(self.protein_dim, self.decoder_heads, self.dropout)
                for _ in range(self.decoder_layers)
            ])

        # ── Four LinearAttention Readout Blocks ──────────────────────────
        if has_drug_seq_attn:
            self.drug_seq_attn = LinearAttention(self.compound_text_dim, self.linears_hidden_dim, self.linear_heads)
        if has_drug_graph_attn:
            self.drug_graph_attn = LinearAttention(self.compound_text_dim, self.linears_hidden_dim, self.linear_heads)
        if has_prot_seq_attn:
            self.protein_seq_attn = LinearAttention(self.protein_dim, self.linears_hidden_dim, self.linear_heads)
        if has_prot_graph_attn:
            self.protein_graph_attn = LinearAttention(self.protein_dim, self.linears_hidden_dim, self.linear_heads)

        # ── Transformer Sequence Encoders (1 layer, 4 heads, FFN 1024, dropout 0.2, Post-LN) ──
        if has_drug_seq:
            enc_layer1 = nn.TransformerEncoderLayer(
                d_model=self.compound_text_dim,
                dim_feedforward=self.feedforward_dim,
                nhead=self.encoder_heads,
                dropout=self.dropout,
                batch_first=True,
                norm_first=False,  # Post-LN per paper Section III-D
            )
            self.transformer_encoder = nn.TransformerEncoder(enc_layer1, num_layers=self.encoder_layers)

        if has_prot_seq:
            enc_layer2 = nn.TransformerEncoderLayer(
                d_model=self.protein_dim,
                dim_feedforward=self.feedforward_dim,
                nhead=self.encoder_heads,
                dropout=self.dropout,
                batch_first=True,
                norm_first=False,  # Post-LN per paper Section III-D
            )
            self.transformer_encoder2 = nn.TransformerEncoder(enc_layer2, num_layers=self.encoder_layers)

        # ── Projections & Layer Norms ────────────────────────────────────
        if has_drug_graph:
            self.fc1 = nn.Linear(self.compound_structure_dim, self.compound_text_dim)
        if has_prot_seq:
            self.fc2 = nn.Linear(self.protein_pretrained_dim, self.protein_dim)
        if has_drug_seq:
            self.fc3 = nn.Linear(self.compound_pretrained_dim, self.compound_text_dim)

        if has_drug_seq:
            self.drug_ln = nn.LayerNorm(self.compound_text_dim)
        if has_drug_graph:
            self.drug_struct_ln = nn.LayerNorm(self.compound_text_dim)
        if has_prot_seq:
            self.target_ln = nn.LayerNorm(self.protein_dim)

        # ── Prediction Head ──────────────────────────────────────────────
        num_streams = (
            (1 if has_drug_seq_attn else 0) +
            (1 if has_drug_graph_attn else 0) +
            (1 if has_prot_seq_attn else 0) +
            (1 if has_prot_graph_attn else 0)
        )
        fusion_dim = num_streams * self.decoder_dim

        if self.objective == 'regression':
            self.lin = nn.Sequential(
                nn.Linear(fusion_dim, 1024),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(256, 1)
            )
        elif self.objective == 'classification':
            self.lin = nn.Sequential(
                nn.Linear(fusion_dim, 512),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(512, 2)
            )

    def generate_masks(self, adj, adj_sizes, n_heads):
        out = torch.ones(adj.shape[0], adj.shape[1], device=adj.device)
        max_size = adj.shape[1]
        if isinstance(adj_sizes, int):
            out[0, adj_sizes:max_size] = 0
        else:
            for e_id, drug_len in enumerate(adj_sizes):
                out[e_id, drug_len: max_size] = 0
        out = out.unsqueeze(1).expand(-1, n_heads, -1)
        return out

    def make_masks(self, atom_num, protein_num, compound_max_len, protein_max_len):
        batch_size = len(atom_num)
        target_device = atom_num.device if torch.is_tensor(atom_num) else (
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        compound_mask = torch.zeros((batch_size, compound_max_len), device=target_device)
        protein_mask = torch.zeros((batch_size, protein_max_len), device=target_device)

        for i in range(batch_size):
            compound_mask[i, :atom_num[i]] = 1
            protein_mask[i, :protein_num[i]] = 1
        compound_mask = compound_mask.unsqueeze(1).unsqueeze(2)
        protein_mask = protein_mask.unsqueeze(1).unsqueeze(2)
        return compound_mask, protein_mask

    def forward(self, data):
        # Infer device from model weights
        target_device = next(self.parameters()).device

        # -----------------------------------------------------------
        # Drug Branch
        # -----------------------------------------------------------
        xd_f1 = None
        xd_f2 = None
        xd_seq = None
        xd_graph = None

        if hasattr(self, 'drug_gcn'):
            compound_x = data['COMPOUND_NODE_FEAT'].to(target_device)
            compound_adj = data['COMPOUND_ADJ'].to(target_device)
            compound_node_max_len = compound_x.shape[1]
            compound_struc = self.drug_gcn(compound_x, compound_adj)
            xd_f1 = self.drug_struct_ln(self.fc1(compound_struc))

        if hasattr(self, 'transformer_encoder'):
            compound_emb = data['COMPOUND_EMBEDDING'].to(target_device)
            compound_smiles_max_len = compound_emb.shape[1]
            drug_seq_pad_mask = torch.arange(compound_smiles_max_len, device=target_device).unsqueeze(0) >= data["COMPOUND_SMILES_LENGTH"].to(target_device).unsqueeze(1)
            compound_smiles = self.transformer_encoder(self.fc3(compound_emb), src_key_padding_mask=drug_seq_pad_mask)
            xd_f2 = self.drug_ln(compound_smiles)

        if hasattr(self, 'drug_seq_cross_atten') or hasattr(self, 'drug_graph_cross_atten'):
            node_mask, smiles_mask = self.make_masks(
                data["COMPOUND_NODE_NUM"],
                data["COMPOUND_SMILES_LENGTH"],
                compound_node_max_len,
                compound_smiles_max_len,
            )
            if hasattr(self, 'drug_seq_cross_atten'):
                # Path 1 (S_d <- G_d): SMILES features query graph features
                xd_seq = xd_f2
                for layer in self.drug_seq_cross_atten:
                    xd_seq = layer(xd_seq, xd_f1, smiles_mask, node_mask)

            if hasattr(self, 'drug_graph_cross_atten'):
                # Path 2 (G_d <- S_d): Graph features query SMILES features
                xd_graph = xd_f1
                for layer in self.drug_graph_cross_atten:
                    xd_graph = layer(xd_graph, xd_f2, node_mask, smiles_mask)
        else:
            # When cross-attention is omitted (A1, A2, A4)
            xd_seq = xd_f2
            xd_graph = xd_f1

        # -----------------------------------------------------------
        # Protein Branch
        # -----------------------------------------------------------
        xt_f1 = None
        xt_f2 = None
        xt_seq = None
        xt_graph = None

        if hasattr(self, 'protein_gcn'):
            protein_x = data['PROTEIN_NODE_FEAT'].to(target_device)
            protein_adj = data['PROTEIN_ADJ'].to(target_device)
            protein_struct_max_len = protein_x.shape[1]
            protein_struc = self.protein_gcn(protein_x, protein_adj)
            xt_f1 = self.protein_struct_ln(self.fc4(protein_struc))

        if hasattr(self, 'transformer_encoder2'):
            target_emb = data['PROTEIN_EMBEDDING'].to(target_device)
            prot_seq_max_len = target_emb.shape[1]
            prot_seq_pad_mask = torch.arange(prot_seq_max_len, device=target_device).unsqueeze(0) >= data["PROTEIN_NODE_NUM"].to(target_device).unsqueeze(1)
            seq_emb = self.transformer_encoder2(self.fc2(target_emb), src_key_padding_mask=prot_seq_pad_mask)
            xt_f2 = self.target_ln(seq_emb)

        if hasattr(self, 'protein_seq_cross_atten') or hasattr(self, 'protein_graph_cross_atten'):
            protein_seq_mask, protein_struct_mask = self.make_masks(
                data["PROTEIN_NODE_NUM"],
                data["PROTEIN_STRUCT_NODE_NUM"],
                prot_seq_max_len,
                protein_struct_max_len,
            )
            if hasattr(self, 'protein_seq_cross_atten'):
                # Path 1 (S_p <- G_p): Sequence features query structure features
                xt_seq = xt_f2
                for layer in self.protein_seq_cross_atten:
                    xt_seq = layer(xt_seq, xt_f1, protein_seq_mask, protein_struct_mask)

            if hasattr(self, 'protein_graph_cross_atten'):
                # Path 2 (G_p <- S_p): Structure features query sequence features
                xt_graph = xt_f1
                for layer in self.protein_graph_cross_atten:
                    xt_graph = layer(xt_graph, xt_f2, protein_struct_mask, protein_seq_mask)
        else:
            # When cross-attention is omitted (A1, A2, A3, A4)
            xt_seq = xt_f2
            xt_graph = xt_f1

        # -----------------------------------------------------------
        # LinearAttention Readout Blocks & Pooling
        # -----------------------------------------------------------
        pooled_streams = []

        # 1. Drug sequence stream (u_1)
        if hasattr(self, 'drug_seq_attn'):
            drug_seq_lin_mask = self.generate_masks(xd_seq, data["COMPOUND_SMILES_LENGTH"], self.linear_heads)
            u_1 = self.drug_seq_attn(xd_seq, drug_seq_lin_mask)          # [B, 128]
            pooled_streams.append(u_1)

        # 2. Drug graph stream (u_2)
        if hasattr(self, 'drug_graph_attn'):
            drug_graph_lin_mask = self.generate_masks(xd_graph, data["COMPOUND_NODE_NUM"], self.linear_heads)
            u_2 = self.drug_graph_attn(xd_graph, drug_graph_lin_mask)      # [B, 128]
            pooled_streams.append(u_2)

        # 3. Protein sequence stream (u_3)
        if hasattr(self, 'protein_seq_attn'):
            prot_seq_lin_mask = self.generate_masks(xt_seq, data["PROTEIN_NODE_NUM"], self.linear_heads)
            u_3 = self.protein_seq_attn(xt_seq, prot_seq_lin_mask)        # [B, 128]
            pooled_streams.append(u_3)

        # 4. Protein graph stream (u_4)
        if hasattr(self, 'protein_graph_attn'):
            prot_graph_lin_mask = self.generate_masks(xt_graph, data["PROTEIN_STRUCT_NODE_NUM"], self.linear_heads)
            u_4 = self.protein_graph_attn(xt_graph, prot_graph_lin_mask)  # [B, 128]
            pooled_streams.append(u_4)

        # -----------------------------------------------------------
        # Joint Concatenation & Task Prediction
        # -----------------------------------------------------------
        z = torch.cat(pooled_streams, dim=-1)
        out = self.lin(z)

        return out


# Backwards compatibility alias
PMMRNet = BiSSANet
