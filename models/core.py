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
      - 4-stream multi-head Linear Attention readouts -> 512-dim joint embedding
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

        # ── Drug graph encoder ──────────────────────────────────────────
        self.drug_gcn = MultiGCN(self.compound_structure_dim, self.gnn_dim)

        # ── Drug decoder: bidirectional cross-attention (Double QKV) ────
        # Path 1 (S_d <- G_d): SMILES queries graph
        self.drug_seq_cross_atten = nn.ModuleList([
            DecoderLayer(self.decoder_dim, self.decoder_heads, self.dropout)
            for _ in range(self.decoder_layers)
        ])
        # Path 2 (G_d <- S_d): Graph queries SMILES
        self.drug_graph_cross_atten = nn.ModuleList([
            DecoderLayer(self.decoder_dim, self.decoder_heads, self.dropout)
            for _ in range(self.decoder_layers)
        ])

        # ── Protein graph encoder ───────────────────────────────────────
        self.protein_gcn = MultiGCN(self.protein_structure_dim, self.gnn_dim)
        self.fc4 = nn.Linear(self.gnn_dim, self.protein_dim)
        self.protein_struct_ln = nn.LayerNorm(self.protein_dim)

        # ── Protein decoder: bidirectional cross-attention (Double QKV) ─
        # Path 1 (S_p <- G_p): Sequence queries structure
        self.protein_seq_cross_atten = nn.ModuleList([
            DecoderLayer(self.protein_dim, self.decoder_heads, self.dropout)
            for _ in range(self.decoder_layers)
        ])
        # Path 2 (G_p <- S_p): Structure queries sequence
        self.protein_graph_cross_atten = nn.ModuleList([
            DecoderLayer(self.protein_dim, self.decoder_heads, self.dropout)
            for _ in range(self.decoder_layers)
        ])

        # ── Four LinearAttention Readout Blocks ──────────────────────────
        self.drug_seq_attn = LinearAttention(self.compound_text_dim, self.linears_hidden_dim, self.linear_heads)
        self.drug_graph_attn = LinearAttention(self.compound_text_dim, self.linears_hidden_dim, self.linear_heads)
        self.protein_seq_attn = LinearAttention(self.protein_dim, self.linears_hidden_dim, self.linear_heads)
        self.protein_graph_attn = LinearAttention(self.protein_dim, self.linears_hidden_dim, self.linear_heads)

        # ── Transformer Sequence Encoders (1 layer, 4 heads, FFN 1024, dropout 0.2, Post-LN) ──
        # Build template layers as local variables to avoid registering unused ghost parameters (A-1, A-2)
        enc_layer1 = nn.TransformerEncoderLayer(
            d_model=self.compound_text_dim,
            dim_feedforward=self.feedforward_dim,
            nhead=self.encoder_heads,
            dropout=self.dropout,
            batch_first=True,
            norm_first=False,  # Post-LN per paper Section III-D
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer1, num_layers=self.encoder_layers)

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
        self.fc1 = nn.Linear(self.compound_structure_dim, self.compound_text_dim)
        self.fc2 = nn.Linear(self.protein_pretrained_dim, self.protein_dim)
        self.fc3 = nn.Linear(self.compound_pretrained_dim, self.compound_text_dim)

        self.drug_ln = nn.LayerNorm(self.compound_text_dim)
        self.drug_struct_ln = nn.LayerNorm(self.compound_text_dim)
        self.target_ln = nn.LayerNorm(self.protein_dim)

        # ── Prediction Head (512-dim joint embedding) ────────────────────
        fusion_dim = self.compound_text_dim * 2 + self.protein_dim * 2  # 128*4 = 512

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

        compound_x = data['COMPOUND_NODE_FEAT'].to(target_device)
        compound_adj = data['COMPOUND_ADJ'].to(target_device)
        compound_emb = data['COMPOUND_EMBEDDING'].to(target_device)

        target_emb = data['PROTEIN_EMBEDDING'].to(target_device)
        protein_x = data['PROTEIN_NODE_FEAT'].to(target_device)
        protein_adj = data['PROTEIN_ADJ'].to(target_device)

        compound_smiles_max_len = data['COMPOUND_EMBEDDING'].shape[1]
        compound_node_max_len = data['COMPOUND_NODE_FEAT'].shape[1]
        protein_struct_max_len = data['PROTEIN_NODE_FEAT'].shape[1]

        # -----------------------------------------------------------
        # Drug Branch
        # -----------------------------------------------------------
        drug_seq_pad_mask = torch.arange(compound_smiles_max_len, device=target_device).unsqueeze(0) >= data["COMPOUND_SMILES_LENGTH"].to(target_device).unsqueeze(1)

        node_mask, smiles_mask = self.make_masks(
            data["COMPOUND_NODE_NUM"],
            data["COMPOUND_SMILES_LENGTH"],
            compound_node_max_len,
            compound_smiles_max_len,
        )

        compound_struc = self.drug_gcn(compound_x, compound_adj)
        xd_f1 = self.drug_struct_ln(self.fc1(compound_struc))

        compound_smiles = self.transformer_encoder(self.fc3(compound_emb), src_key_padding_mask=drug_seq_pad_mask)
        xd_f2 = self.drug_ln(compound_smiles)

        # Path 1 (S_d <- G_d): SMILES features query graph features
        xd_seq = xd_f2
        for layer in self.drug_seq_cross_atten:
            xd_seq = layer(xd_seq, xd_f1, smiles_mask, node_mask)

        # Path 2 (G_d <- S_d): Graph features query SMILES features
        xd_graph = xd_f1
        for layer in self.drug_graph_cross_atten:
            xd_graph = layer(xd_graph, xd_f2, node_mask, smiles_mask)

        # -----------------------------------------------------------
        # Protein Branch
        # -----------------------------------------------------------
        prot_seq_max_len = target_emb.shape[1]
        prot_seq_pad_mask = torch.arange(prot_seq_max_len, device=target_device).unsqueeze(0) >= data["PROTEIN_NODE_NUM"].to(target_device).unsqueeze(1)

        protein_struc = self.protein_gcn(protein_x, protein_adj)
        xt_f1 = self.protein_struct_ln(self.fc4(protein_struc))

        seq_emb = self.transformer_encoder2(self.fc2(target_emb), src_key_padding_mask=prot_seq_pad_mask)
        xt_f2 = self.target_ln(seq_emb)

        protein_seq_mask, protein_struct_mask = self.make_masks(
            data["PROTEIN_NODE_NUM"],
            data["PROTEIN_STRUCT_NODE_NUM"],
            target_emb.shape[1],
            protein_struct_max_len,
        )

        # Path 1 (S_p <- G_p): Sequence features query structure features
        xt_seq = xt_f2
        for layer in self.protein_seq_cross_atten:
            xt_seq = layer(xt_seq, xt_f1, protein_seq_mask, protein_struct_mask)

        # Path 2 (G_p <- S_p): Structure features query sequence features
        xt_graph = xt_f1
        for layer in self.protein_graph_cross_atten:
            xt_graph = layer(xt_graph, xt_f2, protein_struct_mask, protein_seq_mask)

        # -----------------------------------------------------------
        # Four LinearAttention Readout Blocks
        # -----------------------------------------------------------
        drug_seq_lin_mask = self.generate_masks(xd_seq, data["COMPOUND_SMILES_LENGTH"], self.linear_heads)
        drug_graph_lin_mask = self.generate_masks(xd_graph, data["COMPOUND_NODE_NUM"], self.linear_heads)
        prot_seq_lin_mask = self.generate_masks(xt_seq, data["PROTEIN_NODE_NUM"], self.linear_heads)
        prot_graph_lin_mask = self.generate_masks(xt_graph, data["PROTEIN_STRUCT_NODE_NUM"], self.linear_heads)

        u_1 = self.drug_seq_attn(xd_seq, drug_seq_lin_mask)          # [B, 128]
        u_2 = self.drug_graph_attn(xd_graph, drug_graph_lin_mask)      # [B, 128]
        u_3 = self.protein_seq_attn(xt_seq, prot_seq_lin_mask)        # [B, 128]
        u_4 = self.protein_graph_attn(xt_graph, prot_graph_lin_mask)  # [B, 128]

        # -----------------------------------------------------------
        # 512-dim Joint Concatenation & Task Prediction
        # -----------------------------------------------------------
        out = self.lin(torch.cat([u_1, u_2, u_3, u_4], dim=-1))      # [B, 512] -> prediction

        return out


# Backwards compatibility alias
PMMRNet = BiSSANet
