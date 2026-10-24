import math
import torch
from models.gnn import MultiGCN
from torch import nn
import torch.nn.functional as F
from models.decoder import DecoderLayer
from torch.nn.utils.weight_norm import weight_norm
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


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



class PMMRNet(torch.nn.Module):
    def __init__(self, args):

        super(PMMRNet, self).__init__()

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



        self.drug_gcn = MultiGCN(self.compound_structure_dim,  self.gnn_dim)

        # -----------------------------------------------------------
        # Bidirectional Decoder Stacks (Double QKV)
        # -----------------------------------------------------------
        # Drug: Path 1 (S_d <- G_d): SMILES queries graph
        self.drug_seq_cross_atten = nn.ModuleList([
            DecoderLayer(self.decoder_dim, self.decoder_heads, self.dropout)
            for _ in range(self.decoder_layers)
        ])
        # Drug: Path 2 (G_d <- S_d): graph queries SMILES
        self.drug_graph_cross_atten = nn.ModuleList([
            DecoderLayer(self.decoder_dim, self.decoder_heads, self.dropout)
            for _ in range(self.decoder_layers)
        ])

        # convolution layers – protein branch (mirrors drug branch)
        self.protein_gcn = MultiGCN(self.protein_structure_dim, self.gnn_dim)
        self.fc4 = nn.Linear(self.gnn_dim, self.protein_dim)
        self.protein_struct_ln = nn.LayerNorm(self.protein_dim)

        # Protein: Path 1 (S_p <- G_p): sequence queries structure
        self.protein_seq_cross_atten = nn.ModuleList([
            DecoderLayer(self.protein_dim, self.decoder_heads, self.dropout)
            for _ in range(self.decoder_layers)
        ])
        # Protein: Path 2 (G_p <- S_p): structure queries sequence
        self.protein_graph_cross_atten = nn.ModuleList([
            DecoderLayer(self.protein_dim, self.decoder_heads, self.dropout)
            for _ in range(self.decoder_layers)
        ])


        # -----------------------------------------------------------
        # Four LinearAttention Readout Blocks
        # -----------------------------------------------------------
        self.drug_seq_attn = LinearAttention(self.compound_text_dim, self.linears_hidden_dim, self.linear_heads)
        self.drug_graph_attn = LinearAttention(self.compound_text_dim, self.linears_hidden_dim, self.linear_heads)
        self.protein_seq_attn = LinearAttention(self.protein_dim, self.linears_hidden_dim, self.linear_heads)
        self.protein_graph_attn = LinearAttention(self.protein_dim, self.linears_hidden_dim, self.linear_heads)



        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.compound_text_dim, dim_feedforward=self.feedforward_dim, nhead=self.encoder_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.encoder_layers)

        self.encoder_layer2 = nn.TransformerEncoderLayer(d_model=self.protein_dim, dim_feedforward=self.feedforward_dim, nhead=self.encoder_heads)
        self.transformer_encoder2 = nn.TransformerEncoder(self.encoder_layer2, num_layers=self.encoder_layers)


        self.fc1 = nn.Linear(self.compound_structure_dim, self.compound_text_dim)
        self.fc2 = nn.Linear(self.protein_pretrained_dim, self.protein_dim)
        self.fc3 = nn.Linear(self.compound_pretrained_dim, self.compound_text_dim)

        self.drug_ln = nn.LayerNorm(self.compound_text_dim)
        self.drug_struct_ln = nn.LayerNorm(self.compound_text_dim)
        self.target_ln = nn.LayerNorm(self.protein_dim)


        # -----------------------------------------------------------
        # Prediction Head: fusion_dim = compound_text_dim*2 + protein_dim*2
        # -----------------------------------------------------------
        fusion_dim = self.compound_text_dim * 2 + self.protein_dim * 2

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
            self.lin = nn.Sequential(nn.Linear(fusion_dim, 512), nn.ReLU(), nn.Dropout(self.dropout),
                                    nn.Linear(512, 2))


    def generate_masks(self, adj, adj_sizes, n_heads):
        out = torch.ones(adj.shape[0], adj.shape[1])
        max_size = adj.shape[1]
        if isinstance(adj_sizes, int):
            out[0, adj_sizes:max_size] = 0
        else:
            for e_id, drug_len in enumerate(adj_sizes):
                out[e_id, drug_len: max_size] = 0
        out = out.unsqueeze(1).expand(-1, n_heads, -1)
        return out.to(device=adj.device)

    def make_masks(self, atom_num, protein_num, compound_max_len, protein_max_len):
        batch_size = len(atom_num)
        compound_mask = torch.zeros((batch_size, compound_max_len)).type_as(atom_num)
        protein_mask = torch.zeros((batch_size, protein_max_len)).type_as(atom_num)

        for i in range(batch_size):
            compound_mask[i, :atom_num[i]] = 1
            protein_mask[i, :protein_num[i]] = 1
        compound_mask = compound_mask.unsqueeze(1).unsqueeze(2)
        protein_mask = protein_mask.unsqueeze(1).unsqueeze(2)
        # Derive device from input tensor instead of global variable
        target_device = atom_num.device if torch.is_tensor(atom_num) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return compound_mask.to(target_device), protein_mask.to(target_device)

    def forward(self, data):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        compound_x = data['COMPOUND_NODE_FEAT'].to(device)
        compound_adj = data['COMPOUND_ADJ'].to(device)
        compound_emb = data['COMPOUND_EMBEDDING'].to(device)

        target_emb = data['PROTEIN_EMBEDDING'].to(device)
        protein_x = data['PROTEIN_NODE_FEAT'].to(device)
        protein_adj = data['PROTEIN_ADJ'].to(device)

        compound_smiles_max_len = data['COMPOUND_EMBEDDING'].shape[1]
        compound_node_max_len = data['COMPOUND_NODE_FEAT'].shape[1]
        protein_struct_max_len = data['PROTEIN_NODE_FEAT'].shape[1]

        # -----------------------------------------------------------
        # Drug masks for cross-attention
        # -----------------------------------------------------------
        # smiles_mask: [B, 1, 1, L_smiles] – masks invalid SMILES positions
        # node_mask:   [B, 1, 1, L_atoms]  – masks invalid atom positions
        node_mask, smiles_mask = self.make_masks(
            data["COMPOUND_NODE_NUM"],
            data["COMPOUND_SMILES_LENGTH"],
            compound_node_max_len,
            compound_smiles_max_len,
        )

        # -----------------------------------------------------------
        # Drug feature extraction
        # -----------------------------------------------------------
        # Structure branch: RDKit molecular graph -> GCN -> project to compound_text_dim
        compound_struc = self.drug_gcn(compound_x, compound_adj)
        xd_f1 = self.drug_struct_ln(self.fc1(compound_struc))  # [B, L_atoms, 128]

        # Sequence branch: ChemBERTa-2 SMILES -> Transformer -> LayerNorm
        compound_smiles = self.transformer_encoder(self.fc3(compound_emb))
        xd_f2 = self.drug_ln(compound_smiles)  # [B, L_smiles, 128]

        # -----------------------------------------------------------
        # Drug Bidirectional Cross-Attention
        # -----------------------------------------------------------
        # Path 1 (S_d <- G_d): SMILES features query graph features
        xd_seq = xd_f2
        for layer in self.drug_seq_cross_atten:
            xd_seq = layer(xd_seq, xd_f1, smiles_mask, node_mask)

        # Path 2 (G_d <- S_d): Graph features query SMILES features
        xd_graph = xd_f1
        for layer in self.drug_graph_cross_atten:
            xd_graph = layer(xd_graph, xd_f2, node_mask, smiles_mask)

        # -----------------------------------------------------------
        # Protein feature extraction
        # -----------------------------------------------------------
        # Structure branch: AlphaFold contact graph -> GCN -> project to protein_dim
        protein_struc = self.protein_gcn(protein_x, protein_adj)
        xt_f1 = self.protein_struct_ln(self.fc4(protein_struc))  # [B, L_struct, 128]

        # Sequence branch: ESM-2 embedding -> Transformer -> LayerNorm
        seq_emb = self.transformer_encoder2(self.fc2(target_emb))
        xt_f2 = self.target_ln(seq_emb)  # [B, L_seq, 128]

        # Protein masks for cross-attention
        protein_seq_mask, protein_struct_mask = self.make_masks(
            data["PROTEIN_NODE_NUM"],
            data["PROTEIN_STRUCT_NODE_NUM"],
            target_emb.shape[1],
            protein_struct_max_len,
        )

        # -----------------------------------------------------------
        # Protein Bidirectional Cross-Attention
        # -----------------------------------------------------------
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

        u_1 = self.drug_seq_attn(xd_seq, drug_seq_lin_mask)      # [B, 128]
        u_2 = self.drug_graph_attn(xd_graph, drug_graph_lin_mask)  # [B, 128]
        u_3 = self.protein_seq_attn(xt_seq, prot_seq_lin_mask)    # [B, 128]
        u_4 = self.protein_graph_attn(xt_graph, prot_graph_lin_mask)  # [B, 128]

        # -----------------------------------------------------------
        # Concatenation & Prediction
        # -----------------------------------------------------------
        out = self.lin(torch.cat([u_1, u_2, u_3, u_4], dim=-1))  # [B, 512] -> prediction

        return out