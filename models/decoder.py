import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
import os



class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.scale = math.sqrt(hid_dim // n_heads)

    def forward(self, query, key, value, mask=None):
        # query = [batch_size, len_Q, hid_dim]
        # key   = [batch_size, len_K, hid_dim]
        # value = [batch_size, len_K, hid_dim]
        # mask  = [batch_size, 1, 1, len_K]  (masks invalid key positions)
        bsz = query.shape[0]
        device = query.device

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        # Q = [batch_size, len_Q, hid_dim]
        # K, V = [batch_size, len_K, hid_dim]

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        # Q = [batch_size, n_heads, len_Q, hid_dim // n_heads]
        # K = [batch_size, n_heads, len_K, hid_dim // n_heads]
        # V = [batch_size, n_heads, len_K, hid_dim // n_heads]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # energy = [batch_size, n_heads, len_Q, len_K]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))

        attention = self.do(F.softmax(energy, dim=-1))
        # attention = [batch_size, n_heads, len_Q, len_K]

        x = torch.matmul(attention, V)
        # x = [batch_size, n_heads, len_Q, hid_dim // n_heads]
        x = x.permute(0, 2, 1, 3).contiguous()
        # x = [batch_size, len_Q, n_heads, hid_dim // n_heads]
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        # x = [batch_size, len_Q, hid_dim]
        x = self.fc(x)
        # x = [batch_size, len_Q, hid_dim]

        return x



class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.pf_dim = pf_dim

        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)  # convolution neural units
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)  # convolution neural units

        self.do = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch_size, sent_len, hid_dim]

        x = x.permute(0, 2, 1)
        # x = [batch_size, hid_dim, sent_len]
        x = self.do(F.relu(self.fc_1(x)))
        # x = [batch_size, pf_dim, sent_len]
        x = self.fc_2(x)
        # x = [batch_size, hid_dim, sent_len]
        x = x.permute(0, 2, 1)
        # x = [batch_size, sent_len, hid_dim]
        return x


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        self.ln_trg = nn.LayerNorm(hid_dim)
        self.ln_src = nn.LayerNorm(hid_dim)
        self.ln_ff = nn.LayerNorm(hid_dim)
        self.sa = SelfAttention(hid_dim, n_heads, dropout)
        self.ea = SelfAttention(hid_dim, n_heads, dropout)
        self.pf = PositionwiseFeedforward(hid_dim, hid_dim, dropout)
        self.do = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        # trg = [batch_size, len_trg, hid_dim]  (Query source)
        # src = [batch_size, len_src, hid_dim]  (Key/Value source)
        # trg_mask = [batch_size, 1, 1, len_trg]
        # src_mask = [batch_size, 1, 1, len_src]

        # Pre-Norm self-attention on trg with residual
        _trg = self.ln_trg(trg)
        trg = trg + self.do(self.sa(_trg, _trg, _trg, trg_mask))

        # Pre-Norm cross-attention: trg queries src, with residual
        _trg = self.ln_src(trg)
        trg = trg + self.do(self.ea(_trg, src, src, src_mask))

        # Pre-Norm feedforward (1D CNN) with residual
        _trg = self.ln_ff(trg)
        trg = trg + self.do(self.pf(_trg))

        return trg
