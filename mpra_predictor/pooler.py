import math
import torch
from torch import nn


class PositionWeight(nn.Module):
    def __init__(self, channels: int, seq_len: int, n_heads: int, hidden_dim: int, pos_emb_dim: int, dropout: float,):
        super().__init__()
        self.out_size = n_heads * channels
        self.channels = channels
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.pos_emb_dim = pos_emb_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.register_buffer("position_embedding", self._init_sinusoidal_pos_emb(seq_len, pos_emb_dim))  # (L, D)
        self.input_norm = nn.LayerNorm(channels)
        self.pos_mlp = nn.Sequential(
            nn.Linear(channels + pos_emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_heads)  # output logits for each head
        )
        self.softmax = nn.Softmax(dim=1)  # softmax over sequence length

    def _init_sinusoidal_pos_emb(self, seq_len, dim):
        position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim // 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        emb = torch.zeros(seq_len, dim)
        emb[:, 0::2] = torch.sin(position * div_term)
        emb[:, 1::2] = torch.cos(position * div_term)
        return emb  # (seq_len, dim)

    def __str__(self):
        return f"PosWeight:{self.n_heads}-{self.hidden_dim}-{self.pos_emb_dim}-{self.dropout}"

    def forward(self, x):
        B, C, L = x.size()
        assert L == self.seq_len, f"Input length {L} != expected {self.seq_len}"
        x = self.input_norm(x.permute(0, 2, 1)).permute(0, 2, 1)  # (B, C, L)
        pos = self.position_embedding.T.unsqueeze(0).expand(B, -1, -1)  # (B, pos_emb_dim, L)

        x_pos = torch.cat([x, pos], dim=1)  # (B, C + pos_emb + [1], L)
        x_pos = x_pos.transpose(1, 2)  # (B, L, D)
        pos_logits = self.pos_mlp(x_pos)  # (B, L, n_heads)
        pos_weights = self.softmax(pos_logits)  # (B, L, n_heads)
        pos_weights = pos_weights.permute(0, 2, 1)  # (B, n_heads, L)

        x_expanded = x[:, :self.channels, :].unsqueeze(1)  # (B, 1, C, L)
        pos_expanded = pos_weights.unsqueeze(2)  # (B, n_heads, 1, L)
        weighted = (x_expanded * pos_expanded).sum(dim=-1)  # (B, n_heads, C)
        pooled = weighted.reshape(B, self.n_heads * self.channels)  # (B, n_heads * C)
        return pooled
