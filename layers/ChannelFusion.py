"""Lightweight low-rank residual channel mixing."""
import torch
from torch import nn


class LightweightChannelFusion(nn.Module):
    def __init__(self, channels, rank=4, dropout=0.0):
        super().__init__()
        channels = int(channels)
        rank = max(1, min(int(rank), channels))
        self.down = nn.Linear(channels, rank, bias=False)
        self.up = nn.Linear(rank, channels, bias=False)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.tensor(0.10))
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.xavier_uniform_(self.up.weight)

    def forward(self, x):
        # x: [B, pred_len, C]
        residual = self.up(self.activation(self.down(x)))
        residual = self.dropout(residual)
        return x + torch.tanh(self.gate) * residual
