"""PatchTST-style temporal patch branch returning normalized forecasts."""
import torch
import torch.nn.functional as F
from torch import nn


class PatchBranch(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.channels = int(configs.enc_in)
        self.patch_len = max(1, int(getattr(configs, "patch_len", 16)))
        self.patch_len = min(self.patch_len, self.seq_len)
        self.stride = max(1, int(getattr(configs, "patch_stride", 8)))
        self.num_patches = (
            (self.seq_len - self.patch_len) // self.stride + 1)
        covered = (self.num_patches - 1) * self.stride + self.patch_len
        self.padding = covered - self.seq_len

        self.d_model = int(getattr(configs, "patch_d_model", 32))
        self.n_heads = int(getattr(configs, "patch_n_heads", 4))
        self.e_layers = int(getattr(configs, "patch_e_layers", 1))
        self.d_ff = int(getattr(
            configs, "patch_d_ff", 2 * self.d_model))
        self.dropout_rate = float(configs.dropout)
        if self.d_model % self.n_heads != 0:
            raise ValueError("patch_d_model must be divisible by patch_n_heads")

        self.patch_embed = nn.Linear(self.patch_len, self.d_model)
        self.position = nn.Parameter(torch.empty(
            1, self.num_patches, self.d_model))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True)
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=self.e_layers)
        self.head = nn.Linear(
            self.num_patches * self.d_model, self.pred_len)
        self.dropout = nn.Dropout(self.dropout_rate)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.patch_embed.weight)
        nn.init.zeros_(self.patch_embed.bias)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x_norm):
        # x_norm: [B, L, C]
        batch, seq_len, channels = x_norm.shape
        if seq_len != self.seq_len:
            raise ValueError("Unexpected patch branch sequence length")
        if self.padding:
            x_norm = F.pad(x_norm, (0, 0, 0, self.padding))
        patches = x_norm.unfold(
            dimension=1, size=self.patch_len, step=self.stride)
        # [B, num_patches, C, patch_len]
        patches = patches.permute(0, 2, 1, 3).reshape(
            batch * channels, self.num_patches, self.patch_len)
        tokens = self.patch_embed(patches) + self.position
        tokens = self.transformer(tokens)
        out = self.head(tokens.reshape(batch * channels, -1))
        out = self.dropout(out)
        return out.reshape(batch, channels, self.pred_len).permute(0, 2, 1)
