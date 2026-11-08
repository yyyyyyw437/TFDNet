"""Frequency, patch, and channel components with independent ablation switches."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.ChannelFusion import LightweightChannelFusion
from layers.FrequencyBranch import FrequencyBranch
from layers.PatchBranch import PatchBranch
from layers.RevIN import RevIN


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.channels = int(configs.enc_in)
        self.use_frequency = bool(getattr(
            configs, "use_freq_branch", 1))
        self.use_patch = bool(getattr(
            configs, "use_patch_branch", 1))
        self.use_channel = bool(getattr(
            configs, "use_channel_fusion", 0))
        if not self.use_frequency and not self.use_patch:
            raise ValueError(
                "At least one of use_freq_branch/use_patch_branch must be on")

        self.revin_layer = RevIN(
            self.channels, affine=False, subtract_last=False)
        self.frequency_branch = (
            FrequencyBranch(configs) if self.use_frequency else None)
        self.patch_branch = (
            PatchBranch(configs) if self.use_patch else None)
        self.channel_fusion = (
            LightweightChannelFusion(
                self.channels,
                rank=int(getattr(configs, "channel_rank", 4)),
                dropout=float(configs.dropout))
            if self.use_channel else None)
        if self.use_frequency and self.use_patch:
            self.branch_logits = nn.Parameter(torch.zeros(2))
        else:
            self.register_parameter("branch_logits", None)
        self.last_branch_weights = None

    def forward(self, x_enc):
        batch, seq_len, channels = x_enc.shape
        if seq_len != self.seq_len or channels != self.channels:
            raise ValueError(
                "Expected [B, {}, {}], got {}".format(
                    self.seq_len, self.channels, tuple(x_enc.shape)))

        x_norm = self.revin_layer(x_enc, "norm")
        predictions = []
        if self.frequency_branch is not None:
            predictions.append(self.frequency_branch(x_norm))
        if self.patch_branch is not None:
            predictions.append(self.patch_branch(x_norm))

        if len(predictions) == 1:
            fused = predictions[0]
            self.last_branch_weights = torch.ones(
                1, device=x_enc.device)
        else:
            weights = F.softmax(self.branch_logits, dim=0)
            fused = weights[0] * predictions[0] + weights[1] * predictions[1]
            self.last_branch_weights = weights.detach()

        if self.channel_fusion is not None:
            fused = self.channel_fusion(fused)
        return self.revin_layer(fused, "denorm")
