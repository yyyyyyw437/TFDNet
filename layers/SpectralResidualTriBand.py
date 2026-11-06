"""Shared-spectrum tri-band TFDNet blocks."""
import math

import torch
import torch.nn.functional as F
from torch import nn

from layers.TriBandPooledSTFT import (
    ComplexDepthwiseTemporalConv,
    ComplexFrequencyFFN,
)


def _logit(value):
    value = min(max(float(value), 1e-4), 1.0 - 1e-4)
    return math.log(value / (1.0 - value))


class LearnableSpectralPartition(nn.Module):
    """Split one STFT into ordered low, mid, and high frequency masks."""

    def __init__(self, n_freq, low_cutoff=0.20, high_cutoff=0.50,
                 transition=0.08):
        super().__init__()
        if not 0.0 < low_cutoff < high_cutoff < 1.0:
            raise ValueError("cutoffs must satisfy 0 < low < high < 1")
        self.n_freq = int(n_freq)
        self.transition = float(transition)
        first = low_cutoff / 0.60
        second = (high_cutoff - low_cutoff) / (1.0 - low_cutoff)
        self.low_cutoff_raw = nn.Parameter(
            torch.tensor(_logit(first), dtype=torch.float32))
        self.high_cutoff_raw = nn.Parameter(
            torch.tensor(_logit(second), dtype=torch.float32))
        self.last_masks = None
        self.last_cutoffs = None

    def forward(self):
        first = 0.60 * torch.sigmoid(self.low_cutoff_raw)
        second = first + (1.0 - first) * torch.sigmoid(
            self.high_cutoff_raw)
        frequencies = torch.linspace(
            0.0, 1.0, self.n_freq, device=first.device, dtype=first.dtype)
        transition = max(self.transition, 1e-3)

        low = torch.sigmoid((first - frequencies) / transition)
        high = torch.sigmoid((frequencies - second) / transition)
        mid = (1.0 - low - high).clamp_min(0.0)
        masks = torch.stack([low, mid, high], dim=0)
        masks = masks / masks.sum(dim=0, keepdim=True).clamp_min(1e-6)

        self.last_masks = masks.detach()
        self.last_cutoffs = torch.stack([first, second]).detach()
        return masks.unsqueeze(-1)


class ComplexDepthwiseBandBlock(nn.Module):
    """Depthwise temporal filtering for one spectral band."""

    def __init__(self, n_freq, kernel_size, dropout=0.0):
        super().__init__()
        self.temporal = ComplexDepthwiseTemporalConv(
            n_freq, kernel_size=kernel_size, dropout=dropout)

    def forward(self, x):
        return self.temporal(x)


class SpectralTriBandEncoder(nn.Module):
    """One STFT followed by spectral band transforms and one inverse STFT."""

    def __init__(
            self, n_channels, seq_len, n_fft, low_kernel=15, mid_kernel=5,
            high_kernel=3, gate_hidden=16, gate_prior=(0.60, 0.50, 0.40),
            mask_transition=0.08, dropout=0.0):
        super().__init__()
        self.n_channels = int(n_channels)
        self.seq_len = int(seq_len)
        self.n_fft = int(n_fft)
        self.hop = int(self.n_fft // 2)
        self.n_freq = int(self.n_fft // 2 + 1)
        self.n_frames = int(self.seq_len // self.hop + 1)

        self.partition = LearnableSpectralPartition(
            self.n_freq, transition=mask_transition)
        self.low_filter = ComplexDepthwiseBandBlock(
            self.n_freq, kernel_size=low_kernel, dropout=dropout)
        self.mid_filter = ComplexDepthwiseBandBlock(
            self.n_freq, kernel_size=mid_kernel, dropout=dropout)
        self.high_filter = ComplexDepthwiseBandBlock(
            self.n_freq, kernel_size=high_kernel, dropout=dropout)
        self.frequency_ffn = ComplexFrequencyFFN(
            self.n_freq, dropout=dropout)

        prior = torch.tensor(gate_prior, dtype=torch.float32)
        if prior.numel() != 3 or torch.any(prior <= 0) or torch.any(prior >= 1):
            raise ValueError("spectral_gate_prior must contain values in (0, 1)")
        self.register_buffer(
            "gate_prior", prior, persistent=False)
        self.gate = nn.Sequential(
            nn.Linear(3, int(gate_hidden)),
            nn.GELU(),
            nn.Linear(int(gate_hidden), 3),
        )
        for module in self.gate.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        with torch.no_grad():
            self.gate[-1].weight.mul_(0.10)
            self.gate[-1].bias.copy_(torch.logit(prior))

        self.dropout = nn.Dropout(dropout)
        self.last_band_gates = None

    def forward(self, x, return_aux=False):
        # x: [B * C, L]
        window = torch.ones(
            self.n_fft, device=x.device, dtype=x.dtype)
        frames = torch.stft(
            x, n_fft=self.n_fft, hop_length=self.hop, window=window,
            return_complex=True)
        masks = self.partition()
        band_frames = [frames * masks[index] for index in range(3)]

        transformed = [
            self.low_filter(band_frames[0]),
            self.mid_filter(band_frames[1]),
            self.high_filter(band_frames[2]),
        ]
        deltas = [
            transformed[index] - band_frames[index] for index in range(3)]

        band_energy = torch.stack([
            values.abs().square().mean(dim=(1, 2))
            for values in band_frames
        ], dim=-1)
        batch_frames = x.shape[0]
        batch = batch_frames // self.n_channels
        band_energy = band_energy.reshape(batch, self.n_channels, 3)
        gate_features = torch.log(band_energy.clamp_min(1e-8))
        gate_features = (
            gate_features - gate_features.mean(dim=-1, keepdim=True))
        band_gates = torch.sigmoid(self.gate(gate_features))
        self.last_band_gates = band_gates.detach()

        flat_gates = band_gates.reshape(batch_frames, 3).view(-1, 3, 1, 1)
        spectrum = frames + (
            flat_gates[:, 0] * deltas[0]
            + flat_gates[:, 1] * deltas[1]
            + flat_gates[:, 2] * deltas[2])
        spectrum = self.frequency_ffn(spectrum)
        out = torch.istft(
            spectrum, n_fft=self.n_fft, hop_length=self.hop, window=window,
            length=self.seq_len)
        out = self.dropout(out)

        if return_aux:
            return out, {
                "spectrum": spectrum,
                "band_spectra": band_frames,
                "band_gates": band_gates,
                "band_masks": self.partition.last_masks,
            }
        return out
