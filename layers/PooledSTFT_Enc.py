"""Original TFDNet seasonal/trend encoders with adaptive frame pooling."""
import math

import torch
import torch.nn.functional as F
from torch import nn


def complex_tanh(x):
    return torch.complex(torch.tanh(x.real), torch.tanh(x.imag))


def _resolve_pooled_frames(n_frames, ratio):
    n_frames = max(1, int(n_frames))
    ratio = float(ratio)
    if 0.0 < ratio < 1.0:
        return max(1, min(n_frames, int(math.floor(n_frames * ratio))))
    return n_frames


def _complex_to_channels(x):
    batch, n_freq, n_frames = x.shape
    channels = torch.view_as_real(x).permute(0, 1, 3, 2)
    channels = channels.reshape(batch * n_freq * 2, 1, n_frames)
    return channels, batch, n_freq


def _channels_to_complex(channels, batch, n_freq, n_frames):
    channels = channels.reshape(batch, n_freq, 2, n_frames)
    channels = channels.permute(0, 1, 3, 2).contiguous()
    return torch.view_as_complex(channels)


class ComplexAdaptivePool1d(nn.Module):
    """Pool complex STFT frames with a shared real-valued operation."""

    def __init__(self, mode="avg"):
        super().__init__()
        if mode not in ("avg", "energy"):
            raise ValueError("pool_mode must be 'avg' or 'energy'")
        self.mode = mode

    def forward(self, x, target_frames):
        n_frames = x.shape[-1]
        target_frames = min(int(target_frames), n_frames)
        if target_frames == n_frames:
            return x

        channels, batch, n_freq = _complex_to_channels(x)
        if self.mode == "avg":
            pooled = F.adaptive_avg_pool1d(channels, target_frames)
        else:
            paired = channels.reshape(batch * n_freq, 2, n_frames)
            energy = paired.square().sum(dim=1)
            pooled = []
            for index in range(target_frames):
                start = int(math.floor(index * n_frames / target_frames))
                end = int(math.ceil(
                    (index + 1) * n_frames / target_frames))
                weights = torch.softmax(energy[..., start:end], dim=-1)
                pooled.append((
                    paired[..., start:end]
                    * weights.unsqueeze(1)).sum(dim=-1))
            pooled = torch.stack(pooled, dim=-1)
            pooled = pooled.reshape(batch * n_freq * 2, 1, target_frames)

        return _channels_to_complex(
            pooled, batch, n_freq, target_frames)


class ComplexUpsample1d(nn.Module):
    """Restore the full STFT frame count after pooling."""

    def __init__(self, mode="linear"):
        super().__init__()
        if mode not in ("linear", "nearest"):
            raise ValueError("upsample_mode must be 'linear' or 'nearest'")
        self.mode = mode

    def forward(self, x, target_frames):
        n_frames = x.shape[-1]
        if n_frames == target_frames:
            return x

        channels, batch, n_freq = _complex_to_channels(x)
        kwargs = {"size": int(target_frames), "mode": self.mode}
        if self.mode == "linear":
            kwargs["align_corners"] = False
        channels = F.interpolate(channels, **kwargs)
        return _channels_to_complex(
            channels, batch, n_freq, target_frames)


class PooledSeasonalEncoder(nn.Module):
    """Original seasonal TFB with optional frame-axis pooling."""

    def __init__(
            self, enc_in, kernel_num, individual, mode, seq_len, n_fft,
            pool_ratio, pool_mode, upsample_mode, dropout):
        super().__init__()
        self.enc = int(enc_in)
        self.n_fft = int(n_fft)
        self.n_freq = int(n_fft // 2 + 1)
        self.n_frames = int(seq_len // (n_fft * 0.5) + 1)
        self.pooled_frames = _resolve_pooled_frames(
            self.n_frames, pool_ratio)
        self.mode = mode

        if mode == "MK":
            self.wg1 = nn.Parameter(torch.empty(
                kernel_num, self.n_freq, self.pooled_frames,
                dtype=torch.complex64))
            self.wc = nn.Parameter(torch.empty(
                kernel_num, self.n_freq,
                self.pooled_frames, self.pooled_frames,
                dtype=torch.complex64))
            nn.init.xavier_normal_(self.wg1)
            nn.init.xavier_normal_(self.wc)
        elif mode == "IK":
            self.wc1 = nn.Parameter(torch.empty(
                self.enc, individual, dtype=torch.complex64))
            self.wc2 = nn.Parameter(torch.empty(
                individual, self.n_freq,
                self.pooled_frames, self.pooled_frames,
                dtype=torch.complex64))
            nn.init.xavier_normal_(self.wc1)
            nn.init.xavier_normal_(self.wc2)
        else:
            raise ValueError("mode must be MK or IK")

        self.wf1 = nn.Parameter(torch.empty(
            self.n_freq, self.n_freq, dtype=torch.complex64))
        self.bf1 = nn.Parameter(torch.empty(
            self.n_freq, 1, dtype=torch.complex64))
        nn.init.xavier_normal_(self.wf1)
        nn.init.xavier_normal_(self.bf1)

        self.pool = ComplexAdaptivePool1d(pool_mode)
        self.upsample = ComplexUpsample1d(upsample_mode)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, return_aux=False):
        window = torch.ones(
            self.n_fft, device=q.device, dtype=q.dtype)
        frames = torch.stft(
            q, n_fft=self.n_fft,
            hop_length=int(self.n_fft * 0.5), window=window,
            return_complex=True)
        full_frames = frames.shape[-1]
        frames = self.pool(frames, self.pooled_frames)

        if self.mode == "MK":
            gates = self.dropout(F.sigmoid(torch.abs(
                torch.einsum("bfn,kfn->bk", frames, self.wg1)))).cfloat()
            experts = torch.einsum(
                "bfn,kfno->bkfo", frames, self.wc)
            core = torch.einsum("bkfo,bk->bfo", experts, gates)
        else:
            batch = int(q.shape[0] / self.enc)
            frames = frames.reshape(
                batch, self.enc, self.n_freq, frames.shape[-1])
            kernel = torch.einsum(
                "ci,ifno->cfno", self.wc1, self.wc2)
            core = torch.einsum("bcfn,cfno->bcfo", frames, kernel)
            core = core.reshape(
                core.shape[0] * core.shape[1], self.n_freq, core.shape[-1])

        full_core = self.upsample(core, full_frames)
        transformed = torch.einsum(
            "bfn,fo->bon", core, self.wf1) + self.bf1
        pooled_spectrum = core + complex_tanh(transformed)
        spectrum = self.upsample(pooled_spectrum, full_frames)

        out = torch.istft(
            spectrum, n_fft=self.n_fft,
            hop_length=int(self.n_fft * 0.5), window=window, length=q.shape[-1])
        out = self.dropout(out)
        if return_aux:
            return out, {
                "coarse": core,
                "full_core": full_core,
                "spectrum": spectrum,
            }
        return out


class PooledTrendEncoder(nn.Module):
    """Original trend TFB with optional frame-axis pooling."""

    def __init__(
            self, seq_len, n_fft, pool_ratio, pool_mode, upsample_mode,
            dropout):
        super().__init__()
        self.n_fft = int(n_fft)
        self.n_freq = int(n_fft // 2 + 1)
        self.n_frames = int(seq_len // (n_fft * 0.5) + 1)
        self.pooled_frames = _resolve_pooled_frames(
            self.n_frames, pool_ratio)
        self.wc = nn.Parameter(torch.empty(
            self.n_freq, self.pooled_frames, self.pooled_frames,
            dtype=torch.complex64))
        self.wf1 = nn.Parameter(torch.empty(
            self.n_freq, self.n_freq, dtype=torch.complex64))
        self.bf1 = nn.Parameter(torch.empty(
            self.n_freq, 1, dtype=torch.complex64))
        nn.init.xavier_normal_(self.wc)
        nn.init.xavier_normal_(self.wf1)
        nn.init.xavier_normal_(self.bf1)
        self.pool = ComplexAdaptivePool1d(pool_mode)
        self.upsample = ComplexUpsample1d(upsample_mode)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, return_aux=False):
        window = torch.ones(
            self.n_fft, device=q.device, dtype=q.dtype)
        frames = torch.stft(
            q, n_fft=self.n_fft,
            hop_length=int(self.n_fft * 0.5), window=window,
            return_complex=True)
        full_frames = frames.shape[-1]
        frames = self.pool(frames, self.pooled_frames)
        core = torch.einsum("bfn,fno->bfo", frames, self.wc)
        full_core = self.upsample(core, full_frames)
        transformed = torch.einsum(
            "bfn,fo->bon", core, self.wf1) + self.bf1
        pooled_spectrum = core + complex_tanh(transformed)
        spectrum = self.upsample(pooled_spectrum, full_frames)
        out = torch.istft(
            spectrum, n_fft=self.n_fft,
            hop_length=int(self.n_fft * 0.5), window=window, length=q.shape[-1])
        out = self.dropout(out)
        if return_aux:
            return out, {
                "coarse": core,
                "full_core": full_core,
                "spectrum": spectrum,
            }
        return out


class Encoder(nn.Module):
    """Pooled seasonal/trend encoder at one STFT scale."""

    def __init__(
            self, enc_in, seq_len, kernel_num, individual, mode, n_fft,
            pool_ratios, pool_mode, upsample_mode, dropout):
        super().__init__()
        self.block_seasonal = PooledSeasonalEncoder(
            enc_in=enc_in,
            kernel_num=kernel_num,
            individual=individual,
            mode=mode,
            seq_len=seq_len,
            n_fft=n_fft,
            pool_ratio=pool_ratios[0],
            pool_mode=pool_mode,
            upsample_mode=upsample_mode,
            dropout=dropout)
        self.block_trend = PooledTrendEncoder(
            seq_len=seq_len,
            n_fft=n_fft,
            pool_ratio=pool_ratios[1],
            pool_mode=pool_mode,
            upsample_mode=upsample_mode,
            dropout=dropout)

    def forward(self, seasonal, trend, return_aux=False):
        if return_aux:
            seasonal_out, seasonal_aux = self.block_seasonal(
                seasonal, return_aux=True)
            trend_out, trend_aux = self.block_trend(
                trend, return_aux=True)
            return (seasonal_out, trend_out), {
                "branches": [seasonal_aux, trend_aux],
            }
        return self.block_seasonal(seasonal), self.block_trend(trend)
