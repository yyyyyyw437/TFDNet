"""Tri-band TFDNet blocks with frame-axis pooling before TFB kernels."""
import math

import torch
import torch.nn.functional as F
from torch import nn


def complex_tanh(x):
    return torch.complex(torch.tanh(x.real), torch.tanh(x.imag))


def _complex_normal_(tensor, std):
    with torch.no_grad():
        tensor.real.normal_(0.0, std)
        tensor.imag.normal_(0.0, std)
    return tensor


def _odd_kernel_size(size, max_size=None):
    size = int(max(1, size))
    if max_size is not None:
        size = min(size, int(max_size))
    if size % 2 == 0:
        size -= 1
    return max(1, size)


class ComplexDropout(nn.Module):
    """Apply one dropout mask to both real and imaginary components."""

    def __init__(self, p=0.0):
        super().__init__()
        if p < 0.0 or p > 1.0:
            raise ValueError("dropout probability must be in [0, 1]")
        self.p = float(p)

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        if self.p == 1.0:
            mask = torch.zeros_like(x.real)
        else:
            keep_prob = 1.0 - self.p
            mask = torch.empty_like(x.real).bernoulli_(keep_prob)
            mask.div_(keep_prob)
        return torch.complex(x.real * mask, x.imag * mask)


def _resolve_pooled_frames(n_frames, pool_ratio, pool_target):
    """Resolve an absolute target or a ratio of the actual frame count."""
    n_frames = max(1, int(n_frames))
    pool_target = int(pool_target)
    if pool_target > 0:
        return min(n_frames, max(1, pool_target))

    pool_ratio = float(pool_ratio)
    if 0.0 < pool_ratio < 1.0:
        target = int(math.floor(n_frames * pool_ratio))
        return min(n_frames, max(1, target))
    return n_frames


class MovingAverage(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = _odd_kernel_size(kernel_size)
        self.pad = (self.kernel_size - 1) // 2
        self.avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1)

    def forward(self, x):
        if self.kernel_size == 1:
            return x
        front = x[:, :1, :].repeat(1, self.pad, 1)
        end = x[:, -1:, :].repeat(1, self.pad, 1)
        x = torch.cat([front, x, end], dim=1).permute(0, 2, 1)
        return self.avg(x).permute(0, 2, 1)


class TriBandDecomposition(nn.Module):
    """Additive low/mid/high decomposition."""

    def __init__(self, seq_len, low_kernel=25, mid_kernel=5):
        super().__init__()
        max_kernel = seq_len if seq_len % 2 == 1 else max(1, seq_len - 1)
        low_kernel = _odd_kernel_size(low_kernel, max_kernel)
        mid_kernel = _odd_kernel_size(mid_kernel, max_kernel)
        if mid_kernel >= low_kernel:
            mid_kernel = max(1, low_kernel - 2)
        self.low_ma = MovingAverage(low_kernel)
        self.mid_ma = MovingAverage(mid_kernel)

    def forward(self, x):
        low = self.low_ma(x)
        mid_anchor = self.mid_ma(x)
        return low, mid_anchor - low, x - mid_anchor


class ComplexAdaptivePool1d(nn.Module):
    """Pool the STFT frame axis while keeping complex coefficients."""

    def __init__(self, mode="avg"):
        super().__init__()
        if mode not in ("avg", "energy"):
            raise ValueError("pool_mode must be 'avg' or 'energy'")
        self.mode = mode

    def forward(self, x, target_frames):
        # x: [B, M, N], complex
        n_frames = x.shape[-1]
        target_frames = min(int(target_frames), n_frames)
        if target_frames == n_frames:
            return x
        batch, n_freq, _ = x.shape
        real = x.real.reshape(batch * n_freq, 1, n_frames)
        imag = x.imag.reshape(batch * n_freq, 1, n_frames)
        if self.mode == "avg":
            real = F.adaptive_avg_pool1d(real, target_frames)
            imag = F.adaptive_avg_pool1d(imag, target_frames)
        else:
            energy = real.square() + imag.square()
            pooled_real, pooled_imag = [], []
            for index in range(target_frames):
                start = int(math.floor(index * n_frames / target_frames))
                end = int(math.ceil((index + 1) * n_frames / target_frames))
                weights = torch.softmax(energy[..., start:end], dim=-1)
                pooled_real.append((weights * real[..., start:end]).sum(-1))
                pooled_imag.append((weights * imag[..., start:end]).sum(-1))
            real = torch.stack(pooled_real, dim=-1)
            imag = torch.stack(pooled_imag, dim=-1)
        real = real.reshape(batch, n_freq, target_frames)
        imag = imag.reshape(batch, n_freq, target_frames)
        return torch.complex(real, imag)


class ComplexUpsample1d(nn.Module):
    """Restore the frame axis before inverse STFT."""

    def __init__(self, mode="linear"):
        super().__init__()
        if mode not in ("linear", "nearest"):
            raise ValueError("upsample_mode must be 'linear' or 'nearest'")
        self.mode = mode

    def forward(self, x, target_frames):
        n_frames = x.shape[-1]
        if n_frames == target_frames:
            return x
        batch, n_freq, _ = x.shape
        real = x.real.reshape(batch * n_freq, 1, n_frames)
        imag = x.imag.reshape(batch * n_freq, 1, n_frames)
        kwargs = {"size": int(target_frames), "mode": self.mode}
        if self.mode == "linear":
            kwargs["align_corners"] = False
        real = F.interpolate(real, **kwargs)
        imag = F.interpolate(imag, **kwargs)
        real = real.reshape(batch, n_freq, target_frames)
        imag = imag.reshape(batch, n_freq, target_frames)
        return torch.complex(real, imag)


class ComplexFullResolutionResidual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight_real = nn.Parameter(torch.ones(1))
        self.weight_imag = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x * torch.complex(self.weight_real, self.weight_imag)


class PooledTFB(nn.Module):
    """Complex frame-mixing kernel at the pooled frame resolution."""

    def __init__(self, n_channels, n_freq, n_frames, mode="shared",
                 kernel_num=4, individual_factor=7):
        super().__init__()
        self.mode = mode
        self.n_channels = int(n_channels)
        self.n_freq = int(n_freq)
        self.n_frames = int(n_frames)
        std = 1.0 / math.sqrt(2.0 * max(1, self.n_frames))

        if mode == "shared":
            self.kernel = nn.Parameter(
                torch.empty(n_freq, n_frames, n_frames,
                            dtype=torch.complex64))
            _complex_normal_(self.kernel, std)
        elif mode == "mk":
            self.kernel_num = int(kernel_num)
            self.wg = nn.Parameter(
                torch.empty(kernel_num, n_freq, n_frames,
                            dtype=torch.complex64))
            self.kernel = nn.Parameter(
                torch.empty(kernel_num, n_freq, n_frames, n_frames,
                            dtype=torch.complex64))
            _complex_normal_(self.wg, std)
            _complex_normal_(self.kernel, std)
        elif mode == "ik":
            self.rank = int(max(1, individual_factor))
            self.wc1 = nn.Parameter(
                torch.empty(n_channels, self.rank, dtype=torch.complex64))
            self.wc2 = nn.Parameter(
                torch.empty(self.rank, n_freq, n_frames, n_frames,
                            dtype=torch.complex64))
            _complex_normal_(self.wc1, 1.0 / math.sqrt(2.0 * n_channels))
            _complex_normal_(self.wc2, std)
        else:
            raise ValueError("Unsupported TFB mode: {}".format(mode))

    def forward(self, x):
        # x: [B, M, N], complex
        if self.mode == "shared":
            return torch.einsum("bmi,mio->bmo", x, self.kernel)
        if self.mode == "mk":
            gates = torch.sigmoid(torch.abs(torch.einsum("bmi,kmi->bk", x, self.wg)))
            gates = gates.to(torch.complex64)
            experts = torch.einsum("bmi,kmio->bkmo", x, self.kernel)
            return torch.einsum("bkmo,bk->bmo", experts, gates)

        batch_frames = x.shape[0]
        batch = batch_frames // self.n_channels
        x = x.reshape(batch, self.n_channels, self.n_freq, self.n_frames)
        effective_real = (
            torch.einsum("df,fmio->dmio", self.wc1.real, self.wc2.real)
            - torch.einsum("df,fmio->dmio", self.wc1.imag, self.wc2.imag)
        )
        effective_imag = (
            torch.einsum("df,fmio->dmio", self.wc1.real, self.wc2.imag)
            + torch.einsum("df,fmio->dmio", self.wc1.imag, self.wc2.real)
        )
        effective = torch.complex(effective_real, effective_imag)
        out = torch.einsum("bdmi,dmio->bdmo", x, effective)
        return out.reshape(batch_frames, self.n_freq, self.n_frames)


class ComplexDepthwiseTemporalConv(nn.Module):
    """Local complex convolution along the frame axis."""

    def __init__(self, n_freq, kernel_size=3, dropout=0.0):
        super().__init__()
        kernel_size = _odd_kernel_size(kernel_size)
        self.pad = (kernel_size - 1) // 2
        self.conv_real = nn.Conv1d(
            n_freq, n_freq, kernel_size=kernel_size, padding=0, groups=n_freq)
        self.conv_imag = nn.Conv1d(
            n_freq, n_freq, kernel_size=kernel_size, padding=0, groups=n_freq)
        self.dropout = ComplexDropout(dropout)

    def forward(self, x):
        real = F.pad(x.real, (self.pad, self.pad))
        imag = F.pad(x.imag, (self.pad, self.pad))
        out_real = self.conv_real(real) - self.conv_imag(imag)
        out_imag = self.conv_real(imag) + self.conv_imag(real)
        out = torch.complex(out_real, out_imag)
        return x + self.dropout(complex_tanh(out))


class ComplexFrequencyFFN(nn.Module):
    def __init__(self, n_freq, dropout=0.0):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(n_freq, n_freq, dtype=torch.complex64))
        self.bias = nn.Parameter(torch.empty(n_freq, 1, dtype=torch.complex64))
        _complex_normal_(self.weight, 1.0 / math.sqrt(2.0 * n_freq))
        _complex_normal_(self.bias, 1.0 / math.sqrt(2.0 * n_freq))
        self.dropout = ComplexDropout(dropout)

    def forward(self, x):
        transformed = torch.einsum("bmi,mo->boi", x, self.weight) + self.bias
        return x + self.dropout(complex_tanh(transformed))


class BandTFEncoder(nn.Module):
    """STFT -> pooled TFB -> upsample -> residual -> frequency FFN -> iSTFT."""

    def __init__(self, n_channels, seq_len, n_fft, band, use_pool=True,
                 pool_ratio=0.0, pool_target=0, pool_mode="avg",
                 upsample_mode="linear",
                 kernel_num=4, individual_factor=7, mode="mk",
                 high_kernel=3, dropout=0.0):
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop = int(n_fft // 2)
        self.n_freq = int(n_fft // 2 + 1)
        self.n_frames = int(seq_len // self.hop + 1)
        self.use_pool = bool(use_pool)
        self.band = band
        self.pool_ratio = float(pool_ratio)
        self.pool_target = int(pool_target)
        self.pooled_frames = (
            self.n_frames if not self.use_pool else
            _resolve_pooled_frames(
                self.n_frames, self.pool_ratio, self.pool_target))
        self.pool = ComplexAdaptivePool1d(pool_mode)
        self.upsample = ComplexUpsample1d(upsample_mode)

        if band == "low":
            self.core = PooledTFB(
                n_channels, self.n_freq, self.pooled_frames, mode="shared")
        elif band == "mid":
            self.core = PooledTFB(
                n_channels, self.n_freq, self.pooled_frames,
                mode=mode.lower(), kernel_num=kernel_num,
                individual_factor=individual_factor)
        elif band == "high":
            self.core = ComplexDepthwiseTemporalConv(
                self.n_freq, kernel_size=high_kernel, dropout=dropout)
        else:
            raise ValueError("band must be low, mid, or high")

        self.residual = ComplexFullResolutionResidual(n_channels)
        self.frequency_ffn = ComplexFrequencyFFN(self.n_freq, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_aux=False):
        # x: [B * C, L]
        window = torch.ones(
            self.n_fft, device=x.device, dtype=x.dtype)
        frames = torch.stft(
            x, n_fft=self.n_fft, hop_length=self.hop, window=window,
            return_complex=True)
        n_frames = frames.shape[-1]
        pooled_core = self.core(self.pool(frames, self.pooled_frames))
        full_core = self.upsample(pooled_core, n_frames)
        full_core = full_core + self.residual(frames)
        spectrum = self.frequency_ffn(full_core)
        out = torch.istft(
            spectrum, n_fft=self.n_fft, hop_length=self.hop, window=window,
            length=x.shape[-1])
        out = self.dropout(out)
        if return_aux:
            return out, {
                "coarse": pooled_core,
                "full_core": full_core,
                "spectrum": spectrum,
            }
        return out


class TriScaleEncoder(nn.Module):
    """Three band-aware encoders at one STFT scale."""

    def __init__(self, n_channels, seq_len, n_fft, use_pool,
                 pool_ratios, pool_targets, pool_mode, upsample_mode, kernel_num,
                 individual_factor, mode, high_kernel, dropout):
        super().__init__()
        self.low = BandTFEncoder(
            n_channels=n_channels, seq_len=seq_len, n_fft=n_fft, band="low",
            use_pool=use_pool, pool_ratio=pool_ratios[0],
            pool_target=pool_targets[0], pool_mode=pool_mode,
            upsample_mode=upsample_mode, kernel_num=kernel_num,
            individual_factor=individual_factor, mode=mode,
            high_kernel=high_kernel, dropout=dropout)
        self.mid = BandTFEncoder(
            n_channels=n_channels, seq_len=seq_len, n_fft=n_fft, band="mid",
            use_pool=use_pool, pool_ratio=pool_ratios[1],
            pool_target=pool_targets[1], pool_mode=pool_mode,
            upsample_mode=upsample_mode, kernel_num=kernel_num,
            individual_factor=individual_factor, mode=mode,
            high_kernel=high_kernel, dropout=dropout)
        self.high = BandTFEncoder(
            n_channels=n_channels, seq_len=seq_len, n_fft=n_fft, band="high",
            use_pool=use_pool, pool_ratio=pool_ratios[2],
            pool_target=pool_targets[2], pool_mode=pool_mode,
            upsample_mode=upsample_mode, kernel_num=kernel_num,
            individual_factor=individual_factor, mode=mode,
            high_kernel=high_kernel, dropout=dropout)

    def forward(self, low, mid, high, return_aux=False):
        modules = (self.low, self.mid, self.high)
        inputs = (low, mid, high)
        if not return_aux:
            return tuple(module(value) for module, value in zip(modules, inputs))

        outputs = []
        aux = {"coarse": [], "full_core": [], "spectrum": []}
        for module, value in zip(modules, inputs):
            output, band_aux = module(value, return_aux=True)
            outputs.append(output)
            for key in aux:
                aux[key].append(band_aux[key])
        return tuple(outputs), aux
