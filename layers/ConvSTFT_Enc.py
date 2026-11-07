"""TFDNet encoders with multi-scale complex dilated convolution TFB."""
import torch
import torch.nn.functional as F
from torch import nn


def complex_tanh(x):
    return torch.complex(torch.tanh(x.real), torch.tanh(x.imag))


def _odd_kernel_size(size):
    size = max(1, int(size))
    return size if size % 2 == 1 else size - 1


class ComplexDropout(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        if p < 0.0 or p > 1.0:
            raise ValueError("dropout probability must be in [0, 1]")
        self.p = float(p)

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        if keep == 0.0:
            mask = torch.zeros_like(x.real)
        else:
            mask = torch.empty_like(x.real).bernoulli_(keep).div_(keep)
        return torch.complex(x.real * mask, x.imag * mask)


class ComplexConvBank(nn.Module):
    """Fuse several complex depthwise temporal convolutions."""

    def __init__(
            self, n_freq, n_frames, n_channels, per_channel, kernels,
            dilations, dropout):
        super().__init__()
        if len(kernels) != len(dilations) or len(kernels) == 0:
            raise ValueError(
                "conv kernels and dilations must have equal nonzero length")
        self.n_freq = int(n_freq)
        self.n_frames = int(n_frames)
        self.n_channels = int(n_channels)
        self.per_channel = bool(per_channel)
        self.kernels = [_odd_kernel_size(value) for value in kernels]
        self.dilations = [int(value) for value in dilations]
        if any(value <= 0 for value in self.dilations):
            raise ValueError("conv dilations must be positive")

        n_groups = self.n_freq
        if self.per_channel:
            n_groups *= self.n_channels
        self.groups = n_groups
        self.conv_real = nn.ModuleList()
        self.conv_imag = nn.ModuleList()
        for kernel, dilation in zip(self.kernels, self.dilations):
            self.conv_real.append(nn.Conv1d(
                n_groups, n_groups, kernel_size=kernel,
                dilation=dilation, groups=n_groups, padding=0, bias=False))
            self.conv_imag.append(nn.Conv1d(
                n_groups, n_groups, kernel_size=kernel,
                dilation=dilation, groups=n_groups, padding=0, bias=False))
        for module in list(self.conv_real) + list(self.conv_imag):
            nn.init.xavier_normal_(module.weight)

        self.gate = nn.Parameter(torch.empty(
            len(self.kernels), self.n_freq, self.n_frames,
            dtype=torch.complex64))
        nn.init.xavier_normal_(self.gate)
        self.dropout = ComplexDropout(dropout)
        self.last_gates = None

    def _conv_branch(self, x, index):
        kernel = self.kernels[index]
        dilation = self.dilations[index]
        padding = dilation * (kernel - 1) // 2
        combined = torch.stack([x.real, x.imag], dim=0)
        combined = combined.reshape(
            combined.shape[0] * combined.shape[1],
            combined.shape[2], combined.shape[3])
        combined = F.pad(combined, (padding, padding))
        conv_real = F.conv1d(
            combined, self.conv_real[index].weight,
            dilation=dilation, groups=self.groups)
        conv_imag = F.conv1d(
            combined, self.conv_imag[index].weight,
            dilation=dilation, groups=self.groups)
        real_part, imag_part = conv_real.chunk(2, dim=0)
        cross_real, cross_imag = conv_imag.chunk(2, dim=0)
        out_real = real_part - cross_imag
        out_imag = imag_part + cross_real
        return torch.complex(out_real, out_imag)

    def forward(self, x):
        # x: [B * C, F, N]
        gates = torch.sigmoid(torch.abs(torch.einsum(
            "bfn,kfn->bk", x, self.gate)))
        self.last_gates = gates.detach()

        if self.per_channel:
            batch_frames, n_freq, n_frames = x.shape
            batch = batch_frames // self.n_channels
            grouped = x.reshape(
                batch, self.n_channels, n_freq, n_frames)
            grouped = grouped.reshape(
                batch, self.n_channels * n_freq, n_frames)
            conv_input = grouped
            out = 0
            gates = gates.reshape(batch, self.n_channels, -1)
            for index in range(len(self.kernels)):
                branch = self._conv_branch(conv_input, index)
                branch = branch.reshape(
                    batch, self.n_channels, n_freq, n_frames)
                gate = gates[:, :, index].view(
                    batch, self.n_channels, 1, 1)
                out = out + branch * gate
            out = out.reshape(batch_frames, n_freq, n_frames)
        else:
            conv_input = x
            out = 0
            for index in range(len(self.kernels)):
                branch = self._conv_branch(conv_input, index)
                out = out + branch * gates[:, index].view(-1, 1, 1)
        return self.dropout(out)


class ConvSeasonalEncoder(nn.Module):
    def __init__(
            self, enc_in, mode, seq_len, n_fft, kernels, dilations,
            dropout):
        super().__init__()
        self.enc = int(enc_in)
        self.n_fft = int(n_fft)
        self.n_freq = int(n_fft // 2 + 1)
        self.n_frames = int(seq_len // (n_fft * 0.5) + 1)
        self.conv = ComplexConvBank(
            self.n_freq,
            self.n_frames,
            self.enc,
            per_channel=(mode == "IK"),
            kernels=kernels,
            dilations=dilations,
            dropout=dropout)
        self.wf1 = nn.Parameter(torch.empty(
            self.n_freq, self.n_freq, dtype=torch.complex64))
        self.bf1 = nn.Parameter(torch.empty(
            self.n_freq, 1, dtype=torch.complex64))
        nn.init.xavier_normal_(self.wf1)
        nn.init.xavier_normal_(self.bf1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q):
        window = torch.ones(
            self.n_fft, device=q.device, dtype=q.dtype)
        frames = torch.stft(
            q, n_fft=self.n_fft,
            hop_length=int(self.n_fft * 0.5), window=window,
            return_complex=True)
        core = self.conv(frames)
        transformed = torch.einsum(
            "bfn,fo->bon", core, self.wf1) + self.bf1
        out = core + complex_tanh(transformed)
        out = torch.istft(
            out, n_fft=self.n_fft,
            hop_length=int(self.n_fft * 0.5), window=window,
            length=q.shape[-1])
        return self.dropout(out)


class ConvTrendEncoder(nn.Module):
    def __init__(
            self, enc_in, seq_len, n_fft, kernels, dilations, dropout):
        super().__init__()
        self.n_fft = int(n_fft)
        self.n_freq = int(n_fft // 2 + 1)
        self.n_frames = int(seq_len // (n_fft * 0.5) + 1)
        self.conv = ComplexConvBank(
            self.n_freq,
            self.n_frames,
            int(enc_in),
            per_channel=False,
            kernels=kernels,
            dilations=dilations,
            dropout=dropout)
        self.wf1 = nn.Parameter(torch.empty(
            self.n_freq, self.n_freq, dtype=torch.complex64))
        self.bf1 = nn.Parameter(torch.empty(
            self.n_freq, 1, dtype=torch.complex64))
        nn.init.xavier_normal_(self.wf1)
        nn.init.xavier_normal_(self.bf1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q):
        window = torch.ones(
            self.n_fft, device=q.device, dtype=q.dtype)
        frames = torch.stft(
            q, n_fft=self.n_fft,
            hop_length=int(self.n_fft * 0.5), window=window,
            return_complex=True)
        core = self.conv(frames)
        transformed = torch.einsum(
            "bfn,fo->bon", core, self.wf1) + self.bf1
        out = core + complex_tanh(transformed)
        out = torch.istft(
            out, n_fft=self.n_fft,
            hop_length=int(self.n_fft * 0.5), window=window,
            length=q.shape[-1])
        return self.dropout(out)


class Encoder(nn.Module):
    def __init__(
            self, enc_in, seq_len, mode, n_fft, kernels, dilations,
            dropout):
        super().__init__()
        self.block_seasonal = ConvSeasonalEncoder(
            enc_in=enc_in,
            mode=mode,
            seq_len=seq_len,
            n_fft=n_fft,
            kernels=kernels,
            dilations=dilations,
            dropout=dropout)
        self.block_trend = ConvTrendEncoder(
            enc_in=enc_in,
            seq_len=seq_len,
            n_fft=n_fft,
            kernels=kernels,
            dilations=dilations,
            dropout=dropout)

    def forward(self, seasonal, trend):
        return self.block_seasonal(seasonal), self.block_trend(trend)
