"""TFDNet seasonal/trend encoders with low-rank temporal TFB kernels."""
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


class LowRankSeasonalEncoder(nn.Module):
    """Seasonal TFB with dense N x N temporal kernels replaced by two factors."""

    def __init__(
            self, enc_in, kernel_num, individual, mode, seq_len, n_fft,
            rank, dropout):
        super().__init__()
        self.enc = int(enc_in)
        self.n_fft = int(n_fft)
        self.n_freq = int(n_fft // 2 + 1)
        self.n_frames = int(seq_len // (n_fft * 0.5) + 1)
        self.rank = max(1, min(int(rank), self.n_frames))
        self.mode = mode

        if mode == "MK":
            self.wg1 = nn.Parameter(torch.empty(
                kernel_num, self.n_freq, self.n_frames,
                dtype=torch.complex64))
            self.upper = nn.Parameter(torch.empty(
                kernel_num, self.n_freq, self.n_frames, self.rank,
                dtype=torch.complex64))
            self.lower = nn.Parameter(torch.empty(
                kernel_num, self.n_freq, self.rank, self.n_frames,
                dtype=torch.complex64))
            nn.init.xavier_normal_(self.wg1)
        elif mode == "IK":
            self.wc1 = nn.Parameter(torch.empty(
                self.enc, individual, dtype=torch.complex64))
            self.upper = nn.Parameter(torch.empty(
                individual, self.n_freq, self.n_frames, self.rank,
                dtype=torch.complex64))
            self.lower = nn.Parameter(torch.empty(
                individual, self.n_freq, self.rank, self.n_frames,
                dtype=torch.complex64))
            nn.init.xavier_normal_(self.wc1)
        else:
            raise ValueError("mode must be MK or IK")

        _complex_normal_(
            self.upper, 1.0 / math.sqrt(2.0 * self.n_frames))
        _complex_normal_(
            self.lower, 1.0 / math.sqrt(2.0 * self.rank))
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

        if self.mode == "MK":
            gates = self.dropout(F.sigmoid(torch.abs(
                torch.einsum("bfn,kfn->bk", frames, self.wg1)))).cfloat()
            latent = torch.einsum(
                "bfn,kfnr->bkfr", frames, self.upper)
            experts = torch.einsum(
                "bkfr,kfro->bkfo", latent, self.lower)
            out = torch.einsum("bkfo,bk->bfo", experts, gates)
        else:
            batch = int(q.shape[0] / self.enc)
            frames = frames.reshape(
                batch, self.enc, self.n_freq, self.n_frames)
            latent = torch.einsum(
                "bcfn,ifnr->bcfir", frames, self.upper)
            basis = torch.einsum(
                "bcfir,ifro->bcfio", latent, self.lower)
            out = torch.einsum("bcfio,ci->bcfo", basis, self.wc1)
            out = out.reshape(
                out.shape[0] * out.shape[1], self.n_freq, self.n_frames)

        out_ = out
        transformed = torch.einsum(
            "bfn,fo->bon", out, self.wf1) + self.bf1
        out = out_ + complex_tanh(transformed)
        out = torch.istft(
            out, n_fft=self.n_fft,
            hop_length=int(self.n_fft * 0.5), window=window,
            length=q.shape[-1])
        return self.dropout(out)


class LowRankTrendEncoder(nn.Module):
    """Trend TFB with a low-rank temporal frequency kernel."""

    def __init__(self, seq_len, n_fft, rank, dropout):
        super().__init__()
        self.n_fft = int(n_fft)
        self.n_freq = int(n_fft // 2 + 1)
        self.n_frames = int(seq_len // (n_fft * 0.5) + 1)
        self.rank = max(1, min(int(rank), self.n_frames))
        self.upper = nn.Parameter(torch.empty(
            self.n_freq, self.n_frames, self.rank,
            dtype=torch.complex64))
        self.lower = nn.Parameter(torch.empty(
            self.n_freq, self.rank, self.n_frames,
            dtype=torch.complex64))
        _complex_normal_(
            self.upper, 1.0 / math.sqrt(2.0 * self.n_frames))
        _complex_normal_(
            self.lower, 1.0 / math.sqrt(2.0 * self.rank))
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
        latent = torch.einsum(
            "bfn,fnr->bfr", frames, self.upper)
        core = torch.einsum(
            "bfr,fro->bfo", latent, self.lower)
        transformed = torch.einsum(
            "bfn,fo->bon", core, self.wf1) + self.bf1
        out = core + complex_tanh(transformed)
        out = torch.istft(
            out, n_fft=self.n_fft,
            hop_length=int(self.n_fft * 0.5), window=window,
            length=q.shape[-1])
        return self.dropout(out)


class Encoder(nn.Module):
    """Low-rank seasonal/trend encoder at one STFT scale."""

    def __init__(
            self, enc_in, seq_len, kernel_num, individual, mode, n_fft,
            rank, dropout):
        super().__init__()
        self.block_seasonal = LowRankSeasonalEncoder(
            enc_in=enc_in,
            kernel_num=kernel_num,
            individual=individual,
            mode=mode,
            seq_len=seq_len,
            n_fft=n_fft,
            rank=rank,
            dropout=dropout)
        self.block_trend = LowRankTrendEncoder(
            seq_len=seq_len,
            n_fft=n_fft,
            rank=rank,
            dropout=dropout)

    def forward(self, seasonal, trend):
        return self.block_seasonal(seasonal), self.block_trend(trend)
