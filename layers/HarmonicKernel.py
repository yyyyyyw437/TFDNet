"""Harmonic-structured parameterisation of the time-frequency kernels.

The original TFB learns M independent N x N kernels (one per frequency bin).
Here the M slices are generated from a small set of shared basis kernels, tied
together by a harmonic (integer-multiple) affinity:

    a[m, j]   = sum_h exp(-(m - h*m0_j)^2 / (2*(alpha*m0_j)^2))     j = 1..R
    a[m, 0]   = beta                                                (free slot)
    a[m, :]  /= a[m, :].sum()                                       (row-normalised)
    W[m]      = sum_j a[m, j] * B[j]

with

    m0_j = n_fft / P_j          (P_j = fundamental period in *samples*)

P_j is parameterised in physical units on purpose: the same period then maps to
a different bin index at each STFT scale S, so one learned value is shared by
all scales.  Only {B, P, alpha, beta} are learned; the affinity is computed.

This module never mixes frequency bins: it only decides *how each bin's kernel
is generated*.  Cross-bin information flow is still the job of Frequency-FFN.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F


def _inv_softplus(y: float) -> float:
    y = max(float(y), 1e-6)
    return math.log(math.expm1(y))


def complex_normal_(t: torch.Tensor, std: float) -> torch.Tensor:
    """Version-independent complex init (normal_ on complex is not portable)."""
    with torch.no_grad():
        t.real.normal_(0.0, std)
        t.imag.normal_(0.0, std)
    return t


class HarmonicAffinity(nn.Module):
    """Compute the (M, R+1) row-stochastic affinity matrix."""

    def __init__(self, n_freq, n_fft, periods, alpha=0.15, beta=0.05,
                 learn_period=True, learn_alpha=True, learn_beta=True,
                 max_harmonics=512):
        super().__init__()
        self.n_freq = int(n_freq)
        self.n_fft = float(n_fft)

        p = torch.as_tensor(list(periods), dtype=torch.float32).flatten().clamp(min=2.0)
        self.n_fam = int(p.numel())

        logp = torch.log(p)
        if learn_period:
            self.log_period = nn.Parameter(logp.clone())
        else:
            self.register_buffer("log_period", logp.clone())

        a0, b0 = _inv_softplus(alpha), _inv_softplus(beta)
        if learn_alpha:
            self.alpha_raw = nn.Parameter(torch.tensor([a0]))
        else:
            self.register_buffer("alpha_raw", torch.tensor([a0]))
        if learn_beta:
            self.beta_raw = nn.Parameter(torch.tensor([b0]))
        else:
            self.register_buffer("beta_raw", torch.tensor([b0]))

        if self.n_fam > 0:
            with torch.no_grad():
                m0_min = float((self.n_fft / p).min().clamp(min=1e-4))
            h_max = int(math.ceil((self.n_freq - 1) / m0_min)) + 2
            h_max = int(min(max(h_max, 8), max_harmonics))
        else:
            h_max = 1
        self.register_buffer("harmonics", torch.arange(1, h_max + 1, dtype=torch.float32))

    # ------------------------------------------------------------------ #
    @property
    def periods(self):
        """Fundamental periods, in samples. Always > 2."""
        return self.log_period.exp().clamp(min=2.0, max=1e5)

    @property
    def alpha(self):
        """Comb width, relative to the harmonic spacing (= m0)."""
        return F.softplus(self.alpha_raw) + 1e-4

    @property
    def beta(self):
        """Baseline weight of the free slot (fallback for non-harmonic bins)."""
        return F.softplus(self.beta_raw) + 1e-4

    def m0(self):
        """Bin index of each fundamental: m0 = n_fft / P."""
        return self.n_fft / self.periods

    def affinity(self):
        M, R = self.n_freq, self.n_fam
        dev = self.harmonics.device
        if R == 0:
            return torch.ones(M, 1, device=dev)

        m = torch.arange(M, device=dev, dtype=torch.float32).view(M, 1, 1)
        m0 = self.m0().view(1, R, 1)
        width = self.alpha.view(1, 1, 1) * m0
        h = self.harmonics.view(1, 1, -1)
        d = m - h * m0
        a = torch.exp(-(d ** 2) / (2.0 * width ** 2 + 1e-12)).sum(-1)      # (M,R)
        w = torch.cat([self.beta.expand(M, 1), a], dim=1)                 # (M,R+1)
        return w / w.sum(dim=1, keepdim=True).clamp_min(1e-6)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def readout(self):
        a = self.affinity().detach().cpu()
        return {
            "periods_samples": self.periods.detach().cpu().tolist(),
            "m0_bin_index": self.m0().detach().cpu().tolist(),
            "alpha": float(self.alpha.detach().cpu()),
            "beta": float(self.beta.detach().cpu()),
            "affinity": a.numpy(),
        }

    def extra_repr(self):
        return (f"n_freq={self.n_freq}, n_fft={int(self.n_fft)}, "
                f"families={self.n_fam}, harmonics={self.harmonics.numel()}")


class HarmonicBank(nn.Module):
    """`n_outer` kernel banks, each harmonic-parameterised over the M bins.

    n_outer = 1  -> trend encoder (one bank shared by all channels)
    n_outer = K  -> Seasonal-TFB-MK (K shared kernels)
    """

    def __init__(self, n_outer, n_freq, n_win, n_fft, periods,
                 alpha=0.15, beta=0.05, learn_period=True):
        super().__init__()
        self.n_outer = int(n_outer)
        self.n_freq = int(n_freq)
        self.n_win = int(n_win)
        self.aff = HarmonicAffinity(n_freq=n_freq, n_fft=n_fft, periods=periods,
                                    alpha=alpha, beta=beta, learn_period=learn_period)
        n_basis = self.aff.n_fam + 1
        self.bases = nn.Parameter(
            torch.empty(self.n_outer, n_basis, n_win, n_win, dtype=torch.complex64))
        complex_normal_(self.bases, self._init_std())

    def _init_std(self):
        """Keep E||W_eff||^2 equal to the original (independent-kernel) init."""
        with torch.no_grad():
            spread = float((self.aff.affinity() ** 2).sum(1).mean().clamp(min=1e-6))
        return 1.0 / math.sqrt(self.n_win * spread)

    def effective(self):
        a = self.aff.affinity()                                            # (M,R+1) real
        re = torch.einsum("mj,kjpq->kmpq", a, self.bases.real)
        im = torch.einsum("mj,kjpq->kmpq", a, self.bases.imag)
        return torch.complex(re, im)                                       # (K,M,N,N)

    def extra_repr(self):
        return f"outer={self.n_outer}, bases={tuple(self.bases.shape)}, {self.aff}"


class HarmonicBankIK(nn.Module):
    """Seasonal-TFB-IK with harmonic structure on top of the channel low-rank.

    W_ind(i)[m] = sum_f W1[i,f] * sum_j a[m,j] * B2[f,j]
    -> two orthogonal factorisations: rank-I over channels, harmonic over bins.
    """

    def __init__(self, n_ch, n_rank, n_freq, n_win, n_fft, periods,
                 alpha=0.15, beta=0.05, learn_period=True):
        super().__init__()
        self.n_ch = int(n_ch)
        self.n_rank = int(n_rank)
        self.n_win = int(n_win)
        self.aff = HarmonicAffinity(n_freq=n_freq, n_fft=n_fft, periods=periods,
                                    alpha=alpha, beta=beta, learn_period=learn_period)
        n_basis = self.aff.n_fam + 1
        self.wc1 = nn.Parameter(torch.empty(n_ch, n_rank, dtype=torch.complex64))
        self.b2 = nn.Parameter(
            torch.empty(n_rank, n_basis, n_win, n_win, dtype=torch.complex64))
        std1 = math.sqrt(2.0 / (n_ch + n_rank))
        complex_normal_(self.wc1, std1)
        with torch.no_grad():
            spread = float((self.aff.affinity() ** 2).sum(1).mean().clamp(min=1e-6))
        std2 = math.sqrt((n_ch + n_rank) / (4.0 * n_win * n_rank * spread))
        complex_normal_(self.b2, std2)

    def effective(self):
        a = self.aff.affinity()                                            # (M,R+1) real
        tr = torch.einsum("mj,fjpq->fmpq", a, self.b2.real)
        ti = torch.einsum("mj,fjpq->fmpq", a, self.b2.imag)
        tmp = torch.complex(tr, ti)                                        # (I,M,N,N)
        rr = torch.einsum("df,fmpq->dmpq", self.wc1.real, tmp.real) - \
             torch.einsum("df,fmpq->dmpq", self.wc1.imag, tmp.imag)
        ri = torch.einsum("df,fmpq->dmpq", self.wc1.real, tmp.imag) + \
             torch.einsum("df,fmpq->dmpq", self.wc1.imag, tmp.real)
        return torch.complex(rr, ri)                                       # (D,M,N,N)

    def extra_repr(self):
        return f"ch={self.n_ch}, rank={self.n_rank}, {self.aff}"


def harmonic_readout(model):
    """Collect every HarmonicAffinity in a model (for logging / plots)."""
    out = {}
    for name, mod in model.named_modules():
        if isinstance(mod, HarmonicAffinity):
            out[name] = mod.readout()
    return out


def summarise_harmonic(model, tag=""):
    """Human readable one-block summary; safe to call on any model."""
    rows = []
    for name, rec in harmonic_readout(model).items():
        P = ", ".join(f"{p:.2f}" for p in rec["periods_samples"])
        m0 = ", ".join(f"{v:.2f}" for v in rec["m0_bin_index"])
        rows.append(f"[{tag}] {name}: P=[{P}] samples | m0=[{m0}] bins | "
                    f"alpha={rec['alpha']:.3f} | beta={rec['beta']:.3f}")
    return "\n".join(rows) if rows else f"[{tag}] no harmonic module"
