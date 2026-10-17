import torch
from torch import nn
import torch.nn.functional as F
from layers.HarmonicKernel import HarmonicBank, HarmonicBankIK


def complex_tanh(input):
    return F.tanh(input.real).type(torch.complex64) + 1j * F.tanh(input.imag).type(torch.complex64)


class seasonal_encoder(nn.Module):
    def __init__(self, enc_in, kernel_num, individual, mode, seq_len, n_fft = [48], dropout = 0.05,
                 harm = False, harm_periods = (), harm_alpha = 0.15, harm_beta = 0.05,
                 harm_learn_period = True):
        super(seasonal_encoder, self).__init__()
        self.enc = enc_in
        self.n_fft = n_fft
        self.window = int((seq_len / (n_fft * 0.5)) + 1)  # window number
        self.window_len = int(n_fft / 2) + 1  # window length

        self.mode = mode
        self.harm = bool(harm)
        if self.mode == 'MK':
            self.wg1 = nn.Parameter(torch.rand(kernel_num, self.window_len, self.window, dtype = torch.cfloat))
            if not self.harm:
                # keep the original RNG consumption order so the baseline is bit-identical
                self.wc = nn.Parameter(
                    torch.rand(kernel_num, self.window_len, self.window, self.window, dtype = torch.cfloat))
            nn.init.xavier_normal_(self.wg1)
            if self.harm:
                # harmonic-tied dictionary: k banks, each generated from R+1 basis kernels
                self.harm_bank = HarmonicBank(n_outer = kernel_num, n_freq = self.window_len,
                                              n_win = self.window, n_fft = n_fft,
                                              periods = harm_periods, alpha = harm_alpha,
                                              beta = harm_beta, learn_period = harm_learn_period)
            else:
                nn.init.xavier_normal_(self.wc)
        elif self.mode == 'IK':
            if self.harm:
                # channel low-rank (rank I) + harmonic factorisation over bins
                self.harm_bank = HarmonicBankIK(n_ch = enc_in, n_rank = individual,
                                                n_freq = self.window_len, n_win = self.window,
                                                n_fft = n_fft, periods = harm_periods,
                                                alpha = harm_alpha, beta = harm_beta,
                                                learn_period = harm_learn_period)
            else:
                self.wc1 = nn.Parameter(torch.rand(enc_in, individual, dtype = torch.cfloat))
                self.wc2 = nn.Parameter(
                    torch.rand(individual, int(n_fft / 2) + 1, self.window, self.window, dtype = torch.cfloat))
                nn.init.xavier_normal_(self.wc1)
                nn.init.xavier_normal_(self.wc2)

        self.wf1 = nn.Parameter(torch.rand(self.window_len, self.window_len, dtype = torch.cfloat))
        self.bf1 = nn.Parameter(torch.rand(int(n_fft / 2) + 1, 1, dtype = torch.cfloat))
        self.wf2 = nn.Parameter(torch.rand(self.window_len, self.window_len, dtype = torch.cfloat))
        self.bf2 = nn.Parameter(torch.rand(self.window_len, 1, dtype = torch.cfloat))
        self.norm = nn.LayerNorm(seq_len)
        self.dropout = nn.Dropout(p = dropout)

        nn.init.xavier_normal_(self.wf1)
        nn.init.xavier_normal_(self.bf1)
        nn.init.xavier_normal_(self.wf2)
        nn.init.xavier_normal_(self.bf2)

    def forward(self, q):
        # STFT
        xq_stft = torch.stft(q, n_fft = self.n_fft, return_complex = True,
                             hop_length = int(self.n_fft * 0.5))  # [B*N,M,N]
        # seasonal-TFB
        if self.mode == 'MK':
            g = self.dropout(F.sigmoid(torch.abs(torch.einsum("bhw,nhw->bn", xq_stft, self.wg1)))).cfloat()  # [B,k]
            wc = self.harm_bank.effective() if self.harm else self.wc  # [k,M,N,N]
            h = torch.einsum("bhi,nhio->bnho", xq_stft, wc)  # [B*C,k,M,N]
            out = torch.einsum("bnhw,bn->bhw", h, g)
        elif self.mode == 'IK':
            xq_stft = xq_stft.reshape(int(q.shape[0] / self.enc), self.enc, xq_stft.shape[1],
                                      xq_stft.shape[2])  # [B,C,M,N]
            wc = self.harm_bank.effective() if self.harm else \
                torch.einsum("fi,ihlw->fhlw", self.wc1, self.wc2)  # [C,M,N,N]
            out = torch.einsum("bfhi,fhio->bfho", xq_stft, wc)
            out = out.reshape(out.shape[0] * out.shape[1], out.shape[2], out.shape[3])
        # Frequency-FFN
        out_ = out
        out = torch.einsum("biw,io->bow", out, self.wf1) + self.bf1.repeat(1, out.shape[2])
        out = complex_tanh(out)
        out = out_ + out

        # Inverse STFB
        out = torch.istft(out, n_fft = self.n_fft, hop_length = int(self.n_fft * 0.5))
        out = self.dropout(out)
        return out


class trend_encoder(nn.Module):
    def __init__(self, seq_len, n_fft = [48], dropout = 0.05,
                 harm = False, harm_periods = (), harm_alpha = 0.15, harm_beta = 0.05,
                 harm_learn_period = True):
        super(trend_encoder, self).__init__()
        self.n_fft = n_fft
        self.window = int((seq_len / (n_fft * 0.5)) + 1)  # window number N
        self.window_len = int(1 * (int(n_fft / 2) + 1))  # window length M
        self.harm = bool(harm)
        if self.harm:
            self.harm_bank = HarmonicBank(n_outer = 1, n_freq = self.window_len,
                                          n_win = self.window, n_fft = n_fft,
                                          periods = harm_periods, alpha = harm_alpha,
                                          beta = harm_beta, learn_period = harm_learn_period)
        else:
            self.wc = nn.Parameter(torch.rand(self.window_len, self.window, self.window, dtype = torch.cfloat))

        self.wf1 = nn.Parameter(torch.rand((int(n_fft / 2) + 1), int(n_fft / 2) + 1, dtype = torch.cfloat))
        self.bf1 = nn.Parameter(torch.rand((int(n_fft / 2) + 1), 1, dtype = torch.cfloat))

        self.dropout = nn.Dropout(p = dropout)
        if not self.harm:
            nn.init.xavier_normal_(self.wc)
        nn.init.xavier_normal_(self.wf1)
        nn.init.xavier_normal_(self.bf1)

    def forward(self, q):
        # STFB
        xq_stft = torch.stft(q, n_fft = self.n_fft, return_complex = True,
                             hop_length = int(self.n_fft * 0.5))  # [B*C,M.N]
        # Trend-TFB
        wc = self.harm_bank.effective().squeeze(0) if self.harm else self.wc  # [M,N,N]
        h = torch.einsum("bhi,hio->bho", xq_stft, wc)  # [B,n_channel,M,N]

        h_ = h
        h = torch.einsum("biw,io->bow", h, self.wf1) + self.bf1.repeat(1, h.shape[2])
        h = complex_tanh(h)
        out = h_ + h
        # Inverse STFB
        out = torch.istft(out, n_fft = self.n_fft, hop_length = int(self.n_fft * 0.5))
        out = self.dropout(out)
        return out


class Encoder(nn.Module):
    def __init__(self, enc_in, seq_len = 512, kernel_num = 16, individual = 7, mode = 'MK', n_fft = [16],
                 dropout = 0.05, harm_seasonal = False, harm_trend = False,
                 harm_periods = (), harm_alpha = 0.15, harm_beta = 0.05,
                 harm_learn_period = True):
        super(Encoder, self).__init__()
        self.block_seasonal = seasonal_encoder(enc_in = enc_in, kernel_num = kernel_num, individual = individual,
                                               mode = mode, seq_len = seq_len,
                                               n_fft = n_fft, dropout = dropout,
                                               harm = harm_seasonal, harm_periods = harm_periods,
                                               harm_alpha = harm_alpha, harm_beta = harm_beta,
                                               harm_learn_period = harm_learn_period)
        self.block_trend = trend_encoder(seq_len = seq_len, n_fft = n_fft, dropout = dropout,
                                         harm = harm_trend, harm_periods = harm_periods,
                                         harm_alpha = harm_alpha, harm_beta = harm_beta,
                                         harm_learn_period = harm_learn_period)

    def forward(self, q1, q2):
        seasonal = self.block_seasonal(q1)
        trend = self.block_trend(q2)
        return seasonal, trend
