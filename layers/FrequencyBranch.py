"""TFDNet frequency branch used as one component of a hybrid model."""
import torch
from torch import nn

from layers.STFT_Enc import Encoder


class MovingAverage(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.avg = nn.AvgPool1d(
            kernel_size=self.kernel_size, stride=1, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(
            1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(
            1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.average = MovingAverage(kernel_size)

    def forward(self, x):
        trend = self.average(x)
        return x - trend, trend


class FrequencyBranch(nn.Module):
    """Original TFDNet frequency path returning normalized forecasts."""

    def __init__(self, configs):
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.channels = int(configs.enc_in)
        self.n_fft = list(configs.n_fft)
        self.encoder_list = nn.ModuleList()
        for n_fft in self.n_fft:
            self.encoder_list.append(Encoder(
                enc_in=self.channels,
                seq_len=self.seq_len,
                n_fft=int(n_fft),
                dropout=float(configs.dropout),
                kernel_num=int(configs.kernel_num),
                individual=int(configs.individual_factor),
                mode=str(configs.mode)))
        self.mlp1 = nn.Linear(len(self.n_fft), 1, bias=False)
        self.mlp2 = nn.Linear(len(self.n_fft), 1, bias=False)
        self.linear = nn.Linear(self.seq_len, self.pred_len)
        self.dropout = nn.Dropout(float(configs.dropout))
        self.decomposition = SeriesDecomposition(
            int(configs.kernel_size))

    def forward(self, x_norm):
        batch, _, channels = x_norm.shape
        seasonal, trend = self.decomposition(x_norm)
        seasonal = seasonal.permute(0, 2, 1).reshape(
            batch * channels, self.seq_len)
        trend = trend.permute(0, 2, 1).reshape(
            batch * channels, self.seq_len)

        seasonal_scales = []
        trend_scales = []
        for encoder in self.encoder_list:
            seasonal_out, trend_out = encoder(seasonal, trend)
            seasonal_scales.append(seasonal_out)
            trend_scales.append(trend_out)
        seasonal = torch.stack(seasonal_scales, dim=-1)
        trend = torch.stack(trend_scales, dim=-1)
        seasonal = self.mlp1(seasonal).squeeze(-1)
        trend = self.mlp2(trend).squeeze(-1)
        out = self.linear(seasonal + trend)
        out = self.dropout(out)
        return out.reshape(batch, channels, self.pred_len).permute(0, 2, 1)
