"""Original TFDNet architecture with optional STFT frame pooling."""
import torch
from torch import nn

from layers.PooledSTFT_Enc import Encoder
from layers.RevIN import RevIN


class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(
            kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(
            1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(
            1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        return x - moving_mean, moving_mean


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.output_attention = bool(configs.output_attention)
        self.channels = int(configs.enc_in)
        self.n_fft = list(configs.n_fft)
        self.use_pool = bool(getattr(configs, "tf_pool", 1))
        self.pool_ratios = list(getattr(
            configs, "tf_pool_ratios", [0.50, 0.75]))
        if len(self.pool_ratios) != 2:
            raise ValueError(
                "tf_pool_ratios must contain [seasonal, trend]")
        if any(not 0.0 < float(value) < 1.0 for value in self.pool_ratios):
            raise ValueError("tf_pool_ratios values must be in (0, 1)")
        self.pool_mode = str(getattr(configs, "tf_pool_mode", "avg"))
        self.upsample_mode = str(getattr(configs, "tf_upsample", "linear"))

        self.encoder_list = nn.ModuleList()
        for n_fft in self.n_fft:
            self.encoder_list.append(Encoder(
                enc_in=self.channels,
                seq_len=self.seq_len,
                kernel_num=int(configs.kernel_num),
                individual=int(configs.individual_factor),
                mode=str(configs.mode),
                n_fft=int(n_fft),
                pool_ratios=self.pool_ratios if self.use_pool else [1.0, 1.0],
                pool_mode=self.pool_mode,
                upsample_mode=self.upsample_mode,
                dropout=float(configs.dropout)))

        self.mlp1 = nn.Linear(len(self.n_fft), 1, bias=False)
        self.mlp2 = nn.Linear(len(self.n_fft), 1, bias=False)
        self.dropout = nn.Dropout(float(configs.dropout))
        self.linear1 = nn.Linear(self.seq_len, self.pred_len)
        self.decompsition = series_decomp(int(configs.kernel_size))
        self.revin_layer = RevIN(
            self.channels, affine=False, subtract_last=False)

    def forward(self, x_enc, return_aux=False):
        batch, seq_len, channels = x_enc.shape
        if seq_len != self.seq_len or channels != self.channels:
            raise ValueError(
                "Expected [B, {}, {}], got {}".format(
                    self.seq_len, self.channels, tuple(x_enc.shape)))

        x_norm = self.revin_layer(x_enc, "norm")
        seasonal, trend = self.decompsition(x_norm)
        seasonal = seasonal.permute(0, 2, 1).reshape(
            batch * channels, self.seq_len)
        trend = trend.permute(0, 2, 1).reshape(
            batch * channels, self.seq_len)

        seasonal_scales = []
        trend_scales = []
        branch_aux = []
        for encoder in self.encoder_list:
            if return_aux:
                (seasonal_out, trend_out), aux = encoder(
                    seasonal, trend, return_aux=True)
                branch_aux.append(aux["branches"])
            else:
                seasonal_out, trend_out = encoder(seasonal, trend)
            seasonal_scales.append(seasonal_out)
            trend_scales.append(trend_out)

        seasonal_stacked = torch.stack(seasonal_scales, dim=-1)
        trend_stacked = torch.stack(trend_scales, dim=-1)
        out_seasonal = self.mlp1(seasonal_stacked).squeeze(-1)
        out_trend = self.mlp2(trend_stacked).squeeze(-1)
        fused = out_seasonal + out_trend
        out = self.linear1(fused)
        out = self.dropout(out)
        out = out.reshape(batch, channels, self.pred_len)
        out = out.permute(0, 2, 1)
        pred_norm = out
        prediction = self.revin_layer(pred_norm, "denorm")

        if return_aux:
            aux = {"pred_norm": pred_norm}
            for key in ("coarse", "full_core", "spectrum"):
                aux[key] = [
                    [scale[0][key], scale[1][key]]
                    for scale in branch_aux
                ]
            aux["branch_output"] = [
                [seasonal_scales[index], trend_scales[index]]
                for index in range(len(seasonal_scales))
            ]
            return prediction, aux
        return prediction
