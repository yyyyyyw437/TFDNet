"""TFDNet variant with tri-band decomposition and pooled time-frequency kernels."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.RevIN import RevIN
from layers.TriBandPooledSTFT import TriBandDecomposition, TriScaleEncoder


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.channels = int(configs.enc_in)
        self.n_fft = list(configs.n_fft)
        self.dropout_rate = float(configs.dropout)

        self.use_pool = bool(getattr(configs, "tf_pool", 1))
        self.pool_ratios = list(getattr(
            configs, "tf_pool_ratios", [0.50, 0.60, 0.75]))
        self.pool_targets = list(getattr(
            configs, "tf_pool_targets", [0, 0, 0]))
        if len(self.pool_ratios) != 3:
            raise ValueError("tf_pool_ratios must contain [low, mid, high]")
        if len(self.pool_targets) != 3:
            raise ValueError("tf_pool_targets must contain [low, mid, high]")
        if not any(int(target) > 0 for target in self.pool_targets):
            if any(
                    not 0.0 < float(ratio) < 1.0
                    for ratio in self.pool_ratios):
                raise ValueError(
                    "tf_pool_ratios values must be in (0, 1) when no "
                    "absolute tf_pool_targets are provided")
        self.pool_mode = str(getattr(configs, "tf_pool_mode", "avg"))
        self.upsample_mode = str(getattr(configs, "tf_upsample", "linear"))
        self.high_kernel = int(getattr(configs, "tri_high_kernel", 3))

        self.revin_layer = RevIN(
            self.channels, affine=False, subtract_last=False)
        self.decomposition = TriBandDecomposition(
            self.seq_len,
            low_kernel=int(getattr(configs, "tri_low_kernel", 25)),
            mid_kernel=int(getattr(configs, "tri_mid_kernel", 5)))

        self.encoder_list = nn.ModuleList()
        for n_fft in self.n_fft:
            self.encoder_list.append(TriScaleEncoder(
                n_channels=self.channels,
                seq_len=self.seq_len,
                n_fft=int(n_fft),
                use_pool=self.use_pool,
                pool_ratios=self.pool_ratios,
                pool_targets=self.pool_targets,
                pool_mode=self.pool_mode,
                upsample_mode=self.upsample_mode,
                kernel_num=int(getattr(configs, "kernel_num", 4)),
                individual_factor=int(getattr(
                    configs, "individual_factor", 7)),
                mode=str(getattr(configs, "mode", "MK")),
                high_kernel=self.high_kernel,
                dropout=self.dropout_rate))

        n_scales = len(self.n_fft)
        self.scale_low = nn.Linear(n_scales, 1, bias=False)
        self.scale_mid = nn.Linear(n_scales, 1, bias=False)
        self.scale_high = nn.Linear(n_scales, 1, bias=False)
        self.linear1 = nn.Linear(self.seq_len, self.pred_len)
        self.dropout = nn.Dropout(self.dropout_rate)

        gate_hidden = int(getattr(configs, "band_gate_hidden", 16))
        self.band_gate = nn.Sequential(
            nn.Linear(3, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 3),
        )
        band_prior = torch.tensor(
            list(getattr(configs, "band_gate_prior", [0.50, 0.30, 0.20])),
            dtype=torch.float32)
        if band_prior.numel() != 3 or torch.any(band_prior <= 0):
            raise ValueError(
                "band_gate_prior must contain three positive values")
        band_prior = band_prior / band_prior.sum()
        self.register_buffer(
            "band_gate_prior", band_prior, persistent=False)
        for module in self.band_gate.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        last_linear = self.band_gate[-1]
        with torch.no_grad():
            last_linear.weight.mul_(0.10)
            last_linear.bias.copy_(band_prior.log())

        self.last_band_weights = None

    def _fuse_scales(self, outputs, projector):
        stacked = torch.stack(outputs, dim=-1)
        return projector(stacked).squeeze(-1)

    def forward(self, x_enc, return_aux=False):
        batch, seq_len, channels = x_enc.shape
        if seq_len != self.seq_len or channels != self.channels:
            raise ValueError(
                "Expected [B, {}, {}], got {}".format(
                    self.seq_len, self.channels, tuple(x_enc.shape)))

        x_norm = self.revin_layer(x_enc, "norm")
        low, mid, high = self.decomposition(x_norm)

        def flatten_band(band):
            return band.permute(0, 2, 1).reshape(
                batch * channels, self.seq_len)

        low_flat = flatten_band(low)
        mid_flat = flatten_band(mid)
        high_flat = flatten_band(high)

        low_scales, mid_scales, high_scales = [], [], []
        layer_aux = (
            {"coarse": [], "full_core": [], "spectrum": []}
            if return_aux else None)
        for encoder in self.encoder_list:
            if return_aux:
                (out_low, out_mid, out_high), scale_aux = encoder(
                    low_flat, mid_flat, high_flat, return_aux=True)
                for key in layer_aux:
                    layer_aux[key].append(scale_aux[key])
            else:
                out_low, out_mid, out_high = encoder(
                    low_flat, mid_flat, high_flat)
            low_scales.append(out_low)
            mid_scales.append(out_mid)
            high_scales.append(out_high)

        low_out = self._fuse_scales(low_scales, self.scale_low)
        mid_out = self._fuse_scales(mid_scales, self.scale_mid)
        high_out = self._fuse_scales(high_scales, self.scale_high)

        low_out = low_out.reshape(batch, channels, self.seq_len)
        mid_out = mid_out.reshape(batch, channels, self.seq_len)
        high_out = high_out.reshape(batch, channels, self.seq_len)
        band_energy = torch.stack([
            low_out.abs().mean(dim=(1, 2)),
            mid_out.abs().mean(dim=(1, 2)),
            high_out.abs().mean(dim=(1, 2)),
        ], dim=-1)
        band_features = torch.log(band_energy.clamp_min(1e-6))
        band_features = (
            band_features - band_features.mean(dim=-1, keepdim=True))
        band_weights = F.softmax(self.band_gate(band_features), dim=-1)
        self.last_band_weights = band_weights.detach()

        fused = (
            band_weights[:, 0].view(batch, 1, 1) * low_out
            + band_weights[:, 1].view(batch, 1, 1) * mid_out
            + band_weights[:, 2].view(batch, 1, 1) * high_out
        )
        fused = fused.reshape(batch * channels, self.seq_len)
        out = self.linear1(fused)
        out = self.dropout(out)
        out = out.reshape(batch, channels, self.pred_len)
        out = out.permute(0, 2, 1)
        pred_norm = out
        prediction = self.revin_layer(pred_norm, "denorm")
        if return_aux:
            layer_aux["band_output"] = [
                [low_scales[index], mid_scales[index], high_scales[index]]
                for index in range(len(low_scales))
            ]
            layer_aux["pred_norm"] = pred_norm
            return prediction, layer_aux
        return prediction
