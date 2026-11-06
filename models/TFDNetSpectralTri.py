"""TFDNet variant with shared-spectrum residual tri-band filtering."""
import torch
import torch.nn as nn

from layers.RevIN import RevIN
from layers.SpectralResidualTriBand import SpectralTriBandEncoder


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.channels = int(configs.enc_in)
        self.n_fft = list(configs.n_fft)
        self.dropout_rate = float(configs.dropout)

        self.revin_layer = RevIN(
            self.channels, affine=False, subtract_last=False)
        self.encoder_list = nn.ModuleList()
        for n_fft in self.n_fft:
            self.encoder_list.append(SpectralTriBandEncoder(
                n_channels=self.channels,
                seq_len=self.seq_len,
                n_fft=int(n_fft),
                low_kernel=int(getattr(
                    configs, "spectral_low_kernel", 15)),
                mid_kernel=int(getattr(
                    configs, "spectral_mid_kernel", 5)),
                high_kernel=int(getattr(
                    configs, "spectral_high_kernel", 3)),
                gate_hidden=int(getattr(
                    configs, "spectral_gate_hidden", 16)),
                gate_prior=getattr(
                    configs, "spectral_gate_prior", [0.60, 0.50, 0.40]),
                mask_transition=float(getattr(
                    configs, "spectral_mask_transition", 0.08)),
                dropout=self.dropout_rate))

        self.scale_fusion = nn.Linear(
            len(self.n_fft), 1, bias=False)
        with torch.no_grad():
            self.scale_fusion.weight.fill_(
                1.0 / max(1, len(self.n_fft)))
        self.linear1 = nn.Linear(self.seq_len, self.pred_len)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.last_band_gates = None
        self.last_band_masks = None

    def forward(self, x_enc, return_aux=False):
        batch, seq_len, channels = x_enc.shape
        if seq_len != self.seq_len or channels != self.channels:
            raise ValueError(
                "Expected [B, {}, {}], got {}".format(
                    self.seq_len, self.channels, tuple(x_enc.shape)))

        x_norm = self.revin_layer(x_enc, "norm")
        flat = x_norm.permute(0, 2, 1).reshape(
            batch * channels, self.seq_len)

        scale_outputs = []
        scale_gates = []
        scale_masks = []
        scale_spectra = []
        for encoder in self.encoder_list:
            if return_aux:
                output, aux = encoder(flat, return_aux=True)
                scale_gates.append(aux["band_gates"])
                scale_masks.append(aux["band_masks"])
                scale_spectra.append(aux["spectrum"])
            else:
                output = encoder(flat)
            scale_outputs.append(output)

        stacked = torch.stack(scale_outputs, dim=-1)
        fused = self.scale_fusion(stacked).squeeze(-1)
        out = self.linear1(fused)
        out = self.dropout(out)
        out = out.reshape(batch, channels, self.pred_len)
        out = out.permute(0, 2, 1)
        pred_norm = out
        prediction = self.revin_layer(pred_norm, "denorm")
        self.last_band_gates = scale_gates
        self.last_band_masks = scale_masks

        if return_aux:
            return prediction, {
                "pred_norm": pred_norm,
                "band_gates": scale_gates,
                "band_masks": scale_masks,
                "spectra": scale_spectra,
            }
        return prediction
