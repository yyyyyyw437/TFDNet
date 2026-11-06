"""CPU smoke tests for shared-spectrum residual tri-band TFDNet."""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.TFDNetSpectralTri import Model


def make_cfg(**overrides):
    cfg = dict(
        seq_len=96,
        pred_len=24,
        enc_in=7,
        dec_in=7,
        c_out=7,
        n_fft=[8, 16, 32],
        dropout=0.05,
        spectral_low_kernel=15,
        spectral_mid_kernel=5,
        spectral_high_kernel=3,
        spectral_gate_hidden=16,
        spectral_gate_prior=[0.60, 0.50, 0.40],
        spectral_mask_transition=0.08,
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def check_masks(model):
    for encoder in model.encoder_list:
        masks = encoder.partition.last_masks
        assert masks.shape == (3, encoder.n_freq)
        assert torch.allclose(
            masks.sum(dim=0), torch.ones(encoder.n_freq), atol=1e-6)
        assert torch.all(masks >= 0)
        assert masks[0, 0] > masks[2, 0]
        assert masks[2, -1] > masks[0, -1]
    print("spectral partition: normalized low/mid/high masks OK")


def main():
    torch.manual_seed(23)
    cfg = make_cfg()
    model = Model(cfg)
    x = torch.randn(2, cfg.seq_len, cfg.enc_in)
    prediction, aux = model(x, return_aux=True)
    loss = prediction.square().mean()
    loss.backward()

    assert prediction.shape == (2, cfg.pred_len, cfg.enc_in)
    assert len(aux["band_gates"]) == len(cfg.n_fft)
    for gates in aux["band_gates"]:
        assert gates.shape == (2, cfg.enc_in, 3)
        assert torch.all(gates > 0) and torch.all(gates < 1)
    check_masks(model)

    trainable = [parameter for parameter in model.parameters()
                 if parameter.requires_grad]
    missing = [name for name, parameter in model.named_parameters()
               if parameter.requires_grad and parameter.grad is None]
    assert not missing, missing[:10]
    print("output={} params={} dropout=train OK".format(
        tuple(prediction.shape), sum(p.numel() for p in trainable)))

    long_cfg = make_cfg(seq_len=336, enc_in=3, dec_in=3, c_out=3)
    long_model = Model(long_cfg)
    with torch.no_grad():
        long_output = long_model(
            torch.randn(1, long_cfg.seq_len, long_cfg.enc_in))
    assert long_output.shape == (1, long_cfg.pred_len, long_cfg.enc_in)
    print("seq_len=336 output={} OK".format(tuple(long_output.shape)))
    print("OK")


if __name__ == "__main__":
    main()
