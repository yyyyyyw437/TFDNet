"""CPU smoke tests for the optional TFDNetTriPool model."""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from layers.TriBandPooledSTFT import ComplexDropout, TriBandDecomposition
from models.TFDNetTriPool import Model


def make_cfg(**overrides):
    cfg = dict(
        seq_len=96,
        pred_len=24,
        enc_in=7,
        dec_in=7,
        c_out=7,
        n_fft=[8, 16, 32],
        kernel_num=4,
        individual_factor=7,
        mode="MK",
        dropout=0.05,
        tf_pool=1,
        tf_pool_ratios=[0.50, 0.60, 0.75],
        tf_pool_targets=[0, 0, 0],
        tf_pool_mode="avg",
        tf_upsample="linear",
        tri_low_kernel=25,
        tri_mid_kernel=5,
        tri_high_kernel=3,
        band_gate_hidden=8,
        band_gate_prior=[0.50, 0.30, 0.20],
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def run_case(name, **overrides):
    cfg = make_cfg(**overrides)
    model = Model(cfg)
    x = torch.randn(2, cfg.seq_len, cfg.enc_in)
    y = model(x)
    loss = y.square().mean()
    loss.backward()

    assert y.shape == (2, cfg.pred_len, cfg.enc_in), y.shape
    assert model.last_band_weights.shape == (2, 3)
    sums = model.last_band_weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)

    trainable = [p for p in model.parameters() if p.requires_grad]
    no_grad = [name for name, p in model.named_parameters()
               if p.requires_grad and p.grad is None]
    print("{}: params={} output={} no_grad={}".format(
        name, sum(p.numel() for p in trainable), tuple(y.shape), len(no_grad)))
    assert not no_grad, no_grad[:10]


def check_decomposition():
    x = torch.randn(3, 96, 7)
    low, mid, high = TriBandDecomposition(96)(x)
    reconstructed = low + mid + high
    assert torch.allclose(reconstructed, x, atol=1e-6)
    print("decomposition: exact additive reconstruction OK")


def check_complex_dropout():
    torch.manual_seed(13)
    layer = ComplexDropout(0.50)
    x = torch.randn(4, 6, 9, dtype=torch.complex64, requires_grad=True)

    layer.eval()
    assert torch.equal(layer(x), x)

    layer.train()
    dropped = layer(x)
    assert torch.equal(dropped.real == 0, dropped.imag == 0)
    assert torch.any(dropped == 0)
    (dropped.real.square().mean() + dropped.imag.square().mean()).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    print("complex dropout: shared real/imag mask and backward OK")


def check_adaptive_pooling():
    cfg = make_cfg(
        seq_len=336,
        enc_in=1,
        dec_in=1,
        c_out=1,
        n_fft=[8, 16, 32],
        kernel_num=2,
        individual_factor=2,
    )
    model = Model(cfg)
    frame_counts = []
    for encoder in model.encoder_list:
        counts = []
        for band in ("low", "mid", "high"):
            band_encoder = getattr(encoder, band)
            assert band_encoder.pooled_frames < band_encoder.n_frames
            counts.append(
                "{}:{}->{}".format(
                    band, band_encoder.n_frames,
                    band_encoder.pooled_frames))
        frame_counts.append(", ".join(counts))
    print("adaptive pooling: " + " | ".join(frame_counts))


def check_band_gate():
    torch.manual_seed(17)
    cfg = make_cfg(enc_in=2, dec_in=2, c_out=2, dropout=0.0)
    model = Model(cfg)
    model.eval()
    model(torch.randn(2, cfg.seq_len, cfg.enc_in))
    first_weights = model.last_band_weights.clone()
    model(torch.randn(2, cfg.seq_len, cfg.enc_in) * 3.0)
    second_weights = model.last_band_weights.clone()

    assert not torch.allclose(
        first_weights, torch.full_like(first_weights, 1.0 / 3.0),
        atol=1e-4)
    assert not torch.allclose(first_weights, second_weights, atol=1e-6)
    print("band gate: non-uniform prior and energy-dependent weights OK")


if __name__ == "__main__":
    torch.manual_seed(7)
    print("torch", torch.__version__)
    check_decomposition()
    check_complex_dropout()
    check_adaptive_pooling()
    check_band_gate()
    run_case("MK/pool/linear")
    run_case("IK/pool/linear", mode="IK", individual_factor=4)
    run_case("MK/no-pool", tf_pool=0)
    run_case("MK/energy/nearest", tf_pool_mode="energy",
             tf_upsample="nearest")
    print("OK")
