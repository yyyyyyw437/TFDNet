"""CPU smoke tests for pooled original TFDNet and distillation."""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.TFDNetPool import Model
from utils.pool_distillation import TFDNetPoolDistillLoss


def make_cfg(tf_pool, **overrides):
    cfg = dict(
        seq_len=192,
        pred_len=24,
        enc_in=3,
        dec_in=3,
        c_out=3,
        n_fft=[16, 32],
        kernel_num=2,
        individual_factor=3,
        mode="IK",
        output_attention=False,
        kernel_size=25,
        dropout=0.05,
        tf_pool=tf_pool,
        tf_pool_ratios=[0.50, 0.75],
        tf_pool_mode="avg",
        tf_upsample="linear",
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def check_pool_frames(pooled, full):
    for pooled_encoder, full_encoder in zip(
            pooled.encoder_list, full.encoder_list):
        for branch in ("block_seasonal", "block_trend"):
            pooled_branch = getattr(pooled_encoder, branch)
            full_branch = getattr(full_encoder, branch)
            assert pooled_branch.pooled_frames < full_branch.n_frames
    print("adaptive pooling: seasonal/trend frames reduced at every scale")


def main():
    torch.manual_seed(29)
    full = Model(make_cfg(tf_pool=0))
    pooled = Model(make_cfg(tf_pool=1))
    x = torch.randn(2, 192, 3)

    full.eval()
    for parameter in full.parameters():
        parameter.requires_grad = False
    with torch.no_grad():
        full_prediction, full_aux = full(x, return_aux=True)

    prediction, student_aux = pooled(x, return_aux=True)
    criterion = TFDNetPoolDistillLoss()
    parts = criterion(
        student_aux, full_aux, full, feature_dim=0)
    (prediction.square().mean() + parts["total"]).backward()

    assert prediction.shape == (2, 24, 3)
    assert full_prediction.shape == prediction.shape
    assert student_aux["coarse"][0][0].shape[-1] < (
        full_aux["coarse"][0][0].shape[-1])
    check_pool_frames(pooled, full)
    assert abs(criterion.auxiliary_weight - 0.0095) < 1e-12
    assert all(parameter.grad is None for parameter in full.parameters())
    missing = [name for name, parameter in pooled.named_parameters()
               if parameter.requires_grad and parameter.grad is None]
    assert not missing, missing[:10]

    long_cfg = make_cfg(
        tf_pool=1, seq_len=336, enc_in=2, dec_in=2, c_out=2)
    long_model = Model(long_cfg)
    with torch.no_grad():
        long_output = long_model(
            torch.randn(1, long_cfg.seq_len, long_cfg.enc_in))
    assert long_output.shape == (1, long_cfg.pred_len, long_cfg.enc_in)

    energy_cfg = make_cfg(
        tf_pool=1, tf_pool_mode="energy", tf_upsample="nearest",
        dropout=0.0)
    energy_model = Model(energy_cfg)
    with torch.no_grad():
        energy_output = energy_model(
            torch.randn(2, energy_cfg.seq_len, energy_cfg.enc_in))
    assert energy_output.shape == (2, energy_cfg.pred_len, energy_cfg.enc_in)

    mk_cfg = make_cfg(tf_pool=1, mode="MK", kernel_num=2, dropout=0.0)
    mk_model = Model(mk_cfg)
    with torch.no_grad():
        mk_output = mk_model(
            torch.randn(2, mk_cfg.seq_len, mk_cfg.enc_in))
    assert mk_output.shape == (2, mk_cfg.pred_len, mk_cfg.enc_in)

    print("full params={} pooled params={}".format(
        sum(parameter.numel() for parameter in full.parameters()),
        sum(parameter.numel() for parameter in pooled.parameters())))
    print("distillation total={:.7f}".format(parts["total"].item()))
    print("OK")


if __name__ == "__main__":
    main()
