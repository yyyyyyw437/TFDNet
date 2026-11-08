"""CPU smoke tests for frequency, patch, and channel hybrid branches."""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.TFDNetHybrid import Model


def make_cfg(**overrides):
    cfg = dict(
        seq_len=96,
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
        use_freq_branch=1,
        use_patch_branch=1,
        use_channel_fusion=0,
        patch_len=16,
        patch_stride=8,
        patch_d_model=32,
        patch_n_heads=4,
        patch_e_layers=1,
        patch_d_ff=64,
        channel_rank=2,
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def run_case(name, **overrides):
    torch.manual_seed(41)
    cfg = make_cfg(**overrides)
    model = Model(cfg)
    x = torch.randn(2, cfg.seq_len, cfg.enc_in)
    prediction = model(x)
    prediction.square().mean().backward()

    assert prediction.shape == (2, cfg.pred_len, cfg.enc_in)
    assert torch.isfinite(prediction).all()
    ignored_suffixes = (
        ".wf2", ".bf2", ".norm.weight", ".norm.bias")
    missing = [parameter_name
               for parameter_name, parameter in model.named_parameters()
               if parameter.requires_grad and parameter.grad is None
               and not parameter_name.endswith(ignored_suffixes)]
    assert not missing, (name, missing[:10])
    if cfg.use_freq_branch and cfg.use_patch_branch:
        assert model.last_branch_weights.shape == (2,)
        assert torch.allclose(
            model.last_branch_weights.sum(), torch.tensor(1.0), atol=1e-6)
    print("{} output={} params={}".format(
        name,
        tuple(prediction.shape),
        sum(parameter.numel() for parameter in model.parameters())))


def main():
    run_case(
        "frequency-only", use_freq_branch=1,
        use_patch_branch=0, use_channel_fusion=0)
    run_case(
        "patch-only", use_freq_branch=0,
        use_patch_branch=1, use_channel_fusion=0)
    run_case(
        "frequency+patch", use_freq_branch=1,
        use_patch_branch=1, use_channel_fusion=0)
    run_case(
        "frequency+patch+channel", use_freq_branch=1,
        use_patch_branch=1, use_channel_fusion=1)

    long_cfg = make_cfg(
        seq_len=336,
        pred_len=96,
        enc_in=2,
        dec_in=2,
        c_out=2,
        n_fft=[8, 16, 32],
        patch_d_model=16,
        patch_n_heads=4,
        patch_d_ff=32,
        use_channel_fusion=1)
    long_model = Model(long_cfg)
    with torch.no_grad():
        long_output = long_model(
            torch.randn(1, long_cfg.seq_len, long_cfg.enc_in))
    assert long_output.shape == (1, long_cfg.pred_len, long_cfg.enc_in)
    print("seq_len=336 output={} OK".format(tuple(long_output.shape)))
    print("OK")


if __name__ == "__main__":
    main()
