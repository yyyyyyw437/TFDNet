"""CPU smoke tests for multi-scale convolution TFB TFDNet."""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.TFDNet import Model as BaseModel
from models.TFDNetConv import Model as ConvModel


def make_cfg(**overrides):
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
        tf_conv_kernels=[3, 5, 9],
        tf_conv_dilations=[1, 2, 4],
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def check_bank(model):
    for encoder in model.encoder_list:
        for branch in ("block_seasonal", "block_trend"):
            bank = getattr(encoder, branch).conv
            assert bank.last_gates is not None
            assert bank.last_gates.shape[1] == len(bank.kernels)
    print("convolution bank: multi-scale gates and outputs OK")


def main():
    torch.manual_seed(37)
    cfg = make_cfg()
    model = ConvModel(cfg)
    x = torch.randn(2, cfg.seq_len, cfg.enc_in)
    prediction = model(x)
    prediction.square().mean().backward()

    assert prediction.shape == (2, cfg.pred_len, cfg.enc_in)
    check_bank(model)
    missing = [name for name, parameter in model.named_parameters()
               if parameter.requires_grad and parameter.grad is None]
    assert not missing, missing[:10]
    print("IK conv output={} params={}".format(
        tuple(prediction.shape),
        sum(parameter.numel() for parameter in model.parameters())))

    mk_cfg = make_cfg(mode="MK", dropout=0.0)
    mk_model = ConvModel(mk_cfg)
    with torch.no_grad():
        mk_output = mk_model(
            torch.randn(2, mk_cfg.seq_len, mk_cfg.enc_in))
    assert mk_output.shape == (2, mk_cfg.pred_len, mk_cfg.enc_in)
    print("MK conv output={} OK".format(tuple(mk_output.shape)))

    base_cfg = make_cfg(dropout=0.0)
    base = BaseModel(base_cfg)
    print("parameters base={} conv={} ratio={:.4f}".format(
        sum(parameter.numel() for parameter in base.parameters()),
        sum(parameter.numel() for parameter in model.parameters()),
        sum(parameter.numel() for parameter in model.parameters())
        / sum(parameter.numel() for parameter in base.parameters())))
    print("OK")


if __name__ == "__main__":
    main()
