"""CPU smoke tests for low-rank temporal TFB TFDNet."""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.TFDNet import Model as BaseModel
from models.TFDNetLowRank import Model as LowRankModel


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
        lowrank_rank=4,
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def check_rank(model):
    for encoder in model.encoder_list:
        for branch in ("block_seasonal", "block_trend"):
            block = getattr(encoder, branch)
            assert 0 < block.rank < block.n_frames
    print("low-rank factors: 0 < rank < frame_count for all branches")


def main():
    torch.manual_seed(31)
    cfg = make_cfg()
    model = LowRankModel(cfg)
    x = torch.randn(2, cfg.seq_len, cfg.enc_in)
    prediction = model(x)
    prediction.square().mean().backward()

    assert prediction.shape == (2, cfg.pred_len, cfg.enc_in)
    check_rank(model)
    missing = [name for name, parameter in model.named_parameters()
               if parameter.requires_grad and parameter.grad is None]
    assert not missing, missing[:10]
    print("IK output={} params={}".format(
        tuple(prediction.shape),
        sum(parameter.numel() for parameter in model.parameters())))

    mk_cfg = make_cfg(mode="MK", kernel_num=2, dropout=0.0)
    mk_model = LowRankModel(mk_cfg)
    with torch.no_grad():
        mk_output = mk_model(
            torch.randn(2, mk_cfg.seq_len, mk_cfg.enc_in))
    assert mk_output.shape == (2, mk_cfg.pred_len, mk_cfg.enc_in)
    print("MK output={} OK".format(tuple(mk_output.shape)))

    base_cfg = make_cfg(dropout=0.0)
    base = BaseModel(base_cfg)
    print("parameters base={} lowrank={} ratio={:.4f}".format(
        sum(parameter.numel() for parameter in base.parameters()),
        sum(parameter.numel() for parameter in model.parameters()),
        sum(parameter.numel() for parameter in model.parameters())
        / sum(parameter.numel() for parameter in base.parameters())))
    print("OK")


if __name__ == "__main__":
    main()
