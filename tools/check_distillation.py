"""CPU smoke test for full-resolution teacher to pooled-student distillation."""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.TFDNetTriPool import Model
from utils.distillation import TFDNetDistillLoss


def make_cfg(tf_pool):
    return SimpleNamespace(
        seq_len=192,
        pred_len=24,
        enc_in=3,
        dec_in=3,
        c_out=3,
        n_fft=[16, 32],
        kernel_num=2,
        individual_factor=3,
        mode="MK",
        dropout=0.05,
        tf_pool=tf_pool,
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


def main():
    torch.manual_seed(11)
    student = Model(make_cfg(tf_pool=1))
    teacher = Model(make_cfg(tf_pool=0))
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    x = torch.randn(2, 192, 3)
    with torch.no_grad():
        _, teacher_aux = teacher(x, return_aux=True)
    prediction, student_aux = student(x, return_aux=True)

    student_coarse_frames = student_aux["coarse"][0][0].shape[-1]
    teacher_coarse_frames = teacher_aux["coarse"][0][0].shape[-1]
    assert student_coarse_frames < teacher_coarse_frames

    criterion = TFDNetDistillLoss()
    assert abs(criterion.auxiliary_weight - 0.0095) < 1e-12
    assert criterion.auxiliary_weight < 0.1 * criterion.prediction_weight
    parts = criterion(
        student_aux, teacher_aux, teacher, feature_dim=0)
    total = prediction.square().mean() + parts["total"]
    total.backward()

    assert prediction.shape == (2, 24, 3)
    assert torch.isfinite(parts["total"])
    assert any(
        parameter.grad is not None
        for parameter in student.parameters()
        if parameter.requires_grad)
    assert all(parameter.grad is None for parameter in teacher.parameters())
    print("coarse frames: student={} teacher={}".format(
        student_coarse_frames, teacher_coarse_frames))
    print("distillation total={:.7f}".format(parts["total"].item()))
    print("weighted prediction={:.7f} auxiliary={:.7f}".format(
        parts["weighted_prediction"].item(),
        parts["weighted_auxiliary"].item()))
    print("OK")


if __name__ == "__main__":
    main()
