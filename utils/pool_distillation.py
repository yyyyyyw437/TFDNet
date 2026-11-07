"""Distillation losses for pooled original TFDNet."""
import torch
import torch.nn.functional as F
from torch import nn

class TFDNetPoolDistillLoss(nn.Module):
    """Match a pooled student to a frozen full-resolution TFDNet teacher."""

    def __init__(
            self, prediction_weight=0.10, coarse_weight=0.002,
            full_weight=0.002, branch_weight=0.005,
            spectrum_weight=0.0005):
        super().__init__()
        self.prediction_weight = float(prediction_weight)
        self.coarse_weight = float(coarse_weight)
        self.full_weight = float(full_weight)
        self.branch_weight = float(branch_weight)
        self.spectrum_weight = float(spectrum_weight)
        self.smooth_l1 = nn.SmoothL1Loss()

    @property
    def auxiliary_weight(self):
        return (
            self.coarse_weight
            + self.full_weight
            + self.branch_weight
            + self.spectrum_weight
        )

    @staticmethod
    def _complex_l1(student, teacher):
        return (
            F.l1_loss(student.real, teacher.real)
            + F.l1_loss(student.imag, teacher.imag)
        )

    def _complex_branch_loss(self, student_aux, teacher_aux, key):
        losses = []
        for student_scale, teacher_scale in zip(
                student_aux[key], teacher_aux[key]):
            for branch in range(2):
                losses.append(self._complex_l1(
                    student_scale[branch], teacher_scale[branch]))
        return torch.stack(losses).mean()

    def _time_branch_loss(self, student_aux, teacher_aux):
        losses = []
        for student_scale, teacher_scale in zip(
                student_aux["branch_output"],
                teacher_aux["branch_output"]):
            for branch in range(2):
                losses.append(F.l1_loss(
                    student_scale[branch], teacher_scale[branch]))
        return torch.stack(losses).mean()

    def _coarse_loss(self, student_aux, teacher_aux, teacher_model):
        teacher_core = (
            teacher_model.module
            if isinstance(teacher_model, nn.DataParallel)
            else teacher_model)
        losses = []
        for scale_index, (student_scale, teacher_scale) in enumerate(zip(
                student_aux["coarse"], teacher_aux["coarse"])):
            branches = (
                teacher_core.encoder_list[scale_index].block_seasonal,
                teacher_core.encoder_list[scale_index].block_trend,
            )
            for branch_index, branch in enumerate(branches):
                student_value = student_scale[branch_index]
                teacher_value = branch.pool(
                    teacher_scale[branch_index],
                    student_value.shape[-1])
                losses.append(self._complex_l1(
                    student_value, teacher_value))
        return torch.stack(losses).mean()

    def forward(self, student_aux, teacher_aux, teacher_model, feature_dim=0):
        prediction = self.smooth_l1(
            student_aux["pred_norm"][:, :, feature_dim:],
            teacher_aux["pred_norm"][:, :, feature_dim:])
        coarse = self._coarse_loss(
            student_aux, teacher_aux, teacher_model)
        full_core = self._complex_branch_loss(
            student_aux, teacher_aux, "full_core")
        spectrum = self._complex_branch_loss(
            student_aux, teacher_aux, "spectrum")
        branch_output = self._time_branch_loss(student_aux, teacher_aux)

        weighted_prediction = self.prediction_weight * prediction
        weighted_auxiliary = (
            self.coarse_weight * coarse
            + self.full_weight * full_core
            + self.branch_weight * branch_output
            + self.spectrum_weight * spectrum
        )
        return {
            "total": weighted_prediction + weighted_auxiliary,
            "prediction": prediction.detach(),
            "coarse": coarse.detach(),
            "full_core": full_core.detach(),
            "spectrum": spectrum.detach(),
            "branch_output": branch_output.detach(),
            "weighted_prediction": weighted_prediction.detach(),
            "weighted_auxiliary": weighted_auxiliary.detach(),
        }
