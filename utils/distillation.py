"""Distillation losses for tri-band pooled TFDNet."""
import torch
import torch.nn.functional as F
from torch import nn


BANDS = ("low", "mid", "high")


class TFDNetDistillLoss(nn.Module):
    """Align student and frozen full-resolution teacher representations.

    The student sees pooled time-frequency frames. Coarse features are directly
    comparable after the teacher applies the same dynamic pool at the student
    frame count. Full-resolution and time-domain features are already aligned.
    """

    def __init__(
            self, prediction_weight=0.10, coarse_weight=0.002,
            full_weight=0.002, band_weight=0.005, spectrum_weight=0.0005,
            high_band_scale=0.40):
        super().__init__()
        self.prediction_weight = float(prediction_weight)
        self.coarse_weight = float(coarse_weight)
        self.full_weight = float(full_weight)
        self.band_weight = float(band_weight)
        self.spectrum_weight = float(spectrum_weight)
        self.high_band_scale = float(high_band_scale)
        self.smooth_l1 = nn.SmoothL1Loss()

    @property
    def auxiliary_weight(self):
        return (
            self.coarse_weight
            + self.full_weight
            + self.band_weight
            + self.spectrum_weight
        )

    @staticmethod
    def _complex_l1(student, teacher):
        return (
            F.l1_loss(student.real, teacher.real)
            + F.l1_loss(student.imag, teacher.imag)
        )

    def _band_complex_loss(self, student_scales, teacher_scales):
        losses = []
        for student_bands, teacher_bands in zip(
                student_scales, teacher_scales):
            for index, band in enumerate(BANDS):
                band_loss = self._complex_l1(
                    student_bands[index], teacher_bands[index])
                scale = self.high_band_scale if band == "high" else 1.0
                losses.append(band_loss * scale)
        return torch.stack(losses).mean()

    def _band_time_loss(self, student_scales, teacher_scales):
        losses = []
        for student_bands, teacher_bands in zip(
                student_scales, teacher_scales):
            for index, band in enumerate(BANDS):
                band_loss = F.l1_loss(
                    student_bands[index], teacher_bands[index])
                scale = self.high_band_scale if band == "high" else 1.0
                losses.append(band_loss * scale)
        return torch.stack(losses).mean()

    def _coarse_loss(self, student_aux, teacher_aux, teacher_model):
        losses = []
        teacher_core = (
            teacher_model.module
            if isinstance(teacher_model, nn.DataParallel)
            else teacher_model)
        for scale_index, (student_bands, teacher_bands) in enumerate(zip(
                student_aux["coarse"], teacher_aux["coarse"])):
            for band_index, band in enumerate(BANDS):
                student_value = student_bands[band_index]
                teacher_value = teacher_bands[band_index]
                encoder = getattr(
                    teacher_core.encoder_list[scale_index], band)
                teacher_value = encoder.pool(
                    teacher_value, student_value.shape[-1])
                band_loss = self._complex_l1(
                    student_value, teacher_value)
                scale = self.high_band_scale if band == "high" else 1.0
                losses.append(band_loss * scale)
        return torch.stack(losses).mean()

    def forward(self, student_aux, teacher_aux, teacher_model, feature_dim=0):
        prediction = self.smooth_l1(
            student_aux["pred_norm"][:, :, feature_dim:],
            teacher_aux["pred_norm"][:, :, feature_dim:])
        coarse = self._coarse_loss(
            student_aux, teacher_aux, teacher_model)
        full_core = self._band_complex_loss(
            student_aux["full_core"], teacher_aux["full_core"])
        spectrum = self._band_complex_loss(
            student_aux["spectrum"], teacher_aux["spectrum"])
        band_output = self._band_time_loss(
            student_aux["band_output"], teacher_aux["band_output"])

        weighted_prediction = self.prediction_weight * prediction
        weighted_auxiliary = (
            self.coarse_weight * coarse
            + self.full_weight * full_core
            + self.band_weight * band_output
            + self.spectrum_weight * spectrum
        )
        weighted = weighted_prediction + weighted_auxiliary
        return {
            "total": weighted,
            "prediction": prediction.detach(),
            "coarse": coarse.detach(),
            "full_core": full_core.detach(),
            "spectrum": spectrum.detach(),
            "band_output": band_output.detach(),
            "weighted_prediction": weighted_prediction.detach(),
            "weighted_auxiliary": weighted_auxiliary.detach(),
        }
