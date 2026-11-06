TFDNet: Time-Frequency Enhanced Decomposed Network for Long-term Time Series Forecasting [【paper】](https://arxiv.org/abs/2308.13386)

## TFDNet Architecture
<img width="705" alt="截屏2024-05-11 16 59 30" src="https://github.com/YuxiaoLuo0013/TFDNet/assets/137262426/f99768d3-952c-4857-b3b4-f39afb4f10b9">

## Time-frequency Block
<img width="703" alt="截屏2024-05-11 17 00 37" src="https://github.com/YuxiaoLuo0013/TFDNet/assets/137262426/d547d7e6-11ca-464f-b849-d3683a2daa10">

## Main Results
<img width="737" alt="截屏2024-05-11 17 01 03" src="https://github.com/YuxiaoLuo0013/TFDNet/assets/137262426/bd99ec47-401e-466a-bfaa-b3fdb3b3f55a">



## Get Started

1. Install Python 3.9, PyTorch 1.12.0.

2. Download data. You can obtain all the five  benchmarks from
ETT https://github.com/zhouhaoyi/ETDataset
Electricity https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014
Traffic http://pems.dot.ca.gov
Weather https://www.bgc-jena.mpg.de/wetter/
Illness https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html

3. Train the model. We provide the experiment scripts of all benchmarks under the folder `./scripts`. You can reproduce the experiment results by:

```bash
bash ./scripts/ETTm1.sh
bash ./scripts/ETTm2.sh
bash ./scripts/ETTh1.sh
bash ./scripts/ETTh1.sh
bash ./scripts/ECL.sh
bash ./scripts/traffic.sh
bash ./scripts/weather.sh
bash ./scripts/ILI.sh
```
4. You can also run run_longExp.py.

## Tri-band pooled TFDNet

`TFDNetTriPool` is an optional research model and does not modify the original
`TFDNet` implementation. It adds an additive low/mid/high decomposition and
complex-aware frame pooling before the TFB kernels. The pooled frame axis is
restored with interpolation plus a full-resolution residual before inverse
STFT. Pool targets default to a fraction of each scale's actual frame count,
so low-resolution and high-resolution STFT branches are all pooled. Absolute
`--tf_pool_targets` values can still be supplied as an override.

The decomposition is exactly additive:

- low: large moving average
- mid: small moving average minus the low component
- high: input minus the small moving average

The model also supports full-resolution teacher to pooled-student distillation.
The teacher uses `--tf_pool 0`; the student uses `--tf_pool 1`. The teacher is
loaded from a checkpoint, frozen, and evaluated for every training batch.

```bash
python -u run_longExp.py \
  --is_training 1 --model TFDNetTriPool --data ETTh1 \
  --root_path ./Datasets --data_path ETTh1.csv \
  --seq_len 336 --pred_len 96 --enc_in 7 --dec_in 7 --c_out 7 \
  --n_fft 8 16 32 --mode MK --kernel_num 4 \
  --tri_low_kernel 25 --tri_mid_kernel 5 \
  --tf_pool 1 --tf_pool_ratios 0.50 0.60 0.75 \
  --tf_pool_mode avg --tf_upsample linear
```

Band fusion starts from `--band_gate_prior 0.50 0.30 0.20` and is then
conditioned on normalized low/mid/high energy. Both real and imaginary parts
of complex dropout use the same mask.

Run the CPU smoke test without datasets:

```bash
python tools/check_tri_pool.py
python tools/check_distillation.py
```

### Teacher-student distillation

First train a full-resolution tri-band teacher:

```bash
python -u run_longExp.py \
  --is_training 1 --model TFDNetTriPool --model_id kd_teacher \
  --data ETTh1 --root_path ./Datasets --data_path ETTh1.csv \
  --seq_len 336 --pred_len 96 --enc_in 7 --dec_in 7 --c_out 7 \
  --n_fft 8 16 32 --mode MK --kernel_num 4 \
  --tri_low_kernel 25 --tri_mid_kernel 5 --tf_pool 0 \
  --des kd_teacher
```

Then train the pooled student. `--kd_teacher_path` accepts either the
`checkpoint.pth` file or its experiment directory:

```bash
python -u run_longExp.py \
  --is_training 1 --model TFDNetTriPool --model_id kd_student \
  --data ETTh1 --root_path ./Datasets --data_path ETTh1.csv \
  --seq_len 336 --pred_len 96 --enc_in 7 --dec_in 7 --c_out 7 \
  --n_fft 8 16 32 --mode MK --kernel_num 4 \
  --tri_low_kernel 25 --tri_mid_kernel 5 \
  --tf_pool 1 --tf_pool_ratios 0.50 0.60 0.75 \
  --tf_pool_mode avg --tf_upsample linear \
  --kd_enable 1 \
  --kd_teacher_path ./checkpoints/kd_teacher_TFDNetTriPool_ETTh1_ftM_sl336_ll48_pl96_dm512_nh8_el2_dl1_df64_fc1_ebtimeF_dtTrue_kd_teacher_0/checkpoint.pth \
  --des kd_student
```

The distilled objective is:

```text
forecast + 0.10 * prediction
         + 0.002 * coarse TFB
         + 0.002 * full-resolution TFB
         + 0.005 * low/mid/high time-domain outputs
         + 0.0005 * complex spectrum
```

The high-frequency auxiliary term is multiplied by `0.40`; all weights can be
changed with the corresponding `--kd_*_weight` arguments. The auxiliary weights
are intentionally one tenth of the previous defaults to reduce their influence
relative to the forecast and prediction-distillation objectives. Teacher and
student must use the same `seq_len`, `enc_in`, and `n_fft` values.



## Acknowledgement

We appreciate the following github repos a lot for their valuable code base or datasets:

https://github.com/thuml/Autoformer

https://github.com/zhouhaoyi/Informer2020

https://github.com/zhouhaoyi/ETDataset

https://github.com/laiguokun/multivariate-time-series-data

https://github.com/cure-lab/LTSFLinear

