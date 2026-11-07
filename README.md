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

## Low-rank temporal TFB

`TFDNetLowRank` keeps the original TFDNet seasonal/trend decomposition,
multi-scale STFT, frequency FFN, and inverse STFT. It only replaces every
dense temporal TFB matrix with two low-rank factors:

```text
dense:     X [B,F,N] x K [F,N,N]
low-rank:  X [B,F,N] x U [F,N,R] x V [F,R,N]
```

The full time-frame resolution is preserved. There is no pooling, upsampling,
teacher, student, or knowledge distillation in this model.

```bash
python -u run_longExp.py \
  --is_training 1 --model TFDNetLowRank --model_id lowrank_tfb \
  --data ETTm2 --root_path ./Datasets --data_path ETTm2.csv \
  --seq_len 336 --pred_len 96 --enc_in 7 --dec_in 7 --c_out 7 \
  --n_fft 8 16 32 --mode IK --individual_factor 7 \
  --lowrank_rank 8 --seed 2021 --itr 1
```

Run the CPU smoke test:

```bash
python tools/check_lowrank_tfdnet.py
```

Run the GPU forward/backward microbenchmark without a dataset:

```bash
python tools/benchmark_lowrank_tfdnet.py \
  --device cuda --batch_size 32 --seq_len 336 --pred_len 96 \
  --enc_in 7 --n_fft 8 16 32 --mode IK \
  --lowrank_rank 8 --warmup 5 --repeats 20
```



## Acknowledgement

We appreciate the following github repos a lot for their valuable code base or datasets:

https://github.com/thuml/Autoformer

https://github.com/zhouhaoyi/Informer2020

https://github.com/zhouhaoyi/ETDataset

https://github.com/laiguokun/multivariate-time-series-data

https://github.com/cure-lab/LTSFLinear

