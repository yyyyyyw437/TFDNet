#!/bin/bash
# Paired comparison: identical settings, ONLY the harmonic flag differs.
# Edit the block below for another dataset; every run keeps the same seed / lr / epochs.
set -e

DATA=ETTh1
DATA_PATH=ETTh1.csv
ENC_IN=7
SEQ=512
PRED=96
EPOCHS=50
LR=0.00005
BS=128
K=4            # kernel_num for MK
I=7            # individual_factor for IK
# Fundamental periods in SAMPLES. Hourly data -> 24 h = 24 samples, 168 h = 168.
PERIODS="24 168"

mkdir -p logs/LongForecasting

run () {  # $1=mode  $2=tag  $3=extra args
  python -u run_longExp.py \
    --is_training 1 --model_id ${DATA}_${SEQ}_${PRED}_$2 --model TFDNet \
    --data ${DATA} --root_path ./Datasets --data_path ${DATA_PATH} \
    --seq_len ${SEQ} --pred_len ${PRED} --enc_in ${ENC_IN} \
    --mode $1 --kernel_num ${K} --individual_factor ${I} \
    --batch_size ${BS} --train_epochs ${EPOCHS} --learning_rate ${LR} \
    --dropout 0.05 --patience 10 --itr 1 --des "$2" \
    $3 > logs/LongForecasting/${DATA}_$2.log
}

echo "== MK baseline ==";  run MK mk_base   ""
echo "== MK harmonic ==";  run MK mk_harm   "--harm_seasonal 1 --harm_periods ${PERIODS}"
echo "== IK baseline ==";  run IK ik_base   ""
echo "== IK harmonic ==";  run IK ik_harm   "--harm_seasonal 1 --harm_periods ${PERIODS}"

echo
echo "done. compare MSE/MAE in:"
echo "  logs/LongForecasting/${DATA}_mk_base.log"
echo "  logs/LongForecasting/${DATA}_mk_harm.log"
echo "  logs/LongForecasting/${DATA}_ik_base.log"
echo "  logs/LongForecasting/${DATA}_ik_harm.log"
