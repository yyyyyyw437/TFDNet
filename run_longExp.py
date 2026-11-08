import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

fix_seed = 2021


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(fix_seed)

parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

# basic config
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='TFDNet',
                    help='model name, options: [Autoformer, Informer, '
                         'Transformer, TFDNet, TFDNetHybrid]')

# data loader
parser.add_argument('--data', type=str, required=False, default='ETTm2', help='dataset type')
parser.add_argument('--root_path', type=str, default='./Datasets', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTm2.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=720, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help ='prediction sequence length')


# DLinear
parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')
#TFDNet
parser.add_argument('--individual_factor', type=int, default=7, help='individual_factor I for IK mode')
parser.add_argument('--kernel_num', type=int, default=4, help='Kernel number for MK mode')
parser.add_argument('--mode', type=str, default='MK', help='mode in seasonal blcok [gate,low_rank]')
parser.add_argument('--n_fft',nargs='+',type=int,default = [8,16,32], help='FFT length')
parser.add_argument('--use_freq_branch', type=int, default=1,
                    help='1: enable the TFDNet frequency branch')
parser.add_argument('--use_patch_branch', type=int, default=1,
                    help='1: enable the PatchTST-style temporal branch')
parser.add_argument('--use_channel_fusion', type=int, default=0,
                    help='1: enable lightweight cross-channel fusion')
parser.add_argument('--patch_len', type=int, default=16,
                    help='temporal patch length')
parser.add_argument('--patch_stride', type=int, default=8,
                    help='temporal patch stride')
parser.add_argument('--patch_d_model', type=int, default=32,
                    help='patch token width')
parser.add_argument('--patch_n_heads', type=int, default=4,
                    help='patch Transformer attention heads')
parser.add_argument('--patch_e_layers', type=int, default=1,
                    help='number of patch Transformer layers')
parser.add_argument('--patch_d_ff', type=int, default=64,
                    help='patch Transformer feed-forward width')
parser.add_argument('--channel_rank', type=int, default=4,
                    help='low rank of cross-channel residual mixing')
# Harmonic prior on the time-frequency kernels (ours). Default OFF = original TFDNet.
parser.add_argument('--harm_seasonal', type=int, default=0,
                    help='1: add harmonic prior to the seasonal TFB kernels (works for MK and IK)')
parser.add_argument('--harm_trend', type=int, default=0,
                    help='1: add harmonic prior to the trend TFB kernel')
parser.add_argument('--harm_periods', nargs='+', type=float, default=[],
                    help='initial fundamental periods, in SAMPLES (e.g. 24 168 for hourly data). '
                         'Empty means: no harmonic families (falls back to one shared kernel).')
parser.add_argument('--harm_alpha', type=float, default=0.15,
                    help='comb width relative to harmonic spacing (small = sharp)')
parser.add_argument('--harm_beta', type=float, default=0.05,
                    help='baseline weight of the free slot (fallback for non-harmonic bins)')
parser.add_argument('--harm_learn_period', type=int, default=1,
                    help='1: learn the fundamental periods end-to-end, 0: freeze them')
# Formers
parser.add_argument('--store_attn', default=False, help='whether to predict unseen future data')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=7,help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=64, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
# parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--seed', type=int, default=2021,
                    help='base random seed; each itr adds one')
parser.add_argument('--itr', type=int, default=3, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=1, help='train epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='Mixtureloss', help='loss function')
parser.add_argument('--lradj', type=str, default='TST', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default='0', help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

Exp = Exp_Main

if args.is_training:
    for ii in range(args.itr):
        current_seed = args.seed + ii
        set_seed(current_seed)
        # setting record of experiments
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_seed{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            current_seed,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)

        torch.cuda.empty_cache()
else:
    ii = 0
    current_seed = args.seed
    set_seed(current_seed)
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_seed{}_{}_{}'.format(args.model_id,
                                                                                                  args.model,
                                                                                                  args.data,
                                                                                                  args.features,
                                                                                                  args.seq_len,
                                                                                                  args.label_len,
                                                                                                  args.pred_len,
                                                                                                  args.d_model,
                                                                                                  args.n_heads,
                                                                                                  args.e_layers,
                                                                                                  args.d_layers,
                                                                                                  args.d_ff,
                                                                                                  args.factor,
                                                                                                  args.embed,
                                                                                                  args.distil,
                                                                                                  current_seed,
                                                                                                  args.des, ii)

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()
