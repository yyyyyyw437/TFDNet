from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, TFDNet, TFDNetPool
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.pool_distillation import TFDNetPoolDistillLoss
from torch.optim import lr_scheduler
from utils.metrics import metric
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import os
import time
import copy
from contextlib import nullcontext

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

class MixtureLoss(nn.Module):
    def __int__(self):
        super(MixtureLoss, self).__init__()

    def forward(self,pred,true):
        criterion =torch.mean((F.tanh(torch.abs(pred - true))) *torch.abs(pred-true) +(1 - F.tanh(torch.abs(pred - true))) * torch.square(pred-true))
        return criterion

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'TFDNet':TFDNet,
            'TFDNetPool': TFDNetPool,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)


        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):

        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.loss=='mse':
            criterion = nn.MSELoss()
        elif self.args.loss=='Mixtureloss':
            criterion = MixtureLoss()
        return criterion

    def _build_distillation_teacher(self):
        if self.args.model != 'TFDNetPool':
            raise ValueError(
                'Knowledge distillation currently requires TFDNetPool')
        if self.args.use_multi_gpu:
            raise ValueError(
                'Knowledge distillation currently supports one GPU only')

        teacher_path = os.path.abspath(self.args.kd_teacher_path)
        if os.path.isdir(teacher_path):
            teacher_path = os.path.join(teacher_path, 'checkpoint.pth')
        if not os.path.isfile(teacher_path):
            raise FileNotFoundError(
                'Teacher checkpoint not found: {}'.format(teacher_path))

        teacher_config = copy.deepcopy(self.args)
        teacher_config.tf_pool = 0
        teacher_config.kd_enable = 0
        teacher = TFDNetPool.Model(teacher_config).float().to(self.device)
        checkpoint = torch.load(teacher_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        elif isinstance(checkpoint, dict) and 'model' in checkpoint:
            checkpoint = checkpoint['model']
        state_dict = {
            key[7:] if key.startswith('module.') else key: value
            for key, value in checkpoint.items()
        }
        teacher.load_state_dict(state_dict, strict=True)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad = False
        return teacher

    def _forward_train(self, batch_x, teacher=None):
        if teacher is None:
            return self.model(batch_x), None, None
        with torch.no_grad():
            _, teacher_aux = teacher(batch_x, return_aux=True)
        outputs, student_aux = self.model(batch_x, return_aux=True)
        return outputs, student_aux, teacher_aux

    def _forward_training_outputs(
            self, batch_x, batch_x_mark, dec_inp, batch_y_mark, teacher):
        if teacher is not None:
            return self._forward_train(batch_x, teacher=teacher)
        if ('Linear' in self.args.model) or ('TFDNet' in self.args.model):
            return self.model(batch_x), None, None
        if self.args.output_attention:
            outputs = self.model(
                batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
        else:
            outputs = self.model(
                batch_x, batch_x_mark, dec_inp, batch_y_mark)
        return outputs, None, None




    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if ('Linear' in self.args.model) or ('TFDNet' in self.args.model):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs= self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if ('Linear' in self.args.model) or ('TFDNet' in self.args.model):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs= self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)


                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        distiller = None
        teacher = None
        if self.args.kd_enable:
            teacher = self._build_distillation_teacher()
            distiller = TFDNetPoolDistillLoss(
                prediction_weight=self.args.kd_pred_weight,
                coarse_weight=self.args.kd_coarse_weight,
                full_weight=self.args.kd_full_weight,
                branch_weight=self.args.kd_branch_weight,
                spectrum_weight=self.args.kd_spec_weight,
            )

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        scheduler = lr_scheduler.CosineAnnealingLR( model_optim, T_max = self.args.train_epochs)
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            distill_totals = {}

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim = 1).float().to(self.device)

                context = (
                    torch.cuda.amp.autocast()
                    if self.args.use_amp else nullcontext())
                with context:
                    outputs, student_aux, teacher_aux = (
                        self._forward_training_outputs(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark,
                            teacher))
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    if distiller is not None:
                        parts = distiller(
                            student_aux, teacher_aux, teacher,
                            feature_dim=f_dim)
                        loss = loss + parts["total"]
                        for key, value in parts.items():
                            if key != "total":
                                distill_totals.setdefault(key, []).append(
                                    value.item())
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            if distill_totals:
                summary = ', '.join(
                    '{}={:.7f}'.format(key, np.mean(values))
                    for key, values in distill_totals.items())
                print('Distillation Losses: {}'.format(summary))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                scheduler.step()
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model
    def save(self,setting,test=0):
        test_data, test_loader = self._get_data(flag = 'test')
        y_save = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float()
            y_save.append(batch_y)
        y_save = np.concatenate(y_save, axis = 0)
        data_name = self.args.model_id + 'pred.npy'
        folder_path = './test_data/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        np.save(folder_path + data_name, y_save)
    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if ('Linear' in self.args.model) or ('TFDNet' in self.args.model):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs= self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if ('Linear' in self.args.model) or ('TFDNet' in self.args.model):
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs= self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
        preds = np.concatenate(preds, axis = 0)
        trues = np.concatenate(trues, axis = 0)
        inputx = np.concatenate(inputx, axis = 0)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}, corr:{}'.format(mse, mae, rse, corr))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if ('Linear' in self.args.model) or ('TFDNet' in self.args.model):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs= self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if ('Linear' in self.args.model) or ('TFDNet' in self.args.model):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs,trend= self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
