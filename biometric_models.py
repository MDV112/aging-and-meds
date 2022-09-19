import os
import numpy as np
import torch
import torch.nn as nn
from data_loader import TorchDataset
import time
from sklearn.metrics import classification_report
from run import Run
from sklearn.model_selection import train_test_split
from extra_functions import BuildCNN
from collections import OrderedDict
from DSU import DistributionUncertainty
from torch.nn.utils import weight_norm
from losses import DomainGeneralization


class SiameseCNN(nn.Module):

    def __init__(self, nbeats, p, num_chann=[128, 128, 64], ker_size=10, stride=1,
                 dial=1, pad=0, drop_out=0.15, num_hidden=[32, 32], pool_ker_size=2):
        super(SiameseCNN, self).__init__()

        inputs = [num_chann, ker_size, stride, dial, pad, drop_out, num_hidden, pool_ker_size]
        # outputs = tuple([[x] for x in inputs if type(x) != list])
        for i, x in enumerate(inputs):
            if type(x) != list:
                inputs[i] = [x]
        num_chann, ker_size, stride, dial, pad, drop_out, num_hidden, pool_ker_size = tuple(inputs)
        inputs = [ker_size, stride, dial, pad, pool_ker_size]
        if len(num_chann) > 1:
            for i, x in enumerate(inputs):
                if len(x) == 1:
                    inputs[i] = x*len(num_chann)
        ker_size, stride, dial, pad, pool_ker_size = tuple(inputs)
        w = [len(p) for p in inputs]
        if not(w == [len(num_chann)]*len(w)):
            raise Exception('One of the convolution components does not equal number of channels')
        if len(drop_out) == 1:
            drop_out *= len(num_chann) + len(num_hidden)

        self.num_hidden = num_hidden
        self.conv = nn.ModuleList()
        # self.soft_max = nn.Softmax(dim=1)
        self.p = p
        self.DG = DomainGeneralization()
        self.dsu = DistributionUncertainty()
        self.conv.append(nn.Sequential(
            weight_norm(nn.Conv1d(1, num_chann[0], kernel_size=ker_size[0], stride=stride[0], dilation=dial[0], padding=pad[0])),
            nn.BatchNorm1d(num_chann[0]),  # VERY IMPORTANT APPARENTLY
            nn.MaxPool1d(pool_ker_size[0], stride=2),
            nn.ReLU(),
            # nn.Dropout(drop_out[0]),
        ))
        L = np.floor(1+(1/stride[0])*(nbeats + 2*pad[0] - dial[0]*(ker_size[0]-1)-1))
        L = np.floor(1 + (1/2)*(L - (pool_ker_size[0]-1)-1))  # calculating after pooling

        for idx in range(1, len(num_chann)):
            self.conv.append(nn.Sequential(
                weight_norm(nn.Conv1d(num_chann[idx - 1], num_chann[idx], kernel_size=ker_size[idx], stride=stride[idx], dilation=dial[idx], padding=pad[idx])),
                nn.BatchNorm1d(num_chann[idx]),
                nn.MaxPool1d(pool_ker_size[idx], stride=2),
                nn.ReLU(),
                # nn.Dropout(drop_out[idx)
            ))
            L = np.floor(1+(1/stride[idx])*(L + 2*pad[idx] - dial[idx]*(ker_size[idx]-1)-1))
            L = np.floor(1 + (1/2)*(L - (pool_ker_size[idx]-1)-1))  # calculating after pooling
        L = torch.as_tensor(L, dtype=torch.int64)
        if len(num_chann) == 1:
            idx = 0
        self.conv.append(nn.Sequential(
            nn.Linear(num_chann[idx]*L, num_hidden[0]),
            nn.Dropout(p=drop_out[idx]),
            nn.BatchNorm1d(num_hidden[0]),
            nn.ReLU(),
            # nn.ReLU(),
            # nn.Dropout(idx + 1)
        ))
        #todo: avoid activation function in last layer?
        for idx_lin in range(1, len(num_hidden)):
            self.conv.append(nn.Sequential(
                nn.Linear(num_hidden[idx_lin - 1], num_hidden[idx_lin]),
                nn.Dropout(p=drop_out[idx_lin]),
                nn.BatchNorm1d(num_hidden[idx_lin]),
                nn.ReLU(),
                # nn.ReLU(),
                # nn.Dropout(idx + idx_lin + 1)
            ))
        del self.conv[-1][-1]  # last Relu
        del self.conv[-1][-1]  # last batchNorm
        del self.conv[-1][-1]  # last dropout
        a=1

    def forward(self, x, flag_aug=False, flag_DSU=False, y=None):
        if not(flag_aug):
            if not (flag_DSU):
                out = self.conv[0](x)
                for i in range(1, len(self.conv) - len(self.num_hidden)):  # todo: check if range is true
                    out = self.conv[i](out)

                # collapse
                out = out.view(out.size(0), -1)
                for i in range(len(self.conv) - len(self.num_hidden), len(self.conv)):  # todo: check if range is true
                # linear layer
                    out = self.conv[i](out)  # NOTICE THAT HERE IT IS NOT CONVOLUTION BUT MLP
                # out = self.soft_max(out)
            else:
                out = self.conv[0](x)
                for i in range(1, len(self.conv) - len(self.num_hidden)):  # todo: check if range is true
                    out = self.conv[i](self.dsu(out))

                # collapse
                out = out.view(out.size(0), -1)
                for i in range(len(self.conv) - len(self.num_hidden), len(self.conv)):  # todo: check if range is true
                    # linear layer
                    out = self.conv[i](out)  # NOTICE THAT HERE IT IS NOT CONVOLUTION BUT MLP
                # out = self.soft_max(out)

            return out
        else:
            ##### This one performes dsu either way (but not on linear layers)
            out = self.conv[0](x)
            for j in range(1, self.p.e2_idx):
                out = self.conv[j](self.dsu(out))
            aug = self.create_aug(out)
            ##### NOTICE STATRTING FROM self.p.e2_idx #######
            for i in range(self.p.e2_idx, len(self.conv) - len(self.num_hidden)):  # todo: check if range is true
                out = self.conv[i](self.dsu(out))
                aug = self.conv[i](self.dsu(aug))
            aug_loss = self.L_aug(out, aug)
            supp_loss = self.L_supp(out, y)

            # collapse
            out = out.view(out.size(0), -1)

            for i in range(len(self.conv) - len(self.num_hidden), len(self.conv)):  # todo: check if range is true
                # linear layer
                out = self.conv[i](out)  # NOTICE THAT HERE IT IS NOT CONVOLUTION BUT MLP
            # out = self.soft_max(out)

            # https://discuss.pytorch.org/t/training-network-with-multiple-outputs-with-multi-gpus/6344/2

            # d = OrderedDict()
            # d['out'] = out
            # d['aug_loss'] = aug_loss

            return out, aug_loss, supp_loss


class BuildTCN():
    def __init__(self):
        super(BuildTCN, self).__init__()