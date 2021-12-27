import os
import numpy as np
import torch
import torch.nn as nn
from data_loader import TorchDataset
import time
from sklearn.metrics import classification_report
from run import Run
from sklearn.model_selection import train_test_split


class BuildCNN(nn.Module):

    def __init__(self, nbeats, num_labels=0, num_chann=[20, 10, 30], ker_size=10, stride=2,
                 dial=1, pad=0, drop_out=0.15, num_hidden=[60, 40]):
        super(BuildCNN, self).__init__()
        if num_labels != 0:
            num_hidden.append(num_labels)
        inputs = [num_chann, ker_size, stride, dial, pad, drop_out, num_hidden]
        # outputs = tuple([[x] for x in inputs if type(x) != list])
        for i, x in enumerate(inputs):
            if type(x) != list:
                inputs[i] = [x]
        num_chann, ker_size, stride, dial, pad, drop_out, num_hidden = tuple(inputs)
        inputs = [ker_size, stride, dial, pad]
        if len(num_chann) > 1:
            for i, x in enumerate(inputs):
                if len(x) == 1:
                    inputs[i] = x*len(num_chann)
        ker_size, stride, dial, pad = tuple(inputs)
        w = [len(p) for p in inputs]
        if not(w == [len(num_chann)]*len(w)):
            raise Exception('One of the convolution components does not equal number of channels')
        if len(drop_out) == 1:
            drop_out *= len(num_chann) + len(num_hidden)

        self.num_hidden = num_hidden
        self.conv = nn.ModuleList()
        self.soft_max = nn.Softmax(dim=1)

        self.conv.append(nn.Sequential(
            nn.Conv1d(1, num_chann[0], kernel_size=ker_size[0], stride=stride[0], dilation=dial[0], padding=pad[0]),
            nn.BatchNorm1d(num_chann[0]),  # VERY IMPORTANT APPARENTLY
            nn.LeakyReLU(),
            nn.Dropout(drop_out[0])
        ))
        L = np.floor(1+(1/stride[0])*(nbeats + 2*pad[0] - dial[0]*(ker_size[0]-1)-1))

        for idx in range(1, len(num_chann)):
            self.conv.append(nn.Sequential(
                nn.Conv1d(num_chann[idx - 1], num_chann[idx], kernel_size=ker_size[idx], stride=stride[idx], dilation=dial[idx], padding=pad[idx]),
                nn.BatchNorm1d(num_chann[idx]),
                nn.LeakyReLU(),
                nn.Dropout(drop_out[idx])
            ))
            L = np.floor(1+(1/stride[idx])*(L + 2*pad[idx] - dial[idx]*(ker_size[idx]-1)-1))
        L = torch.as_tensor(L, dtype=torch.int64)
        # if len(num_chann) == 1:
        #     idx = 0
        self.conv.append(nn.Sequential(
            nn.Linear(num_chann[idx]*L, num_hidden[0]),
            nn.BatchNorm1d(num_hidden[0]),
            nn.LeakyReLU(),
            nn.Dropout(drop_out[idx + 1])
        ))
        #todo: avoid activation function in last layer?
        for idx_lin in range(1, len(num_hidden)):
            self.conv.append(nn.Sequential(
                nn.Linear(num_hidden[idx_lin - 1], num_hidden[idx_lin]),
                nn.BatchNorm1d(num_hidden[idx_lin]),
                nn.LeakyReLU(),
                nn.Dropout(drop_out[idx + idx_lin + 1])
            ))

    def forward(self, x):
        out = self.conv[0](x)
        for i in range(1, len(self.conv) - len(self.num_hidden)):  # todo: check if range is true
            out = self.conv[i](out)

        # collapse
        out = out.view(out.size(0), -1)
        for i in range(len(self.conv) - len(self.num_hidden), len(self.conv)):  # todo: check if range is true
            # linear layer
            out = self.conv[i](out)
        # out = self.soft_max(out)

        return out
