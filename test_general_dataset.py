from data_loader import Dataloader
# from run import Run
from models import Models
from torch.utils.data import Dataset
from run import *
import tensorflow as tf
from sklearn.metrics import f1_score, make_scorer
from data_loader import TorchDataset
# from deep import AE
import torch.nn as nn
import torch
import torch
from dim_reduction import DimRed
from deep_models import DeepModels
# from deep import CNN
# from data_loader import TorchDataset
import seaborn as sns
from deep_models import ContrastiveLoss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler
import pickle


# todo: the inputs are Dataframes: THINK IF WE SHOULD REALLY SCALE OR NOT
# scaled_train =  [tr_x_c, tr_x_a, tr_y_c, tr_y_a]
# scaled_test[ts_x_c, ts_x_a, ts_y_c, ts_y_a]


class TorchDatasetEnd2End(Dataset):

    def __init__(self, data, max_age, net_mode='mlp', T=3):
        # data = [tr_x_c, tr_x_a, tr_y_c, tr_y_a] or [ts_x_c, ts_x_a, ts_y_c, ts_y_a]
        self.data = data
        self.net_mode = net_mode  # either 'koopman' or 'mlp'
        self.T = T
        tag = self.data[0].index  # all data elements have the same tags at the same order
        y_mlp = np.array([max_age[int(x)] for x in tag])
        self.y_mlp = y_mlp - np.asarray(self.data[2]['Age'], int)  # WE NEED THE TIME FROM THE PREDICTIONS OF KOOPMAN THUS WE NEED THE AGE AT Y DATA
        self.X = torch.stack([torch.from_numpy(self.data[0].values.astype(np.double)), torch.from_numpy(self.data[1].values.astype(np.double))]).float()
        self.Y = torch.stack([torch.from_numpy(self.data[2].values.astype(np.double)), torch.from_numpy(self.data[3].values.astype(np.double))]).float()
        self.y_mlp = torch.from_numpy(self.y_mlp)

    def __len__(self):
        return int(self.X.shape[1]/self.T)
        #
        # if self.net_mode == 'koopman':
        #     return int(self.X.shape[1]/self.T)
        # else:
        #     return self.X.shape[0]

    def __getitem__(self, idx, med_mode=0):
        if self.net_mode == 'koopman':
            pass
            # x = self.X[:, self.T*idx:self.T*(idx+1), :]
            # y = self.Y[:, self.T*idx:self.T*(idx+1), :]
            # sample = {GT_TENSOR_INPUTS_KEY: x, GT_TENSOR_PREDICITONS_KEY: y}
        else:
            x = self.X[:, idx:idx+1, :]
            # x.squeeze()
            # x = x.T  # not sure it is needed
            y = self.y_mlp[idx]
            # if self.hash_id is not None:
            #     y = torch.tensor(self.hash_id[tag.item()], dtype=torch.int64)  # convert labels ranging fro 0 to C-1
            # else:
            # y = torch.tensor(tag, dtype=torch.int64)
            sample = (x, y)
            return sample


class MLP(torch.nn.Module):

    def __init__(self, input_shape=19, num_hidden=[25, 10], dropout=0.2):
        super(MLP, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_shape, num_hidden[0]),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(num_hidden[0], num_hidden[1]),
            nn.ReLU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(num_hidden[1], 1),
        )

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        out = out.squeeze()
        return out