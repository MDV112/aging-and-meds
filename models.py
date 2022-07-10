from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
# import tensorflow as tf
# from tensorflow.keras import layers, losses
# from tensorflow.keras.models import Model
# from tensorflow import keras
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import Dense, Dropout, Activation, Conv1D, MaxPool1D, Flatten, BatchNormalization
# from tensorflow.keras import utils
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
# from tensorflow.keras.optimizers import SGD
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Models:
    def __init__(self, data, ylabel='id', model_name='rfc', mode='train', nbeats=250, **kwargs):
        self.model = None
        self.data = data
        self.nbeats = nbeats
        self.model_name = model_name
        self.mode = mode
        self.ylabel = ylabel
        self.kwargs = kwargs

    def set_model(self, mode='train'):
        if self.model_name == 'log_reg':
            self.model = LogisticRegression(**self.kwargs)
        elif self.model_name == 'svm':
            self.model = SVC(**self.kwargs)
        elif self.model_name == 'rfc':
            self.model = RandomForestClassifier(**self.kwargs)
        elif self.model_name == 'xgb':
            self.model = xgb.XGBClassifier(**self.kwargs)
        # elif self.model_name == 'AE':
        #     self.model = AE(self.nbeats)
        # elif self.model_name == 'CNN':
        #     if self.mode == 'train':
        #         self.model = CNN(len(np.unique(self.data.y_train_specific[self.ylabel])), self.nbeats)
        #     else:
        #         self.model = CNN(len(np.unique(self.data.y_test_specific[self.ylabel])), self.nbeats)
        # else:
        #     self.model = self.create_model(input_shape=self.data.dataset_name, **self.kwargs)
        return self.model
    #
    # @staticmethod
    # def create_model(input_shape=250, window_size=60, len_sub_window=10, n_filters_start=64, n_hidden_start=512, dropout=0.5, lr=0.01, momentum=0, **kwargs):
    #     model = Sequential()
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    #     tf.keras.backend.clear_session()
    #     # tf.executing_eagerly()
    #
    #     config = tf.compat.v1.ConfigProto()
    #     config.gpu_options.per_process_gpu_memory_fraction = 0.2
    #     tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
    #
    #     model.add(Conv1D(n_filters_start, len_sub_window, activation='relu', input_shape=(input_shape, 1)))
    #     model.add(BatchNormalization())
    #     model.add(Conv1D(2 * n_filters_start, len_sub_window, activation='relu'))
    #     model.add(BatchNormalization())
    #     model.add(MaxPool1D())
    #     model.add(Conv1D(4 * n_filters_start, len_sub_window, activation='relu'))
    #     model.add(BatchNormalization())
    #     model.add(Dropout(dropout))
    #     model.add(Flatten())
    #     model.add(Dense(n_hidden_start, activation='relu'))
    #     model.add(BatchNormalization())
    #     model.add(Dense(int(n_hidden_start / 2), activation='relu'))
    #     model.add(BatchNormalization())
    #     model.add(Dense(int(n_hidden_start / 4), activation='relu'))
    #     model.add(BatchNormalization())
    #     model.add(Dropout(dropout))
    #     model.add(Dense(1, activation='softmax'))
    #     optimizer = SGD(lr=lr, momentum=momentum)
    #     model.compile(optimizer=optimizer, metrics=['accuracy'], loss='binary_crossentropy')
    #     return model


# class AE(Model):
#
#     def __init__(self, nbeats=250):
#         super(AE, self).__init__()
#         self.encoder = tf.keras.Sequential([
#             layers.Dense(32, activation="relu"),
#             layers.Dense(16, activation="relu"),
#             layers.Dense(8, activation="relu")])
#
#         self.decoder = tf.keras.Sequential([
#             layers.Dense(16, activation="relu"),
#             layers.Dense(32, activation="relu"),
#             layers.Dense(nbeats, activation="sigmoid")])
#
#     def call(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded

# autoencoder = AE()


class CNN(nn.Module):

    def __init__(self, num_labels, nbeats, num_chann=[20, 10], ker_size=[10, 10], stride=[2, 2],
                 dial=[1, 1], pad=[0, 0], num_hidden=60):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, num_chann[0], kernel_size=ker_size[0], stride=stride[0], dilation=dial[0], padding=pad[0]),
            nn.BatchNorm1d(num_chann[0]),  # VERY IMPORTANT APPARENTLY
            nn.LeakyReLU(),
            # nn.Dropout(0.1),
        )
        self.L1 = np.floor(1+(1/stride[0])*(nbeats + 2*pad[0] - dial[0]*(ker_size[0]-1)-1))
        # self.L1 = np.floor(0.5*(nbeats-10) + 1)
        self.conv2 = nn.Sequential(
            nn.Conv1d(num_chann[0], num_chann[1], kernel_size=ker_size[1], stride=stride[1], dilation=dial[1], padding=pad[1]),
            nn.BatchNorm1d(num_chann[1]),
            nn.LeakyReLU(),
            # nn.Dropout(0.25)
        )
        self.L2 = np.floor(1+(1/stride[1])*(self.L1 + 2*pad[1] - dial[1]*(ker_size[1]-1)-1))
        self.L2 = torch.as_tensor(self.L2, dtype=torch.int64)
        # self.L2 = torch.tensor(np.floor(0.5*(self.L1-10) + 1), dtype=torch.int64)
        self.fc1 = nn.Sequential(
            nn.Linear(num_chann[1]*self.L2, num_hidden),
            nn.BatchNorm1d(num_hidden),
            nn.LeakyReLU(),
            # nn.ReLU(),
            # nn.Dropout(0.5)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(num_hidden, num_labels),
            nn.BatchNorm1d(num_labels),
            nn.ReLU(),
            # nn.Dropout(0.5)
        )

        # self.soft_max = nn.Softmax(1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        # collapse
        out = out.view(out.size(0), -1)
        # linear layer
        out = self.fc1(out)
        # output layer
        out = self.fc2(out)
        # out = self.soft_max(out) #NO NEED

        return out


class TruncatedCNN(nn.Module):

    def __init__(self, CNN_list):
        super(TruncatedCNN, self).__init__()

        self.conv1 = CNN_list[0]
        self.conv2 = CNN_list[1]
        self.fc = CNN_list[2]

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        # collapse
        out = out.view(out.size(0), -1)
        # linear layer
        out = self.fc(out)

        return out


class AdverserailCNN(nn.Module):

    def __init__(self, nbeats, num_chann=[20, 10], ker_size=[10, 10], stride=[2, 2],
                 dial=[1, 1], pad=[0, 0], num_hidden=60):
        super(AdverserailCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, num_chann[0], kernel_size=ker_size[0], stride=stride[0], dilation=dial[0], padding=pad[0]),
            nn.BatchNorm1d(num_chann[0]),  # VERY IMPORTANT APPARENTLY
            nn.LeakyReLU(),
            # nn.Dropout(0.1),
        )
        self.L1 = np.floor(1+(1/stride[0])*(nbeats + 2*pad[0] - dial[0]*(ker_size[0]-1)-1))
        # self.L1 = np.floor(0.5*(nbeats-10) + 1)
        self.conv2 = nn.Sequential(
            nn.Conv1d(num_chann[0], num_chann[1], kernel_size=ker_size[1], stride=stride[1], dilation=dial[1], padding=pad[1]),
            nn.BatchNorm1d(num_chann[1]),
            nn.LeakyReLU(),
            # nn.Dropout(0.25)
        )
        self.L2 = np.floor(1+(1/stride[1])*(self.L1 + 2*pad[1] - dial[1]*(ker_size[1]-1)-1))
        self.L2 = torch.as_tensor(self.L2, dtype=torch.int64)
        # self.L2 = torch.tensor(np.floor(0.5*(self.L1-10) + 1), dtype=torch.int64)
        self.fc1 = nn.Sequential(
            nn.Linear(num_chann[1]*self.L2, num_hidden),
            nn.BatchNorm1d(num_hidden),
            nn.LeakyReLU(),
            # nn.ReLU(),
            # nn.Dropout(0.5)
        )

        self.soft_max = nn.Softmax(1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        # collapse
        out = out.view(out.size(0), -1)
        # linear layer
        out = self.fc1(out)
        # out = self.soft_max(out)

        return out