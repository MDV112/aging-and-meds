import numpy as np
from deep_models import Advrtset
import torch.nn as nn
import torch

class ProSet():
    def __init__(self, mult_gpu=1):
        self.entity = "morandv_team"  #wandb init
        # data paths:
        self.train_path = '/home/smorandv/DynamicalSystems/DynamicalSystems/running_scripts/single_exp/x_y.pkl'
        self.test_path = '/home/smorandv/DynamicalSystems/DynamicalSystems/running_scripts/single_exp/no_exp_test.pkl'
        # cosine loss parameters:
        self.b = -0.8
        self.lmbda = 1000
        self.flag = 0
        self.phi = np.pi
        # training hyperparameters:
        self.num_epochs = 4
        self.lr = 0.001
        self.batch_size = 2 ** 10
        self. weight_decay = 1  # optimizer
        # model hyperparmeters:
        self.ker_size = 2
        self.stride = 1
        self.dial = 1
        # gpu:
        self.mult_gpu = mult_gpu
        self.device_ids = [1, 3, 4, 5, 6]  # always a list even if there is only one gpu
        self.device = torch.device('cuda:' + str(self.device_ids[0]) if torch.cuda.is_available() else 'cpu')
