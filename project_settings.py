import numpy as np
import torch


class ProSet:
    def __init__(self):

        self.entity = "morandv_team"  # wandb init
        self.med_mode = 'a'  # control or abk ('a')
        self.log_path = '/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/logs/'
        self.n_train = None
        self.n_val = None
        self.n_test = None
        self.n_individuals_train = None
        self.n_individuals_val = None
        self.n_individuals_test = None
        self.train_ages = None
        self.test_ages = None
        # data paths:
        # self.train_path = '/home/smorandv/DynamicalSystems/DynamicalSystems/running_scripts/single_exp/x_y.pkl'
        self.train_path = '/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/rr_data.pkl'
        # self.test_path = '/home/smorandv/DynamicalSystems/DynamicalSystems/running_scripts/single_exp/no_exp_test.pkl'
        self.test_path = '/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/rr_data.pkl'
        # splitting:
        self.proper = True
        self.val_size = 0.25
        self.seed = 42
        # cosine loss hyperparameters:
        self.b = -0.5  # -0.8
        self.lmbda = 1  # 1000
        self.flag = 0
        self.phi = np.pi
        # training hyperparameters:
        self.num_epochs = 3
        self.pretraining_epoch = 30
        self.reg_aug = 1/30
        self.reg_supp = 1/20
        self.lr = 0.0000001  # 0.000001
        self.batch_size = 2 ** 4
        self. weight_decay = 1  # optimizer
        # model hyperparmeters:
        self.e2_idx = 1
        self.ker_size = 5
        self.stride = 2
        self.dial = 1
        self.pad = 0
        # gpu:
        self.cpu = False
        self.mult_gpu = False
        self.device_ids = [1, 3, 4, 5, 6]  # always a list even if there is only one gpu
        self.device = torch.device('cuda:' + str(self.device_ids[0]) if not(self.cpu) else 'cpu')

        self.calc_metric = True
        self.sig_type = 'rr'
        self.feat2drop = []
