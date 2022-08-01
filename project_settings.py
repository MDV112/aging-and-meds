import numpy as np
import torch


class ProSet:
    def __init__(self, read_txt=False, txt_path=None):
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
        # self.train_path = '/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/rr_human_train_data.pkl'
        # self.test_path = '/home/smorandv/DynamicalSystems/DynamicalSystems/running_scripts/single_exp/no_exp_test.pkl'
        self.test_path = '/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/rr_data.pkl'
        # splitting:
        self.proper = True
        self.val_size = 0.2
        self.seed = 42
        self.samp_per_id = 60
        self.human_flag = False
        # cosine loss hyperparameters:
        self.b = -0.5  # -0.8
        self.lmbda = 1  # 1000
        self.flag = 0
        self.phi = np.pi
        # training hyperparameters:
        self.num_epochs = 50
        self.pretraining_epoch = 201
        self.reg_aug = 0.0
        self.reg_supp = 0.0
        self.lr = 1e-5   # 0.000001
        self.momentum = 0.7
        self.dampening = 0
        self.batch_size = 2 ** 4
        self. weight_decay = 1  # optimizer
        # model hyperparmeters:
        self.e2_idx = 2
        self.stride = 1
        self.dial = 1
        self.pad = 0
        self.num_chann = [128, 128, 64]  # [128, 128, 64]
        self.ker_size = 15  # 10
        self.pool_ker_size = 2
        self.drop_out = 0.25  # 0.15
        self.num_hidden = [128, 64, 32]
        # gpu:
        self.cpu = False
        self.mult_gpu = False
        self.device_ids = [1, 3, 4, 5, 6]  # always a list even if there is only one gpu
        # self.device = torch.device('cuda:' + str(self.device_ids[0]) if not(self.cpu) else 'cpu')
        self.device = torch.device('cuda:' + str(5) if not (self.cpu) else 'cpu')

        self.calc_metric = True
        self.sig_type = 'rr'
        self.feat2drop = []
        if read_txt:
            if txt_path is None:
                raise Exception('txt path must be given when read_txt is true')
            else:
                self.read_txt(txt_path)
    def read_txt(self, txt_path):
        with open(txt_path + 'README.txt') as f:
            lines = f.readlines()
            lines = lines[4:]
            if len(lines) != len(vars(self)):
                raise Exception('Chosen txt does not have the same attributes of current run')
            else:
                for idx, attr, _ in enumerate(vars(self).items()):
                    self.attr = lines[idx][1+lines[idx].find('='):lines[idx].find('\n')].strip()
        a=1
        pass
