import numpy as np
import torch
from torch.utils.data import Dataset
import argparse

def init_parser(parent, add_help=False):
    """
    This function converts ProSet class to parser.
    :param parent: empty_parser (in wandb_main)
    :param add_help:
    :return: parser
    """
    parser = argparse.ArgumentParser(parents=[parent], add_help=add_help)
    p = ProSet()
    for idx, (attr, val) in enumerate(vars(p).items()):
        parser.add_argument('--' + attr, default=val)
        # example: parser.add_argument('--entity' , default='morandv_team', type=str, help='wandb init')
    return parser


class ProSet:
    def __init__(self, read_txt=False, txt_path=None):
        self.entity = "morandv_team"  # wandb init
        self.med_mode = 'c'  # control ('c'), abk ('a') or both ('both')
        self.log_path = '/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/logs/'
        self.mkdir = False
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
        # self.train_path = '/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/rr_data.pkl'
        self.train_path = '/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/rr_human_train_data.pkl'
        # self.test_path = '/home/smorandv/DynamicalSystems/DynamicalSystems/running_scripts/single_exp/no_exp_test.pkl'
        self.test_path = '/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/rr_data.pkl'
        # splitting:
        self.proper = True
        self.remove_mean = False
        self.test_mode = False
        self.val_size = 0.2
        self.seed = 42
        self.samp_per_id = 60
        self.human_flag = False
        self.wandb_enable = True
        # early stopping:
        self.patience = 5
        self.lr_ker_size = 5
        self.lr_counter = 20
        self.lr_factor = 0.9  # has to be in (0,1)
        self.curr_train_epoch = 1
        self.curr_val_epoch = 1
        self.first_epoch_time = 200
        self.epoch_factor = 30
        self.error_txt = None
        # cosine loss hyperparameters:
        self.b = -0.5  # -0.8
        self.lmbda = 1.0  # 1000
        self.flag = 0
        self.phi = np.pi
        # training hyperparameters:
        self.num_epochs = 500
        self.pretraining_epoch = 0
        self.flag_DSU = False
        self.reg_aug = 0.0  # silenced either way for now in the code
        self.reg_supp = 0.0 # silenced either way for now in the code
        self.lr = 1e-8   # 0.000001
        self.momentum = 0.7
        self.dampening = 0.0
        self.batch_size = 2 ** 4
        self. weight_decay = 1.0  # optimizer
        # model hyperparmeters:
        self.e2_idx = 2
        self.stride = 1
        self.dial = 1
        self.pad = 0
        self.num_chann = [128, 128, 64]  # probably should be ascending order
        self.ker_size = 15  # 10
        self.pool_ker_size = 2
        self.drop_out = 0.25  # 0.15
        self.num_hidden = [128, 64, 32]  # [128, 64, 32]
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


class HRVDataset(Dataset):

    def __init__(self, x, y, mode=0):
        # self.x = x.clone().detach().requires_grad_(True)
        super().__init__()
        self.x = x.copy()
        self.y = y.copy()
        self.mode = mode


    def __len__(self):
        return self.x.shape[0]  # since we transpose

    def __getitem__(self, idx):
        """
        Two datasets are build for Siamease network. We have to make sure that within a batch
        the comparison is made 50% of the time with negative example and 50% with negative. The two
        datasets have shuffle=False. mode=1 will be only for the second dataset
        :param idx: index of sample
        :param mode: used for second dataset to have 50% of the mice compared to the first dataset to be different
        :return:
        """
        # np.random.seed(5)
        if self.mode:
            y = self.y[idx, 0]
            r = np.random.randint(2)  # 50%-50%
            if r:  # find negative example
                neg_list = np.argwhere(self.y[:, 0] != y)
                idx = neg_list[np.random.randint(0, len(neg_list))].item()
            else:
                idx_temp = None
                pos_list = np.argwhere(self.y[:, 0] == y)
                while (idx_temp is None) or (idx_temp == idx):  # avoid comparing the same signals
                    idx_temp = pos_list[np.random.randint(0, len(pos_list))].item()
                idx = idx_temp
        x = self.x[idx:idx+1, :]
        y = self.y[idx, :]
        sample = (torch.from_numpy(x).requires_grad_(True).type(torch.FloatTensor), torch.from_numpy(y).type(torch.IntTensor))  # just here convert to torch
        return sample


class TE2TDataset(Dataset):
    pass

# if __name__ == '__main__':
#     empty_parser = argparse.ArgumentParser()
#     parser = init_parser(parent=empty_parser)
#     run_config = parser.parse_args()  # parse args from cli
#     a=1