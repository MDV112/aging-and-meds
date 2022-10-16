import numpy as np
import torch
import torch.nn as nn
from data_loader import TorchDataset
import time
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pickle
from deep_models import Advrtset
from deep_models import cosine_loss
from tqdm import tqdm
from torch.utils.data import Dataset
import wandb
from single_model_functions import *
from project_settings import ProSet
from project_settings import init_parser
import subprocess as sp
import os
import argparse
from tsai.all import *


if __name__ == '__main__':
    ## TO RUN WANDB, CHANGE WANDB_ENABLE TO TRUE AND ACTIVATE WANDB.INIT WITH ADEQUATE INDENTATION
    p = ProSet()
    p.wandb_enable = False
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    device_list = [0,1,2,3,4,5,6,7]
    for device in device_list:
        if memory_free_values[device] > 4000:
            p.device = device
            break
    if p.wandb_enable:
        wandb.login()
    empty_parser = argparse.ArgumentParser()
    parser = init_parser(parent=empty_parser)
    p = parser.parse_args()
    # with wandb.init(entity=p.entity, config=p):
    #     p = argparse.Namespace(**wandb.config)
    p.device = device
    p.wandb_enable = False
    p.test_mode = True
    if not(p.test_mode):
        print('Med mode is : {}'.format(p.med_mode))
        if p.human_flag:
            p.train_path = '/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/rr_human_train_data.pkl'
            p.test_path = '/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/rr_human_test_data.pkl'
        else:
            p.train_path = '/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/rr_data.pkl'
            p.test_path = p.train_path
        tr_dataset_1 = load_datasets(p.train_path, p, human_flag=p.human_flag, samp_per_id=p.samp_per_id)
        x_tr, y_tr, x_val, y_val = split_dataset(tr_dataset_1, p)
        tr_dataset_1, val_dataset_1, scaler1 = scale_dataset(x_tr, y_tr, x_val, y_val)
        tr_dataset_2 = load_datasets(p.train_path, p, mode=1, human_flag=p.human_flag, samp_per_id=p.samp_per_id)
        x_tr, y_tr, x_val, y_val = split_dataset(tr_dataset_2, p)
        tr_dataset_2, val_dataset_2, scaler2 = scale_dataset(x_tr, y_tr, x_val, y_val, mode=1)

        torch.manual_seed(p.seed)


        model = TCN(1,1, layers=[128, 128, 128], ks=30)
        model = model.to(p.device)
        # model = Advrtset(tr_dataset_1.x.shape[1], p, ker_size=p.ker_size, stride=p.stride, pool_ker_size=p.pool_ker_size, dial=p.dial,
        #                  drop_out=p.drop_out, num_chann=p.num_chann, num_hidden=p.num_hidden,).to(p.device)
        if p.mult_gpu:
            model = nn.DataParallel(model, device_ids=p.device_ids)

        optimizer = torch.optim.Adam(model.parameters(), lr=p.lr, weight_decay=p.weight_decay)  # , momentum=p.momentum, dampening=p.dampening
        ############## TRAINING SET ##########################
        trainloader1 = torch.utils.data.DataLoader(
            tr_dataset_1, batch_size=p.batch_size, shuffle=False, num_workers=0, drop_last=True)
        trainloader2 = torch.utils.data.DataLoader(
            tr_dataset_2, batch_size=p.batch_size, shuffle=False, num_workers=0, drop_last=True)
        ############## VALIDATION SET ###########################
        valloader1 = torch.utils.data.DataLoader(
            val_dataset_1, batch_size=p.batch_size, shuffle=False, num_workers=0, drop_last=True)
        valloader2 = torch.utils.data.DataLoader(
            val_dataset_2, batch_size=p.batch_size, shuffle=False, num_workers=0, drop_last=True)
        if p.wandb_enable:
            wandb.watch(model, cosine_loss, log="all", log_freq=30)
        train_model(model, p, optimizer, trainloader1, trainloader2, valloader1, valloader2)
        if p.wandb_enable:
            wandb.finish()


    #todo: WE CAN LOAD THE BEST MODEL AND APPLY ON THE TEST



    ####### TAKING FULL TRAINING SET AND RUN with TEST
    else:
        exp_name = ['abk_min_val_loss','control_min_val_loss','both_min_val_loss','abk_max_val_acc','control_max_val_acc',
                    'control_max_val_acc'] # for now we take only the max val_acc values and not the highest mean
        # chosen_epoch = [80, 200, 319, 274, 410, 10]
        hyperparameter_dict = {'abk_min_val_loss': dict(num_epochs = 3, b = -0.5928304672016774, batch_size = 16, drop_out = 0.33817387726694303,
                                                        ker_size = 30, lmbda = 0.15788530065565576, lr = 2.779077656310891e-07,
                                                        med_mode = 'a', momentum = 0.6971956087960214, weight_decay = 9.85273609898957),
                               'control_min_val_loss': dict(num_epochs = 200, b = 0.4759263501595812, batch_size = 8, drop_out = 0.12609304729484835,
                                                            ker_size = 30, lmbda = 9.613353037151509, lr = 3.435639818473589e-07,
                                                            med_mode = 'c', momentum = 0.755330893385039, weight_decay = 1.1752777398646077),
                               'both_min_val_loss': dict(num_epochs = 319, b = 0.4770143418377288, batch_size = 128, drop_out = 0.30722983323578223,
                                                         ker_size = 10, lmbda = 1.3150498005398192, lr = 4.078531801074058e-07,
                                                         med_mode = 'both', momentum = 0.10744132018019596, weight_decay = 0.17254862160126153),
                               'abk_max_val_acc': dict(num_epochs = 274, b = -0.9562638379209744, batch_size = 128, drop_out = 0.17056978300508333,
                                                       ker_size = 10, lmbda = 3.116931692726408, lr = 4.052054605371503e-09,
                                                       med_mode = 'a', momentum = 0.1242686110085278, weight_decay = 9.940616806701842),
                               'control_max_val_acc': dict(num_epochs = 410, b = -0.35933104620886336, batch_size = 8, drop_out = 0.24671384595337603,
                                                           ker_size = 30, lmbda = 0.0002452993185841912, lr = 2.2402385543478617e-07,
                                                           med_mode = 'c', momentum = 0.3939188582826322, weight_decay = 7.947725851063474),
                               'both_max_val_acc': dict(num_epochs = 10, b = 0.49879753665203497, batch_size = 64, drop_out = 0.372252749360447,
                                                        ker_size = 10, lmbda = 0.17068996497656697, lr = 3.827153538598344e-07,
                                                        med_mode = 'both', momentum = 0.3889915168297784, weight_decay = 0.1554402334425353)}
        lines = []
        for idx_exp, exp in enumerate(hyperparameter_dict.keys()):
            for var_name in p.__dict__:
                if var_name in hyperparameter_dict[exp].keys():
                    p.__dict__[var_name] = hyperparameter_dict[exp][var_name]
            if p.human_flag:
                p.train_path = '/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/rr_human_train_data.pkl'
                p.test_path = '/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/rr_human_test_data.pkl'
            else:
                p.train_path = '/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/rr_data.pkl'
                p.test_path = p.train_path
            tr_dataset_1 = load_datasets(p.train_path, p, human_flag=p.human_flag, samp_per_id=p.samp_per_id)
            ts_dataset_1 = load_datasets(p.test_path, p, human_flag=p.human_flag, samp_per_id=p.samp_per_id, train_mode=False)
            tr_dataset_1, ts_dataset_1 = scale_dataset(tr_dataset_1, ts_dataset_1, should_scale=False)
            tr_dataset_2 = load_datasets(p.train_path, p, human_flag=p.human_flag, samp_per_id=p.samp_per_id, mode=1)
            ts_dataset_2 = load_datasets(p.test_path, p, human_flag=p.human_flag, samp_per_id=p.samp_per_id, train_mode=False, mode=1)
            tr_dataset_2, ts_dataset_2 = scale_dataset(tr_dataset_2, ts_dataset_2, mode=1, should_scale=False)

            acc_vector = torch.zeros(3)
            for i in range(3):
                if i == 2:
                    p.mkdir = True
                else:
                    p.mkdir = False
                model = Advrtset(tr_dataset_1.x.shape[1], p, ker_size=p.ker_size, stride=p.stride, pool_ker_size=p.pool_ker_size,
                                 dial=p.dial,
                                 drop_out=p.drop_out, num_chann=p.num_chann, num_hidden=p.num_hidden, ).to(p.device)
                if p.mult_gpu:
                    model = nn.DataParallel(model, device_ids=p.device_ids)

                optimizer = torch.optim.Adam(model.parameters(), lr=p.lr,
                                            weight_decay=p.weight_decay)  #, momentum=p.momentum, dampening=p.dampening
                ############## TRAINING SET ##########################
                trainloader1 = torch.utils.data.DataLoader(
                    tr_dataset_1, batch_size=p.batch_size, shuffle=False, num_workers=0, drop_last=True)
                trainloader2 = torch.utils.data.DataLoader(
                    tr_dataset_2, batch_size=p.batch_size, shuffle=False, num_workers=0, drop_last=True)
                ############## TESTING SET ###########################
                tsloader1 = torch.utils.data.DataLoader(
                    ts_dataset_1, batch_size=p.batch_size, shuffle=False, num_workers=0, drop_last=True)
                tsloader2 = torch.utils.data.DataLoader(
                    ts_dataset_2, batch_size=p.batch_size, shuffle=False, num_workers=0, drop_last=True)

                train_model(model, p, optimizer, trainloader1, trainloader2, tsloader1, tsloader2)
                _, _, acc_vector[i] = eval_model(model, p, 1, tsloader1, tsloader2)
                if p.wandb_enable:
                    wandb.finish()
            lines += ['In exp. {}, mean test results were  {:.2f}% with {:.2f}% std'.format(exp, 100*acc_vector.nanmean(),
                                                                                            100*acc_vector.std())]
        with open(p.log_path + '/README.txt', 'w') as f:
            f.write('\n'.join(lines))
