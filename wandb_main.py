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


if __name__ == '__main__':
    p = ProSet()
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    device_list = [0,1,2,3,4,5,6,7]
    for device in device_list:
        if memory_free_values[device] > 4000:
            p.device = device
            break
    wandb.login()
    empty_parser = argparse.ArgumentParser()
    parser = init_parser(parent=empty_parser)
    p = parser.parse_args()
    with wandb.init('test2', entity=p.entity, config=p):
        # wandb.run.name = 'stam'
        # wandb.run.save()
        # config = wandb.config
        p = argparse.Namespace(**wandb.config)
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
            model = Advrtset(tr_dataset_1.x.shape[1], p, ker_size=p.ker_size, stride=p.stride, pool_ker_size=p.pool_ker_size, dial=p.dial,
                             drop_out=p.drop_out, num_chann=p.num_chann, num_hidden=p.num_hidden,).to(p.device)
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

            wandb.watch(model, cosine_loss, log="all", log_freq=30)
            train_model(model, p, optimizer, trainloader1, trainloader2, valloader1, valloader2)

            wandb.finish()


    #todo: WE CAN LOAD THE BEST MODEL AND APPLY ON THE TEST



    ####### TAKING FULL TRAINING SET AND RUN with TEST
        else:
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

            for i in range(3):
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
                ############## VALIDATION SET ###########################
                tsloader1 = torch.utils.data.DataLoader(
                    ts_dataset_1, batch_size=p.batch_size, shuffle=False, num_workers=0, drop_last=True)
                tsloader2 = torch.utils.data.DataLoader(
                    ts_dataset_2, batch_size=p.batch_size, shuffle=False, num_workers=0, drop_last=True)

                train_model(model, p, optimizer, trainloader1, trainloader2, tsloader1, tsloader2)

            wandb.finish()

