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

if __name__ == '__main__':
    # p = ProSet(read_txt=True, txt_path='/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/logs/Jul-25-2022_11_23_04/')
    p = ProSet()
    # wandb.login()
    # wandb.init('test', entity=p.entity)
    # config = dict(n_epochs=p.num_epochs, batch_size=p.batch_size)
    print('Med mode is : {}'.format(p.med_mode))
    if p.train_path == '/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/rr_human_train_data.pkl':
        p.human_flag = 1
        p.test_path = '/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/rr_human_test_data.pkl'
    else:
        p.human_flag = 0
    tr_dataset_1 = load_datasets(p.train_path, p, human_flag=p.human_flag, samp_per_id=p.samp_per_id)
    x_tr, y_tr, x_val, y_val = split_dataset(tr_dataset_1, p)
    tr_dataset_1, val_dataset_1, scaler1 = scale_dataset(x_tr, y_tr, x_val, y_val)
    tr_dataset_2 = load_datasets(p.train_path, p, mode=1, human_flag=p.human_flag, samp_per_id=p.samp_per_id)
    x_tr, y_tr, x_val, y_val = split_dataset(tr_dataset_2, p)
    tr_dataset_2, val_dataset_2, scaler2 = scale_dataset(x_tr, y_tr, x_val, y_val, mode=1)

    ##### NOTICE  WHEN TO USE SCALER FOR TESTING. IF WE USE FULL DATASET FOR LEARNING THUS OUTSIDE SCALER IS NOT NEEDED.

    model = Advrtset(tr_dataset_1.x.shape[1], p, ker_size=p.ker_size, stride=p.stride, dial=p.dial,
                     drop_out=p.drop_out, num_chann=p.num_chann, num_hidden=p.num_hidden,).to(p.device)
    if p.mult_gpu:
        model = nn.DataParallel(model, device_ids=p.device_ids)

    # optimizer = torch.optim.Adam(model.parameters(), lr=p.lr, weight_decay=p.weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), lr=p.lr)
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

    #IN VALIDATION SET, IT SHOULD BE VAL_DATASET. ENLARGE
    # NUM OF MICE. CHANGE SHUFFLE AS NEEDED FOR DOMAIN. CHECK WHY VAL AND TRAIN LOSS ARE NOT THE SAME.
    train_model(model, p, optimizer, trainloader1, trainloader2, valloader1, valloader2)

    # wandb.finish()
