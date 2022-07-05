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
import pickle
from comp2ecg_single_model import HRVDataset

#todo: ADD THE SCALING AND SPLITTING TO VALIDATION


def load_datasets(full_pickle_path, med_mode='c', mode=0):
    with open(full_pickle_path, 'rb') as f:
        e = pickle.load(f)
        data = e[0:4]
        # max_age = e[-1]
    x_c, x_a = (data[0], data[1])
    y_c, y_a = (x_c.index, x_a.index)
    if med_mode == 'c':
        label_dataset = y_c
        np_dataset = np.array(x_c.values, dtype=np.float)
        dataset = HRVDataset(np_dataset, label_dataset, mode=mode)
    elif med_mode == 'a':
        label_dataset = y_a
        np_dataset = np.array(x_a.values, dtype=np.float)
        dataset = HRVDataset(np_dataset, label_dataset, mode=mode)
    else:
        pass
    return dataset


def train_model(model, p, *args, calc_metric=0):
    # * args should be: optimizer, trainloader1, trainloader1, valloader1, valloader1
    model.train()
    train_epochs(model, p, calc_metric, *args)  # without "*" it would have built a tuple in a tuple


def train_epochs(model, p, calc_metric, *args):
    for epoch in range(1, p.num_epochs + 1):
        epoch_time = time.time()
        training_loss = train_batches(model, p, calc_metric, *args)
        training_loss /= len(args[1])  # len of trainloader
        validation_loss = eval_model(model, p, calc_metric=calc_metric, *args)
        validation_loss /= len(args[3])  # len of valloader
        log = "Epoch: {} | Training loss: {:.4f}  | Validation loss: {:.4f}   ".format(epoch, training_loss, validation_loss)
        epoch_time = time.time() - epoch_time
        log += "Epoch Time: {:.2f} secs".format(epoch_time)
        print(log)
    pass


def train_batches(model, p, calc_metric, *args):
    optimizer, dataloader1, dataloader2, _, _ = args
    running_loss = 0.0
    for i, data in enumerate(tqdm(zip(dataloader1, dataloader2), total=len(dataloader1)), 0):
        # get the inputs
        inputs1, labels1 = data[0]
        inputs2, labels2 = data[1]
        # send them to device
        inputs1 = inputs1.to(p.device)
        labels1 = labels1.to(p.device)
        inputs2 = inputs2.to(p.device)
        labels2 = labels2.to(p.device)
        # forward
        outputs1 = model(inputs1)  # forward pass
        outputs2 = model(inputs2)
        # backward + optimize
        loss = cosine_loss(outputs1, outputs2, labels1, labels2, flag=p.flag, lmbda=p.lmbda, b=p.b)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)  # backpropagation
        optimizer.step()
        # accumulate mean loss
        running_loss += loss.data.item()
        if calc_metric:
            pass
    return running_loss


def eval_model(model, p, *args, calc_metric=0):
    # * args should be: optimizer, dataloader1, dataloader2
    model.eval()
    eval_loss = eval_batches(model, p, calc_metric, *args)  # without "*" it would have built a tuple in a tuple
    return eval_loss


def eval_batches(model, p, calc_metric, *args):
    if len(args) == 5:  # validation (thus training is also there)
        _, _, _, dataloader1, dataloader2 = args
    else:  # (=2) only testing
        dataloader1, dataloader2 = args
    running_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(tqdm(zip(dataloader1, dataloader2), total=len(dataloader1)), 0):
            # get the inputs
            inputs1, labels1 = data[0]
            inputs2, labels2 = data[1]
            # send them to device
            inputs1 = inputs1.to(p.device)
            labels1 = labels1.to(p.device)
            inputs2 = inputs2.to(p.device)
            labels2 = labels2.to(p.device)
            # forward
            outputs1 = model(inputs1)  # forward pass
            outputs2 = model(inputs2)
            # calculate loss
            loss = cosine_loss(outputs1, outputs2, labels1, labels2, flag=p.flag, lmbda=p.lmbda, b=p.b)
            running_loss += loss.data.item()
            if calc_metric:
                pass
    return running_loss



