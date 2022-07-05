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
    p = ProSet()

    # wandb.login()
    # wandb.init('test', entity=p.entity)
    # config = dict(n_epochs=p.num_epochs, batch_size=p.batch_size)

    tr_dataset_1 = load_datasets(p.train_path)
    tr_dataset_2 = load_datasets(p.train_path, mode=1)

    model = Advrtset(tr_dataset_1.x.shape[1], ker_size=p.ker_size, stride=p.stride, dial=p.dial).to(p.device)
    if p.mult_gpu:
        model = nn.DataParallel(model, device_ids=p.device_ids)

    optimizer = torch.optim.Adam(model.parameters(), lr=p.lr, weight_decay=p.weight_decay)

    trainloader1 = torch.utils.data.DataLoader(
        tr_dataset_1, batch_size=p.batch_size, shuffle=False, num_workers=0)
    trainloader2 = torch.utils.data.DataLoader(
        tr_dataset_2, batch_size=p.batch_size, shuffle=False, num_workers=0)

    train(model, p, optimizer, trainloader1, trainloader2)

    # wandb.finish()
