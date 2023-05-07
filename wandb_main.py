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
import shutil


if __name__ == '__main__':
    # v1 = np.array([10, 25, 50, 75])
    # w = np.arange(100, 1050, 50)
    # nbeats = np.hstack([v1, w])
    # xl_file = '/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/res_tbl_nbeats_.xlsx'
    # for n in nbeats:
    #     curr_folder = '/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/rr_datasets/n_beats_' + str(n) + '/inference/rearrange_True/'
    #     if not(os.path.isdir(curr_folder)):
    #         os.mkdir(curr_folder)
    #     shutil.copy(xl_file, curr_folder)
    #     os.rename(curr_folder + xl_file[-20:], curr_folder + xl_file[-20:-5] + str(n) + '_True.xlsx')
    #     if os.path.isfile(curr_folder + xl_file[-20:-5] + str(n) + '.xlsx'):
    #         os.remove(curr_folder + xl_file[-20:-5] + str(n) + '.xlsx')
    #     curr_folder = '/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/rr_datasets/n_beats_' + str(
    #         n) + '/inference/rearrange_False/'
    #     if not(os.path.isdir(curr_folder)):
    #         os.mkdir(curr_folder)
    #     shutil.copy(xl_file, curr_folder)
    #     os.rename(curr_folder + xl_file[-20:], curr_folder + xl_file[-20:-5] + str(n) + '_False.xlsx')
    #     if os.path.isfile(curr_folder + xl_file[-20:-5] + str(n) + '.xlsx'):
    #         os.remove(curr_folder + xl_file[-20:-5] + str(n) + '.xlsx')

    ## TO RUN WANDB, CHANGE WANDB_ENABLE TO TRUE AND ACTIVATE WANDB.INIT WITH ADEQUATE INDENTATION
    p = ProSet()
    p.wandb_enable = False
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    device_list = [0, 1, 2, 3, 4, 5, 6, 7]
    for device in device_list:
        if memory_free_values[device] > 1400:
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
    # p.test_mode = True

    # [25, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850,900
    # ,950, 1000]
    # p.n_beats = 1000  # [, , , , , , , , , , , , , , , , , , ,,, ]
    # kernel size issues with 10
    w = np.arange(100, 650, 50)
    v1 = np.array([10, 25, 50, 75])
    nbeats4run = np.hstack([v1, w])
    nbeats4run = nbeats4run[(nbeats4run != 10)]  # 10 was not trained because of the kernel size
    # nbeats4run = nbeats4run[(nbeats4run != 25)]  # 25 has issues at least with basal state
    # nbeats4run = nbeats4run[(nbeats4run != 900)]  # has probably not enough samples starting at the age of 24
    # nbeats4run = nbeats4run[(nbeats4run != 950)]  # probably the same issue as before also in lower age
    # nbeats4run = nbeats4run[(nbeats4run != 1000)]  # probably the same issue as before also in lower age

    ages = [6]  # np.arange(6, 27, 3) or [6]
    p.run_saved_models = False  # IF YOU SET THIS TO FALSE, YOU SHOULD SET ages=[6] , set bas_on_int to False, the loop should be range(1)
    # set p.up2_21 to whatever it should be and change for single nbeats4run in the inner loop below
    p.bas_on_int = False
    p.up2_21 = True

    for jj in range(p.bootstrap_total_folds + 1):  # p.bootstrap_total_folds +
        p.bootstrap_idx = jj
        # if p.bootstrap_idx == 0:
        #     p.stratify = True
        # else:
        #     p.stratify = False
        for rearrange in [True, False]:
            p.rearrange = rearrange
            for curr_nbeats in nbeats4run:  # [nbeats4run[13]]  or nbeats4run
                p.n_beats = curr_nbeats
                lines = []
                for age in ages:
                    p.age = age
                    print(p.n_beats)
                    np.random.seed(p.seed)
                    torch.manual_seed(p.seed)
                    torch.cuda.manual_seed_all(p.seed)
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False

                    if not(p.test_mode):
                        print('Med mode is : {}'.format(p.med_mode))
                        if p.human_flag:
                            p.train_path = '/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/rr_human_train_data.pkl'
                            p.test_path = '/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/rr_human_test_data.pkl'
                        else:
                            if p.equal_non_equal == 'equal':
                                curr_folder = '/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/rr_datasets_equal/n_beats_' + str(
                                    p.n_beats) + '/'
                            else:
                                curr_folder = '/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/rr_datasets/n_beats_' + str(
                                    p.n_beats) + '/'
                            p.working_folder = curr_folder
                            p.train_path = p.working_folder + 'rr_data_age_' + str(p.age) + '_nbeats_' + str(p.n_beats) + '.pkl'
                            # p.train_path = '/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/rr_data.pkl'
                            p.test_path = p.train_path
                        tr_dataset_1 = load_datasets(p.train_path, p, human_flag=p.human_flag, samp_per_id=p.samp_per_id)
                        x_tr, y_tr, x_val, y_val = split_dataset(tr_dataset_1, p)
                        tr_dataset_1, val_dataset_1, scaler1 = scale_dataset(p, x_tr, y_tr, x_val, y_val)
                        if p.rearrange:
                            tr_dataset_1, val_dataset_1 = rearrange_dataset(p, tr_dataset_1, val_dataset_1)
                        tr_dataset_2 = load_datasets(p.train_path, p, mode=1, human_flag=p.human_flag, samp_per_id=p.samp_per_id)
                        x_tr, y_tr, x_val, y_val = split_dataset(tr_dataset_2, p)
                        tr_dataset_2, val_dataset_2, scaler2 = scale_dataset(p, x_tr, y_tr, x_val, y_val, mode=1)
                        if p.rearrange:
                            tr_dataset_2, val_dataset_2 = rearrange_dataset(p, tr_dataset_2, val_dataset_2, mode=1)

                        # torch.manual_seed(p.seed)


                        # model = TCN(1,1, layers=[128, 128, 128], ks=30)
                        # model = model.to(p.device)
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
                        if p.wandb_enable:
                            wandb.watch(model, cosine_loss, log="all", log_freq=30)
                        train_model(model, p, optimizer, trainloader1, trainloader2, valloader1, valloader2)
                        if p.wandb_enable:
                            wandb.finish()





                    ####### TAKING FULL TRAINING SET AND RUN with TEST
                    else:  # in p.test_mode == True, we can have either TRAINING WITHOUT VALIDATION and then test or use saved models
                        # and evaluate performance.
                        if p.run_saved_models:
                            print('Notice: loaded models are used')

                        exp_name = ['abk_min_val_loss', 'control_min_val_loss', 'both_min_val_loss', 'abk_max_val_acc', 'control_max_val_acc',
                                    'control_max_val_acc'] # for now we take only the max val_acc values and not the highest mean
                        # chosen_epoch = [80, 200, 319, 274, 410, 10]


                        # saved_models_path = ['/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/logs/Nov-16-2022_17_57_41/',
                        #                      '/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/logs/Nov-16-2022_18_02_17/',
                        #                      '/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/logs/Nov-16-2022_18_07_51/']
                        saved_models_path = ['/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/logs/Nov-24-2022_23_05_46/',
                                            '/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/logs/Nov-24-2022_23_08_24/',
                                            '/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/logs/Nov-24-2022_23_11_42/']
                        model_type = ['final_model.pt']  # ker_size are 30 30 10  num_epochs = 80 200 319
                        hyperparameter_dict = {'abk_min_val_loss': dict(num_epochs=80, b=-0.5928304672016774, batch_size=16, drop_out=0.33817387726694303,
                                                 ker_size=30, lmbda=0.15788530065565576, lr=2.779077656310891e-07,
                                                 med_mode='a', momentum=0.6971956087960214, weight_decay=9.85273609898957),
                                                'control_min_val_loss': dict(num_epochs=200, b=0.4759263501595812, batch_size=8, drop_out=0.12609304729484835,
                                                     ker_size=30, lmbda=9.613353037151509, lr=3.435639818473589e-07,
                                                     med_mode='c', momentum=0.755330893385039, weight_decay=1.1752777398646077),

                                               'both_min_val_loss': dict(num_epochs=319, b=0.4770143418377288, batch_size=128, drop_out=0.30722983323578223,
                                                                         ker_size=30, lmbda=1.3150498005398192, lr=4.078531801074058e-07,
                                                                         med_mode='both', momentum=0.10744132018019596, weight_decay=0.17254862160126153)}

                                               # 'abk_max_val_acc': dict(num_epochs = 274, b = -0.9562638379209744, batch_size = 128, drop_out = 0.17056978300508333,
                                               #                         ker_size = 10, lmbda = 3.116931692726408, lr = 4.052054605371503e-09,
                                               #                         med_mode = 'a', momentum = 0.1242686110085278, weight_decay = 9.940616806701842),
                                               # 'control_max_val_acc': dict(num_epochs = 410, b = -0.35933104620886336, batch_size = 8, drop_out = 0.24671384595337603,
                                               #                             ker_size = 30, lmbda = 0.0002452993185841912, lr = 2.2402385543478617e-07,
                                               #                             med_mode = 'c', momentum = 0.3939188582826322, weight_decay = 7.947725851063474),
                                               # 'both_max_val_acc': dict(num_epochs = 10, b = 0.49879753665203497, batch_size = 64, drop_out = 0.372252749360447,
                                               #                          ker_size = 10, lmbda = 0.17068996497656697, lr = 3.827153538598344e-07,
                                               #                          med_mode = 'both', momentum = 0.3889915168297784, weight_decay = 0.1554402334425353)}

                        #todo: make dir with nbeats as a name so we can change ker_size approprietly

                        for idx_exp, exp in enumerate(hyperparameter_dict.keys()):
                            for var_name in p.__dict__:
                                if var_name in hyperparameter_dict[exp].keys():
                                    p.__dict__[var_name] = hyperparameter_dict[exp][var_name]
                            # p.ker_size = int(np.minimum(np.ceil(0.1*p.n_beats), 30))
                            p.ker_size = int(np.ceil(0.1 * p.n_beats))
                            if p.human_flag:
                                p.train_path = '/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/rr_human_train_data.pkl'
                                p.test_path = '/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/rr_human_test_data.pkl'
                            else:
                                if p.up2_21:
                                    curr_folder = '/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/rr_datasets_equal_up2_21/n_beats_' + str(
                                            p.n_beats) + '/'
                                else:
                                    if p.equal_non_equal == 'equal':
                                        curr_folder = '/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/rr_datasets_equal/n_beats_' + str(
                                            p.n_beats) + '/'
                                    else:
                                        curr_folder = '/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/rr_datasets/n_beats_' + str(
                                            p.n_beats) + '/'
                                p.working_folder = curr_folder
                                p.train_path = p.working_folder + 'rr_data_age_' + str(p.age) + '_nbeats_' + str(p.n_beats) + '.pkl'
                                # p.train_path = '/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/rr_data.pkl'
                                p.test_path = p.train_path
                            tr_dataset_1 = load_datasets(p.train_path, p, human_flag=p.human_flag, samp_per_id=p.samp_per_id)
                            ts_dataset_1 = load_datasets(p.test_path, p, human_flag=p.human_flag, samp_per_id=p.samp_per_id, train_mode=False)
                            tr_dataset_1, ts_dataset_1 = scale_dataset(p, tr_dataset_1, ts_dataset_1, should_scale=False)
                            if p.rearrange:
                                if p.bas_on_int and p.med_mode == 'c':
                                    rearrange_bas_on_int(p, tr_dataset_1, ts_dataset_1)
                                else:
                                    tr_dataset_1, ts_dataset_1 = rearrange_dataset(p, tr_dataset_1, ts_dataset_1)
                            tr_dataset_2 = load_datasets(p.train_path, p, human_flag=p.human_flag, samp_per_id=p.samp_per_id, mode=1)
                            ts_dataset_2 = load_datasets(p.test_path, p, human_flag=p.human_flag, samp_per_id=p.samp_per_id, train_mode=False, mode=1)
                            tr_dataset_2, ts_dataset_2 = scale_dataset(p, tr_dataset_2, ts_dataset_2, mode=1, should_scale=False)
                            if p.rearrange:
                                if p.bas_on_int and p.med_mode == 'c':
                                    rearrange_bas_on_int(p, tr_dataset_2, ts_dataset_2, mode=1)
                                else:
                                    tr_dataset_2, ts_dataset_2 = rearrange_dataset(p, tr_dataset_2, ts_dataset_2, mode=1)
                            curr_batch_size = p.batch_size
                            while (np.floor(ts_dataset_1.x.shape[0] / curr_batch_size) < 1) and (curr_batch_size != 1):
                                curr_batch_size /= 2
                            curr_batch_size = int(np.ceil(curr_batch_size))
                            p.batch_size = curr_batch_size
                            acc_vector = torch.zeros(1)
                            for i in range(1):
                                # if i == 0:
                                #     p.mkdir = True
                                # else:
                                #     p.mkdir = False

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
                                if p.run_saved_models:
                                    if p.rearrange:
                                        pth = p.working_folder + 'trained_on_6m_rearrange_True'
                                    else:
                                        pth = p.working_folder + 'trained_on_6m_rearrange_False'
                                    if p.med_mode == 'a':
                                        pth = pth + '/' + 'intrinsic/'
                                    elif p.med_mode == 'c':
                                        pth = pth + '/' + 'basal/'
                                    elif p.med_mode == 'both':
                                        pth = pth + '/' + 'combined/'
                                    # p.log_path = saved_models_path[idx_exp]
                                    # model = torch.load(p.log_path + model_type[0])
                                    if os.path.isfile(pth + model_type[0]):
                                        model = torch.load(pth + model_type[0]) #, map_location=lambda storage, loc: storage.cuda(1))
                                        # p.device = 1
                                        # both two code lines above force the model and the data to be on GPU cuda:1
                                        # the codel line belo makes sure that the running device is the same as it was when the model was saved
                                        p.device = model.p.device
                                        _, _, acc_vector[i] = eval_model(model, p, 1, tsloader1, tsloader2)
                                else:
                                    train_model(model, p, optimizer, trainloader1, trainloader2, tsloader1, tsloader2)
                                    _, _, acc_vector[i] = eval_model(model, p, 1, tsloader1, tsloader2)
                                    if p.wandb_enable:
                                        wandb.finish()
                            lines += ['In exp. {}, mean test results were  {:.2f}% with {:.2f}% std'.format(exp, 100*acc_vector.nanmean(),
                                                                                                            100*acc_vector.std())]
                            lines += ['{} training mice with {} heartbeat windows, {} testing mice with {}'
                                      ' heartbeat windows at age of {}'.format(p.n_individuals_train, p.n_train, p.n_individuals_test, p.n_test, age)]
                with open(p.log_path + '/README_' + str(p.bootstrap_idx) + '.txt', 'w') as f:
                    f.write('\n'.join(lines))
