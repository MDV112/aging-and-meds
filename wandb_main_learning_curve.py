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
    p.learning_curve = True
    w = np.arange(100, 650, 50)
    v1 = np.array([10, 25, 50, 75])
    nbeats4run = np.hstack([v1, w])
    nbeats4run = nbeats4run[(nbeats4run != 10)]  # 10 was not trained because of the kernel size

    ages = [6]  # np.arange(6, 27, 3) or [6]
    p.bas_on_int = False
    p.up2_21 = True

    chosen_nbeat = 50
    p.learning_curve_folds = 50

    p.run_saved_models = False
    # Once set to FALSE, comment the ########  ######### lines. After run is done, set to TRUE
    # and run again for inference. This should end with an error. When done you may uncomment the ###########  ######## lines.


########################################################################################
    # with open('/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/rr_datasets_equal_up2_21/n_beats_' + str(chosen_nbeat) + '/res_temp_train.npy',
    #           'rb') as f:
    #     res_temp_train = np.load(f)
    # with open('/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/rr_datasets_equal_up2_21/n_beats_' + str(chosen_nbeat) + '/res_temp_test.npy',
    #           'rb') as g:
    #     res_temp_test = np.load(g)
    # cd_pd = [0, 1, 0, 1, 0, 1]  # 0, 1: CD, PD
    # exp = [0, 0, 1, 1, 2, 2]  # 0, 1, 2: abk, bas, comb
    # cd_pd_dict = {0: 'CD', 1: 'PD'}
    # exp_dict = {0: 'ABK', 1: 'BAS', 2: 'COMB'}
    # for val in zip(cd_pd, exp):
    #     fig, ax = plt.subplots(figsize=(20, 20))
    #     x = np.arange(0, 100, 1)
    #     ax.plot(x[0:-1:2], res_temp_train[val[0], val[1], 0, :])
    #     ax.plot(x[0:-1:2], res_temp_test[val[0], val[1], 0, :])
    #     ax.set_xticks(x[0:-1:4])
    #     plt.xticks(fontsize=26, weight='bold')
    #     plt.yticks(fontsize=26, weight='bold')
    #     ax.set_xticklabels(x[0:-1:4], fontsize=26, weight='bold')
    #     # ax.set_yticklabels(np.around(res_temp_test[val[0], val[1], 0, :], decimals=2), fontsize=26, weight='bold')
    #     ax.set_xlabel(xlabel="Fraction of training set size [%]", fontsize=26, weight='bold')
    #     ax.set_ylabel(ylabel="Loss", fontsize=26, weight='bold')
    #     for _, s in ax.spines.items():
    #         s.set_linewidth(5)
    #     ax.legend(['Train', 'Test'], fontsize=30)
    #     plt.title(cd_pd_dict[val[0]] + ' , ' + exp_dict[val[1]], fontsize=30, weight='bold')
    #     sns.despine(fig=None, ax=ax, top=True, right=True, left=False, bottom=False, offset=None, trim=False)
    #     plt.show()
    #     b=1
    # a=1


##########################################################################################################


    idx = np.argwhere(nbeats4run == chosen_nbeat).item()  # just to make sure I've picked nbeat that exists in the array
    chosen_nbeat = nbeats4run[idx]

    if p.run_saved_models:
        res_temp_train = np.zeros((2, 3, 7, p.learning_curve_folds))
        res_temp_test = np.zeros_like(res_temp_train)


    for jj in range(p.learning_curve_folds):
        p.curr_fold = jj
        h_rearrange = 0
        for rearrange in [True, False]:
            p.rearrange = rearrange
            for curr_nbeats in [chosen_nbeat]:
                p.n_beats = curr_nbeats
                lines = []
                h_age = 0
                for age in ages:
                    if (p.run_saved_models and age < 7) or not(p.run_saved_models):
                        p.age = age
                        print(p.n_beats)
                        np.random.seed(p.seed)
                        torch.manual_seed(p.seed)
                        torch.cuda.manual_seed_all(p.seed)
                        torch.backends.cudnn.deterministic = True
                        torch.backends.cudnn.benchmark = False

                        if not(p.test_mode):
                            pass
                        ####### TAKING FULL TRAINING SET AND RUN with TEST
                        else:  # in p.test_mode == True, we can have either TRAINING WITHOUT VALIDATION and then test or use saved models
                            # and evaluate performance.
                            if p.run_saved_models:
                                print('Notice: loaded models are used')

                            exp_name = ['abk_min_val_loss', 'control_min_val_loss', 'both_min_val_loss', 'abk_max_val_acc', 'control_max_val_acc',
                                        'control_max_val_acc'] # for now we take only the max val_acc values and not the highest mean
                            model_type = ['final_model_']  # ker_size are 30 30 10  num_epochs = 80 200 319
                            hyperparameter_dict = {'abk_min_val_loss': dict(num_epochs=80, b=-0.5928304672016774, batch_size=16, drop_out=0.33817387726694303,
                                                     ker_size=30, lmbda=0.15788530065565576, lr=2.779077656310891e-07,
                                                     med_mode='a', momentum=0.6971956087960214, weight_decay=9.85273609898957),
                                                    'control_min_val_loss': dict(num_epochs=200, b=0.4759263501595812, batch_size=8, drop_out=0.12609304729484835,
                                                         ker_size=30, lmbda=9.613353037151509, lr=3.435639818473589e-07,
                                                         med_mode='c', momentum=0.755330893385039, weight_decay=1.1752777398646077),

                                                   'both_min_val_loss': dict(num_epochs=319, b=0.4770143418377288, batch_size=128, drop_out=0.30722983323578223,
                                                                             ker_size=30, lmbda=1.3150498005398192, lr=4.078531801074058e-07,
                                                                             med_mode='both', momentum=0.10744132018019596, weight_decay=0.17254862160126153)}
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
                                        if os.path.isfile(pth + model_type[0] + str(jj) + '.pt'):
                                            model = torch.load(pth + model_type[0] + str(jj) + '.pt') #, map_location=lambda storage, loc: storage.cuda(1))
                                            # p.device = 1
                                            # both two code lines above force the model and the data to be on GPU cuda:1
                                            # the codel line belo makes sure that the running device is the same as it was when the model was saved
                                            p.device = model.p.device
                                            running_loss_train, eer_train, _ = eval_model(model, p, 1, trainloader1, trainloader2)
                                            running_loss_test, eer_test, _ = eval_model(model, p, 1, tsloader1, tsloader2)
                                            res_temp_train[h_rearrange, idx_exp, h_age, p.curr_fold] = running_loss_train
                                            res_temp_test[h_rearrange, idx_exp, h_age, p.curr_fold] = running_loss_test

                                    else:
                                        train_model(model, p, optimizer, trainloader1, trainloader2, tsloader1, tsloader2)
                                        _, _, acc_vector[i] = eval_model(model, p, 1, tsloader1, tsloader2)
                                        if p.wandb_enable:
                                            wandb.finish()
                                lines += ['In exp. {}, mean test results were  {:.2f}% with {:.2f}% std'.format(exp, 100*acc_vector.nanmean(),
                                                                                                                100*acc_vector.std())]
                                lines += ['{} training mice with {} heartbeat windows, {} testing mice with {}'
                                          ' heartbeat windows at age of {}'.format(p.n_individuals_train, p.n_train, p.n_individuals_test, p.n_test, age)]
                        h_age += 1
                with open(p.log_path + '/README_' + str(p.bootstrap_idx) + '.txt', 'w') as f:
                    f.write('\n'.join(lines))
            h_rearrange += 1
    with open('/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/rr_datasets_equal_up2_21/n_beats_' + str(chosen_nbeat) + '/res_temp_train.npy', 'wb') as f:
        np.save(f, res_temp_train)
    with open('/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/rr_datasets_equal_up2_21/n_beats_' + str(chosen_nbeat) + '/res_temp_test.npy', 'wb') as g:
        np.save(g, res_temp_test)
    a=1