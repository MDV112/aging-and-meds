import copy
import os
import pickle
import time
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler as MinMax
from sklearn.utils import shuffle
from tqdm import tqdm
from scipy import stats
from project_settings import HRVDataset
from deep_models import cosine_loss
import wandb
from torch.optim.lr_scheduler import LambdaLR
import seaborn as sns
from sklearn.model_selection import StratifiedKFold


def choose_win(rr, lbls, samp_per_id=0):
    """
    This function chooses random time window for different patients so the RR within it would hopefully be stationary.
    Since the data comes from holter, the HRV might vary a lot during the day.
    :param rr: rr_matrix at the size of (beats X number of examples)
    :param lbls: labels matrix at the size of (number of examples X 2).
    :param samp_per_id: number of consecutive examples to extract per patient.
    :return: tuple of rr matrix at the shape of (n X number of examples) and matrix of labels at the shape of
    (number of examples X 2).
    """
    if samp_per_id != 0:
        tags = np.unique(lbls[:, 0])
        rr_new = np.zeros((1, rr.shape[1]))
        lbls_new = np.zeros((1, lbls.shape[1]), dtype=int)
        for tag in tags:
            idx = np.argwhere(lbls[:, 0] == tag)
            if len(idx) > samp_per_id - 1:
                r_start = np.random.randint(0, len(idx) - samp_per_id - 1)
                rr_new = np.vstack(
                    [rr_new, rr[idx[0, :].item() + r_start: idx[0, :].item() + r_start + samp_per_id, :]])
                lbls_new = np.vstack(
                    [lbls_new, lbls[idx[0, :].item() + r_start: idx[0, :].item() + r_start + samp_per_id, :]])
            else:
                rr_new = np.vstack([rr_new, rr])
                lbls_new = np.vstack([lbls_new, lbls])
        return rr_new[1:, :], lbls_new[1:, :]
    else:
        return rr, lbls


def load_datasets_auxiliary(x, y, p, mode, train_mode, x_a_orig, y_a_orig):

    if train_mode:
        p.n_train = x.shape[1]
        dataset = HRVDataset(x.T, y, p, mode=mode)  # transpose should fit HRVDataset
    else:
        if p.bas_on_int:
            rel_idx1 = int(np.floor(p.bootstrap[p.bootstrap_idx][0]*x_a_orig.shape[1]))
            rel_idx2 = int(np.ceil(p.bootstrap[p.bootstrap_idx][1]*x_a_orig.shape[1]))
            x_a = x_a_orig[:, rel_idx1:rel_idx2]
            y_a = y_a_orig[rel_idx1:rel_idx2, :]
            p.n_test = x_a.shape[1]
            dataset = HRVDataset(x_a.T, y_a, p, mode=mode)  # transpose should fit HRVDataset
        else:
            rel_idx1 = int(np.floor(p.bootstrap[p.bootstrap_idx][0] * x.shape[1]))
            rel_idx2 = int(np.ceil(p.bootstrap[p.bootstrap_idx][1] * x.shape[1]))
            x_new = x[:, rel_idx1:rel_idx2]
            y_new = y[rel_idx1:rel_idx2, :]
            # x_new = x[:, p.bootstrap[p.bootstrap_idx][0]: p.bootstrap[p.bootstrap_idx][1]]
            # y_new = y[p.bootstrap[p.bootstrap_idx][0]:p.bootstrap[p.bootstrap_idx][1], :]
            p.n_test = x_new.shape[1]
            dataset = HRVDataset(x_new.T, y_new, p, mode=mode)  # transpose should fit HRVDataset
    return dataset

def load_datasets(full_pickle_path: str, p: object, mode: int = 0, train_mode: bool = True, human_flag=0, samp_per_id=0) \
        -> object:
    """
    This function is used for loading pickls of training and proper testing prepared ahead and rearrange them as
     HRVDataset objects. It is recommended to have pickle with all jrv features since we can drop whatever we want here.
    :param full_pickle_path: fullpath of training pickle or testing pickle (including '.pkl' ending).
    :param med_mode: type of treatment: 'c' for control and 'a' for abk.
    :param mode: see HRVDataset.
    :param feat2drop: HRV features to drop.
    :param sig_type: rr or HRV signal (The HRV is in koopman mode for now).
    :param train_mode: extract train or test.
    :return: HRVDataset object
    """
    if p.sig_type == 'rr':
        if human_flag:
            with open(full_pickle_path, 'rb') as f:
                e = pickle.load(f)
            chosen_x, chosen_y = choose_win(e[0], e[1], samp_per_id=samp_per_id)
            x, y = shuffle(chosen_x, chosen_y, random_state=0)  # for domain mixing
            if p.remove_mean:
                x = x.T - x.mean(axis=1)
                x = x.T
            p.med_mode = 'c'  # for now human are only NSR
            if train_mode:
                p.n_train = x.shape[0]
            else:
                p.n_test = x.shape[0]
            dataset = HRVDataset(x, y, p, mode=mode)
            return dataset
        with open(full_pickle_path, 'rb') as f:
            e = pickle.load(f)
            if train_mode:
                x = e.x_train_specific
                if p.remove_mean:
                    p.mu = x.mean()
                    x = x - p.mu
                    # x = x - x.mean(axis=0)
                    # x = x - x.min()
                y = e.y_train_specific
                p.train_ages = np.unique(y['age'])
                p.n_individuals_train = len(np.unique(y['id']))
                print('Ages used in training set are {}'.format(np.unique(y['age'])))
            else:
                x = e.x_test_specific
                if p.remove_mean:
                    x = x - p.mu
                    # x = x - x.mean(axis=0)
                    # x = x - x.min(axis=0)
                y = e.y_test_specific
                p.test_ages = np.unique(y['age'])
                print('Ages used in training set are {}'.format(np.unique(y['age'])))
                p.n_individuals_test = len(np.unique(y['id']))
        x_c, x_a = x[:, y['med'] == 0], x[:, y['med'] == 1]
        y_c, y_a = y[['id', 'age']][y['med'] == 0].values.astype(int), y[['id', 'age']][y['med'] == 1].values.astype(
            int)


        x_c_T, y_c = shuffle(x_c.T, y_c, random_state=0)  # shuffle is used here for shuffeling ages
        # because in the dataloader it should be false. sklearn should make the shuffle the same. Notice the transpose
        # in x since we want to shuffle the columns fo RR
        x_a_T, y_a = shuffle(x_a.T, y_a, random_state=0)

        if train_mode and p.learning_curve and not(p.rearrange):
            N = x_c_T.shape[0]
            step = int(np.floor(N/p.learning_curve_folds))
            stop = (p.curr_fold + 1) * step - 1
            if stop <= p.batch_size:
                stop = p.batch_size + 1
            x_c_T, x_a_T, y_c, y_a = x_c_T[0:stop, :], x_a_T[0:stop, :], y_c[0:stop, :], y_a[0:stop, :]
            # print(x_c_T.shape[0])


        x_c = x_c_T.T
        x_a = x_a_T.T
        x_ca = np.concatenate((x_c, x_a), axis=1)
        y_ca = np.concatenate((y_c, y_a))
        x_ca_T, y_ca = shuffle(x_ca.T, y_ca, random_state=0)
        x_ca = x_ca_T.T

        if p.med_mode == 'c':
            dataset = load_datasets_auxiliary(x_c, y_c, p, mode, train_mode, x_a, y_a)
        elif p.med_mode == 'a':
            dataset = load_datasets_auxiliary(x_a, y_a, p, mode, train_mode, x_a, y_a)
        elif p.med_mode == 'both':
            dataset = load_datasets_auxiliary(x_ca, y_ca, p, mode, train_mode, x_a, y_a)


        # if p.med_mode == 'c':
        #     if train_mode:
        #         p.n_train = x_c.shape[1]
        #         dataset = HRVDataset(x_c.T, y_c, p, mode=mode)  # transpose should fit HRVDataset
        #     else:
        #         if p.bas_on_int:
        #             # x_a = x_a[:, p.bootstrap[p.bootstrap_idx][0]:p.bootstrap[p.bootstrap_idx][1]]
        #             p.n_test = x_a.shape[1]
        #             dataset = HRVDataset(x_a.T, y_a, p, mode=mode)  # transpose should fit HRVDataset
        #         else:
        #             # x_c = x_c[:, p.bootstrap[p.bootstrap_idx][0]: p.bootstrap[p.bootstrap_idx][1]]
        #             p.n_test = x_c.shape[1]
        #             dataset = HRVDataset(x_c.T, y_c, p, mode=mode)  # transpose should fit HRVDataset
        #
        # elif p.med_mode == 'a':
        #     if train_mode:
        #         p.n_train = x_a.shape[1]
        #         dataset = HRVDataset(x_a.T, y_a, p, mode=mode)  # transpose should fit HRVDataset
        #     else:
        #         if p.bas_on_int:
        #             # x_a = x_a[:, p.bootstrap[p.bootstrap_idx][0]:p.bootstrap[p.bootstrap_idx][1]]
        #             p.n_test = x_a.shape[1]
        #             dataset = HRVDataset(x_a.T, y_a, p, mode=mode)  # transpose should fit HRVDataset
        #         else:
        #             # x_a = x_a[:, p.bootstrap[p.bootstrap_idx][0]:p.bootstrap[p.bootstrap_idx][1]]
        #             p.n_test = x_a.shape[1]
        #             dataset = HRVDataset(x_a.T, y_a, p, mode=mode)  # transpose should fit HRVDataset
        # elif p.med_mode == 'both':
        #     if train_mode:
        #         p.n_train = x_ca.shape[1]
        #         dataset = HRVDataset(x_ca.T, y_ca, p, mode=mode)  # transpose should fit HRVDataset
        #     else:
        #         if p.bas_on_int:
        #             p.n_test = x_a.shape[1]
        #             dataset = HRVDataset(x_a.T, y_a, p, mode=mode)  # transpose should fit HRVDataset
        #         else:
        #             p.n_test = x_ca.shape[1]
        #             dataset = HRVDataset(x_ca.T, y_ca, p, mode=mode)  # transpose should fit HRVDataset
        else:  # other medications
            raise NotImplementedError
    else:  # Koopman HRV
        # todo: add age tags
        with open(full_pickle_path, 'rb') as f:
            e = pickle.load(f)
            data = e[0:4]
        x_c, x_a = (data[0], data[1])  # (data[2], data[3]) are the y of Koopman, meaning the samples in the future
        y_c, y_a = (x_c.index, x_a.index)
        if p.med_mode == 'c':
            label_dataset = y_c
            if len(p.feat2drop) != 0:
                x_c.drop(p.feat2drop, axis=1, inplace=True)
            np_dataset = np.array(x_c.values, dtype=np.float)
            dataset = HRVDataset(np_dataset, label_dataset, p, mode=mode)
        elif p.med_mode == 'a':
            label_dataset = y_a
            if len(p.feat2drop) != 0:
                x_a.drop(p.feat2drop, axis=1, inplace=True)
            np_dataset = np.array(x_a.values, dtype=np.float)
            dataset = HRVDataset(np_dataset, label_dataset, p, mode=mode)
        else:  # other medications
            raise NotImplementedError
    return dataset


def split_dataset(dataset: object, p: object, seed: int = 42) -> tuple:
    """
    This function splits the training dataset into training and validation.
    :param dataset: HRVDataset of training.
    :param val_size: validation size which is a fraction in the range of (0,1).
    :param seed: seed for random choice of mice.
    :param proper: if True the split is done like in real testing, i.e. the training and validation sets contain data
     from different mice. If False, then only the examples are different (came form different time windows).
    :return: 4 numpy arrays.
    """
    if p.proper:  # meaning splitting train and val into different mice or just different time windows
        np.random.seed(seed)
        tags = np.unique(dataset.y[:, 0])
        val_tags = np.random.choice(tags, int(np.floor(p.val_size * len(tags))), replace=False)
        val_tags = [730, 727, 765, 708, 743, 731, 555, 717, 751]
        train_tags = np.setdiff1d(tags, val_tags)
        p.n_individuals_train = len(train_tags)
        p.n_individuals_val = len(val_tags)
        train_mask = np.isin(dataset.y[:, 0], train_tags)
        val_mask = np.isin(dataset.y[:, 0], val_tags)
        x_train = dataset.x[train_mask, :]
        y_train = dataset.y[train_mask, :]
        x_val = dataset.x[val_mask, :]
        y_val = dataset.y[val_mask, :]
    else:  # todo: check if split is done correctly here
        if p.stratify:
            x_train, x_val, y_train, y_val = train_test_split(dataset.x, dataset.y, test_size=p.val_size,
                                                              random_state=42, stratify=dataset.y)  # random_state is to make sure both dataloaders will have the same split and thus we can preserve 50%-50% tagging using HRVDataset __getitem__
        else:
            x_train, x_val, y_train, y_val = train_test_split(dataset.x, dataset.y, test_size=p.val_size,
                                                              random_state=42)
    p.n_train = x_train.shape[0]
    p.n_val = x_val.shape[0]
    return x_train, y_train, x_val, y_val


def scale_dataset(p, *args, input_scaler=None, mode=0, should_scale: bool = False) -> tuple:
    """
    Scaling the data properly according to the training set. If split is made, then scaling is performed for training
    set and then validation and testing are scaled by the same fitted scaler. This means that we call the function twice;
    once for scaling training and validation (len(args)==4) and once for testing (len(args)==1). Second option is to use
    the full dataset for final tuning after choosing our best hyperparmeters and then we call this function once with
    HRVDatasets of training and testing.
    :param args: Can either [output of split_datasets (4 numpy arrays)] or [HRVdataset(train) and HRVdataset(test)] or
    HRVdataset(test).
    :param input_scaler: fitted sklearn MinMax scaler.
    :param mode: see HRVDataset.
    :param should_scale: More for rr. Notice if transpose is needed or not.
    :return: Three options: 1) two scaled HRVDatasets (training & validation) and a fitted scaler.
                            2) two scaled HRVDatasets (training & testing).
                            3) One scaled HRVdataset (testing).
    """
    if input_scaler is None:
        scaler = MinMax()
        if len(args) == 4:  # x_train, y_train, x_val, y_val
            if should_scale:
                x_train = scaler.fit_transform(args[0])
                x_val = scaler.transform(args[2])
            else:  # todo: remove mean rr from every example
                x_train = args[0]
                x_val = args[2]
            return HRVDataset(x_train, args[1], p, mode=mode), HRVDataset(x_val, args[3], p, mode=mode), scaler
        elif len(args) == 2:  # HRVdataset(train) and HRVdataset(test)
            if should_scale:
                args[0].x = scaler.fit_transform(
                    args[0].x)  # Notice that this won't do anything if training is already scale
                args[1].x = scaler.transform(args[1].x)
            return args
    else:
        if len(args) != 1:
            raise Exception('Only test set can be an input')
        else:
            if should_scale:
                args[0].x = input_scaler.transform(args[0].x)
            return args


def rearrange_dataset(p, *args, mode=0) -> tuple:
    if len(args) == 2:
        train_data = args[0]
        test_data = args[1]
    mixed_x_data = np.vstack((train_data.x, test_data.x))
    mixed_y_data = np.vstack((train_data.y, test_data.y))
    if p.med_mode == 'both':
        a=1
    if p.stratify:
        x_train, x_val, y_train, y_val = train_test_split(mixed_x_data, mixed_y_data, test_size=p.val_size,
                                                          random_state=p.seed, stratify=mixed_y_data[:, 0])  # random_state is to make sure both dataloaders will have the same split and thus we can preserve 50%-50% tagging using HRVDataset __getitem__
    else:
        x_train, x_val, y_train, y_val = train_test_split(mixed_x_data, mixed_y_data, test_size=p.val_size,
                                                          random_state=p.seed)
    if p.learning_curve:
        N = x_train.shape[0]
        step = int(np.floor(N / p.learning_curve_folds))
        stop = (p.curr_fold + 1) * step - 1
        if stop <= p.batch_size:
            stop = p.batch_size + 1
        x_train, y_train = x_train[0:stop, :], y_train[0:stop, :]
    p.n_train = x_train.shape[0]
    p.n_test = x_val.shape[0]
    p.r = np.random.randint(0, 2, p.n_train)
    train_dataset = HRVDataset(x_train, y_train, p, mode=mode)
    p.r = np.random.randint(0, 2, p.n_test)
    test_dataset = HRVDataset(x_val, y_val, p, mode=mode)
    return train_dataset, test_dataset


def rearrange_bas_on_int(p, *args, mode=0) -> tuple:
    train_bas = args[0]
    test_bas = args[1]
    mixed_x_bas_data = np.vstack((train_bas.x, test_bas.x))
    mixed_y_bas_data = np.vstack((train_bas.y, test_bas.y))
    if p.stratify:
        x_train, _, y_train, _ = train_test_split(mixed_x_bas_data, mixed_y_bas_data, test_size=p.val_size,
                                                      random_state=p.seed, stratify=mixed_y_bas_data[:, 0])
    else:
        x_train, _, y_train, _ = train_test_split(mixed_x_bas_data, mixed_y_bas_data, test_size=p.val_size,
                                                      random_state=p.seed)
    p.n_train = x_train.shape[0]
    p.r = np.random.randint(0, 2, p.n_train)
    train_dataset = HRVDataset(x_train, y_train, p, mode=mode)

    p.med_mode = 'a'
    tr_dataset_int = load_datasets(p.train_path, p, human_flag=p.human_flag, samp_per_id=p.samp_per_id, mode=mode)
    ts_dataset_int = load_datasets(p.test_path, p, human_flag=p.human_flag, samp_per_id=p.samp_per_id, train_mode=False,
                                   mode=mode)
    tr_dataset_int, ts_dataset_int = scale_dataset(p, tr_dataset_int, ts_dataset_int, should_scale=False, mode=mode)
    train_int = tr_dataset_int
    test_int = ts_dataset_int
    mixed_x_int_data = np.vstack((train_int.x, test_int.x))
    mixed_y_int_data = np.vstack((train_int.y, test_int.y))
    if p.stratify:
        _, x_val, _, y_val = train_test_split(mixed_x_int_data, mixed_y_int_data, test_size=p.val_size,
                                                          random_state=p.seed, stratify=mixed_y_int_data[:, 0])
    else:
        _, x_val, _, y_val = train_test_split(mixed_x_int_data, mixed_y_int_data, test_size=p.val_size,
                                                          random_state=p.seed)
    p.n_test = x_val.shape[0]
    p.r = np.random.randint(0, 2, p.n_test)
    test_dataset = HRVDataset(x_val, y_val, p, mode=mode)
    p.med_mode = 'c'
    return train_dataset, test_dataset


def train_model(model: object, p: object, *args):
    """
    Training the model.
    :param model: chosen neural network.
    :param p: ProSet (project setting) object.
    :param args: Can be either [optimizer, trainloader1, trainloader1, valloader1, valloader2] or
                                [optimizer, trainloader1, trainloader1]
    :param calc_metric: calculate metrics such as FAR, FPR etc.
    :return: void (model is learned).
    """
    model.train()
    train_epochs(model, p, *args)  # without "*" it would have built a tuple in a tuple


def train_epochs(model: object, p: object, *args):
    """
    This function runs the model in epochs and evaluates the validation sets if exist (see more details in scale_dataset).
    :param: inputs from train_model function.
    :return: prints logs of epochs.
    """
    now = datetime.now()
    d_loss = 0.0
    training_loss_vector = np.zeros(p.num_epochs)
    val_loss_vector = np.zeros(p.num_epochs)
    training_err_vector = np.zeros(p.num_epochs)
    val_err_vector = np.zeros(p.num_epochs)
    train_acc_vector = np.zeros(p.num_epochs)
    val_acc_vector = np.zeros(p.num_epochs)
    best_val_ERR = 1
    best_val_acc = 0
    best_ERR_diff = 1
    best_ACC_diff = 1
    # if p.mkdir:
        # pth = '/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/logs/' + now.strftime("%b-%d-%Y_%H_%M_%S")
    if p.rearrange:
        pth = p.working_folder + 'trained_on_6m_rearrange_True'
        if not (os.path.isdir(pth)):
            os.mkdir(pth)
    else:
        pth = p.working_folder + 'trained_on_6m_rearrange_False'
        if not (os.path.isdir(pth)):
            os.mkdir(pth)
    p.log_path = p.working_folder  # USED TO BE pth (pth = '/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/logs/' + now.strftime("%b-%d-%Y_%H_%M_%S"))
        # os.mkdir(pth)
    counter_train = 0
    counter_val = 0
    counter_lr = 0
    if p.med_mode == 'a':
        if not(os.path.isdir(pth + '/' + 'intrinsic')):
            os.mkdir(pth + '/' + 'intrinsic')
        pth = pth + '/' + 'intrinsic/'
    elif p.med_mode == 'c':
        if not(os.path.isdir(pth + '/' + 'basal')):
            os.mkdir(pth + '/' + 'basal')
        pth = pth + '/' + 'basal/'
    elif p.med_mode == 'both':
        if not(os.path.isdir(pth + '/' + 'combined')):
            os.mkdir(pth + '/' + 'combined')
        pth = pth + '/' + 'combined/'

    for epoch in range(1, p.num_epochs + 1):
        epoch_time = time.time()
        if p.calc_metric:
            training_loss, training_err_vector[epoch - 1], train_acc_vector[epoch - 1] = train_batches(model, p, epoch,
                                                                                                       *args)
            validation_loss, val_err_vector[epoch - 1], val_acc_vector[epoch - 1] = eval_model(model, p, epoch, *args)
            if p.wandb_enable:
                # wandb.log({'epoch':epoch,'training_err':training_err_vector[epoch - 1],
                #            'training_acc':train_acc_vector[epoch - 1], 'val_err':val_err_vector[epoch - 1],
                #            'val_acc':val_acc_vector[epoch - 1]})
                wandb.log({'training_acc':train_acc_vector[epoch - 1],'val_acc':val_acc_vector[epoch - 1],'remaining epochs': p.num_epochs - epoch})
            if training_err_vector[epoch - 1] >= 0.999:
                d = epoch - p.curr_train_epoch
                p.curr_train_epoch = epoch
                if d == 1:
                    counter_train += 1
                else:
                    counter_train = 0
                if counter_train > p.patience:
                    p.error_txt = 'Stopped running since {} training epochs EER in a row were 1.00'.format(
                        counter_train)
                    print(p.error_txt)
                    break
            if val_err_vector[epoch - 1] >= 0.999:
                d = epoch - p.curr_val_epoch
                p.curr_val_epoch = epoch
                if d == 1:
                    counter_val += 1
                else:
                    counter_val = 0
                if counter_val > p.patience:
                    p.error_txt = 'Stopped running since {} validation epochs EER in a row were 1.00'.format(
                        counter_val)
                    print(p.error_txt)
                    break
            ############## SAVING BEST MODELS ########################
            if p.learning_curve:
                if val_err_vector[epoch - 1] < best_val_ERR:
                    best_val_ERR = val_err_vector[epoch - 1]
                    torch.save(copy.deepcopy(model), pth + 'best_ERR_model_' + str(p.curr_fold) + '.pt')
                if val_acc_vector[epoch - 1] > best_val_acc:
                    best_val_acc = val_acc_vector[epoch - 1]
                    torch.save(copy.deepcopy(model), pth + 'best_ACC_model_' + str(p.curr_fold) + '.pt')
                if np.abs(val_err_vector[epoch - 1] - training_err_vector[epoch - 1]) < best_ERR_diff:
                    best_ERR_diff = np.abs(val_err_vector[epoch - 1] - training_err_vector[epoch - 1])
                    torch.save(copy.deepcopy(model), pth + 'best_ERR_diff_model_' + str(p.curr_fold) + '.pt')
                if np.abs(val_acc_vector[epoch - 1] - train_acc_vector[epoch - 1]) < best_ACC_diff:
                    best_ACC_diff = np.abs(val_acc_vector[epoch - 1] - train_acc_vector[epoch - 1])
                    torch.save(copy.deepcopy(model), pth + 'best_ACC_diff_model_' + str(p.curr_fold) + '.pt')
                if epoch == p.num_epochs:
                    torch.save(copy.deepcopy(model), pth + 'final_model_' + str(p.curr_fold) + '.pt')
            else:
                if val_err_vector[epoch - 1] < best_val_ERR:
                    best_val_ERR = val_err_vector[epoch - 1]
                    torch.save(copy.deepcopy(model), pth + 'best_ERR_model.pt')
                if val_acc_vector[epoch - 1] > best_val_acc:
                    best_val_acc = val_acc_vector[epoch - 1]
                    torch.save(copy.deepcopy(model), pth + 'best_ACC_model.pt')
                if np.abs(val_err_vector[epoch - 1] - training_err_vector[epoch - 1]) < best_ERR_diff:
                    best_ERR_diff = np.abs(val_err_vector[epoch - 1] - training_err_vector[epoch - 1])
                    torch.save(copy.deepcopy(model), pth + 'best_ERR_diff_model.pt')
                if np.abs(val_acc_vector[epoch - 1] - train_acc_vector[epoch - 1]) < best_ACC_diff:
                    best_ACC_diff = np.abs(val_acc_vector[epoch - 1] - train_acc_vector[epoch - 1])
                    torch.save(copy.deepcopy(model), pth + 'best_ACC_diff_model.pt')
                if epoch == p.num_epochs:
                    torch.save(copy.deepcopy(model), pth + 'final_model.pt')
            # if val_err_vector[epoch - 1] < best_val_ERR:
            #     best_val_ERR = val_err_vector[epoch - 1]
            #     torch.save(copy.deepcopy(model.state_dict()), pth + '/best_ERR_model.pt')
            # if val_acc_vector[epoch - 1] > best_val_acc:
            #     best_val_acc = val_acc_vector[epoch - 1]
            #     torch.save(copy.deepcopy(model.state_dict()), pth + '/best_ACC_model.pt')
            # if np.abs(val_err_vector[epoch - 1] - training_err_vector[epoch - 1]) < best_ERR_diff:
            #     best_ERR_diff = np.abs(val_err_vector[epoch - 1] - training_err_vector[epoch - 1])
            #     torch.save(copy.deepcopy(model.state_dict()), pth + '/best_ERR_diff_model.pt')
            # if np.abs(val_acc_vector[epoch - 1] - train_acc_vector[epoch - 1]) < best_ACC_diff:
            #     best_ACC_diff = np.abs(val_acc_vector[epoch - 1] - train_acc_vector[epoch - 1])
            #     torch.save(copy.deepcopy(model.state_dict()), pth + '/best_ACC_diff_model.pt')
            #################################################################
        else:
            training_loss = train_batches(model, p, epoch, *args)
            validation_loss = eval_model(model, p, epoch, *args)
        training_loss /= len(args[1])  # len of trainloader
        training_loss_vector[epoch - 1] = training_loss
        if p.wandb_enable:
            wandb.log({'training_loss':training_loss})  # , 'learning_rate': p.lr, 'd_loss':d_loss})
        if epoch > p.lr_ker_size:
            m1 = training_loss_vector[epoch - p.lr_ker_size - 1: epoch - 1].mean()
            m2 = training_loss_vector[epoch - p.lr_ker_size : epoch].mean()
            d_loss = np.abs(m2/m1)
            lr_lmbda = lambda d_loss: p.lr_factor if ((d_loss > 0.99) and (d_loss < 1.01)) else 1
            scheduler = LambdaLR(args[0], lr_lambda=lr_lmbda)
            # scheduler.step()
            # wandb.log({'learning_rate':scheduler.get_last_lr()})

        # if counter_lr > p.lr_counter:
        #     p.error_txt = 'Stopped running since learning was reduced {} times with factor of {:.2f}}.'\
        #         .format(counter_lr, p.lr_factor)
        #     print(p.error_txt)
        #     break
        if len(args) > 3:  # meaning validation exists.
            validation_loss /= len(args[3])  # len of valloader
            val_loss_vector[epoch - 1] = validation_loss
            if p.wandb_enable:
                wandb.log({'val_loss':validation_loss})
        if p.calc_metric:
            log = "Epoch: {} | Training loss: {:.4f}  | Validation loss: {:.4f}  |  Training ERR: {:.4f}  |" \
                  "  Validation ERR: {:.4f}  |  ".format(epoch, training_loss, validation_loss,
                                                         training_err_vector[epoch - 1],
                                                         val_err_vector[epoch - 1])
        else:
            log = "Epoch: {} | Training loss: {:.4f}  | Validation loss: {:.4f}   ".format(epoch, training_loss,
                                                                                           validation_loss)
        epoch_time = time.time() - epoch_time
        if p.wandb_enable:
            wandb.log({'epoch_time':epoch_time})
        if epoch_time > p.epoch_factor*p.first_epoch_time:
            p.error_txt = 'Stopped running since epoch time was {:.2f} which is at least {} time larger then the first' \
                          ' epoch time which was {:2f}.'.format(epoch_time, p.epoch_factor, p.first_epoch_time)
            print(p.error_txt)
            break
        log += "Epoch Time: {:.2f} secs".format(epoch_time)
        print(log)
        if epoch == 1:
            p.first_epoch_time = epoch_time
    if p.calc_metric:
        pass
        ################## SAVE PLOTS AND TEXT ########################################
        # idx_val_min = np.argmin(val_err_vector)
        # idx_val_max = np.argmax(val_acc_vector)
        # idx_min_diff_err = np.argmin(np.abs(val_err_vector - training_err_vector))
        # idx_min_diff_acc = np.argmin(np.abs(val_acc_vector - train_acc_vector))
        # array_test = np.unique(np.array([1 + idx_val_min, 1 + idx_val_max, 1 + idx_min_diff_err, 1 + idx_min_diff_acc]))
        # str_test = [str(x) for x in array_test]
        # str_test_new = []
        # for x in str_test:
        #     if len(x) == 1:
        #         str_test_new.append('0' + x)
        #     else:
        #         str_test_new.append(x)
        # for file in os.listdir(pth):
        #     if (file.endswith(".png") or file.endswith(".pt")) and ('epoch' in file):
        #         if not (np.any([s in file for s in str_test_new])):
        #             os.remove(pth + '/' + file)
        # plt.plot(np.arange(1, p.num_epochs + 1), 100 * training_err_vector, np.arange(1, p.num_epochs + 1),
        #          100 * val_err_vector)
        # plt.legend(['ERR Train', 'ERR Test'])
        # plt.ylabel('ERR [%]')
        # plt.xlabel('epochs')
        # plt.savefig(pth + '/err.png')
        # plt.close()
        # plt.plot(np.arange(1, p.num_epochs + 1), training_loss_vector, np.arange(1, p.num_epochs + 1), val_loss_vector)
        # plt.legend(['Train loss', 'Validation loss'])
        # plt.ylabel('loss [N.U]')
        # plt.xlabel('epochs')
        # plt.savefig(pth + '/loss.png')
        # plt.close()
        # # todo:  threshold
        # lines = ['Minimal validation ERR was {:.2f}% in epoch number {}. Training ERR at the same epoch was: {:.2f}%.'
        #          .format(100 * np.min(val_err_vector), 1 + idx_val_min, 100 * training_err_vector[idx_val_min]),
        #          'Maximal validation accuracy was {:.2f}% in epoch number {}. Training accuracy at the same epoch was: {:.2f}%.'
        #          .format(100 * np.max(val_acc_vector), 1 + idx_val_max, 100 * train_acc_vector[idx_val_max]),
        #          'Minimal absolute value EER difference was {:.2f} in epoch number {}.'.format(
        #              np.abs(val_err_vector[idx_min_diff_err]
        #                     - training_err_vector[idx_min_diff_err]), 1 + idx_min_diff_err),
        #          'Minimal absolute value ACC difference was {:.2f} in epoch number {}.'.format(
        #              np.abs(val_acc_vector[idx_min_diff_acc] - train_acc_vector[idx_min_diff_acc]),
        #              1 + idx_min_diff_acc)
        #          ]
        #
        # write2txt(lines, pth, p)
        #
        # plt.plot(np.arange(1, p.num_epochs + 1), 100 * train_acc_vector, np.arange(1, p.num_epochs + 1),
        #          100 * val_acc_vector)
        # plt.legend(['ACC Train', 'ACC Test'])
        # plt.ylabel('ACC [%]')
        # plt.xlabel('epochs')
        # plt.savefig(pth + '/acc.png')
        # plt.close()
        # # todo: delete all irrelevant images. Save best conf_mats
        # # plt.show()
        ######################################################################################
    fig1, ax = plt.subplots()
    sns.violinplot(y=val_acc_vector, ax=ax)
    if p.wandb_enable:
        wandb.log({'val_acc_vln': wandb.Image(fig1)})
    plt.close()
    fig2, ax = plt.subplots()
    sns.violinplot(y=train_acc_vector, ax=ax)
    if p.wandb_enable:
        wandb.log({'train_acc_vln': wandb.Image(fig2)})
    plt.close()
    q = np.nanquantile(val_acc_vector, [0, 0.25, 0.5, 0.75, 1])
    val_acc_desc = ['{:.3f}'.format(x) for x in q]
    val_acc_desc.append('{:.3f}'.format(np.mean(val_acc_vector)))
    val_acc_desc.append('{}'.format(1 + np.argmax(val_acc_vector)))
    if p.wandb_enable:
        wandb.log({'val_acc_stats': wandb.Table(columns=['min', 'Q1', 'med', 'Q3', 'max', 'mean', 'max_epoch'],
                                            data=[val_acc_desc])})  # additional square brackets are important
    return


def train_batches(model, p, epoch, *args) -> float:
    """
    This function runs over the mini-batches in a single complete epoch using cosine loss.
    :param: inputs from train_model function.
    :param epoch: current epoch to check if pretraining is over
    :return: accumalting loss over the epoch.
    """
    if len(args) == 3:  # validation does not exist
        optimizer, dataloader1, dataloader2 = args
    else:
        optimizer, dataloader1, dataloader2, _, _ = args
    running_loss = 0.0
    scores_list = []
    y_list = []
    # scheduler = ExponentialLR(optimizer, gamma=0.9)
    n_count_one = 0
    for i, data in enumerate(tqdm(zip(dataloader1, dataloader2), total=len(dataloader1)), 0):
        # get the inputs
        inputs1, labels1 = data[0]
        inputs2, labels2 = data[1]
        # send them to device
        inputs1 = inputs1.to(p.device)
        labels1 = labels1.to(p.device)
        inputs2 = inputs2.to(p.device)
        labels2 = labels2.to(p.device)
        n_count_one += torch.sum(torch.abs(torch.sum(inputs1 - inputs2,
                                                     axis=2).squeeze()) < 1e-10).item()  # count how many compared pairs contained the same RR
        # forward
        if epoch > p.pretraining_epoch:
            # outputs1 = model(inputs1, flag_DSU=p.flag_DSU)  # forward pass
            # outputs2 = model(inputs2, flag_DSU=p.flag_DSU)  # forward pass
            outputs1 = model(inputs1)
            outputs2 = model(inputs2)
            # outputs2, aug_loss, supp_loss = model(inputs2, flag_aug=False, flag_DSU=True, y=labels2[:, 1])
            task_loss = cosine_loss(outputs1, outputs2, labels1, labels2, flag=p.flag,
                                    lmbda=p.lmbda, b=p.b)
            loss = task_loss
            # supp_loss
            # backward + optimize
            # if aug_loss.detach().item() > 0:
            #     aug_task_ratio = np.abs(task_loss.detach().item() / aug_loss.detach().item())
            # else:
            #     aug_task_ratio = 0.0
            # if supp_loss.detach().item() > 0:
            #     supp_task_ratio = np.abs(task_loss.detach().item() / supp_loss.detach().item())
            # else:
            #     supp_task_ratio = 0.0
            # loss = -p.reg_aug * aug_task_ratio * aug_loss + p.reg_supp * supp_task_ratio * supp_loss + task_loss

        else:
            outputs1 = model(inputs1)  # forward pass
            outputs2 = model(inputs2)
            # backward + optimize
            loss = cosine_loss(outputs1, outputs2, labels1, labels2, flag=p.flag, lmbda=p.lmbda, b=p.b)
        # loss is calculated as mean over a batch
        optimizer.zero_grad()
        loss.backward()  # retain_graph=False even though there are multioutput from forward because we define
        # loss= loss1 + loss2 +loss3. This way, we can free up space in cuda. We mght also want to detach the values
        # when we assign them to loss vector, acc vector etc
        optimizer.step()
        # accumulate mean loss
        running_loss += loss.data.item()
        res_temp, y_temp = cosine_loss(outputs1, outputs2, labels1, labels2, flag=1, lmbda=p.lmbda, b=p.b)
        scores_list.append(0.5 * (res_temp + 1))  # making the cosine similarity as probability
        y_list.append(y_temp)
    # scheduler.step()
    # wandb.log({'learning_rate':scheduler.get_last_lr()})
    if p.calc_metric:
        err, best_acc, conf, y = calc_metric(scores_list, y_list, epoch, p)
        # error_analysis(dataloader1, dataloader2, conf, y)
        print('In training, {:.1f}% of the pairs contain the same RR'.format(
            100 * n_count_one / (dataloader1.batch_size * len(dataloader1))))
        return running_loss/len(dataloader1), err, best_acc
    print('In training, {:.1f}% of the pairs contain the same RR'.format(
        100 * n_count_one / (dataloader1.batch_size * len(dataloader1))))
    return running_loss/len(dataloader1)


def eval_model(model, p, epoch, *args):
    """
    This function evaluates the current learned model on validation set in every epoch or on testing set in a "single
    epoch".
    :param args: can be either [optimizer, trainloader1, trainloader1, valloader1, valloader2]
                            or [tesloader1, testloader2].
    :param: see train_model.
    :return:
    """
    p.log_path = p.working_folder
    if len(args) == 3:  # no validation
        raise NotImplementedError
        return eval_loss
    else:
        model.eval()
        eval_loss = eval_batches(model, p, epoch, *args)  # without "*" it would have built a tuple in a tuple
        return eval_loss


def eval_batches(model, p, epoch, *args) -> float:
    """
    This function runs evaluation over batches.
    :param: see eval_model
    :return: accumulating loss over an epoch
    """

    if len(args) == 5:  # validation (thus training is also there)
        _, _, _, dataloader1, dataloader2 = args
    else:  # (=2) only testing
        dataloader1, dataloader2 = args
    running_loss = 0.0
    scores_list = []
    y_list = []
    n_count_one = 0
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
            n_count_one += torch.sum(torch.abs(torch.sum(inputs1 - inputs2, axis=2).squeeze()) < 1e-10).item()  # count how many compared pairs contained the same RR
            # print('inputs1:{},inputs2:{},labels1:{},labels2:{}'.format(inputs1.shape, inputs2.shape, labels1.shape, labels2.shape))

            if epoch > p.pretraining_epoch:
                # outputs1 = model(inputs1, flag_DSU=p.flag_DSU)  # forward pass
                # outputs2 = model(inputs2, flag_DSU=p.flag_DSU)
                outputs1 = model(inputs1)
                outputs2 = model(inputs2)
                # outputs2, aug_loss, supp_loss = model(inputs2, flag_aug=False, flag_DSU=True, y=labels2[:, 1])
                task_loss = cosine_loss(outputs1, outputs2, labels1, labels2, flag=p.flag,
                                        lmbda=p.lmbda, b=p.b)
                loss = task_loss
                # supp_loss
                # backward + optimize
                # if aug_loss.detach().item() > 0:
                #     aug_task_ratio = np.abs(task_loss.detach().item() / aug_loss.detach().item())
                # else:
                #     aug_task_ratio = 0.0
                # if supp_loss.detach().item() > 0:
                #     supp_task_ratio = np.abs(task_loss.detach().item() / supp_loss.detach().item())
                # else:
                #     supp_task_ratio = 0.0
                # loss = -p.reg_aug * aug_task_ratio * aug_loss + p.reg_supp * supp_task_ratio * supp_loss + task_loss
            else:
                outputs1 = model(inputs1)  # forward pass
                outputs2 = model(inputs2)
                # backward + optimize
                loss = cosine_loss(outputs1, outputs2, labels1, labels2, flag=p.flag, lmbda=p.lmbda, b=p.b)
            # forward
            running_loss += loss.data.item()
            res_temp, y_temp = cosine_loss(outputs1, outputs2, labels1, labels2, flag=1, lmbda=p.lmbda, b=p.b)
            scores_list.append(0.5 * (res_temp + 1))  # making the cosine similarity as probability
            y_list.append(y_temp)
        if p.calc_metric:
            err, best_acc, best_th, y = calc_metric(scores_list, y_list, epoch, p, train_mode='Testing',
                                                    n_one_win=100*n_count_one/(dataloader1.batch_size*len(dataloader1)))
            if epoch == p.num_epochs:
                # error_analysis(dataloader1, dataloader2, model, best_th, p)
                pass
            print('In testing, {:.1f}% of the pairs contain the same RR'.format(100*n_count_one/(dataloader1.batch_size*len(dataloader1))))
            return running_loss/len(dataloader1), err, best_acc
    print('In testing, {:.1f}% of the pairs contain the same RR'.format(100*n_count_one/(dataloader1.batch_size*len(dataloader1))))
    return running_loss/len(dataloader1)


def calc_metric(scores_list, y_list, epoch, p, train_mode='Training', n_one_win=None):
    """
    This function calculates metrics relevant to verification task such as FAR, FRR, ERR, confusion matrix etc.
    :param scores_list: list of tensors (mini-batches) that are probability-like.
    :param y_list: list of tensors (mini-batches) where every example can have the value of 0 (not verified) or 1 (verified).
    :param epoch: current epoch number.
    :param train_mode: Training/Testing for title.
    :return: ERR + plotting every 10 epochs and priniting confusion matrix every epoch.
    """


    if not scores_list:
        y = 1
        err = 1
        best_acc = 0
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
        with open(pth + 'thresh.pkl', 'rb') as f:
            th_err, _, _, err_idx, acc_idx, F1_idx = pickle.load(f)
            best_th = th_err
        return err, best_acc, best_th.to(p.device), y  #.to(p.device)
    y = torch.cat(y_list)
    scores = torch.cat(scores_list)
    print(torch.sum(y) / len(y))
    # fpr, tpr, thresholds = metrics.roc_curve(y.detach().cpu(), scores.detach().cpu())
    # # https://stats.stackexchange.com/questions/272962/are-far-and-frr-the-same-as-fpr-and-fnr-respectively
    # far, frr, = fpr, 1 - tpr  # since frr = fnr
    # # thresholds -= 1
    # tr = np.flip(thresholds)
    # err_idx = np.argmin(np.abs(frr - far))
    # err = 0.5 * (frr[err_idx] + far[err_idx])
    # optimal_thresh = tr[err_idx]
    # res = torch.clone(scores)
    # res[scores >= optimal_thresh] = 1
    # res[scores < optimal_thresh] = 0
    #
    res_orig = 2 * scores - 1
    res_orig = res_orig.cpu().detach()
    y = y.cpu().detach()
    if train_mode == 'Training':
        thresh = torch.linspace(scores.min().item() - 1, scores.max().item() + 1, 100)
        far = torch.zeros_like(thresh)  # fpr
        frr = torch.zeros_like(thresh)  # 1 - tpr
        acc = torch.zeros_like(thresh)
        F1 = torch.zeros_like(thresh)
        conf_mat_tensor = torch.zeros((thresh.shape[0], 2, 2))
        for idx, trh in enumerate(thresh):
            res2 = torch.clone(res_orig)
            conf_mat = torch.zeros((2, 2))
            res2[res_orig >= trh] = 1
            res2[res_orig < trh] = 0
            conf = (res2 == y)
            conf_mat[0, 0] += torch.sum(1 * (conf[res2 == 0] == 1))
            conf_mat[0, 1] += torch.sum(1 * (conf[res2 == 1] == 0))
            conf_mat[1, 0] += torch.sum(1 * (conf[res2 == 0] == 0))
            conf_mat[1, 1] += torch.sum(1 * (conf[res2 == 1] == 1))
            far[idx] = conf_mat[0, 1] / conf_mat.sum(axis=1)[0]
            frr[idx] = 1 - (conf_mat[1, 1] / conf_mat.sum(axis=1)[1])
            acc[idx] = conf_mat.trace() / conf_mat.sum()
            PPV = conf_mat[1, 1]/(conf_mat.sum(axis=0)[1])
            Se = conf_mat[1, 1]/(conf_mat.sum(axis=1)[1])
            F1[idx] = 2*(PPV*Se)/(PPV+Se)
            conf_mat_tensor[idx, :, :] = conf_mat
        far[torch.isnan(far)] = 1
        frr[torch.isnan(frr)] = 1
        acc[torch.isnan(acc)] = 0
        F1[torch.isnan(F1)] = 0
        err_idx = np.argmin(np.abs(frr - far))
        err = 0.5 * (frr[err_idx] + far[err_idx])
        if p.wandb_enable:
            # if train_mode == 'Training':
            #     wandb.log({'training_thresh_err': thresh[err_idx]})
            # else:
            #     wandb.log({'testing_thresh_err': thresh[err_idx]})
            pass
        acc_idx = torch.argmax(acc)
        best_acc = acc[acc_idx]
        F1_idx = torch.argmax(F1)
        best_F1 = F1[F1_idx]
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
        with open(pth + 'thresh.pkl', 'wb') as f:
            pickle.dump([thresh[err_idx], thresh[acc_idx], thresh[F1_idx], err_idx, acc_idx, F1_idx], f)
        print('With ERR threshold of {:.2f} we got maximal accuracy of {:.2f} and maximal F1 of {:.2f}'
              .format(thresh[err_idx], acc[err_idx], F1[err_idx]))
        print(conf_mat_tensor[err_idx])
        print('With ACC threshold of {:.2f} we got maximal accuracy of {:.2f} and maximal F1 of {:.2f}'
              .format(thresh[acc_idx], acc[acc_idx], F1[acc_idx]))
        best_th = thresh[acc_idx]
        print(conf_mat_tensor[acc_idx])
        print('With F1 threshold of {:.2f} we got maximal accuracy of {:.2f} and maximal F1 of {:.2f}'
              .format(thresh[F1_idx], acc[F1_idx], F1[F1_idx]))
        print(conf_mat_tensor[F1_idx])

            # if epoch == p.num_epochs:

    else:
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
        with open(pth + 'thresh.pkl', 'rb') as f:
            th_err, th_acc, th_f1, err_idx, acc_idx, F1_idx = pickle.load(f)
            conf_mat_tensor = torch.zeros((3, 2, 2))
            stat_res = torch.zeros(3)  # three rows for different types of thresh and two columns for the results (acc and F1)
            th_list = [th_err, th_acc, th_f1]
            for idx, trh in enumerate(th_list):
                res2 = torch.clone(res_orig)
                conf_mat = torch.zeros((2, 2))
                res2[res_orig >= trh] = 1
                res2[res_orig < trh] = 0
                conf = (res2 == y)
                conf_mat[0, 0] += torch.sum(1 * (conf[res2 == 0] == 1))
                conf_mat[0, 1] += torch.sum(1 * (conf[res2 == 1] == 0))
                conf_mat[1, 0] += torch.sum(1 * (conf[res2 == 0] == 0))
                conf_mat[1, 1] += torch.sum(1 * (conf[res2 == 1] == 1))
                far = conf_mat[0, 1] / conf_mat.sum(axis=1)[0]
                frr = 1 - (conf_mat[1, 1] / conf_mat.sum(axis=1)[1])
                acc = conf_mat.trace() / conf_mat.sum()
                PPV = conf_mat[1, 1] / (conf_mat.sum(axis=0)[1])
                Se = conf_mat[1, 1] / (conf_mat.sum(axis=1)[1])
                F1 = 2 * (PPV * Se) / (PPV + Se)
                conf_mat_tensor[idx, :, :] = conf_mat
                far[torch.isnan(far)] = 1
                frr[torch.isnan(frr)] = 1
                acc[torch.isnan(acc)] = 0
                F1[torch.isnan(F1)] = 0
                err = 0.5 * (far + frr)
                if idx == 0:
                    stat_res[idx] = err
                elif idx == 1:
                    stat_res[idx] = acc
                else:
                    stat_res[idx] = F1

                best_acc = acc
                if p.wandb_enable:
                    # if train_mode == 'Training':
                    #     wandb.log({'training_thresh_err': thresh[err_idx]})
                    # else:
                    #     wandb.log({'testing_thresh_err': thresh[err_idx]})
                    pass
        # stat_res[0, 0]
        # print('With ERR threshold of {:.2f} we got minimal ERR of {:.2f} and maximal F1 of {:.2f}'
        #       .format(th_err, stat_res[0, 0], stat_res[0, 1]))
        # print(conf_mat_tensor[0])
        # print('With ACC threshold of {:.2f} we got maximal accuracy of {:.2f} and maximal F1 of {:.2f}'
        #       .format(th_acc, stat_res[1, 0], stat_res[1, 1]))
        # print(conf_mat_tensor[1])
        # print('With F1 threshold of {:.2f} we got maximal accuracy of {:.2f} and maximal F1 of {:.2f}'
        #       .format(th_f1, stat_res[2, 0], stat_res[2, 1]))
        print('With ERR threshold of {:.2f} we got minimal ERR of {:.2f}'
              .format(th_err, stat_res[0]))
        print(conf_mat_tensor[0])
        print('With ACC threshold of {:.2f} we got maximal accuracy of {:.2f}'
              .format(th_acc, stat_res[1]))
        print(conf_mat_tensor[1])
        print('With F1 threshold of {:.2f} we got  maximal F1 of {:.2f}'
              .format(th_f1, stat_res[2]))
        print(conf_mat_tensor[2])
        # ii = torch.argmax(stat_res[:, 1])
        best_th = th_list[2]  # meaning gave the best F1 on test
        d = {}
        if p.run_saved_models:
            if p.rearrange:
                d['Rearrange'] = 'T'
                # xl_pth = '/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/rr_datasets/n_beats_' \
                #          + str(p.n_beats) + '/inference/rearrange_True/res_tbl_nbeats_' \
                #                             + str(p.n_beats)  + '_True.xlsx'
            else:
                d['Rearrange'] = 'F'
            #     xl_pth = '/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/rr_datasets/n_beats_' \
            #              + str(p.n_beats) + '/inference/rearrange_False/res_tbl_nbeats_' \
            #              + str(p.n_beats) + '_False.xlsx'
            # xl_file = pd.read_excel(xl_pth)
            if p.med_mode == 'a':
                d['State'] = 'int'
                # xl_file.loc[xl_file['State'] == 'int', p.age] = stat_res[:, 1].numpy()
            elif p.med_mode == 'c':
                d['State'] = 'basal'
                # xl_file.loc[xl_file['State'] == 'basal', p.age] = stat_res[:, 1].numpy()
            elif p.med_mode == 'both':
                d['State'] = 'comb'
                # xl_file.loc[xl_file['State'] == 'comb', p.age] = stat_res[:, 1].numpy()
            # os.remove(xl_pth)
            # xl_file.to_excel(xl_pth, columns=xl_file.columns)
            if p.up2_21 and not p.bas_on_int:
                'up2_21_bas_on_int'
                mode = 'up2_21'
                load_and_dump(mode, d, p, n_one_win, stat_res)
            elif p.up2_21 and p.bas_on_int:
                mode = 'up2_21_bas_on_int'
                load_and_dump(mode, d, p, n_one_win, stat_res)
            else:
                if p.equal_non_equal == 'equal':
                    if p.bas_on_int:
                        mode = 'bas_on_int'
                        load_and_dump(mode, d, p, n_one_win, stat_res)
                        # with open('/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/rr_datasets_equal/res_df_bas_on_int.pkl',
                        #           'rb') as f:
                        #     res_df = pickle.load(f)
                        # res_df.loc[(res_df.loc[:, 'Rearrange'] == d['Rearrange']) & (res_df.loc[:, 'State'] == d['State'])
                        #            & (res_df.loc[:, 'nbeats'] == p.n_beats), str(p.age)] = stat_res.numpy()
                        # with open('/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/rr_datasets_equal/res_df_bas_on_int.pkl',
                        #           'wb') as f:
                        #     pickle.dump(res_df, f)

                    else:
                        mode = 'equal'
                        load_and_dump(mode, d, p, n_one_win, stat_res)
                        # with open('/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/rr_datasets_equal/res_df_equal.pkl', 'rb') as f:
                        #      res_df = pickle.load(f)
                        # res_df.loc[(res_df.loc[:, 'Rearrange'] == d['Rearrange']) & (res_df.loc[:, 'State'] == d['State'])
                        #            & (res_df.loc[:, 'nbeats'] == p.n_beats), str(p.age)] = stat_res.numpy()
                        # with open('/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/rr_datasets_equal/res_df_equal.pkl', 'wb') as f:
                        #      pickle.dump(res_df, f)
                else:
                    mode = 'non_equal'
                    load_and_dump(mode, d, p, n_one_win, stat_res)
                    # with open('/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/rr_datasets/res_df.pkl',
                    #           'rb') as f:
                    #     res_df = pickle.load(f)
                    # res_df.loc[(res_df.loc[:, 'Rearrange'] == d['Rearrange']) & (res_df.loc[:, 'State'] == d['State'])
                    #            & (res_df.loc[:, 'nbeats'] == p.n_beats), str(p.age)] = stat_res.numpy()
                    # with open('/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/rr_datasets/res_df.pkl',
                    #           'wb') as f:
                    #     pickle.dump(res_df, f)

    if epoch < 10:
        str_epoch = '0' + str(epoch)
    else:
        str_epoch = str(epoch)
    if train_mode == 'Training':
        name1 = 'conf_err_train_epoch_'
        name2 = 'conf_acc_train_epoch_'
        name3 = 'conf_f1_train_epoch_'
        if epoch == p.num_epochs:
            torch.save(conf_mat_tensor[err_idx], p.log_path + '/' + name1 + str_epoch + '.pt')
            torch.save(conf_mat_tensor[acc_idx], p.log_path + '/' + name2 + str_epoch + '.pt')
            torch.save(conf_mat_tensor[F1_idx], p.log_path + '/' + name3 + str_epoch + '.pt')
            res2 = torch.clone(res_orig)
            # paint orange res2[res2>= thresh[err_idx]] and blue res2[res < thresh[err_idx]], x_axis is res2 itself. add dashed
            # line of optimal threshold. Do the same for accuracy, do it wuth subplot
            sns.histplot(x=res2, hue=y, bins=50)  # , stat='probability', bins=int(np.ceil(0.4*len(y))))
            plt.axvline(x=thresh[acc_idx], color='r', linestyle='dashed')
            name1 = '/histo_train_epoch_'
            name2 = '/err_acc_train_epoch_'
            plt.xlabel('Cosine similarity [N.U]')
            plt.title('{} mode: ACC = {:.2f}%, tr = {:.2f}'.format(train_mode, 100 * best_acc, thresh[acc_idx]))
            if p.rearrange:
                plt.savefig(p.log_path + '/trained_on_6m_rearrange_True' + name1 + str_epoch + '.png')
            else:
                plt.savefig(p.log_path + '/trained_on_6m_rearrange_False' + name1 + str_epoch + '.png')
            plt.close()
            # if np.mod(epoch, 10) == 0:
            plt.plot(thresh, far, thresh, frr)  # , thresh, acc)
            plt.legend(['FAR', 'FRR'])  # , 'ACC'])
            # plt.title('{} mode: ERR = {:.2f}%, tr = {:.2f}, ACC = {:.2f}%, tr = {:.2f}'.format(train_mode, 100*err, thresh[err_idx], 100*best_acc, thresh[acc_idx]))
            plt.xlabel('Cosine similarity [N.U]')
            plt.ylabel('Error [N.U]')
            plt.xlim([-1, 1])
            plt.title('{} mode: ERR = {:.2f}%, tr = {:.2f}'.format(train_mode, 100 * err, thresh[err_idx]))
            if p.rearrange:
                plt.savefig(p.log_path + '/trained_on_6m_rearrange_True' + name2 + str_epoch + '.png')
            else:
                plt.savefig(p.log_path + '/trained_on_6m_rearrange_False' + name2 + str_epoch + '.png')
            # plt.savefig(p.log_path + name2 + str_epoch + '.png')
            plt.close()
    else:
        if (epoch == p.num_epochs) or p.run_saved_models:
            name1 = 'conf_err_val_epoch_'
            name2 = 'conf_acc_val_epoch_'
            name3 = 'conf_f1_val_epoch_'
            if p.rearrange:
                test_path = p.log_path + 'inference/' + 'rearrange_True/'
                if not(os.path.isdir(test_path)):
                    if not (os.path.isdir(p.log_path + 'inference/')):
                        os.mkdir(p.log_path + 'inference/')
                        os.mkdir(test_path)
                    else:
                        os.mkdir(test_path)
            else:
                test_path = p.log_path + 'inference/' + 'rearrange_False/'
                if not(os.path.isdir(test_path)):
                    if not (os.path.isdir(p.log_path + 'inference/')):
                        os.mkdir(p.log_path + 'inference/')
                        os.mkdir(test_path)
                    else:
                        os.mkdir(test_path)
            if p.med_mode == 'a':
                if not (os.path.isdir(test_path + '/' + 'intrinsic')):
                    os.mkdir(test_path + '/' + 'intrinsic')
                test_path = test_path + '/' + 'intrinsic/'
            elif p.med_mode == 'c':
                if not (os.path.isdir(test_path + '/' + 'basal')):
                    os.mkdir(test_path + '/' + 'basal')
                test_path = test_path + '/' + 'basal/'
            elif p.med_mode == 'both':
                if not (os.path.isdir(test_path + '/' + 'combined')):
                    os.mkdir(test_path + '/' + 'combined')
                test_path = test_path + '/' + 'combined/'
            torch.save(conf_mat_tensor[0], test_path + str(p.age) + 'm_' + name1 + str_epoch + '.pt')
            torch.save(conf_mat_tensor[1], test_path + str(p.age) + 'm_' + name2 + str_epoch + '.pt')
            torch.save(conf_mat_tensor[2], test_path + str(p.age) + 'm_' + name3 + str_epoch + '.pt')


            res2 = torch.clone(res_orig)
            name1 = 'histo_val_epoch_'
            fig = plt.subplots(figsize=(15, 15))
            ax = sns.histplot(x=res2, hue=y, legend=False, kde=False, bins=50)
            ax.set_xlabel(xlabel="Cosine similarity [N.U]", fontsize=24, weight='bold')
            # ax.set_ylabel(ylabel='Estimated probability density')
            ax.set_ylabel(ylabel="Count", fontsize=24, weight='bold')
            ax.set_xticklabels([str("{:.1f}".format(i)) for i in ax.get_xticks()], fontsize=24, weight='bold')
            ax.set_yticklabels([str("{:.1f}".format(i)) for i in ax.get_yticks()], fontsize=24, weight='bold')

            # ax.set_yticklabels(ax.get_yticks(), fontsize=24, weight='bold')
            for _, s in ax.spines.items():
                s.set_linewidth(5)
            plt.legend(['Positive GT', 'Negative GT'], fontsize=34)
            plt.axvline(x=th_acc, color='r', linestyle='dashed', lw=6)
            sns.despine(fig=None, ax=ax, top=True, right=True, left=False, bottom=False, offset=None, trim=False)
            # plt.show()
            if p.bas_on_int:
                plt.savefig(test_path + str(p.age) + 'm_bas_on_int' + name1 + str_epoch + '.png')
            else:
                plt.savefig(test_path + str(p.age) + 'm_' + name1 + str_epoch + '.png')
            plt.close()
    return err, best_acc, best_th.to(scores.device), y.to(scores.device)


def load_and_dump(mode, d, p, n_one_win, stat_res):
    pkl_pth = '/home/smorandv/ac8_and_aging_NEW/ac8_and_aging/rr_datasets'
    mode2fig = {'equal': '_equal/', 'non_equal': '/', 'bas_on_int': '_equal/bas_on_int/', 'up2_21': '_equal_up2_21/',
                'up2_21_bas_on_int': '_equal_up2_21/bas_on_int/'}
    pkl_pth += mode2fig[mode] + 'res_df_' + str(p.bootstrap_idx) + '.pkl'
    with open(pkl_pth, 'rb') as f:
        res_df = pickle.load(f)
    res_df.loc[(res_df.loc[:, 'Rearrange'] == d['Rearrange']) & (res_df.loc[:, 'State'] == d['State'])
               & (res_df.loc[:, 'nbeats'] == p.n_beats), ['n_one_beats_' + str(p.age), str(p.age)]] = \
        np.hstack([np.tile(n_one_win, (3, 1)), np.expand_dims(stat_res.numpy(), axis=1)])
    with open(pkl_pth, 'wb') as f:
        pickle.dump(res_df, f)


def error_analysis(dataloader1, dataloader2, model, best_th, p):
    misrejected_pairs = []
    misaccepted_pairs = []
    correctly_rejected_pairs = []
    correctly_accepted_pairs = []
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
            outputs1 = model(inputs1)
            outputs2 = model(inputs2)
            res_temp, y_temp = cosine_loss(outputs1, outputs2, labels1, labels2, flag=1, lmbda=p.lmbda, b=p.b)
            res2 = torch.clone(res_temp)
            res2[res_temp >= best_th] = 1
            res2[res_temp < best_th] = 0
            conf = (res2 == y_temp)
            misrejected = torch.argwhere(torch.logical_and((conf==False), (y_temp==1))).squeeze()  # meaning they are the same but predicted that they are not
            misaccepted = torch.argwhere(torch.logical_and((conf==False), (y_temp==0))).squeeze()
            correctly_rejected = torch.argwhere(torch.logical_and((conf==True), (y_temp==0))).squeeze()
            correctly_accepted = torch.argwhere(torch.logical_and((conf==True), (y_temp==1))).squeeze()
            # todo: check what happens if one of the above is an empty tensor
            misrejected_pairs.append((inputs1[misrejected], inputs2[misrejected]))
            misaccepted_pairs.append((inputs1[misaccepted], inputs2[misaccepted]))
            correctly_rejected_pairs.append((inputs1[correctly_rejected], inputs2[correctly_rejected]))
            correctly_accepted_pairs.append((inputs1[correctly_accepted], inputs2[correctly_accepted]))
        # rr_diff = dict(misrejected_pairs=[], misaccepted_pairs=[], correctly_rejected_pairs=[], correctly_accepted_pairs=[])
        rr_diff_mean = dict([(0, []), (1, []), (2, []), (3, [])])
        for pair_tup in zip(misrejected_pairs, misaccepted_pairs, correctly_rejected_pairs, correctly_accepted_pairs):
            for jj in range(len(pair_tup)):
                if pair_tup[jj][0].nelement() > 0:
                    # rr_diff = pair_tup[jj][0] - pair_tup[jj][1]
                    if pair_tup[jj][0].ndim < 3:
                        rr_diff_mean[jj].append(torch.abs(60/(p.mu + pair_tup[jj][0].mean(axis=1).squeeze())-60/(p.mu + pair_tup[jj][1].mean(axis=1).squeeze())))
                    else:
                        rr_diff_mean[jj].append(torch.abs(60/(p.mu + pair_tup[jj][0].mean(axis=2).squeeze())-60/(p.mu + pair_tup[jj][1].mean(axis=2).squeeze())))
        f11 = lambda tensor_list: [val.unsqueeze(dim=0) if val.ndim < 1 else val for val in tensor_list]
        for key in rr_diff_mean:
            rr_diff_mean[key] = f11(rr_diff_mean[key])
            rr_diff_mean[key] = torch.cat(rr_diff_mean[key])
            print('Median={:.2f}, Q1={:.2f}, Q3={:.2f}, min={:.2f}, max={:.2f}'.format(torch.median(rr_diff_mean[key]).item(),
                                                                                       torch.quantile(rr_diff_mean[key], 0.25).item(),
                                                                                       torch.quantile(rr_diff_mean[key], 0.75).item(),
                                                                                       torch.min(rr_diff_mean[key]).item(),
                                                                                       torch.max(rr_diff_mean[key]).item()))
    ax = sns.boxplot(data=[rr_diff_mean[0].cpu(),rr_diff_mean[1].cpu(),rr_diff_mean[2].cpu(),rr_diff_mean[3].cpu()])
    ax.set_xticklabels(["misrejected", "misaccepted", "corr_rejected", "corr_accepted"])
    ax.set_ylabel('Avg. HR difference [bpm]')
    plt.show()
    plt.savefig(p.log_path + '/inference/' + 'HR_diff.png', ax)
    plt.close()
    # plt.show()
    a=1


def write2txt(lines, pth, p):
    # lines += ['n_training = {}, n_validation = {}.'.format(p.n_train, p.n_val),
    #           ]
    lines += ['{} = {}'.format(attr, value) for attr, value in vars(p).items()]
    with open(pth + '/README.txt', 'w') as f:
        f.write('\n'.join(lines))


