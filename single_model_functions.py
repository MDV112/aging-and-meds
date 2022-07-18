import os

import numpy as np
import torch
import torch.nn as nn
from data_loader import TorchDataset
import time
import copy
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler as MinMax
from sklearn import metrics
from sklearn.utils import shuffle
from datetime import datetime


def load_datasets(full_pickle_path: str, p: object, mode: int = 0, train_mode: bool = True) -> object:
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
        with open(full_pickle_path, 'rb') as f:
            e = pickle.load(f)
            if train_mode:
                x = e.x_train_specific
                # x = x - x.mean(axis=0)
                # x = x - x.min()
                y = e.y_train_specific
                p.train_ages = np.unique(y['age'])
                print('Ages used in training set are {}'. format(np.unique(y['age'])))
            else:
                x = e.x_test_specific
                # x = x - x.mean(axis=0)
                # x = x - x.min(axis=0)
                y = e.y_test_specific
                p.test_ages = np.unique(y['age'])
                print('Ages used in training set are {}'. format(np.unique(y['age'])))
        x_c, x_a = x[:, y['med'] == 0], x[:, y['med'] == 1]
        y_c, y_a = y[['id', 'age']][y['med'] == 0].values.astype(int), y[['id', 'age']][y['med'] == 1].values.astype(int)
        x_c_T, y_c = shuffle(x_c.T, y_c, random_state=0)  # shuffle is used here for shuffeling ages
        # because in the dataloader it should be false. sklearn should make the shuffle the same. Notice the transpose
        # in x since we want to shuffle the columns fo RR
        x_a_T, y_a = shuffle(x_a.T, y_a, random_state=0)
        x_c = x_c_T.T
        x_a = x_a_T.T
        if p.med_mode == 'c':
            p.n_train = x_c.shape[1]
            dataset = HRVDataset(x_c.T, y_c, mode=mode)  # transpose should fit HRVDataset
        elif p.med_mode == 'a':
            p.n_train = x_c.shape[1]
            dataset = HRVDataset(x_a.T, y_a, mode=mode)  # transpose should fit HRVDataset
        else:  # other medications
            raise NotImplementedError
    else:  # Koopman HRV
        #todo: add age tags
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
            dataset = HRVDataset(np_dataset, label_dataset, mode=mode)
        elif p.med_mode == 'a':
            label_dataset = y_a
            if len(p.feat2drop) != 0:
                x_a.drop(p.feat2drop, axis=1, inplace=True)
            np_dataset = np.array(x_a.values, dtype=np.float)
            dataset = HRVDataset(np_dataset, label_dataset, mode=mode)
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
        tags = np.unique(dataset.y[:,0])
        val_tags = np.random.choice(tags, int(np.floor(p.val_size*len(tags))), replace=False)
        train_tags = np.setdiff1d(tags, val_tags)
        p.n_individuals_train = len(train_tags)
        p.n_individuals_val = len(val_tags)
        train_mask = np.isin(dataset.y[:, 0], train_tags)
        val_mask = np.isin(dataset.y[:, 0], val_tags)
        x_train = dataset.x[train_mask, :]
        y_train = dataset.y[train_mask, :]
        x_val = dataset.x[val_mask, :]
        y_val = dataset.y[val_mask, :]
    else:  #todo: check if split is done correctly here
        x_train, x_val, y_train, y_val = train_test_split(dataset.x, dataset.y[:,0], test_size=p.val_size)
    p.n_train = x_train.shape[0]
    p.n_val = x_val.shape[0]
    return x_train, y_train, x_val, y_val


def scale_dataset(*args, input_scaler=None, mode=0, should_scale: bool = False) -> tuple:
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
            else:  #todo: remove mean rr from every example
                x_train = args[0]
                x_val = args[2]
            return HRVDataset(x_train, args[1], mode=mode), HRVDataset(x_val, args[3], mode=mode), scaler
        elif len(args) == 2:  # HRVdataset(train) and HRVdataset(test)
            if should_scale:
                args[0].x = scaler.fit_transform(args[0].x)  # Notice that this won't do anything if training is already scale
                args[1].x = scaler.transform(args[1].x)
            return args
    else:
        if len(args) != 1:
            raise Exception('Only test set can be an input')
        else:
            if should_scale:
                args[0].x = input_scaler.transform(args[0].x)
            return args


def train_model(model: object, p: object, *args):
    """
    Training the model.
    :param model: chosen neural network.
    :param p: ProSet (project setting) object.
    :param args: Can be either [optimizer, trainloader1, trainloader1, valloader1, valloader2] or
                                [optimizer, trainloader1, trainloader1]
    :param calc_metric: caluclate metrics such as FAR, FPR etc.
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
    training_err_vector = np.zeros(p.num_epochs)
    val_err_vector = np.zeros(p.num_epochs)
    train_acc_vector = np.zeros(p.num_epochs)
    val_acc_vector = np.zeros(p.num_epochs)
    best_val_ERR = 1
    best_val_acc = 0
    best_ERR_diff = 1
    best_ACC_diff = 1
    pth = p.log_path + now.strftime("%b-%d-%Y_%H_%M_%S")
    os.mkdir(pth)
    for epoch in range(1, p.num_epochs + 1):
        epoch_time = time.time()
        if p.calc_metric:
            training_loss, training_err_vector[epoch - 1], train_acc_vector[epoch - 1] = train_batches(model, p, epoch, *args)
            validation_loss, val_err_vector[epoch - 1], val_acc_vector[epoch - 1] = eval_model(model, p, epoch, *args)
            if val_err_vector[epoch - 1] < best_val_ERR:
                best_val_ERR = val_err_vector[epoch - 1]
                torch.save(copy.deepcopy(model.state_dict()), pth + '/best_ERR_model.pt')
            if val_acc_vector[epoch - 1] > best_val_acc:
                best_val_acc = val_acc_vector[epoch - 1]
                torch.save(copy.deepcopy(model.state_dict()), pth + '/best_ACC_model.pt')
            if np.abs(val_err_vector[epoch - 1] - training_err_vector[epoch - 1]) < best_ERR_diff:
                best_ERR_diff = np.abs(val_err_vector[epoch - 1] - training_err_vector[epoch - 1])
                torch.save(copy.deepcopy(model.state_dict()), pth + '/best_ERR_diff_model.pt')
            if np.abs(val_acc_vector[epoch - 1] - train_acc_vector[epoch - 1]) < best_ACC_diff:
                best_ACC_diff = np.abs(val_acc_vector[epoch - 1] - train_acc_vector[epoch - 1])
                torch.save(copy.deepcopy(model.state_dict()), pth + '/best_ACC_diff_model.pt')
        else:
            training_loss = train_batches(model, p, epoch, *args)
            validation_loss = eval_model(model, p, epoch, *args)
        training_loss /= len(args[1])  # len of trainloader
        if len(args) > 3:  # meaning validation exists.
            validation_loss /= len(args[3])  # len of valloader
        if p.calc_metric:
            log = "Epoch: {} | Training loss: {:.4f}  | Validation loss: {:.4f}  |  Training ERR: {:.4f}  |" \
                  "  Validation ERR: {:.4f}  |  ".format(epoch, training_loss, validation_loss, training_err_vector[epoch - 1],
                                                    val_err_vector[epoch - 1])
        else:
            log = "Epoch: {} | Training loss: {:.4f}  | Validation loss: {:.4f}   ".format(epoch, training_loss, validation_loss)
        epoch_time = time.time() - epoch_time
        log += "Epoch Time: {:.2f} secs".format(epoch_time)
        print(log)
    if p.calc_metric:
        idx_val_min = np.argmin(val_err_vector)
        idx_val_max = np.argmax(val_acc_vector)
        idx_min_diff_err = np.argmin(np.abs(val_err_vector - training_err_vector))
        idx_min_diff_acc = np.argmin(np.abs(val_acc_vector - train_acc_vector))

        plt.plot(np.arange(1, p.num_epochs + 1), 100*training_err_vector, np.arange(1, p.num_epochs + 1), 100*val_err_vector)
        plt.legend(['ERR Train', 'ERR Test'])
        plt.ylabel('ERR [%]')
        plt.xlabel('epochs')
        plt.show()
        # todo:  threshold
        lines = ['Minimal validation ERR was {:.2f}% in epoch number {}. Training ERR at the same epoch was: {:.2f}%.'
                 .format(100 * np.min(val_err_vector), 1 + idx_val_min, 100 * training_err_vector[idx_val_min]),
                 'Maximal validation accuracy was {:.2f}% in epoch number {}. Training accuracy at the same epoch was: {:.2f}%.'
                 .format(100 * np.max(val_acc_vector), 1 + idx_val_max, 100 * train_acc_vector[idx_val_max]),
                 'Minimal absolute value EER difference was {:.2f} in epoch number {}.'.format(np.abs(val_err_vector[idx_min_diff_err]
                                                        - training_err_vector[idx_min_diff_err]), 1 + idx_min_diff_err),
                 'Minimal absolute value ACC difference was {:.2f} in epoch number {}.'.format(
                     np.abs(val_acc_vector[idx_min_diff_acc]- train_acc_vector[idx_min_diff_acc]), 1 + idx_min_diff_acc)
                 ]

        write2txt(lines, pth, p)


        plt.plot(np.arange(1, p.num_epochs + 1), 100 * train_acc_vector, np.arange(1, p.num_epochs + 1),
                 100 * val_acc_vector)
        plt.legend(['ACC Train', 'ACC Test'])
        plt.ylabel('ACC [%]')
        plt.xlabel('epochs')
        plt.show()
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
        if epoch > p.pretraining_epoch:
            outputs1= model(inputs1)  # forward pass
            outputs2, aug_loss, supp_loss = model(inputs2, flag_aug=True, y=labels2[:, 1])
            # supp_loss
            # backward + optimize
            loss = p.reg_aug*aug_loss + p.reg_supp*supp_loss  + cosine_loss(outputs1, outputs2, labels1, labels2, flag=p.flag,
                                                                   lmbda=p.lmbda, b=p.b)

        else:
            outputs1 = model(inputs1)  # forward pass
            outputs2 = model(inputs2)
            # backward + optimize
            loss = cosine_loss(outputs1, outputs2, labels1, labels2, flag=p.flag, lmbda=p.lmbda, b=p.b)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)  # backpropagation
        optimizer.step()
        # accumulate mean loss
        running_loss += loss.data.item()
        res_temp, y_temp = cosine_loss(outputs1, outputs2, labels1, labels2, flag=1, lmbda=p.lmbda, b=p.b)
        scores_list.append(0.5 * (res_temp + 1))  # making the cosine similarity as probability
        y_list.append(y_temp)
    optimizer.zero_grad()
    if p.calc_metric:
        err, best_acc, conf, y = calc_metric(scores_list, y_list, epoch)
        error_analysis(dataloader1, dataloader2, conf, y)
        return running_loss, err, best_acc
    return running_loss

def eval_model(model, p, epoch, *args):
    """
    This function evaluates the current learned model on validation set in every epoch or on testing set in a "single
    epoch".
    :param args: can be either [optimizer, trainloader1, trainloader1, valloader1, valloader2]
                            or [tesloader1, testloader2].
    :param: see train_model.
    :return:
    """
    if len(args) == 3:  # no validation
        eval_loss = 0
        return eval_loss
    else:
        model.eval()
        eval_loss = eval_batches(model, p, epoch, *args)  # without "*" it would have built a tuple in a tuple
        return eval_loss

def eval_batches(model, p,  epoch, *args) -> float:
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

            if epoch > p.pretraining_epoch:
                outputs1 = model(inputs1)  # forward pass
                outputs2, aug_loss, supp_loss = model(inputs2, flag_aug=True, y=labels2[:, 1])
                # backward + optimize
                # supp_loss
                loss = p.reg_aug*aug_loss + p.reg_supp*supp_loss + cosine_loss(outputs1, outputs2, labels1, labels2,
                                                                       flag=p.flag, lmbda=p.lmbda, b=p.b)
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
            err, best_acc, conf, y = calc_metric(scores_list, y_list, epoch, train_mode='Testing')
            return running_loss, err, best_acc
    return running_loss

def calc_metric(scores_list, y_list, epoch, train_mode='Training'):
    """
    This function calculates metrics relevant to verification task such as FAR, FRR, ERR, confusion matrix etc.
    :param scores_list: list of tensors (mini-batches) that are probability-like.
    :param y_list: list of tensors (mini-batches) where every example can have the value of 0 (not verified) or 1 (verified).
    :param epoch: current epoch number.
    :param train_mode: Training/Testing for title.
    :return: ERR + plotting every 10 epochs and priniting confusion matrix every epoch.
    """
    scores = torch.cat(scores_list)
    y = torch.cat(y_list)

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
    thresh = torch.linspace(scores.min().item() - 1, scores.max().item() + 1, 30)
    far = torch.zeros_like(thresh)  # fpr
    frr = torch.zeros_like(thresh)  # 1 - tpr
    acc = torch.zeros_like(thresh)
    conf_mat_tensor = torch.zeros((thresh.shape[0], 2, 2))
    for idx, trh in enumerate(thresh):
        res2 = torch.clone(res_orig)
        conf_mat = torch.zeros((2, 2))
        res2[res_orig >= trh] = 1
        res2[res_orig < trh] = 0
        conf = (res2 == y)
        conf_mat[0, 0] += torch.sum(1*(conf[res2 == 0] == 1))
        conf_mat[0, 1] += torch.sum(1*(conf[res2 == 1] == 0))
        conf_mat[1, 0] += torch.sum(1*(conf[res2 == 0] == 0))
        conf_mat[1, 1] += torch.sum(1*(conf[res2 == 1] == 1))
        far[idx] = conf_mat[0, 1]/conf_mat.sum(axis=1)[0]
        frr[idx] = 1 - (conf_mat[1, 1]/conf_mat.sum(axis=1)[1])
        acc[idx] = conf_mat.trace()/conf_mat.sum()
        conf_mat_tensor[idx, :, :] = conf_mat
    err_idx = np.argmin(np.abs(frr - far))
    err = 0.5 * (frr[err_idx] + far[err_idx])
    acc_idx = torch.argmax(acc)
    best_acc = acc[acc_idx]
    print('With threshold of {:.2f} we got minimal ERR of {:.2f} and accuracy of  {:.2f}'.format(thresh[err_idx], err,
                                                                                               acc[err_idx]))
    print(conf_mat_tensor[err_idx])
    print('With threshold of {:.2f} we got maximal accuracy of {:.2f} and ERR of  {:.2f}'.format(thresh[acc_idx],
                                                                        best_acc, 0.5 * (frr[acc_idx] + far[acc_idx])))
    print(conf_mat_tensor[acc_idx])

    if np.mod(epoch, 10) == 0:
        plt.plot(thresh, far, thresh, frr, thresh, acc)
        plt.legend(['FAR', 'FRR', 'ACC'])
        plt.title('{} mode: ERR = {:.2f}%, epoch = {}, ACC = {:.2f}%'.format(train_mode, 100*err, epoch, 100*best_acc))
        plt.show()
    return err, best_acc, conf.to(scores.device), y.to(scores.device)

def error_analysis(dataloader1, dataloader2, conf, y):
    # wrong_pairs = (dataloader1.dataset[conf == 0], dataloader2.dataset[conf == 0])
    # misrejected = (wrong_pairs[0][y == 1], wrong_pairs[1][y == 1])  # meaning they are the same but predicted that they are not
    # misaccepted = (wrong_pairs[0][y == 0], wrong_pairs[1][y == 0])

    pass

def write2txt(lines, pth, p):
    # lines += ['n_training = {}, n_validation = {}.'.format(p.n_train, p.n_val),
    #           ]
    lines += ['{} = {}'.format(attr, value) for attr, value in vars(p).items()]
    with open(pth + '/README.txt', 'w') as f:
        f.write('\n'.join(lines))



