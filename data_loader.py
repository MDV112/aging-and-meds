import pandas as pd
import numpy as np
import h5py
import scipy.io as sio
from pathlib import Path
import os
from torch.utils.data import Dataset
import torch.nn as nn
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import random
import pickle


class Dataloader:
    def __init__(self, data_path=os.getcwd(), group='aging', dataset_name=[3, 0], input_type='features'):
        self.data_path = data_path  # wherever the h5 files are located at
        self.group = group  # either aging or ac8
        self.dataset_name = dataset_name  # can be either number of beats or a 2 element list such as [3, 0] for
        # win_len and phase
        self.input_type = input_type  # either features or raw
        self.loaded = False  # flag for loading before splitting
        self.splitted = False
        self.ds_input = None
        self.ds_output = None
        self.feat_names = None
        self.X_train = None
        self.x_test = None
        self.Y_train = None
        self.y_test = None
        self.x_train_specific = None
        self.y_train_specific = None
        self.x_test_specific = None
        self.y_test_specific = None

    def load(self):
        """
        This function loads the correct dataset according to request
        :return:
        input: Either HRV features (pandas) or RR intervals
        output: [id, age, med, num_win] (pandas)
        feat_names: HRV features names
        """
        if self.input_type == 'features':
            if type(self.dataset_name) != list:
                Exception('dataset_name has to be a list representing win_len and phase')
            fname = Path.joinpath(Path(self.data_path), 'hrv.h5')
            f = h5py.File(fname, "r")
            ds_keys = []
            f.visit(lambda key: ds_keys.append(key) if type(f[key]) is h5py._hl.dataset.Dataset else None)
            matching = [s for s in ds_keys if str(self.dataset_name[0]) in s and str(self.dataset_name[1]) in s]
            ds_input = f[matching[0]].value
            ds_output = f[matching[1]].value.astype(int)
            mat_files = [f for f in os.listdir(Path(self.data_path)) if f.endswith('_featuresNames.mat')]
            new_list = self.dataset_name.copy()
            if self.group == 'aging':
                new_list.append('aging')
            else:
                new_list.append('ac8')
            for file in mat_files:
                t = [True for c in new_list if file.find(str(c)) != -1]
                if all(t):
                    w = sio.loadmat(Path.joinpath(Path(self.data_path), file))
                    feat_names_obj = w['bas_vars']
                    feat_names = [feat_names_obj[0][i][0] for i in range(feat_names_obj.size)]
                    self.feat_names = feat_names
                else:
                    raise Exception('Adequate mat file for features names was not found in ' + str(Path(self.data_path)))
            self.ds_input = pd.DataFrame(ds_input.T, columns=feat_names)
            self.ds_output = pd.DataFrame(ds_output.T, columns=['id', 'age', 'med', 'win_num'])
            self.loaded = True
            lbls_drop = self.ds_input[(self.ds_output['id'] == 726) & (self.ds_output['age'] == 27)].index
            self.ds_input.drop(labels=lbls_drop, inplace=True)
            self.ds_output.drop(labels=lbls_drop, inplace=True)
            # self.ds_output[(self.ds_output['id'] == 726) & (self.ds_output['age'] == 27)].drop(inpalce=True)
            return self.ds_input, self.ds_output, self.feat_names
        else:
            if type(self.dataset_name) != int:
                raise Exception('dataset_name has to be an integer representing number of beats')
            fname = Path.joinpath(Path(self.data_path), 'rr.h5')
            f = h5py.File(fname, "r")
            ds_keys = []
            f.visit(lambda key: ds_keys.append(key) if type(f[key]) is h5py._hl.dataset.Dataset else None)
            matching = [s for s in ds_keys if str(self.dataset_name) in s]
            self.ds_input = f[matching[0]][()]
            ds_output = f[matching[1]][()].astype(int)
            self.ds_output = pd.DataFrame(ds_output.T, columns=['id', 'age', 'med', 'win_num'])
            self.loaded = True
            #todo: check why dropping 726 in the age of 27 does not work in main

            # lbls_drop = self.ds_input[(self.ds_output['id'] == 726) & (self.ds_output['age'] == 27)].index
            # self.ds_input.drop(labels=lbls_drop, inplace=True)
            # self.ds_output.drop(labels=lbls_drop, inplace=True)
            return self.ds_input, self.ds_output

    def split(self, seed=42, test_size=0.2):
        """
        This function splits the data into training and testing so the mice are different in both groups
        :param test_size: test size
        :param seed: for repeatability of same mice
        :return:
        X_train: HRV features or RR for training
        Y_train: Labels for training
        x_test: HRV features or RR for testing
        y_test: Labels for testing
        feat_names: original HRV features calculated in PhysioZoo
        """
        if self.loaded:
            id = np.unique(self.ds_output['id'])
            np.random.seed(seed=seed)
            rand_idx = np.random.randint(id.shape[0], size=(int(np.ceil(test_size*id.shape[0])),))
            age_30 = [559, 727, 730, 751]  # tags of age 30
            idx_30_list = [np.where(id == val) for val in age_30]  # indices in id of 30 months old mice
            if np.any(np.in1d(idx_30_list, rand_idx)):  # make sure that at least one of the 30 months old mice is in the test set
                pass
            else:
                rand_idx[-1] = idx_30_list[np.random.randint(len(idx_30_list))][0].item()

            test_mice = list(id[rand_idx])
            p = list(id)
            train_mice = [i for j, i in enumerate(p) if j not in rand_idx]
            if self.input_type == 'features':
                self.X_train = self.ds_input[[x in train_mice for x in self.ds_output['id']]]
                self.x_test = self.ds_input[[x in test_mice for x in self.ds_output['id']]]
            else:
                self.X_train = self.ds_input[:, [x in train_mice for x in self.ds_output['id']]]
                self.x_test = self.ds_input[:, [x in test_mice for x in self.ds_output['id']]]
            self.Y_train = self.ds_output[[x in train_mice for x in self.ds_output['id']]]
            self.y_test = self.ds_output[[x in test_mice for x in self.ds_output['id']]]
            self.splitted = True
            return self.X_train, self.x_test, self.Y_train, self.y_test, self.feat_names
        else:
            raise Exception('Data have to be loaded first!')

    def clean(self, thresh=[0.05, 0.15], feat2drop=['RR', 'NN'], **kwargs):
        """
        This function removes selected features and nan values in one of three methods depending on the fraction of nans.
        If fraction of nans is lower then first threshold argument then an imputation is performed according
        to **kwargs. Imputation in test set is performed according to training statistics. If it is in between the
        thresholds then random sampling is made. Again both according to training data. If it is higher then the feature
        is dropped in both training and testing.
        :param thresh: 2 element list. Values are in an ascending order and can range between 0 and 1
        :param feat2drop: features to drop without relating its nan values or anything else
        :param kwargs: used for SimpleImputer, e.g. changing strategy to median instead of default mean
        :return: training and testing set without nans.
        """
        if self.splitted:
            if self.input_type == 'features':
                self.X_train.drop(columns=feat2drop, inplace=True)
                self.x_test.drop(columns=feat2drop, inplace=True)
                self.feat_names = [ele for ele in self.feat_names if ele not in feat2drop]

                tr = self.X_train.isna().sum()/self.X_train.shape[0]
                ts = self.x_test.isna().sum()/self.x_test.shape[0]

                imputer = SimpleImputer(**kwargs)
                imputer.fit(self.X_train)
                feat_impute_train = self.X_train.columns[(tr > 0) & (tr < thresh[0])]
                feat_impute_test = self.x_test.columns[(ts > 0) & (ts < thresh[0])]
                nan_rows_train = self.X_train.loc[:, feat_impute_train].isna()
                nan_rows_test = self.x_test.loc[:, feat_impute_test].isna()
                for feat in nan_rows_train:
                    self.X_train.loc[nan_rows_train[feat], feat] = imputer.statistics_[list(self.X_train.columns)
                        .index(feat)]
                    print('{}/{} elements were imputed for the feature {} from the training set in the training set'
                          .format(1*nan_rows_train[feat].sum(), len(nan_rows_train[feat]), feat))
                for feat in nan_rows_test:
                    self.x_test.loc[nan_rows_test[feat], feat] = imputer.statistics_[list(self.x_test.columns)
                        .index(feat)]
                    print('{}/{} elements were imputed for the feature {} from the training set in testing set'
                          .format(1*nan_rows_test[feat].sum(), len(nan_rows_test[feat]), feat))

                feat_cdf_train = self.X_train.columns[(tr > thresh[0]) & (tr < thresh[1])]
                feat_cdf_test = self.x_test.columns[(ts > thresh[0]) & (ts < thresh[1])]
                for val in feat_cdf_train:
                    n = self.X_train[[val]].isna().sum().item()
                    samples = self.X_train[[val]].dropna().sample(n=n).values
                    self.X_train.loc[self.X_train.loc[:, val].isna(), val] = samples.squeeze()
                    print('{}/{} elements were randomly sampled for the feature {} from the training set and '
                          'implemented in the training set'.format(n, self.X_train[[val]].shape[0], val))
                for val in feat_cdf_test:
                    n = self.x_test[[val]].isna().sum().item()
                    samples = self.X_train[[val]].dropna().sample(n=n).values # take samples from X_train!!
                    self.x_test.loc[self.x_test.loc[:, val].isna(), val] = samples.squeeze()
                    print('{}/{} elements were randomly sampled for the feature {} from the training set and '
                          'implemented in the testing set'.format(n, self.x_test[[val]].shape[0], val))

                total_drop = set()
                feat_drop_train = self.X_train.columns[tr > thresh[1]]
                for val in feat_drop_train:
                    total_drop.add(val)
                feat_drop_test = self.x_test.columns[ts > thresh[1]]
                for val in feat_drop_test:
                    total_drop.add(val)
                self.X_train.drop(columns=total_drop, inplace=True)
                self.x_test.drop(columns=total_drop, inplace=True)
                self.feat_names = [ele for ele in self.feat_names if ele not in total_drop]
                print('The features {} were dropped as selected by the user or by default argument.'.format(feat2drop))
                if len(total_drop) == 0:
                    print('No additional features were dropped. These are the features of the dataset: {}'
                          .format(self.feat_names))
                else:
                    print('The features {} were dropped. Features left are {}'.format(total_drop, self.feat_names))

        else:
            raise Exception('Data have to be splitted first!')

    def choose_specific_xy(self, label_dict={'k_id': 'all', 'med': 'all', 'age': 'all', 'win_num': 'all', 'seed': 42}):
        """
        This function should output specific training and testing sets according to user requests upon number of mice,
        type of medicine and age.
        :param label_dict: k_id should be an index (up to the number of mice in X_train) to choose random mice from
        X_train. med should be a list of medications (represented as integers). age should be a list of  ages.
        Both med and age have to be list even if they consist only one element.
        :param seed: seed for repeatability of random choice of mice
        :return: training and testing sets.
        """

        random.seed(label_dict['seed'])
        id = list(set(self.Y_train['id']))
        if self.input_type == 'features':
            if label_dict['k_id'] == 'all':
                self.x_train_specific = self.X_train
                self.y_train_specific = self.Y_train
            else:
                id = random.choices(id, k=label_dict['k_id'])
                self.x_train_specific = self.X_train.loc[[x in id for x in self.Y_train['id']], :]
                self.y_train_specific = self.Y_train[[x in id for x in self.Y_train['id']]]

            self.x_test_specific = self.x_test
            self.y_test_specific = self.y_test

            if label_dict['med'] == 'all':
                pass
            else:
                self.x_train_specific = self.x_train_specific.loc[[x in label_dict['med'] for x in self.y_train_specific['med']], :]
                self.y_train_specific = self.y_train_specific[[x in label_dict['med'] for x in self.y_train_specific['med']]]
                self.x_test_specific = self.x_test_specific.loc[[x in label_dict['med'] for x in self.y_test_specific['med']], :]
                self.y_test_specific = self.y_test_specific[[x in label_dict['med'] for x in self.y_test_specific['med']]]

            if label_dict['age'] == 'all':
                pass
            else:
                self.x_train_specific = self.x_train_specific.loc[[x in label_dict['age'] for x in self.y_train_specific['age']], :]
                self.y_train_specific = self.y_train_specific[[x in label_dict['age'] for x in self.y_train_specific['age']]]
                self.x_test_specific = self.x_test_specific.loc[[x in label_dict['age'] for x in self.y_test_specific['age']], :]
                self.y_test_specific = self.y_test_specific[[x in label_dict['age'] for x in self.y_test_specific['age']]]

            if label_dict['win_num'] == 'all':
                pass
            else:
                if isinstance(label_dict['win_num'], int):
                    self.x_train_specific = self.x_train_specific.loc[[x in np.arange(1, label_dict['win_num'] + 1) for x in self.y_train_specific['win_num']], :]
                    self.y_train_specific = self.y_train_specific[[x in np.arange(1, label_dict['win_num'] + 1) for x in self.y_train_specific['win_num']]]
                    self.x_test_specific = self.x_test_specific.loc[[x in np.arange(1, label_dict['win_num'] + 1) for x in self.y_test_specific['win_num']], :]
                    self.y_test_specific = self.y_test_specific[[x in np.arange(1, label_dict['win_num'] + 1) for x in self.y_test_specific['win_num']]]
                else:
                    self.x_train_specific = self.x_train_specific.loc[[x in label_dict['win_num'] for x in self.y_train_specific['win_num']], :]
                    self.y_train_specific = self.y_train_specific[[x in label_dict['win_num'] for x in self.y_train_specific['win_num']]]
                    self.x_test_specific = self.x_test_specific.loc[[x in label_dict['win_num'] for x in self.y_test_specific['win_num']], :]
                    self.y_test_specific = self.y_test_specific[[x in label_dict['win_num'] for x in self.y_test_specific['win_num']]]


            # x_val_specific = self.X_train.loc[[x in win_num for x in self.Y_train['win_num']], :]
            # y_val_specific = self.Y_train[[x in win_num for x in self.Y_train['win_num']]]
            # x_train_specific = self.X_train.loc[[x not in win_num for x in self.Y_train['win_num']], :]
            # y_train_specific = self.Y_train[[x not in win_num for x in self.Y_train['win_num']]]
            #
            # x_train_specific = x_train_specific.values
            # x_val_specific = x_val_specific.values

        else:  # i.e. if it is a raw signal
            if label_dict['k_id'] == 'all':
                self.x_train_specific = self.X_train
                self.y_train_specific = self.Y_train
            else:
                id = random.choices(id, k=label_dict['k_id'])
                self.x_train_specific = self.X_train[:, [x in id for x in self.Y_train['id']]]
                self.y_train_specific = self.Y_train[[x in id for x in self.Y_train['id']]]

            self.x_test_specific = self.x_test
            self.y_test_specific = self.y_test

            if label_dict['med'] == 'all':
                pass
            else:
                self.x_train_specific = self.x_train_specific[:, [x in label_dict['med'] for x in self.y_train_specific['med']]]
                self.y_train_specific = self.y_train_specific[[x in label_dict['med'] for x in self.y_train_specific['med']]]
                self.x_test_specific = self.x_test_specific[:, [x in label_dict['med'] for x in self.y_test_specific['med']]]
                self.y_test_specific = self.y_test_specific[[x in label_dict['med'] for x in self.y_test_specific['med']]]
            if label_dict['age'] == 'all':
                pass
            else:
                self.x_train_specific = self.x_train_specific[:, [x in label_dict['age'] for x in self.y_train_specific['age']]]
                self.y_train_specific = self.y_train_specific[[x in label_dict['age'] for x in self.y_train_specific['age']]]
                self.x_test_specific = self.x_test_specific[:, [x in label_dict['age'] for x in self.y_test_specific['age']]]
                self.y_test_specific = self.y_test_specific[[x in label_dict['age'] for x in self.y_test_specific['age']]]

            if label_dict['win_num'] == 'all':
                pass
            else:
                if isinstance(label_dict['win_num'], int):
                    self.x_train_specific = self.x_train_specific[:, [x in np.arange(1, label_dict['win_num'] + 1) for x in self.y_train_specific['win_num']]]
                    self.y_train_specific = self.y_train_specific[[x in np.arange(1, label_dict['win_num'] + 1) for x in self.y_train_specific['win_num']]]
                    self.x_test_specific = self.x_test_specific[:, [x in np.arange(1, label_dict['win_num'] + 1) for x in self.y_test_specific['win_num']]]
                    self.y_test_specific = self.y_test_specific[[x in np.arange(1, label_dict['win_num'] + 1) for x in self.y_test_specific['win_num']]]
                else:
                    self.x_train_specific = self.x_train_specific[:, [x in label_dict['win_num'] for x in self.y_train_specific['win_num']]]
                    self.y_train_specific = self.y_train_specific[[x in label_dict['win_num'] for x in self.y_train_specific['win_num']]]
                    self.x_test_specific = self.x_test_specific[:, [x in label_dict['win_num'] for x in self.y_test_specific['win_num']]]
                    self.y_test_specific = self.y_test_specific[[x in label_dict['win_num'] for x in self.y_test_specific['win_num']]]
        # return self.x_train_specific, self.y_train_specific, self.x_test_specific, self.y_test_specific
    ###################################################
        # id = list(set(self.Y_train['id']))
        # id_ovrfit = id[0:]
        # self.x4overfit = self.X_train[:, [x in id_ovrfit for x in self.Y_train['id']]]
        # self.y4overfit = self.Y_train[[x in id_ovrfit for x in self.Y_train['id']]]
        # self.x4overfit = self.x4overfit[:, [x==True for x in self.y4overfit['med']==0]]
        # self.y4overfit = self.y4overfit[[x==True for x in self.y4overfit['med']==0]]
        # self.x4overfit = self.x4overfit[:, [x==True for x in self.y4overfit['age']==6]]
        # self.y4overfit = self.y4overfit[[x==True for x in self.y4overfit['age']==6]]
        # return self.x4overfit, self.y4overfit
    #######################################################


class TorchDataset(Dataset):
    def __init__(self, data, label='id', mode='train'):
        self.data = data
        self.hash_id = None
        if mode == 'train':
            if label == 'id':
                self.hash_id = {tag: idx for idx, tag in enumerate(set(data.y_train_specific[label]))}
            self.y = torch.tensor(data.y_train_specific[label].values, dtype=torch.int64)
            self.X = torch.tensor(data.x_train_specific, dtype=torch.float, requires_grad=True)
        if mode == 'test':
            if label == 'id':
                self.hash_id = {tag: idx for idx, tag in enumerate(set(data.y_test_specific['id']))}
            self.y = torch.tensor(data.y_test_specific[label].values, dtype=torch.int64)
            self.X = torch.tensor(data.x_test_specific, dtype=torch.float, requires_grad=True)
        if mode == 'train_val':
            X, y = data
            if label == 'id':
                self.hash_id = {tag: idx for idx, tag in enumerate(set(y.detach().numpy()))}
            self.X = torch.transpose(X, 0, 1)
            self.y = y
        if mode == 'test_val':
            X, y = data
            if label == 'id':
                self.hash_id = {tag: idx for idx, tag in enumerate(set(y.detach().numpy()))}
            self.X = torch.transpose(X, 0, 1)
            self.y = y
        # if ds_name == 'ovrfit':
        #     self.hash_id = {tag : idx for idx, tag in enumerate(set(data.y4overfit['id']))}
        #     self.y = torch.tensor(data.y4overfit['id'].values, dtype=torch.int64)
        #     self.X = torch.tensor(data.x4overfit, dtype=torch.float, requires_grad=True)
        # else:
        #     self.hash_id = {tag : idx for idx, tag in enumerate(set(data.Y_train['id']))}
        #     self.y = torch.tensor(data.Y_train['id'].values, dtype=torch.int64)
        #     self.X = torch.tensor(data.X_train, dtype=torch.float, requires_grad=True)

    def __len__(self):
        return self.X.shape[1]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        X = torch.transpose(self.X, 0, 1)
        # X = torch.unsqueeze(X, 1)
        x = X[idx:idx+1, :]
        tag = self.y[idx]
        if self.hash_id is not None:
            y = torch.tensor(self.hash_id[tag.item()], dtype=torch.int64)  # convert labels ranging fro 0 to C-1
        else:
            y = torch.tensor(tag, dtype=torch.int64)
        sample = (x, y)
        return sample
    # def scale(self, method='standard'):
    #     """
    #     This function scales the data. Notice that we probably shouldn't use it here because in the training part we
    #     scale training and validation apart.
    #     :param method: Either standard or minmax
    #     :return: Scaled features or RR
    #     """
    #     if method == 'standard':
    #         scaler = StandardScaler()
    #     elif method == 'minmax':
    #         scaler = MinMaxScaler()
    #     else:
    #         Exception('Undefined scaling method!')
    #     if self.input_type == 'features':
    #         self.X_train = pd.DataFrame(scaler.fit_transform(self.X_train), columns=self.feat_names)
    #         self.x_test = pd.DataFrame(scaler.transform(self.x_test), columns=self.feat_names)
    #     else:
    #         self.X_train = scaler.fit_transform(self.X_train)
    #         self.x_test = scaler.transform(self.x_test)
    #     return self.X_train, self.x_test

