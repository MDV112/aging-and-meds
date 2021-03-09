import pandas as pd
import numpy as np
import h5py
import scipy.io as sio
from pathlib import Path
import os
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer


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
                    Exception('Adequate mat file for features names was not found in ' + str(Path(self.data_path)))
            self.ds_input = pd.DataFrame(ds_input.T, columns=feat_names)
            self.ds_output = pd.DataFrame(ds_output.T, columns=['id', 'age', 'med', 'win_num'])
            self.loaded = True
            return self.ds_input, self.ds_output, self.feat_names
        else:
            if type(self.dataset_name) != int:
                Exception('dataset_name has to be an integer representing number of beats')
            fname = Path.joinpath(Path(self.data_path), 'rr.h5')
            f = h5py.File(fname, "r")
            ds_keys = []
            f.visit(lambda key: ds_keys.append(key) if type(f[key]) is h5py._hl.dataset.Dataset else None)
            matching = [s for s in ds_keys if str(self.dataset_name) in s]
            self.ds_input = f[matching[0]].value
            ds_output = f[matching[1]].value.astype(int)
            self.ds_output = pd.DataFrame(ds_output.T, columns=['id', 'age', 'med', 'win_num'])
            self.loaded = True
            return self.ds_input, self.ds_output

    def split(self, seed=42, test_size=0.2):
        """
        This function splits the data into training and testing so the mice are different in both groups
        :param test_size: test size
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
            Exception('Data have to be loaded first!')

    def clean(self, thresh=[0.05, 0.15], feat2drop=['RR','NN'], **kwargs):
        """
        This function removes selected features and nan values in one of three methods depending on the fraction of nans.
        If fraction of nans is lower then first threshold argument then an imputation is performed according
        to **kwargs. Imputation in test set is performed according to training statistics. If it is in between the
        thresholds then random sampling is made. Again both according to training data. If it is higher then the feature
        is dropped in both training and testing.
        :param thresh: 2 element list. Values are in an ascending order and can range between 0 and 1
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
            Exception('Data have to be splitted first!')

    # def scale(self, method='standard'):
    #     """
    #     This funcion scales the data. Notice that we probably shouldn't use it here because in the training part we
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

