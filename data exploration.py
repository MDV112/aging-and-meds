from data_loader import Dataloader
# from run import Run
from models import Models
from torch.utils.data import Dataset
from run import *
import tensorflow as tf
from sklearn.metrics import f1_score, make_scorer
from data_loader import TorchDataset
# from deep import AE
import torch.nn as nn
import torch
import torch
from dim_reduction import DimRed
from deep_models import DeepModels
# from deep import CNN
# from data_loader import TorchDataset
import seaborn as sns
from deep_models import ContrastiveLoss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler
import pickle

# device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# data = Dataloader(dataset_name=[1, 0])  # input_type='raw', dataset_name=250)
# data.load()
# data.split(test_size=0.05)
# drop_indices = [0, 1, 4, 9, 10, 12, 13, 15, 17]
# feat_list = data.feat_names
# feat2drop = [feat_list[i] for i in drop_indices]
# data.clean(feat2drop=feat2drop)
# win_num = 10//data.dataset_name[0]  # num of windows in 10 min
# label_dict = {'k_id': 'all', 'med': [0, 1], 'age': 'all', 'win_num': win_num, 'seed': 42}
# # dim_red_dict = dict(perplexity=10.5, init='pca')
# data.choose_specific_xy(label_dict=label_dict)


def cv(data):
    yy = np.unique(data.y_train_specific['id'].values)
    cv_mat = np.zeros((len(yy), data.x_train_specific.shape[1]), dtype=np.float64)
    for i, tag in enumerate(yy):
        mouse = data.x_train_specific.loc[data.y_train_specific['id'] == tag, :].values
        cv_mat[i, :] = mouse.std(axis=0)/np.abs(mouse.mean(axis=0))
    cv_pd = pd.DataFrame(cv_mat, columns=data.x_train_specific.columns)
    cv_pd.index = list(yy)
    return cv_pd
# cv_pd = cv(data)


def set_data(dx, dy):
    Age = np.unique(dy['age'].values)
    yy = np.unique(dy['id'].values)
    control = dx.loc[dy['med'] == 0, :]
    y_control = dy.loc[dy['med'] == 0, :]
    abk = dx.loc[dy['med'] == 1, :]
    y_abk = dy.loc[dy['med'] == 1, :]
    s1 = pd.Series([])
    s2 = pd.Series([])
    x_control = pd.Series([])
    yy_control = pd.Series([])
    x_abk = pd.Series([])
    yy_abk = pd.Series([])
    tag_age = []
    tag_age_x = []
    tag_age_y = []
    for idx_tag, tag in enumerate(yy):
        curr_control = control.loc[dy['id'] == tag, :].copy()
        curr_y_control = y_control.loc[dy['id'] == tag, :].copy()
        curr_abk = abk.loc[dy['id'] == tag, :].copy()
        curr_y_abk = y_abk.loc[dy['id'] == tag, :].copy()
        for age in Age:
            mean_control = curr_control.loc[curr_y_control['age'] == age, :].mean(skipna=True)
            mean_abk = curr_abk.loc[curr_y_abk['age'] == age, :].mean(skipna=True)
            if mean_abk.isnull().all() or age == 30:  # mean_abk and mean_control get nan row at the same locations
                if age == 30 and not(mean_abk.isnull().all()):
                    tag_age.append(str(tag) + '_' + str(age))
                    s1 = pd.concat([s1, mean_control], axis=1, ignore_index=True)
                    s2 = pd.concat([s2, mean_abk], axis=1, ignore_index=True)
                tag_age_x += tag_age[0:-1]
                tag_age_y += tag_age[1:]
                temp_control = s1.T.iloc[1:]
                X_control = temp_control.iloc[0:-1]
                Y_control = temp_control.iloc[1:]
                x_control = pd.concat([x_control, X_control], axis=0, ignore_index=False)
                if idx_tag == 0:
                    x_control = x_control.iloc[:, 1:]
                x_control.index = tag_age_x
                yy_control = pd.concat([yy_control, Y_control], axis=0, ignore_index=False)
                if idx_tag == 0:
                    yy_control = yy_control.iloc[:, 1:]
                yy_control.index = tag_age_y
                s1 = pd.Series([])
                temp_abk = s2.T.iloc[1:]
                X_abk = temp_abk.iloc[0:-1]
                Y_abk = temp_abk.iloc[1:]
                x_abk = pd.concat([x_abk, X_abk], axis=0, ignore_index=False)
                if idx_tag == 0:
                    x_abk = x_abk.iloc[:, 1:]
                x_abk.index = tag_age_x
                yy_abk = pd.concat([yy_abk, Y_abk], axis=0, ignore_index=False)
                if idx_tag == 0:
                    yy_abk = yy_abk.iloc[:, 1:]
                yy_abk.index = tag_age_y
                s2 = pd.Series([])
                tag_age = []
                break
            #     break inner loop
            s1 = pd.concat([s1, mean_control], axis=1, ignore_index=True)
            s2 = pd.concat([s2, mean_abk], axis=1, ignore_index=True)
            tag_age.append(str(tag) + '_' + str(age))
    return x_control, x_abk, yy_control, yy_abk


# x_control_train, x_abk_train, yy_control_train, yy_abk_train = set_data(data.x_train_specific, data.y_train_specific)
# x_control_test, x_abk_test, yy_control_test, yy_abk_test = set_data(data.x_test_specific, data.y_test_specific)


def add_age_col(df):
    tag = []
    age = []
    for tag_name in df.index:
        tag.append(tag_name[0:3])
        age.append(tag_name[4:])
    df.index = tag
    df['Age'] = age

#
# for df in (x_control_train, x_abk_train, yy_control_train, yy_abk_train):
#     add_age_col(df)
#
# for df in (x_control_test, x_abk_test, yy_control_test, yy_abk_test):
#     add_age_col(df)




# yy_train = np.unique(x_control_train.index)
# yy_test = np.unique(x_control_test.index)
#
# train_tuple = (x_control_train, x_abk_train, yy_control_train, yy_abk_train)
# test_tuple = (x_control_test, x_abk_test, yy_control_test, yy_abk_test)


def reshape_df_by_id(df_x_control, df_x_abk, df_yy_control, df_yy_abk, yy, T=3):
    x_c = pd.Series([])
    x_a = pd.Series([])
    y_c = pd.Series([])
    y_a = pd.Series([])
    for tag in yy:
        curr_x_control = df_x_control.loc[df_x_control.index == str(tag)]
        curr_x_abk = df_x_abk.loc[df_x_abk.index == str(tag)]
        curr_y_control = df_yy_control.loc[df_yy_control.index == str(tag)]
        curr_y_abk = df_yy_abk.loc[df_yy_abk.index == str(tag)]
        if len(curr_y_abk) < T:
            pass
        else:
            for i in range(len(curr_x_control.index)-(T-1)):
                x_c = pd.concat([x_c, curr_x_control.iloc[i:i+T]], axis=0, ignore_index=False)
                x_a = pd.concat([x_a, curr_x_abk.iloc[i:i+T]], axis=0, ignore_index=False)
                y_c = pd.concat([y_c, curr_y_control.iloc[i:i+T]], axis=0, ignore_index=False)
                y_a = pd.concat([y_a, curr_y_abk.iloc[i:i+T]], axis=0, ignore_index=False)
    x_c = x_c.iloc[:, 1:]
    x_a = x_a.iloc[:, 1:]
    y_c = y_c.iloc[:, 1:]
    y_a = y_a.iloc[:, 1:]
    return x_c, x_a, y_c, y_a



# tr_x_c, tr_x_a, tr_y_c, tr_y_a = reshape_df_by_id(train_x_c, train_x_a, train_y_c, train_y_a, yy_train)
# tr_x_c, tr_x_a, tr_y_c, tr_y_a = reshape_df_by_id(train_tuple[0], train_tuple[1], train_tuple[2], train_tuple[3], yy_train)
# ts_x_c, ts_x_a, ts_y_c, ts_y_a = reshape_df_by_id(test_tuple[0], test_tuple[1], test_tuple[2], test_tuple[3], yy_test)

# scaler = MinMaxScaler()

#todo: Should normaliztion be applied after or before reshape_df_by_id?

#todo: normalization of test according to full training


def scale_data(train_list, test_list, scaler=MinMaxScaler()):
    scaled_train = []
    scaled_test = []
    for tr, ts in zip(train_list, test_list):
        scaled_train.append(scaler.fit_transform(tr))
        scaled_test.append(scaler.transform(ts))
    return scaled_train, scaled_test

# scaled_train, scaled_test = scale_data([tr_x_c, tr_x_a, tr_y_c, tr_y_a], [ts_x_c, ts_x_a, ts_y_c, ts_y_a])


#todo: Uncomment the following


def split_train(train_list, test_size=0.05):
    train_inds, val_inds = next(GroupShuffleSplit(test_size=test_size, n_splits=2, random_state=7)
                                .split(x_control_train, groups=train_list[0].index))
    train_tuple = tuple(element.iloc[train_inds] for element in train_list)
    val_tuple = tuple(element.iloc[val_inds] for element in train_list)
    return train_tuple, val_tuple


# train_tuple, val_tuple = split_train([x_control_train, x_abk_train, yy_control_train, yy_abk_train])
# yy_val = np.unique(val_tuple[0].index)
# tr_x_c, tr_x_a, tr_y_c, tr_y_a = reshape_df_by_id(train_tuple[0], train_tuple[1], train_tuple[2], train_tuple[3], yy_train)
# v_x_c, v_x_a, v_y_c, v_y_a = reshape_df_by_id(val_tuple[0], val_tuple[1], val_tuple[2], val_tuple[3], yy_val)

#todo: normalization of validation according to training

#todo: convert to torch
def convert2torch(scaled_train, scaled_test):
    X_train = torch.stack([torch.from_numpy(scaled_train[0]), torch.from_numpy(scaled_train[1])])
    Y_train = torch.stack([torch.from_numpy(scaled_train[2]), torch.from_numpy(scaled_train[3])])
    X_test = torch.stack([torch.from_numpy(scaled_test[0]), torch.from_numpy(scaled_test[1])])
    Y_test = torch.stack([torch.from_numpy(scaled_test[2]), torch.from_numpy(scaled_test[3])])
    return X_train, Y_train, X_test, Y_test

# X_train, Y_train, X_test, Y_test = convert2torch(scaled_train, scaled_test)
# train_X_c = torch.from_numpy(tr_x_c)
# train_X_a = torch.from_numpy(tr_x_a)
# train_Y_c = torch.from_numpy(tr_y_c)
# train_Y_a = torch.from_numpy(tr_y_a)
#
# X_train = torch.stack([train_X_c, train_X_a])
# Y_train = torch.stack([train_Y_c, train_Y_a])
#
# val_X_c = torch.from_numpy(v_x_c)
# val_X_a = torch.from_numpy(v_x_a)
# val_Y_c = torch.from_numpy(v_y_c)
# val_Y_a = torch.from_numpy(v_y_a)
#
# X_val = torch.stack([val_X_c, val_X_a])
# Y_val = torch.stack([val_Y_c, val_Y_a])

if __name__ == '__main__':

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    data = Dataloader(dataset_name=[1, 0])  # input_type='raw', dataset_name=250)
    data.load()
    data.split(test_size=0.1)
    drop_indices = [0, 1]  # , 4, 9, 10, 12, 13, 15, 17]
    feat_list = data.feat_names
    feat2drop = [feat_list[i] for i in drop_indices]
    data.clean(feat2drop=feat2drop)
    win_num = 10//data.dataset_name[0]  # num of windows in 10 min
    label_dict = {'k_id': 'all', 'med': [0, 1], 'age': 'all', 'win_num': win_num, 'seed': 42}
    # dim_red_dict = dict(perplexity=10.5, init='pca')
    data.choose_specific_xy(label_dict=label_dict)

    x_control_train, x_abk_train, yy_control_train, yy_abk_train = set_data(data.x_train_specific, data.y_train_specific)
    x_control_test, x_abk_test, yy_control_test, yy_abk_test = set_data(data.x_test_specific, data.y_test_specific)

    for df in (x_control_train, x_abk_train, yy_control_train, yy_abk_train):
        add_age_col(df)
    for df in (x_control_test, x_abk_test, yy_control_test, yy_abk_test):
        add_age_col(df)

    # todo: 726 at the age of 27 was already dropped completely in data_loader.load() since there were no peaks detected
    # in control

    yy_train = np.unique(x_control_train.index)
    yy_test = np.unique(x_control_test.index)

    train_tuple = (x_control_train, x_abk_train, yy_control_train, yy_abk_train)
    test_tuple = (x_control_test, x_abk_test, yy_control_test, yy_abk_test)

    tr_x_c, tr_x_a, tr_y_c, tr_y_a = reshape_df_by_id(train_tuple[0], train_tuple[1], train_tuple[2], train_tuple[3], yy_train, T=2)
    ts_x_c, ts_x_a, ts_y_c, ts_y_a = reshape_df_by_id(test_tuple[0], test_tuple[1], test_tuple[2], test_tuple[3], yy_test, T=2)

    scaler = MinMaxScaler()

    scaled_train, scaled_test = scale_data([tr_x_c, tr_x_a, tr_y_c, tr_y_a], [ts_x_c, ts_x_a, ts_y_c, ts_y_a])

    X_train, Y_train, X_test, Y_test = convert2torch(scaled_train, scaled_test)

    with open('new_train_test.pkl', 'wb') as f:
        pickle.dump([X_train, X_test, Y_train, Y_test], f)

    train_tuple, val_tuple = split_train([x_control_train, x_abk_train, yy_control_train, yy_abk_train])
    yy_val = np.unique(val_tuple[0].index)
    tr_x_c, tr_x_a, tr_y_c, tr_y_a = reshape_df_by_id(train_tuple[0], train_tuple[1], train_tuple[2], train_tuple[3], yy_train, T=2)
    v_x_c, v_x_a, v_y_c, v_y_a = reshape_df_by_id(val_tuple[0], val_tuple[1], val_tuple[2], val_tuple[3], yy_val, T=2)

    scaled_train, scaled_val = scale_data([tr_x_c, tr_x_a, tr_y_c, tr_y_a], [v_x_c, v_x_a, v_y_c, v_y_a])

    X_train, Y_train, X_val, Y_val = convert2torch(scaled_train, scaled_val)

    with open('new_train_val.pkl', 'wb') as f:
        pickle.dump([X_train, X_val, Y_train, Y_val], f)


    a=1

# with open('train_val.pkl', 'rb') as handle:
#     b = pickle.load(handle)





