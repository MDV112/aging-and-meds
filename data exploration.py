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


class TorchDataset(Dataset):

    def __init__(self, data, label='id', mode='train'):
        self.data = data
        self.X = data[0]
        max_age = data[1]
        tag = self.X.index
        y = np.array([max_age[int(x)] for x in tag])
        self.y = y - np.asarray(self.X['Age'], int)
        self.X.drop(['Age'], axis=1, inplace=True)
        self.X = torch.from_numpy(self.X.values).float()
        self.y = torch.from_numpy(self.y).float()
        # self.y = torch.tensor(self.y, requires_grad=True) #, dtype=torch.int64)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        # X = torch.tensor.transpose(self.X, 0, 1)

        # self.X = torch.unsqueeze(self.X, 1)
        x = self.X[idx:idx+1, :]
        x = x.T
        y = self.y[idx]
        # if self.hash_id is not None:
        #     y = torch.tensor(self.hash_id[tag.item()], dtype=torch.int64)  # convert labels ranging fro 0 to C-1
        # else:
        # y = torch.tensor(tag, dtype=torch.int64)
        sample = (x, y)
        return sample

class MLP(torch.nn.Module):

    def __init__(self, input_shape, num_hidden=[25, 10], dropout=0.2):
        super(MLP, self).__init__()
        self.fc1 = nn.Sequential(
            # nn.BatchNorm1d(input_shape, track_running_stats=False),
            nn.Linear(input_shape, num_hidden[0]),
            # nn.BatchNorm1d(num_hidden[0], track_running_stats=False),
            nn.ReLU(),
            # nn.ReLU(),
            # nn.Dropout(dropout)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(num_hidden[0], num_hidden[1]),
            # nn.BatchNorm1d(num_hidden[1], track_running_stats=False),
            nn.ReLU(),
            # nn.Dropout(dropout)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(num_hidden[1], 1),
            # nn.BatchNorm1d(num_hidden[1], track_running_stats=False),
            # nn.ReLU(),
            # nn.Dropout(dropout)
        )


    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        # collapse
        # out = out.view(out.size(0), -1)
        # out = self.soft_max(out) #NO NEED
        # out = self.sigmoid(out)
        out = out.squeeze()
        return out


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

def set_exp_data(dx, dy, smp2mean=2):
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
            curr_age_control = curr_control.loc[curr_y_control['age'] == age, :]
            curr_age_abk = curr_abk.loc[curr_y_abk['age'] == age, :]
            if curr_age_control.mean(skipna=True).isnull().all() or age == 30:  # mean_abk and mean_control get nan row at the same locations
                if age == 30 and not(curr_age_control.mean(skipna=True).isnull().all()):
                    for p, q in zip(curr_age_control.index, curr_age_abk.index):
                        if p + smp2mean <= curr_age_control.index.max():
                            mean_control = curr_age_control.loc[p: p + smp2mean, :].mean(skipna=True)
                            mean_abk = curr_age_abk.loc[q: q + smp2mean, :].mean(skipna=True)
                            tag_age.append(str(tag) + '_' + str(age))
                            s1 = pd.concat([s1, mean_control], axis=1, ignore_index=True)
                            s2 = pd.concat([s2, mean_abk], axis=1, ignore_index=True)
                        else:
                            break
                tag_age_x += tag_age
                # tag_age_y += tag_age[1:]
                X_control = s1.T.iloc[1:]
                # X_control = temp_control
                # Y_control = temp_control.iloc[1:]
                x_control = pd.concat([x_control, X_control], axis=0, ignore_index=False)
                if idx_tag == 0:
                    x_control = x_control.iloc[:, 1:]
                x_control.index = tag_age_x

                # yy_control = pd.concat([yy_control, Y_control], axis=0, ignore_index=False)
                # if idx_tag == 0:
                #     yy_control = yy_control.iloc[:, 1:]
                # yy_control.index = tag_age_y

                s1 = pd.Series([])
                X_abk = s2.T.iloc[1:]
                # X_abk = temp_abk.iloc[0:-1]
                # Y_abk = temp_abk.iloc[1:]
                x_abk = pd.concat([x_abk, X_abk], axis=0, ignore_index=False)
                if idx_tag == 0:
                    x_abk = x_abk.iloc[:, 1:]
                x_abk.index = tag_age_x
                # yy_abk = pd.concat([yy_abk, Y_abk], axis=0, ignore_index=False)
                # if idx_tag == 0:
                #     yy_abk = yy_abk.iloc[:, 1:]
                # yy_abk.index = tag_age_y
                s2 = pd.Series([])
                tag_age = []
                break
            else:
                for p, q in zip(curr_age_control.index, curr_age_abk.index):
                    if p + smp2mean <= curr_age_control.index.max():
                        mean_control = curr_age_control.loc[p: p + smp2mean, :].mean(skipna=True)
                        mean_abk = curr_age_abk.loc[q: q + smp2mean, :].mean(skipna=True)
                        s1 = pd.concat([s1, mean_control], axis=1, ignore_index=True)
                        s2 = pd.concat([s2, mean_abk], axis=1, ignore_index=True)
                        tag_age.append(str(tag) + '_' + str(age))
                    else:
                        break
    return x_control, x_abk


def add_age_col(df):
    tag = []
    age = []
    for tag_name in df.index:
        tag.append(tag_name[0:3])
        age.append(tag_name[4:])
    df.index = tag
    df['Age'] = age

def set_y_exp(df_x_control, df_x_abk):
    tags = np.unique(df_x_control.index.values)
    y_control = df_x_control[df_x_control['Age'].astype(int) > 6]
    y_abk = df_x_abk[df_x_abk['Age'].astype(int) > 6]
    map_max_age = {tag: df_x_control.loc[str(tag), 'Age'].astype(int).max() for tag in tags}
    s = pd.Series([])
    for tag in tags:
        s = pd.concat([s, df_x_control.loc[tag, 'Age'].astype(int) < map_max_age[tag]], axis=0)
    x_control = df_x_control[s]
    x_abk = df_x_abk[s]
    return x_control, x_abk, y_control, y_abk


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


def reshape_exp(df_tuple, T=3, trj_per_mouse_per_age=5):
    df_out = []
    for df in df_tuple:
        tags = np.unique(df.index.values)
        output = pd.Series([])
        for tag in tags:
            curr_df = df.loc[tag, :]
            for j in range(curr_df['Age'].astype(int).min(), curr_df['Age'].astype(int).max() - 3*(T-2), 3):
                trj_smple = curr_df.loc[curr_df['Age'].isin([str(j+3*idx) for idx in range(T)]), :]
                low = [np.argwhere(trj_smple['Age'] == str(j+3*idx)).min() for idx in range(T)]
                high = [np.argwhere(trj_smple['Age'] == str(j+3*idx)).max() for idx in range(T)]
                if np.any(np.array(low) >= np.array(high)):
                    print('tag number is {} at minimum age of {}, low={}, high={}'.format(tag, j, low, high))
                    break
                rand_idx = np.random.randint(low=low, high=high, size=(trj_per_mouse_per_age, T))
                for i in range(rand_idx.shape[0]):
                    output = pd.concat([output, trj_smple.iloc[rand_idx[i, :]]], axis=0, ignore_index=False)
        df_out.append(output.iloc[:, 1:])
    return df_out[0], df_out[1], df_out[2], df_out[3]

#todo: Should normaliztion be applied after or before reshape_df_by_id?

#todo: normalization of test according to full training


def scale_data(train_list, test_list, scaler=MinMaxScaler()):
    scaled_train = []
    scaled_test = []
    for tr, ts in zip(train_list, test_list):
        scaled_train.append(scaler.fit_transform(tr))
        scaled_test.append(scaler.transform(ts))
    return scaled_train, scaled_test


def split_train(train_list, test_size=0.05):
    train_inds, val_inds = next(GroupShuffleSplit(test_size=test_size, n_splits=2, random_state=7)
                                .split(x_control_train, groups=train_list[0].index))
    train_tuple = tuple(element.iloc[train_inds] for element in train_list)
    val_tuple = tuple(element.iloc[val_inds] for element in train_list)
    return train_tuple, val_tuple


def convert2torch(scaled_train, scaled_test):
    X_train = torch.stack([torch.from_numpy(scaled_train[0]), torch.from_numpy(scaled_train[1])])
    Y_train = torch.stack([torch.from_numpy(scaled_train[2]), torch.from_numpy(scaled_train[3])])
    X_test = torch.stack([torch.from_numpy(scaled_test[0]), torch.from_numpy(scaled_test[1])])
    Y_test = torch.stack([torch.from_numpy(scaled_test[2]), torch.from_numpy(scaled_test[3])])
    return X_train, Y_train, X_test, Y_test


if __name__ == '__main__':
    exp_flag = 1
    mlp = 1
    T = 3  # trajectories
    tpm = 20  # , trj_per_mouse_per_age
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    data = Dataloader(dataset_name=[1, 0])  # input_type='raw', dataset_name=250)
    data.load()
    t = np.unique(data.ds_output['id'])
    max_age = {}
    for p in t:
        temp = data.ds_output.loc[data.ds_output['id'] == p, 'age']
        max_age[p] = temp.max()
    data.split(test_size=0.2)
    drop_indices = [0, 1, 4, 9, 10, 12, 13, 15, 17]
    feat_list = data.feat_names
    feat2drop = [feat_list[i] for i in drop_indices]
    data.clean(feat2drop=feat2drop)
    win_num = 10//data.dataset_name[0]  # num of windows in 10 min
    label_dict = {'k_id': 'all', 'med': [0, 1], 'age': 'all', 'win_num': win_num, 'seed': 42}
    data.choose_specific_xy(label_dict=label_dict)
    if exp_flag:
        x_control_train, x_abk_train = set_exp_data(data.x_train_specific, data.y_train_specific)
        x_control_test, x_abk_test = set_exp_data(data.x_test_specific, data.y_test_specific)
    else:
        x_control_train, x_abk_train, yy_control_train, yy_abk_train = set_data(data.x_train_specific, data.y_train_specific)
        x_control_test, x_abk_test, yy_control_test, yy_abk_test = set_data(data.x_test_specific, data.y_test_specific)


    if exp_flag:
        for df in (x_control_train, x_abk_train):
            add_age_col(df)
        for df in (x_control_test, x_abk_test):
            add_age_col(df)
    # #todo: NOTICE THAT IF WE USE TORCHDATASER LATER ON, THEN X AGE ENDS 3 MONTHS EARLIER THAN Y
    #     if mlp:
    #         #todo: build dataset composed of both abk and control with correct labels0
    #
    #         # hyper-parameters

        x_control_train, x_abk_train, yy_control_train, yy_abk_train = set_y_exp(x_control_train, x_abk_train)
        x_control_test, x_abk_test, yy_control_test, yy_abk_test = set_y_exp(x_control_test, x_abk_test)
    else:
        for df in (x_control_train, x_abk_train, yy_control_train, yy_abk_train):
            add_age_col(df)
        for df in (x_control_test, x_abk_test, yy_control_test, yy_abk_test):
            add_age_col(df)



    yy_train = np.unique(x_control_train.index)
    yy_test = np.unique(x_control_test.index)

    train_tuple = (x_control_train, x_abk_train, yy_control_train, yy_abk_train)
    test_tuple = (x_control_test, x_abk_test, yy_control_test, yy_abk_test)

    if exp_flag:
        tr_x_c, tr_x_a, tr_y_c, tr_y_a = reshape_exp(train_tuple, T=T, trj_per_mouse_per_age=tpm)
        ts_x_c, ts_x_a, ts_y_c, ts_y_a = reshape_exp(test_tuple, T=T, trj_per_mouse_per_age=tpm)
    else:
        tr_x_c, tr_x_a, tr_y_c, tr_y_a = reshape_df_by_id(train_tuple[0], train_tuple[1], train_tuple[2], train_tuple[3], yy_train, T=T)
        ts_x_c, ts_x_a, ts_y_c, ts_y_a = reshape_df_by_id(test_tuple[0], test_tuple[1], test_tuple[2], test_tuple[3], yy_test, T=T)


    batch_size = int(np.ceil(0.1*tr_x_c.shape[0]))
    learning_rate = 5e-4
    epochs = 350
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    # tr_x_c = torch.from_numpy(tr_x_c.X.values).float()
    # ts_x_c = torch.from_numpy(ts_x_c.X.values).float()
    trainset = TorchDataset((tr_x_c, max_age))
    testset = TorchDataset((ts_x_c, max_age))
    # dataloaders - creating batches and shuffling the data
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=1)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=1)
    # device - cpu or gpu?

    # loss criterion
    criterion = nn.MSELoss()
    # build our model and send it to the device
    model = MLP(tr_x_c.shape[1]) # no need for parameters as we alredy defined them in the class
    # optimizer - SGD, Adam, RMSProp...
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(1, epochs + 1):
        model.train()  # put in training mode
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            # send them to device
            inputs = inputs.to(device)
            labels = labels.to(device)
            # forward + backward + optimize
            outputs = model(inputs)  # forward pass
            loss = criterion(outputs, labels)  # calculate the loss
            # always the same 3 steps
            optimizer.zero_grad()  # zero the parameter gradients
            loss.backward()  # backpropagation
            optimizer.step()  # update parameters
            # print statistics
            new_outputs = torch.zeros_like(outputs).to(device)
            ref = torch.arange(0, 30, 3).to(device)
            for idx, out in enumerate(outputs):
                new_outputs[idx] = ref[torch.argmin(torch.abs(ref-out))]
            new_loss = criterion(new_outputs, labels)
            running_loss += new_loss.data.item()
            # Normalizing the loss by the total number of train batches
        running_loss /= len(trainloader)
        log = "Epoch: {} | Loss: {:.4f}".format(epoch, running_loss)
        print(log)
    for i, data in enumerate(testloader, 0):
        model.eval()
        inputs, labels = data
        # send them to device
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)  # forward pass
        new_outputs = torch.zeros_like(outputs).to(device)
        ref = torch.arange(0, 30, 3).to(device)
        for idx, out in enumerate(outputs):
            new_outputs[idx] = ref[torch.argmin(torch.abs(ref-out))]
        outputs = new_outputs
        loss = criterion(outputs, labels)  # calculate the loss
        running_loss += loss.data.item()
        # Normalizing the loss by the total number of train batches
    running_loss /= len(testloader)
    log = "The loss of the batch test number {} is {:.4f}".format(i, running_loss)
    print(log)






    scaler = MinMaxScaler()

    scaled_train, scaled_test = scale_data([tr_x_c, tr_x_a, tr_y_c, tr_y_a], [ts_x_c, ts_x_a, ts_y_c, ts_y_a])

    X_train, Y_train, X_test, Y_test = convert2torch(scaled_train, scaled_test)

    with open('new_train_test.pkl', 'wb') as f:
        pickle.dump([X_train, X_test, Y_train, Y_test, T], f)

    # train_tuple, val_tuple = split_train([x_control_train, x_abk_train, yy_control_train, yy_abk_train])
    # yy_val = np.unique(val_tuple[0].index)
    # tr_x_c, tr_x_a, tr_y_c, tr_y_a = reshape_df_by_id(train_tuple[0], train_tuple[1], train_tuple[2], train_tuple[3], yy_train, T=2)
    # v_x_c, v_x_a, v_y_c, v_y_a = reshape_df_by_id(val_tuple[0], val_tuple[1], val_tuple[2], val_tuple[3], yy_val, T=2)
    #
    # scaled_train, scaled_val = scale_data([tr_x_c, tr_x_a, tr_y_c, tr_y_a], [v_x_c, v_x_a, v_y_c, v_y_a])
    #
    # X_train, Y_train, X_val, Y_val = convert2torch(scaled_train, scaled_val)
    #
    # with open('new_train_val.pkl', 'wb') as f:
    #     pickle.dump([X_train, X_val, Y_train, Y_val, T], f)


    a=1

# with open('train_val.pkl', 'rb') as handle:
#     b = pickle.load(handle)







