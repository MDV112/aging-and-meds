from data_loader import Dataloader
from run import *
import torch
from data_exploration import set_data, set_exp_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from dim_reduction import DimRed
from sklearn.decomposition import PCA
from sklearn.svm import SVC


def plot_feat_sep(x, y, win_num, feat=[1, 15, 16], num_signals=None, flag_plot=None):
    """
    This function can plot either 4 subplots of 3 features and one separation or one separation.
    Normally x = data.x_train_specific, y = data.y_train_specific
    :return:
    """
    age = []

    if flag_plot is None:
        fig, axes = plt.subplots(1, len(feat)+1)
    else:
        fig, axes = plt.subplots(1, 1)
    if num_signals is not None:
        res = np.zeros(num_signals)
        res_std = np.zeros((win_num, len(res)))
    else:
        res = np.zeros(int(len(x)//win_num))
        res_std = np.zeros((win_num, len(res)))
    for i in range(len(res)):
        temp = np.zeros((len(feat), win_num))
        for j, val in enumerate(feat):
            curr_feat = x.iloc[win_num*i:win_num*(i+1), val]
            if flag_plot is None:
                axes[j].plot(np.arange(win_num), curr_feat)
                # axes[j].plot(np.arange(win_num), np.ones_like(np.arange(win_num))*curr_feat.mean())
            temp[j, :] = np.expand_dims(curr_feat.values, axis=0)
        age.append(y.iloc[win_num*i, 0])
        res[i] = temp.std(axis=0).mean()
        res_std[:, i] = temp.std(axis=0)
    # res_plot = np.exp(res/10)
    res_plot = res
    ones_vec = np.ones_like(np.arange(win_num))
    for i in range(len(res)):
        if flag_plot is None:
            axes[len(feat)].plot(np.arange(win_num), np.log(ones_vec*res_plot[i]))
            # axes[len(feat)].scatter(np.arange(win_num), res_std[:, i])
        else:
            axes.plot(np.arange(win_num), ones_vec*res_plot[i])
            plt.text(9 + 1.2*np.random.uniform(), 0.996*res_plot[i], str(age[i]), fontsize=6)
            plt.subplots_adjust(right=0.8)
    sort_array = np.diff(np.sort(res_plot))
    me = np.median(sort_array)
    dist = (1/len(sort_array))*np.dot((sort_array - me), (sort_array - me))
    plt.title("Med = {}, Age = {}, separation = {:.2f}".format(med, curr_age, dist))
    plt.show()


# def sigcomb(res, num_plots):
#     for i in range(len(res)):
#         temp = np.zeros((3, win_num))
#         for j, val in enumerate(feat):
#             curr_feat = x.iloc[win_num*i:win_num*(i+1), val]
#             if num_plots == 4:
#                 axes[j].plot(np.arange(win_num), curr_feat)
#             temp[j, :] = np.expand_dims(curr_feat.values, axis=0)
#         age.append(y.iloc[win_num*i, 0])
#         res[i] = temp.std(axis=0).mean()


def plot_sep_by_sep(x, y, win_num, win_train=6, feat=[1, 15, 16], num_signals=None):
    age = []
    fig, axes = plt.subplots(1, 2)
    if num_signals is None:
        res = np.zeros((2, int(len(x)//win_num)))
    else:
        res = np.zeros((2, num_signals))

    x_train = x.loc[y['win_num'] <= win_train, :]
    y_train = y.loc[y['win_num'] <= win_train, :]
    x_test = x.loc[y['win_num'] > win_train, :]
    y_test = y.loc[y['win_num'] > win_train, :]
    for i in range(res.shape[1]):
        train_temp = np.zeros((len(feat), win_train))
        test_temp = np.zeros((len(feat), win_num-win_train))
        for j, val in enumerate(feat):
            train_feat = x_train.iloc[win_train*i:win_train*(i+1), val]
            train_temp[j, :] = np.expand_dims(train_feat.values, axis=0)
            test_feat = x_test.iloc[(win_num-win_train)*i:(win_num-win_train)*(i+1), val]
            test_temp[j, :] = np.expand_dims(test_feat.values, axis=0)
        age.append(y.iloc[win_num*i, 0])
        res[0, i] = train_temp.std(axis=0).mean()
        res[1, i] = test_temp.std(axis=0).mean()
        # res[1, i] = 5.34 + res[1, i]/1.11
    # res_plot = np.exp(res/10)
    res_plot = res
    ones_vec = np.ones_like(np.arange(win_num))
    for i in range(res.shape[1]):
            axes[0].plot(np.arange(win_num), ones_vec*res_plot[0, i])
            axes[0].text(0 - 1.2*np.random.uniform(), 0.996*res_plot[0, i], str(age[i]), fontsize=6)
            plt.subplots_adjust(left=0.05)
            axes[1].plot(np.arange(win_num), ones_vec*res_plot[1, i])
            axes[1].text(9 + 1.2*np.random.uniform(), 0.996*res_plot[1, i], str(age[i]), fontsize=6)
            plt.subplots_adjust(right=0.8)
    # sort_array = np.diff(np.sort(res_plot))
    # me = np.median(sort_array)
    # dist = (1/len(sort_array))*np.dot((sort_array - me), (sort_array - me))
    # plt.title("Med = {}, Age = {}, separation = {:.2f}".format(med, curr_age, dist))
    plt.show()
    plt.scatter(res[0, :], res[1, :])
    z = np.polyfit(res[0, :], res[1, :], 1)
    p = np.poly1d(z)
    t = np.linspace(res[0, :].min(), res[0, :].max(), num=res.shape[1])
    plt.plot(res[0, :], p(res[0, :]))
    plt.xlabel(str(x_train.columns[feat]) + " training transformation")
    plt.ylabel(str(x_train.columns[feat]) + " testing transformation")
    mse = (1/len(res[1, :]))*(np.dot(p(res[0, :]) - res[1, :], p(res[0, :]) - res[1, :]))
    if z[1] > 0:
        plt.title("y = {:.2f}x + {:.2f}, MSE = {:.2f}, trainset = {}".format(z[0], z[1], mse, win_train))
    else:
        plt.title("y = {:.2f}x {:.2f}, MSE = {:.2f}, trainset = {}".format(z[0], z[1], mse, win_train))
    plt.show()
    return z, mse


if __name__ == '__main__':
    exp_flag = 0
    mlp = 1
    T = 3  # trajectories
    tpm = 250  # , trj_per_mouse_per_age
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    data = Dataloader(dataset_name=[1, 0])  # input_type='raw', dataset_name=250)
    data.load()
    t = np.unique(data.ds_output['id'])
    max_age = {}
    for p in t:
        temp = data.ds_output.loc[data.ds_output['id'] == p, 'age']
        max_age[p] = temp.max()
    data.split(test_size=0.2)
    drop_indices = [0, 1, 2, 4, 9, 10, 12, 13, 15, 17]
    feat_list = data.feat_names
    feat2drop = [feat_list[i] for i in drop_indices]
    data.clean(feat2drop=feat2drop)
    win_num = 10//data.dataset_name[0]  # num of windows in 10 min
    win_num = 9

    # med = 0
    # curr_age = 6
    for curr_age in range(6, 33, 3):
        for med in [0, 1]:
            label_dict = {'k_id': 'all', 'med': [med], 'age': [curr_age], 'win_num': win_num, 'seed': 42}
            data.choose_specific_xy(label_dict=label_dict)
            x = data.x_train_specific
            y = data.y_train_specific
            z_mse = []
            # for win in [1, 2, 3, 4, 5, 6, 7, 8]:
            win_num=5
            z, mse = plot_sep_by_sep(x, y, win_num, win_train=5, feat=[1, 15, 16], num_signals=10)
            z_mse.append((z, mse))
            plot_feat_sep(x, y, win_num, feat=[1, 6, 15], num_signals=10)
            a=1
            # age = []
            #
            # fig, axes = plt.subplots(1, 1)
            # # for i in range(int(len(x)//win_num)):
            # feat = [1, 15, 16]  # feat = [1, 15, 16] is true when drop_indices = [0, 1, 2, 4, 9, 10, 12, 13, 15, 17]
            # res = np.zeros(int(len(x)//win_num))
            # # res = np.zeros(10)
            # for i in range(len(res)):
            #     temp = np.zeros((3, win_num))
            #     for j, val in enumerate(feat):
            #         curr_feat = x.iloc[win_num*i:win_num*(i+1), val]
            #         # axes[j].plot(np.arange(win_num), curr_feat)
            #         temp[j, :] = np.expand_dims(curr_feat.values, axis=0)
            #     age.append(y.iloc[win_num*i, 0])
            #     res[i] = temp.std(axis=0).mean()
            # res_plot = np.exp(res/10)
            # ones_vec = np.ones_like(np.arange(win_num))
            # for i in range(len(res)):
            #     # axes[3].plot(np.arange(win_num), np.log(ones_vec*res_plot[i]))
            #     axes.plot(np.arange(win_num), ones_vec*res_plot[i])
            #     plt.text(9 + 1.2*np.random.uniform(), 0.996*res_plot[i], str(age[i]), fontsize=6)
            #     plt.subplots_adjust(right=0.8)
            # sort_array = np.diff(np.sort(res_plot))
            # me = np.median(sort_array)
            # dist = (1/len(sort_array))*np.dot((sort_array - me), (sort_array - me))
            # plt.title("Med = {}, Age = {}, separation = {:.2f}".format(med, curr_age, dist))
            # plt.show()

    med = [1]
    curr_age = [12]
    label_dict = {'k_id': 'all', 'med': med, 'age': curr_age, 'win_num': win_num, 'seed': 42}
    lbl = 'id'
    data.choose_specific_xy(label_dict=label_dict)
    x = data.x_train_specific
    new_feat = x[['pNN5', 'PSS', 'PAS']].std(axis=1)
    x['new_feat'] = new_feat
    x.drop(columns=['pNN5', 'PSS', 'PAS'], inplace=True)

    # new_feat2 = x[['SampEn', 'PIP']].std(axis=1)
    # x['new_feat2'] = new_feat2
    # x.drop(columns=['SampEn', 'PIP'], inplace=True)
    y = data.y_train_specific
    if lbl == 'id':
        x_train = x.loc[y['win_num'] <= 5, :]
        y_train = y.loc[y['win_num'] <= 5, :]
        # x_test = x_train
        # y_test = y_train
        x_test = x.loc[y['win_num'] > 5, :]
        x_test['new_feat'] = 5.34 + x_test['new_feat']/1.11
        y_test = y.loc[y['win_num'] > 5, :]
    else:
        x_test = data.x_test_specific
        y_test = data.y_test_specific


    d = {tag: idx for idx, tag in enumerate(set(y[lbl]))}
    y_tag_train = y_train[lbl].values
    y_tag_test = y_test[lbl].values
    y_tag_train_new = np.zeros_like(y_tag_train)
    y_tag_test_new = np.zeros_like(y_tag_test)
    for i in range(len(y_tag_train)):
        y_tag_train_new[i] = d[y_tag_train[i]]
    for i in range(len(y_tag_test)):
        y_tag_test_new[i] = d[y_tag_test[i]]
    scaler = StandardScaler()
    x = scaler.fit_transform(x_train)
    y = y_tag_train_new
    clf = LogisticRegression(solver='saga', penalty='l1', max_iter=10000)
    # clf = SVC()
    # clf = RandomForestClassifier()
    clf.fit(x, y)
    w_6 = clf.coef_
    print(clf.score(scaler.transform(x_test), y_tag_test_new))
    print(np.argsort(np.abs(w_6).sum(axis=0)))

    # med = [1]
    # curr_age = [15, 18, 21, 24]
    # curr_age = 9

    label_dict = {'k_id': 'all', 'med': med, 'age': curr_age, 'win_num': win_num, 'seed': 42}
    data.choose_specific_xy(label_dict=label_dict)
    dim_red = DimRed(data, n_components=2)
    x = data.x_train_specific
    y = data.y_train_specific
    d = {tag: idx for idx, tag in enumerate(set(y[lbl]))}
    y_tag = y[lbl].values
    for i in range(len(y_tag)):
        y_tag[i] = d[y_tag[i]]
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    y = y_tag
    clf = LogisticRegression(solver='saga', penalty='l1', max_iter=10000)
    clf.fit(x, y)
    w_9 = clf.coef_
    print(clf.score(x, y))
    print(np.argsort(np.abs(w_6).sum(axis=0)))
    print(np.argsort(np.abs(w_9).sum(axis=0)))
    a=1
    # if exp_flag:
    #     x_control_train, x_abk_train = set_exp_data(data.x_train_specific, data.y_train_specific)
    #     x_control_test, x_abk_test = set_exp_data(data.x_test_specific, data.y_test_specific)
    # else:
    #     x_control_train, x_abk_train, yy_control_train, yy_abk_train = set_data(data.x_train_specific, data.y_train_specific)
    #     x_control_test, x_abk_test, yy_control_test, yy_abk_test = set_data(data.x_test_specific, data.y_test_specific)
    # a=1