from data_loader import Dataloader
# from run import Run
from models import Models
from run import *
# import tensorflow as tf
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
import pickle


if __name__ == '__main__':
    # with open('y_label.pkl', 'rb') as f:
    #     yy = pickle.load(f)
    # with open('x_y.pkl', 'rb') as f:
    #     tr_x_c, tr_x_a, tr_y_c, tr_y_a, ts_x_c, ts_x_a, ts_y_c, ts_y_a, max_age = pickle.load(f)
    red_dim = False  # apply dimensionality reduction
    vis = False
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    data = Dataloader(input_type='raw', dataset_name=250)
    data.load()
    data.split()
    drop_indices = [0, 1, 4, 9, 10, 12, 13, 15, 17]
    label_dict = {'k_id': 'all', 'med': 'all', 'age': 'all', 'win_num': 'all', 'seed': 0}
    dim_red_dict = dict(perplexity=10.5, init='pca')
    n_components = 2
    n_nets = 1
    if data.input_type == 'features':
        feat_list = data.feat_names
        feat2drop = [feat_list[i] for i in drop_indices]
        data.clean(feat2drop=feat2drop)
        data.choose_specific_xy(label_dict=label_dict)
        if red_dim:
            dim_red = DimRed(data, n_components=n_components, plot=vis, save=True)
            # dim_red_dict = dict(kernel='rbf', gamma=0.0005)
            obj_vis = dim_red.apply(label_dict, fname='file27', **dim_red_dict)
        model_dict = {}
        model_obj = Models(data, model_name='rfc', **model_dict)
        if model_obj.model_name not in ['log_reg', 'svm', 'rfc', 'xgb']:
            raise Exception('This type of model does not exist in Models class!')
        model = model_obj.set_model()
        # model = model.to(device)
        # criterion{“gini”, “entropy”}
        param_grid = {'criterion': ['entropy', 'gini'], 'max_leaf_nodes': [2, 10, None], 'warm_start': [False, True],
                      'max_features': ['auto', None], 'min_samples_leaf': [1, 0.1, 0.05]}
        # param_grid = {}
        param_grid = {'model__' + k: v for k, v in param_grid.items()}  # adding 'model__' to keys for GridSearchCV
        # param_grid = {}
        runner = Run(data, model, label='med')
        f1 = make_scorer(f1_score, average='macro')
        runner.train(scoring='roc_auc', refit=True, param_grid=param_grid)
        runner.infer()
    else:  # i.e. raw data
        data.clean()
        data.choose_specific_xy(label_dict=label_dict)
        if red_dim:  # todo: check red_dim stucks the run
            dim_red = DimRed(data, n_components=n_components, plot=vis)
            # dim_red_dict = dict(kernel='rbf', gamma=0.0005)
            obj_vis = dim_red.apply(**dim_red_dict)
        with open('rr_data.pkl', 'wb') as f:
            pickle.dump(data, f)
        mod = DeepModels(data, 1, device)
        mod.choose_model('CNN', label='id', mode='val', **dict(num_chann=[20, 30, 50, 20], ker_size=10, drop_out=0.2,
                                                               num_hidden=[40, 30, 10]))
        mod.set_model(lr=1e-3, optimizer=torch.optim.SGD, **dict(momentum=0.9))
        loss = nn.CrossEntropyLoss()
        mod.train(loss, epochs=300)
        mod.train('constructive_cosine', epochs=300, n_nets=2)
        # mod.load_state_dict(torch.load('./checkpoints/cifar_cnn_ckpt.pth'))
        # mod.choose_model('TruncCNNtest', mode='val')  #, **dict(num_chann=[20, 30, 50, 20], ker_size=10, drop_out=0.2, num_hidden=[40, 30, 3]))
        # # The above works perfectly for 3 mice, Adagrad with lr of 1e-2 and 1500 epochs
        # mod.set_model(lr=1e-1, optimizer=torch.optim.SGD, **dict(momentum=0.9))  # , **dict(momentum=0.9))
        # # for param in mod.model.parameters():
        # #     print(param.data)
        # # loss_train, loss_val, acc_train, acc_val = mod.train('constructive_cosine', epochs=300)
        # mod.train('constructive_cosine', epochs=300)
        # # loss_train, loss_val, acc_train, acc_val = mod.train(nn.BCEWithLogitsLoss(), epochs=1000)
        # # for param in mod.model.parameters():
        # #     print(param.data)
        # plt.figure()
        # plt.plot(range(len(loss_train)), loss_train)
        # plt.plot(range(len(loss_val)), loss_val)
        # plt.legend(['train', 'val'])
        # plt.title('CrossEntropy loss')
        # plt.figure()
        # plt.plot(range(len(acc_train)), acc_train)
        # plt.plot(range(len(acc_val)), acc_val)
        # plt.legend(['train', 'val'])
        # plt.title('Accuracy')
        # plt.show()
        # # tr = [0.1, 0.2, 0.3, 0.45, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        # tr = np.linspace(0.01, 0.99, 15)  # for cosine
        # acc, conf = mod.infer(nn.CrossEntropyLoss())
        # tn = conf[0, 0]
        # tp = conf[1, 1]
        # fp = conf[0, 1]
        # fn = conf[1, 0]
        # mcc = ((tp * tn) - (fp * fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        # print('MCC is {:.2f}'.format(mcc))
        tr = np. linspace(0, 1, 50)  # for MI that is still under 'cosine loss' in mod.infer() argument
        FAR = []
        FRR = []
        CONF_list = []
        # acc_list = []
        for j, t in enumerate(tr):
            conf, acc = mod.infer('cosine_loss', t)
            # todo: maybe cosine inference is not good and we should combine it with MI. Plus need to think about a way to visualise it. Maybe eigen vectors of MI of embedded
            # acc_list.append(acc)
            CONF_list.append(np.zeros([2, 3]))
            CONF_list[j][0:, 0:2] = conf
            FAR.append(conf[0, 1] / (conf[0, 0] + conf[0, 1]))
            CONF_list[j][0, 2] = FAR[j].item()
            FRR.append(conf[1, 0] / (conf[1, 0] + conf[1, 1]))
            CONF_list[j][1, 2] = FRR[j].item()
        plt.plot(tr, FAR)
        plt.plot(tr, FRR)
        # plt.plot(tr, acc_list)
        # plt.plot(tr, 1 / len(np.unique(mod.ds_test.y))*np.ones_like(tr))
        plt.title('Identification accuracy of {:.2f}% and naive estimator factor of {:.2f}'.format(100*acc, acc*len(np.unique(mod.ds_test.y))))
        plt.legend(('FAR', 'FRR', 'id_acc', 'baseline'))
        plt.show()
        acc, conf = mod.infer('cosine_loss')
        conf_plot = conf / np.tile(np.sum(conf, axis=0), (conf.shape[0], 1))
        ax_heat = sns.heatmap(conf_plot, annot=True)
        ax_heat.set_xlabel('Pred')
        ax_heat.set_ylabel('True')
        ax_heat.set_title('Accuracy is {:.2f} %'.format(acc))
        plt.show()
        mod.choose_model('TruncCNN')
        # for param in mod.model.parameters():
        #     print(param.data)
        # for param in mod.model2.parameters():
        #     print(param.data)
        mod.train('cosine_loss', epochs=100)
        mod.infer(thresh=0.9)
        a=1
    # x_train_specific = data.x_train_specific
    # x_val_specific = data.x_test_specific
    # y_train_specific = data.y_train_specific
    # y_val_specific = data.y_test_specific
    # if hasattr(x_train_specific, 'values'):
    #     x_train_4_dim_red = x_train_specific.values
    #     x_val_4_dim_red = x_val_specific.values
    # else:
    #     x_train_4_dim_red = np.transpose(x_train_specific)
    #     x_val_4_dim_red = np.transpose(x_val_specific)
    # # data.choose_xy()
    # ds = TorchDataset(data, device)
    a = 1
    ###############################################################
    # ae = AE(250)
    # tf.keras.backend.clear_session()
    # # tf.executing_eagerly()
    #
    # config = tf.compat.v1.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.2
    # tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
    # X_train = data.X_train.T
    # # X_train = np.expand_dims(X_train, axis=2)
    # # X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    # ae.compile(optimizer='adam', loss='mae')
    # x_test = data.x_test
    # x_test = x_test.T
    # # x_test = np.expand_dims(x_test, axis=2)
    # history = ae.fit(X_train, X_train,
    #                          validation_data=(x_test, x_test),
    #                          epochs=10,
    #                          shuffle=True)
    # plt.plot(history.history["loss"], label="Training Loss")
    # plt.plot(history.history["val_loss"], label="Validation Loss")
    # plt.legend()
    # plt.show()
    #########################################################################
    # trainloader = torch.utils.data.DataLoader(TorchDataset(data))
    # model = CharCNN(len(labels))

    #########################################################################
    # d = {'max_depth': 3}

    a=1
    # todo: consider using comet_ml

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
