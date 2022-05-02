from data_loader import Dataloader
# from run import Run
from models import Models
import json
from run import *
import tensorflow as tf
from sklearn.metrics import f1_score, make_scorer
from data_loader import TorchDataset
# from deep import AE
import torch.nn as nn
import torch
import matplotlib.cm as cm
import torch
# from deep import CNN
# from data_loader import TorchDataset
import time
from models import TruncatedCNN
from models import AdverserailCNN
import seaborn as sns
import sklearn
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
import pydiffmap.diffusion_map as diffmap
import pydiffmap.kernel as diffker
from sklearn.manifold import TSNE
import random


class DimRed():
    def __init__(self, data, method='t-SNE', n_components=3, plot=True, save=False):  # ,  **kwargs
        self.data = data
        self.method = method
        self.n_components = n_components
        self.plot = plot
        self.obj = None
        self.save_file = save
        # self.kwargs == kwargs

    def apply(self, label_dict, fname=None, fit_mode=1, **kwargs):
        # fit mode = 1 means fit_transform(x_train) and show it
        # fit_mode = 2 means fit_transform(x_train) and show transform(x_test)
        # fit mode = 3 means fit_transform(x_test) and show it
        x_train_specific = self.data.x_train_specific
        x_val_specific = self.data.x_test_specific
        y_train = self.data.y_train_specific
        y_test = self.data.y_test_specific
        if hasattr(x_train_specific, 'values'):
            x_train = x_train_specific.values
            x_test = x_val_specific.values
        else:
            x_train = np.transpose(x_train_specific)
            x_test = np.transpose(x_val_specific)
        if self.method == 'Diffusion map':
            # kwargs can include t
            self.obj = diffmap.DiffusionMap.from_sklearn(n_evecs=self.n_components, **kwargs)
            trans_mat = self.obj.construct_Lmat(x_train)
            Z_train = self.obj.fit_transform(x_train)
            Z_test = self.obj.transform(x_test)
            # U_fitted = self.obj.fit(x_train)
            # Psi = U_fitted.evecs
            # lmbda = np.diag(U_fitted.evals)
            # t = 1
            # Psi_t = Psi@(lmbda**t)
        elif self.method == 'PCA' or not(isinstance(self.n_components, int)):
            self.obj = KernelPCA(n_components=self.n_components, **kwargs)
            Z_train = self.obj.fit_transform(x_train)
            Z_test = self.obj.transform(x_test)
        elif self.method == 't-SNE':
            if fit_mode == 2:
                raise Exception('TSNE does not have transform function, only fit_transform')
            self.obj = TSNE(n_components=self.n_components, **kwargs) #perplexity=0.1)
            Z_train = self.obj.fit_transform(x_train)
            Z_test = self.obj.fit_transform(x_test)
        else:
            raise Exception('This type of dimensionality reduction is not implemented here. Check typing')
        if self.plot:
            if fit_mode == 1:
                Z = Z_train
                y = y_train
            elif fit_mode == 2:
                Z = Z_test
                y = y_test
            else:
                Z = self.obj.fit_transform(x_test)
                y = y_test
            obj_vis = self.visualize(Z, y, fname, label_dict)
            return obj_vis
        else:
            return Z_train, Z_test

    def visualize(self, Z, y, fname, label_dict, plot_labels=True):
        if fname == None:
            self.save_file = False
        if self.n_components == 3:
            id = y['id']
            id_unique = np.unique(id.values)
            rand_color = np.arange(len(id_unique))
            hashing = {tag: col for tag, col in zip(id_unique, rand_color)}
            color_id = [hashing[x] for x in id]
            fig = plt.figure(figsize=(25, 25))
            w = [[0, 1], [0, 2], [1, 2]]
            for j, rel_ax in enumerate(w):
                ax = fig.add_subplot(1, 3, j+1)
                ax.scatter(Z[:, rel_ax[0]], Z[:, rel_ax[1]], c=color_id, cmap='jet_r')
                if plot_labels:
                    for i, txt in enumerate(id.values):
                        ax.annotate(str(txt), (Z[i, rel_ax[0]], Z[i, rel_ax[1]]), xycoords='data')
                # ax.text(Z_val[i, 0], Z_val[i, 1], Z_val[i, 2], str(txt))
                ax.set_xlim([Z[:, rel_ax[0]].min(), Z[:, rel_ax[0]].max()])
                ax.set_ylim([Z[:, rel_ax[1]].min(), Z[:, rel_ax[1]].max()])
                # ax.set_zlim(Z_val[:, 2].min(), Z_val[:, 2].max())
                ax.set_xlabel('U_' + str(rel_ax[0]) + '[N.U]')
                ax.set_ylabel('U_' + str(rel_ax[1]) + '[N.U]')
                ax.set_title('2D ' + self.method)
                if not(self.save_file):
                    plt.show()
                    return self.obj
                else:
                    return self.save(fname, label_dict)
        else:  # apply 2d visualization for both n_components=2 or not an integer
            id = y['id']
            id_unique = np.unique(id.values)
            rand_color = np.arange(len(id_unique))
            hashing = {tag: col for tag, col in zip(id_unique, rand_color)}
            color_id = [hashing[x] for x in id]
            fig = plt.figure(figsize=(25, 25))
            ax = fig.add_subplot(111)
            ax.scatter(Z[:, 0], Z[:, 1], c=color_id, cmap="jet_r", linewidths=6)
            if plot_labels:
                for i, txt in enumerate(id.values):
                    ax.annotate(str(txt), (Z[i, 0], Z[i, 1]), xycoords='data', **dict(fontsize='xx-large'))
            # ax.text(Z_val[i, 0], Z_val[i, 1], Z_val[i, 2], str(txt))
            ax.set_xlim([Z[:, 0].min(), Z[:, 0].max()])
            ax.set_ylim([Z[:, 1].min(), Z[:, 1].max()])
            ax.tick_params(axis='both', labelsize=30)
            ax.set_xlabel('$U_1 [N.U]$', fontsize=30, fontweight='bold')
            ax.set_ylabel('$U_2 [N.U]$', fontsize=30, fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # ax.set_title('2D ' + self.method)
            if not(self.save_file):
                plt.show()
                return self.obj
            else:
                return self.save(fname, label_dict)

    def save(self, fname, label_dict):
        if hasattr(self.obj, 'get_params'):
            params = self.obj.get_params()
        else:
            params = 'params are not available'
        if (fname + '.txt') in os.listdir():
            raise Exception('Change file name')
        else:
            img_path = os.path.join(os.getcwd(), fname + '.jpg')
            txt_path = os.path.join(os.getcwd(), fname + '.txt')
        plt.savefig(img_path)
        plt.show()
        txt_list = [params, self.data.feat_names, label_dict]
        with open(txt_path, 'w') as file:
            # file.write(json.dumps(txt_list, indent=4))
            for data in txt_list:
                file.write(json.dumps(data, indent=4))
        return self.obj



# if __name__ == '__main__':
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     learning_rate = 1e-4
#     # batch_size = 3500  # 1000 works well for 5 ids with both med and no med
#     data_loader = Dataloader(input_type='raw', dataset_name=250)
#     data_loader.load()
#     data_loader.split()
#     label_dict = {'k_id': 20, 'med': [0], 'age': [6], 'win_num': 10, 'seed': 1}
#     if data_loader.input_type == 'features':
#         feat_list = data_loader.feat_names
#         good_indices = [0, 1, 4, 9, 10, 12, 13, 15, 17]
#         feat2drop = [feat_list[i] for i in good_indices]
#         data_loader.clean(feat2drop=feat2drop)
#     else:
#         data_loader.clean()
#     # data_loader.choose_specific_xy(label_dict=label_dict)
#     # x_train_specific = data_loader.x_train_specific
#     # x_val_specific = data_loader.x_test_specific
#     # y_train_specific = data_loader.y_train_specific
#     # y_val_specific = data_loader.y_test_specific
#     # if hasattr(x_train_specific, 'values'):
#     #     x_train_4_dim_red = x_train_specific.values
#     #     x_val_4_dim_red = x_val_specific.values
#     # else:
#     #     x_train_4_dim_red = np.transpose(x_train_specific)
#     #     x_val_4_dim_red = np.transpose(x_val_specific)
#
#     dim_red = DimRed(n_components=2)
#     d = dict(perplexity=100, init='pca')
#     # d = dict(kernel='rbf', gamma=0.0005)
#     obj_vis = dim_red.apply(**d)
#     if hasattr(obj_vis, 'get_params'):
#         params = obj_vis.get_params()
#     else:
#         params = 'params are not available'
#     # data_loader.feat_names
#     # # label_dict
#     fname = 'file15'
#     if (fname + '.txt') in os.listdir():
#         raise Exception('Change file name')
#     else:
#         img_path = os.path.join(os.getcwd(), fname + '.jpg')
#         txt_path = os.path.join(os.getcwd(), fname + '.txt')
#     plt.savefig(img_path)
#     plt.show()
#     txt_list = [params, data_loader.feat_names, label_dict]
#     with open(txt_path, 'w') as file:
#         # file.write(json.dumps(txt_list, indent=4))
#         for data in txt_list:
#             file.write(json.dumps(data, indent=4))
#     a=1
#     # win_num = [1]
#     # med = [0]
#     # x_val_specific = data_loader.X_train.loc[[x in win_num for x in data_loader.Y_train['win_num']], :]
#     # y_val_specific = data_loader.Y_train[[x in win_num for x in data_loader.Y_train['win_num']]]
#     # x_val_specific = x_val_specific.loc[[x in med for x in y_val_specific['med']], :]
#     # y_val_specific = y_val_specific[[x in med for x in y_val_specific['med']]]
#     # x_train_specific = data_loader.X_train.loc[[x not in win_num for x in data_loader.Y_train['win_num']], :]
#     # y_train_specific = data_loader.Y_train[[x not in win_num for x in data_loader.Y_train['win_num']]]
#     # x_train_specific = x_train_specific.loc[[x in med for x in y_train_specific['med']], :]
#     # y_train_specific = y_train_specific[[x in med for x in y_train_specific['med']]]
#     # x_train_specific = x_train_specific.values
#     # x_val_specific = x_val_specific.values
#     # # else:
#     #
#     # data_loader.choose_specific_xy_rr(label_dict=label_dict)
#     # x_train_specific = data_loader.x_train_specific
#     # x_val_specific = data_loader.x_test_specific
#     # y_train_specific = data_loader.y_train_specific
#     # y_val_specific = data_loader.y_test_specific
#
# #---------------------------Implement your code here:------------------------
#
#     n_components = 2
#     ker = diffker
#     neighbor_params = {'n_jobs': -1, 'algorithm': 'kd_tree'}
#     diff_map = diffmap.DiffusionMap.from_sklearn(n_evecs=3, alpha=8, epsilon=0.001)
#     # trans_mat = diff_map.construct_Lmat(np.transpose(x_train_specific))#, k=32, neighbor_params=neighbor_params)
#     # Z_train = diff_map.fit_transform(np.transpose(x_train_specific))
#     # Z_val = diff_map.transform(np.transpose(x_val_specific))
#     # U_fitted = diff_map.fit(np.transpose(x_train_specific))
#     # Psi = U_fitted.evecs
#     # lmbda = np.diag(U_fitted.evals)
#     # t = 1
#     # Psi_t = Psi@(lmbda**t)
#
#
#     # CONSIDER USING ONLY FIT_TRASNFORM WITHOUT FIT
#     Z_train = TSNE(n_components=2, perplexity=20).fit_transform(np.transpose(x_train_specific))
#     # Z_val = TSNE(n_components=2, perplexity=1).fit_transform(np.transpose(x_val_specific))
#     Z_val = Z_train
#     id = y_train_specific['id']
#     fig = plt.figure(figsize=(18, 18))
#     ax = fig.add_subplot(111)  # , projection='3d')
#     ax.scatter(Z_val[:, 0], Z_val[:, 1], c=id.values) #, Z_val[:, 2])
#
#     # for i, txt in enumerate(id.values):
#     #     ax.annotate(str(txt), (Z_val[i, 0], Z_val[i, 1]), xycoords='data')
#         # ax.text(Z_val[i, 0], Z_val[i, 1], Z_val[i, 2], str(txt))
#     ax.set_xlim([Z_val[:, 0].min(), Z_val[:, 0].max()])
#     ax.set_ylim([Z_val[:, 1].min(), Z_val[:, 1].max()])
#     # ax.set_zlim(Z_val[:, 2].min(), Z_val[:, 2].max())
#     ax.set_xlabel('$U_1 [N.U]$')
#     ax.set_ylabel('$U_2 [N.U]$')
#     ax.set_title('2D t-SNE')
#     plt.show()
#
#
#     # pca = PCA(n_components=n_components, whiten=True)
#     pca = KernelPCA(n_components=n_components, kernel='rbf', gamma=0.001)
#
#     # apply PCA transformation
#     Z_train = pca.fit_transform(np.transpose(x_train_specific))
#     Z_val = pca.transform(np.transpose(x_val_specific))
#
#     fig = plt.figure(figsize=(18, 18))
#     ax = fig.add_subplot(111) #, projection='3d')
#     ax.scatter(Z_val[:, 0], Z_val[:, 1]) #, Z_val[:, 2])
#     id = y_val_specific['id']
#     for i, txt in enumerate(id.values):
#         ax.annotate(str(txt), (Z_val[i, 0], Z_val[i, 1]), xycoords='data')
#     # ax.annotate(id, (Z_val[:, 0], Z_val[:, 1]))
#     # ax.scatter(X_pca[y==1, 0], X_pca[y==1, 1], color='r')
#     # ax.legend(('B', 'M'))
#     # ax.plot([0], [0], "ko")
#     # ax.arrow(0, 0, 0, 1, head_width=0.05, length_includes_head=True, head_length=0.1, fc='k', ec='k')
#     # ax.arrow(0, 0, 1, 0, head_width=0.05, length_includes_head=True, head_length=0.1, fc='k', ec='k')
#     plt.xlim([Z_val[:, 0].min(), Z_val[:, 0].max()])
#     plt.ylim([Z_val[:, 1].min(), Z_val[:, 1].max()])
#     # ax.set_zlim(Z_val[:, 2].min(), Z_val[:, 2].max())
#     ax.set_xlabel('$U_1$')
#     ax.set_ylabel('$U_2$')
#     ax.set_title('2D PCA')
#     plt.show()
#
#     yy_test = data_loader.Y_train.loc[:, data_loader.Y_train.loc['win_num'] == 1]
#     yy_test = yy_test.loc[:, yy_test.loc['med'] == 0]
#     a=1