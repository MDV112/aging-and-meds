from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
# from sklearn.metrics import plot_confusion_matrix, roc_auc_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv1D, MaxPool1D, Flatten, BatchNormalization
from tensorflow.keras import utils
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import os
from tensorflow.keras.optimizers import SGD
import sys
# todo: use Pipeline. Add here plots such as confusion matrix and radar plots.

tf.compat.v1.enable_eager_execution()


class Run:
    def __init__(self, data, model, label='med'):
        self.label = label
        self.best_estimator = None
        self.best_params = None
        self.data = data
        self.model = model

    @staticmethod
    def calc_stat(y, y_pred, y_pred_proba):
        # if np.unique(y).shape[0] > 2:
        #     f1_micro = f1_score(y, y_pred, average='micro')
        #     f1_macro = f1_score(y, y_pred, average='macro')
        #     f1_weighted = f1_score(y, y_pred, average='weighted')
        # else:
        #     pass
        # acc = balanced_accuracy_score(y, y_pred)
        if type(y).__name__ == 'Tensor':
            y = y.cpu().detach().numpy()
            y_pred = y_pred.cpu().detach().numpy()
        calc_tn = lambda y_true, y_predicted: confusion_matrix(y_true, y_predicted)[0, 0]
        calc_fp = lambda y_true, y_predicted: confusion_matrix(y_true, y_predicted)[0, 1]
        calc_fn = lambda y_true, y_predicted: confusion_matrix(y_true, y_predicted)[1, 0]
        calc_tp = lambda y_true, y_predicted: confusion_matrix(y_true, y_predicted)[1, 1]
        tn = calc_tn(y, y_pred)
        fp = calc_fp(y, y_pred)
        fn = calc_fn(y, y_pred)
        tp = calc_tp(y, y_pred)
        se = tp/(tp+fn)
        sp = tn/(tn+fp)
        ppv = tp/(tp+fp)
        npv = tn/(tn+fn)
        acc = (tp+tn)/(tp+tn+fp+fn)
        f1 = (2*se*ppv)/(se+ppv)
        mcc = ((tp * tn) - (fp * fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        print('Sensitivity is {:.2f}. \nSpecificity is {:.2f}. \nPPV is {:.2f}. \nNPV is {:.2f}. \nAccuracy is {:.2f}.'
              ' \nF1 is {:.2f}. \n MCC is {:.2f}'.format(se, sp, ppv, npv, acc, f1, mcc))
        if type(y_pred_proba) == int:
            print('AUROC could not be calculated')
        else:
            print('AUROC is {:.2f}'.format(roc_auc_score(y, y_pred_proba[:, 1])))

    def train(self, param_grid={}, method='standard', pca=None, scoring=['accuracy', 'f1', 'precision', 'recall', 'roc_auc'], refit='roc_auc', **skf_kwargs):
        # X_train = self.data.X_train
        # y = self.data.Y_train[self.label]
        X_train = self.data.x_train_specific
        y = self.data.y_train_specific[self.label]
        print('n_training = {}'.format(y.shape[0]))
        # y.replace([6, 9, 12, 15, 18, 21, 24, 27, 30], [1, 2, 3, 4, 5, 6, 7, 8, 9], inplace=True)
        # y[y <= 21] = 0
        # y[y > 21] = 1
        lbl_ratio = 100 * y.value_counts(normalize=True)
        lbl_ratio.sort_index(inplace=True)
        for val in lbl_ratio.keys():
            print(r' %s %d constitutes %.2f%% of the training set.' % (self.label, val, lbl_ratio[val]))
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise Exception('Undefined scaling method!')
        if self.data.input_type == 'features':
            skf = StratifiedKFold(**skf_kwargs)
            if pca is None:
                pipe = Pipeline(steps=[('scale', scaler), ('model', self.model)])
            else:
                pipe = Pipeline(steps=[('scale', scaler), ('pca', pca()), ('model', self.model)])
            clf = GridSearchCV(estimator=pipe, param_grid=param_grid,
                               scoring=scoring, cv=skf,
                               refit=refit, verbose=3, return_train_score=True)
            clf.fit(X_train, y)
            print("Best: %f using %s" % (clf.best_score_, clf.best_params_))
            self.best_estimator = clf.best_estimator_
            self.best_params = clf.best_params_
        else:
            # os.environ["CUDA_VISIBLE_DEVICES"] = "3"
            # tf.keras.backend.clear_session()
            # # tf.executing_eagerly()
            #
            # config = tf.compat.v1.ConfigProto()
            # config.gpu_options.per_process_gpu_memory_fraction = 0.2
            # tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
            X_train = X_train.T
            X_train = np.expand_dims(X_train, axis=2)
            # X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            # if hasattr(self.model, 'decoder'):
            #     self.model.compile(optimizer='adam', loss='mae')
            #     x_test = self.data.x_test
            #     x_test = x_test.T
            #     x_test = np.expand_dims(x_test, axis=2)
            #     history = self.model.fit(X_train, X_train,
            #                              validation_data=(x_test, x_test),
            #                               epochs=10,
            #                               shuffle=True)
            #     plt.plot(history.history["loss"], label="Training Loss")
            #     plt.plot(history.history["val_loss"], label="Validation Loss")
            #     plt.legend()
            #     plt.show()

            model = KerasClassifier(build_fn=self.model, verbose=1, epochs=10)
            # batch_size = [2000]
            # n_filters_start = [32]
            # # dropout = [0.1]
            # param_grid = dict(batch_size=batch_size, n_filters_start=n_filters_start)
            grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3)
            grid_result = model.fit(X_train, y)
            print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    def infer(self):
        # todo: Change also test shape if data type is raw
        if self.best_estimator is None:
            raise Exception('No model was fitted!')
        else:
            # y = self.data.y_test[self.label]
            y = self.data.y_test_specific[self.label]
            print('n_testing = {}'.format(y.shape[0]))
            # y.replace([6, 9, 12, 15, 18, 21, 24, 27, 30], [1, 2, 3, 4, 5, 6, 7, 8, 9], inplace=True)
            # y[y <= 21] = 0
            # y[y > 21] = 1
            lbl_ratio = 100 * y.value_counts(normalize=True)
            lbl_ratio.sort_index(inplace=True)
            target_names = [self.label + '_' + str(k) for k in lbl_ratio.keys()]
            for val in lbl_ratio.keys():
                print(r' %s %d constitutes %.2f%% of the testing set.' % (self.label, val, lbl_ratio[val]))
            # y_pred = self.best_estimator.predict(self.data.x_test)
            y_pred = self.best_estimator.predict(self.data.x_test_specific)
            if hasattr(self.model, 'probability'):
                if self.model.probability is False:
                    y_pred_proba = 0
                else:
                    # y_pred_proba = self.best_estimator.predict_proba(self.data.x_test)
                    y_pred_proba = self.best_estimator.predict_proba(self.data.x_test_specific)
            elif lbl_ratio.shape[0] > 2:
                y_pred_proba = 0
            else:
                # y_pred_proba = self.best_estimator.predict_proba(self.data.x_test)
                y_pred_proba = self.best_estimator.predict_proba(self.data.x_test_specific)
            print(classification_report(y, y_pred))
            self.calc_stat(y, y_pred, y_pred_proba)



