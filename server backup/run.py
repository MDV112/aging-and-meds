from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import plot_confusion_matrix, roc_auc_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv1D, MaxPool1D, Flatten, BatchNormalization
from tensorflow.keras import utils
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import SGD
import sys
# todo: use Pipeline. Add here plots such as confusion matrix and radar plots.


class Run:
    def __init__(self, data, model, label='med'):
        self.label = label
        self.best_estimator = None
        self.best_params = None
        self.data = data
        self.model = model

    @staticmethod
    def calc_stat(y, y_pred, y_pred_proba):
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
        print('Sensitivity is {:.2f}. \nSpecificity is {:.2f}. \nPPV is {:.2f}. \nNPV is {:.2f}. \nAccuracy is {:.2f}.'
              ' \nF1 is {:.2f}. '.format(se, sp, ppv, npv, acc, f1))
        if type(y_pred_proba) == int:
            print('AUROC could not be calculated')
        else:
            print('AUROC is {:.2f}'.format(roc_auc_score(y, y_pred_proba[:, 1])))

    def train(self, param_grid={}, method='standard', pca=None, refit='roc_auc', **skf_kwargs):
        X_train = self.data.X_train
        y = self.data.Y_train[self.label]
        lbl_ratio = 100 * y.value_counts(normalize=True)
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
                               scoring=['accuracy', 'f1', 'precision', 'recall', 'roc_auc'], cv=skf,
                               refit=refit, verbose=3, return_train_score=True)
            clf.fit(X_train, y)
            self.best_estimator = clf.best_estimator_
            self.best_params = clf.best_params_
        else:
            tf.keras.backend.clear_session()
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.2
            tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
            X_train = X_train.T
            X_train = np.expand_dims(X_train, axis=2)
            # X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            model = KerasClassifier(build_fn=self.model, verbose=1, epochs=100)
            #batch_size = [2000]
            #n_filters_start = [32]
            # dropout = [0.1]
            #param_grid = dict(batch_size=batch_size, n_filters_start=n_filters_start)
            #grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3)
            #grid_result = grid.fit(X_train, y)
            self.best_estimator = model
            self.best_estimator.fit(X_train,y)
            #print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


    def infer(self):
        if self.best_estimator is None:
            raise Exception('No model was fitted!')
        else:
            y = self.data.y_test[self.label]
            x_test = self.data.x_test.T
            x_test = np.expand_dims(x_test, axis=2)
            y_pred = self.best_estimator.predict(x_test)
            if hasattr(self.model, 'probability'):
                if self.model.probability is False:
                    y_pred_proba = 0
                else:
                    y_pred_proba = self.best_estimator.predict_proba(x_test)
            else:
                y_pred_proba = self.best_estimator.predict_proba(x_test)
            self.calc_stat(y, y_pred, y_pred_proba)



