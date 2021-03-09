from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
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
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            Exception('Undefined scaling method!')
        skf = StratifiedKFold(**skf_kwargs)
        if pca == None:
            pipe = Pipeline(steps=[('scale', scaler), ('model', self.model)])
        else:
            pipe = Pipeline(steps=[('scale', scaler), ('pca', pca()), ('model', self.model)])
        clf = GridSearchCV(estimator=pipe, param_grid=param_grid,
                           scoring=['accuracy', 'f1', 'precision', 'recall', 'roc_auc'], cv=skf,
                           refit=refit, verbose=3, return_train_score=True)
        clf.fit(X_train, y)
        self.best_estimator = clf.best_estimator_
        self.best_params = clf.best_params_

    def infer(self):
        y = self.data.y_test[self.label]
        y_pred = self.best_estimator.predict(self.data.x_test)
        if hasattr(self.model, 'probability'):
            if self.model.probability == False:
                y_pred_proba = 0
        else:
            y_pred_proba = self.best_estimator.predict_proba(self.data.x_test)
        self.calc_stat(y, y_pred, y_pred_proba)


        # plot_confusion_matrix(self.best_estimator, x_test, y, cmap=plt.cm.Blues)
        # plt.grid(False)
        # plt.show()
        a=1
        # y_pred = self.best_estimator(x_test)
        pass


