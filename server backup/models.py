from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv1D, MaxPool1D, Flatten, BatchNormalization
from tensorflow.keras import utils
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import SGD


class Models:
    def __init__(self, model_name='rfc', **kwargs):
        self.model = None
        self.model_name = model_name
        self.kwargs = kwargs

    def set_model(self):
        if self.model_name == 'log_reg':
            self.model = LogisticRegression(**self.kwargs)
        elif self.model_name == 'svm':
            self.model = SVC(**self.kwargs)
        elif self.model_name == 'rfc':
            self.model = RandomForestClassifier(**self.kwargs)
        else:
            self.model = self.create_model
            # todo: deep approach

            pass
        return self.model

    @staticmethod
    def create_model(window_size=60, len_sub_window=10, n_filters_start=64, n_hidden_start=512, dropout=0.5, lr=0.01, momentum=0):
        model = Sequential()
        model.add(Conv1D(n_filters_start, len_sub_window, activation='relu', input_shape=(250, 1)))
        model.add(BatchNormalization())
        model.add(Conv1D(2 * n_filters_start, len_sub_window, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPool1D())
        model.add(Conv1D(4 * n_filters_start, len_sub_window, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
        model.add(Flatten())
        model.add(Dense(n_hidden_start, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(int(n_hidden_start / 2), activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(int(n_hidden_start / 4), activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
        model.add(Dense(1, activation='sigmoid'))
        optimizer = SGD(lr=lr, momentum=momentum)
        model.compile(optimizer=optimizer, metrics=['accuracy'], loss='binary_crossentropy')
        return model
