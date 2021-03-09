from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


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
            # todo: deep approach
            pass
        return self.model
