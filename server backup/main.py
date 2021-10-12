from data_loader import Dataloader
from run import Run
from models import Models
from run import *
if __name__ == '__main__':
    data = Dataloader(input_type='raw', dataset_name=250)
    data.load()
    data.split()
    data.clean()
    d = {'C': 0.5}
    # d = {}
    model = Models(model_name='deep').set_model()
    # param_grid = {'model__C': [0.5, 1, 100]}
    param_grid = {}
    runner = Run(data, model)
    runner.train()
    runner.infer()
    a=1
    # todo: consider using comet_ml

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
