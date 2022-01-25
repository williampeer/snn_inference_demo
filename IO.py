import torch
import os
from datetime import datetime

import data_util

PATH = './saved/'
PLOT_PATH = 'plot_data/'
JUST_DATA_PATH = 'data/'
fname_ext = '.pt'


def makedir_if_not_exists(path):
    path_parts = path.split('/')
    aggr_path = ''
    for i in range(len(path_parts)):  # needs to create one dir at a time.
        aggr_path += path_parts[i] + '/'
        if not os.path.exists(aggr_path):
            os.mkdir(aggr_path)


def save_entire_model(model, uuid, fname='test_model'):
    makedir_if_not_exists(PATH + uuid)

    torch.save(model, PATH+uuid+'/'+fname+fname_ext)


def save_model_params(model, fname='test_model_params'):
    full_path = data_util.prefix + data_util.target_data_path + data_util.matlab_export
    makedir_if_not_exists(full_path)

    torch.save(model.state_dict(), full_path+'/'+fname+fname_ext)


def save_poisson_rates(rates, uuid, fname='default_poisson_rates'):
    makedir_if_not_exists(PATH + uuid)

    torch.save(rates, PATH+uuid+'/'+fname+fname_ext)


def save(model, loss, uuid, fname='test_exp_dict'):
    makedir_if_not_exists(PATH + uuid)

    torch.save({
        'model': model,
        'loss': loss
    }, PATH+uuid+'/'+fname+fname_ext)


def save_plot_data(data, uuid, plot_fn='unknown', fname=False):
    makedir_if_not_exists(PATH+PLOT_PATH+uuid)

    if not fname:
        fname = plot_fn + dt_descriptor()
    torch.save({
        'plot_data': data,
        'plot_fn': plot_fn
    }, PATH+PLOT_PATH+uuid+'/'+fname+fname_ext)


def save_data(data, uuid, description='default', fname=False):
    makedir_if_not_exists(PATH+JUST_DATA_PATH+uuid)

    if not fname:
        fname = 'saved_data_' + dt_descriptor()
    torch.save({
        'data': data,
        'description': description
    }, PATH+JUST_DATA_PATH+uuid+'/'+fname+fname_ext)


def import_data(uuid, fname):
    return torch.load(PATH+JUST_DATA_PATH+uuid+'/'+fname+fname_ext)


def dt_descriptor():
    return datetime.utcnow().strftime('%m-%d_%H-%M-%S-%f')[:-3]
