import os
import sys

import numpy as np
import torch

import IO
import data_util
from Models.GLIF import GLIF
from Models.GLIF_no_cell_types import GLIF_no_cell_types
from Models.LIF import LIF
from Models.LIF_no_cell_types import LIF_no_cell_types
from Models.microGIF import microGIF
from spike_train_matlab_export import simulate_and_save_model_spike_train

man_seed = 3
torch.manual_seed(man_seed)
np.random.seed(man_seed)

duration = 60 * 1000

load_fname = 'snn_model_target_GD_test'
model_class_lookup = { 'LIF': LIF, 'GLIF': GLIF, 'microGIF': microGIF,
                       'LIF_no_cell_types': LIF_no_cell_types, 'GLIF_no_cell_types': GLIF_no_cell_types }

GT_path = '/home/william/repos/snn_inference/Test/saved/'
GT_model_by_type = {'LIF': '12-09_11-49-59-999',
                    'GLIF': '12-09_11-12-47-541',
                    'mesoGIF': '12-09_14-56-20-319',
                    'microGIF': '12-09_14-56-17-312'}
# archive_name = 'data/'
# plot_data_path = experiments_path + 'plot_data/'
model_types = list(GT_model_by_type.keys())

for model_type_str in model_types:
    GT_euid = GT_model_by_type[model_type_str]
    tar_fname = 'snn_model_target_GD_test'
    model_name = model_type_str
    if model_type_str == 'mesoGIF':
        model_name = 'microGIF'
    load_data_target = torch.load(GT_path + model_name + '/' + GT_euid + '/' + tar_fname + IO.fname_ext)
    snn = load_data_target['model']
    model_class = snn.__class__
    euid = GT_model_by_type[model_type_str]
    cur_fname = 'target_GT_model_{}_N_{}'.format(model_type_str, snn.N)

    if not os.path.exists(data_util.prefix + data_util.target_data_path + data_util.matlab_export + cur_fname + '.mat'):
        simulate_and_save_model_spike_train(model=snn, t=duration, exp_num=euid, model_name=model_class.__name__, fname=cur_fname)
    else:
        print('file exists. skipping..')

sys.exit()
