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
from analysis.analysis_util import get_lfn_from_plot_data_in_folder
from spike_train_matlab_export import simulate_and_save_model_spike_train

man_seed = 3
torch.manual_seed(man_seed)
np.random.seed(man_seed)

duration = 30 * 1000

load_fname = 'snn_model_target_GD_test'
model_class_lookup = { 'LIF': LIF, 'GLIF': GLIF, 'microGIF': microGIF,
                       'LIF_no_cell_types': LIF_no_cell_types, 'GLIF_no_cell_types': GLIF_no_cell_types }

# experiments_path = '/home/william/repos/snn_inference/Test/saved/GT/'
# experiments_path = '/media/william/p6/archive_14122021/archive/saved/sleep_data_no_types/'
# experiments_path = '/media/william/p6/archive_30122021_full/archive/saved/sleep_data_no_types/'  # GLIF, LIF
# experiments_path = '/media/william/p6/archive_30122021_full/archive/saved/sleep_data/'  # microGIF / SGIF

# exp_paths = ['/media/william/p6/archive_30122021_full/archive/saved/sleep_data_no_types/',
#              '/media/william/p6/archive_30122021_full/archive/saved/sleep_data/']

# exp_paths = ['/home/william/repos/archives_snn_inference/archive_0201/archive/saved/Synthetic/']
exp_paths = ['/home/william/repos/archives_snn_inference/archive_mesoGIF_and_LIF_GIF_pscapes_0401/archive/saved/Synthetic/']

for experiments_path in exp_paths:
    # archive_name = 'data/'
    # plot_data_path = experiments_path + 'plot_data/'
    model_type_dirs = os.listdir(experiments_path)
    # model_type_dirs = ['microGIF']

    for model_type_str in model_type_dirs:
        if not model_type_str.__contains__("plot_data"):
            model_class = model_class_lookup[model_type_str]
            # model_class = microGIF
            if os.path.exists(experiments_path + model_type_str):
                exp_uids = os.listdir(experiments_path + '/' + model_type_str)
                for euid in exp_uids:
                    load_data = torch.load(experiments_path + '/' + model_type_str + '/' + euid + '/' + load_fname + IO.fname_ext)
                    snn = load_data['model']

                    single_exp_plot_data_path = experiments_path.replace('saved/', 'saved/plot_data/') + model_type_str + '/' + euid + '/'
                    exp_lfn = get_lfn_from_plot_data_in_folder(single_exp_plot_data_path)
                    # saved_target_losses = load_data['loss']
                    # num_neurons = snn.N
                    cur_fname = 'nuovo_synthetic_v3_mesoGIF__spikes_mt_{}_lfn_{}_euid_{}'.format(model_class.__name__, exp_lfn, euid)

                    if not os.path.exists(data_util.prefix + data_util.target_data_path + data_util.matlab_export + cur_fname + '.mat'):
                        print('simulating data for: {}'.format(experiments_path + model_type_str))
                        simulate_and_save_model_spike_train(model=snn, t=duration, exp_num=euid, model_name=model_class.__name__, fname=cur_fname)
                    else:
                        print('file exists. skipping..')
            else:
                print('path does not exist: {}'.format(experiments_path + '/' + model_type_str))

sys.exit()
