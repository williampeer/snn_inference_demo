import os

import numpy as np
import torch


def get_lfn_from_plot_data_in_folder(exp_folder):
    folder_files = os.listdir(exp_folder)
    loss_file = list(filter(lambda x: x.__contains__('plot_loss'), folder_files))
    if len(loss_file) == 0:
        return False
    else:
        loss_file = loss_file[0]
    plot_data = torch.load(exp_folder + loss_file)['plot_data']
    custom_title = plot_data['custom_title']
    lfn = custom_title.split(',')[0].strip('Loss ')
    return lfn


experiments_path = '/media/william/p6/archive_30122021_full/archive/saved/sleep_data_no_types/'
experiments_path_plot_data = '/media/william/p6/archive_30122021_full/archive/saved/plot_data/sleep_data_no_types/'

euid_to_sleep_exp_num = {}
euid_to_lfn = {}
model_type_dirs = ['LIF_no_cell_types', 'GLIF_no_cell_types']
for model_type_str in model_type_dirs:
    euid_to_sleep_exp_num[model_type_str] = {}
    euid_to_lfn[model_type_str] = {}
    exp_uids = os.listdir(experiments_path + model_type_str)
    euid_num = 0
    assert len(exp_uids) == 2*7*10, "len(exp_uids): {} != 2*7*20 exps. please add logic to account for exps.".format(len(exp_uids))
    for euid in exp_uids:
        lfn = get_lfn_from_plot_data_in_folder(experiments_path_plot_data + model_type_str + '/' + euid + '/')
        estimated_exp_num = int(np.floor(euid_num / 10)) % 7
        euid_to_sleep_exp_num[model_type_str][euid] = estimated_exp_num
        euid_to_lfn[model_type_str][euid] = lfn
        euid_num += 1

experiments_path_sleep_data_microGIF = '/media/william/p6/archive_30122021_full/archive/saved/sleep_data/'
experiments_path_plot_sleep_data_microGIF = '/media/william/p6/archive_30122021_full/archive/saved/plot_data/sleep_data/'
model_type_str = 'microGIF'
euid_to_sleep_exp_num[model_type_str] = {}
euid_to_lfn[model_type_str] = {}
exp_uids = os.listdir(experiments_path_sleep_data_microGIF + model_type_str)
euid_num = 0
assert len(exp_uids) == 2*7*20, "len(exp_uids): {} != 2*7*20 exps. please add logic to account for exps.".format(len(exp_uids))
for euid in exp_uids:
    lfn = get_lfn_from_plot_data_in_folder(experiments_path_plot_sleep_data_microGIF + model_type_str + '/' + euid + '/')
    estimated_exp_num = int(np.floor(euid_num / 20)) % 7
    euid_to_sleep_exp_num[model_type_str][euid] = estimated_exp_num
    euid_to_lfn[model_type_str][euid] = lfn
    euid_num += 1

print(euid_to_sleep_exp_num)

