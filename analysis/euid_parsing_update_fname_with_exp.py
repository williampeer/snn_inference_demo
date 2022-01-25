import os

import torch

from analysis.euid_to_sleep_exp import euid_to_sleep_exp_num


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


exp_paths = ['/media/william/p6/archive_14122021/archive/saved/sleep_data_no_types/',
             '/home/william/repos/snn_inference/Test/saved/sleep_data/']

model_type_dirs = ['LIF_no_cell_types', 'GLIF_no_cell_types', 'microGIF']

euid_to_lfn = {}
for model_str in model_type_dirs:
    for exp_path in exp_paths:
        if os.path.exists(exp_path + model_str):
            euid_dirs = os.listdir(exp_path + model_str)
            for euid in euid_dirs:
                single_exp_plot_data_path = exp_path.replace('saved/', 'saved/plot_data/') + model_str + '/' + euid + '/'
                exp_lfn = get_lfn_from_plot_data_in_folder(single_exp_plot_data_path)
                if exp_lfn:
                    euid_to_lfn[euid] = exp_lfn

matlab_export_path = '/home/william/data/target_data/matlab_export/'
exported_files = os.listdir(matlab_export_path)
for fname in exported_files:
    if fname.__contains__('.mat') and fname.__contains__('_euid_') and not fname.__contains__('_lfn_'):
        euid = fname.split('_euid_')[1].strip('.mat')
        new_fname = '{}_exp_{}.mat'.format(fname.strip('.mat'), euid_to_sleep_exp_num[euid])
        # os.rename(matlab_export_path + fname, matlab_export_path + new_fname)
        print(fname, new_fname)

matlab_results_path = '/home/william/repos/pnmf-fork/results/'
results_files = os.listdir(matlab_results_path)
for fname in results_files:
    if fname.__contains__('.mat') and fname.__contains__('_euid_') and not fname.__contains__('_lfn_'):
        euid = fname.split('_euid_')[1][:-6]
        new_fname = '{}_exp_{}_4.mat'.format(fname[:-6], euid_to_sleep_exp_num[euid])
        # os.rename(matlab_results_path + fname, matlab_results_path + new_fname)
        print(fname, new_fname)
