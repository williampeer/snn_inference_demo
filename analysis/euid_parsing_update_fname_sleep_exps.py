import os

import torch

from analysis.euid_to_sleep_exp import euid_to_sleep_exp_num, euid_to_lfn


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


exp_paths = ['/media/william/p6/archive_30122021_full/archive/saved/sleep_data_no_types/',
             '/media/william/p6/archive_30122021_full/archive/saved/sleep_data/']
matlab_export_path = '/home/william/data/target_data/matlab_export/'
matlab_results_path = '/home/william/repos/pnmf-fork/results/'
# model_type_dirs = ['LIF_no_cell_types', 'GLIF_no_cell_types', 'microGIF']
model_type_dirs = ['LIF_no_cell_types', 'GLIF_no_cell_types']

for mt_str in model_type_dirs:
    exported_files = os.listdir(matlab_export_path)
    model_type_export_files = list(filter(lambda x: x.__contains__('_{}_'.format(mt_str)) and x.__contains__('_sleep_v2_'), exported_files))
    for fname in model_type_export_files:
        euid = fname.split('_euid_')[1].strip('.mat').strip('.pt')
        if euid.__contains__('_exp_'):
            euid = euid.split('_exp_')[0]
        _ = (euid_to_sleep_exp_num[mt_str][euid])
        if fname.__contains__('.mat') and fname.__contains__('_euid_') and not fname.__contains__('_lfn_'):
            new_fname = '{}_exp_{}_lfn_{}.mat'.format(fname.strip('.mat'), euid_to_sleep_exp_num[mt_str][euid], euid_to_lfn[mt_str][euid])
            # os.rename(matlab_export_path + fname, matlab_export_path + new_fname)
            print(fname, new_fname)

    results_files = os.listdir(matlab_results_path)
    model_type_results_files = list(filter(lambda x: x.__contains__('_{}_'.format(mt_str)) and x.__contains__('_sleep_v2_'), exported_files))
    for fname in model_type_results_files:
        euid = fname.split('_euid_')[1].strip('.mat').strip('.pt')
        if euid.__contains__('_exp_'):
            euid = euid.split('_exp_')[0]
        _ = (euid_to_sleep_exp_num[mt_str][euid])
        if fname.__contains__('.mat') and fname.__contains__('_euid_') and not fname.__contains__('_lfn_'):
            new_fname = '{}_exp_{}_lfn_{}_4.mat'.format(fname[:-4], euid_to_sleep_exp_num[mt_str][euid], euid_to_lfn[mt_str][euid])
            # os.rename(matlab_results_path + fname, matlab_results_path + new_fname)
            print(fname, new_fname)
