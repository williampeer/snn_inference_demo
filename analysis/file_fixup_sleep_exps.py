import os

import numpy as np


def update_fnames_if_sleep_len(base_path, sleep_exp_files):
    sleep_exp_files.sort()
    assert len(sleep_exp_files) == 280, "expected len 280, was: {}".format(len(sleep_exp_files))

    exp_ctr = 0
    for fname in sleep_exp_files:
        estimated_exp_num = int(np.floor(exp_ctr / 20)) % 7
        if fname.__contains__('.mat') and fname.__contains__('_euid_') and not fname.__contains__('_exp_'):
            # euid = fname.split('_euid_')[1].strip('.mat')
            new_fname = '{}_exp_{}.mat'.format(fname.strip('.mat'), estimated_exp_num)
            # os.rename(base_path + fname, base_path + new_fname)
            assert os.path.exists(base_path + fname), "os.path.exists(base_path + fname): {}".format(base_path + fname)
            # assert os.path.exists(base_path + new_fname), "os.path.exists(base_path + new_fname): {}".format(base_path + new_fname)
            print('A: {}, \nB: {}'.format(fname, new_fname))

        exp_ctr += 1

sleep_exps = ['exp108', 'exp109', 'exp124', 'exp126', 'exp138', 'exp146', 'exp147']

matlab_export_path = '/home/william/data/target_data/matlab_export/'
exported_files = os.listdir(matlab_export_path)
exported_files_LIF = list(filter(lambda x: x.__contains__('_sdnt_') and x.__contains__('_LIF_') and x.__contains__('.mat'), exported_files))
exported_files_GLIF = list(filter(lambda x: x.__contains__('_sdnt_') and x.__contains__('_LIF_') and x.__contains__('.mat'), exported_files))
# exported_files_microGIF = list(filter(lambda x: x.__contains__('_sdnt_') and x.__contains__('_LIF_') and x.__contains__('.mat'), exported_files))
# exported_files_LIF = list(filter(lambda x: x.__contains__('_sleep_v2_') and x.__contains__('_LIF_') and x.__contains__('.mat'), exported_files))
# exported_files_GLIF = list(filter(lambda x: x.__contains__('_sleep_v2_') and x.__contains__('_GLIF_') and x.__contains__('.mat'), exported_files))
exported_files_microGIF = list(filter(lambda x: x.__contains__('_sleep_v2_') and x.__contains__('_microGIF_') and x.__contains__('.mat'), exported_files))

matlab_results_path = '/home/william/repos/pnmf-fork/results/'
matlab_exported_files = os.listdir(matlab_results_path)
matlab_exported_files_LIF = list(filter(lambda x: x.__contains__('_sdnt_') and x.__contains__('_LIF_') and x.__contains__('.mat'), matlab_exported_files))
matlab_exported_files_GLIF = list(filter(lambda x: x.__contains__('_sdnt_') and x.__contains__('_LIF_') and x.__contains__('.mat'), matlab_exported_files))
# matlab_exported_files_microGIF = list(filter(lambda x: x.__contains__('_sdnt_') and x.__contains__('_LIF_') and x.__contains__('.mat'), matlab_exported_files))
# matlab_exported_files_LIF = list(filter(lambda x: x.__contains__('_sleep_v2_') and x.__contains__('_LIF_') and x.__contains__('.mat'), matlab_exported_files))
# matlab_exported_files_GLIF = list(filter(lambda x: x.__contains__('_sleep_v2_') and x.__contains__('_GLIF_') and x.__contains__('.mat'), matlab_exported_files))
matlab_exported_files_microGIF = list(filter(lambda x: x.__contains__('_sleep_v2_') and x.__contains__('_microGIF_') and x.__contains__('.mat'), matlab_exported_files))

update_fnames_if_sleep_len(matlab_export_path, exported_files_LIF)
update_fnames_if_sleep_len(matlab_export_path, exported_files_GLIF)
update_fnames_if_sleep_len(matlab_export_path, exported_files_microGIF)

update_fnames_if_sleep_len(matlab_results_path, matlab_exported_files_LIF)
update_fnames_if_sleep_len(matlab_results_path, matlab_exported_files_GLIF)
update_fnames_if_sleep_len(matlab_results_path, matlab_exported_files_microGIF)
