import os
import sys

import numpy as np
import torch

from IO import makedir_if_not_exists
from analysis.spike_train_matlab_export import load_and_export_sim_data
from data_util import prefix, target_data_path


def main(argv):
    print('Argument List:', str(argv))
    offset = 42

    experiments_path = '/home/william/repos/archives_snn_inference/archive_osx_2009/archive/saved/'  # Done
    # experiments_path = '/media/william/p6/archives_snn_inference/PLACEHOLDER/saved/'

    archive_name = 'data/'
    plot_data_path = experiments_path + 'plot_data/'
    folders = os.listdir(experiments_path)
    # pdata_files = os.listdir(plot_data_path)

    spike_train_files_post_training = {}
    for folder_path in folders:
        # print(folder_path)

        full_folder_path = experiments_path + folder_path + '/'
        if not folder_path.__contains__('.DS_Store'):
            files = os.listdir(full_folder_path)
            id = folder_path.split('-')[-1]
        else:
            files = []
            id = 'None'

        for f in files:
            if f.__contains__('exp_num'):
                model_type = f.split('_exp_num_')[0]
                # if model_type not in ['LIF', 'LIF_R']:  # mt mask
                #     pass
                # else:
                exp_num = int(f.split('_exp_num_')[1].split('_')[0])
                print('exp_num: {}'.format(exp_num))

                pdata_files = os.listdir(plot_data_path + folder_path)
                pdata_loss_files = []
                for pdata_f in pdata_files:
                    if pdata_f.__contains__('plot_losses'):
                        pdata_loss_files.append(pdata_f)

                rand_seed = int(exp_num) + len(pdata_loss_files) + 1
                torch.manual_seed(rand_seed)
                np.random.seed(rand_seed)

                print('exp_num: {}, rand_seed: {}'.format(exp_num, rand_seed))

                pdata_loss_files.sort()
                if len(pdata_loss_files) > exp_num-offset:
                    cur_exp_pdata_loss_file = pdata_loss_files[exp_num-offset]
                    loss_data = torch.load(plot_data_path + folder_path + '/' + cur_exp_pdata_loss_file)
                    custom_title = loss_data['plot_data']['custom_title']
                    optimiser = custom_title.split(', ')[1].strip(' ')
                    # model_type = custom_title.split(',')[0].split('(')[-1]
                    lr = custom_title.split(', ')[-1].strip(' =lr').strip(')').replace('.', '')
                    lfn = loss_data['plot_data']['fname'].split('loss_fn_')[1].split('_tau')[0]
                    exp_type = loss_data['plot_data']['exp_type']

                    # cur_fname = 'spikes_{}_{}_{}_{}_{}_{}_exp_num_{}_60s'.format(exp_type, model_type, optimiser, lfn, lr, id, exp_num).replace('=', '_')
                    cur_fname = 'nuovo_spikes_mt_{}_et_{}_seed_{}'.format(model_type, exp_type, exp_num)
                    save_file_name = prefix + target_data_path + archive_name + cur_fname + '.mat'

                    # if lfn == 'FIRING_RATE_DIST':
                    print('checking: {}'.format(save_file_name))
                    if not os.path.exists(prefix + target_data_path + archive_name) or not os.path.exists(save_file_name):
                        makedir_if_not_exists('./figures/default/plot_imported_model/' + archive_name)
                        load_and_export_sim_data(full_folder_path + f, fname=archive_name + cur_fname)
                    else:
                        print('file exists. skipping..')
                    # else:
                    #     print('.. 💩 {}, {}, {}, {}, {}'.format(lfn, model_type, optimiser, lr, custom_title))
                    #     load_and_export_sim_data(f, optim='Adam_frdvrda_001')


if __name__ == "__main__":
    main(sys.argv[1:])
