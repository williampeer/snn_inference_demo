import os
import sys

import numpy as np
import torch
from torch import FloatTensor as FT

import IO
from Models.GLIF import GLIF
from Models.LIF import LIF
from Models.microGIF import microGIF
from experiments import draw_from_uniform, zip_dicts
from plot import bar_plot_pair_custom_labels


class_lookup = { 'LIF': LIF, 'GLIF': GLIF, 'microGIF': microGIF }

def get_init_params(model_class, rand_seed, N=12):
    torch.manual_seed(rand_seed)
    np.random.seed(rand_seed)

    w_mean = 0.3; w_var = 0.2; programmatic_neuron_types = torch.ones((N,))
    for n_i in range(int(2 * N / 3), N):
        programmatic_neuron_types[n_i] = -1
    neuron_types = programmatic_neuron_types
    rand_ws = (w_mean - w_var) + 2 * w_var * torch.rand((N, N))
    for i in range(len(neuron_types)):
        if neuron_types[i] == -1:
            rand_ws[i, :] = -torch.abs(FT(rand_ws[i, :]))
        elif neuron_types[i] == 1:
            rand_ws[i, :] = torch.abs(FT(rand_ws[i, :]))
        else:
            raise NotImplementedError()

    return zip_dicts({'w': rand_ws}, draw_from_uniform(model_class.parameter_init_intervals, N))


def euclid_dist(p1, p2):
    return np.sqrt(np.power((p1 - p2), 2).sum()) / len(p1)


def main():
    all_exps_path = '/home/william/repos/archives_snn_inference/GENERIC/archive/saved/plot_data/'
    folders = os.listdir(all_exps_path)
    experiment_averages = {}
    optim_to_include = 'Adam'
    # res_per_exp = {}
    for exp_folder in folders:
        full_folder_path = all_exps_path + exp_folder + '/'

        if not exp_folder.__contains__('.DS_Store'):
            files = os.listdir(full_folder_path)
            id = exp_folder.split('-')[-1]
        else:
            files = []
            id = 'None'

        param_files = []; optimiser = None; model_type = ''
        for f in files:
            if f.__contains__('plot_all_param_pairs_with_variance'):
                param_files.append(f)
            elif optimiser is None and f.__contains__('plot_losses'):
                f_data = torch.load(full_folder_path + f)
                custom_title = f_data['plot_data']['custom_title']
                spf = custom_title.split('spf=')[-1].split(')')[0]
                optimiser = custom_title.split(', ')[1].strip(' ')
                model_type = custom_title.split(',')[0].split('(')[-1]
                lr = custom_title.split(', ')[-1].strip(' =lr').strip(')')
                lfn = f_data['plot_data']['fname'].split('loss_fn_')[1].split('_tau')[0]

        if optimiser == optim_to_include and model_type in ['LIF', 'GLIF', 'microGIF'] and spf == 'None' and len(param_files) == 1:
            print('Succes! Processing exp: {}'.format(exp_folder + '/' + param_files[0]))
            exp_data = torch.load(full_folder_path + param_files[0])
            # param_names = exp_data['plot_data']['param_names']
            param_names = class_lookup[model_type].parameter_names
            m_p_by_exp = exp_data['plot_data']['param_means']

            GT_path = '/home/william/repos/snn_inference/Test/saved/'
            GT_model_by_type = {'LIF': '12-09_11-49-59-999',
                                'GLIF': '12-09_11-12-47-541',
                                'mesoGIF': '12-09_14-56-20-319',
                                'microGIF': '12-09_14-56-17-312'}
            GT_euid = GT_model_by_type[model_type]
            tar_fname = 'snn_model_target_GD_test'
            model_name = model_type
            if model_type == 'mesoGIF':
                model_name = 'microGIF'
            load_data_target = torch.load(GT_path + model_name + '/' + GT_euid + '/' + tar_fname + IO.fname_ext)
            target_model = load_data_target['model']
            # model_class = tar_model.__class__

            if(len(m_p_by_exp)>0):
                model_N = m_p_by_exp[1][0][0].shape[0]

                # config = '{}_{}_{}_{}'.format(model_type, optimiser, lfn, lr.replace('.', '_'))
                config = '{}_{}_{}'.format(model_type, optimiser, lfn)
                if not experiment_averages.__contains__(config):
                    experiment_averages[config] = { 'dist' : {}, 'std': {}, 'init_dist': {}, 'init_std': {}}
                    for k in range(len(m_p_by_exp)):
                        experiment_averages[config]['dist'][param_names[k]] = []
                        experiment_averages[config]['std'][param_names[k]] = []
                        experiment_averages[config]['init_dist'][param_names[k]] = []
                        experiment_averages[config]['init_std'][param_names[k]] = []

                for p_i in range(len(m_p_by_exp)):
                    per_exp = []
                    for e_i in range(len(m_p_by_exp[p_i])):
                        init_model_params = get_init_params(class_lookup[model_type], e_i, N=model_N)
                        # if(param_names[p_i] in init_model_params.keys()):
                        t_p_by_exp = target_model.params_wrapper()
                        c_d = euclid_dist(init_model_params[param_names[p_i]].numpy(), t_p_by_exp[p_i])
                        per_exp.append(c_d)
                    experiment_averages[config]['init_dist'][param_names[p_i]].append(np.mean(per_exp))
                    experiment_averages[config]['init_std'][param_names[p_i]].append(np.std(per_exp))

                for p_i in range(len(m_p_by_exp)):
                    per_exp = []
                    for e_i in range(len(m_p_by_exp[p_i])):
                        t_p_by_exp = target_model.params_wrapper()
                        c_d = euclid_dist(m_p_by_exp[p_i][e_i][0], t_p_by_exp[p_i])
                        per_exp.append(c_d)
                    experiment_averages[config]['dist'][param_names[p_i]].append(np.mean(per_exp))
                    experiment_averages[config]['std'][param_names[p_i]].append(np.std(per_exp))


    # unpack
    exp_avg_ds = []; exp_avg_stds = []; exp_avg_init_ds = []; exp_avg_init_stds = []
    keys_list = list(experiment_averages.keys())
    keys_list.sort()
    labels = []
    for k_i, k_v in enumerate(keys_list):
        model_type = k_v.split('_{}'.format(optim_to_include))[0]
        param_names = class_lookup[model_type].parameter_names
        label_param_names = map(lambda x: '${}$'.format(x.replace('delta_theta_', '\delta\\theta_').replace('delta_V', '\delta_V').replace('tau', '\\tau')), param_names)
        # if not (k_v.__contains__('vrdfrda') or k_v.__contains__('pnll')):
        # if not (k_v.__contains__('vrdfrda') or k_v.__contains__('pnll')):
        if True:
        # if k_v not in ['LIF_Adam_vrdfrda_0_05', 'LIF_Adam_pnll_0_05']:
            labels.append(k_v
                          .replace('LIF_SGD_', 'LIF\n')
                          .replace('LIF_R_SGD_', 'R\n')
                          .replace('LIF_ASC_SGD_', 'A\n')
                          .replace('LIF_R_ASC_SGD_', 'R_A\n')
                          .replace('GLIF_SGD_', 'GLIF\n')
                          .replace('LIF_RMSprop_', 'LIF\n')
                          .replace('LIF_R_RMSprop_', 'R\n')
                          .replace('LIF_ASC_RMSprop_', 'A\n')
                          .replace('LIF_R_ASC_RMSprop_', 'R_A\n')
                          .replace('GLIF_RMSprop_', 'GLIF\n')
                          # .replace('_', '\n')
                          # .replace('LIF_Adam_', '').replace('LIF_SGD_', '').replace('_', '\n')
                          # .replace('frdvrda', '$d_A$').replace('frdvrd', '$d_C$')
                          .replace('FIRING_RATE_DIST', '$d_F$')
                          .replace('RATE_PCC_HYBRID', '$d_P$')
                          .replace('VAN_ROSSUM_DIST', '$d_V$')
                          .replace('MSE', '$mse$'))
            print('processing exp results for config: {}'.format(k_v))
            flat_ds = []; flat_stds = []
            for d_i, d in enumerate(experiment_averages[k_v]['dist'].values()):
                flat_ds.append(d[0])
            for s_i, s in enumerate(experiment_averages[k_v]['std'].values()):
                flat_stds.append(s[0])
            flat_ds_init = []; flat_stds_init = []
            for d_i, d in enumerate(experiment_averages[k_v]['init_dist'].values()):
                flat_ds_init.append(d[0])
            for s_i, s in enumerate(experiment_averages[k_v]['init_std'].values()):
                flat_stds_init.append(s[0])

            # norm_kern = np.array(flat_ds_init)
            norm_kern = np.ones_like(flat_ds_init)
            norm_kern[np.isnan(norm_kern)] = 1.0

            bar_plot_pair_custom_labels(np.array(flat_ds_init)/norm_kern, np.array(flat_ds)/norm_kern,
                                        np.array(flat_stds_init)/norm_kern, np.array(flat_stds)/norm_kern,
                                        label_param_names, 'export', 'test',
                                        'exp_export_all_euclid_dist_params_{}.png'.format(k_v),
                                        'Avg Euclid dist per param for configuration {}'.format(k_v.replace('0_0', '0.0')).replace('_', ', '),
                                        legend=['Initial model', 'Fitted model'])

            exp_avg_ds.append(np.mean(np.array(flat_ds)/norm_kern))
            exp_avg_stds.append(np.std(np.array(flat_ds)/norm_kern))
            # exp_avg_stds.append(np.mean(flat_stds))
            exp_avg_init_ds.append(np.mean(np.array(flat_ds_init)/norm_kern))
            exp_avg_init_stds.append(np.std(np.array(flat_ds_init)/norm_kern))
            # exp_avg_init_stds.append(np.mean(flat_stds_init))

    bar_plot_pair_custom_labels(np.array(exp_avg_init_ds), np.array(exp_avg_ds),
                                np.array(exp_avg_init_stds), np.array(exp_avg_stds),
                                labels, 'export', 'test',
                                'exp_export_all_euclid_dist_params_across_exp_{}.png'.format(optim_to_include),
                                'Avg Euclid dist for all parameters across experiments',
                                legend=['Initial model', 'Fitted model'], baseline=1.0)


if __name__ == "__main__":
    main()
    sys.exit(0)
