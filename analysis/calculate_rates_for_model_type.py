import os

import numpy as np
import torch

import IO
import plot
import analysis_util

man_seed = 3
torch.manual_seed(man_seed)
np.random.seed(man_seed)

load_fname = 'snn_model_target_GD_test'
# model_class_lookup = { 'LIF': LIF, 'GLIF': GLIF, 'mesoGIF': microGIF, 'microGIF': microGIF,
#                        'LIF_no_cell_types': LIF_no_cell_types, 'GLIF_no_cell_types': GLIF_no_cell_types }

# experiments_path = '/home/william/repos/snn_inference/Test/saved/'
# experiments_path_plot_data = '/home/william/repos/snn_inference/Test/saved/plot_data/'
experiments_path = '/home/william/repos/snn_inference/Test/saved/GT/'
experiments_path_plot_data = '/home/william/repos/snn_inference/Test/saved/plot_data/GT/'
# experiments_path = '/media/william/p6/archive_14122021/archive/saved/sleep_data_no_types/'
# archive_name = 'data/'
# plot_data_path = experiments_path + 'plot_data/'
# model_type_dirs = os.listdir(experiments_path)
model_type_dirs = ['LIF', 'GLIF', 'mesoGIF', 'microGIF']
# model_type_dirs = ['LIF']


plot_exp_type = 'export_metrics'
global_fname_rates = 'export_rates_only_{}_all.eps'.format(experiments_path.split('/')[-2])
xticks = []
mean_rates = []; std_rates = []; target_rates = []
for model_type_str in model_type_dirs:
    target_model = analysis_util.get_target_model(model_type_str)
    target_rate = analysis_util.get_mean_rate_for_model(target_model)
    cur_fname = 'export_rates_{}_{}_N_{}.eps'.format(model_type_str, experiments_path.split('/')[-2], target_model.N)

    plot_uid = model_type_str
    full_path = './figures/' + plot_exp_type + '/' + plot_uid + '/'
    # mean_rates_by_lfn = { 'frd': [], 'vrd': [], 'bernoulli_nll': [], 'poisson_nll': [] }
    if model_type_str == 'microGIF' or model_type_str == 'mesoGIF':
        mean_dist_by_lfn = {'bernoulli_nll': [], 'poisson_nll': []}
        mean_rates_by_lfn = {'bernoulli_nll': [], 'poisson_nll': []}
    else:
        mean_dist_by_lfn = {'frd': [], 'vrd': []}
        mean_rates_by_lfn = {'frd': [], 'vrd': []}

    # model_class = microGIF
    model_type_path = model_type_str
    if model_type_str == 'mesoGIF':
        model_type_path = 'microGIF'
    exp_uids = os.listdir(experiments_path + model_type_path)
    lfn_ctr = 0
    for euid in exp_uids:
        lfn = analysis_util.get_lfn_from_plot_data_in_folder(experiments_path_plot_data + model_type_path + '/' + euid + '/')

        load_data = torch.load(experiments_path + '/' + model_type_path + '/' + euid + '/' + load_fname + IO.fname_ext)
        cur_model = load_data['model']
        if cur_model.N == target_model.N:
            cur_rate = analysis_util.get_mean_rate_for_model(cur_model)
            if not np.isnan(cur_rate) and (cur_rate < 1.75 * target_rate and cur_rate > 0.25 * target_rate):
                # cur_dists.append(mean_dist_by_lfn[lfn][r_i])
                mean_rates_by_lfn[lfn].append(cur_rate)
            else:  # diverged, fallback to init. model
                # init_p_dist, init_p_dist_std = analysis_util.get_init_param_dist(target_model)
                # cur_dists.append(init_p_dist)
                neuron_types = False
                if hasattr(target_model, 'neuron_types'):
                    neuron_types = target_model.neuron_types
                init_seed = 23 + int(lfn_ctr*20/len(exp_uids))
                init_model = analysis_util.get_init_model(target_model.__class__, seed=init_seed, N=target_model.N, neuron_types=neuron_types)
                # correct rate to init for diverged fit
                cur_rate = analysis_util.get_mean_rate_for_model(init_model)
                mean_rates_by_lfn[lfn].append(cur_rate)
        lfn_ctr += 1

    for lfn in mean_rates_by_lfn.keys():
        target_rates.append(target_rate)
        cur_mean_rate = np.mean(list(filter(lambda x: not np.isnan(x) and (x < 1.75 * target_rate and x > 0.25 * target_rate), mean_rates_by_lfn[lfn])))
        cur_std_rate = np.std(list(filter(lambda x: not np.isnan(x) and (x < 1.75 * target_rate and x > 0.25 * target_rate), mean_rates_by_lfn[lfn])))
        if np.isnan(cur_mean_rate):
            cur_mean_rate = 0.; cur_std_rate = 0.
        mean_rates.append(cur_mean_rate)
        std_rates.append(cur_std_rate)
        xticks.append('{},\n${}$'.format(model_type_str.replace('microGIF', 'miGIF').replace('mesoGIF', 'meGIF'),
                                         lfn.replace('poisson_nll', 'P_{NLL}').replace('bernoulli_nll', 'B_{NLL}')))
# plot.bar_plot(np.asarray(mean_dists), np.asarray(std_dists), labels=xticks, exp_type=plot_exp_type, uuid='all', fname=global_fname)
# import importlib
# importlib.reload(plot)
# For non-micro:
# plot.bar_plot_neuron_rates(target_rates[:-2], np.asarray(mean_rates)[:-2], 0., np.asarray(std_rates)[:-2], plot_exp_type, 'all',
#                            custom_legend=['Target models', 'Fitted models'],
#                            fname=global_fname_rates, xticks=xticks, custom_colors=['Green', 'Magenta'])
plot.bar_plot_neuron_rates(target_rates, np.asarray(mean_rates), 0., np.asarray(std_rates), plot_exp_type, 'all',
                           custom_legend=['Target models', 'Fitted models'],
                           fname=global_fname_rates, xticks=xticks, custom_colors=['Green', 'Magenta'])

# sys.exit()
