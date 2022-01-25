import os

import numpy as np
import torch

import IO
import plot
from Models.GLIF import GLIF
from Models.GLIF_no_cell_types import GLIF_no_cell_types
from Models.LIF import LIF
from Models.LIF_no_cell_types import LIF_no_cell_types
from Models.microGIF import microGIF
from analysis import analysis_util
from analysis.euid_to_sleep_exp import euid_to_sleep_exp_num

man_seed = 3
torch.manual_seed(man_seed)
np.random.seed(man_seed)

model_class_lookup = { 'LIF': LIF, 'GLIF': GLIF, 'microGIF': microGIF,
                       'LIF_no_cell_types': LIF_no_cell_types, 'GLIF_no_cell_types': GLIF_no_cell_types }

experiments_path = '/media/william/p6/archive_30122021_full/archive/saved/sleep_data_no_types/'
experiments_path_plot_data = '/media/william/p6/archive_30122021_full/archive/saved/plot_data/sleep_data_no_types/'

model_type_dirs = ['LIF_no_cell_types', 'GLIF_no_cell_types']
exp_folder_name = experiments_path.split('/')[-2]


sleep_exps = ['exp108', 'exp109', 'exp124', 'exp126', 'exp138', 'exp146', 'exp147']
rate_per_exp = {}
loss_per_exp = {}
target_rates = []
target_rate_stds = []
plot_exp_type = 'export_metrics'
for model_type_str in model_type_dirs:
    for exp_str in sleep_exps:
        target_rate, target_rate_std = analysis_util.get_target_rate_for_sleep_exp(exp_str)
        target_rates.append(target_rate)
        target_rate_stds.append(target_rate_std)
        cur_fname = 'export_rates_{}_{}_{}.eps'.format(model_type_str, experiments_path.split('/')[-2], exp_str)

        plot_uid = model_type_str
        full_path = './figures/' + plot_exp_type + '/' + plot_uid + '/'
        # mean_rates_by_lfn = { 'frd': [], 'vrd': [], 'bernoulli_nll': [], 'poisson_nll': [] }
        rate_per_exp[model_type_str]= { 'frd': {}, 'vrd': {} }
        loss_per_exp[model_type_str] = { 'frd': {}, 'vrd': {} }

    if os.path.exists(experiments_path + model_type_str):
        exp_uids = os.listdir(experiments_path + model_type_str)
        for euid in exp_uids:
            sleep_exp = euid_to_sleep_exp_num[model_type_str][euid]
            lfn, loss = analysis_util.get_lfn_loss_from_plot_data_in_folder(experiments_path_plot_data + model_type_str + '/' + euid + '/')

            load_fname = 'snn_model_target_GD_test'
            load_data = torch.load(experiments_path + '/' + model_type_str + '/' + euid + '/' + load_fname + IO.fname_ext)
            cur_model = load_data['model']
            if rate_per_exp[model_type_str][lfn].keys().__contains__(sleep_exp):
                rate_per_exp[model_type_str][lfn][sleep_exp].append(analysis_util.get_mean_rate_for_model(cur_model))
                loss_per_exp[model_type_str][lfn][sleep_exp].append(loss)
            else:
                rate_per_exp[model_type_str][lfn][sleep_exp] = [(analysis_util.get_mean_rate_for_model(cur_model))]
                loss_per_exp[model_type_str][lfn][sleep_exp] = [(loss)]

results_dict = { 'rate_per_exp': rate_per_exp, 'loss_per_exp': loss_per_exp,
                 'target_rates': target_rates, 'target_rate_stds': target_rate_stds }
torch.save(results_dict, './save_stuff/results_dict_LIF_GLIF_rates_losses.pt')

sleep_exp_labels = list(map(lambda x: 'exp ' + str(sleep_exps.index(x)), sleep_exps))
sleep_exp_mean_rates = { 'LIF_no_cell_types': { 'frd': [], 'vrd': [] }, 'GLIF_no_cell_types': { 'frd': [], 'vrd': [] }}

LIF_GLIF_processed_res = {}
for mt_str in rate_per_exp.keys():
    mean_rates_frd = []; mean_rate_std_frd = []
    # mean_rates_vrd = []; mean_rate_std_vrd = []
    mean_loss_frd = []; mean_loss_std_frd = []
    # mean_loss_vrd = []; mean_loss_std_vrd = []

    for sleep_exp_num in range(7):
        mean_rates_frd.append(np.mean(rate_per_exp[mt_str]['frd'][sleep_exp_num]))
        mean_rate_std_frd.append(np.std(rate_per_exp[mt_str]['frd'][sleep_exp_num]))
        # mean_rates_vrd.append(np.mean(rate_per_exp[mt_str]['vrd'][sleep_exp_num]))
        # mean_rate_std_vrd.append(np.std(rate_per_exp[mt_str]['vrd'][sleep_exp_num]))

        mean_loss_frd.append(np.mean(loss_per_exp[mt_str]['frd'][sleep_exp_num]))
        mean_loss_std_frd.append(np.std(loss_per_exp[mt_str]['frd'][sleep_exp_num]))
        # mean_loss_vrd.append(np.mean(loss_per_exp[mt_str]['vrd'][sleep_exp_num]))
        # mean_loss_std_vrd.append(np.std(loss_per_exp[mt_str]['vrd'][sleep_exp_num]))

    # plot.bar_plot_neuron_rates(mean_rates_frd, mean_rate_std_frd, mean_rates_vrd, mean_rate_std_vrd,
    #                            xticks=sleep_exp_labels, exp_type='export_sleep', uuid='export_sleep', fname='approx_rate_across_exp_{}.eps'.format('mt_str'),
    #                            custom_legend=['frd', 'vrd'])

    plot.bar_plot_neuron_rates(mean_rates_frd, mean_rate_std_frd, target_rates, target_rate_stds,
                               xticks=sleep_exp_labels, exp_type='export_sleep', uuid='export_sleep',
                               fname='approx_rate_across_exp_{}_frd_vs_fitted.eps'.format(mt_str),
                               custom_legend=['Fitted model', 'Target model'])
    # plot.bar_plot_neuron_rates(mean_rates_vrd, mean_rate_std_vrd, target_rates, target_rate_stds,
    #                            xticks=sleep_exp_labels, exp_type='export_sleep', uuid='export_sleep', fname='approx_rate_across_exp_{}_bernoulli_nll_vs_fitted.eps'.format(mt_str),
    #                            custom_legend=['Fitted model', 'Target model'])

    plot.bar_plot(mean_loss_frd, mean_loss_std_frd, labels=sleep_exp_labels, exp_type='export_sleep', uuid='export_sleep',
                  fname='approx_loss_across_exp_{}.eps'.format(mt_str), ylabel='Loss (frd)')

    processed_results_dict = { 'mean_rates_frd': mean_rates_frd, 'mean_rate_std_frd': mean_rate_std_frd,
                               # 'mean_rates_vrd': mean_rates_vrd, 'mean_rate_std_vrd': mean_rate_std_vrd,
                               'mean_loss_frd': mean_loss_frd, 'mean_loss_std_frd': mean_loss_std_frd }
                               # 'mean_loss_vrd': mean_loss_vrd, 'mean_loss_std_vrd': mean_loss_std_vrd }
    LIF_GLIF_processed_res[mt_str] = processed_results_dict

plot.bar_plot_neuron_rates(LIF_GLIF_processed_res['LIF_no_cell_types']['mean_rates_frd'], LIF_GLIF_processed_res['LIF_no_cell_types']['mean_rate_std_frd'],
                           LIF_GLIF_processed_res['GLIF_no_cell_types']['mean_rates_frd'], LIF_GLIF_processed_res['GLIF_no_cell_types']['mean_rate_std_frd'],
                           xticks=sleep_exp_labels, exp_type='export_sleep', uuid='export_sleep', ylabel='Loss (frd)',
                           fname='approx_loss_both_model_types_LIF_GLIF.eps',
                           custom_legend=['LIF', 'GLIF'])

torch.save(LIF_GLIF_processed_res, './save_stuff/LIF_GLIF_processed_res.pt')
