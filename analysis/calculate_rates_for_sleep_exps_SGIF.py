import numpy as np
import torch

import plot
from Models.GLIF import GLIF
from Models.GLIF_no_cell_types import GLIF_no_cell_types
from Models.LIF import LIF
from Models.LIF_no_cell_types import LIF_no_cell_types
from Models.microGIF import microGIF
from analysis import analysis_util

man_seed = 3
torch.manual_seed(man_seed)
np.random.seed(man_seed)

model_class_lookup = { 'LIF': LIF, 'GLIF': GLIF, 'microGIF': microGIF,
                       'LIF_no_cell_types': LIF_no_cell_types, 'GLIF_no_cell_types': GLIF_no_cell_types }

experiments_path = '/media/william/p6/archive_30122021_full/archive/saved/sleep_data_no_types/'
experiments_path_plot_data = '/media/william/p6/archive_30122021_full/archive/saved/plot_data/sleep_data_no_types/'
experiments_path_sleep_data_microGIF = '/media/william/p6/archive_30122021_full/archive/saved/sleep_data/'
experiments_path_sleep_data_microGIF_plot_data = '/media/william/p6/archive_30122021_full/archive/saved/plot_data/sleep_data/'
model_type_dirs = ['LIF_no_cell_types', 'GLIF_no_cell_types', 'microGIF']
exp_folder_name = experiments_path.split('/')[-2]


sleep_exps = ['exp108', 'exp109', 'exp124', 'exp126', 'exp138', 'exp146', 'exp147']
plot_exp_type = 'export_metrics'

loaded = torch.load('./save_stuff/calc_rates_for_sleep_exps_res_SGIF.pt')
print('loaded: ', loaded)
print('loaded keys: ', loaded.keys())
rate_per_exp = loaded['rate_per_exp']
loss_per_exp = loaded['loss_per_exp']
print(rate_per_exp)
target_rates = []
target_rate_stds = []

for exp_str in rate_per_exp.keys():
    target_rate, tar_rate_std = analysis_util.get_target_rate_for_sleep_exp(exp_str)
    target_rates.append(target_rate)
    target_rate_stds.append(tar_rate_std)

ctr = 0
res_rates = {}
res_rates_std = {}
res_loss = {}
res_loss_std = {}

sut_rate = { 'microGIF': { 'bernoulli_nll':[], 'poisson_nll': [] } }  #, 'LIF_no_cell_types': [], 'GLIF_no_cell_types': [] }
sut_rate_std = { 'microGIF': { 'bernoulli_nll':[], 'poisson_nll': [] } }  #, 'LIF_no_cell_types': [], 'GLIF_no_cell_types': [] }
sut_loss = { 'microGIF': { 'bernoulli_nll':[], 'poisson_nll': [] } }  #, 'LIF_no_cell_types': [], 'GLIF_no_cell_types': [] }
sut_loss_std = { 'microGIF': { 'bernoulli_nll':[], 'poisson_nll': [] } }  #, 'LIF_no_cell_types': [], 'GLIF_no_cell_types': [] }
for exp_name in rate_per_exp.keys():
    target_rate = target_rates[ctr]
    ctr += 1
    cur_exp_res = rate_per_exp[exp_name]
    mean_rates = []; std_rates = []; xticks = []
    mean_losses = []; std_losses = []
    for model_type_name in cur_exp_res.keys():
        for lfn in cur_exp_res[model_type_name].keys():
            converged_indices = list(map(lambda x: not np.isnan(x) and (x < 10. * target_rate and x > 0.2 * target_rate), cur_exp_res[model_type_name][lfn]))
            cur_mean_rate = np.mean(np.asarray(cur_exp_res[model_type_name][lfn])[converged_indices])
            cur_std_rate = np.std(np.asarray(cur_exp_res[model_type_name][lfn])[converged_indices])
            res_rates['{}_{}_{}'.format(exp_name, model_type_name, lfn)] = np.mean(np.asarray(rate_per_exp[exp_name][model_type_name][lfn])[converged_indices])
            res_rates_std['{}_{}_{}'.format(exp_name, model_type_name, lfn)] = np.std(np.asarray(rate_per_exp[exp_name][model_type_name][lfn])[converged_indices])
            if np.isnan(cur_mean_rate):
                cur_mean_rate = 0.; cur_std_rate = 0.
            mean_rates.append(cur_mean_rate)
            std_rates.append(cur_std_rate)
            xticks.append('{},\n${}$'.format(model_type_name.replace('microGIF', 'miGIF').replace('mesoGIF', 'meGIF'),
                                             lfn.replace('poisson_nll', 'P_{NLL}').replace('bernoulli_nll', 'B_{NLL}')))

            mean_losses.append(np.mean(np.asarray(loss_per_exp[exp_name][model_type_name][lfn])[converged_indices]))
            std_losses.append(np.std(np.asarray(loss_per_exp[exp_name][model_type_name][lfn])[converged_indices]))
            res_loss['{}_{}_{}'.format(exp_name, model_type_name, lfn)] = np.mean(np.asarray(loss_per_exp[exp_name][model_type_name][lfn])[converged_indices])
            res_loss_std['{}_{}_{}'.format(exp_name, model_type_name, lfn)] = np.std(np.asarray(loss_per_exp[exp_name][model_type_name][lfn])[converged_indices])

            sut_rate[model_type_name][lfn].append(np.mean(np.asarray(rate_per_exp[exp_name][model_type_name][lfn])[converged_indices]))
            sut_rate_std[model_type_name][lfn].append(np.std(np.asarray(rate_per_exp[exp_name][model_type_name][lfn])[converged_indices]))
            sut_loss[model_type_name][lfn].append(np.mean(np.asarray(loss_per_exp[exp_name][model_type_name][lfn])[converged_indices]))
            sut_loss_std[model_type_name][lfn].append(np.std(np.asarray(loss_per_exp[exp_name][model_type_name][lfn])[converged_indices]))

    model_type_fname = 'export_rates_sleep_data_test_{}_{}_all.eps'.format(exp_name, exp_folder_name)
    plot.bar_plot_neuron_rates(np.asarray([target_rate]), np.asarray(mean_rates), 0., np.asarray(std_rates), plot_exp_type, 'all',
                               custom_legend=['Sleep data', 'Fitted models'],
                               fname=model_type_fname, xticks=xticks, custom_colors=['Red', 'Cyan'])
    plot.bar_plot(mean_losses, std_losses, labels=xticks, exp_type=plot_exp_type, uuid='all', fname=model_type_fname,
                  custom_colors=['Red', 'Cyan'], custom_legend=['Sleep data', 'Fitted models'])

sleep_exp_labels = list(map(lambda x: 'exp ' + str(sleep_exps.index(x)), sleep_exps))
plot.bar_plot_neuron_rates(sut_rate['microGIF']['bernoulli_nll'], sut_rate_std['microGIF']['bernoulli_nll'], sut_rate['microGIF']['poisson_nll'], sut_rate_std['microGIF']['poisson_nll'],
                           xticks=sleep_exp_labels, exp_type='export_sleep', uuid='export_sleep', fname='approx_rate_across_exp_{}.eps'.format('microGIF'),
                           custom_legend=['Bernoulli NLL', 'Poisson NLL'])

plot.bar_plot_neuron_rates(sut_rate['microGIF']['poisson_nll'], sut_rate_std['microGIF']['poisson_nll'], target_rates, target_rate_stds,
                           xticks=sleep_exp_labels, exp_type='export_sleep', uuid='export_sleep', fname='approx_rate_across_exp_{}_poisson_nll_vs_fitted.eps'.format('microGIF'),
                           custom_legend=['Fitted model', 'Target model'])
plot.bar_plot_neuron_rates(sut_rate['microGIF']['bernoulli_nll'], sut_rate_std['microGIF']['bernoulli_nll'], target_rates, target_rate_stds,
                           xticks=sleep_exp_labels, exp_type='export_sleep', uuid='export_sleep', fname='approx_rate_across_exp_{}_bernoulli_nll_vs_fitted.eps'.format('microGIF'),
                           custom_legend=['Fitted model', 'Target model'])

plot.bar_plot_neuron_rates(sut_loss['microGIF']['bernoulli_nll'], sut_loss_std['microGIF']['bernoulli_nll'], sut_loss['microGIF']['poisson_nll'], sut_loss_std['microGIF']['poisson_nll'],
                           xticks=sleep_exp_labels, exp_type='export_sleep', uuid='export_sleep', fname='approx_loss_across_exp_{}.eps'.format('microGIF'),
                           custom_legend=['Bernoulli NLL', 'Poisson NLL'], ylabel='Loss (NLL)')
# for v, k in enumerate(res_rates):
#     sleep_data_approx_rates = v
#     sleep_data_approx_rate_stds = res_rates_std[k]
#     plot.bar_plot(sleep_data_approx_rates, sleep_data_approx_rate_stds, sleep_exp_labels, exp_type='export_sleep', uuid='export_sleep', fname='approx_rate_per_exp_{}.eps'.format(k))

print(res_rates)
print(res_rates_std)
print(res_loss)
print(res_loss_std)

print(sut_rate)
print(sut_rate_std)
print(sut_loss)
print(sut_loss_std)

# sys.exit()
