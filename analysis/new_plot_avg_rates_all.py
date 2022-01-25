import os

import numpy as np
import torch

import plot
import stats
from TargetModels import TargetModels
from experiments import sine_modulated_white_noise_input
from model_util import generate_model_data

colours = ['Green', 'Red']


def plot_stats_across_experiments(avg_statistics_per_exp, archive_name):
    for m_i, m_k in enumerate(avg_statistics_per_exp):
        # flat arrays per model type
        labels = []
        avg_model_rates = []
        avg_target_rates = []
        avg_model_rates_std = []
        avg_target_rates_std = []
        for lfn_i, lfn_k in enumerate(avg_statistics_per_exp[m_k]):
            for lr_i, lr_k in enumerate(avg_statistics_per_exp[m_k][lfn_k]):
                avg_stats_exps = avg_statistics_per_exp[m_k][lfn_k][lr_k]
                print('processing: {}'.format(avg_stats_exps))

                labels.append(lfn_k)
                # labels.append('{}\ninit\nmodels'.format(m_k))

                cur_model_mean = np.mean(avg_stats_exps['avg_model_rate'])
                cur_model_std = np.std(avg_stats_exps['avg_model_rate'])
                cur_target_mean = np.mean(avg_stats_exps['avg_target_rate'])
                cur_target_std = np.std(avg_stats_exps['avg_target_rate'])

                avg_model_rates.append(cur_model_mean)
                avg_target_rates.append(cur_target_mean)
                avg_model_rates_std.append(cur_model_std)
                avg_target_rates_std.append(cur_target_std)

                print('plotting for {}, {}, {}'.format(m_k, lfn_k, lr_k))
                plot.bar_plot(y=cur_model_mean, y_std=cur_model_std,
                              labels=labels, exp_type='export',
                              uuid=m_k, fname='rate_bar_plot_model_avg_rate_within_exp_{}_{}_{}_{}'.format(m_k, lfn_k, lr_k.replace('.', '_'), archive_name),
                              # title='Model rates across GBO experiments ({}, {}, {})'.format(m_k, lfn_k, lr_k),
                              title='',
                              ylabel='Firing rate ($Hz$)', xlabel='$\\alpha$', custom_colors=colours)
                plot.bar_plot_pair_custom_labels(y1=cur_model_mean, y2=cur_target_mean,
                                                 y1_std=cur_model_std, y2_std=cur_target_std,
                                                 labels=labels,
                                                 exp_type='export', uuid=m_k, fname='rate_bar_plot_pair_avg_rate_within_exp_{}_{}_{}_{}'
                                                 .format(m_k, lfn_k, lr_k.replace('.', '_'), archive_name),
                                                 # title='Model rates across GBO experiments ({}, {}, {})'.format(m_k, lfn_k, lr_k),
                                                 title='',
                                                 ylabel='Firing rate ($Hz$)', xlabel='$f_{dist}$', colours=colours)

                plot.bar_plot_pair_custom_labels(y1=cur_model_std/cur_model_mean,
                                                 y2=cur_target_std/cur_target_mean,
                                                 y1_std=np.zeros_like(cur_model_std),
                                                 y2_std=np.zeros_like(cur_target_std),
                                                 labels=labels,
                                                 exp_type='export', uuid=m_k, fname='rate_bar_plot_pair_avg_rate_CV_within_exp_{}_{}_{}.png'.format(m_k, lfn_k, lr_k.replace('.', '_')),
                                                 # title='Avg. CV for rates across GBO experiments ({}, {}, {})'.format(m_k, lfn_k, lr_k),
                                                 title='',
                                                 ylabel='Rate CV', colours=['Purple', 'Yellow'])

        plot.bar_plot(y=avg_model_rates, y_std=avg_model_rates_std,
                      labels=labels, exp_type='export',
                      uuid=m_k, fname='rate_bar_plot_model_avg_rate_across_exp_{}_{}'.format(m_k, archive_name),
                      # title='Model rates across GBO experiments ({})'.format(m_k),
                      title='',
                      ylabel='Firing rate ($Hz$)', xlabel='$\\alpha$', custom_colors=colours)
        plot.bar_plot_pair_custom_labels(y1=avg_model_rates, y2=avg_target_rates,
                                         y1_std=avg_model_rates_std, y2_std=avg_target_rates_std,
                                         labels=labels,
                                         exp_type='export', uuid=m_k,
                                         fname='rate_bar_plot_pair_avg_rate_across_exp_{}_{}'.format(m_k, archive_name),
                                         # title='Model rates across GBO experiments ({})'.format(m_k),
                                         title='',
                                         ylabel='Firing rate ($Hz$)', xlabel='$f_{dist}$', colours=colours)

        plot.bar_plot_pair_custom_labels(y1=np.array(avg_model_rates_std) / np.array(avg_model_rates),
                                         y2=np.array(avg_target_rates_std) / np.array(avg_target_rates),
                                         y1_std=np.zeros_like(cur_model_std),
                                         y2_std=np.zeros_like(cur_target_std),
                                         labels=labels,
                                         exp_type='export', uuid=m_k,
                                         fname='rate_bar_plot_pair_avg_rate_CV_across_exp_{}.png'.format(m_k),
                                         # title='Avg. CV for rates across GBO experiments ({})'.format(m_k),
                                         title='',
                                         ylabel='Rate CV', colours=['Purple', 'Yellow'])

load_paths = []
# load_paths.append('/home/william/repos/archives_snn_inference/archive_0908/archive/saved/')
# load_paths.append('/home/william/repos/archives_snn_inference/archive_0208_LIF_R/archive/saved/')
# load_paths.append('/home/william/repos/archives_snn_inference/archive_1108_full_some_diverged/archive/saved/')
# load_paths.append('/home/william/repos/archives_snn_inference/archive_1208_GLIF_3_LIF_R_AND_ASC_10_PLUSPLUS/archive/saved/')
# load_paths.append('/home/william/repos/archives_snn_inference/archive_3008_all_seed_64_and_sbi_3_and_4/archive/saved/')
# load_paths.append('/home/william/repos/archives_snn_inference/archive_partial_0109/archive/saved/')
# load_paths.append('/media/william/p6/archive_1009/archive/saved/')
# load_paths.append('/media/william/p6/archive_1109/archive/saved/')
# load_paths.append('/home/william/repos/archives_snn_inference/archive_1309_last_SBI/archive/saved/')
# load_paths.append('/home/william/repos/archives_snn_inference/archive_1509_new_runs/archive/saved/')
# load_paths.append('/home/william/repos/archives_snn_inference/archive_1609/archive/saved/')
# load_paths.append('/home/william/repos/archives_snn_inference/archive_1809_q/archive/saved/')
# load_paths.append('/home/william/repos/archives_snn_inference/archive_2009_tmp/archive/saved/')
load_paths.append('/home/william/repos/archives_snn_inference/archive_osx_2009/archive/saved/')

experiment_averages = {}


def get_target_model_for(model, cur_tar_seed):
    tar_model_fn_lookup = {'LIF': TargetModels.lif_continuous_ensembles_model_dales_compliant,
                           'LIF_R': TargetModels.lif_r_continuous_ensembles_model_dales_compliant,
                           'LIF_R_ASC': TargetModels.lif_r_asc_continuous_ensembles_model_dales_compliant,
                           'GLIF': TargetModels.glif_continuous_ensembles_model_dales_compliant}
    tar_model = tar_model_fn_lookup[model.name()](cur_tar_seed, model.N)
    return tar_model


for experiments_path in load_paths:
    archive_name = experiments_path.split('archives_snn_inference/')[-1].split('/')[0]
    folders = os.listdir(experiments_path)
    for folder_path in folders:
        print('folder: {}'.format(folder_path))

        path_models = experiments_path + folder_path + '/'
        path_plot_data = experiments_path + 'plot_data/' + folder_path + '/'
        if not path_plot_data.__contains__('.DS_Store') and not folder_path.__contains__('data'):
            plot_files = os.listdir(path_plot_data)
            id = folder_path.split('-')[-1]
        else:
            plot_files = []
            id = 'None'
        plot_spiketrains_files = []
        model_type = None
        for f in plot_files:
            # print(f)
            if f.__contains__('plot_spiketrains_side_by_side'):
                plot_spiketrains_files.append(f)
                # print('appended {}'.format(f))
            elif f.__contains__('plot_losses'):
                exp_plot_data = torch.load(path_plot_data + f)
                custom_title = exp_plot_data['plot_data']['custom_title']
                optimiser = custom_title.split(', ')[1].strip(' ')
                model_type = custom_title.split(',')[0].split('(')[-1]
                lr = custom_title.split(', ')[-2].strip(' =lr').strip(')').replace('.', '_')
                lfn = exp_plot_data['plot_data']['fname'].split('loss_fn_')[1].split('_tau')[0]
                # break
                print('plot_losses')

        if model_type is None:
            print('exp did not converge.')
            pass
        # if len(plot_spiketrains_files) != 21 * 3 or model_type in ['LIF', 'LIF_no_grad']:  # file mask
        elif len(plot_spiketrains_files) == 0 or model_type in ['LIF', 'LIF_no_grad']:  # or optimiser == 'SGD':  # file mask
            print('Skipping: {}, {}, {}. #spike_train_files {}'.format(model_type, lfn, optimiser, len(plot_spiketrains_files)))
            # print("Incomplete exp. len should be 5 exp * 11 plots. was: {}".format(len(plot_spiketrains_files)))
            pass
        else:
            print('Success! Processing exp: {}'.format(folder_path))

            if not path_models.__contains__('.DS_Store'):
                files = os.listdir(path_models)
                id = folder_path.split('-')[-1]
            else:
                files = []
                id = 'None'

            if not experiment_averages.__contains__(model_type):
                experiment_averages[model_type] = { lfn: {
                    lr: { 'avg_model_rate': [], 'stds_model_rates' : [],
                          'avg_target_rate': [], 'stds_target_rates' : [] } }
                }
            if not experiment_averages[model_type].__contains__(lfn):
                experiment_averages[model_type][lfn] = { lr: {'avg_model_rate': [], 'stds_model_rates' : [],
                                                       'avg_target_rate': [], 'stds_target_rates' : []} }
            if not experiment_averages[model_type][lfn].__contains__(lr):
                experiment_averages[model_type][lfn][lr] = { 'avg_model_rate': [], 'stds_model_rates' : [],
                                                       'avg_target_rate': [], 'stds_target_rates' : [] }

            f_ctr = 0
            mean_model_rates = []
            mean_tar_rates = []
            for f in files:
                # import, simulate, plot
                exp_res = torch.load(path_models + f)
                model = exp_res['model']
                poisson_rate = exp_res['rate']
                print('Loaded model data.')

                inputs = sine_modulated_white_noise_input(rate=10., t=10000, N=model.N)
                model_spike_train = generate_model_data(model, inputs).clone().detach()
                cur_neuronal_rates = stats.rate_Hz(model_spike_train)
                mean_model_rates.append(np.mean(cur_neuronal_rates.numpy()))

                cur_tar_seed = 3 + f_ctr % 3
                tar_model = get_target_model_for(model, cur_tar_seed)
                tar_inputs = sine_modulated_white_noise_input(rate=10., t=10000, N=model.N)
                tar_spike_train = generate_model_data(model, tar_inputs).clone().detach()
                cur_tar_rates = stats.rate_Hz(tar_spike_train)
                mean_tar_rates.append(np.mean(cur_tar_rates.numpy()))

                f_ctr += 1

            experiment_averages[model_type][lfn][lr]['avg_model_rate'].append(np.mean(mean_model_rates))
            experiment_averages[model_type][lfn][lr]['stds_model_rates'].append(np.std(mean_model_rates))
            experiment_averages[model_type][lfn][lr]['avg_target_rate'].append(np.mean(mean_tar_rates))
            experiment_averages[model_type][lfn][lr]['stds_target_rates'].append(np.std(mean_tar_rates))

    # plot_stats_across_experiments(avg_statistics_per_exp=experiment_averages, archive_name=archive_name)

plot_stats_across_experiments(avg_statistics_per_exp=experiment_averages, archive_name='all')

    # if __name__ == "__main__":
    #     main(sys.argv[1:])
