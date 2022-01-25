import os

import numpy as np
import torch

import plot
import stats


def plot_stats_across_experiments(avg_statistics_per_exp):
    for m_i, m_k in enumerate(avg_statistics_per_exp):
        res_std_m = []; res_std_m_std = []
        res_std_t = []; res_std_t_std = []
        res_mu_m = []; res_mu_m_std = []
        res_mu_t = []; res_mu_t_std = []
        res_CV_m = []; res_CV_m_std = []
        res_CV_t = []; res_CV_t_std = []

        avg_diag_corrs = []
        avg_diag_corrs_std = []
        labels = []
        for o_i, o_k in enumerate(avg_statistics_per_exp[m_k]):
            avg_statistics_per_exp[m_k][o_k].pop('vrdfrda', None)
            avg_statistics_per_exp[m_k][o_k].pop('pnll', None)
            corr_avgs = []; corr_avgs_stds = []
            for lfn_i, lfn_k in enumerate(sorted(list(avg_statistics_per_exp[m_k][o_k].keys()))):
                for lr_i, lr_k in enumerate(sorted(list(avg_statistics_per_exp[m_k][o_k][lfn_k].keys()))):
                    # for lr_i, lr_k in enumerate(avg_statistics_per_exp[m_k][o_k][lfn_k]):
                    avg_stats_exps = avg_statistics_per_exp[m_k][o_k][lfn_k][lr_k]

                    # res_std_m.append(np.mean(avg_stats_exps['std_model']))
                    # res_std_t.append(np.mean(avg_stats_exps['std_target']))
                    # res_std_m_std.append(np.std(avg_stats_exps['std_model']))
                    # res_std_t_std.append(np.std(avg_stats_exps['std_target']))
                    res_mu_m.append(np.mean(avg_stats_exps['mu_model']))
                    res_mu_t.append(np.mean(avg_stats_exps['mu_target']))
                    res_mu_m_std.append(np.std(avg_stats_exps['mu_model']))
                    res_mu_t_std.append(np.std(avg_stats_exps['mu_target']))
                    res_CV_m.append(np.mean(avg_stats_exps['CV_model']))
                    res_CV_t.append(np.mean(avg_stats_exps['CV_target']))
                    res_CV_m_std.append(np.std(avg_stats_exps['CV_model']))
                    res_CV_t_std.append(np.std(avg_stats_exps['CV_target']))

                    # for c_i in range(len(avg_stats_exps['all_corrcoeff'])):
                    avg_corrcoeff = np.mean(avg_stats_exps['all_corrcoeff'])
                    avg_corrcoeff_std = np.std(avg_stats_exps['all_corrcoeff'])
                    # avg_diag_corr = (np.eye(12) * avg_corrcoeff).sum() / 12.
                    # avg_diag_corr_std = (np.eye(12) * avg_corrcoeff_std).sum() / 12.
                    # print('avg_diag_corr: {} for config ({}, {}, {})'.format(avg_diag_corr, m_k, o_k, lfn_k))
                    print('avg_diag_corr: {} for config ({}, {}, {})'.format(avg_corrcoeff, m_k, o_k, lfn_k))
                    # corr_avgs.append(avg_diag_corr)

                    # corr_avgs.append(avg_diag_corr)
                    # avg_diag_corrs.append(avg_diag_corr)
                    # avg_diag_corrs_std.append(avg_diag_corr_std)
                    avg_diag_corrs.append(avg_corrcoeff)
                    avg_diag_corrs_std.append(avg_corrcoeff_std)

                    labels.append(lfn_k.replace('frdvrda', '$d_A$').replace('frdvrd', '$d_C$').replace('frd', '$d_r$').replace('vrd', '$d_v$'))

        # i_mum, i_stdm, i_CVm, i_mut, i_stdt, i_CVt, i_mean_avg_corrcoeff = get_LIF_init_models_stats()
        # res_std_m.append(np.mean(i_stdm))
        # res_std_t.append(np.mean(i_stdt))
        # res_std_m_std.append(np.std(i_stdm))
        # res_std_t_std.append(np.std(i_stdt))
        # res_mu_m.append(np.mean(i_mum))
        # res_mu_t.append(np.mean(i_mut))
        # res_mu_m_std.append(np.std(i_mum))
        # res_mu_t_std.append(np.std(i_mut))
        # res_CV_m.append(np.mean(i_CVm))
        # res_CV_t.append(np.mean(i_CVt))
        # res_CV_m_std.append(np.std(i_CVm))
        # res_CV_t_std.append(np.std(i_CVt))
        # avg_diag_corrs.append(np.mean(i_mean_avg_corrcoeff))
        # avg_diag_corrs_std.append(np.std(i_mean_avg_corrcoeff))
        #
        # labels.append('init\nmodels')

        plot.bar_plot_pair_custom_labels_two_grps(y1=res_mu_m, y2=res_mu_t, y1_std=res_mu_m_std, y2_std=res_mu_t_std, labels=labels,
                                         exp_type='export', uuid='ho_stats' + '/' + custom_uuid, fname='bar_plot_avg_mu_across_exp_{}.eps'.format(m_k),
                                         title='Avg. spike count across experiments ({})'.format(m_k),
                                                  ylabel='Firing rate ($Hz$)',
                                                  legend=['$d_f$', '$\\rho+d_f$', '$d_V$'])
                                                  # legend= 1000 * ['test'])
        # plot.bar_plot_pair_custom_labels(y1=res_std_m, y2=res_std_t, y1_std=res_std_m_std, y2_std=res_std_t_std,
        #                                  labels=labels,
        #                                  exp_type='export', uuid=m_k, fname='bar_plot_avg_std_across_exp_{}.eps'.format(m_k),
        #                                  title='Avg. spike standard deviation across experiments ({})'.format(m_k))
        plot.bar_plot_pair_custom_labels_two_grps(y1=res_CV_m, y2=res_CV_t, y1_std=res_CV_m_std, y2_std=res_CV_t_std,
                                         labels=labels,
                                         exp_type='export', uuid='ho_stats' + '/' + custom_uuid, fname='bar_plot_avg_avg_CV_{}.eps'.format(m_k),
                                         title='Avg. CV for spike count across experiments ({})'.format(m_k),
                                                  ylabel='Coefficient of variation')

        print('m_k', m_k)
        # baseline = 0.202
        # baseline = 0.325
        baseline = 0.
        # if m_k is 'LIF':
        #     baseline = 0.202
        # elif m_k is 'GLIF':
        #     baseline = 0.325
        # for l_i in range(len(labels)):
        #     labels[l_i] = labels[l_i].replace('')
        # plot.bar_plot_crosscorrdiag(y1=avg_diag_corrs, y1_std=avg_diag_corrs_std, labels=labels,
        #                                  exp_type='export', uuid=m_k, fname='bar_plot_avg_diag_corrs_{}.eps'.format(m_k),
        #                                  title='Avg. diag. corrs. across experiments ({})'.format(m_k), baseline=baseline)
        lh = int(len(avg_diag_corrs)/2)
        if not (np.any(np.isnan(avg_diag_corrs)) or np.any(np.isnan(avg_diag_corrs_std))):
            if lh>0:
                plot.bar_plot_two_grps(y1=avg_diag_corrs[:lh], y1_std=avg_diag_corrs_std[:lh],
                                       y2=avg_diag_corrs[lh:], y2_std=avg_diag_corrs_std[lh:],
                                       labels=labels,
                                       exp_type='export', uuid='ho_stats' + '/' + custom_uuid, fname='bar_plot_avg_diag_corrs_{}.eps'.format(m_k),
                                       title='Avg. diag. corrs. across experiments ({})'.format(m_k), baseline=baseline,
                                       ylabel='Correlation coefficient')
            else:  # only one elem.
                plot.bar_plot_two_grps(y1=avg_diag_corrs, y1_std=avg_diag_corrs_std,
                                       y2=avg_diag_corrs, y2_std=avg_diag_corrs_std,
                                       labels=labels,
                                       exp_type='export', uuid='ho_stats' + '/' + custom_uuid,
                                       fname='bar_plot_avg_diag_corrs_{}.eps'.format(m_k),
                                       title='Avg. diag. corrs. across experiments ({})'.format(m_k), baseline=baseline,
                                       ylabel='Correlation coefficient')

# def main(argv):
# print('Argument List:', str(argv))

# experiments_path = '/Users/william/repos/archives_snn_inference/archive 13/saved/plot_data/'
# experiments_path = '/home/william/repos/archives_snn_inference/archive/saved/plot_data/'
# experiments_path = '/home/william/repos/archives_snn_inference/archive_1607/saved/plot_data/'
# experiments_path = '/media/william/p6/archive (8)/saved/'
# experiments_path = '/media/william/p6/archives_pre_0907/archive (5)/saved/plot_data/'
# experiments_path = '/media/william/p6/archive_0907/archive/saved/plot_data/'
# experiments_path = '/home/william/repos/archives_snn_inference/archive_2607/saved/plot_data/'
# experiments_path = '/home/william/repos/archives_snn_inference/archive_0908/archive/saved/plot_data/'
experiments_path = '/media/william/p6/archive_1009/archive/saved/plot_data/'

custom_uuid = 'data'
folders = os.listdir(experiments_path)
experiment_averages = {}
for folder_path in folders:
    # print(folder_path)

    full_folder_path = experiments_path + folder_path + '/'
    if not folder_path.__contains__('.DS_Store'):
        files = os.listdir(full_folder_path)
        id = folder_path.split('-')[-1]
    else:
        files = []
        id = 'None'
    plot_spiketrains_files = []
    plot_losses_files = []
    for f in files:
        if f.__contains__('plot_spiketrains_side_by_side'):
            plot_spiketrains_files.append(f)
        elif f.__contains__('plot_losses'):
            f_data = torch.load(full_folder_path + f)
            custom_title = f_data['plot_data']['custom_title']
            optimiser = custom_title.split(', ')[1].strip(' ')
            model_type = custom_title.split(',')[0].split('(')[-1]
            lr = custom_title.split(', ')[-1].strip(' =lr').strip(')')
            lfn = f_data['plot_data']['fname'].split('loss_fn_')[1].split('_tau')[0]
            # break
            plot_losses_files.append(f)

    # if len(plot_spiketrains_files) != 55 or model_type != 'LIF' or lfn not in ['frd', 'vrd', 'frdvrd', 'frdvrda']: # file mask
    # if model_type != 'LIF' or lfn not in ['frd', 'vrd', 'frdvrd', 'frdvrda']: # file mask
    if len(plot_losses_files) == 0:
        print("Incomplete exp.: No loss files.")
        # print(len(plot_spiketrains_files))
        pass
    # elif model_type != 'LIF_R' or lfn not in ['frd']: # file mask
    # elif lfn not in ['frd'] or model_type == 'GLIF' or model_type == 'LIF': # file mask
    # elif lfn not in ['frd']: # file mask
    #     # print("Incomplete exp. len should be 5 exp * 11 plots. was: {}".format(len(plot_spiketrains_files)))
    #     # print(len(plot_spiketrains_files))
    #     pass
    else:
        print('Processing final spike trains for configuration: {}, {}, {}'.format(model_type, optimiser, lfn))
        plot_spiketrains_files.sort()  # check that alphabetically

        if not experiment_averages.__contains__(model_type):
            experiment_averages[model_type] = {
                optimiser: {lfn: {lr: {'all_corrcoeff': [], 'mu_model': [], 'std_model': [],
                                       'mu_target': [], 'std_target': [],
                                       'CV_model': [], 'CV_target': []}}}}
        if not experiment_averages[model_type].__contains__(optimiser):
            experiment_averages[model_type][optimiser] = {}
        if not experiment_averages[model_type][optimiser].__contains__(lfn):
            # experiment_averages[model_type][optimiser][lfn] = {}
            experiment_averages[model_type][optimiser][lfn] = {lr: {'all_corrcoeff': [], 'mu_model': [],
                                                                      'std_model': [],
                                                                      'mu_target': [], 'std_target': [],
                                                                      'CV_model': [], 'CV_target': []}}
        if not experiment_averages[model_type][optimiser][lfn].__contains__(lr):
            experiment_averages[model_type][optimiser][lfn][lr] = {'corrcoeff': [], 'mu_model': [],
                                                                      'std_model': [],
                                                                      'mu_target': [], 'std_target': [],
                                                                      'CV_model': [], 'CV_target': []}

        # avg_rates_model = []; avg_rates_target = []
        # corrcoeff_sum = None; mum = []; mut = []; stdm = []; stdt = []; CVm = []; CVt = []
        # for exp_i in range(int(len(plot_spiketrains_files) / 21)):  # gen data for [0 + 11 * i]
        N_exp = 4
        for exp_i in range(N_exp):  # gen data for [0 + 11 * i]
            modulus_op = int(len(plot_spiketrains_files) / N_exp)
            print('exp_i: {}'.format(exp_i))
            cur_full_path = full_folder_path + plot_spiketrains_files[modulus_op * exp_i]

            data = torch.load(cur_full_path)
            plot_data = data['plot_data']
            model_spike_train = plot_data['model_spikes'].detach().numpy()
            target_spike_train = plot_data['target_spikes'].detach().numpy()
            N = model_spike_train[0].shape[0]

            # plot.plot_spiketrains_side_by_side(torch.tensor(model_spike_train), torch.tensor(target_spike_train),
            #                                    'export', model_type,
            #                                    title='Final spike trains {}, {}, {}'.format(model_type, optimiser, lfn),
            #                                    fname='spike_train_{}_{}_{}_exp_{}.eps'.format(model_type, optimiser, lfn, exp_i))

            corrcoeff, mu1, std1, mu2, std2, CV1, CV2 = stats.higher_order_stats(model_spike_train, target_spike_train, bin_size=100)

            if not np.isnan(corrcoeff[N:, :N]).any():
                avg_diag_corr = (np.eye(N) * corrcoeff[N:, :N]).sum() / float(N)
                # avg_diag_corr_std = (np.eye(12) * corrcoeff).sum() / 12.
                experiment_averages[model_type][optimiser][lfn][lr]['all_corrcoeff'].append(np.copy(avg_diag_corr))
            else:
                print('corrcoeff NaN for {}, {}, {}'.format(model_type, optimiser, lfn))

            cur_hyperconf = 'Correlation coefficient, {}, {}, {}'.format(model_type, optimiser, lfn)
            fname_prefix = model_type + '_' + optimiser + '_' + lfn

            id = cur_full_path.split('/')[-2]
            save_fname = '{}_{}_exp_num_{}.png'.format(fname_prefix, id, exp_i)

            # plot.heatmap_spike_train_correlations(corrcoeff[12:, :12], axes=['Fitted model', 'Target model'],
            #                                       exp_type='export', uuid=model_type+'/single_exp',
            #                                       fname='heatmap_bin_{}_{}.eps'.format(20, save_fname),
            #                                       bin_size=20, custom_title=cur_hyperconf)

            # if corrcoeff_sum is None:
                # corrcoeff_sum = np.zeros_like(corrcoeff) + corrcoeff
            # else:
            #     corrcoeff_sum = corrcoeff_sum + corrcoeff


            # mum.append(mu1)
            # mut.append(mu2)
            # stdm.append(std1)
            # stdt.append(std2)
            # CVm.append(CV1)
            # CVt.append(CV2)
            if not np.isnan(mu1).any():
                experiment_averages[model_type][optimiser][lfn][lr]['mu_model'].append(np.mean(mu1))
            # experiment_averages[model_type][optimiser][lfn]['std_model'].append(np.std(mum))
            if not np.isnan(mu2).any():
                experiment_averages[model_type][optimiser][lfn][lr]['mu_target'].append(mu2)
            # experiment_averages[model_type][optimiser][lfn]['std_target'].append(np.std(mut))
            if not np.isnan(CV1).any():
                experiment_averages[model_type][optimiser][lfn][lr]['CV_model'].append(CV1)
            if not np.isnan(CV2).any():
                experiment_averages[model_type][optimiser][lfn][lr]['CV_target'].append(CV2)

        # if corrcoeff_sum is not None:
        #     if exp_i > 2:
        #         avg_corrcoeff = (corrcoeff_sum / float(exp_i+1))[12:, :12]
        #     else:
        #         avg_corrcoeff = np.copy(corrcoeff_sum[12:, :12])
        #     # print('avg_corrcoeff: {}'.format(avg_corrcoeff))
        #     for i in range(avg_corrcoeff.shape[0]):
        #         for j in range(avg_corrcoeff.shape[1]):
        #             if np.isnan(avg_corrcoeff[i][j]):
        #                 avg_corrcoeff[i][j] = 0.
        #     cur_hyperconf = 'Average corrcoeff, {}, {}, {}'.format(model_type, optimiser, lfn)
            # plot.heatmap_spike_train_correlations(avg_corrcoeff, axes=['Fitted model', 'Target model'],
            #                                       exp_type=plot_data['exp_type'], uuid='export',
            #                                       fname='heatmap_bin_{}_avg_{}_exp_{}.eps'.format(20, fname_prefix.replace('.', ''), id),
            #                                       bin_size=20, custom_title=cur_hyperconf)

            # experiment_averages[model_type][optimiser][lfn]['all_corrcoeff'].append(np.copy(avg_corrcoeff))

        # plot.bar_plot_pair_custom_labels(y1=mum, y2=mut,
        #                                  y1_std=stdm,
        #                                  y2_std=stdt,
        #                                  labels=False,
        #                                  exp_type='export', uuid=model_type,
        #                                  fname='bar_plot_avg_avg_{}.eps'.format(
        #                                      model_type + '_' + optimiser + '_' + lfn).replace('.', ''),
        #                                  title='Average spike count within experiment', xlabel='Random seed')
        # plot.bar_plot_pair_custom_labels(y1=CVm, y2=CVt,
        #                                  y1_std=np.std(CVm),
        #                                  y2_std=np.std(CVt),
        #                                  labels=False,
        #                                  exp_type='export', uuid=model_type, fname='bar_plot_avg_avg_CV_{}.eps'.format(model_type + '_' + optimiser + '_' + lfn).replace('.', ''),
        #                                  title='Avg. CV for spike count within experiment', xlabel='Random seed')


        # cur_std_model, cur_rate_model = stats.binned_avg_firing_rate_per_neuron(model_spike_train, bin_size=400)
        # cur_std_target, cur_rate_target = stats.binned_avg_firing_rate_per_neuron(target_spike_train, bin_size=400)

plot_stats_across_experiments(experiment_averages)

# if __name__ == "__main__":
#     main(sys.argv[1:])
