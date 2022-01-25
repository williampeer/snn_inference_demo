import os

import numpy as np
import torch
from matplotlib import pyplot as plt

import plot
from TargetModels import TargetModels
from experiments import draw_from_uniform, generate_synthetic_data
from spike_metrics import euclid_dist

load_paths = []
# load_paths.append('/home/william/repos/archives_snn_inference/archive_0908/archive/saved/')
# load_paths.append('/home/william/repos/archives_snn_inference/archive_0208_LIF_R/archive/saved/')
# load_paths.append('/home/william/repos/archives_snn_inference/archive_1108_full_some_diverged/archive/saved/')
# load_paths.append('/home/william/repos/archives_snn_inference/archive_1208_GLIF_3_LIF_R_AND_ASC_10_PLUSPLUS/archive/saved/')
# load_paths.append('/home/william/repos/archives_snn_inference/archive_3008_all_seed_64_and_sbi_3_and_4/archive/saved/')
# load_paths.append('/media/william/p6/archive_1009/archive/saved/')
# load_paths.append('/home/william/repos/archives_snn_inference/archive_1309_last_SBI/archive/saved/')
# load_paths.append('/home/william/repos/archives_snn_inference/archive_1509_new_runs/archive/saved/')
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


def export_rates_models(model, tar_model, descriptor, legend=['Fitted', 'Target']):
    model_spike_train, _ = generate_synthetic_data(model, poisson_rate=10., t=10000)
    model_spike_rate = 1000. * model_spike_train.sum(dim=0) / model_spike_train.shape[0]

    tar_spike_train, _ = generate_synthetic_data(tar_model, poisson_rate=10., t=10000)
    tar_spike_rate = 1000. * tar_spike_train.sum(dim=0) / tar_spike_train.shape[0]

    custom_uuid = 'GBO'
    plt.figure()
    plot.bar_plot_pair_custom_labels(y1=model_spike_rate.numpy(), y2=tar_spike_rate.numpy(),
                                     y1_std=np.zeros_like(model_spike_rate), y2_std=np.zeros_like(tar_spike_rate),
                                     labels=range(model_spike_train.shape[1]),
                                     exp_type='export', uuid='ho_stats' + '/' + custom_uuid,
                                     fname='export_bar_plot_avg_rate_GBO_{}_{}.eps'.format(model.__class__.__name__, descriptor),
                                     title='Avg. rates for GBO parameters ({})'.format(model.__class__.__name__),
                                     ylabel='Firing rate ($Hz$)', xlabel='Neuron',
                                     legend=legend)
    plt.close()
    model_spike_train = None; tar_spike_train = None
    return model_spike_rate, tar_spike_rate


def plot_param_dist_combined(model, init_model, tar_model, fname, lr):
    # dist_per_param = parameter_distance.euclid_dist(model.get_parameters(), tar_model.get_parameters())
    p_init = init_model.get_parameters(); p_model = model.get_parameters(); p_tar = tar_model.get_parameters()
    dist_per_param_fitted = []
    dist_per_param_init = []
    for p_i in range(len(p_model)):
        dist_per_param_fitted.append(np.sqrt(np.power((p_tar[p_i] - p_model[p_i]), 2).sum()) / len(p_model))
        dist_per_param_init.append(np.sqrt(np.power((p_tar[p_i] - p_init[p_i]), 2).sum()) / len(p_init))

    dist_per_param_fitted = np.array(dist_per_param_fitted)
    dist_per_param_init = np.array(dist_per_param_init)

    plot.bar_plot_pair_custom_labels(dist_per_param_fitted, dist_per_param_init,
                                     np.zeros_like(dist_per_param_fitted), np.zeros_like(dist_per_param_init),
                                     model.__class__.parameter_names,
                                     'export', 'param_dist_per_exp', fname=fname, legend=['Fitted', 'Initial'],
                                     title='Parameter distance GBO, {}, $\\alpha={}$'.format(model.__class__.__name__, lr),
                                     ylabel='Distance', xlabel='Parameter $p$')

    return dist_per_param_fitted, dist_per_param_init


def plot_param_dist_across_exp_combined(fitted_param_dists, init_param_dists, model_class, fname, lr, title=False):
    if not title:
        title = 'Parameter distance exp avg GBO, {}, $\\alpha={}$'.format(model_class.__name__, lr)
    plot.bar_plot_pair_custom_labels(np.mean(fitted_param_dists, axis=0), np.mean(init_param_dists, axis=0),
                                     np.std(fitted_param_dists, axis=0), np.std(init_param_dists, axis=0),
                                     model_class.parameter_names,
                                     'export', 'param_dist_per_exp', fname=fname, legend=['Fitted', 'Initial'],
                                     title=title, ylabel='Distance', xlabel='Parameter $p$')


def plot_param_dist(model, tar_model, fname, lr):
    # dist_per_param = parameter_distance.euclid_dist(model.get_parameters(), tar_model.get_parameters())
    p1 = model.get_parameters(); p2 = tar_model.get_parameters()
    dist_per_param = []
    for p_i in range(len(p1)):
        dist_per_param.append(np.sqrt(np.power((p1[p_i] - p2[p_i]), 2).sum()) / len(p1))

    dist_per_param = np.array(dist_per_param)
    plot.bar_plot(dist_per_param, np.zeros_like(dist_per_param), model.__class__.parameter_names,
                  'export', 'param_dist_per_exp',
                  fname=fname,
                  title='Parameter distance, GBO {}, $\\alpha={}$'.format(model.__class__.__name__, lr))

    return dist_per_param


results_dict = {}
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
        test_losses = []
        run_converged = []
        exp_nums = []
        for f in plot_files:
            print(f)
            if f.__contains__('plot_spiketrains_side_by_side'):
                plot_spiketrains_files.append(f)
                print('appended {}'.format(f))
            elif f.__contains__('plot_losses'):
                exp_plot_data = torch.load(path_plot_data + f)
                custom_title = exp_plot_data['plot_data']['custom_title']
                optimiser = custom_title.split(', ')[1].strip(' ')
                model_type = custom_title.split(',')[0].split('(')[-1]
                lr = custom_title.split(', ')[-2].strip(' =lr').strip(')')
                lfn = exp_plot_data['plot_data']['fname'].split('loss_fn_')[1].split('_tau')[0]
                cur_test_losses = exp_plot_data['plot_data']['test_loss']
                test_losses.append(cur_test_losses)
            if f.__contains__('exp_num'):
                exp_num = int(f.split('_exp_num_')[1].split('_')[0])
                print('exp_num: {}'.format(exp_num))
                exp_nums.append(exp_num)

        if model_type is None:
            print('exp did not converge.')
            pass
        elif len(plot_spiketrains_files) == 0 or model_type in ['LIF', 'LIF_no_grad'] or lfn != 'FIRING_RATE_DIST':
            print('Skipping: {}, {}, {}. #spike_train_files {}'.format(model_type, lfn, optimiser, len(plot_spiketrains_files)))
            pass
        else:
            print('Success! Processing exp: {}'.format(folder_path))

            if not path_models.__contains__('.DS_Store'):
                files = os.listdir(path_models)
                id = folder_path.split('-')[-1]
            else:
                files = []
                id = 'None'

            start_seed = 42; N_exp = 3
            # start_seed = exp_nums[0]; N_exp = len(exp_nums)
            print('start_seed: {}, N_exp: {}'.format(start_seed, N_exp))
            model_class = None

            f_ctr = 0
            model_rates = []
            init_model_rates = []
            tar_rates = []
            param_dist_per_param_fitted = []
            param_dist_per_param_init = []
            for f in files:
                cur_exp_num = f_ctr % N_exp  # shouldn't be needed?
                if(f_ctr > cur_exp_num):
                    print('WARNING: f_ctr > exp_num: {} > {}'.format(f_ctr, cur_exp_num))

                exp_res = torch.load(path_models + f)
                model = exp_res['model']
                poisson_rate = exp_res['rate']
                print('Loaded model data.')

                non_overlapping_offset = start_seed + N_exp + 1
                torch.manual_seed(non_overlapping_offset + cur_exp_num)
                # torch.manual_seed(non_overlapping_offset)
                np.random.seed(non_overlapping_offset + cur_exp_num)
                init_params_model = draw_from_uniform(model.__class__.parameter_init_intervals, model.N)
                model_class = model.__class__
                programmatic_neuron_types = torch.ones((model.N,))
                for n_i in range(int(2 * model.N / 3), model.N):
                    programmatic_neuron_types[n_i] = -1
                neuron_types = programmatic_neuron_types
                init_model = model.__class__(N=model.N, parameters=init_params_model, neuron_types=neuron_types)

                cur_tar_seed = 3 + f_ctr % N_exp
                tar_model = get_target_model_for(model, cur_tar_seed)

                descriptor = folder_path + 'exp_num_{}'.format(cur_exp_num)
                init_model_rate, tar_model_rate = export_rates_models(init_model, tar_model, descriptor, legend=['Initial', 'Target'])
                model_rate, _ = export_rates_models(model, tar_model, descriptor, legend=['Fitted', 'Target'])
                converged = euclid_dist(init_model_rate, tar_model_rate) > euclid_dist(model_rate, tar_model_rate)

                if converged:
                    fname_combined = 'export_param_dist_converged_combined_{}_exp_{}.png'.format(folder_path, cur_exp_num)
                    dist_per_param_fitted, dist_per_param_init = plot_param_dist_combined(model, init_model, tar_model, fname=fname_combined, lr=lr)
                    param_dist_per_param_fitted.append(dist_per_param_fitted)
                    param_dist_per_param_init.append(dist_per_param_init)
                    init_model_rates.append(init_model_rate.numpy())
                    model_rates.append(model_rate.numpy())
                    tar_rates.append(tar_model_rate.numpy())

                f_ctr += 1

                model.reset(); tar_model.reset()
                model = None; tar_model = None; model_rate = None; tar_model_rate = None; init_model_rate = None

            fname_exp = 'export_param_dist_converged_exp_avg_all_exp_{}.png'.format(folder_path)
            plot_param_dist_across_exp_combined(param_dist_per_param_fitted, param_dist_per_param_init, model_class, fname_exp, lr=lr)

            descriptor = folder_path + '_converged'
            plot.bar_plot_pair_custom_labels(y1=np.mean(model_rates, axis=0), y2=np.mean(tar_rates, axis=0),
                                             y1_std=np.std(model_rates, axis=0), y2_std=np.std(tar_rates, axis=0),
                                             labels=range(len(model_rates[0])),
                                             exp_type='export', uuid='ho_stats' + '/' + 'GBO',
                                             fname='export_bar_plot_converged_avg_rate_GBO_{}_{}.eps'.format(model_type, descriptor),
                                             title='Avg. rates over ({}) converged runs ({})'.format(
                                                 len(model_rates), model_type),
                                             ylabel='Firing rate ($Hz$)', xlabel='Neuron',
                                             legend=['Fitted', 'Target'])
            plot.bar_plot_pair_custom_labels(y1=np.mean(init_model_rates, axis=0), y2=np.mean(tar_rates, axis=0),
                                             y1_std=np.std(init_model_rates, axis=0), y2_std=np.std(tar_rates, axis=0),
                                             labels=range(len(init_model_rates[0])),
                                             exp_type='export', uuid='ho_stats' + '/' + 'GBO',
                                             fname='export_bar_plot_converged_init_avg_rate_GBO_{}_{}.eps'.format(model_type, descriptor),
                                             title='Avg. rates initial models over ({}) converged runs ({})'.format(
                                                 len(model_rates), model_type),
                                             ylabel='Firing rate ($Hz$)', xlabel='Neuron',
                                             legend=['Initial', 'Target'])

            config_key = '{}_{}'.format(model_type, lr)  # optim, lfn const.
            if results_dict.keys().__contains__(config_key):
                for p_i in range(len(param_dist_per_param_fitted)):
                    results_dict[config_key]['fitted'].append(param_dist_per_param_fitted[p_i])
                    results_dict[config_key]['init'].append(param_dist_per_param_init[p_i])
                    results_dict[config_key]['model_class'] = model_class
            else:
                results_dict[config_key] = { 'fitted' : param_dist_per_param_fitted,
                                             'init' : param_dist_per_param_init,
                                             'model_class' : model_class }


for k_i, k_v in enumerate(results_dict):
    fname_k_v = 'export_param_dist_converged_exp_avg_all_config_{}.png'.format(k_v)
    model_class = results_dict[k_v]['model_class']
    plot_param_dist_across_exp_combined(results_dict[k_v]['fitted'], results_dict[k_v]['init'], model_class, fname=fname_k_v, lr=lr)

# plot_stats_across_experiments(avg_statistics_per_exp=experiment_averages, archive_name='all')

    # if __name__ == "__main__":
    #     main(sys.argv[1:])
