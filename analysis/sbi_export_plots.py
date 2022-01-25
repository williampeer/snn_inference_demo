import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

import plot
from Models.no_grad.GLIF_soft_no_grad import GLIF_soft_no_grad
from analysis import parameter_distance
from analysis.sbi_import_export_spikes import convert_posterior_to_model_params_dict
from dev_sbi_main_multi import get_binned_spike_counts, export_plots
from experiments import generate_synthetic_data


def export_stats_model_target(model, observation, descriptor):
    # spike_rates = 1000. * spike_train.sum(dim=0) / spike_train.shape[0]
    # for spike_iters in range(10-1):
    #     spike_train, _ = generate_synthetic_data(model, poisson_rate=10., t=10000)
    #     spike_rates = torch.cat([spike_rates, 1000. * spike_train.sum(dim=0) / spike_train.shape[0]])
    n_samples = 10
    spike_counts_per_sample = None
    spike_count_list = []
    for spike_iters in range(n_samples-1):
        spike_train, _ = generate_synthetic_data(model, t=10000)
        cur_cur_spike_count = torch.reshape(get_binned_spike_counts(spike_train.clone().detach()), (-1,))
        if spike_counts_per_sample is None:
            spike_counts_per_sample = cur_cur_spike_count
        else:
            # spike_counts_per_sample = torch.vstack((spike_counts_per_sample, cur_cur_spike_count))
            spike_counts_per_sample = spike_counts_per_sample + cur_cur_spike_count
        spike_count_list.append(cur_cur_spike_count)
    mean_model_spike_counts = spike_counts_per_sample / n_samples
    std_model_spike_counts = torch.zeros_like(mean_model_spike_counts)
    for s_i in range(len(spike_count_list)):
        std_model_spike_counts = std_model_spike_counts + torch.pow((spike_count_list[s_i] - mean_model_spike_counts), 2)
    std_model_spike_counts = torch.sqrt(std_model_spike_counts / (len(spike_count_list)-1))

    # spike_rates = torch.reshape(spike_rates, (-1, model.N))
    # mean_spike_rates = torch.mean(spike_rates, dim=0)
    # rate_stds = torch.std(spike_rates, dim=0)

    custom_uuid = 'sbi'
    plt.figure()
    # reshaped_tar = torch.reshape(observation, (-1, model.N))
    assert len(mean_model_spike_counts) == len(observation), \
        "mean_model_spike_counts ({}) should be same len as observation ({}).\n{}\n{}".\
            format(len(mean_model_spike_counts), len(observation), mean_model_spike_counts, observation)
    print('INFO; mean model spike counts and observation to follow:\n{}\n{}'.format(mean_model_spike_counts, observation))
    plot.bar_plot_pair_custom_labels(y1=mean_model_spike_counts, y2=observation,
                                     y1_std=std_model_spike_counts, y2_std=np.zeros_like(observation),
                                     labels=range(len(mean_model_spike_counts)),
                                     exp_type='export', uuid='ho_stats' + '/' + custom_uuid,
                                     fname='export_bar_plot_spike_count_sbi_{}.eps'.format(descriptor),
                                     title='Binned spike counts for SBI parameters ({})'.format(descriptor),
                                     ylabel='Binned spike count', xlabel='Neuron',
                                     legend=['Fitted', 'Target'])
    plt.close()

    # correlation:
    # Not too informative, unless the same input is used. However, correlation between neurons within model may be informative about that model, but so is NMF.

    return mean_model_spike_counts


def export_stats_top_samples(mean_model_spike_counts, std_model_spike_counts, targets, descriptor, N_samples=20):
    plt.figure()
    plot.bar_plot_pair_custom_labels(y1=mean_model_spike_counts, y2=targets,
                                     y1_std=std_model_spike_counts, y2_std=torch.zeros_like(std_model_spike_counts),
                                     labels=range(len(mean_model_spike_counts)),
                                     exp_type='export', uuid='ho_stats' + '/' + 'sbi',
                                     fname='export_bar_plot_avg_rate_sbi_{}.eps'.format(descriptor),
                                     title='Avg. spike counts {} most likely samples ({})'.format(N_samples, descriptor),
                                     ylabel='Binned spike count', xlabel='Neuron',
                                     legend=['Fitted', 'Target'])
    plt.close()


def limits_for_class(model_class, N):
    # parsed_weights = torch.zeros((N ** 2 - N,))
    limits_low = torch.zeros((N ** 2 - N,))
    limits_high = torch.ones((N ** 2 - N,))

    for i in range(1, len(model_class.parameter_names)):
        limits_low = torch.hstack((limits_low, torch.ones((N,)) * model_class.param_lin_constraints[i][0]))
        limits_high = torch.hstack((limits_high, torch.ones((N,)) * model_class.param_lin_constraints[i][1]))

    return limits_low, limits_high


def plot_param_dist(parameter_distance, title, fname):
    plot.bar_plot(parameter_distance, parameter_distance, False, 'export',
                  'sbi_param_dist', 'export_sbi_param_dist_{}.png'.format(fname), title)


def main():
    # experiments_path = '/media/william/p6/archive_3008_all_seed_64_and_sbi_3_and_4/archive/saved/data/'
    # experiments_path = '/home/william/repos/archives_snn_inference/archive_SINE_mod_input_2409/archive/saved/data/'
    experiments_path = '/home/william/repos/archives_snn_inference/archive_2709/saved/data/'
    # experiments_path = '/media/william/p6/archives_snn_inference/PLACEHOLDER/saved/'
    # experiments_path = '/home/william/repos/snn_inference/saved/data/'

    custom_uuid = 'data'
    files_sbi_res = os.listdir(experiments_path + 'sbi_res/')

    for sbi_res_file in files_sbi_res:
        plt.close('all ')
        print(sbi_res_file)

        sbi_res_path = experiments_path + 'sbi_res/' + sbi_res_file
        print('Loading: {}'.format(sbi_res_path))

        tar_seed = int(sbi_res_file.split('tar_seed_')[-1].strip('.pt'))
        print('tar seed: {}'.format(tar_seed))

        res_load = torch.load(sbi_res_path)
        print('sbi_res load successful.')
        sbi_res = res_load['data']
        if sbi_res.keys().__contains__('SNPE'):
            method = 'SNPE'
        elif sbi_res.keys().__contains__('SNLE'):
            method = 'SNLE'
        elif sbi_res.keys().__contains__('SNRE'):
            method = 'SNRE'
        else:
            raise NotImplementedError("Another method than SNPE,LE,RE")
        posterior = sbi_res[method]
        # if sbi_res.keys()
        model_class = sbi_res['model_class']
        m_name = model_class.__name__
        tar_model_class = GLIF_soft_no_grad
        N = sbi_res['N']
        dt_descriptor = sbi_res['dt_descriptor']
        if 'param_num' in sbi_res:
            param_num = sbi_res['param_num']
            corresponding_samples_fname = 'samples_method_{}_m_name_{}_param_num_{}_dt_{}_tar_seed_{}.pt'.format(method, m_name, param_num, dt_descriptor, tar_seed)
            print('single param. passing for now..')
            pass
        else:
            pre_path = experiments_path + 'sbi_samples/'
            corresponding_samples_fname = 'samples_method_{}_m_name_{}_dt_{}_tar_seed_{}.pt'\
                .format(method, m_name.strip('_no_grad').strip('_lower_dim').strip('_soft'), dt_descriptor, tar_seed)
            full_data_path = pre_path + corresponding_samples_fname
            if os.path.exists(pre_path + corresponding_samples_fname):
                # try:
                data_arr = torch.load(full_data_path)['data']
                print('sbi_samples load successful.')
                samples = data_arr['samples']
                # observation = data_arr['observation'][:,0]
                observation = data_arr['observation']
                points = data_arr['tar_parameters']
                m_name = data_arr['m_name']

                lim_low, lim_high = limits_for_class(model_class, N=N)

                # -------------------------------------------------------
                # print('DEBUG VERBOSE EXTRA PLOTTING')
                # log_probability = posterior.log_prob(samples, x=observation)
                # print('log_probability: {}'.format(log_probability))
                # samples = posterior.sample((10000,), x=observation)
                # -------------------------------------------------------
                export_plots(samples, points, lim_low, lim_high, N, method, m_name, dt_descriptor, model_class)

                N_samples = 20
                print('Drawing the {} most likely samples..'.format(N_samples))
                posterior_params = posterior.sample((N_samples,), x=observation)
                print('\nposterior_params: {}'.format(posterior_params))

                mean_model_spike_counts = torch.tensor([])
                converged_mean_model_spike_counts = torch.tensor([])
                # std_model_rates = torch.tensor([])

                avg_param_dist_across_samples = []
                converged_avg_param_dist_across_samples = []
                for s_i in range(N_samples):
                    model_params = convert_posterior_to_model_params_dict(model_class, posterior_params[s_i], tar_model_class, points, N)
                    programmatic_neuron_types = torch.ones((N,))
                    for n_i in range(int(2 * N / 3), N):
                        programmatic_neuron_types[n_i] = -1

                    model = model_class(parameters=model_params, N=N, neuron_types=programmatic_neuron_types)
                    cur_mean_spike_counts = export_stats_model_target(model, observation=observation,
                                                                      descriptor='{}_parallel_sbi_{}_sample_N_{}'.
                                                                      format(m_name, dt_descriptor, s_i))
                    mean_model_spike_counts = torch.cat((mean_model_spike_counts, cur_mean_spike_counts))


                    more_than_one_third_fairly_silent = (cur_mean_spike_counts < 1.).sum() > 0.333 * len(cur_mean_spike_counts)
                    if not more_than_one_third_fairly_silent:
                        converged_mean_model_spike_counts = torch.cat((converged_mean_model_spike_counts, cur_mean_spike_counts))

                    current_avg_dist_per_p = []
                    model_parameter_list = model.get_parameters()
                    for p_i in range(len(model_class.parameter_names)):
                        cur_p_name = model_class.parameter_names[p_i]
                        dist_p_i = parameter_distance.euclid_dist(model_parameter_list[cur_p_name], points[p_i])
                        current_avg_dist_per_p.append(dist_p_i)
                    plot_param_dist(np.array(current_avg_dist_per_p), 'Parameter distance for sample: {}'.format(s_i),
                                    '{}_N_{}_parallel_sbi_{}_sample_num_{}'.format(m_name, N, dt_descriptor, s_i))
                    avg_param_dist_across_samples.append(current_avg_dist_per_p)
                    if not more_than_one_third_fairly_silent:
                        converged_avg_param_dist_across_samples.append(current_avg_dist_per_p)

                mean_across_exps = np.mean(avg_param_dist_across_samples, axis=1)
                plot_param_dist(mean_across_exps, 'Parameter distance across samples',
                                'sbi_samples_avg_param_dist_{}_N_{}_{}'.format(m_name, N, dt_descriptor))
                converged_mean_p_dist = np.mean(converged_avg_param_dist_across_samples, axis=1)
                # if not hasattr(converged_mean_p_dist, 'len'):
                #     converged_mean_p_dist = np.array([converged_mean_p_dist])
                plot_param_dist(converged_mean_p_dist, 'Parameter distance across samples forming non-silent models',
                                'sbi_samples_converged_non_silent_avg_param_dist_{}_N_{}_{}'.format(m_name, N, dt_descriptor))

                # std_model_rates.append(cur_std_model_rate)
                mean_model_spike_counts = torch.reshape(mean_model_spike_counts, (N_samples, -1))
                converged_mean_model_spike_counts = torch.reshape(converged_mean_model_spike_counts, (-1, len(observation)))
                export_stats_top_samples(torch.mean(mean_model_spike_counts, dim=0), torch.std(mean_model_spike_counts, dim=0),
                                         observation, '{}_{}_sbi_parallel_{}'.format(method, m_name, dt_descriptor), N_samples=len(mean_model_spike_counts))
                converged_mean_model_spike_counts = torch.mean(converged_mean_model_spike_counts, dim=0)
                # if not hasattr(converged_mean_model_rates, 'len'):
                #     converged_mean_model_rates = np.array([converged_mean_model_rates])
                export_stats_top_samples(converged_mean_model_spike_counts, torch.std(converged_mean_model_spike_counts, dim=0),
                                         observation, 'converged_non_silent_{}_{}_sbi_parallel_{}'.format(method, m_name, dt_descriptor), N_samples=len(converged_mean_model_spike_counts))

            else:
                print('Error: Did not find corresponding .pt-file: {}'.format(corresponding_samples_fname))

if __name__ == "__main__":
    main()
    sys.exit(0)
