import sys

import numpy as np

import experiments
import model_util
import plot
from IO import *
from Models.microGIF import microGIF
from analysis import analysis_util
from analysis.spike_train_matlab_export import simulate_and_save_model_spike_train


def OU_process_input(t=10000):
    mu = torch.tensor([1., 1., 0, 1.])
    tau = torch.tensor([2000., 2000., 1., 2000.])
    q = torch.tensor([0.5, 0.5, 0, 0.5])
    dW = torch.randn((t, 4))
    # dW = torch.zeros((t, 4))

    I_0 = torch.tensor([0.5, 0.5, 0, 0.5])
    I_interval = I_0.clone().detach()
    I = I_0.clone().detach()
    for t_i in range(t-1):
        dI = (I - mu)/tau + torch.sqrt(2/tau) * q * dW[t_i, :]
        I = I + dI
        I_interval = torch.vstack((I_interval, dI))

    assert I_interval.shape[0] == t and I_interval.shape[1] == 4, "I_interval should be {}x{}. Was: {}".format(t, 4, I_interval.shape)
    # return torch.zeros((t, 4))
    return I_interval


def get_model_activity(model, inputs):
    if model.__class__ is microGIF:
        s_lambdas, _, _ = model_util.feed_inputs_sequentially_return_args(model=model, inputs=inputs.clone().detach())
    else:
        s_lambdas, _ = model_util.feed_inputs_sequentially_return_tuple(model=model, inputs=inputs.clone().detach())
    return s_lambdas


def activity_RMSE(model_spikes, target_spikes, bin_size):
    num_pops = 4
    assert model_spikes.shape[1] == num_pops, "assuming row by column shape, i.e. (t, 4): {}".format(model_spikes.shape)
    assert target_spikes.shape[1] == num_pops, "assuming row by column shape, i.e. (t, 4): {}".format(target_spikes.shape)
    assert model_spikes.shape[0] % bin_size == 0, "time should be a multiple of bin_size: {}, model_spikes.shape: {}".format(bin_size, model_spikes.shape)
    num_bins = int(model_spikes.shape[0] / bin_size)
    m_binned_spikes = torch.zeros((num_bins, num_pops))
    t_binned_spikes = torch.zeros((num_bins, num_pops))
    for b_i in range(num_bins):
        for bp_i in range(num_pops):
            m_binned_spikes[b_i, bp_i] = model_spikes[b_i * bin_size:(b_i + 1) * bin_size, bp_i].sum()
            t_binned_spikes[b_i, bp_i] = target_spikes[b_i * bin_size:(b_i + 1) * bin_size, bp_i].sum()
    rmse = torch.sqrt(torch.sum(torch.mean(torch.pow(m_binned_spikes - t_binned_spikes, 2))) / num_pops)
    return rmse.detach().numpy()


def activity_correlations(model_act, tar_act, bin_size):
    num_pops = 4
    assert model_act.shape[1] == num_pops, "assuming row by column shape, i.e. (t, 4): {}".format(model_act.shape)
    assert tar_act.shape[1] == num_pops, "assuming row by column shape, i.e. (t, 4): {}".format(tar_act.shape)
    assert model_act.shape[0] % bin_size == 0, "time should be a multiple of bin_size: {}, m_act.shape: {}".format(bin_size, model_act.shape)
    num_bins = int(model_act.shape[0] / bin_size)
    m_bins = torch.zeros((num_bins, num_pops))
    t_bins = torch.zeros((num_bins, num_pops))
    for b_i in range(num_bins):
        for bp_i in range(num_pops):
            m_bins[b_i, bp_i] = model_act[b_i * bin_size:(b_i + 1) * bin_size, bp_i].sum()
            t_bins[b_i, bp_i] = tar_act[b_i * bin_size:(b_i + 1) * bin_size, bp_i].sum()
    m_bin_avg = torch.mean(m_bins, dim=0)
    t_bin_avg = torch.mean(t_bins, dim=0)
    assert len(m_bin_avg) == num_pops, "len mactavg {} should be numpops {}".format(len(m_bin_avg), num_pops)
    rho = torch.tensor(0.)
    for p_i in range(num_pops):
        rho+= ((t_bins[:, p_i] - t_bin_avg[p_i]) * (m_bins[:, p_i] - m_bin_avg[p_i]) / torch.sqrt(torch.pow((t_bins[:, p_i] - t_bin_avg[p_i]), 2) * torch.pow((m_bins[:, p_i] - m_bin_avg[p_i]), 2))).sum()
    return (rho / num_pops).detach().numpy()


def convert_posterior_to_model_params_dict(model_class, posterior_params, target_class, target_points, N):
    if posterior_params.shape[0] == 1:
        posterior_params = posterior_params[0]
    model_params = {}
    p_i = 0
    for p_name in model_class.free_parameters:
        if p_i == 0:
            assert p_name == 'w'
            model_params[p_name] = posterior_params[:(N**2-N)]
        else:
            model_params[p_name] = posterior_params[(N**2-N)+N*(p_i-1):(N**2-N)+N*p_i]
        p_i += 1

    t_p_i = 1
    for t_p_name in target_class.free_parameters:
        if t_p_name not in model_params:
            model_params[t_p_name] = target_points[(N**2-N)+N*(t_p_i-1):(N**2-N)+N*t_p_i]
        t_p_i += 1

    return model_params


# experiments_path = '/home/william/repos/archives_snn_inference/GENERIC/archive/saved/data/'
experiments_path = '/home/william/repos/archives_snn_inference/archive_1612/archive/saved/data/'

files_sbi_res = os.listdir(experiments_path + 'sbi_res/')

correlations_OU_per_model_type = { 'microGIF': [] }
activity_rmse_OU_per_model_type = { 'microGIF': [] }
correlations_wn_per_model_type = { 'microGIF': [] }
activity_rmse_wn_per_model_type = { 'microGIF': [] }

init_correlations_OU_per_model_type = {}
init_activity_rmse_OU_per_model_type = {}
init_correlations_wn_per_model_type = {}
init_activity_rmse_wn_per_model_type = {}
for sbi_res_file in files_sbi_res:
    print(sbi_res_file)

    sbi_res_path = experiments_path + 'sbi_res/' + sbi_res_file
    print('Loading: {}'.format(sbi_res_path))
    res_load = torch.load(sbi_res_path)
    print('sbi_res load successful.')

    sbi_res = res_load['data']
    assert sbi_res.keys().__contains__('SNPE'), "method SNPE expected"
    method = 'SNPE'
    posterior = sbi_res[method]
    sut_description = sbi_res['dt_descriptor']
    model_class = sbi_res['model_class']

    if model_class is microGIF:
        m_name = model_class.__name__
        N = sbi_res['N']
        dt_descriptor = sbi_res['dt_descriptor']
        tar_seed = False
        if sbi_res_file.__contains__('tar_seed'):
            tar_seed = int(sbi_res_file.split('tar_seed_')[-1].split('.pt')[0])

        if tar_seed:
            corresponding_samples_fname = 'samples_method_{}_m_name_{}_dt_{}_tar_seed_{}.pt'.format(method, m_name, dt_descriptor, tar_seed)
        else:
            corresponding_samples_fname = 'samples_method_{}_m_name_{}_dt_{}.pt'.format(method, m_name, dt_descriptor)

        data_arr = torch.load(experiments_path + 'sbi_samples/' + corresponding_samples_fname)['data']
        print('sbi_samples load successful.')

        save_fname = 'spikes_sbi_{}_tar_seed_{}'.format(corresponding_samples_fname.strip('.pt')+'', tar_seed)

        torch.manual_seed(tar_seed)
        np.random.seed(tar_seed)

        # samples = data_arr['samples']
        observation = data_arr['observation']
        # points = data_arr['tar_parameters']
        m_name = data_arr['m_name']

        # log_probability = posterior.log_prob(samples, x=observation)
        # print('log_probability: {}'.format(log_probability))

        N_samples = 15
        t = 10000
        print('Drawing the {} most likely samples..'.format(N_samples))
        posterior_params = posterior.sample((N_samples,), x=observation)
        print('\nposterior_params: {}'.format(posterior_params))
        tar_m_name = m_name
        if m_name == 'microGIF':
            tar_m_name = 'mesoGIF'
        target_model = analysis_util.get_target_model(tar_m_name)
        programmatic_neuron_types = N * [1]
        n_inhib = int(N / 4)
        programmatic_neuron_types[-n_inhib:] = n_inhib * [-1]
        init_params_dict = experiments.draw_from_uniform(model_class.parameter_init_intervals, target_model.N)
        if m_name == 'microGIF':
            init_model = model_class(parameters=init_params_dict, N=N)
        else:
            init_model = model_class(parameters=init_params_dict, N=N, neuron_types=programmatic_neuron_types)

        burn_in_inputs = torch.rand((t,4))
        _ = get_model_activity(init_model, burn_in_inputs)
        _ = get_model_activity(target_model, burn_in_inputs)
        eval_inputs = OU_process_input(t=t)
        t_act_OU = get_model_activity(target_model, eval_inputs)
        init_act_OU = get_model_activity(init_model, eval_inputs)

        init_model.reset()
        target_model.reset()

        burn_in_inputs = torch.rand((t, 4))
        _ = get_model_activity(init_model, burn_in_inputs)
        _ = get_model_activity(target_model, burn_in_inputs)
        white_noise = torch.rand((t,))
        init_act_wn = get_model_activity(init_model, white_noise)
        t_act_wn = get_model_activity(target_model, white_noise)

        init_correlations_wn_per_model_type[m_name] = activity_correlations(init_act_wn, t_act_wn, bin_size=int(t/8))
        init_activity_rmse_wn_per_model_type[m_name] = activity_RMSE(init_act_wn, t_act_wn, bin_size=int(t/8))
        init_correlations_OU_per_model_type[m_name] = activity_correlations(init_act_OU, t_act_OU, bin_size=int(t/8))
        init_activity_rmse_OU_per_model_type[m_name] = activity_RMSE(init_act_OU, t_act_OU, bin_size=int(t/8))

        for s_i in range(N_samples):
            model_params = convert_posterior_to_model_params_dict(model_class, posterior_params[s_i], target_class=model_class, target_points=[], N=N)
            programmatic_neuron_types = torch.ones((N,))
            for n_i in range(int(2 * N / 3), N):
                programmatic_neuron_types[n_i] = -1
            if model_class is microGIF:
                model = model_class(parameters=model_params, N=N)
            else:
                model = model_class(parameters=model_params, N=N, neuron_types=programmatic_neuron_types)

            # burn in:
            burn_in_inputs = OU_process_input(t=t)
            _ = get_model_activity(model, burn_in_inputs)
            eval_inputs = OU_process_input(t=t)
            m_act_OU = get_model_activity(model, eval_inputs)

            model.reset()

            white_noise_burn_in = torch.rand((t,4))
            _ = get_model_activity(model, white_noise_burn_in)
            white_noise = torch.rand((t,4))
            m_act_wn = get_model_activity(model, white_noise)

            activity_correlation_OU_process = activity_correlations(m_act_OU, t_act_OU, bin_size=int(t/8))
            activity_RMSE_OU_process = activity_RMSE(m_act_OU, t_act_OU, bin_size=int(t/8))
            activity_correlation_white_noise = activity_correlations(m_act_wn, t_act_wn, bin_size=int(t/8))
            activity_RMSE_white_noise = activity_RMSE(m_act_wn, t_act_wn, bin_size=int(t/8))

            correlations_OU_per_model_type[m_name].append(activity_correlation_OU_process)
            activity_rmse_OU_per_model_type[m_name].append(activity_RMSE_OU_process)
            correlations_wn_per_model_type[m_name].append(activity_correlation_white_noise)
            activity_rmse_wn_per_model_type[m_name].append(activity_RMSE_white_noise)


xticks = []
correlations_wn = []; rmse_wn = []; corr_wn_std = []; rmse_wn_std = []
correlations_OU = []; rmse_OU = []; corr_OU_std = []; rmse_OU_std = []
init_corrs_wn = []; init_corrs_OU = []; init_rmse_wn = []; init_rmse_wn_std = []
init_rmse_OU = []
for m_k in correlations_wn_per_model_type.keys():
    correlations_wn.append(np.mean(correlations_wn_per_model_type[m_k]))
    corr_wn_std.append(np.std(correlations_wn_per_model_type[m_k]))
    rmse_wn.append(np.mean(activity_rmse_wn_per_model_type[m_k]))
    rmse_wn_std.append(np.std(activity_rmse_wn_per_model_type[m_k]))

    correlations_OU.append(np.mean(correlations_OU_per_model_type[m_k]))
    corr_OU_std.append(np.std(correlations_OU_per_model_type[m_k]))
    rmse_OU.append(np.mean(activity_rmse_OU_per_model_type[m_k]))
    rmse_OU_std.append(np.std(activity_rmse_OU_per_model_type[m_k]))

    init_corrs_wn.append(init_correlations_wn_per_model_type[m_k])
    init_rmse_wn.append(init_activity_rmse_wn_per_model_type[m_k])
    init_corrs_OU.append(init_correlations_OU_per_model_type[m_k])
    init_rmse_OU.append(init_correlations_OU_per_model_type[m_k])

    xticks.append(m_k.replace('microGIF', 'miGIF').replace('mesoGIF', 'meGIF'))

plot_exp_type = 'export_sbi'
plot.bar_plot_neuron_rates(init_corrs_wn, correlations_wn, 0, 0,
                           exp_type=plot_exp_type, uuid='all', fname='sbi_correlations_wn_all.eps', xticks=xticks,
                           custom_legend=['Init. model', 'Posterior models'], ylabel='Avg. activity correlation',
                           custom_colors=['Gray', 'Brown'])
plot.bar_plot_neuron_rates(init_corrs_OU, correlations_OU, 0, 0,
                           exp_type=plot_exp_type, uuid='all', fname='sbi_correlations_OU_all.eps', xticks=xticks,
                           custom_legend=['Init. model', 'Posterior models'], ylabel='Avg. activity correlation',
                           custom_colors=['Gray', 'Brown'])

# not_undef = lambda x: not np.isnan(x) and not np.isinf(x)
# init_rmse_wn = list(filter(not_undef, init_rmse_wn))
# rmse_wn = list(filter(not_undef, rmse_wn))
# init_rmse_OU = list(filter(not_undef, init_rmse_OU))
# rmse_OU = list(filter(not_undef, rmse_OU))
#
# correlations_wn = list(filter(not_undef, correlations_wn))
# correlations_OU = list(filter(not_undef, correlations_OU))
# init_corrs_wn = list(filter(not_undef, init_corrs_wn))
# init_corrs_OU = list(filter(not_undef, init_corrs_OU))

import torch.tensor as T
n_init_rmse_wn = T(np.asarray(init_rmse_wn))/T(np.asarray(init_rmse_wn)); n_rmse_wn = T(rmse_wn) / T(np.asarray(init_rmse_wn))
# n_rmse_wn[1] = torch.min(n_rmse_wn[1], T(4.))
n_init_rmse_OU = T(np.asarray(init_rmse_OU))/T(np.asarray(init_rmse_OU)); n_rmse_OU = T(rmse_OU) / T(np.asarray(init_rmse_OU))
# n_rmse_OU[1] = torch.min(n_rmse_OU[1], T(4.))
plot.bar_plot(n_rmse_wn.numpy(), y_std=0, exp_type=plot_exp_type, uuid='all',
              ylabel='RMSE', fname='plot_rmse_wn_all.eps', labels=xticks, baseline=1.,
              custom_colors=['Purple'], custom_legend=['Init. model', 'Posterior models'])
plot.bar_plot(n_rmse_OU.numpy(), y_std=0, exp_type=plot_exp_type, uuid='all',
              ylabel='RMSE', fname='plot_rmse_OU_all.eps', labels=xticks, baseline=1.,
              custom_colors=['Purple'], custom_legend=['Init. model', 'Posterior models'])

print('init_rmse_wn', init_rmse_wn)
print('rmse_wn', rmse_wn)

print('init_rmse_OU', init_rmse_OU)
print('rmse_OU', rmse_OU)
# ------

print('n_init_rmse_wn', n_init_rmse_wn)
print('n_rmse_wn', n_rmse_wn)

print('n_init_rmse_OU', n_init_rmse_OU)
print('n_rmse_OU', n_rmse_OU)

print('init_corrs_wn', init_corrs_wn)
print('correlations_wn', correlations_wn)

print('init_corrs_OU', init_corrs_OU)
print('correlations_OU', correlations_OU)

# sys.exit()
