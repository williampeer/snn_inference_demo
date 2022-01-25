import numpy as np

import experiments
import model_util
import plot
from IO import *
from Models.microGIF import microGIF
from analysis import analysis_util


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


def activity_RMSE_per_pop(model_spikes, target_spikes, bin_size):
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
    rmse_per_pop = torch.sqrt((torch.mean(torch.pow(m_binned_spikes - t_binned_spikes, 2), dim=0)))
    return rmse_per_pop.detach().numpy()


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
        rho += ((t_bins[:, p_i] - t_bin_avg[p_i]) * (m_bins[:, p_i] - m_bin_avg[p_i]) / torch.sqrt(torch.pow((t_bins[:, p_i] - t_bin_avg[p_i]), 2) * torch.pow((m_bins[:, p_i] - m_bin_avg[p_i]), 2))).sum()
    return (rho / num_pops).detach().numpy()


def activity_correlations_per_pop(model_act, tar_act, bin_size):
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

    corrs = []
    import scipy.stats
    for bp_i in range(num_pops):
        corrs.append(scipy.stats.pearsonr(m_bins[:, bp_i].detach().numpy(), t_bins[:, bp_i].detach().numpy()))
    return corrs



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


tar_seed = 3
torch.manual_seed(tar_seed)
np.random.seed(tar_seed)

# experiments_path = '/home/william/repos/archives_snn_inference/GENERIC/archive/saved/data/'
# experiments_path = '/home/william/repos/archives_snn_inference/archive_1612/archive/saved/data/'
# experiments_path = '/home/william/repos/archives_snn_inference/archive_1612/archive/saved/data/'
experiments_path = '/home/william/repos/archives_snn_inference/archive_mesoGIF_and_LIF_GIF_pscapes_0401/archive/saved/Synthetic/'
model_type_str = 'microGIF'
load_fname = 'snn_model_target_GD_test.pt'

exp_folders = os.listdir(experiments_path)

correlations_OU_per_model_type = { 'microGIF': [] }
activity_rmse_OU_per_model_type = { 'microGIF': [] }
correlations_wn_per_model_type = { 'microGIF': [] }
activity_rmse_wn_per_model_type = { 'microGIF': [] }
correlations_OU_per_model_type_per_pop = { 'microGIF': [] }
activity_rmse_OU_per_model_type_per_pop = { 'microGIF': [] }
correlations_wn_per_model_type_per_pop = { 'microGIF': [] }
activity_rmse_wn_per_model_type_per_pop = { 'microGIF': [] }

init_correlations_OU_per_model_type = {}
init_activity_rmse_OU_per_model_type = {}
init_correlations_wn_per_model_type = {}
init_activity_rmse_wn_per_model_type = {}
init_correlations_OU_per_model_type_per_pop = {}
init_activity_rmse_OU_per_model_type_per_pop = {}
init_correlations_wn_per_model_type_per_pop = {}
init_activity_rmse_wn_per_model_type_per_pop = {}
exp_uids = os.listdir(experiments_path + '/' + model_type_str)

method = 'GBO'
t = 9000
# t = 1200
BIN_SIZE = int(t/9)
target_model = analysis_util.get_target_model('mesoGIF')
init_params_dict = experiments.draw_from_uniform(microGIF.parameter_init_intervals, target_model.N)
init_model = microGIF(parameters=init_params_dict, N=target_model.N)

# burn_in_inputs = torch.rand((t,4))
# _ = get_model_activity(init_model, burn_in_inputs)
# _ = get_model_activity(target_model, burn_in_inputs)
# eval_inputs = OU_process_input(t=t)
# t_act_OU = get_model_activity(target_model, eval_inputs)
# init_act_OU = get_model_activity(init_model, eval_inputs)
#
# init_model.reset()
# target_model.reset()
#
# burn_in_inputs = torch.rand((t, 4))
# _ = get_model_activity(init_model, burn_in_inputs)
# _ = get_model_activity(target_model, burn_in_inputs)
# white_noise = torch.rand((t,))
# init_act_wn = get_model_activity(init_model, white_noise)
# t_act_wn = get_model_activity(target_model, white_noise)
#
# init_correlations_wn_per_model_type['microGIF'] = activity_correlations(init_act_wn, t_act_wn, bin_size=BIN_SIZE)
# init_activity_rmse_wn_per_model_type['microGIF'] = activity_RMSE(init_act_wn, t_act_wn, bin_size=BIN_SIZE)
# init_correlations_OU_per_model_type['microGIF'] = activity_correlations(init_act_OU, t_act_OU, bin_size=BIN_SIZE)
# init_activity_rmse_OU_per_model_type['microGIF'] = activity_RMSE(init_act_OU, t_act_OU, bin_size=BIN_SIZE)
#
# init_correlations_wn_per_model_type_per_pop['microGIF'] = activity_correlations_per_pop(init_act_wn, t_act_wn, bin_size=BIN_SIZE)
# init_activity_rmse_wn_per_model_type_per_pop['microGIF'] = activity_RMSE_per_pop(init_act_wn, t_act_wn, bin_size=BIN_SIZE)
# init_correlations_OU_per_model_type_per_pop['microGIF'] = activity_correlations_per_pop(init_act_OU, t_act_OU, bin_size=BIN_SIZE)
# init_activity_rmse_OU_per_model_type_per_pop['microGIF'] = activity_RMSE_per_pop(init_act_OU, t_act_OU, bin_size=BIN_SIZE)
#
#
# for euid in exp_uids:
#     load_data = torch.load(experiments_path + '/' + model_type_str + '/' + euid + '/' + load_fname)
#     model = load_data['model']
#     loss_data = load_data['loss']
#     exp_lfn = loss_data['lfn']
#
#     cur_fname = 'nuovo_synthetic_v3_mesoGIF__spikes_mt_{}_lfn_{}_euid_{}'.format(microGIF.__name__, exp_lfn, euid)
#     m_name = loss_data['model_type']
#     N = model.N
#     dt_descriptor = euid
#
#     # burn in:
#     burn_in_inputs = OU_process_input(t=t)
#     _ = get_model_activity(model, burn_in_inputs)
#     eval_inputs = OU_process_input(t=t)
#     m_act_OU = get_model_activity(model, eval_inputs)
#
#     model.reset()
#
#     white_noise_burn_in = torch.rand((t,4))
#     _ = get_model_activity(model, white_noise_burn_in)
#     white_noise = torch.rand((t,4))
#     m_act_wn = get_model_activity(model, white_noise)
#
#     activity_correlation_OU_process = activity_correlations(m_act_OU, t_act_OU, bin_size=BIN_SIZE)
#     activity_RMSE_OU_process = activity_RMSE(m_act_OU, t_act_OU, bin_size=BIN_SIZE)
#     activity_correlation_white_noise = activity_correlations(m_act_wn, t_act_wn, bin_size=BIN_SIZE)
#     activity_RMSE_white_noise = activity_RMSE(m_act_wn, t_act_wn, bin_size=BIN_SIZE)
#     activity_correlation_OU_process_per_pop = activity_correlations_per_pop(m_act_OU, t_act_OU, bin_size=BIN_SIZE)
#     activity_RMSE_OU_process_per_pop = activity_RMSE_per_pop(m_act_OU, t_act_OU, bin_size=BIN_SIZE)
#     activity_correlation_white_noise_per_pop = activity_correlations_per_pop(m_act_wn, t_act_wn, bin_size=BIN_SIZE)
#     activity_RMSE_white_noise_per_pop = activity_RMSE_per_pop(m_act_wn, t_act_wn, bin_size=BIN_SIZE)
#
#     correlations_OU_per_model_type[m_name].append(activity_correlation_OU_process)
#     activity_rmse_OU_per_model_type[m_name].append(activity_RMSE_OU_process)
#     correlations_wn_per_model_type[m_name].append(activity_correlation_white_noise)
#     activity_rmse_wn_per_model_type[m_name].append(activity_RMSE_white_noise)
#     correlations_OU_per_model_type_per_pop[m_name].append(activity_correlation_OU_process_per_pop)
#     activity_rmse_OU_per_model_type_per_pop[m_name].append(activity_RMSE_OU_process_per_pop)
#     correlations_wn_per_model_type_per_pop[m_name].append(activity_correlation_white_noise_per_pop)
#     activity_rmse_wn_per_model_type_per_pop[m_name].append(activity_RMSE_white_noise_per_pop)
# massive_dict = { 'correlations_OU_per_model_type': correlations_OU_per_model_type, 'activity_rmse_OU_per_model_type': activity_rmse_OU_per_model_type,
#                  'correlations_wn_per_model_type': correlations_wn_per_model_type, 'activity_rmse_wn_per_model_type': activity_rmse_wn_per_model_type,
#                  'correlations_OU_per_model_type_per_pop': correlations_OU_per_model_type_per_pop, 'activity_rmse_OU_per_model_type_per_pop': activity_rmse_OU_per_model_type_per_pop,
#                  'correlations_wn_per_model_type_per_pop': correlations_wn_per_model_type_per_pop, 'activity_rmse_wn_per_model_type_per_pop': activity_rmse_wn_per_model_type_per_pop }
# torch.save(massive_dict, './save_stuff/massive_dict_SGIF_perf_metrics_bin_size_{}.pt'.format(BIN_SIZE))


massive_dict = torch.load('./save_stuff/massive_dict_SGIF_perf_metrics_bin_size_{}.pt'.format(BIN_SIZE))
print('massive_dict keys:', massive_dict.keys())
correlations_OU_per_model_type_per_pop = massive_dict['correlations_OU_per_model_type_per_pop']
activity_rmse_OU_per_model_type_per_pop = massive_dict['activity_rmse_OU_per_model_type_per_pop']
correlations_wn_per_model_type_per_pop = massive_dict['correlations_wn_per_model_type_per_pop']
activity_rmse_wn_per_model_type_per_pop = massive_dict['activity_rmse_wn_per_model_type_per_pop']


def print_all_and_converged_corrs(corrs, description, corr_filter_threshold=0.2):
    print(description, 'unfiltered mean:', np.mean(corrs, axis=0))
    corr_considered_converged = list(filter(lambda x: np.mean(x) > corr_filter_threshold, corrs))
    print('num \"converged\": {}'.format(len(corr_considered_converged)))
    print(description, 'filtered w threshold: {}'.format(corr_filter_threshold), np.mean(corr_considered_converged, axis=0))
    print()

print_all_and_converged_corrs(correlations_OU_per_model_type_per_pop['microGIF'][:20], description='correlations_OU_per_model_type_per_pop BNLL')
print_all_and_converged_corrs(correlations_wn_per_model_type_per_pop['microGIF'][:20], description='correlations_wn_per_model_type_per_pop BNLL')
print_all_and_converged_corrs(correlations_OU_per_model_type_per_pop['microGIF'][20:], description='correlations_wn_per_model_type_per_pop PNLL')
print_all_and_converged_corrs(correlations_wn_per_model_type_per_pop['microGIF'][20:], description='correlations_wn_per_model_type_per_pop PNLL')

print('activity_rmse_OU_per_model_type_per_pop BNLL', np.mean(activity_rmse_OU_per_model_type_per_pop['microGIF'][:20], axis=0))
print('activity_rmse_wn_per_model_type_per_pop BNLL', np.mean(activity_rmse_wn_per_model_type_per_pop['microGIF'][:20], axis=0))
print('activity_rmse_OU_per_model_type_per_pop PNLL', np.mean(activity_rmse_OU_per_model_type_per_pop['microGIF'][20:], axis=0))
print('activity_rmse_wn_per_model_type_per_pop PNLL', np.mean(activity_rmse_wn_per_model_type_per_pop['microGIF'][20:], axis=0))

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

    # init_corrs_wn.append(init_correlations_wn_per_model_type[m_k])
    # init_rmse_wn.append(init_activity_rmse_wn_per_model_type[m_k])
    # init_corrs_OU.append(init_correlations_OU_per_model_type[m_k])
    # init_rmse_OU.append(init_correlations_OU_per_model_type[m_k])

    xticks.append(m_k.replace('microGIF', 'miGIF').replace('mesoGIF', 'meGIF'))

# plot_exp_type = 'export_sbi'
# plot.bar_plot_neuron_rates(init_corrs_wn, correlations_wn, 0, 0,
#                            exp_type=plot_exp_type, uuid='all', fname='GBO_correlations_wn_all.eps', xticks=xticks,
#                            custom_legend=['Init. model', 'Posterior models'], ylabel='Avg. activity correlation',
#                            custom_colors=['Gray', 'Brown'])
# plot.bar_plot_neuron_rates(init_corrs_OU, correlations_OU, 0, 0,
#                            exp_type=plot_exp_type, uuid='all', fname='GBO_correlations_OU_all.eps', xticks=xticks,
#                            custom_legend=['Init. model', 'Posterior models'], ylabel='Avg. activity correlation',
#                            custom_colors=['Gray', 'Brown'])
#
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
#
# plot.bar_plot(np.asarray(rmse_wn), y_std=0, exp_type=plot_exp_type, uuid='all',
#               ylabel='RMSE', fname='plot_rmse_wn_all_GBO.eps', labels=xticks, baseline=1.,
#               custom_colors=['Purple'], custom_legend=['Init. model', 'Posterior models'])
# plot.bar_plot(np.asarray(rmse_OU), y_std=0, exp_type=plot_exp_type, uuid='all',
#               ylabel='RMSE', fname='plot_rmse_OU_all_GBO.eps', labels=xticks, baseline=1.,
#               custom_colors=['Purple'], custom_legend=['Init. model', 'Posterior models'])

# ------
# print('rmse_wn', rmse_wn)
# print('rmse_OU', rmse_OU)
# print('correlations_wn', correlations_wn)
# print('correlations_OU', correlations_OU)

# print('rmse_wn_per_pop', rmse_wn_per_pop)
# print('rmse_OU_per_pop', rmse_OU_per_pop)
# print('correlations_wn_per_pop', correlations_wn_per_pop)
# print('correlations_OU_per_pop', correlations_OU_per_pop)

# sys.exit()
