import torch

import Log
from plot import plot_all_param_pairs_with_variance

current_params_for_optim = torch.load('/home/william/repos/snn_inference/Dev/saved/test/test.pt')
target_parameters = torch.load('/home/william/repos/snn_inference/Dev/saved/exported_model_params/generated_spike_train_random_glif_1_model_t_300s_rate_0_6_params.pt')
param_names = ['f_I', 'C_m', 'G', 'R_I', 'f_v', 'E_L', 'b_s', 'b_v', 'a_v', 'delta_theta_s', 'delta_V', 'theta_inf', 'I_A']
optim_name = 'DE'

# print('current_params_for_optim', current_params_for_optim)
# print('target_parameters', target_parameters)
corrected_params = {}
enumerated_targets = {}

for i in range(2, len(list(current_params_for_optim.values()))):
    corrected_params[i-2] = current_params_for_optim[i]

target_parameters.pop('w')
for i in range(len(list(target_parameters.values()))):
    enumerated_targets[i] = list(target_parameters.values())[i]


# plot_all_param_pairs_with_variance(corrected_params,
#                                    exp_type='nevergrad',
#                                    uuid='single_objective_optim',
#                                    target_params=enumerated_targets,
#                                    param_names=param_names,
#                                    custom_title="KDEs for values across experiments ({})".format(optim_name),
#                                    logger=Log.Logger('test_plot_param_pairs_nevergrad'),
#                                    fname='single_objective_KDE_optim_{}'.format(optim_name))
