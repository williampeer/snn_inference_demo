import numpy as np
import torch
from torch import tensor as T

from Models.GLIF import GLIF
from Models.LIF import LIF
from experiments import randomise_parameters, zip_tensor_dicts


def lif(random_seed, N = 12):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    weights_std = 0.05
    pop_size = int(N / 2)
    pop_size_last = N - pop_size

    params_pop1 = {'tau_m': 2.8, 'E_L': -56., 'tau_s': 3.5, 'spike_threshold': 30.}
    hand_coded_params_pop1 = {'preset_weights': (torch.ones((pop_size, 1)) + (2*weights_std * torch.randn((pop_size, N))) - weights_std) *
                                                torch.cat([T(pop_size * [0.55]), T(pop_size_last * [0.5])])}

    params_pop2 = {'tau_m': 1.8, 'E_L': -64., 'tau_s': 2., 'spike_threshold': 30.}
    hand_coded_params_pop2 = {'preset_weights': (torch.ones((pop_size_last, 1)) + (2*weights_std * torch.randn((pop_size_last, N))) - weights_std) *
                                                torch.cat([T(pop_size * [0.55]), T(pop_size_last * [0.5])])}

    params_pop1 = randomise_parameters(params_pop1, coeff=T(0.025), N_dim=pop_size)
    params_pop1 = zip_tensor_dicts(params_pop1, hand_coded_params_pop1)
    params_pop2 = randomise_parameters(params_pop2, coeff=T(0.025), N_dim=pop_size)
    params_pop2 = zip_tensor_dicts(params_pop2, hand_coded_params_pop2)
    randomised_params = zip_tensor_dicts(params_pop1, params_pop2)

    neuron_types = np.ones((N,))
    for i in range(int(N / 2)):
        neuron_types[-(1 + i)] = -1
    return LIF(parameters=randomised_params, N=N, neuron_types=neuron_types)


def glif(random_seed, N = 12):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    pop_size = int(N / 2)
    pop_size_last = N - pop_size
    params_pop1 = {'tau_m': 3.4, 'G': 0.7, 'E_L': -52., 'delta_theta_s': 18., 'b_s': 0.4, 'f_v': 0.14, 'delta_V': 10.,
                   'f_I': 0.5, 'b_v': 0.3, 'a_v': 0.2, 'theta_inf': -12., 'tau_s': 4.5}
    weights_std = 0.05
    # weights_std = 0
    hand_coded_params_pop1 = {'preset_weights': (torch.ones((pop_size, 1)) +
                                                 (2*weights_std * torch.randn((pop_size, N))) - weights_std) *
                                                torch.cat([T(pop_size * [0.55]), T(pop_size_last * [0.45])])}

    params_pop2 = {'tau_m': 2.6, 'G': 0.8, 'E_L': -62., 'delta_theta_s': 18., 'b_s': 0.25, 'f_v': 0.14, 'delta_V': 12.,
                   'f_I': 0.35, 'b_v': 0.4, 'a_v': 0.3, 'theta_inf': -11., 'tau_s': 2.4}
    hand_coded_params_pop2 = {'preset_weights': (torch.ones((pop_size_last, 1)) +
                                                 (2*weights_std * torch.randn((pop_size_last, N))) - weights_std) *
                                                torch.cat([T(pop_size * [.5]), T(pop_size_last * [.35])])}

    params_pop1 = randomise_parameters(params_pop1, coeff=T(0.025), N_dim=pop_size)
    params_pop1 = zip_tensor_dicts(params_pop1, hand_coded_params_pop1)
    params_pop2 = randomise_parameters(params_pop2, coeff=T(0.025), N_dim=pop_size)
    params_pop2 = zip_tensor_dicts(params_pop2, hand_coded_params_pop2)
    randomised_params = zip_tensor_dicts(params_pop1, params_pop2)

    neuron_types = np.ones((N,))
    for i in range(int(N / 2)):
        neuron_types[-(1 + i)] = -1
    return GLIF(parameters=randomised_params, N=N, neuron_types=neuron_types)
