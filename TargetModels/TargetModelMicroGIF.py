import numpy as np
import torch
from torch import tensor as T

from Models.microGIF import microGIF
from Models.nonDifferentiableMicroGIF import nonDifferentiableMicroGIF
from experiments import randomise_parameters, zip_tensor_dicts


def draw_weights_for_pop(pop_num, pop_sizes, probabilities, weights):
    N = int(np.sum(np.asarray(pop_sizes)))
    Ws_pop_to_N = torch.zeros((pop_sizes[pop_num], N))
    # for neuron in population
    #   for each other population
    #       sample connectivity to population by drawing for each connection
    for neur_i in range(pop_sizes[pop_num]):
        neur_ind_offset = 0
        for pop_i in range(len(pop_sizes)):
            for pop_node in range(pop_sizes[pop_i]):
                connected = torch.rand((1,)) < probabilities[pop_i]
                if connected:
                    Ws_pop_to_N[neur_i, neur_ind_offset + pop_node] = weights[pop_i]
            neur_ind_offset += pop_sizes[pop_i]
    return Ws_pop_to_N


def get_low_dim_micro_GIF_transposed(random_seed):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    N = 4
    # neuron_types = torch.tensor([1, 1, -1, -1]).float()
    pop_sizes = [2, 2]

    # weights_excit_L4 = draw_weights_for_pop(pop_num=0, pop_sizes=pop_sizes, probabilities=2*[1.], weights=torch.mul(T([2.482, 1.245]), T([0.0497, 0.0794])))
    # weights_inhib_L4 = draw_weights_for_pop(pop_num=1, pop_sizes=pop_sizes, probabilities=2*[1.], weights=torch.mul(T(2 * [-4.964]), T([0.1350, 0.1597])))
    c = 0.3
    params_pop_excit_L4 = {'tau_m': 10., 'tau_s': 3., 'Delta_u': 5., 'tau_theta': 1000., 'c': c,
                           'J_theta': 1., 'E_L': -0.1, 'pop_sizes': 438}
    # hand_coded_params_pop_excit_L4 = {'preset_weights': weights_excit_L4}
    params_pop_inhib_L4 = {'tau_m': 10., 'tau_s': 6., 'Delta_u': 5., 'tau_theta': 1000., 'c': c,
                           'J_theta': 0., 'E_L': -0.1, 'pop_sizes': 109}
    # hand_coded_params_pop_inhib_L4 = {'preset_weights': weights_inhib_L4}

    params_pop_excit_L4 = randomise_parameters(params_pop_excit_L4, coeff=T(0.), N_dim=2)
    # params_pop_excit_L4 = zip_tensor_dicts(params_pop_excit_L4, hand_coded_params_pop_excit_L4)
    params_pop_inhib_L4 = randomise_parameters(params_pop_inhib_L4, coeff=T(0.), N_dim=2)
    # params_pop_inhib_L4 = zip_tensor_dicts(params_pop_inhib_L4, hand_coded_params_pop_inhib_L4)

    sut_weights_params = {}
    # rand_ws = torch.abs(4. + 1. * torch.rand((N, N))).clamp(0., 2.)
    # rand_ws = 2.*torch.ones((N,N))
    # rand_ws[2:, :] = 0.
    # rand_ws = torch.zeros((N,N))
    rand_ws = 10.* torch.ones((N,N))
    rand_ws[:, 2:] = -4.
    sut_weights_params['preset_weights'] = rand_ws

    randomised_params = zip_tensor_dicts(params_pop_excit_L4, params_pop_inhib_L4)
    randomised_params = zip_tensor_dicts(randomised_params, sut_weights_params)

    return pop_sizes, microGIF(parameters=randomised_params, N=N)


def micro_gif_populations_model_full_size(random_seed):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Read Schwalger et al. (2017), really figure out the parameters here. Also, have another look at whether this
    #   should work, or if we need to work with larger bins etc.
    # Conclusion: E_L set too high, but they implemented tonically firing neurons, i.e. model with stationary
    #  firing rates. I've tested that the rates differ without input, and without weights.

    # weights_std = 0

    pop_sizes = [8, 2, 9, 2]
    N = int(np.sum(np.asarray(pop_sizes)))
    pop_types = [1, -1, 1, -1]
    neuron_types = torch.ones((N,))
    for n_i in range(len(pop_sizes)):
        for n_j in range(pop_sizes[n_i]):
            if n_i > 0:
                ind = np.sum(np.asarray(pop_sizes[:n_i])) + n_j
            else:
                ind = n_j
            neuron_types[ind] = pop_types[n_i]

    # weights_excit_L2_3 = draw_weights_for_pop(pop_num=0, pop_sizes=[8,2,9,2], probabilities=[0.1009, 0.1346, 0.0077, 0.0691], weights=4*[1.245])
    # weights_inhib_L2_3 = draw_weights_for_pop(pop_num=1, pop_sizes=[8,2,9,2], probabilities=[0.1689, 0.1371, 0.0059, 0.0029], weights=4*[-4.964])
    # weights_excit_L4 = draw_weights_for_pop(pop_num=2, pop_sizes=[8,2,9,2], probabilities=[0.0437, 0.0316, 0.0497, 0.0794], weights=[1.245, 1.245, 2.482, 1.245])
    # weights_inhib_L4 = draw_weights_for_pop(pop_num=3, pop_sizes=[8,2,9,2], probabilities=[0.0818, 0.0515, 0.1350, 0.1597], weights=4*[-4.964])
    weights_excit_L2_3 = draw_weights_for_pop(pop_num=0, pop_sizes=[8,2,9,2], probabilities=4*[1.], weights=torch.mul(T(4*[1.245]), T([0.1009, 0.1346, 0.0077, 0.0691])))
    weights_inhib_L2_3 = draw_weights_for_pop(pop_num=1, pop_sizes=[8, 2, 9, 2], probabilities=4*[1.], weights=torch.mul(T(4 * [-4.964]), T([0.1689, 0.1371, 0.0059, 0.0029])))
    weights_excit_L4 = draw_weights_for_pop(pop_num=2, pop_sizes=[8, 2, 9, 2], probabilities=4*[1.], weights=torch.mul(T([1.245, 1.245, 2.482, 1.245]), T([0.0437, 0.0316, 0.0497, 0.0794])))
    weights_inhib_L4 = draw_weights_for_pop(pop_num=3, pop_sizes=[8, 2, 9, 2], probabilities=4*[1.], weights=torch.mul(T(4 * [-4.964]), T([0.0818, 0.0515, 0.1350, 0.1597])))

    E_L_origo = 0.
    params_pop_excit_L2_3 = {'tau_m': 10., 'tau_s': 3., 'Delta_u': 5., 'tau_theta': 1000., 'c': 0.1,
                          'J_theta': 1., 'E_L': E_L_origo+0., 'R_m': 0.001, 'pop_sizes': 413}
    hand_coded_params_pop_excit_L2_3 = {'preset_weights': weights_excit_L2_3 }
    params_pop_inhib_L2_3 = {'tau_m': 10., 'tau_s': 6., 'Delta_u': 5., 'tau_theta': 1000., 'c': 0.1,
                          'J_theta': 0., 'E_L': E_L_origo+0., 'R_m': 0.001, 'pop_sizes': 116}
    hand_coded_params_pop_inhib_L2_3 = {'preset_weights': weights_inhib_L2_3 }

    params_pop_excit_L4 = {'tau_m': 10., 'tau_s': 3., 'Delta_u': 5., 'tau_theta': 1000., 'c': 0.1,
                          'J_theta': 1., 'E_L': E_L_origo+7, 'R_m': 19., 'pop_sizes': 438}
    hand_coded_params_pop_excit_L4 = {'preset_weights': weights_excit_L4 }
    params_pop_inhib_L4 = {'tau_m': 10., 'tau_s': 6., 'Delta_u': 5., 'tau_theta': 1000., 'c': 0.1,
                          'J_theta': 0., 'E_L': E_L_origo+2, 'R_m': 11.964, 'pop_sizes': 109}
    hand_coded_params_pop_inhib_L4 = {'preset_weights': weights_inhib_L4 }

    params_pop_excit_L2_3 = randomise_parameters(params_pop_excit_L2_3, coeff=T(0.), N_dim=pop_sizes[0])
    params_pop_excit_L2_3 = zip_tensor_dicts(params_pop_excit_L2_3, hand_coded_params_pop_excit_L2_3)
    params_pop_inhib_L2_3 = randomise_parameters(params_pop_inhib_L2_3, coeff=T(0.), N_dim=pop_sizes[1])
    params_pop_inhib_L2_3 = zip_tensor_dicts(params_pop_inhib_L2_3, hand_coded_params_pop_inhib_L2_3)
    params_pop_excit_L4 = randomise_parameters(params_pop_excit_L4, coeff=T(0.), N_dim=pop_sizes[2])
    params_pop_excit_L4 = zip_tensor_dicts(params_pop_excit_L4, hand_coded_params_pop_excit_L4)
    params_pop_inhib_L4 = randomise_parameters(params_pop_inhib_L4, coeff=T(0.), N_dim=pop_sizes[3])
    params_pop_inhib_L4 = zip_tensor_dicts(params_pop_inhib_L4, hand_coded_params_pop_inhib_L4)

    rand_params_L2_3 = zip_tensor_dicts(params_pop_excit_L2_3, params_pop_inhib_L2_3)
    rand_params_L4 = zip_tensor_dicts(params_pop_excit_L4, params_pop_inhib_L4)
    randomised_params = zip_tensor_dicts(rand_params_L2_3, rand_params_L4)

    return pop_sizes, microGIF(parameters=randomised_params, N=N)


def meso_gif_populations_model(random_seed, pop_size=1, N_pops=2):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Read Schwalger et al. (2017), really figure out the parameters here. Also, have another look at whether this
    #   should work, or if we need to work with larger bins etc.
    # Conclusion: E_L set too high, but they implemented tonically firing neurons, i.e. model with stationary
    #  firing rates. I've tested that the rates differ without input, and without weights.

    weights_std = 0.0
    # weights_std = 0

    N = N_pops * pop_size

    if N_pops == 2:
        weights_excit = (torch.ones((pop_size, 1)) + (2 * weights_std * torch.randn((pop_size, N))) - weights_std) * \
                          torch.cat([T(pop_size * [2.482*0.0497]), T(pop_size * [1.245*0.0794])])
        weights_inhib = (torch.ones((pop_size, 1)) + (2 * weights_std * torch.randn((pop_size, N))) - weights_std) * \
                          torch.cat([T(pop_size * [-4.964*0.1350]), T(pop_size * [-4.964*0.1597])])

        # Excitatory:
        params_pop_excit = {'tau_m': 10., 'tau_s': 3., 'Delta_u': 5., 'tau_theta': 1000., 'c': 0.1,
                              'J_theta': 1., 'E_L': 2., 'R_m': 19., 'pop_sizes': 438 }
        hand_coded_params_pop_excit = {'preset_weights': weights_excit}

        # Inhibitory
        params_pop_inhib = {'tau_m': 10., 'tau_s': 6., 'Delta_u': 5., 'tau_theta': 1000., 'c': 0.1,
                              'J_theta': 1., 'E_L': 1.5, 'R_m': 11.964, 'pop_sizes': 109 }
        hand_coded_params_pop_inhib = {'preset_weights': weights_inhib}

        params_pop_excit = randomise_parameters(params_pop_excit, coeff=T(0.), N_dim=pop_size)
        params_pop_excit = zip_tensor_dicts(params_pop_excit, hand_coded_params_pop_excit)
        params_pop_inhib = randomise_parameters(params_pop_inhib, coeff=T(0.), N_dim=pop_size)
        params_pop_inhib = zip_tensor_dicts(params_pop_inhib, hand_coded_params_pop_inhib)

        randomised_params = zip_tensor_dicts(params_pop_excit, params_pop_inhib)
    elif N_pops == 4:
        # weights_excit_L2_3 = (torch.ones((pop_size, 1)) + (
        #             2 * weights_std * torch.randn((pop_size, N))) - weights_std) * torch.cat(
        #     [T(pop_size * [1.245*0.1009]), T(pop_size * [1.245*0.1346]), T(pop_size * [1.245*0.0077]), T(pop_size * [1.245*0.0691])])
        # weights_inhib_L2_3 = (torch.ones((pop_size, 1)) + (
        #             2 * weights_std * torch.randn((pop_size, N))) - weights_std) * torch.cat(
        #     [T(pop_size * [-4.964*0.1689]), T(pop_size * [-4.964*0.1371]), T(pop_size * [-4.964*0.0059]), T(pop_size * [-4.964*0.0029])])
        # weights_excit_L4 = (torch.ones((pop_size, 1)) + (
        #             2 * weights_std * torch.randn((pop_size, N))) - weights_std) * torch.cat(
        #     [T(pop_size * [1.245*0.0437]), T(pop_size * [1.245*0.0316]), T(pop_size * [2.482*0.0497]), T(pop_size * [1.245*0.0794])])
        # weights_inhib_L4 = (torch.ones((pop_size, 1)) + (
        #             2 * weights_std * torch.randn((pop_size, N))) - weights_std) * torch.cat(
        #     [T(pop_size * [-4.964*0.0818]), T(pop_size * [-4.964*0.0515]), T(pop_size * [-4.964*0.1350]), T(pop_size * [-4.964*0.1597])])
        # weights_excit_L2_3 = (torch.ones((pop_size, 1)) + (
        #         2 * weights_std * torch.randn((pop_size, N))) - weights_std) * torch.cat(
        #     [T(pop_size * [0.1009]), T(pop_size * [0.1346]), T(pop_size * [0.0077]),
        #      T(pop_size * [0.0691])])
        # weights_inhib_L2_3 = (torch.ones((pop_size, 1)) + (
        #         2 * weights_std * torch.randn((pop_size, N))) - weights_std) * torch.cat(
        #     [T(pop_size * [0.1689]), T(pop_size * [0.1371]), T(pop_size * [0.0059]),
        #      T(pop_size * [0.0029])])
        # weights_excit_L4 = (torch.ones((pop_size, 1)) + (
        #         2 * weights_std * torch.randn((pop_size, N))) - weights_std) * torch.cat(
        #     [T(pop_size * [0.0437]), T(pop_size * [0.0316]), T(pop_size * [0.0497]),
        #      T(pop_size * [0.0794])])
        # weights_inhib_L4 = (torch.ones((pop_size, 1)) + (
        #         2 * weights_std * torch.randn((pop_size, N))) - weights_std) * torch.cat(
        #     [T(pop_size * [0.0818]), T(pop_size * [0.0515]), T(pop_size * [0.1350]),
        #      T(pop_size * [0.1597])])
        weights_excit_L2_3 = (torch.ones((pop_size, 1)) + (
                    2 * weights_std * torch.randn((pop_size, N))) - weights_std) * torch.cat(
            [T(pop_size * [1.245]), T(pop_size * [1.245]), T(pop_size * [1.245]), T(pop_size * [1.245])])
        weights_inhib_L2_3 = (torch.ones((pop_size, 1)) + (
                    2 * weights_std * torch.randn((pop_size, N))) - weights_std) * torch.cat(
            [T(pop_size * [-4.964]), T(pop_size * [-4.964]), T(pop_size * [-4.964]), T(pop_size * [-4.964])])
        weights_excit_L4 = (torch.ones((pop_size, 1)) + (
                    2 * weights_std * torch.randn((pop_size, N))) - weights_std) * torch.cat(
            [T(pop_size * [1.245]), T(pop_size * [1.245]), T(pop_size * [2.482]), T(pop_size * [1.245])])
        weights_inhib_L4 = (torch.ones((pop_size, 1)) + (
                    2 * weights_std * torch.randn((pop_size, N))) - weights_std) * torch.cat(
            [T(pop_size * [-4.964]), T(pop_size * [-4.964]), T(pop_size * [-4.964]), T(pop_size * [-4.964])])
        # weights_excit_L2_3 = torch.ones((pop_size, N))
        # weights_inhib_L2_3 = torch.ones((pop_size, N))
        # weights_excit_L4 = torch.ones((pop_size, N))
        # weights_inhib_L4 = torch.ones((pop_size, N))

        E_L_origo = 0.
        params_pop_excit_L2_3 = {'tau_m': 10., 'tau_s': 3., 'Delta_u': 5., 'tau_theta': 1000., 'c': 0.1,
                              'J_theta': 1., 'E_L': E_L_origo+0., 'R_m': 0.001, 'pop_sizes': 413}
        hand_coded_params_pop_excit_L2_3 = {'preset_weights': weights_excit_L2_3 }
        params_pop_inhib_L2_3 = {'tau_m': 10., 'tau_s': 6., 'Delta_u': 5., 'tau_theta': 1000., 'c': 0.1,
                              'J_theta': 0., 'E_L': E_L_origo+0., 'R_m': 0.001, 'pop_sizes': 116}
        hand_coded_params_pop_inhib_L2_3 = {'preset_weights': weights_inhib_L2_3 }

        params_pop_excit_L4 = {'tau_m': 10., 'tau_s': 3., 'Delta_u': 5., 'tau_theta': 1000., 'c': 0.1,
                              'J_theta': 1., 'E_L': E_L_origo+7, 'R_m': 19., 'pop_sizes': 438}
        hand_coded_params_pop_excit_L4 = {'preset_weights': weights_excit_L4 }
        params_pop_inhib_L4 = {'tau_m': 10., 'tau_s': 6., 'Delta_u': 5., 'tau_theta': 1000., 'c': 0.1,
                              'J_theta': 0., 'E_L': E_L_origo+2, 'R_m': 11.964, 'pop_sizes': 109}
        hand_coded_params_pop_inhib_L4 = {'preset_weights': weights_inhib_L4 }

        params_pop_excit_L2_3 = randomise_parameters(params_pop_excit_L2_3, coeff=T(0.), N_dim=pop_size)
        params_pop_excit_L2_3 = zip_tensor_dicts(params_pop_excit_L2_3, hand_coded_params_pop_excit_L2_3)
        params_pop_inhib_L2_3 = randomise_parameters(params_pop_inhib_L2_3, coeff=T(0.), N_dim=pop_size)
        params_pop_inhib_L2_3 = zip_tensor_dicts(params_pop_inhib_L2_3, hand_coded_params_pop_inhib_L2_3)
        params_pop_excit_L4 = randomise_parameters(params_pop_excit_L4, coeff=T(0.), N_dim=pop_size)
        params_pop_excit_L4 = zip_tensor_dicts(params_pop_excit_L4, hand_coded_params_pop_excit_L4)
        params_pop_inhib_L4 = randomise_parameters(params_pop_inhib_L4, coeff=T(0.), N_dim=pop_size)
        params_pop_inhib_L4 = zip_tensor_dicts(params_pop_inhib_L4, hand_coded_params_pop_inhib_L4)

        rand_params_L2_3 = zip_tensor_dicts(params_pop_excit_L2_3, params_pop_inhib_L2_3)
        rand_params_L4 = zip_tensor_dicts(params_pop_excit_L4, params_pop_inhib_L4)
        randomised_params = zip_tensor_dicts(rand_params_L2_3, rand_params_L4)
    else:
        raise NotImplementedError("Model only supports 2 or 4 populations.")

    # neuron_types = T([1,1, -1,-1, 1,1, -1,-1])
    pop_types = [1, -1, 1, -1]
    neuron_types = torch.ones((N,))
    for n_i in range(N_pops):
        for n_j in range(pop_size):
            ind = n_i*pop_size + n_j
            neuron_types[ind] = pop_types[n_i]
    return microGIF(parameters=randomised_params, N=N, neuron_types=neuron_types)


def non_differentiable_micro_gif_populations_model(random_seed, pop_size=1, N_pops=2):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Read Schwalger et al. (2017), really figure out the parameters here. Also, have another look at whether this
    #   should work, or if we need to work with larger bins etc.
    # Conclusion: E_L set too high, but they implemented tonically firing neurons, i.e. model with stationary
    #  firing rates. I've tested that the rates differ without input, and without weights.

    weights_std = 0.0
    # weights_std = 0

    N = N_pops * pop_size
    origo_E_L = 0.
    time_scale = 1.0

    if N_pops == 2:
        weights_excit_1 = (torch.ones((pop_size, 1)) + (2 * weights_std * torch.randn((pop_size, N))) - weights_std) * \
                          torch.cat([T(pop_size * [2.482*0.0497]), T(pop_size * [1.245*0.0794])])
        weights_inhib_1 = (torch.ones((pop_size, 1)) + (2 * weights_std * torch.randn((pop_size, N))) - weights_std) * \
                          torch.cat([T(pop_size * [-4.964*0.1350]), T(pop_size * [-4.964*0.1597])])

        # Excitatory:
        params_pop_excit_1 = {'tau_m': time_scale*10., 'tau_s': time_scale*3., 'Delta_u': 5., 'tau_theta': time_scale*1000.,
                              'c': 0.1, 'J_theta': 1., 'E_L': origo_E_L+2., 'R_m': 19., 'pop_sizes': 438 }
        hand_coded_params_pop_excit_1 = {'preset_weights': weights_excit_1}

        # Inhibitory
        params_pop_inhib_1 = {'tau_m': time_scale*10., 'tau_s': time_scale*6., 'Delta_u': 5., 'tau_theta': time_scale*1000.,
                              'c': 0.1, 'J_theta': 1., 'E_L': origo_E_L+1.5, 'R_m': 11.964, 'pop_sizes': 109 }
        hand_coded_params_pop_inhib_1 = {'preset_weights': weights_inhib_1}

        params_pop_excit_1 = randomise_parameters(params_pop_excit_1, coeff=T(0.), N_dim=pop_size)
        params_pop_excit_1 = zip_tensor_dicts(params_pop_excit_1, hand_coded_params_pop_excit_1)
        params_pop_inhib_1 = randomise_parameters(params_pop_inhib_1, coeff=T(0.), N_dim=pop_size)
        params_pop_inhib_1 = zip_tensor_dicts(params_pop_inhib_1, hand_coded_params_pop_inhib_1)

        randomised_params = zip_tensor_dicts(params_pop_excit_1, params_pop_inhib_1)
    elif N_pops == 4:
        weights_excit_1 = (torch.ones((pop_size, 1)) + (
                    2 * weights_std * torch.randn((pop_size, N))) - weights_std) * torch.cat(
            [T(pop_size * [1.245*0.1009]), T(pop_size * [1.245*0.1346]), T(pop_size * [1.245*0.0077]), T(pop_size * [1.245*0.0691])])
        weights_excit_2 = (torch.ones((pop_size, 1)) + (
                    2 * weights_std * torch.randn((pop_size, N))) - weights_std) * torch.cat(
            [T(pop_size * [1.245*0.1689]), T(pop_size * [1.245*0.1371]), T(pop_size * [2.482*0.0059]), T(pop_size * [1.245*0.0029])])
        weights_inhib_1 = (torch.ones((pop_size, 1)) + (
                    2 * weights_std * torch.randn((pop_size, N))) - weights_std) * torch.cat(
            [T(pop_size * [-4.964*0.0437]), T(pop_size * [-4.964*0.0316]), T(pop_size * [-4.964*0.0497]), T(pop_size * [-4.964*0.0794])])
        weights_inhib_2 = (torch.ones((pop_size, 1)) + (
                    2 * weights_std * torch.randn((pop_size, N))) - weights_std) * torch.cat(
            [T(pop_size * [-4.964*0.0818]), T(pop_size * [-4.964*0.0515]), T(pop_size * [-4.964*0.1350]), T(pop_size * [-4.964*0.1597])])

        # Excitatory:
        params_pop_excit_1 = {'tau_m': time_scale*10., 'tau_s': time_scale*3., 'Delta_u': 5., 'tau_theta': time_scale*1000.,
                              'c': 0.1, 'J_theta': 1., 'E_L': origo_E_L+0., 'R_m': 0.001, 'pop_sizes': 413}
        hand_coded_params_pop_excit_1 = {'preset_weights': weights_excit_1 }
        params_pop_excit_2 = {'tau_m': time_scale*10., 'tau_s': 3., 'Delta_u': 5., 'tau_theta': time_scale*1000.,
                              'c': 0.1, 'J_theta': 1., 'E_L': origo_E_L+7., 'R_m': 19., 'pop_sizes': 438}
        hand_coded_params_pop_excit_2 = {'preset_weights': weights_excit_2 }

        # Inhibitory
        params_pop_inhib_1 = {'tau_m': time_scale*10., 'tau_s': time_scale*6., 'Delta_u': 5., 'tau_theta': time_scale*1000.,
                              'c': 0.1, 'J_theta': 0., 'E_L': origo_E_L+0., 'R_m': 0.001, 'pop_sizes': 116}
        hand_coded_params_pop_inhib_1 = {'preset_weights': weights_inhib_1 }
        params_pop_inhib_2 = {'tau_m': time_scale*10., 'tau_s': time_scale*6., 'Delta_u': 5., 'tau_theta': time_scale*1000.,
                              'c': 0.1, 'J_theta': 0., 'E_L': origo_E_L+5., 'R_m': 11.964, 'pop_sizes': 109}
        hand_coded_params_pop_inhib_2 = {'preset_weights': weights_inhib_2 }

        # params_pop_excit_1 = {'tau_m': 100., 'tau_s': 30., 'Delta_u': 5., 'tau_theta': 10000., 'c': 1.,
        #                       'J_theta': 1., 'E_L': 18., 'R_m': 0.001, 'pop_sizes': 413}
        # hand_coded_params_pop_excit_1 = {'preset_weights': weights_excit_1}
        # params_pop_excit_2 = {'tau_m': 10., 'tau_s': 30., 'Delta_u': 5., 'tau_theta': 10000., 'c': 1.,
        #                       'J_theta': 1., 'E_L': 25., 'R_m': 19., 'pop_sizes': 438}
        # hand_coded_params_pop_excit_2 = {'preset_weights': weights_excit_2}
        #
        # # Inhibitory
        # params_pop_inhib_1 = {'tau_m': 100., 'tau_s': 60., 'Delta_u': 5., 'tau_theta': 10000., 'c': 1.,
        #                       'J_theta': 0., 'E_L': 18., 'R_m': 0.001, 'pop_sizes': 116}
        # hand_coded_params_pop_inhib_1 = {'preset_weights': weights_inhib_1}
        # params_pop_inhib_2 = {'tau_m': 100., 'tau_s': 60., 'Delta_u': 5., 'tau_theta': 10000., 'c': 1.,
        #                       'J_theta': 0., 'E_L': 20., 'R_m': 11.964, 'pop_sizes': 109}
        # hand_coded_params_pop_inhib_2 = {'preset_weights': weights_inhib_2}

        params_pop_excit_1 = randomise_parameters(params_pop_excit_1, coeff=T(0.), N_dim=pop_size)
        params_pop_excit_1 = zip_tensor_dicts(params_pop_excit_1, hand_coded_params_pop_excit_1)
        params_pop_excit_2 = randomise_parameters(params_pop_excit_2, coeff=T(0.), N_dim=pop_size)
        params_pop_excit_2 = zip_tensor_dicts(params_pop_excit_2, hand_coded_params_pop_excit_2)
        params_pop_inhib_1 = randomise_parameters(params_pop_inhib_1, coeff=T(0.), N_dim=pop_size)
        params_pop_inhib_1 = zip_tensor_dicts(params_pop_inhib_1, hand_coded_params_pop_inhib_1)
        params_pop_inhib_2 = randomise_parameters(params_pop_inhib_2, coeff=T(0.), N_dim=pop_size)
        params_pop_inhib_2 = zip_tensor_dicts(params_pop_inhib_2, hand_coded_params_pop_inhib_2)

        rand_params_excit_pops = zip_tensor_dicts(params_pop_excit_1, params_pop_excit_2)
        rand_params_inhib_pops = zip_tensor_dicts(params_pop_inhib_1, params_pop_inhib_2)
        randomised_params = zip_tensor_dicts(rand_params_excit_pops, rand_params_inhib_pops)
    else:
        raise NotImplementedError("Model only supports 2 or 4 populations.")

    neuron_types = np.ones((N,))
    for i in range(int(N / 2)):
        neuron_types[-(1 + i)] = -1
    return nonDifferentiableMicroGIF(parameters=randomised_params, N=N, neuron_types=neuron_types)
