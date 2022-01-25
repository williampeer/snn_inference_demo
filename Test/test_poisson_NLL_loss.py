import torch
from torch.nn.functional import poisson_nll_loss

import model_util
from Models.GLIF import GLIF
from experiments import sine_modulated_white_noise_input, randomise_parameters, zip_dicts


def test_poisson_NLL():
    tar_spikes = sine_modulated_white_noise_input(1. * torch.ones((12,)), t=100, N=12)
    model_spikes = sine_modulated_white_noise_input(1. * torch.ones((12,)), t=100, N=12)
    zeros = torch.zeros_like(tar_spikes)
    print('num of sample model spikes: {}'.format(model_spikes.sum()))
    print('num of sample target spikes: {}'.format(tar_spikes.sum()))
    print('num of spikes in zeros: {}'.format(zeros.sum()))

    loss = poisson_nll_loss(model_spikes, tar_spikes)
    print('poisson nll tar vs spikes: {}'.format(loss))

    loss_zeros = poisson_nll_loss(zeros.clone(), zeros)
    print('poisson nll zeros: {}'.format(loss_zeros))
    assert loss_zeros == 1.0, "distance between silent trains should be approximately zero. was: {}".format(loss_zeros)

    loss_model_spikes_zeros = poisson_nll_loss(model_spikes, zeros)
    print('loss spikes vs zeros: {}'.format(loss_model_spikes_zeros))
    assert loss_model_spikes_zeros > loss_zeros, "spikes should result in greater loss with spikes than no spikes with no spikes as target"


def test_poisson_NLL_models():
    static_parameters = {'N': 3}
    free_parameters = {'w_mean': 0.2, 'w_var': 0.3, 'tau_m': 1.5, 'tau_g': 4.0, 'v_rest': -60.0}
    m1 = GLIF(device='cpu', parameters=zip_dicts(static_parameters, free_parameters))
    m2 = GLIF(device='cpu', parameters=zip_dicts(static_parameters, randomise_parameters(free_parameters, coeff=torch.tensor(0.25))))

    inputs = sine_modulated_white_noise_input(0.5, t=500, N=static_parameters['N'])
    membrane_potentials, spikes1 = model_util.feed_inputs_sequentially_return_tuple(m1, inputs)
    membrane_potentials, spikes1_2 = model_util.feed_inputs_sequentially_return_tuple(m1, inputs)
    membrane_potentials, spikes2 = model_util.feed_inputs_sequentially_return_tuple(m2, inputs)

    print('num of sample model1 spikes1: {}'.format(spikes1.sum()))
    print('num of sample model1 spikes2: {}'.format(spikes1_2.sum()))
    print('num of sample model2 spikes: {}'.format(spikes2.sum()))

    loss = poisson_nll_loss(spikes1, spikes1_2)
    print('poisson nll s1 vs s1_2: {}'.format(loss))

    loss2 = poisson_nll_loss(spikes1, spikes2)
    print('poisson nll s1 vs s2: {}'.format(loss2))

# --------------------------------------
test_poisson_NLL()
test_poisson_NLL_models()
