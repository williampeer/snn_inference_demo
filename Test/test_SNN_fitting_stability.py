import torch

from Constants import Constants
from Models.GLIF import GLIF
from Models.LIF import LIF
from experiments import sine_modulated_white_noise_input, zip_dicts
from fit import fit_batches
from model_util import generate_model_data
from plot import plot_neuron, plot_losses


def test_stability_with_matching_configurations_deprecated(model, gen_model, rate_factor, tau_vr, learn_rate, optim):
    t=4000

    poisson_rate = torch.tensor(rate_factor)
    gen_inputs = sine_modulated_white_noise_input(poisson_rate, t, gen_model.N)
    gen_membrane_potentials, gen_spiketrain = generate_model_data(model=gen_model, inputs=gen_inputs)

    optims = [optim(list(model.parameters()), lr=learn_rate),
              optim([poisson_rate], lr=learn_rate)]
    for neur_ind in range(gen_membrane_potentials.shape[1]):
        plot_neuron(gen_membrane_potentials[:, neur_ind].data, title='Generative neuron model #{}'.format(neur_ind),
                    fname_ext='_test_1_neuron_{}'.format(neur_ind))

    avg_batch_loss = fit_batches(model=model, gen_inputs=gen_inputs,
                                 target_spiketrain=gen_spiketrain,
                                 poisson_input_rate=poisson_rate,
                                 optimiser=optims,
                                 constants=Constants(0, 0, 0, 500, tau_vr, 0.6, 2000, optim, 'van_rossum_dist', 1))

    for param_i, param in enumerate(list(model.parameters())):
        # print('parameter #{}: {}'.format(param_i, param))
        assert param.grad is not None, "gradient was none. param #{}, \nparam: {}\nparam.grad: {}" \
            .format(param_i, param, param.grad)
        assert torch.abs(param.grad.sum()) > 1e-08, "gradients should not be zero. param #{}, " \
                                                    "\nparam: {}\nparam.grad: {}".format(param_i, param, param.grad)

    return avg_batch_loss


def test_stability_with_matching_configurations(model, gen_model, rate_factor, tau_vr, learn_rate, optim, train_iter_cap):
    t=4000

    gen_model_rate = torch.tensor(rate_factor)
    model_rate = gen_model_rate.clone().detach()
    optims = [optim(list(model.parameters()), lr=learn_rate),
              optim([model_rate], lr=learn_rate)]

    batch_losses = []
    train_iter = 0; avg_batch_loss = 10
    while train_iter < train_iter_cap and avg_batch_loss > 5.0:
        gen_inputs = sine_modulated_white_noise_input(gen_model_rate, t, gen_model.N)
        gen_model.reset_hidden_state()
        # for gen spiketrain this may be thresholded to binary values:
        gen_membrane_potentials, targets = generate_model_data(model=gen_model, inputs=gen_inputs)
        gen_membrane_potentials = gen_membrane_potentials.clone().detach()
        gen_spiketrain = targets.clone().detach()
        plot_neuron(gen_membrane_potentials.data, title='LIF_test neuron plot ({:.2f} spikes)'.format(gen_spiketrain.sum()), fname_ext='test_LIF_poisson_input')
        assert gen_spiketrain.sum() > 10., "gen spike train had almost no spike. #spikes: {}".format(gen_spiketrain.sum())

        # for neur_ind in range(gen_membrane_potentials.shape[1]):
        #     plot_neuron(gen_membrane_potentials[:, neur_ind].data, title='Generative neuron model #{}'.format(neur_ind),
        #                 fname_ext='_test_2_neuron_{}'.format(neur_ind))
        del gen_membrane_potentials, targets, gen_inputs

        avg_batch_loss = fit_batches(model=model, gen_inputs=None, target_spiketrain=gen_spiketrain, poisson_input_rate=model_rate,
                                     optimiser=optims,
                                     constants=Constants(0, 0, 0, 500, tau_vr, model_rate, 2000, optim, 'van_rossum_dist', 1))
        model_rate = model_rate.clone().detach()  # reset

        # for param_i, param in enumerate(list(model.parameters())):
        #     # print('parameter #{}: {}'.format(param_i, param))
        #     # if param.grad is not None:
        #     assert param.grad is not None, "gradient was none. param #{}, \nparam: {}\nparam.grad: {}"\
        #         .format(param_i, param, param.grad)
        #     assert torch.abs(param.grad.sum()) != 0, "gradients should not be zero. param #{}, " \
        #                                                 "\nparam: {}\nparam.grad: {}".format(param_i, param, param.grad)

        train_iter += 1
        batch_losses.append(avg_batch_loss)

    plot_losses(batch_losses, [], 1, 'test_SNN_fitting_stability',
                custom_title='Avg. batch loss ({}, {}, lr={})'.format(model.__name__, optim.__name__, learn_rate))



# gen_model = BaselineSNN.BaselineSNN(device='cpu', parameters={}, N=12, w_mean=0.8, w_var=0.6)
# model = BaselineSNN.BaselineSNN(device='cpu', parameters={}, N=12, w_mean=0.8, w_var=0.6)
# test_stability_with_matching_configurations_deprecated(gen_model, model, rate_factor=0.7, tau_vr=tau_vr, learn_rate=0.05)
# test_stability_with_matching_configurations(gen_model, model, rate_factor=0.7, tau_vr=tau_vr, learn_rate=learn_rate, optim=torch.optim.SGD)

# gen_model = LIF.LIF(device='cpu', parameters={}, tau_m=6.5, N=3, w_mean=0.2, w_var=0.25)
# model = LIF.LIF(device='cpu', parameters={}, tau_m=6.5, N=3, w_mean=0.2, w_var=0.25)
# test_stability_with_matching_configurations_and_training_input_noise(gen_model, model, rate_factor=0.5)
# test_stability_with_matching_configurations_different_training_input_noise(gen_model, model, rate_factor=0.5)
#
# gen_model = Izhikevich.Izhikevich(device='cpu', parameters={}, N=3, tau_g=1.0, a=0.1, b=0.27, w_mean=0.15, w_var=0.25)
# model = Izhikevich.Izhikevich(device='cpu', parameters={}, N=3, tau_g=1.0, a=0.1, b=0.27, w_mean=0.15, w_var=0.25)
# test_stability_with_matching_configurations_and_training_input_noise(gen_model, model, rate_factor=0.25)
# test_stability_with_matching_configurations_different_training_input_noise(gen_model, model, rate_factor=0.25)

# gen_model = Izhikevich.Izhikevich_constrained(device='cpu', parameters={}, N=3, a=0.1, b=0.28, w_mean=0.1, w_var=0.25)
# model = Izhikevich.Izhikevich_constrained(device='cpu', parameters={}, N=3, a=0.1, b=0.28, w_mean=0.1, w_var=0.25)
# test_stability_with_matching_configurations_and_training_input_noise(gen_model, model, rate_factor=0.25)
# test_stability_with_matching_configurations_different_training_input_noise(gen_model, model, rate_factor=0.25)

tau_vr = torch.tensor(5.0)
learn_rate = 0.1
static_parameters = {'N': 3, 'w_mean': 0.2, 'w_var': 0.3}
free_parameters = {'tau_g': 2.0}
params = zip_dicts(static_parameters, free_parameters)

gen_model = LIF(device='cpu', parameters=params, R_I=42.)
model = GLIF(device='cpu', parameters=params, R_I=20.)
# model = LIF(device='cpu', parameters=params)
# test_stability_with_matching_configurations_deprecated(model, gen_model, 0.7, tau_vr, learn_rate, optim=torch.optim.Adam)
test_stability_with_matching_configurations_deprecated(model, gen_model, 0.7, tau_vr, learn_rate, optim=torch.optim.SGD)

# test_stability_with_matching_configurations(gen_model, model, rate_factor=0.7, tau_vr=tau_vr,
#                                             learn_rate=learn_rate, optim=torch.optim.SGD, train_iter_cap=15)
# test_stability_with_matching_configurations(gen_model, model, rate_factor=0.7, tau_vr=tau_vr,
#                                             learn_rate=learn_rate, optim=torch.optim.Adam, train_iter_cap=15)
