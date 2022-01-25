import torch
import numpy as np

import model_util
import spike_metrics
from TargetModels import TargetEnsembleModels
from experiments import sine_modulated_white_noise_input
from plot import plot_spike_trains_side_by_side

for random_seed in range(15):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    num_neurons = 12
    # params_model = draw_from_uniform(GLIF.parameter_init_intervals, num_neurons)
    # snn = GLIF(parameters=params_model, N=num_neurons)

    # static_parameters = {'N': 10}
    # free_parameters = {'w_mean': 0.2, 'w_var': 0.3}
    # snn = GLIF(device='cpu', parameters=zip_dicts(static_parameters, free_parameters))
    # snn = TargetEnsembleModels.glif_ensembles_model(random_seed=random_seed, N = 12); ext_name = 'ensembles_1'
    snn = TargetEnsembleModels.glif_ensembles_model_dales_compliant(random_seed=random_seed, N = 12)
    ext_name = 'ensembles_{}_dales_GLIF'.format(random_seed)
    # snn = TargetEnsembleModels.lif_ensembles_model_dales_compliant(random_seed=random_seed, N = 12)
    # ext_name = 'ensembles_{}_dales_LIF'.format(random_seed)
    # snn = TargetModels.glif1(N = 12); ext_name = '1'
    # snn = TargetModels.glif1_2(N = 12); ext_name = '1_2'
    # snn = TargetModels.glif2(N = 12); ext_name = '2'
    # snn = TargetModels.glif3(N = 12); ext_name = '3'
    # snn = TargetModels.glif_async(N = 12); ext_name = 'glif_async'
    # snn = TargetModels.glif_slower_more_synchronous(N = 12); ext_name = 'glif_slower_more_synchronous'
    # print(list(snn.parameters())[0])

    rate = 10.
    inputs = sine_modulated_white_noise_input(rate, t=12000, N=snn.N)  # rate in Hz
    print('#inputs: {}'.format(inputs.sum()))
    # membrane_potentials, spikes = model_util.feed_inputs_sequentially_return_spikes_and_potentials(snn, inputs)
    spikes = model_util.feed_inputs_sequentially_return_spike_train(snn, inputs)
    print('#spikes: {}'.format(torch.round(spikes).sum(dim=0)))
    # plot_neuron(membrane_potentials.data, title='GLIF neuron plot ({:.2f} spikes)'.format(spikes.sum()), fname_ext='test_GLIF_poisson_input' + '_' + str(random_seed))
    assert spikes.sum() < 12. * 12 * 12 * 2, "should be less spikes than input rate * 2 per neuron. spikes.sum(): {}".format(spikes.sum())

    zeros = torch.zeros_like(inputs)
    # membrane_potentials_zeros, spikes_zeros = model_util.feed_inputs_sequentially_return_spikes_and_potentials(snn, zeros)
    spikes_zeros = model_util.feed_inputs_sequentially_return_spike_train(snn, zeros)
    # plot_neuron(membrane_potentials_zeros.data, title='Neuron plot ({:.2f} spikes)'.format(spikes_zeros.sum()), fname_ext='test_GLIF_no_input'  + '_' + str(random_seed))

    plot_spike_trains_side_by_side(spikes, spikes_zeros, 'test_SNNs', title='{} random ({} Hz) and zero input'.format(ext_name, rate),
                                   legend=['Poisson input', 'No input'])

    tau_vr = torch.tensor(4.0)
    loss = spike_metrics.van_rossum_dist(spikes, spikes_zeros, tau=tau_vr)
    print('tau_vr: {}, loss: {}'.format(tau_vr, loss))
    assert loss > 1e-03, "should have loss for input vs. no input"
