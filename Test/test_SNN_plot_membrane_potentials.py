import torch
import numpy as np

import model_util
import plot
import spike_metrics
from Models.GLIF import GLIF
from Models.LIF import LIF
from Models.LIF_ASC import LIF_ASC
from Models.LIF_R import LIF_R
from Models.LIF_R_ASC import LIF_R_ASC
from TargetModels import TargetModels
from experiments import sine_modulated_white_noise, draw_from_uniform
from plot import plot_neuron, plot_spike_trains_side_by_side

num_neurons = 4

for random_seed in range(1, 2):
    # snn = lif_ensembles_model_dales_compliant(random_seed=random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    model_class = LIF
    init_params_model = draw_from_uniform(model_class.parameter_init_intervals, N=num_neurons)
    snn = model_class(init_params_model, N=num_neurons, neuron_types=[1, 1, 1, -1])
    # snn = TargetModels.lif_HS_17_continuous_ensembles_model_dales_compliant(random_seed=random_seed)
    # snn = TargetModels.lif_continuous_ensembles_model_dales_compliant(random_seed=random_seed)
    # snn = TargetModels.lif_r_continuous_ensembles_model_dales_compliant(random_seed=random_seed)
    # snn = TargetModels.lif_asc_continuous_ensembles_model_dales_compliant(random_seed=random_seed)
    # snn = TargetModels.lif_r_asc_continuous_ensembles_model_dales_compliant(random_seed=random_seed)
    # snn = TargetModels.glif_continuous_ensembles_model_dales_compliant(random_seed=random_seed)

    inputs = sine_modulated_white_noise(10., t=200, N=snn.N)  # now assumes rate in Hz

    print('- SNN test for class {} -'.format(snn.__class__.__name__))
    print('#inputs: {}'.format(inputs.sum()))
    # membrane_potentials, spikes = model_util.feed_inputs_sequentially_return_spikes_and_potentials(snn, inputs)
    spikes = model_util.feed_inputs_sequentially_return_spike_train(snn, inputs)
    # print('snn weights: {}'.format(snn.w))
    print('#spikes: {}'.format(torch.round(spikes).sum()))
    print('=========avg. rate: {}'.format(1000*torch.round(spikes).sum() / (spikes.shape[1] * spikes.shape[0])))
    # plot_neuron(membrane_potentials.detach().numpy(), title='{} neuron plot ({:.2f} spikes)'.format(snn.__class__.__name__, spikes.sum()),
    #             uuid='test', fname='test_{}_poisson_input'.format(snn.__class__.__name__) + '_' + str(random_seed))

    zeros = torch.zeros_like(inputs)
    # membrane_potentials_zeros, spikes_zeros = model_util.feed_inputs_sequentially_return_spikes_and_potentials(snn, zeros)
    spikes_zeros = model_util.feed_inputs_sequentially_return_spike_train(snn, zeros)
    # plot_neuron(membrane_potentials_zeros.detach().numpy(), title='Neuron plot ({:.2f} spikes)'.format(spikes_zeros.sum()),
    #             uuid='test', fname='test_{}_no_input'.format(snn.__class__.__name__)  + '_' + str(random_seed))
    print('#spikes no input: {}'.format(torch.round(spikes_zeros).sum()))

    # plot_spike_trains_side_by_side(spikes, spikes_zeros, 'test_LIF', title='Test LIF spiketrains random and zero input', legend=['Poisson input', 'No input'])
    snn.w = torch.nn.Parameter(torch.zeros((snn.v.shape[0],snn.v.shape[0])), requires_grad=True)
    # membrane_potentials_zero_weights, spikes_zero_weights = model_util.feed_inputs_sequentially_return_spikes_and_potentials(snn, inputs)
    spikes_zero_weights = model_util.feed_inputs_sequentially_return_spike_train(snn, inputs)
    print('#spikes no weights: {}'.format(torch.round(spikes_zero_weights).sum()))
    # plot_neuron(membrane_potentials.detach().numpy(), title='LIF neuron plot ({:.2f} spikes)'.format(spikes.sum()),
    #             uuid='test', fname='test_{}_poisson_input'.format(snn.__class__.__name__) + '_' + str(random_seed))
    plot_spike_trains_side_by_side(spikes, spikes_zero_weights, 'test_{}'.format(snn.__class__.__name__),
                                   title='Test {} spiketrains random input'.format(snn.__class__.__name__),
                                   legend=['Random weights', 'No weights'])

    tau_vr = torch.tensor(4.5)
    loss = spike_metrics.van_rossum_dist(spikes, spikes_zeros, tau=tau_vr)
    plot.plot_neuron(spikes.detach().numpy(), 'test_plot_s', exp_type='test', ylabel='$S$', fname='plot_S')
    plot.plot_neuron(spike_metrics.torch_van_rossum_convolution(spikes, tau_vr).detach().numpy(), 'test_plot_S',
                     exp_type='test', ylabel='$f_{conv}(S)$', fname='plot_S_conv')

    print('tau_vr: {}, loss: {}'.format(tau_vr, loss))
