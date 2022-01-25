import sys

import numpy as np
import torch

import experiments
import model_util
import spike_metrics
from Models.microGIF import microGIF
from TargetModels import TargetModelMicroGIF
from plot import plot_spike_trains_side_by_side, plot_neuron

# num_pops = 2
num_pops = 4
pop_size = 2
# pop_size = 2
# pop_size = 1
t = 1200

for random_seed in range(3, 4):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    pop_sizes, snn = TargetModelMicroGIF.get_low_dim_micro_GIF_transposed(random_seed=random_seed)
    snn = microGIF(snn.get_parameters(), snn.N)

    N = snn.N

    A_coeffs = []
    A_coeffs.append(torch.randn((4,)))
    phase_shifts = []
    phase_shifts.append(torch.randn((4,)))
    # input_types = [1, 2, 2, 2]
    input_types = [1, 1, 1, 1]
    all_inputs = experiments.generate_composite_input_of_white_noise_modulated_sine_waves(t, A_coeffs, phase_shifts, input_types)

    print('- SNN test for class {} -'.format(snn.__class__.__name__))
    print('#inputs sum: {}'.format(all_inputs.sum()))
    snn.reset()
    _, spikes, vs = model_util.feed_inputs_sequentially_return_args(snn, all_inputs)
    spikes = spikes.clone().detach()
    # spikes = model_util.feed_inputs_sequentially_return_spike_train(snn, inputs)
    # print('snn weights: {}'.format(snn.w))
    hard_thresh_spikes_sum = torch.round(spikes).sum()
    print('spikes sum: {}'.format(hard_thresh_spikes_sum))
    soft_thresh_spikes_sum = (spikes > 0.333).sum()
    zero_thresh_spikes_sum = (spikes > 0).sum()
    print('thresholded spikes sum: {}'.format(torch.round(spikes).sum()))
    print('=========avg. hard rate: {}'.format(1000*hard_thresh_spikes_sum / (spikes.shape[1] * spikes.shape[0])))
    print('=========avg. soft rate: {}'.format(1000*soft_thresh_spikes_sum / (spikes.shape[1] * spikes.shape[0])))
    print('=========avg. zero thresh rate: {}'.format(1000*zero_thresh_spikes_sum / (spikes.shape[1] * spikes.shape[0])))

    zeros = torch.zeros_like(all_inputs)
    snn.reset()
    _, spikes_zero_input, membrane_potentials_zero_input = model_util.feed_inputs_sequentially_return_args(snn, zeros)
    print('#spikes no input: {}'.format(torch.round(spikes_zero_input).sum()))

    snn.w = torch.nn.Parameter(torch.zeros((snn.N, snn.N)), requires_grad=True)

    snn.reset()
    _, spikes_zero_weights, membrane_potentials_zero_weights = model_util.feed_inputs_sequentially_return_args(snn, all_inputs)

    print('#spikes no weights: {}'.format(torch.round(spikes_zero_weights).sum()))
    plot_neuron(all_inputs.detach().numpy(), title='I_ext'.format(snn.name(), spikes.sum()),
                uuid='test_{}'.format(snn.__class__.__name__), fname='test_ext_input'.format(snn.name()) + '_' + str(random_seed))
    plot_neuron(vs.detach().numpy(), title='{} neuron plot ({:.2f} spikes)'.format(snn.name(), spikes.sum()),
                uuid='test_{}'.format(snn.__class__.__name__), fname='test_membrane_potential_{}_ext_input'.format(snn.name()) + '_' + str(random_seed))
    plot_neuron(membrane_potentials_zero_weights.detach().numpy(), title='{} neuron plot ({:.2f} spikes)'.format(snn.name(), spikes.sum()),
                uuid='test_{}'.format(snn.__class__.__name__), fname='test_membrane_potential_{}_no_weights'.format(snn.name()) + '_' + str(random_seed))
    plot_neuron(membrane_potentials_zero_input.detach().numpy(), title='{} neuron plot ({:.2f} spikes)'.format(snn.name(), spikes.sum()),
                uuid='test_{}'.format(snn.__class__.__name__), fname='test_membrane_potential_{}_zero_input'.format(snn.name()) + '_' + str(random_seed))
    plot_spike_trains_side_by_side(spikes, spikes_zero_weights, 'test_{}'.format(snn.__class__.__name__),
                                   title='Test {} spiketrains random input'.format(snn.__class__.__name__),
                                   legend=['Random weights', 'No weights'])
    plot_spike_trains_side_by_side(spikes, spikes_zero_input, 'test_{}'.format(snn.__class__.__name__),
                                   title='Test {} spiketrains no input'.format(snn.__class__.__name__),
                                   legend=['Rand (weights & input)', 'Rand weights, no input'])

    tau_vr = torch.tensor(5.0)
    loss = spike_metrics.van_rossum_dist(spikes, spikes_zero_input, tau=tau_vr)
    print('tau_vr: {}, loss: {}'.format(tau_vr, loss))
    loss_rate = spike_metrics.firing_rate_distance(spikes, spikes_zero_weights)
    print('firing rate loss: {}'.format(loss_rate))


sys.exit(0)
