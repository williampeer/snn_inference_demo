import sys

import numpy as np
import torch
from torch import tensor as T

import model_util
import spike_metrics
from Models.LIF import LIF
from TargetModels import TargetModelsSoft
import experiments
from plot import plot_spike_trains_side_by_side, plot_spike_train_projection, plot_neuron

# num_pops = 2
num_pops = 4
# pop_size = 1
pop_size = 2
# pop_size = 4

for random_seed in range(3, 5):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    # tar_snn = TargetModelsSoft.lif_pop_model(random_seed=random_seed, pop_size=pop_size, N_pops=num_pops)
    num_neurons = num_pops * pop_size
    pop_types = [1, -1, 1, -1]
    neuron_types = np.ones((num_neurons,))
    N_pops = 4
    pop_size = int(num_neurons / N_pops)
    for n_i in range(N_pops):
        for n_j in range(pop_size):
            ind = n_i * pop_size + n_j
            neuron_types[ind] = pop_types[n_i]

    init_params_model = experiments.draw_from_uniform(LIF.parameter_init_intervals, num_neurons)
    snn = LIF(parameters=init_params_model, N=num_neurons, neuron_types=neuron_types)

    inputs = experiments.sine_modulated_white_noise(t=4800, N=snn.N, neurons_coeff=torch.cat([T(int(snn.N/4) * [0.]), T(int(snn.N/4) * [0.]), T(int(snn.N/4) * [0.5]), T(int(snn.N/4) * [0.])]))
    # inputs = experiments.strong_sine_modulated_white_noise(t=4800, N=snn.N, neurons_coeff=torch.cat([T(int(snn.N / 2) * [1.]), T(int(snn.N / 2) * [0.])]))
    # inputs = experiments.sine_input(t=4800, N=snn.N, neurons_coeff=torch.cat([T(int(snn.N / 4) * [1.]), T(int(snn.N / 4) * [0.]), T(int(snn.N / 2) * [0.])]))

    print('- SNN test for class {} -'.format(snn.__class__.__name__))
    print('#inputs: {}'.format(inputs.sum()))
    membrane_potentials, spikes = model_util.feed_inputs_sequentially_return_tuple(snn, inputs)
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
    plot_spike_train_projection(spikes, fname='test_projection_{}_ext_input'.format(snn.__class__.__name__) + '_' + str(random_seed))

    # zeros = torch.zeros_like(inputs)
    # spikes_zeros = model_util.feed_inputs_sequentially_return_spike_train(snn, zeros)
    # print('#spikes no input: {}'.format(torch.round(spikes_zeros).sum()))

    # snn.w = torch.nn.Parameter(torch.zeros((snn.v.shape[0],snn.v.shape[0])), requires_grad=True)
    # spikes_zero_weights = model_util.feed_inputs_sequentially_return_spike_train(snn, inputs)
    # print('#spikes no weights: {}'.format(torch.round(spikes_zero_weights).sum()))
    plot_neuron(inputs.detach().numpy(), title='I_ext'.format(snn.name(), spikes.sum()),
                uuid='test', fname='test_ext_input_{}'.format(snn.name()) + '_' + str(random_seed))
    plot_neuron(inputs.detach().numpy(), title='membrane potential {} spikes: {}'.format(snn.name(), spikes.sum()),
                uuid='test', fname='test_V_{}'.format(snn.name()) + '_' + str(random_seed))
    # plot_spike_trains_side_by_side(spikes, spikes_zero_weights, 'test_{}'.format(snn.__class__.__name__),
    #                                title='Test {} spiketrains random input'.format(snn.__class__.__name__),
    #                                legend=['Random weights', 'No weights'])

    # tau_vr = torch.tensor(5.0)
    # loss = spike_metrics.van_rossum_dist(spikes, spikes_zeros, tau=tau_vr)
    # print('tau_vr: {}, loss: {}'.format(tau_vr, loss))
    # loss_rate = spike_metrics.firing_rate_distance(spikes, spikes_zero_weights)
    # print('firing rate loss: {}'.format(loss_rate))


sys.exit(0)
