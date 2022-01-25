import sys

import numpy as np
import torch

import experiments
import model_util
import spike_metrics
from Models.GLIF import GLIF
from Models.GLIF_no_cell_types import GLIF_no_cell_types
from Models.LIF import LIF
from Models.LIF_no_cell_types import LIF_no_cell_types
from plot import plot_spike_trains_side_by_side, plot_neuron

for random_seed in range(23, 30):
    # snn = lif_ensembles_model_dales_compliant(random_seed=random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    # model_class = LIF_soft_weights_only
    # init_params_model = draw_from_uniform(model_class.parameter_init_intervals, num_neurons)
    # snn = model_class(init_params_model)
    # snn = TargetModels.lif_continuous_ensembles_model_dales_compliant(random_seed=random_seed, N=num_neurons)
    # snn = TargetModels.lif_r_continuous_ensembles_model_dales_compliant(random_seed=random_seed, N=num_neurons)
    # snn = TargetModels.lif_r_asc_continuous_ensembles_model_dales_compliant(random_seed=random_seed, N=num_neurons)
    # snn = TargetModelsBestEffort.glif(random_seed=random_seed, N=4)
    # snn = TargetModelsBestEffort.lif(random_seed=random_seed, N=10)
    N = 10
    # model_class = GLIF
    # model_class = GLIF_no_cell_types
    model_class = LIF_no_cell_types
    params_model = experiments.draw_from_uniform(model_class.parameter_init_intervals, N=N)
    neuron_types = N * [1]
    neuron_types[-int(N/3):] = int(N/3) * [-1]
    # snn = model_class(parameters=params_model, N=N, neuron_types=neuron_types)
    snn = model_class(parameters=params_model, N=N)
    # snn = Izhikevich(parameters=params_model, N=4)


    # inputs = sine_modulated_white_noise(10., t=2400., N=snn.N)
    t = 1200
    white_noise = torch.rand((t, snn.N))
    input_type = 'white_noise'
    inputs = white_noise
    # input_type = 'sine_modulated_white_noise_input'
    # inputs = experiments.sine_modulated_input(white_noise)

    print('- SNN test for class {} -'.format(snn.__class__.__name__))
    print('#inputs: {}'.format(inputs.sum()))
    # membrane_potentials, spikes = model_util.feed_inputs_sequentially_return_spikes_and_potentials(snn, inputs)
    vs, spikes = model_util.feed_inputs_sequentially_return_tuple(snn, inputs)
    # print('snn weights: {}'.format(snn.w))
    hard_thresh_spikes_sum = torch.round(spikes).sum()
    print('spikes sum: {}'.format(hard_thresh_spikes_sum))
    soft_thresh_spikes_sum = (spikes > 0.333).sum()
    zero_thresh_spikes_sum = (spikes > 0).sum()
    print('thresholded spikes sum: {}'.format(torch.round(spikes).sum()))
    print('=========avg. hard rate: {}'.format(1000*hard_thresh_spikes_sum / (spikes.shape[1] * spikes.shape[0])))
    print('=========avg. soft rate: {}'.format(1000*soft_thresh_spikes_sum / (spikes.shape[1] * spikes.shape[0])))
    print('=========avg. zero thresh rate: {}'.format(1000*zero_thresh_spikes_sum / (spikes.shape[1] * spikes.shape[0])))
    # plot_spike_train_projection(spikes, fname='test_projection_{}_poisson_input'.format(snn.__class__.__name__) + '_' + str(random_seed))

    zeros = torch.zeros_like(inputs)
    vs_zeros, spikes_zeros = model_util.feed_inputs_sequentially_return_tuple(snn, zeros)
    print('#spikes no input: {}'.format(torch.round(spikes_zeros).sum()))

    # plot_spiketrains_side_by_side(spikes, spikes_zeros, 'test_LIF', title='Test LIF spiketrains random and zero input', legend=['Poisson input', 'No input'])
    snn.w = torch.nn.Parameter(torch.zeros((snn.v.shape[0],snn.v.shape[0])), requires_grad=True)
    vs_zero_weights, spikes_zero_weights = model_util.feed_inputs_sequentially_return_tuple(snn, inputs)
    print('#spikes no weights: {}'.format(torch.round(spikes_zero_weights).sum()))

    plot_neuron(vs.detach().numpy(), title='{} neuron plot ({:.2f} spikes)'.format(snn.__class__.__name__, spikes.sum()),
                uuid='test_{}'.format(snn.__class__.__name__), fname='test_{}_fn_input'.format(snn.__class__.__name__) + '_' + str(random_seed) + '_' + input_type)
    plot_neuron(vs_zeros.detach().numpy(), title='Neuron plot ({:.2f} spikes)'.format(spikes_zeros.sum()),
                uuid='test_{}'.format(snn.__class__.__name__), fname='test_{}_no_input'.format(snn.__class__.__name__)  + '_' + str(random_seed) + '_' + input_type)
    plot_neuron(vs_zero_weights.detach().numpy(), title='LIF neuron plot ({:.2f} spikes)'.format(spikes.sum()),
                uuid='test_{}'.format(snn.__class__.__name__), fname='test_{}_fn_input_no_weights'.format(snn.__class__.__name__) + '_' + str(random_seed) + '_' + input_type)
    plot_spike_trains_side_by_side(spikes, spikes_zero_weights, 'test_{}'.format(snn.__class__.__name__),
                                   title='Test {} spiketrains random input'.format(snn.__class__.__name__),
                                   fname='spike_train_fn_input_and_no_weights_{}_seed_{}_{}'.format(snn.__class__.__name__, random_seed,input_type),
                                   legend=['Random weights', 'No weights'])

    tau_vr = torch.tensor(5.0)
    loss = spike_metrics.van_rossum_dist(spikes, spikes_zeros, tau=tau_vr)
    print('tau_vr: {}, loss: {}'.format(tau_vr, loss))
    loss_rate = spike_metrics.firing_rate_distance(spikes, spikes_zero_weights)
    print('firing rate loss: {}'.format(loss_rate))


sys.exit(0)
