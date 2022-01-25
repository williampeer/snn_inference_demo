import torch
from torch import tensor as T

import model_util
from TargetModels import SleepModelWrappers
from experiments import sine_modulated_white_noise_input
from plot import plot_neuron, plot_spike_train
from stats import firing_rate_per_neuron


def test_sleep_model(snn):
    rate = 0.5
    inputs = sine_modulated_white_noise_input(rate=rate, t=4000, N=snn.N)
    membrane_potentials, spikes = model_util.feed_inputs_sequentially_return_tuple(snn, inputs)
    mean_rates = firing_rate_per_neuron(spikes)
    print('mean_rates (Poisson): {}'.format(mean_rates))
    plot_neuron(membrane_potentials.data, title='Neuron plot ({:.2f} spikes) Poisson stim.'.format(spikes.sum()), fname_ext='test_LIF_complex_poisson_input')
    snn.reset_hidden_state()

    wake_mask = torch.cat([T(4*[1.]), T(4*[0.]), T(4*[0.])])
    membrane_potentials, spikes_wake = model_util.feed_inputs_sequentially_return_tuple(snn, inputs * wake_mask)
    # plot_neuron(membrane_potentials.data, title='Neuron plot ({:.2f} spikes) wake stim.'.format(spikes_wake.sum()), fname_ext='test_LIF_complex_poisson_wake_input')
    mean_rates = firing_rate_per_neuron(spikes_wake)
    print('mean_rates (wake): {}'.format(mean_rates))
    snn.reset_hidden_state()

    rem_mask = torch.cat([T(4*[0.]), T(4*[1.0]), T(4*[0.])])
    membrane_potentials, spikes_rem = model_util.feed_inputs_sequentially_return_tuple(snn, inputs * rem_mask)
    # plot_neuron(membrane_potentials.data, title='Neuron plot ({:.2f} spikes) REM stim.'.format(spikes_rem.sum()), fname_ext='test_LIF_complex_poisson_rem_input')
    mean_rates = firing_rate_per_neuron(spikes_rem)
    print('mean_rates (rem): {}'.format(mean_rates))
    snn.reset_hidden_state()

    nrem_mask = torch.cat([T(4*[0.]), T(4*[0.]), T(4*[1.0])])
    membrane_potentials, spikes_nrem = model_util.feed_inputs_sequentially_return_tuple(snn, inputs * nrem_mask)
    # plot_neuron(membrane_potentials.data, title='Neuron plot ({:.2f} spikes) NREM stim.'.format(spikes_nrem.sum()), fname_ext='test_LIF_complex_poisson_nrem_input')
    mean_rates = firing_rate_per_neuron(spikes_nrem)
    print('mean_rates (nrem): {}'.format(mean_rates))
    snn.reset_hidden_state()

    # zeros = torch.zeros((1000, snn.N))
    # membrane_potentials_zeros, spikes_zeros = model_util.feed_inputs_sequentially_return_spikes_and_potentials(snn, zeros)
    # plot_neuron(membrane_potentials_zeros.data, title='Neuron plot ({:.2f} spikes) no stimulus'.format(spikes_zeros.sum()), fname_ext='test_LIF_complex_no_input')
    # mean_rates_zeros = firing_rate_per_neuron(spikes_zeros)
    # print('mean_rates_zeros: {}'.format(mean_rates_zeros))

    plot_spike_train(spikes, title='Test {} spiketrain Poisson input'.format(snn.__class__.__name__), uuid='test_SleepModels',
                     fname='spiketrain_{}_Poisson_input_{}'.format(snn.__class__.__name__, str(rate).replace('.', '_')))
    # plot_spiketrains_side_by_side(spikes_rem, spikes_nrem, 'test_LIF_complex', title='Test {} spiketrains REM & NREM input'.format(snn.__class__.__name__), legend=['REM', 'NREM'])
    # plot_all_spiketrains([spikes, spikes_wake, spikes_rem, spikes_nrem], 'test_{}_complex'.format(snn.__class__.__name__), title='Test {} spiketrains'.format(snn.__class__.__name__), legend=['Poisson', 'Wake', 'REM', 'NREM'])


# snn = SleepModels.LIF()
# snn = SleepModelWrappers.lif_sleep_model()
# test_sleep_model(snn)

snn = SleepModelWrappers.glif_sleep_model()
test_sleep_model(snn)

# snn = SleepModels.IzhikevichStable()
# snn = SleepModelWrappers.izhikevich_sleep_model()
# test_sleep_model(snn)
