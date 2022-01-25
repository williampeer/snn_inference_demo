import torch
import plot
import spike_metrics
from TargetModels.TargetModels import lif_r_asc_continuous_ensembles_model_dales_compliant
from experiments import sine_modulated_white_noise_input
from model_util import feed_inputs_sequentially_return_spike_train


def test_plot_van_rossum_convolution():
    # sample_spiketrain = 1.0 * (torch.rand((200, 3)) < 0.25)
    sample_inputs = sine_modulated_white_noise_input(rate=10., t=800, N=3)
    # sample_spiketrain_2 = 1.0 * (torch.rand((200, 3)) < 0.25)
    sample_inputs_2 = sine_modulated_white_noise_input(rate=10., t=800, N=3)

    plot.plot_neuron(sample_inputs, uuid='van_rossum', title='Sample input perturbation 1', fname='plot_input_1')
    plot.plot_neuron(sample_inputs_2, uuid='van_rossum', title='Sample input perturbation 2', fname='plot_input_2')

    model = lif_r_asc_continuous_ensembles_model_dales_compliant(42, N=3)
    spikes1 = feed_inputs_sequentially_return_spike_train(model, sample_inputs)
    spikes2 = feed_inputs_sequentially_return_spike_train(model, sample_inputs_2)

    plot.plot_spike_train(spikes1, uuid='van_rossum', title="Spike train 1", fname='spike_train_1')
    plot.plot_spike_train(spikes2, uuid='van_rossum', title="Spike train 2", fname='spike_train_2')
    plot.plot_spike_trains_side_by_side(spikes1, spikes2, 'van_rossum',
                                        title="Sample side-by-side spike trains")

    convolved1 = spike_metrics.torch_van_rossum_convolution(spikes1, tau=torch.tensor(25.0))
    # convolved = spike_metrics.convolve_van_rossum_using_clone(sample_spiketrain, tau=torch.tensor(5.0))
    plot.plot_neuron(convolved1[:, 0], uuid='van_rossum', title='Spiketrain 1 node 1 van Rossum convolved', fname='plot_1_neuron_conv_1')
    plot.plot_neuron(convolved1[:, 1], uuid='van_rossum', title='Spiketrain 1 node 2 van Rossum convolved', fname='plot_1_neuron_conv_2')
    plot.plot_neuron(convolved1[:, 2], uuid='van_rossum', title='Spiketrain 1 node 3 van Rossum convolved', fname='plot_1_neuron_conv_3')

    convolved2 = spike_metrics.torch_van_rossum_convolution(spikes2, tau=torch.tensor(25.0))
    plot.plot_neuron(convolved2[:, 0], uuid='van_rossum', title='Spiketrain 2 node 1 van Rossum convolved', fname='plot_2_neuron_conv_1')
    plot.plot_neuron(convolved2[:, 1], uuid='van_rossum', title='Spiketrain 2 node 2 van Rossum convolved', fname='plot_2_neuron_conv_2')
    plot.plot_neuron(convolved2[:, 2], uuid='van_rossum', title='Spiketrain 2 node 3 van Rossum convolved', fname='plot_2_neuron_conv_3')


def test_plot_parameter_pairs_with_variance():
    sample_param_1_means = []
    sample_param_2_means = []
    sample_targets = {0: 6.5, 1: -65.0}
    # sample_param_3_means = []

    for i in range(10):
        sample_param_1_means.append(6.0 + torch.rand(1))
        sample_param_2_means.append(-66.0 + 2.0 * torch.rand(1))
        # sample_param_3_means.append(torch.tensor(9000. + i))

    fitted_param_means = {0: sample_param_1_means, 1: sample_param_2_means}
    param_names = ['p1', 'p2']

    # plot.plot_parameter_pair_with_variance(fitted_param_means[0], fitted_param_means[1], sample_targets)
    plot.plot_all_param_pairs_with_variance(param_means=fitted_param_means, target_params=sample_targets, exp_type='test',
                                            param_names=param_names,
                                            uuid='test_plots', fname='test_parameter_kdes_1', custom_title='Test plot', logger=False)

    fitted_param_means[2] = torch.rand((3, 3)).data
    # sample_targets.append(9000.)
    sample_targets[2] = torch.rand((3, 3)).data

    plot.plot_all_param_pairs_with_variance(param_means=fitted_param_means, target_params=sample_targets,
                                            param_names=param_names,
                                            exp_type='test', uuid='test_plots', fname='test_parameter_kdes_2',
                                            custom_title='Test plot', logger=False)


def test_plot_spiketrains_side_by_side():
    sample_spiketrain = 1.0 * (torch.rand((200, 3)) < 0.25)
    sample_spiketrain_2 = 1.0 * (torch.rand((200, 3)) < 0.25)

    plot.plot_spike_trains_side_by_side(sample_spiketrain, sample_spiketrain_2, 'test_uuid', title='Test plot_spiketrains_side_by_side')


test_plot_van_rossum_convolution()
test_plot_parameter_pairs_with_variance()
test_plot_spiketrains_side_by_side()
