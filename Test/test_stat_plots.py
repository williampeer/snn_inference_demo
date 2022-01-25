import torch

import data_util
from experiments import generate_synthetic_data
from plot import bar_plot_neuron_rates, heatmap_spike_train_correlations
from stats import binned_avg_firing_rate_per_neuron, spike_train_corr_new


def test_stat_plots():
    t = 12000; N = 12; bin_size=400
    node_indices, spike_times, spike_indices, states = data_util.load_data(4)
    assert len(node_indices) == N, "should have the same number of nodes. len(node_indices): {}, N: {}".format(len(node_indices), N)

    _, s1 = data_util.get_spike_train_matrix(index_last_step=0, advance_by_t_steps=t, spike_times=spike_times,
                                             spike_indices=spike_indices, node_numbers=node_indices)

    model = torch.load('./IzhikevichStable_sample.pt')['model']
    model.reset_hidden_state()
    s2 = generate_synthetic_data(model, 0.6, t=t).detach().numpy()
    model.reset_hidden_state()

    std1, r1 = binned_avg_firing_rate_per_neuron(s1, bin_size)
    std2, r2 = binned_avg_firing_rate_per_neuron(s2, bin_size)
    assert r1.shape[0] == N and r2.shape[0] == N, "should be rate per neuron. r1.shape: {}, N: {}".format(r1.shape, N)

    bar_plot_neuron_rates(r1, r2, std1, std2, bin_size=bin_size,
                          exp_type='default', uuid='test_bar_plot_rates', fname='test_bar_plot_neuron_rates')

    corrs_vars = spike_train_corr_new(s1, s2, bin_size=bin_size)
    heatmap_spike_train_correlations(corrs_vars, axes=['Data set nodes', 'Model neurons'], exp_type='default', uuid='test_heatmap_corrs',
                                     fname='test_heatmap_plot', bin_size=100)


test_stat_plots()
