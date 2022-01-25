import stats
from experiments import sine_modulated_white_noise_input
from plot import heatmap_spike_train_correlations


def test_spike_train_correlation():
    t = 12000; N=12; bin_size = 400
    s1 = (sine_modulated_white_noise_input(0.6, t=t, N=N) > 0).float()
    s2 = (sine_modulated_white_noise_input(0.4, t=t, N=N) > 0).float()

    corrs_vars = stats.spike_train_correlation(s1, s2, bin_size=bin_size)
    assert corrs_vars.shape[0] == N and corrs_vars.shape[1] == N, \
        "spiketrain correlations should be NxN. correlations shape: {}, N: {}".format(corrs_vars.shape, N)

    for corr_i in range(corrs_vars.shape[0]):
        for corr_j in range(corrs_vars.shape[1]):
            assert corrs_vars[corr_i][corr_j] < 0.5, \
                "should have no strongly correlated bins for poisson input. i: {}, j: {}, corr: {}"\
                    .format(corr_i, corr_j, corrs_vars[corr_i][corr_j])


def test_plot_spike_train_correlations():
    t = 12000; N = 12; bin_size = 400
    s1 = (sine_modulated_white_noise_input(0.6, t=t, N=N) > 0).float()
    s2 = (sine_modulated_white_noise_input(0.4, t=t, N=N) > 0).float()

    corrs_vars = stats.spike_train_correlation(s1, s2, bin_size=bin_size)

    heatmap_spike_train_correlations(corrs_vars, exp_type='default', uuid='test_heatmap_corrs', fname='test_heatmap_plot')


def test_plot_spike_train_correlation_self():
    t = 12000; N = 12; bin_size = 400
    spiketrain = (sine_modulated_white_noise_input(0.5, t=t, N=N) > 0).float()

    corrs_vars = stats.spike_train_correlation(spiketrain, spiketrain, bin_size=bin_size)

    heatmap_spike_train_correlations(corrs_vars, exp_type='default', uuid='test_heatmap_corrs', fname='test_heatmap_self_plot')


test_spike_train_correlation()
test_plot_spike_train_correlations()
test_plot_spike_train_correlation_self()
