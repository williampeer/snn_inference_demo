import sys

import torch

import PDF_metrics
import experiments


def main(argv):
    print('Argument List:', str(argv))
    t = 4000
    N = 21
    neurons_coeff = torch.tensor(N * [1.])
    bin_size = 100

    # extreme_silent = torch.zeros((t, N))
    extreme_chaotic = torch.ones((t, N))
    sinusoidal_fast = experiments.sine_input(t, N, neurons_coeff=neurons_coeff, period=200.).clip(0., 1.)
    sinusoidal_slow = experiments.sine_input(t, N, neurons_coeff=neurons_coeff, period=1000.).clip(0., 1.)

    spike_probs_main = torch.randn((t, N)).clip(0., 1.)
    m_bernoulli = torch.distributions.bernoulli.Bernoulli(spike_probs_main)
    sample_spikes_main = m_bernoulli.sample()
    sample_spikes_second = m_bernoulli.sample()

    spike_probas_other = torch.randn((t, N)).clip(0., 1.)
    m_bernoulli_other = torch.distributions.bernoulli.Bernoulli(spike_probas_other)
    sample_spikes_other = m_bernoulli_other.sample()
    sample_spikes_other_second = m_bernoulli_other.sample()

    # TEST
    lowest_nll = PDF_metrics.poisson_nll(spike_probs_main, sample_spikes_main, bin_size=bin_size)
    lowest_nll_second = PDF_metrics.poisson_nll(spike_probs_main, sample_spikes_second, bin_size=bin_size)
    print('lowest_nll: {}'.format(lowest_nll))
    print('lowest_nll_second: {}'.format(lowest_nll_second))
    assert lowest_nll < 1.5 * lowest_nll_second and lowest_nll_second < 1.5 * lowest_nll, "lowest_nll should be fairly similar to lowest_2nd"

    pnll_chaotic = PDF_metrics.poisson_nll(spike_probs_main, extreme_chaotic, bin_size=bin_size)
    print('pnll chaotic: {}'.format(pnll_chaotic))
    sine_fast = PDF_metrics.poisson_nll(spike_probs_main, sinusoidal_fast.round(), bin_size=bin_size)
    sine_slow = PDF_metrics.poisson_nll(spike_probs_main, sinusoidal_slow.round(), bin_size=bin_size)
    print('pnll sine fast: {}'.format(sine_fast))
    print('pnll sine_slow: {}'.format(sine_slow))

    # assert lowest_nll < PDF_metrics.poisson_nll(spike_probs_main, extreme_silent, bin_size=bin_size)
    assert lowest_nll < pnll_chaotic, "lowest nll: {}, pnll_chaotic: {}".format(lowest_nll, pnll_chaotic)
    assert lowest_nll < sine_fast
    assert lowest_nll < sine_slow

    pnll_main_to_other = PDF_metrics.poisson_nll(spike_probs_main, sample_spikes_other, bin_size=bin_size)
    pnll_main_to_second_other = PDF_metrics.poisson_nll(spike_probs_main, sample_spikes_other_second, bin_size=bin_size)
    print('pnll pnll_main_to_other: {}'.format(pnll_main_to_other))
    print('pnll pnll_main_to_second_other: {}'.format(pnll_main_to_second_other))

    assert lowest_nll < pnll_main_to_other
    assert lowest_nll < pnll_main_to_second_other

    assert PDF_metrics.poisson_nll(spike_probas_other, sample_spikes_other, bin_size) < \
           PDF_metrics.poisson_nll(spike_probas_other, sample_spikes_main, bin_size)

    pnll_sine_fast_to_slow = PDF_metrics.poisson_nll(sinusoidal_fast, sinusoidal_slow.round(), bin_size)
    pnll_sine_fast_to_fast = PDF_metrics.poisson_nll(sinusoidal_fast, sinusoidal_fast.round(), bin_size)
    pnll_sine_slow_to_slow = PDF_metrics.poisson_nll(sinusoidal_slow, sinusoidal_slow.round(), bin_size)
    pnll_sine_slow_to_fast = PDF_metrics.poisson_nll(sinusoidal_slow, sinusoidal_fast.round(), bin_size)

    print('pnll_sine_fast_to_fast: {}'.format(pnll_sine_fast_to_fast))
    print('pnll_sine_slow_to_slow: {}'.format(pnll_sine_slow_to_slow))
    print('pnll_sine_fast_to_slow: {}'.format(pnll_sine_fast_to_slow))
    print('pnll_sine_slow_to_fast: {}'.format(pnll_sine_slow_to_fast))

    assert pnll_sine_fast_to_fast < pnll_sine_fast_to_slow
    assert pnll_sine_slow_to_slow < pnll_sine_slow_to_fast


if __name__ == "__main__":
    main(sys.argv[1:])
    sys.exit(0)
