from enum import Enum

import torch


def get_binned_spike_counts(out, bin_size=400):
    n_bins = int(out.shape[0] / bin_size)
    out_counts = torch.zeros((n_bins, out.shape[1]))
    for b_i in range(n_bins):
        out_counts[b_i] = (out[b_i * bin_size:(b_i + 1) * bin_size].sum(dim=0))
    return out_counts


def poisson_nll(spike_probabilities, target_spikes, bin_size):
    lambda_spikes = get_binned_spike_counts(spike_probabilities, bin_size).clamp(1., bin_size)
    target_spike_counts = get_binned_spike_counts(target_spikes, bin_size=bin_size).clamp(1., bin_size)
    m_Poiss = torch.distributions.poisson.Poisson(lambda_spikes)
    return torch.sum(-m_Poiss.log_prob(target_spike_counts))
    # return torch.mean(-m_Poiss.log_prob(target_spike_counts))


def bernoulli_nll(spike_probabilities, target_spikes):
    m = torch.distributions.bernoulli.Bernoulli(spike_probabilities.clamp(0., 1.))
    return torch.sum(-m.log_prob(target_spikes))
    # return torch.mean(-m.log_prob(target_spikes))


def calculate_loss(spike_probabilities, target_spikes, loss_fn, bin_size=100):
    lfn = PDF_LFN[loss_fn]
    if lfn == PDF_LFN.BERNOULLI:
        return bernoulli_nll(spike_probabilities, target_spikes)
    elif lfn == PDF_LFN.POISSON:
        return poisson_nll(spike_probabilities, target_spikes, bin_size)
    else:
        raise NotImplementedError()


class PDF_LFN(Enum):
    BERNOULLI = 'BERNOULLI'
    POISSON = 'POISSON'
