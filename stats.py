import numpy as np
import torch


def mean_firing_rate(spikes, bin_size=1.):
    assert spikes.shape[0] > spikes.shape[1], "should be bins (1ms) by nodes (rows by cols)"
    return spikes.sum(axis=0) / (spikes.shape[0] * bin_size)


def sums_helper(spikes1, spikes2):
    assert spikes1.shape[0] > spikes1.shape[1], "expected one column per neuron. spikes1.shape: {}".format(spikes1.shape)
    # sum over bins
    sum_spikes1 = spikes1.sum(axis=1)
    sum_spikes2 = spikes2.sum(axis=1)
    return torch.reshape(torch.cat([sum_spikes1, sum_spikes2]), (2, -1))  # N by spikes


def firing_rate_per_neuron(spikes):
    assert spikes.shape[0] > spikes.shape[1], "should be bins (1ms) by nodes (rows by cols)"
    return torch.mean(spikes, dim=0)


def binned_avg_firing_rate_per_neuron(spikes, bin_size):
    spikes = np.array(spikes)
    assert spikes.shape[0] > spikes.shape[1], "should be bins (1ms) by nodes (rows by cols)"

    std_per_node = torch.zeros((spikes.shape[1],))
    mean_per_node = torch.zeros((spikes.shape[1],))
    for node_i in range(spikes.shape[1]):
        avgs = binned_firing_rates(spikes[:, node_i], bin_size)
        std_per_node[node_i] = torch.std(torch.tensor(avgs))
        mean_per_node[node_i] = torch.mean(torch.tensor(avgs))
    return std_per_node, mean_per_node


def binned_firing_rates(vec, bin_size):
    num_intervals = int(vec.shape[0] / bin_size)
    avgs = np.zeros((num_intervals,))

    for i in range(num_intervals):
        cur_interval = vec[i*bin_size:(i+1)*bin_size]
        avgs[i] = np.mean(cur_interval)

    return avgs


def rate_Hz(spike_train):
    rate_Hz = spike_train.sum(dim=0) * 1000. / spike_train.shape[0]
    return rate_Hz

# ----------------------------------------

def sub_sums(s, bin_size):
    assert s.shape[0] > s.shape[1], "assert more rows than columns. shape: {}".format(s.shape)

    sums = s[:bin_size].sum(axis=0)
    for ctr in range(1, int(s.shape[0]/bin_size)):
        sums = np.vstack((sums, s[ctr*bin_size:(ctr+1)*bin_size].sum(axis=0)))
    return sums


def spike_train_corr_new(s1, s2, bin_size=20):
    s1 = np.array(s1); s2 = np.array(s2)
    assert s1.shape[0] == s2.shape[0] and s1.shape[1] == s2.shape[
        1], "shapes should be equal. s1.shape: {}, s2.shape: {}".format(s1.shape, s2.shape)

    sums1 = sub_sums(s1, bin_size=bin_size)
    sums2 = sub_sums(s2, bin_size=bin_size)

    corrcoef = np.corrcoef(sums1, sums2, rowvar=False)
    return corrcoef


def higher_order_stats(s1, s2, bin_size=20):
    s1 = np.array(s1); s2 = np.array(s2)
    assert s1.shape[0] == s2.shape[0] and s1.shape[1] == s2.shape[1], \
        "shapes should be equal. s1.shape: {}, s2.shape: {}".format(s1.shape, s2.shape)

    sums1 = sub_sums(s1, bin_size=bin_size)
    sums2 = sub_sums(s2, bin_size=bin_size)

    mu1 = np.mean(sums1)
    std1 = np.std(sums1)
    mu2 = np.mean(sums2)
    std2 = np.std(sums2)

    CV1 = std1 / mu1
    CV2 = std2 / mu2

    corrcoef = np.corrcoef(sums1, sums2, rowvar=False)
    s1_spikes = s1.sum(axis=0)
    s2_spikes = s2.sum(axis=0)
    N = s1.shape[1]
    for m_i in range(N):
        if s1_spikes[m_i] < 0.01 or s2_spikes[m_i] < 0.01:
            corrcoef[N + m_i, :] = 0.
            corrcoef[:, m_i] = 0.

    return np.abs(corrcoef), mu1, std1, mu2, std2, CV1, CV2
