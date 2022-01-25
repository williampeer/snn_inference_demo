import torch

BIN_SIZE = 400


# an approximation using torch.where
def torch_van_rossum_convolution(spikes, tau):
    decay_kernel = torch.exp(-torch.tensor(1.) / tau)
    convolved_spiketrain = spikes.clone()
    one_row_of_zeros = torch.ones((1, spikes.shape[1]))
    for i in range(int(3*tau)):
        tmp_shifted_conv = torch.cat([one_row_of_zeros, convolved_spiketrain[:-1]])
        convolved_spiketrain = torch.where(spikes < 0.5, tmp_shifted_conv.clone() * decay_kernel, spikes.clone())
    return convolved_spiketrain


def torch_van_rossum_convolution_two_sided(spikes, tau):
    decay_kernel = torch.exp(-torch.tensor(1.) / tau)
    convolved_spiketrain = spikes.clone()
    # convolved_spiketrain_backwards = spikes.clone()
    row_of_zeros = torch.zeros((1, spikes.shape[1]))
    for i in range(int(3*tau)):
        tmp_shifted_conv = torch.cat([row_of_zeros, convolved_spiketrain[:-1]])
        tmp_shifted_backwards = torch.cat([convolved_spiketrain[1:], row_of_zeros.clone().detach()])
        convolved_spiketrain = torch.where(spikes < 0.75, torch.max(tmp_shifted_conv * decay_kernel, tmp_shifted_backwards * decay_kernel), spikes.clone())
    return convolved_spiketrain


def van_rossum_dist(spikes, target_spikes, tau):
    c1 = torch_van_rossum_convolution(spikes=spikes, tau=tau)
    c2 = torch_van_rossum_convolution(spikes=target_spikes, tau=tau)
    return euclid_dist(c1, c2)


def van_rossum_dist_two_sided(spikes, target_spikes, tau):
    c1 = torch_van_rossum_convolution_two_sided(spikes=spikes, tau=tau)
    c2 = torch_van_rossum_convolution_two_sided(spikes=target_spikes, tau=tau)
    return euclid_dist(c1, c2)


def van_rossum_dist_one_to_K(spikes, target_spikes, tau):
    c1 = torch_van_rossum_convolution(spikes=spikes.reshape((-1, 1)), tau=tau)
    c1 = torch.ones((1, target_spikes.shape[1])) * c1.reshape((-1, 1))  # broadcast
    c2 = torch_van_rossum_convolution(spikes=target_spikes, tau=tau)
    pow_res = torch.pow(torch.sub(c1, c2), 2)
    euclid_per_node = torch.sqrt(pow_res + 1e-12).sum(dim=0)
    return euclid_per_node


def greedy_shortest_dist_vr(spikes, target_spikes, tau):
    assert spikes.shape[0] > spikes.shape[1], "each time step as a row expected, meaning column by node"
    num_nodes = spikes.shape[1]
    indices_left = torch.arange(0, num_nodes)
    min_distances = torch.zeros((spikes.shape[1],))
    for s_i in range(0, num_nodes):
        d_i_J = van_rossum_dist_one_to_K(spikes[:, s_i], target_spikes[:, indices_left], tau)
        min_i_J = d_i_J[0]; min_index = 0
        for ind in range(1, d_i_J.shape[0]):
            if d_i_J[ind] < min_i_J:
                min_i_J = d_i_J[ind]
                min_index = ind
        min_distances[s_i] = min_i_J
        indices_left = indices_left[indices_left != min_index]

    return torch.mean(min_distances)


def euclid_dist(vec1, vec2):
    pow_res = torch.pow(torch.sub(vec2, vec1), 2)
    return torch.sqrt(pow_res.sum() + 1e-12)


def mse(s1, s2):
    return torch.pow(torch.sub(s2, s1), 2).sum() / (s1.shape[0] * s1.shape[1])


def van_rossum_squared_distance(s1, s2, tau):
    c1 = torch_van_rossum_convolution(spikes=s1, tau=tau)
    c2 = torch_van_rossum_convolution(spikes=s2, tau=tau)
    return mse(c1, c2)


def silent_penalty_term(spikes, targets):
    normalised_spike_rate = spikes.sum(dim=0) * 1000. / (spikes.shape[0])  # Hz
    normalised_target_rate = targets.sum(dim=0) * 1000. / (targets.shape[0])  # Hz
    # f_penalty(x,y) = se^(-x/(N*t))
    return torch.pow(torch.exp(-normalised_spike_rate) - torch.exp(-normalised_target_rate), 2).sum()


def firing_rate_distance(model_spikes, target_spikes):
    mean_model_rate = model_spikes.sum(dim=0) * 1000. / model_spikes.shape[0]  # Hz
    mean_targets_rate = target_spikes.sum(dim=0) * 1000. / target_spikes.shape[0]  # Hz
    return euclid_dist(mean_targets_rate, mean_model_rate)


def fano_factor_dist(out, tar, bins=BIN_SIZE):
    bin_len = int(out.shape[0]/bins)
    out_counts = torch.zeros((bins,out.shape[1]))
    tar_counts = torch.zeros((bins,tar.shape[1]))
    for b_i in range(bins):
        out_counts[b_i] = (out[b_i*bin_len:(b_i+1)*bin_len].sum(dim=0))
        tar_counts[b_i] = (tar[b_i*bin_len:(b_i+1)*bin_len].sum(dim=0))

    F_out = torch.var(out_counts) / torch.mean(out_counts)
    F_tar = torch.var(tar_counts) / torch.mean(tar_counts)
    return euclid_dist(F_out, F_tar)


def calc_pearsonr(counts_out, counts_tar):
    mu_out = torch.mean(counts_out, dim=0)
    std_out = torch.std(counts_out, dim=0)  # * counts_out.shape[0]  # Bessel correction correction
    mu_tar = torch.mean(counts_tar, dim=0)
    std_tar = torch.std(counts_tar, dim=0)  # * counts_out.shape[0]  # Bessel correction correction

    pcorrcoeff = (counts_out - torch.ones_like(counts_out) * mu_out) * (counts_tar - torch.ones_like(counts_tar) * mu_tar) / (std_out * std_tar + 1e-12)

    assert torch.isnan(pcorrcoeff).sum() == 0, "found nan-values in pcorrcoeff: {}".format(pcorrcoeff)

    return pcorrcoeff


# correlation metric over binned spike counts
def correlation_metric_distance(out, tar, bin_size=BIN_SIZE):
    n_bins = int(out.shape[0] / bin_size)
    out_counts = torch.zeros((n_bins, out.shape[1]))
    tar_counts = torch.zeros((n_bins, tar.shape[1]))
    # assert n_bins >= 3, "n_bins should be at least 3. was: {}\nout.shape: {}, bin_size: {}".format(n_bins, out.shape, bin_size)
    for b_i in range(n_bins):
        out_counts[b_i] = (out[b_i * bin_size:(b_i + 1) * bin_size].sum(dim=0))
        tar_counts[b_i] = (tar[b_i * bin_size:(b_i + 1) * bin_size].sum(dim=0))

    pcorrcoeff = calc_pearsonr(tar_counts, out_counts)
    neg_dist = torch.ones_like(pcorrcoeff) - pcorrcoeff  # max 0.
    squared_dist = torch.pow(neg_dist, 2)
    dist = torch.sqrt(squared_dist + 1e-12).sum()
    return dist


def spike_proba_metric(sprobs, spikes, target_spikes):
    m = torch.distributions.bernoulli.Bernoulli(sprobs)
    # nll_target = -m.log_prob(target_spikes).sum()
    frd = firing_rate_distance(spikes, target_spikes)
    nll_output = -m.log_prob(target_spikes).sum()
    return frd*nll_output


def test_metric(sprobs, spikes, target_spikes):
    m = torch.distributions.bernoulli.Bernoulli(sprobs)
    spikes = m.sample()
    frd = firing_rate_distance(spikes, target_spikes)
    return frd


def CV_dist(out, tar, bins=BIN_SIZE):
    bin_len = int(out.shape[0]/bins)
    out_counts = torch.zeros((bins,out.shape[1]))
    tar_counts = torch.zeros((bins,tar.shape[1]))
    for b_i in range(bins):
        out_counts[b_i] = (out[b_i*bin_len:(b_i+1)*bin_len].sum(dim=0))
        tar_counts[b_i] = (tar[b_i*bin_len:(b_i+1)*bin_len].sum(dim=0))

    F_out = torch.std(out_counts) / torch.mean(out_counts)
    F_tar = torch.std(tar_counts) / torch.mean(tar_counts)
    return euclid_dist(F_out, F_tar)


def normalised_overall_activity_term(model_spikes):
    # overall-activity penalty:
    return (model_spikes.sum() + 1e-09) / model_spikes.shape[1]


def shortest_dist_rates(spikes, target_spikes):
    assert spikes.shape[0] > spikes.shape[1], "each time step as a row expected, meaning column by node"

    spike_rates = spikes.sum(dim=0) * 1000. / spikes.shape[0]
    spike_rates, _ = torch.sort(spike_rates)
    target_rates = target_spikes.sum(axis=0) * 1000. / target_spikes.shape[0]
    target_rates, _ = torch.sort(target_rates)

    return euclid_dist(spike_rates, target_rates)


def shortest_dist_rates_w_silent_penalty(spikes, target_spikes):
    assert spikes.shape[0] > spikes.shape[1], "each time step as a row expected, meaning column by node"

    spike_rates = spikes.sum(dim=0) * 1000. / spikes.shape[0]
    spike_rates, _ = torch.sort(spike_rates)
    target_rates = target_spikes.sum(dim=0) * 1000. / target_spikes.shape[0]
    target_rates, _ = torch.sort(target_rates)

    pow_res = torch.pow(torch.exp(-spike_rates) - torch.exp(target_rates), 2)

    silent_penalty = torch.sqrt(pow_res.sum() + 1e-12)
    return euclid_dist(spike_rates, target_rates) + silent_penalty

