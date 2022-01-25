from experiments import sine_modulated_white_noise_input
from plot import plot_neuron
from spike_metrics import *

import torch


def test_greedy_shortest_dist_vr():
    tar_spikes = sine_modulated_white_noise_input(20., t=1000, N=3)
    model_spikes = sine_modulated_white_noise_input(20. * torch.ones((3,)), t=1000, N=3)
    print('num of sample model spikes: {}'.format(model_spikes.sum()))
    print('num of sample target spikes: {}'.format(tar_spikes.sum()))

    dist_poisson_spikes_tau_4 = greedy_shortest_dist_vr(spikes=model_spikes, target_spikes=tar_spikes, tau=torch.tensor(4.0))
    dist_poisson_spikes_tau_20 = greedy_shortest_dist_vr(spikes=model_spikes, target_spikes=tar_spikes, tau=torch.tensor(20.0))
    print('(tau=4) van rossum dist.: {}'.format(dist_poisson_spikes_tau_4))
    print('(tau=20) van rossum dist.: {}'.format(dist_poisson_spikes_tau_20))

    zeros = torch.zeros_like(tar_spikes)
    print('num of spikes in zeros: {}'.format(zeros.sum()))

    dist_zeros = greedy_shortest_dist_vr(zeros.clone(), zeros, tau=torch.tensor(4.0))
    print('van rossum zero dist.: {}'.format(dist_zeros))
    assert 0 <= dist_zeros < 1e-06, "distance between silent trains should be approximately zero. was: {}".format(dist_zeros)

    distance_model_spikes_zeros_tau_vr_4 = greedy_shortest_dist_vr(model_spikes, zeros, torch.tensor(4.0))
    distance_model_spikes_zeros_tau_vr_20 = greedy_shortest_dist_vr(model_spikes, zeros, torch.tensor(20.0))
    print('distance_model_spikes_zeros_tau_vr_4: {}'.format(distance_model_spikes_zeros_tau_vr_4))
    print('distance_model_spikes_zeros_tau_vr_20: {}'.format(distance_model_spikes_zeros_tau_vr_20))
    assert distance_model_spikes_zeros_tau_vr_20 > distance_model_spikes_zeros_tau_vr_4, "tau 20 should result in greater loss than 4 when compared to no spikes as target"


def test_van_rossum_dist():
    tar_spikes = sine_modulated_white_noise_input(10., t=1000, N=3)
    model_spikes = sine_modulated_white_noise_input(10., t=1000, N=3)
    print('num of sample model spikes: {}'.format(model_spikes.sum()))
    print('num of sample target spikes: {}'.format(tar_spikes.sum()))

    dist_poisson_spikes_tau_4 = van_rossum_dist(spikes=model_spikes, target_spikes=tar_spikes, tau=torch.tensor(4.0))
    dist_poisson_spikes_tau_20 = van_rossum_dist(spikes=model_spikes, target_spikes=tar_spikes, tau=torch.tensor(20.0))
    print('(tau=4) van rossum dist.: {}'.format(dist_poisson_spikes_tau_4))
    print('(tau=20) van rossum dist.: {}'.format(dist_poisson_spikes_tau_20))

    zeros = torch.zeros_like(tar_spikes)
    print('num of spikes in zeros: {}'.format(zeros.sum()))

    dist_zeros = van_rossum_dist(zeros.clone(), zeros, tau=torch.tensor(4.0))
    print('van rossum zero dist.: {}'.format(dist_zeros))
    assert 0 <= dist_zeros < 1e-08, "distance between silent trains should be approximately zero. was: {}".format(dist_zeros)

    distance_model_spikes_zeros_tau_vr_4 = van_rossum_dist(model_spikes, zeros, torch.tensor(4.0))
    distance_model_spikes_zeros_tau_vr_20 = van_rossum_dist(model_spikes, zeros, torch.tensor(20.0))
    print('distance_model_spikes_zeros_tau_vr_4: {}'.format(distance_model_spikes_zeros_tau_vr_4))
    print('distance_model_spikes_zeros_tau_vr_20: {}'.format(distance_model_spikes_zeros_tau_vr_20))
    assert distance_model_spikes_zeros_tau_vr_20 > distance_model_spikes_zeros_tau_vr_4, "tau 20 should result in greater loss than 4 when compared to no spikes as target"


def test_optimised_van_rossum():
    tau = torch.tensor(20.0)
    # spikes = (torch.rand((100, 3)) > 0.85).float()
    spikes = sine_modulated_white_noise_input(20., 200., 3)

    torch_conv = torch_van_rossum_convolution(spikes, tau)
    plot_neuron(torch_conv, "van Rossum convolved spike train", fname_ext="_vr_test")

    print('no. spikes: {}, torch conv. sum: {}'.format(spikes.sum(), torch_conv.sum()))
    assert torch_conv.sum() - spikes.sum() > 0., "check torch conv. impl."


def test_optimised_van_rossum_two_sided():
    tau = torch.tensor(10.0)
    # spikes = (torch.rand((100, 3)) > 0.85).float()
    spikes = sine_modulated_white_noise_input(20., 200., 3)

    torch_conv = torch_van_rossum_convolution(spikes, tau)
    plot_neuron(torch_conv, "van Rossum convolved spike train", fname_ext="_vr_test_one_sided")

    print('no. spikes: {}, torch conv. sum: {}'.format(spikes.sum(), torch_conv.sum()))
    assert torch_conv.sum() - spikes.sum() > 0., "check torch conv. impl."

    torch_conv = torch_van_rossum_convolution_two_sided(spikes, tau)
    plot_neuron(torch_conv, "van Rossum convolved spike train two sided", fname_ext="_vr_test_two_sided")

    print('no. spikes: {}, torch conv. sum: {}'.format(spikes.sum(), torch_conv.sum()))
    assert torch_conv.sum() - spikes.sum() > 0., "check torch conv. impl."


def test_different_taus_van_rossum_dist():
    t = 400; N=12
    zeros = torch.zeros((t, N))
    sample_spikes = sine_modulated_white_noise_input(10., t, N)
    cur_tau = torch.tensor(25.0)
    cur_dist = van_rossum_dist(sample_spikes, zeros, cur_tau)
    print('cur_tau: {:.2f}, cur_dist: {:.4f}'.format(cur_tau, cur_dist))
    for tau_i in range(24):
        prev_dist = cur_dist.clone()
        cur_tau = cur_tau - 1
        cur_dist = van_rossum_dist(sample_spikes, zeros, cur_tau)
        # print('cur_tau: {:.2f}, cur_dist: {:.4f}'.format(cur_tau, cur_dist))
        assert cur_dist < prev_dist, "decreasing tau should decrease loss when comparing poisson spikes to zero spikes"


def test_van_rossum_convolution():
    t = 400; N = 12
    tau_vr = torch.tensor(20.0)

    zeros = torch.zeros((t, N))
    sample_spikes = (sine_modulated_white_noise_input(10., t, N) > 0).float()
    print('no. of spikes: {}'.format(sample_spikes.sum()))

    lower_avg_rate = 6. * (t/1000.) * N
    assert sample_spikes.sum() > lower_avg_rate, "spike sum: {} should be greater than an approximate lower avg. rate: {}"\
        .format(sample_spikes.sum(), lower_avg_rate)

    dist = van_rossum_dist(sample_spikes, zeros, tau=tau_vr)
    conv = torch_van_rossum_convolution(sample_spikes, tau=tau_vr)
    dist_approx = euclid_dist(conv, torch.zeros_like(conv))
    assert dist == dist_approx, "distance: {:.4f} was not approx. dist: {:.4f}".format(dist, dist_approx)
    assert conv.sum() > sample_spikes.sum(), "conv sum should be greater than sum of spikes"


# --------------------------------------
test_greedy_shortest_dist_vr()
test_van_rossum_dist()
test_optimised_van_rossum()
test_different_taus_van_rossum_dist()
test_van_rossum_convolution()
test_optimised_van_rossum_two_sided()
