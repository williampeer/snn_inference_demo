from experiments import sine_modulated_white_noise_input

import torch

from plot import plot_spike_trains_side_by_side, plot_losses
from spike_metrics import firing_rate_distance
from stats import firing_rate_per_neuron


def test_sum_per_node():
    t = 400; N = 12; tau_vr = torch.tensor(20.0)

    # zeros = torch.zeros((t, N))
    s1 = (sine_modulated_white_noise_input(0.8, t, N) > 0).float()
    s2 = (sine_modulated_white_noise_input(0.8, t, N) > 0).float()
    # mse_nodes = mse_per_node(s1, s2)
    # assert len(mse_nodes) == s1.shape[1], "mse should be per column (node). mse shape: {}, s1.shape: {}"\
    #     .format(mse_nodes.shape, s1.shape)

    # vr_dist_nodes = van_rossum_dist_per_node(s1, s2, tau_vr)
    # assert len(vr_dist_nodes) == s1.shape[1], "vr_dist_nodes should be per column (node). vr_dist_nodes shape: {}, s1.shape: {}"\
    #     .format(vr_dist_nodes.shape, s1.shape)

    # vr_squared_nodes = van_rossum_squared_per_node(s1, s2, tau_vr)
    # assert len(vr_squared_nodes) == s1.shape[1], "vr_squared_nodes should be per column (node). vr_squared_nodes shape: {}, s1.shape: {}" \
    #     .format(vr_squared_nodes.shape, s1.shape)


def spiketrain_is_approx_rate(spiketrain, rate=10.):
    print("\n\n======= def spiketrain_is_approx_rate(): =======\n")
    # rates = firing_rate_per_neuron(spiketrain) * t / 1000.
    rates = spiketrain.sum(dim=0) * 1000. / spiketrain.shape[0]  # Hz

    assert torch.all((rate - 0.1 * rate < rates).bool()), \
        "rate: {} should be approx. rates: {}".format(rate, rates)
    assert torch.all((rates < rate + rate * 0.1).bool()), \
        "rate: {} should be approx. rates: {}".format(rate, rates)

    print('spiketrain sum: {}'.format(spiketrain.sum()))
    print('rates: {}'.format(rates))


def test_firing_rate_per_neuron():
    print("\n\n======= def test_firing_rate_per_neuron(): =======\n")
    N = 12; t=1000000; rate = 20.
    # spikes = poisson_input(rate=rate, t=t, N=N)
    # spikes2 = (poisson_input(rate=rate, t=t, N=N) > 0.5).float()
    # spikes_higher_rate = (poisson_input(rate=rate*1.1, t=t, N=N) > 0.5).float()
    # spikes_lower_rate = (poisson_input(rate=rate*0.9, t=t, N=N) > 0.5).float()
    spikes = torch.rand((t, N)) < (rate/1000.)
    spikes2 = torch.rand((t, N)) < (rate/1000.)
    higher_rate = 1.15*rate
    lower_rate = 0.85*rate
    spikes_higher_rate = torch.rand((t, N)) < (higher_rate/1000.)
    spikes_lower_rate = torch.rand((t, N)) < (lower_rate/1000.)

    spiketrain_is_approx_rate(spikes, rate)
    spiketrain_is_approx_rate(spikes2, rate)
    spiketrain_is_approx_rate(spikes_higher_rate, higher_rate)
    spiketrain_is_approx_rate(spikes_lower_rate, lower_rate)
    assert spikes.shape[1] == N, "spikes should be per node"
    assert spikes2.shape[1] == N, "spikes should be per node"
    assert spikes_higher_rate.shape[1] == N, "spikes should be per node"
    assert spikes_lower_rate.shape[1] == N, "spikes should be per node"
    # assert torch.all((broadcast_rates - 0.05 * broadcast_rates < rates < broadcast_rates + broadcast_rates * 0.05).bool()), \
        # "broadcast_rates: {} should be approx. rates: {}".format(broadcast_rates, rates)

    # relatively low frd:
    sut_loss_same_rate = firing_rate_distance(model_spikes=spikes, target_spikes=spikes2)
    sut_loss_to_higher_rate = firing_rate_distance(model_spikes=spikes, target_spikes=spikes_higher_rate)
    sut_loss_to_lower_rate = firing_rate_distance(model_spikes=spikes, target_spikes=spikes_lower_rate)

    assert sut_loss_same_rate < sut_loss_to_higher_rate, \
        "loss same rate: {} should be lower than loss to train with a higher rate: {}".format(sut_loss_same_rate, sut_loss_to_higher_rate)
    assert sut_loss_same_rate < sut_loss_to_lower_rate, \
        "loss same rate: {} should be lower than loss to train with a lower rate: {}".format(sut_loss_same_rate, sut_loss_to_lower_rate)

    print('loss same rate: {}'.format(sut_loss_same_rate))
    print('loss higher rate: {}'.format(sut_loss_to_higher_rate))
    print('loss lower rate: {}'.format(sut_loss_to_lower_rate))

    assert sut_loss_to_higher_rate * 0.9 < sut_loss_to_lower_rate and sut_loss_to_higher_rate * 1.1 > sut_loss_to_lower_rate, \
        "rate deviations above and below result in approx the same losses: {}, {}".format(sut_loss_to_higher_rate, sut_loss_to_lower_rate)

    silent_train = torch.zeros((t, N))
    almost_silent_train = torch.zeros((t, N))
    almost_silent_train[100, 0] = 1.0
    almost_silent_train[1000, 1] = 1.0
    almost_silent_train[6000, 2] = 1.0
    sut_loss_silent_to_normal = firing_rate_distance(spikes, silent_train)
    sut_loss_almost_silent_to_normal = firing_rate_distance(spikes, silent_train)
    print('sut_loss_silent_to_normal: {}'.format(sut_loss_silent_to_normal))
    print('sut_loss_almost_silent_to_normal: {}'.format(sut_loss_almost_silent_to_normal))

    assert sut_loss_silent_to_normal > sut_loss_same_rate, "loss normal to silent should be higher than loss same rates"
    assert sut_loss_silent_to_normal > sut_loss_to_higher_rate, "loss normal to silent should be higher than loss higher rates"
    assert sut_loss_silent_to_normal > sut_loss_to_lower_rate, "loss normal to silent should be higher than loss lower rates"

    assert sut_loss_almost_silent_to_normal > sut_loss_same_rate, "loss normal to almost silent should be higher than loss same rates"
    assert sut_loss_almost_silent_to_normal > sut_loss_to_higher_rate, "loss normal to almost silent should be higher than loss higher rates"
    assert sut_loss_almost_silent_to_normal > sut_loss_to_lower_rate, "loss normal to almost silent should be higher than loss lower rates"


def test_gd_emulation_sanity():
    print("\n\n======= def test_gd_emulation_sanity(): =======\n")
    N = 3; t = 12000; tar_rate = 10.
    t_spikes = (torch.rand((t, N)) < (tar_rate / 1000.)).float()

    prev_loss = 10e10
    print('Emulating GD towards correct rate, interval length t={}..'.format(t))
    losses = []
    for i in range(11):
        m_rate = 20. - i
        m_spikes = (torch.rand((t, N)) < (m_rate / 1000.)).float()

        cur_loss = firing_rate_distance(m_spikes, t_spikes)
        print('loss: {}, train iter #{}, m_rate: {}'.format(cur_loss, i, m_rate))
        # evaluate_loss(model, None, m_rate, t_spikes.clone().detach(), label='', exp_type='test', train_i=None, exp_num='test')
        plot_spike_trains_side_by_side(m_spikes, t_spikes, uuid='test', exp_type='test',
                                       title='Spike trains test set ({}, loss: {:.3f})'.format('test', cur_loss),
                                       fname='spiketrains_test_set_{}_exp_{}_train_iter_{}'.format('frd_test', 0, i))

        assert prev_loss > cur_loss, "prev loss: {} should be gt cur_loss: {} w rate: {} (closer to t_rate: {})."\
            .format(prev_loss, cur_loss, m_rate, tar_rate)
        prev_loss = cur_loss.clone().detach()
        losses.append(prev_loss.clone().detach())
    plot_losses(training_loss=losses, test_loss=[], uuid='test', exp_type='test',
                custom_title='Loss ({}, {}, lr={}, spf={})'.format('GD emul', 'GD emul', 'GD emul', 'None'),
                fname='training_and_test_loss_exp_{}_loss_fn_{}_tau_vr_{}'.format(0, 'emul', 'None'))


def test_gd_emulation_towards_silent_sanity():
    print("\n\n======= def test_gd_emulation_towards_silent_sanity(): =======\n")
    N = 3; t = 12000; tar_rate = 10.
    t_spikes = (torch.rand((t, N)) < (tar_rate / 1000.)).float()

    prev_loss = 1e-03
    print('Emulating GD towards zero rate, interval length t={}..'.format(t))
    losses = []
    for i in range(6):
        m_rate = 10. - 2.*i
        m_spikes = (torch.rand((t, N)) < (m_rate / 1000.)).float()

        cur_loss = firing_rate_distance(m_spikes, t_spikes)
        print('loss: {}, train iter #{}, m_rate: {}'.format(cur_loss, i, m_rate))
        # evaluate_loss(model, None, m_rate, t_spikes.clone().detach(), label='', exp_type='test', train_i=None, exp_num='test')
        plot_spike_trains_side_by_side(m_spikes, t_spikes, uuid='test', exp_type='test_towards_zero',
                                       title='Spike trains test set ({}, loss: {:.3f})'.format('test', cur_loss),
                                       fname='spiketrains_test_set_{}_exp_{}_train_iter_{}'.format('frd_test', 0, i))

        assert prev_loss < cur_loss, "prev loss: {} should be lt cur_loss: {} w lower rate: {} (further away from t_rate: {})."\
            .format(prev_loss, cur_loss, m_rate, tar_rate)
        prev_loss = cur_loss.clone().detach()
        losses.append(prev_loss.clone().detach())
    plot_losses(training_loss=losses, test_loss=[], uuid='test', exp_type='test_towards_zero',
                custom_title='Loss ({}, {}, lr={}, spf={})'.format('GD emul', 'GD emul', 'GD emul', 'None'),
                fname='training_and_test_loss_exp_{}_loss_fn_{}_tau_vr_{}'.format(0, 'emul', 'None'))


# test_sum_per_node()
test_firing_rate_per_neuron()
test_gd_emulation_sanity()
test_gd_emulation_towards_silent_sanity()
