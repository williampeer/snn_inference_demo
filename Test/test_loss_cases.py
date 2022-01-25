from experiments import sine_modulated_white_noise_input
from spike_metrics import van_rossum_dist

import torch
from torch import tensor

tau_vr = tensor(20.0)
sample_spikes = tensor(1.0) * (sine_modulated_white_noise_input(0.15, t=150, N=10) > 0)
print('sample spikes sum: {}'.format(sample_spikes.sum()))
assert sample_spikes.sum() > 0.15 * 150, 'spiking should occur in more than 10 % of the bins across all neurons. sum: {}'.format(sample_spikes.sum())

loss_vs_silent_model = van_rossum_dist(sample_spikes, torch.zeros_like(sample_spikes), tau=tau_vr)
print('loss_vs_silent_model: {}'.format(loss_vs_silent_model))
assert loss_vs_silent_model > 0, "loss_vs_silent_model should be above 0. loss: {}".format(loss_vs_silent_model)

sample_spikes_shifted_10 = sample_spikes[:-10, :].clone().detach()
sample_spikes_shifted_10 = torch.cat([torch.zeros((10, 10)), sample_spikes_shifted_10])
assert sample_spikes_shifted_10.shape[0] == sample_spikes.shape[0] and sample_spikes_shifted_10.shape[1] == sample_spikes.shape[1], \
    "sample spikes shifted and sample spikes shapes should be equal. shape of shifted: {}, shape of sample spikes: {}"\
        .format(sample_spikes_shifted_10.shape, sample_spikes.shape)

loss_sample_vs_shifted_10 = van_rossum_dist(sample_spikes, sample_spikes_shifted_10, tau=tau_vr)
loss_shifted_10_vs_sample = van_rossum_dist(sample_spikes_shifted_10, sample_spikes, tau=tau_vr)
print('loss_sample_vs_shifted_10: {}'.format(loss_sample_vs_shifted_10))
assert loss_sample_vs_shifted_10 > 0
assert loss_shifted_10_vs_sample > 0
assert loss_sample_vs_shifted_10 == loss_shifted_10_vs_sample, "losses should be equal. sample vs shifted: {}, shifted vs sample: {}"\
    .format(loss_sample_vs_shifted_10, loss_shifted_10_vs_sample)
assert loss_vs_silent_model > loss_sample_vs_shifted_10, "silent model should be worse than shifted spikes. loss silent: {}, loss shifted: {}"\
    .format(loss_vs_silent_model, loss_sample_vs_shifted_10)

loss_prev_more_shifted = loss_sample_vs_shifted_10
for i in range(1, 10):
    sample_spikes_shifted = sample_spikes[:-(10-i), :].clone().detach()
    sample_spikes_shifted = torch.cat([torch.zeros((10-i, 10)), sample_spikes_shifted])
    assert sample_spikes_shifted.shape[0] == sample_spikes.shape[0] and sample_spikes_shifted.shape[1] == sample_spikes.shape[1], \
        "sample spikes shifted and sample spikes shapes should be equal. shape of shifted: {}, shape of sample spikes: {}" \
            .format(sample_spikes_shifted.shape, sample_spikes.shape)

    loss_sample_vs_shifted = van_rossum_dist(sample_spikes, sample_spikes_shifted, tau=tau_vr)
    loss_shifted_vs_sample = van_rossum_dist(sample_spikes_shifted, sample_spikes, tau=tau_vr)
    print('loss_sample_vs_shifted: {}'.format(loss_sample_vs_shifted))
    assert loss_sample_vs_shifted > 0
    assert loss_shifted_vs_sample > 0
    assert loss_sample_vs_shifted == loss_shifted_vs_sample, "losses should be equal. sample vs shifted: {}, shifted vs sample: {}" \
        .format(loss_sample_vs_shifted, loss_shifted_vs_sample)
    assert loss_shifted_vs_sample < loss_prev_more_shifted, "more shifted should mean more loss. current: {}, prev: {}"\
        .format(loss_shifted_vs_sample, loss_prev_more_shifted)
    assert loss_vs_silent_model > loss_shifted_vs_sample, "silent model should be worse than shifted spikes. loss silent: {}, loss shifted: {}" \
        .format(loss_vs_silent_model, loss_shifted_vs_sample)
    loss_prev_more_shifted = loss_shifted_vs_sample.clone().detach()
