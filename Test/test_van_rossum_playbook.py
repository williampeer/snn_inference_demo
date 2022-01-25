import torch

import experiments
from spike_metrics import van_rossum_dist


# s1 = sine_modulated_white_noise_input(10., 4000, 12)
# s2 = sine_modulated_white_noise_input(10., 4000, 12)
s1 = torch.round(torch.rand((1200, 4)))
s2 = torch.round(torch.rand((1200, 4)))

t2 = float(s2.sum())

s2_silent_neuron = s2.clone().detach()
s2_silent_neuron[:, 0] = torch.zeros((s2.shape[0],))
# s2_silent_neuron[:, 1] = torch.zeros((s2.shape[0],))

tau_start = 25.0

for i in range(5):
    vrd = van_rossum_dist(s1, s2, tau=torch.tensor(tau_start + 0.1 * tau_start * i))
    print(i, vrd)
    vrd_one_silent = van_rossum_dist(s1, s2_silent_neuron, tau=torch.tensor(tau_start + 3.0 * i))
    print(i, vrd_one_silent)
    assert vrd_one_silent > vrd, "distance should be greater between a silent spike train and spike train than between two spike trains of similar rates."


assert t2 == float(s2.sum()), "sum should be unmodified. in-place modification in vrd?"
