import torch

from experiments import sine_modulated_white_noise_input
from spike_metrics import silent_penalty_term

N = 12
s1 = sine_modulated_white_noise_input(10., 4000, N)
s2 = sine_modulated_white_noise_input(10., 4000, N)

sut = silent_penalty_term(s1, s2)
relatively_low = 1e-03
assert sut < relatively_low, "silent penalty for two poisson processes with same rate should be very low"

s2_silent_neuron = s2.clone().detach()
s2_silent_neuron[:, -1] = torch.zeros((s2.shape[0],))

sut_penalised = silent_penalty_term(s2, s2_silent_neuron)
assert sut_penalised > sut, "one silent neuron should give higher penalty"
moderate_penalty = 1./(N+1)
assert sut_penalised > moderate_penalty, "one silent neuron compared to stochastic point process should be penalised"
