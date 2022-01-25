import torch
import numpy as np

from Models.GLIF import GLIF
from experiments import sine_modulated_white_noise_input, draw_from_uniform
from model_util import feed_inputs_sequentially_return_spike_train

torch.random.manual_seed(0)
np.random.seed(0)

num_neurons = 12
params_model = draw_from_uniform(GLIF.parameter_init_intervals, num_neurons)
m1 = GLIF(parameters=params_model, N=num_neurons)
m2 = GLIF(parameters=params_model, N=num_neurons)

t_inputs = sine_modulated_white_noise_input(rate=10., t=2000, N=num_neurons)

spikes1 = m1(t_inputs.clone().detach())
spikes2 = feed_inputs_sequentially_return_spike_train(m2, t_inputs)

assert spikes1.sum() == spikes2.sum(), "spike trains should have the same sum. sums: {}, {}".format(spikes1.sum(), spikes2.sum())
