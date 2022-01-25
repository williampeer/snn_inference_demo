from Models.GLIF import GLIF
from experiments import draw_from_uniform, sine_modulated_white_noise_input
from model_util import feed_inputs_sequentially_return_spike_train
from plot import plot_spike_train

import numpy as np
import torch

np.random.seed(0)
torch.manual_seed(0)


num_neurons = 12
params_model = draw_from_uniform(GLIF.parameter_init_intervals, num_neurons)

model = GLIF(parameters=params_model, N=num_neurons)
t_inputs = sine_modulated_white_noise_input(rate=20., t=10000, N=num_neurons)

sut_outs = feed_inputs_sequentially_return_spike_train(model, t_inputs)

print('sum of sut_outs: {}'.format(sut_outs.sum()))

plot_spike_train(sut_outs, 'Test init params out spike train', uuid='test')
