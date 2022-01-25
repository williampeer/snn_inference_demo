import sys

import torch

import experiments
import model_util
from Models.Izhikevich import Izhikevich
from Test.parameter_landscape_test import plot_param_landscape

A_coeffs = [torch.randn((4,))]
phase_shifts = [torch.rand((4,))]
input_types = [1, 1, 1, 1]
t = 1200
num_steps = 100

# prev_timestamp = '12-03_13-26-09-778'
# fname = 'snn_model_target_GD_test'
# load_data = torch.load(IO.PATH + Izhikevich.__name__ + '/' + prev_timestamp + '/' + fname + IO.fname_ext)
# snn_target = load_data['model']
params_model = experiments.draw_from_uniform(Izhikevich.parameter_init_intervals, N=4)
snn_target = Izhikevich(parameters=params_model, N=4)
# current_inputs = experiments.generate_composite_input_of_white_noise_modulated_sine_waves(t, A_coeffs, phase_shifts, input_types)
white_noise = torch.rand((1200, snn_target.N))
current_inputs = white_noise
target_vs, target_spikes = model_util.feed_inputs_sequentially_return_tuple(snn_target, current_inputs.clone().detach())

# other_parameters = experiments.draw_from_uniform(microGIF.parameter_init_intervals, N=snn_target.N)
other_parameters = snn_target.get_parameters()
other_parameters['N'] = snn_target.N
# other_parameters = snn_target.parameters()
# plot_param_landscape(microGIF, [0.01, 1.], [-5., 10.], 'c', 'E_L', other_parameters, target_spikes, num_steps=num_steps,
#                      inputs=current_inputs.clone().detach())
# parameter_init_intervals = {'a': [0.02, 0.05], 'b': [0.25, 0.27], 'c': [-65., -55.], 'd': [4., 8.], 'R_I': [40., 50.],
#                                 'tau_s': [2., 3.5]}
plot_param_landscape(Izhikevich, [0.01, 0.2], [0.2, 0.3], 'a', 'b', other_parameters, target_spikes, num_steps=num_steps, inputs=current_inputs.clone().detach(), fname_addition='white_noise')
plot_param_landscape(Izhikevich, [0.01, 0.2], [-70., -40.], 'a', 'c', other_parameters, target_spikes, num_steps=num_steps, inputs=current_inputs.clone().detach(), fname_addition='white_noise')
# plot_param_landscape(Izhikevich, [0.2, 0.3], [-70., -40.], 'b', 'c', other_parameters, target_spikes, num_steps=num_steps, inputs=current_inputs.clone().detach(), fname_addition='white_noise')
# plot_param_landscape(Izhikevich, [0.15, 0.35], [1., 10.], 'b', 'd', other_parameters, target_spikes, num_steps=num_steps, inputs=current_inputs.clone().detach(), fname_addition='white_noise')

# plot_param_landscape(Izhikevich, [0., 1.], [-1., 0.], 'w_excit', 'w_inhib', other_parameters, target_spikes, num_steps=num_steps, inputs=current_inputs.clone().detach())

# plot_param_landscape(Izhikevich, [0.01, 1.], [1., 20.], 'tau_m', 'tau_s', other_parameters, target_spikes, num_steps=num_steps,
#                      inputs=current_inputs.clone().detach())
# plot_param_landscape(Izhikevich, [0.01, 1.], [1., 20.], 'E_L', 'tau_m', other_parameters, target_spikes, num_steps=num_steps,
#                      inputs=current_inputs.clone().detach())
# plot_param_landscape(Izhikevich, [0.01, 1.], [1., 20.], 'E_L', 'tau_s', other_parameters, target_spikes, num_steps=num_steps,
#                      inputs=current_inputs.clone().detach())

sys.exit(0)
