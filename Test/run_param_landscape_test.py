import sys

import torch

import IO
import experiments
import model_util
from Models.microGIF import microGIF
from Test.parameter_landscape_test import plot_param_landscape

A_coeffs = [torch.randn((4,))]
phase_shifts = [torch.rand((4,))]
input_types = [1, 1, 1, 1]
t = 1200
num_steps = 100

prev_timestamp = '11-16_11-21-13-903'
fname = 'snn_model_target_GD_test'
load_data = torch.load(IO.PATH + microGIF.__name__ + '/' + prev_timestamp + '/' + fname + IO.fname_ext)
snn_target = load_data['model']
current_inputs = experiments.generate_composite_input_of_white_noise_modulated_sine_waves(t, A_coeffs, phase_shifts, input_types)
_, target_spikes, target_vs = model_util.feed_inputs_sequentially_return_args(snn_target, current_inputs.clone().detach())

# other_parameters = experiments.draw_from_uniform(microGIF.parameter_init_intervals, N=snn_target.N)
other_parameters = snn_target.get_parameters()
# plot_param_landscape(microGIF, [0.01, 1.], [-5., 10.], 'c', 'E_L', other_parameters, target_spikes, num_steps=num_steps,
#                      inputs=current_inputs.clone().detach())
plot_param_landscape(microGIF, [0.01, 1.], [1., 20.], 'c', 'tau_m', other_parameters, target_spikes, num_steps=num_steps,
                     inputs=current_inputs.clone().detach())
plot_param_landscape(microGIF, [0.01, 1.], [1., 20.], 'c', 'tau_s', other_parameters, target_spikes, num_steps=num_steps,
                     inputs=current_inputs.clone().detach())
plot_param_landscape(microGIF, [0.01, 1.], [1., 20.], 'c', 'Delta_u', other_parameters, target_spikes, num_steps=num_steps,
                     inputs=current_inputs.clone().detach())
plot_param_landscape(microGIF, [0.01, 1.], [0.1, 3.], 'c', 'J_theta', other_parameters, target_spikes, num_steps=num_steps,
                     inputs=current_inputs.clone().detach())
plot_param_landscape(microGIF, [0.01, 1.], [1., 20.], 'tau_m', 'tau_s', other_parameters, target_spikes, num_steps=num_steps,
                     inputs=current_inputs.clone().detach())
plot_param_landscape(microGIF, [0.01, 1.], [1., 20.], 'E_L', 'tau_m', other_parameters, target_spikes, num_steps=num_steps,
                     inputs=current_inputs.clone().detach())
plot_param_landscape(microGIF, [0.01, 1.], [1., 20.], 'E_L', 'tau_s', other_parameters, target_spikes, num_steps=num_steps,
                     inputs=current_inputs.clone().detach())

sys.exit(0)
