import sys

import torch

import IO
import model_util
from Models.LIF import LIF
from Test.parameter_landscape_test import plot_param_landscape

A_coeffs = [torch.randn((4,))]
phase_shifts = [torch.rand((4,))]
input_types = [1, 1, 1, 1]
t = 1200
num_steps = 100

prev_timestamp = '12-09_11-49-59-999'
fname = 'snn_model_target_GD_test'
load_data = torch.load('./Test/' + IO.PATH + LIF.__name__ + '/' + prev_timestamp + '/' + fname + IO.fname_ext)
snn_target = load_data['model']
# params_model = experiments.draw_from_uniform(GLIF.parameter_init_intervals, N=4)
# snn_target = TargetModelsBestEffort.glif(random_seed=42, N=4)
# current_inputs = experiments.generate_composite_input_of_white_noise_modulated_sine_waves(t, A_coeffs, phase_shifts, input_types)
white_noise = torch.rand((1200, snn_target.N))
current_inputs = white_noise
target_vs, target_spikes = model_util.feed_inputs_sequentially_return_tuple(snn_target, current_inputs.clone().detach())

# other_parameters = experiments.draw_from_uniform(microGIF.parameter_init_intervals, N=snn_target.N)
other_parameters = snn_target.get_parameters()
other_parameters['N'] = snn_target.N
other_parameters['neuron_types'] = snn_target.neuron_types.detach().numpy()
# other_parameters = snn_target.parameters()
# free_parameters = ['w', 'E_L', 'tau_m', 'G', 'f_v', 'f_I', 'delta_theta_s', 'b_s', 'a_v', 'b_v', 'theta_inf', 'delta_V', 'tau_s']
plot_param_landscape(LIF, [-70., -40.], [1.5, 10.], 'E_L', 'tau_m', other_parameters, target_spikes, num_steps=num_steps, inputs=current_inputs.clone().detach(), N=snn_target.N, fname_addition='white_noise', lfn='frd')
plot_param_landscape(LIF, [-70., -40.], [1., 12.], 'E_L', 'tau_s', other_parameters, target_spikes, num_steps=num_steps, inputs=current_inputs.clone().detach(), N=snn_target.N, fname_addition='white_noise', lfn='frd')
plot_param_landscape(LIF, [1.5, 10.], [1., 12.], 'tau_m', 'tau_s', other_parameters, target_spikes, num_steps=num_steps, inputs=current_inputs.clone().detach(), N=snn_target.N, fname_addition='white_noise', lfn='frd')

sys.exit(0)
