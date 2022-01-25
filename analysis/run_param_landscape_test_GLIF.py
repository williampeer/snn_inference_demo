import sys

import torch

import IO
import model_util
from Models.GLIF import GLIF
from Test.parameter_landscape_test import plot_param_landscape

A_coeffs = [torch.randn((4,))]
phase_shifts = [torch.rand((4,))]
input_types = [1, 1, 1, 1]
t = 1200
num_steps = 100

# prev_timestamp = '12-09_11-14-47-449'
prev_timestamp = '12-09_11-12-47-541'
fname = 'snn_model_target_GD_test'
load_data = torch.load('./Test/' + IO.PATH + GLIF.__name__ + '/' + prev_timestamp + '/' + fname + IO.fname_ext)
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
# other_parameters = snn_target.parameters()
# free_parameters = ['w', 'E_L', 'tau_m', 'G', 'f_v', 'f_I', 'delta_theta_s', 'b_s', 'a_v', 'b_v', 'theta_inf', 'delta_V', 'tau_s']
plot_param_landscape(GLIF, [-68., -40.], [1.5, 10.], 'E_L', 'tau_m', other_parameters, target_spikes, num_steps=num_steps, inputs=current_inputs.clone().detach(), N=snn_target.N, fname_addition='white_noise', GIF_flag=False, lfn='frd')
plot_param_landscape(GLIF, [0.3, 1.0], [1.5, 10.], 'G', 'tau_m', other_parameters, target_spikes, num_steps=num_steps, inputs=current_inputs.clone().detach(), N=snn_target.N, fname_addition='white_noise', GIF_flag=False, lfn='frd')
plot_param_landscape(GLIF, [1., 12.], [1.5, 10.], 'tau_s', 'tau_m', other_parameters, target_spikes, num_steps=num_steps, inputs=current_inputs.clone().detach(), N=snn_target.N, fname_addition='white_noise', GIF_flag=False, lfn='frd')
plot_param_landscape(GLIF, [0.1, 1.0], [0.1, 1.0], 'f_v', 'f_I', other_parameters, target_spikes, num_steps=num_steps, inputs=current_inputs.clone().detach(), N=snn_target.N, fname_addition='white_noise', GIF_flag=False, lfn='frd')
plot_param_landscape(GLIF, [0.1, 1.0], [0.1, 1.0], 'b_s', 'a_v', other_parameters, target_spikes, num_steps=num_steps, inputs=current_inputs.clone().detach(), N=snn_target.N, fname_addition='white_noise', GIF_flag=False, lfn='frd')
plot_param_landscape(GLIF, [0.1, 1.0], [0.1, 1.0], 'a_v', 'b_v', other_parameters, target_spikes, num_steps=num_steps, inputs=current_inputs.clone().detach(), N=snn_target.N, fname_addition='white_noise', GIF_flag=False, lfn='frd')
plot_param_landscape(GLIF, [2., 30.], [-20., 0.], 'delta_theta_s', 'theta_inf', other_parameters, target_spikes, num_steps=num_steps, inputs=current_inputs.clone().detach(), N=snn_target.N, fname_addition='white_noise', GIF_flag=False, lfn='frd')
plot_param_landscape(GLIF, [4., 20.], [-20., 0.], 'delta_V', 'theta_inf', other_parameters, target_spikes, num_steps=num_steps, inputs=current_inputs.clone().detach(), N=snn_target.N, fname_addition='white_noise', GIF_flag=False, lfn='frd')

sys.exit(0)
