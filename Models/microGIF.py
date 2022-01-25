import torch
import torch.nn as nn
from torch import FloatTensor as FT

from Models.TORCH_CUSTOM import static_clamp_for, static_clamp_for_matrix


class microGIF(nn.Module):
    free_parameters = ['w', 'E_L', 'tau_m', 'tau_s', 'tau_theta', 'J_theta', 'c', 'Delta_u']
    parameter_init_intervals = { 'E_L': [0., 3.], 'tau_m': [7., 9.], 'tau_s': [4., 8.], 'tau_theta': [950., 1050.],
                                 'J_theta': [0.9, 1.1], 'c': [0.15, 0.2], 'Delta_u': [3.5, 4.5] }
    param_lin_constraints = [[0., 2.], [-5., 25.], [1., 20.], [1., 20.], [800., 1500.], [0.1, 2.], [0.01, 1.], [1., 20.]]

    # def __init__(self, parameters, N=4, neuron_types=[1, -1]):
    def __init__(self, parameters, N=4):
        super(microGIF, self).__init__()

        if parameters is not None:
            for key in parameters.keys():
                if key == 'tau_m':
                    tau_m = FT(torch.ones((N,)) * parameters[key])
                elif key == 'tau_s':
                    tau_s = FT(torch.ones((N,)) * parameters[key])
                elif key == 'tau_theta':
                    tau_theta = FT(torch.ones((N,)) * parameters[key])
                elif key == 'J_theta':
                    J_theta = FT(torch.ones((N,)) * parameters[key])
                elif key == 'E_L':
                    E_L = FT(torch.ones((N,)) * parameters[key])
                elif key == 'c':
                    c = FT(torch.ones((N,)) * parameters[key])
                elif key == 'Delta_u':
                    Delta_u = FT(torch.ones((N,)) * parameters[key])

        __constants__ = ['N', 'self_recurrence_mask']
        self.N = N

        if parameters.__contains__('preset_weights'):
            print('Setting w to preset weights in {}.'.format(self.__class__.__name__))
            rand_ws = parameters['preset_weights']
            assert rand_ws.shape[0] == N and rand_ws.shape[1] == N, "shape of weights matrix should be NxN"
        else:
            # rand_ws = torch.abs((0.5 - 0.25) + 2 * 0.25 * torch.rand((self.N, self.N)))
            rand_ws = 8. * torch.randn((N, N))
        # nt = torch.tensor(neuron_types).float()
        # self.neuron_types = torch.transpose((nt * torch.ones((self.N, self.N))), 0, 1)
        # self.neuron_types = (torch.ones((self.N, 1)) * nt).T
        # self.neuron_types = nt
        self.w = nn.Parameter(FT(rand_ws).clip(-10., 10.), requires_grad=True)  # initialise with positive weights only
        self.self_recurrence_mask = torch.ones((self.N, self.N)) - torch.eye(self.N, self.N)
        # self.self_recurrence_mask = torch.ones((self.N, self.N))

        self.v = E_L * torch.ones((self.N,))
        self.spiked = torch.zeros((self.N,))
        # self.g = torch.zeros((self.N,))
        self.time_since_spike = 1e4 * torch.ones((N,))

        self.E_L = nn.Parameter(FT(E_L), requires_grad=True)  # Rest potential
        self.tau_m = nn.Parameter(FT(tau_m), requires_grad=True)
        self.tau_s = nn.Parameter(FT(tau_s), requires_grad=True)
        self.tau_theta = nn.Parameter(FT(tau_theta), requires_grad=True)  # Adaptation time constant
        self.J_theta = nn.Parameter(FT(J_theta), requires_grad=True)  # Adaptation strength
        self.c = nn.Parameter(FT(c), requires_grad=True)
        self.Delta_u = nn.Parameter(FT(Delta_u), requires_grad=True)  # Noise level
        self.Delta_delay = 1.  # Transmission delay
        self.theta_inf = FT(torch.ones((N,)) * 15.)
        self.reset_potential = 0.
        self.theta_v = FT(self.theta_inf * torch.ones((N,)))
        # self.R_m = FT(R_m)
        self.t_refractory = 2.

        self.register_backward_clamp_hooks()

    def reset(self):
        for p in self.parameters():
            p.grad = None
        self.reset_hidden_state()

    def reset_hidden_state(self):
        self.v = self.v.clone().detach()
        self.spiked = self.spiked.clone().detach()
        self.time_since_spike = self.time_since_spike.clone().detach()
        self.theta_v = self.theta_v.clone().detach()

    def register_backward_clamp_hooks(self):
        self.E_L.register_hook(lambda grad: static_clamp_for(grad, -5., 25., self.E_L, 'E_L'))
        self.tau_m.register_hook(lambda grad: static_clamp_for(grad, 2., 20., self.tau_m, 'tau_m'))
        self.tau_s.register_hook(lambda grad: static_clamp_for(grad, 1., 20., self.tau_s, 'tau_s'))
        self.tau_theta.register_hook(lambda grad: static_clamp_for(grad, 800., 1500, self.tau_theta, 'tau_theta'))
        self.J_theta.register_hook(lambda grad: static_clamp_for(grad, 0.1, 2., self.J_theta, 'J_theta'))
        self.c.register_hook(lambda grad: static_clamp_for(grad, 0.01, 1., self.c, 'c'))
        self.Delta_u.register_hook(lambda grad: static_clamp_for(grad, 1., 20., self.Delta_u, 'Delta_u'))

        self.w.register_hook(lambda grad: static_clamp_for_matrix(grad, -10., 10., self.w))

    def get_parameters(self):
        params_dict = {}

        params_dict['w'] = self.w.data
        params_dict['E_L'] = self.E_L.data
        params_dict['tau_m'] = self.tau_m.data
        params_dict['tau_s'] = self.tau_s.data
        params_dict['tau_theta'] = self.tau_theta.data
        params_dict['J_theta'] = self.J_theta.data
        params_dict['Delta_u'] = self.Delta_u.data
        params_dict['c'] = self.c.data

        # params_dict['R_m'] = self.R_m
        params_dict['theta_inf'] = self.theta_inf

        return params_dict

    def name(self):
        return self.__class__.__name__

    def forward(self, I_ext):
        #   adaptation_kernel can be rewritten to:
        dtheta_v = (self.theta_inf - self.theta_v + self.J_theta * self.spiked) / self.tau_theta
        self.theta_v = self.theta_v + dtheta_v

        epsilon_spike_pulse = (1 + torch.tanh(self.time_since_spike - self.Delta_delay)) * torch.exp(
            -(self.time_since_spike - self.Delta_delay) / self.tau_s) / self.tau_s

        # W_syn = self.self_recurrence_mask * self.w * self.neuron_types
        W_syn = self.self_recurrence_mask * self.w
        I_syn = ((W_syn).matmul(epsilon_spike_pulse))
        # I_syn = ((W_syn) * (epsilon_spike_pulse)).sum(dim=0)
        dv = (self.E_L - self.v + I_ext) / self.tau_m + I_syn
        v_next = self.v + dv

        not_refractory = torch.where(self.time_since_spike > self.t_refractory, 1, 0)

        spikes_lambda = not_refractory * (self.c * torch.exp((v_next - self.theta_v) / self.Delta_u))
        spikes_lambda = spikes_lambda.clip(0., 1.)
        spikes_lambda[torch.isnan(spikes_lambda)] = 1.  # tmp nan-fix

        m = torch.distributions.bernoulli.Bernoulli(spikes_lambda)
        spiked = m.sample()

        self.spiked = spiked
        not_spiked = (spiked - 1) / -1

        self.time_since_spike = not_spiked * (self.time_since_spike + 1)
        self.v = not_spiked * v_next + spiked * self.reset_potential

        return spikes_lambda, spiked, self.v
