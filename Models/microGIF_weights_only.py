import torch
import torch.nn as nn
from torch import FloatTensor as FT

from Models.TORCH_CUSTOM import static_clamp_for_matrix


class microGIF_weights_only(nn.Module):
    free_parameters = ['w']
    param_lin_constraints = [[0., 2.]]

    def __init__(self, parameters, N=4, neuron_types=[1, -1]):
        super(microGIF_weights_only, self).__init__()

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
            rand_ws = 0.5 * torch.randn((N, N))
        nt = torch.tensor(neuron_types).float()
        self.neuron_types = nt
        # self.neuron_types = (torch.ones((self.N, 1)) * nt).T
        self.w = nn.Parameter(FT(rand_ws).clip(-10., 10.), requires_grad=True)  # initialise with positive weights only
        self.self_recurrence_mask = torch.ones((self.N, self.N)) - torch.eye(self.N, self.N)

        self.v = E_L * torch.ones((self.N,))
        self.spiked = torch.zeros((self.N,))
        self.time_since_spike = 1e4 * torch.ones((N,))

        self.E_L = FT(E_L)  # Rest potential
        self.tau_m = FT(tau_m)
        self.tau_s = FT(tau_s)
        self.tau_theta = FT(tau_theta)  # Adaptation time constant
        self.J_theta = FT(J_theta)  # Adaptation strength
        self.c = FT(c)
        self.Delta_u = FT(Delta_u)  # Noise level
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
        self.w.register_hook(lambda grad: static_clamp_for_matrix(grad, -10., 10., self.w))

    def get_parameters(self):
        return { 'w': self.w.data }

    def name(self):
        return self.__class__.__name__

    def forward(self, I_ext):
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

        m = torch.distributions.bernoulli.Bernoulli(spikes_lambda)
        spiked = m.sample()

        self.spiked = spiked
        not_spiked = (spiked - 1) / -1

        self.time_since_spike = not_spiked * (self.time_since_spike + 1)
        self.v = not_spiked * v_next + spiked * self.reset_potential

        # return spikes_lambda, spiked
        return spikes_lambda, spiked, self.v
