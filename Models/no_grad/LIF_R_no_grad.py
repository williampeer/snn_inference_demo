import torch
import torch.nn as nn
from torch import FloatTensor as FT
from torch import tensor as T

from Models.LIF_R import LIF_R


class LIF_R_no_grad(nn.Module):
    parameter_names = ['w', 'E_L', 'tau_m', 'G', 'f_v', 'delta_theta_s', 'b_s', 'delta_V', 'tau_s']
    parameter_init_intervals = {'E_L': [-64., -55.], 'tau_m': [3.5, 4.0], 'G': [0.7, 0.8],
                                'f_v': [0.25, 0.35], 'delta_theta_s': [10., 20.], 'b_s': [0.25, 0.35],
                                'delta_V': [8., 14.], 'tau_s': [5., 6.]}
    param_lin_constraints = [[0., 1.], [-80., -35.], [1.2, 8.], [0.01, 0.99], [0.01, 0.99], [6., 30.], [0.01, 0.95],
                             [1., 35.], [1.5, 12.]]

    def __init__(self, parameters, N=12, w_mean=0.4, w_var=0.25, neuron_types=T([1, -1])):
        super(LIF_R_no_grad, self).__init__()
        # self.device = device

        if parameters:
            for key in parameters.keys():
                if key == 'tau_m':
                    tau_m = FT(torch.ones((N,)) * parameters[key])
                elif key == 'E_L':
                    E_L = FT(torch.ones((N,)) * parameters[key])
                elif key == 'G':
                    G = FT(torch.ones((N,)) * parameters[key])
                elif key == 'tau_s':
                    tau_s = FT(torch.ones((N,)) * parameters[key])
                elif key == 'w_mean':
                    w_mean = float(parameters[key])
                elif key == 'w_var':
                    w_var = float(parameters[key])
                elif key == 'delta_theta_s':
                    delta_theta_s = FT(torch.ones((N,)) * parameters[key])
                elif key == 'b_s':
                    b_s = FT(torch.ones((N,)) * parameters[key])
                elif key == 'f_v':
                    f_v = FT(torch.ones((N,)) * parameters[key])
                elif key == 'delta_V':
                    delta_V = FT(torch.ones((N,)) * parameters[key])

        __constants__ = ['N', 'norm_R_const', 'self_recurrence_mask']
        self.N = N
        self.norm_R_const = (delta_theta_s - E_L) * 1.1

        self.v = torch.zeros((self.N,))
        self.s = torch.zeros_like(self.v)  # syn. conductance
        self.theta_s = delta_theta_s * torch.ones((self.N,))

        self.self_recurrence_mask = torch.ones((self.N, self.N)) - torch.eye(self.N, self.N)
        if parameters.__contains__('preset_weights'):
            # print('DEBUG: Setting w to preset weights: {}'.format(parameters['preset_weights']))
            # print('Setting w to preset weights.')
            rand_ws = torch.abs(parameters['preset_weights'])
            assert rand_ws.shape[0] == N and rand_ws.shape[1] == N, "shape of weights matrix should be NxN"
        else:
            rand_ws = (w_mean - w_var) + 2 * w_var * torch.rand((self.N, self.N))
        nt = T(neuron_types).float()
        self.neuron_types = torch.transpose((nt * torch.ones((self.N, self.N))), 0, 1)
        self.w = FT(rand_ws)

        self.E_L = FT(E_L).clamp(-80., -35.)
        self.b_s = FT(b_s).clamp(0.01, 0.99)
        self.G = FT(G)
        self.tau_m = FT(tau_m).clamp(1.5, 8.)
        self.tau_s = FT(tau_s).clamp(1., 12.)
        self.delta_theta_s = FT(delta_theta_s).clamp(6., 30.)
        self.f_v = FT(f_v).clamp(0.01, 0.99)
        self.delta_V = FT(delta_V).clamp(1., 35.)

    def reset(self):
        for p in self.parameters():
            p.grad = None
        self.reset_hidden_state()

        self.v = self.E_L.clone().detach() * torch.ones((self.N,))
        self.s = torch.zeros_like(self.v)  # syn. conductance

    def reset_hidden_state(self):
        self.v = self.v.clone().detach()
        self.s = self.s.clone().detach()
        self.theta_s = self.theta_s.clone().detach()

    def name(self):
        return LIF_R.__name__

    def get_parameters(self):
        params_dict = {}

        params_dict['w'] = self.w.data
        params_dict['E_L'] = self.E_L.data
        params_dict['tau_m'] = self.tau_m.data
        params_dict['G'] = self.G.data
        params_dict['f_v'] = self.f_v.data
        params_dict['delta_theta_s'] = self.delta_theta_s.data
        params_dict['b_s'] = self.b_s.data
        params_dict['delta_V'] = self.delta_V.data
        params_dict['tau_s'] = self.tau_s.data

        return params_dict

    def params_wrapper(self):
        return { 0: self.w.data.numpy(), 1: self.E_L.data.numpy(), 2: self.tau_m.data.numpy(), 3: self.tau_s.data.numpy(),
                 4: self.G.data.numpy(), 5: self.f_v.data.numpy(), 6: self.delta_theta_s.data.numpy(),
                 7: self.b_s.data.numpy(), 8: self.delta_V.data.numpy() }

    def forward(self, I_ext):
        W_syn = self.w * self.neuron_types
        I_syn = (self.s).matmul(self.self_recurrence_mask * W_syn)

        dv = (self.G * (self.E_L - self.v) + (I_syn + I_ext) * self.norm_R_const) / self.tau_m
        v_next = torch.add(self.v, dv)

        gating = (v_next / self.theta_s).clamp(0., 1.)
        dv_max = (self.theta_s - self.E_L)
        ds = (-self.s + gating * (dv / dv_max).clamp(0., 1.)) / self.tau_s
        self.s = self.s + ds

        # non-differentiable, hard threshold for nonlinear reset dynamics
        spiked = (v_next >= self.theta_s).float()
        not_spiked = (spiked - 1.) / -1.

        self.theta_s = torch.add((1 - self.b_s) * self.theta_s, spiked * self.delta_theta_s)
        v_reset = self.E_L + self.f_v * (self.v - self.E_L) - self.delta_V
        self.v = torch.add(spiked * v_reset, not_spiked * v_next)

        # return self.v, self.s * self.tau_s
        # return self.s * self.tau_s  # use synaptic current as spike signal
        # return self.s * (self.tau_s + 1) / 2.  # return readout of synaptic current as spike signal

        # differentiable soft threshold
        soft_spiked = torch.sigmoid(torch.sub(v_next, self.theta_s))
        return soft_spiked  # return sigmoidal spiked
        # return gating
