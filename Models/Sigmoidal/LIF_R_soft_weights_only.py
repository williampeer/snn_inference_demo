import torch
import torch.nn as nn
from torch import FloatTensor as FT
from torch import tensor as T

from Models.LIF_R import LIF_R
from Models.TORCH_CUSTOM import static_clamp_for, static_clamp_for_matrix


class LIF_R_soft_weights_only(nn.Module):
    parameter_names = ['w']#, 'E_L', 'tau_m', 'G', 'f_v', 'delta_theta_s', 'b_s', 'delta_V', 'tau_g']
    # parameter_init_intervals = {'E_L': [-64., -52.], 'tau_m': [3., 4.], 'G': [0.7, 0.8],
    #                             'f_v': [0.2, 0.4], 'delta_theta_s': [10., 20.], 'b_s': [0.2, 0.4],
    #                             'delta_V': [8., 14.], 'tau_g': [4., 5.]}

    def __init__(self, parameters, N=12, w_mean=0.3, w_var=0.2, neuron_types=T([1, -1])):
        super(LIF_R_soft_weights_only, self).__init__()
        # self.device = device

        if parameters:
            for key in parameters.keys():
                if key == 'tau_m':
                    tau_m = FT(torch.ones((N,)) * parameters[key])
                elif key == 'tau_g':
                    tau_g = FT(torch.ones((N,)) * parameters[key])
                elif key == 'E_L':
                    E_L = FT(torch.ones((N,)) * parameters[key])
                elif key == 'G':
                    G = FT(torch.ones((N,)) * parameters[key])
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

        __constants__ = ['N', 'norm_R_const']
        self.N = N
        self.norm_R_const = (delta_theta_s - E_L) * 1.1

        self.v = torch.zeros((self.N,))
        self.g = torch.zeros_like(self.v)  # syn. conductance
        self.spiked = torch.zeros_like(self.v)  # spike prop. for next time-step
        self.theta_s = delta_theta_s * torch.ones((self.N,))

        if parameters.__contains__('preset_weights'):
            rand_ws = torch.abs(parameters['preset_weights'])
            assert rand_ws.shape[0] == N and rand_ws.shape[1] == N, "shape of weights matrix should be NxN"
        else:
            rand_ws = (w_mean - w_var) + 2 * w_var * torch.rand((self.N, self.N))
        nt = T(neuron_types).float()
        self.neuron_types = torch.transpose((nt * torch.ones((self.N, self.N))), 0, 1)
        self.w = nn.Parameter(FT(rand_ws), requires_grad=True)  # initialise with positive weights only
        self.self_recurrence_mask = torch.ones((self.N, self.N)) - torch.eye(self.N, self.N)

        self.E_L = FT(E_L).clamp(-80., -35.)
        self.b_s = FT(b_s).clamp(0.01, 0.99)
        self.G = FT(G)
        self.tau_m = FT(tau_m).clamp(1.5, 8.)
        self.tau_g = FT(tau_g).clamp(1., 12.)
        self.delta_theta_s = FT(delta_theta_s).clamp(6., 30.)
        self.f_v = FT(f_v).clamp(0.01, 0.99)
        self.delta_V = FT(delta_V).clamp(1., 35.)

        self.register_backward_clamp_hooks()

    def register_backward_clamp_hooks(self):
        self.w.register_hook(lambda grad: static_clamp_for_matrix(grad, 0., 1., self.w))

    def reset(self):
        for p in self.parameters():
            p.grad = None
        self.reset_hidden_state()

        self.v = self.E_L.clone().detach() * torch.ones((self.N,))
        self.spiked = torch.zeros_like(self.v)  # spike prop. for next time-step
        self.g = torch.zeros_like(self.v)  # syn. conductance

    def reset_hidden_state(self):
        self.v = self.v.clone().detach()
        self.g = self.g.clone().detach()
        self.theta_s = self.theta_s.clone().detach()
        self.spiked = self.spiked.clone().detach()

    def name(self):
        return LIF_R.__name__

    def get_parameters(self):
        params_list = []
        # parameter_names = ['w', 'E_L', 'tau_m', 'tau_s', 'G', 'f_v', 'delta_theta_s', 'b_s', 'delta_V']
        params_list.append(self.w.data)
        # params_list.append(self.E_L.data)
        # params_list.append(self.tau_m.data)
        # params_list.append(self.G.data)
        # params_list.append(self.f_v.data)
        # params_list.append(self.delta_theta_s.data)
        # params_list.append(self.b_s.data)
        # params_list.append(self.delta_V.data)
        # params_list.append(self.tau_g.data)

        return params_list

    def forward(self, I_ext):
        W_syn = self.w * self.neuron_types
        I_syn = (self.g).matmul(self.self_recurrence_mask * W_syn)

        dv = (self.G * (self.E_L - self.v) + (I_syn + I_ext) * self.norm_R_const) / self.tau_m
        v_next = self.v + dv

        # differentiability
        self.spiked = torch.sigmoid(torch.sub(v_next, self.theta_s))
        # non-differentiable, hard threshold
        spiked = (v_next >= self.theta_s).float()
        not_spiked = (spiked - 1.) / -1.

        v_reset = self.E_L + self.f_v * (self.v - self.E_L) - self.delta_V
        self.v = spiked * v_reset + not_spiked * v_next

        self.theta_s = (1-self.b_s) * self.theta_s + spiked * self.delta_theta_s

        dg = -torch.div(self.g, self.tau_g)  # -g/tau_g
        self.g = torch.add(spiked * torch.ones_like(self.g), not_spiked * torch.add(self.g, dg))

        # return self.v, self.spiked
        return self.spiked
