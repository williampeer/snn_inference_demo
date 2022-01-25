import torch
import torch.nn as nn
from torch import FloatTensor as FT

from Models.LIF_R_ASC import LIF_R_ASC
from Models.TORCH_CUSTOM import static_clamp_for, static_clamp_for_matrix


class LIF_R_ASC_soft(nn.Module):
    parameter_names = ['w', 'E_L', 'tau_m', 'G', 'f_v', 'f_I', 'delta_theta_s', 'b_s', 'delta_V', 'tau_g']  # 0,2,3,6,9
    parameter_init_intervals = {'E_L': [-68., -45.], 'tau_m': [4., 4.5], 'G': [0.7, 0.8], 'f_v': [0.2, 0.4], 'f_I': [0.3, 0.4],
                                'delta_theta_s': [10., 20.], 'b_s': [0.2, 0.4], 'delta_V': [8., 14.], 'tau_g': [4.7, 5.7]}

    def __init__(self, parameters, N=12, w_mean=0.2, w_var=0.15,
                 neuron_types=torch.tensor([1, -1])):
        super(LIF_R_ASC_soft, self).__init__()

        if parameters is not None:
            for key in parameters.keys():
                if key == 'tau_m':
                    tau_m = FT(torch.ones((N,)) * parameters[key])
                elif key == 'tau_g':
                    tau_g = FT(torch.ones((N,)) * parameters[key])
                elif key == 'G':
                    G = FT(torch.ones((N,)) * parameters[key])
                elif key == 'E_L':
                    E_L = FT(torch.ones((N,)) * parameters[key])
                elif key == 'delta_theta_s':
                    delta_theta_s = FT(torch.ones((N,)) * parameters[key])
                elif key == 'b_s':
                    b_s = FT(torch.ones((N,)) * parameters[key])
                elif key == 'f_v':
                    f_v = FT(torch.ones((N,)) * parameters[key])
                elif key == 'delta_V':
                    delta_V = FT(torch.ones((N,)) * parameters[key])
                elif key == 'f_I':
                    f_I = FT(torch.ones((N,)) * parameters[key])
                elif key == 'w_mean':
                    w_mean = FT(torch.ones((N,)) * parameters[key])
                elif key == 'w_var':
                    w_var = FT(torch.ones((N,)) * parameters[key])

        __constants__ = ['N', 'norm_R_const', 'self_recurrence_mask']
        self.N = N
        self.norm_R_const = (delta_theta_s - E_L) * 1.1

        self.v = E_L * torch.ones((self.N,))
        self.g = torch.zeros_like(self.v)  # syn. conductance
        self.spiked = torch.zeros_like(self.v)  # spike prop. for next time-step
        self.theta_s = delta_theta_s * torch.ones((self.N,))
        # self.theta_v = torch.ones((self.N,))
        self.I_additive = torch.zeros((self.N,))

        if parameters.__contains__('preset_weights'):
            # print('DEBUG: Setting w to preset weights: {}'.format(parameters['preset_weights']))
            # print('Setting w to preset weights.')
            rand_ws = parameters['preset_weights']
            assert rand_ws.shape[0] == N and rand_ws.shape[1] == N, "shape of weights matrix should be NxN"
        else:
            rand_ws = (w_mean - w_var) + 2 * w_var * torch.rand((self.N, self.N))
        nt = torch.tensor(neuron_types).float()
        self.neuron_types = torch.transpose((nt * torch.ones((self.N, self.N))), 0, 1)
        self.w = nn.Parameter(FT(rand_ws), requires_grad=True)  # initialise with positive weights only
        self.self_recurrence_mask = torch.ones((self.N, self.N)) - torch.eye(self.N, self.N)

        self.E_L = nn.Parameter(FT(E_L).clamp(-80., -35.), requires_grad=True)
        self.tau_m = nn.Parameter(FT(tau_m).clamp(1.5, 8.), requires_grad=True)
        self.tau_g = nn.Parameter(FT(tau_g).clamp(1., 12.), requires_grad=True)
        self.G = nn.Parameter(FT(G).clamp(0.01, 0.99), requires_grad=True)
        self.f_v = nn.Parameter(FT(f_v).clamp(0.01, 0.99), requires_grad=True)
        self.f_I = nn.Parameter(FT(f_I).clamp(0.01, 0.99), requires_grad=True)
        self.delta_theta_s = nn.Parameter(FT(delta_theta_s).clamp(6., 30.), requires_grad=True)
        self.b_s = nn.Parameter(FT(b_s).clamp(0.01, 0.99), requires_grad=True)
        self.delta_V = nn.Parameter(FT(delta_V).clamp(1., 35.), requires_grad=True)
        # self.I_A = nn.Parameter(FT(I_A).clamp(0.5, 3.), requires_grad=True)

        self.register_backward_clamp_hooks()

    def get_parameters(self):
        params_list = []
        # parameter_names = ['w', 'E_L', 'tau_m', 'G', 'f_v', 'f_I', 'delta_theta_s', 'b_s', 'delta_V', 'tau_s']
        params_list.append(self.w.data)
        params_list.append(self.E_L.data)
        params_list.append(self.tau_m.data)
        params_list.append(self.G.data)
        params_list.append(self.f_v.data)
        params_list.append(self.f_I.data)
        params_list.append(self.delta_theta_s.data)
        params_list.append(self.b_s.data)
        params_list.append(self.delta_V.data)
        params_list.append(self.tau_g.data)

        return params_list

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
        self.spiked = self.spiked.clone().detach()

        self.theta_s = self.theta_s.clone().detach()
        self.I_additive = self.I_additive.clone().detach()

    def register_backward_clamp_hooks(self):
        self.E_L.register_hook(lambda grad: static_clamp_for(grad, -80., -35., self.E_L))
        self.tau_m.register_hook(lambda grad: static_clamp_for(grad, 1.5, 8., self.tau_m))
        self.tau_g.register_hook(lambda grad: static_clamp_for(grad, 1., 12., self.tau_g))
        self.G.register_hook(lambda grad: static_clamp_for(grad, 0.01, 0.99, self.G))
        self.f_v.register_hook(lambda grad: static_clamp_for(grad, 0.01, 0.99, self.f_v))
        self.f_I.register_hook(lambda grad: static_clamp_for(grad, 0.01, 0.99, self.f_I))
        self.delta_theta_s.register_hook(lambda grad: static_clamp_for(grad, 6., 30., self.delta_theta_s))
        self.b_s.register_hook(lambda grad: static_clamp_for(grad, 0.01, 0.99, self.b_s))
        self.delta_V.register_hook(lambda grad: static_clamp_for(grad, 1., 35., self.delta_V))
        # self.I_A.register_hook(lambda grad: static_clamp_for(grad, 0.5, 3., self.I_A))

        self.w.register_hook(lambda grad: static_clamp_for_matrix(grad, 0., 1., self.w))

    def name(self):
        return LIF_R_ASC.__name__

    def forward(self, x_in):
        # assuming input weights to be Eye(N,N)
        W_syn = self.w * self.neuron_types
        # I = (self.I_additive + self.s).matmul(self.self_recurrence_mask * W_syn) + 1.75 * x_in
        I_tot = ((self.I_additive + self.g) / 2).matmul(self.self_recurrence_mask * W_syn) + 1.75 * x_in

        # dv = (self.G * (self.E_L - self.v) + I * self.R_I) / self.tau_m
        # I_syn = self.I_additive.matmul(self.w)
        # I_tot = 2 * torch.sigmoid(I_syn + 6 * x_in) - 1  # in (-1, 1)

        dv = (self.G * (self.E_L - self.v) + I_tot * self.norm_R_const) / self.tau_m
        v_next = self.v + dv

        # differentiable
        self.spiked = torch.sigmoid(torch.sub(v_next, self.theta_s))
        # non-differentiable, hard threshold
        spiked = (v_next >= self.theta_s).float()  # thresholding when spiked isn't use for grad.s (non-differentiable)
        not_spiked = (spiked - 1.) / -1.  # flips the boolean mat.

        v_reset = self.E_L + self.f_v * (self.v - self.E_L) - self.delta_V
        self.v = spiked * v_reset + not_spiked * v_next

        # theta_s_next = self.theta_s - self.b_s * self.theta_s
        # self.theta_s = spiked * (self.theta_s + self.delta_theta_s) + not_spiked * theta_s_next
        self.theta_s = (1. - self.b_s) * self.theta_s + spiked * self.delta_theta_s  # always decay

        # I_additive_decayed = (torch.ones_like(self.f_I) - self.f_I) * self.I_additive
        # self.I_additive = spiked * (self.I_additive + self.I_A) + not_spiked * I_additive_decayed
        self.I_additive = self.I_additive - self.f_I * self.I_additive + spiked * self.f_I

        dg = -torch.div(self.g, self.tau_g)  # -g/tau_g
        self.g = torch.add(spiked * torch.ones_like(self.g), not_spiked * torch.add(self.g, dg))

        # return self.v, self.spiked
        return self.spiked
