import torch
import torch.nn as nn
from torch import FloatTensor as FT

from Models.LIF_R_ASC import LIF_R_ASC
from Models.TORCH_CUSTOM import static_clamp_for, static_clamp_for_matrix


class LIF_R_ASC_lower_dim(nn.Module):
    parameter_names = ['w', 'tau_m',
                       # 'E_L', 'G', 'f_v', 'f_I', 'delta_theta_s', 'b_s', 'delta_V',
                       'tau_s']  # 0,2,3,6,9
    parameter_init_intervals = { 'tau_m': [4., 4.5],
                                 # 'E_L': [-68., -45.],'G': [0.7, 0.8], 'f_v': [0.2, 0.4], 'f_I': [0.3, 0.4], 'delta_theta_s': [10., 20.], 'b_s': [0.2, 0.4], 'delta_V': [8., 14.],
                                 'tau_s': [4.7, 5.7]}

    def __init__(self, parameters, N=12, w_mean=0.2, w_var=0.15,
                 neuron_types=torch.tensor([1, -1])):
        super(LIF_R_ASC_lower_dim, self).__init__()

        if parameters is not None:
            for key in parameters.keys():
                if key == 'tau_m':
                    tau_m = FT(torch.ones((N,)) * parameters[key])
                elif key == 'tau_s':
                    tau_s = FT(torch.ones((N,)) * parameters[key])
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

        self.tau_m = nn.Parameter(FT(tau_m).clamp(1.5, 8.), requires_grad=True)
        self.tau_s = nn.Parameter(FT(tau_s).clamp(1., 12.), requires_grad=True)
        self.E_L = FT(E_L).clamp(-80., -35.)
        self.G = FT(G).clamp(0.01, 0.99)
        self.f_v = FT(f_v).clamp(0.01, 0.99)
        self.f_I = FT(f_I).clamp(0.01, 0.99)
        self.delta_theta_s = FT(delta_theta_s).clamp(6., 30.)
        self.b_s = FT(b_s).clamp(0.01, 0.99)
        self.delta_V = FT(delta_V).clamp(1., 35.)
        # self.I_A = nn.Parameter(FT(I_A).clamp(0.5, 3.), requires_grad=True)

        self.register_backward_clamp_hooks()

    def get_parameters(self):
        params_list = []
        params_list.append(self.w.data)
        params_list.append(self.tau_m.data)
        # params_list.append(self.E_L.data)
        # params_list.append(self.G.data)
        # params_list.append(self.f_v.data)
        # params_list.append(self.f_I.data)
        # params_list.append(self.delta_theta_s.data)
        # params_list.append(self.b_s.data)
        # params_list.append(self.delta_V.data)
        params_list.append(self.tau_s.data)

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
        self.tau_m.register_hook(lambda grad: static_clamp_for(grad, 1.5, 8., self.tau_m))
        self.tau_s.register_hook(lambda grad: static_clamp_for(grad, 1., 12., self.tau_s))
        # self.E_L.register_hook(lambda grad: static_clamp_for(grad, -80., -35., self.E_L))
        # self.G.register_hook(lambda grad: static_clamp_for(grad, 0.01, 0.99, self.G))
        # self.f_v.register_hook(lambda grad: static_clamp_for(grad, 0.01, 0.99, self.f_v))
        # self.f_I.register_hook(lambda grad: static_clamp_for(grad, 0.01, 0.99, self.f_I))
        # self.delta_theta_s.register_hook(lambda grad: static_clamp_for(grad, 6., 30., self.delta_theta_s))
        # self.b_s.register_hook(lambda grad: static_clamp_for(grad, 0.01, 0.99, self.b_s))
        # self.delta_V.register_hook(lambda grad: static_clamp_for(grad, 1., 35., self.delta_V))
        # self.I_A.register_hook(lambda grad: static_clamp_for(grad, 0.5, 3., self.I_A))

        self.w.register_hook(lambda grad: static_clamp_for_matrix(grad, 0., 1., self.w))

    def name(self):
        return LIF_R_ASC.__name__

    def forward(self, I_ext):
        # assuming input weights to be Eye(N,N)
        W_syn = self.w * self.neuron_types
        I_syn = ((self.I_additive + self.s) / 2).matmul(self.self_recurrence_mask * W_syn)

        dv = (self.G * (self.E_L - self.v) + (I_syn + I_ext) * self.norm_R_const) / self.tau_m
        v_next = self.v + dv
        # non-differentiable, hard threshold
        spiked = (v_next >= self.theta_s).float()
        not_spiked = (spiked - 1.) / -1.

        gating = (v_next / self.theta_s).clamp(0., 1.)
        dv_max = (self.theta_s - self.E_L)
        ds = (-self.s + gating * (dv / dv_max).clamp(0., 1.)) / self.tau_s
        self.s = self.s + ds

        v_reset = self.E_L + self.f_v * (self.v - self.E_L) - self.delta_V
        self.v = torch.add(spiked * v_reset, not_spiked * v_next)

        theta_s_next = (1 - self.b_s) * self.theta_s
        self.theta_s = spiked * (self.theta_s + self.delta_theta_s) + not_spiked * theta_s_next

        # self.I_additive = (1. - self.f_I) * self.I_additive + spiked * self.I_A
        self.I_additive = self.I_additive - self.f_I * self.I_additive + spiked * self.f_I

        # return self.v, self.s * self.tau_s
        # return self.s * self.tau_s  # use synaptic current as spike signal
        # return self.s * (self.tau_s + 1) / 2.  # return readout of synaptic current as spike signal

        # return self.v, self.spiked
        # return self.spiked

        # differentiable soft threshold
        soft_spiked = torch.sigmoid(torch.sub(v_next, self.theta_s))
        return soft_spiked  # return sigmoidal spiked
        # return gating