import torch
import torch.nn as nn
from torch import FloatTensor as FT

from Models.TORCH_CUSTOM import static_clamp_for, static_clamp_for_matrix


class GLIF(nn.Module):
    free_parameters = ['w', 'E_L', 'tau_m', 'G', 'f_v', 'f_I', 'delta_theta_s', 'b_s', 'a_v', 'b_v', 'theta_inf',
                       'delta_V', 'tau_s']
    parameter_init_intervals = {'E_L': [-62., -56.], 'tau_m': [3., 3.2], 'G': [0.75, 0.8],  'f_v': [0.25, 0.35],
                                'f_I': [0.35, 0.45], 'delta_theta_s': [14., 26.], 'b_s': [0.28, 0.32], 'a_v': [0.17, 0.2],
                                'b_v': [0.28, 0.32], 'theta_inf': [-10., -8.], 'delta_V': [9., 11.],
                                'tau_s': [3., 4.]}
    param_lin_constraints = [[0., 1.], [-70., -40.], [1.5, 10.], [0.01, 0.99], [0.01, 0.99], [0.01, 0.99], [6., 30.],
                             [0.01, 0.99], [0.01, 0.99], [0.01, 0.99], [-25., 0.], [1., 35.], [1., 12.]]

    def __init__(self, parameters, N=4, w_mean=0.15, w_var=0.1,
                 neuron_types=[1, 1, -1, -1]):
        super(GLIF, self).__init__()

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
                elif key == 'b_v':
                    b_v = FT(torch.ones((N,)) * parameters[key])
                elif key == 'a_v':
                    a_v = FT(torch.ones((N,)) * parameters[key])
                elif key == 'theta_inf':
                    theta_inf = FT(torch.ones((N,)) * parameters[key])
                elif key == 'w_mean':
                    w_mean = FT(torch.ones((N,)) * parameters[key])
                elif key == 'w_var':
                    w_var = FT(torch.ones((N,)) * parameters[key])

        __constants__ = ['N', 'norm_R_const', 'self_recurrence_mask', 'Theta_max']
        self.N = N

        self.Theta_max = (delta_theta_s / b_s - (a_v / b_v) * E_L + delta_V)
        # self.Theta_max = delta_theta_s/(1+b_s) + delta_V/(1+b_v)
        # self.norm_R_const = R_factor * self.Theta_max - E_L
        self.norm_R_const = self.Theta_max

        self.v = E_L * torch.ones((self.N,))
        # self.spiked = torch.zeros_like(self.v)  # spike prop. for next time-step
        self.s = torch.zeros_like(self.v)
        self.theta_s = delta_theta_s * torch.ones((self.N,))
        self.theta_v = delta_V * torch.ones((self.N,))
        self.I_additive = torch.zeros((self.N,))

        self.self_recurrence_mask = torch.ones((self.N, self.N)) - torch.eye(self.N, self.N)
        if parameters.__contains__('preset_weights'):
            # print('DEBUG: Setting w to preset weights: {}'.format(parameters['preset_weights']))
            # print('Setting w to preset weights.')
            rand_ws = torch.abs(parameters['preset_weights'])
            assert rand_ws.shape[0] == N and rand_ws.shape[1] == N, "shape of weights matrix should be NxN"
        else:
            rand_ws = (w_mean - w_var) + 2 * w_var * torch.rand((self.N, self.N))
            rand_ws = torch.abs(rand_ws)
        nt = torch.tensor(neuron_types).float()
        # self.neuron_types = torch.transpose((nt * torch.ones((self.N, self.N))), 0, 1)
        self.neuron_types = nt * torch.ones((self.N, self.N))
        self.w = nn.Parameter(FT(rand_ws), requires_grad=True)  # initialise with positive weights only

        self.E_L = nn.Parameter(FT(E_L).clamp(-80., -35.), requires_grad=True)
        self.tau_m = nn.Parameter(FT(tau_m).clamp(1.5, 8.), requires_grad=True)
        self.tau_s = nn.Parameter(FT(tau_s).clamp(1., 12.), requires_grad=True)
        self.G = nn.Parameter(FT(G).clamp(0.01, 0.99), requires_grad=True)
        self.f_v = nn.Parameter(FT(f_v).clamp(0.01, 0.99), requires_grad=True)
        self.f_I = nn.Parameter(FT(f_I).clamp(0.01, 0.99), requires_grad=True)
        self.delta_theta_s = nn.Parameter(FT(delta_theta_s).clamp(6., 30.), requires_grad=True)
        self.b_s = nn.Parameter(FT(b_s).clamp(0.01, 0.99), requires_grad=True)
        self.a_v = nn.Parameter(FT(a_v).clamp(0.01, 0.99), requires_grad=True)
        self.b_v = nn.Parameter(FT(b_v).clamp(0.01, 0.99), requires_grad=True)
        self.theta_inf = nn.Parameter(FT(theta_inf).clamp(-25., 0.), requires_grad=True)
        self.delta_V = nn.Parameter(FT(delta_V).clamp(1., 35.), requires_grad=True)

        self.register_backward_clamp_hooks()

    def reset(self):
        for p in self.parameters():
            p.grad = None
        self.reset_hidden_state()

    def reset_hidden_state(self):
        self.v = self.v.clone().detach()
        self.s = self.s.clone().detach()
        self.theta_s = self.theta_s.clone().detach()
        self.theta_v = self.theta_v.clone().detach()
        self.I_additive = self.I_additive.clone().detach()

    def register_backward_clamp_hooks(self):
        self.E_L.register_hook(lambda grad: static_clamp_for(grad, -70., -40., self.E_L))
        self.tau_m.register_hook(lambda grad: static_clamp_for(grad, 1.5, 8., self.tau_m))
        self.tau_s.register_hook(lambda grad: static_clamp_for(grad, 1., 10., self.tau_s))
        self.G.register_hook(lambda grad: static_clamp_for(grad, 0.01, 0.99, self.G))
        self.f_v.register_hook(lambda grad: static_clamp_for(grad, 0.01, 0.99, self.f_v))
        self.f_I.register_hook(lambda grad: static_clamp_for(grad, 0.01, 0.99, self.f_I))
        self.delta_theta_s.register_hook(lambda grad: static_clamp_for(grad, 6., 30., self.delta_theta_s))
        self.b_s.register_hook(lambda grad: static_clamp_for(grad, 0.01, 0.99, self.b_s))
        self.a_v.register_hook(lambda grad: static_clamp_for(grad, 0.01, 0.99, self.a_v))
        self.b_v.register_hook(lambda grad: static_clamp_for(grad, 0.01, 0.99, self.b_v))
        self.theta_inf.register_hook(lambda grad: static_clamp_for(grad, -25., 0., self.theta_inf))
        self.delta_V.register_hook(lambda grad: static_clamp_for(grad, 1., 35., self.delta_V))

        self.w.register_hook(lambda grad: static_clamp_for_matrix(grad, 0., 1., self.w))

    def get_parameters(self):
        params_dict = {}
        # parameter_names = ['w', 'E_L', 'tau_m', 'G', 'f_v', 'f_I', 'delta_theta_s', 'b_s', 'a_v', 'b_v', 'theta_inf', 'delta_V', 'tau_s']
        params_dict['w'] = self.w.data
        params_dict['E_L'] = self.E_L.data
        params_dict['tau_m'] = self.tau_m.data
        params_dict['tau_s'] = self.tau_s.data
        params_dict['G'] = self.G.data
        params_dict['f_v'] = self.f_v.data
        params_dict['f_I'] = self.f_I.data
        params_dict['delta_theta_s'] = self.delta_theta_s.data
        params_dict['b_s'] = self.b_s.data
        params_dict['a_v'] = self.a_v.data
        params_dict['b_v'] = self.b_v.data
        params_dict['theta_inf'] = self.theta_inf.data
        params_dict['delta_V'] = self.delta_V.data

        return params_dict

    def name(self):
        return self.__class__.__name__

    def forward(self, x_in):
        W_syn = self.self_recurrence_mask * self.w * self.neuron_types
        I_tot = ((self.I_additive + self.s) / 2).matmul(W_syn) + x_in

        dv = (self.G * (self.E_L - self.v) + I_tot * self.norm_R_const) / self.tau_m
        v_next = self.v + dv
        # non-differentiable, hard threshold
        spiked = (v_next >= self.theta_s + self.theta_v).int()
        not_spiked = (spiked - 1) / -1
        soft_spiked = torch.sigmoid(torch.sub(v_next, self.theta_s + self.theta_v))

        # gating = ((v_next-self.theta_inf) / (self.theta_s + self.theta_v)).clamp(0., 1.)  # sub-threshold currents above theta_inf
        # # gating = ((v_next) / (self.theta_s + self.theta_v)).clamp(0., 1.)  # sub-threshold currents above theta_inf
        # dv_max = (self.theta_s + self.theta_v - self.E_L)
        # # dv_max = self.Theta_max
        # ds = (-self.s + gating * (dv / dv_max).clamp(0., 1.0)) / self.tau_s
        # self.s = self.s + ds
        ds = -self.s/self.tau_s
        self.s = spiked + not_spiked * (self.s + ds)

        v_reset = self.E_L + self.f_v * (self.v - self.E_L) - self.delta_V
        self.v = torch.add(spiked * v_reset, not_spiked * v_next)

        self.theta_s = (1. - self.b_s) * self.theta_s + spiked * self.delta_theta_s  # always decay
        d_theta_v = self.a_v * (self.v - self.E_L) - self.b_v * (self.theta_v - self.theta_inf)
        self.theta_v = self.theta_v + not_spiked * d_theta_v

        # self.I_additive = (1. - self.f_I) * self.I_additive + spiked * self.I_A
        self.I_additive = self.I_additive - self.f_I * self.I_additive + spiked * self.f_I

        # differentiable soft threshold
        # readouts = self.W_out.matmul(soft_spiked)
        # return self.v, readouts
        return self.v, soft_spiked
        # return gating
