import torch
import torch.nn as nn
from torch import FloatTensor as FT

from Models.GLIF import GLIF
from Models.TORCH_CUSTOM import static_clamp_for, static_clamp_for_matrix


class GLIF_soft_no_grad(nn.Module):
    parameter_names = ['w', 'E_L', 'tau_m', 'G', 'f_v', 'f_I', 'delta_theta_s', 'b_s', 'a_v', 'b_v', 'theta_inf', 'delta_V', 'tau_g']
    parameter_init_intervals = {'E_L': [-64., -58.], 'tau_m': [2.7, 2.8], 'G': [0.7, 0.8],  'f_v': [0.25, 0.35], 'f_I': [0.35, 0.45],
                                'delta_theta_s': [10., 20.], 'b_s': [0.25, 0.35], 'a_v': [0.15, 0.2], 'b_v': [0.25, 0.35],
                                'theta_inf': [-10., -8.], 'delta_V': [8., 14.], 'tau_g': [3., 4.]}
    param_lin_constraints = [[0., 1.], [-80., -35.], [1.2, 8.], [0.01, 0.99], [0.01, 0.99], [0.01, 0.99], [6., 30.],
                             [0.01, 0.95], [0.01, 0.95], [0.01, 0.95], [-25., 0.], [1., 35.], [1.5, 12.]]

    def __init__(self, parameters, N=12, w_mean=0.2, w_var=0.15, neuron_types=torch.tensor([1, -1])):
        super(GLIF_soft_no_grad, self).__init__()

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
        self.g = torch.zeros_like(self.v)  # syn. conductance
        self.spiked = torch.zeros_like(self.v)  # spike prop. for next time-step
        self.theta_s = delta_theta_s * torch.ones((self.N,))
        self.theta_v = torch.ones((self.N,))
        self.I_additive = torch.zeros((self.N,))

        if parameters.__contains__('preset_weights'):
            # print('DEBUG: Setting w to preset weights: {}'.format(parameters['preset_weights']))
            # print('Setting w to preset weights.')
            rand_ws = torch.abs(parameters['preset_weights'])
            assert rand_ws.shape[0] == N and rand_ws.shape[1] == N, "shape of weights matrix should be NxN"
        else:
            rand_ws = (w_mean - w_var) + 2 * w_var * torch.rand((self.N, self.N))
        nt = torch.tensor(neuron_types).float()
        self.neuron_types = torch.transpose((nt * torch.ones((self.N, self.N))), 0, 1)
        self.w = nn.Parameter(FT(rand_ws), requires_grad=True)  # initialise with positive weights only
        self.self_recurrence_mask = torch.ones((self.N, self.N)) - torch.eye(self.N, self.N)

        self.E_L = FT(E_L).clamp(-80., -35.)
        self.tau_m = FT(tau_m).clamp(1.5, 8.)
        self.tau_g = FT(tau_g).clamp(1., 12.)
        self.G = FT(G).clamp(0.01, 0.99)
        self.f_v = FT(f_v).clamp(0.01, 0.99)
        self.f_I = FT(f_I).clamp(0.01, 0.99)
        self.delta_theta_s = FT(delta_theta_s).clamp(6., 30.)
        self.b_s = FT(b_s).clamp(0.01, 0.99)
        self.a_v = FT(a_v).clamp(0.01, 0.99)
        self.b_v = FT(b_v).clamp(0.01, 0.99)
        self.theta_inf = FT(theta_inf).clamp(-25., 0.)
        self.delta_V = FT(delta_V).clamp(1., 35.)

        self.register_backward_clamp_hooks()

    def get_parameters(self):
        params_dict = {}

        params_dict['w'] = self.w.data
        params_dict['E_L'] = self.E_L.data
        params_dict['tau_m'] = self.tau_m.data
        params_dict['G'] = self.G.data
        params_dict['f_v'] = self.f_v.data
        params_dict['f_I'] = self.f_I.data
        params_dict['delta_theta_s'] = self.delta_theta_s.data
        params_dict['b_s'] = self.b_s.data
        params_dict['a_v'] = self.a_v.data
        params_dict['b_v'] = self.b_v.data
        params_dict['theta_inf'] = self.theta_inf.data
        params_dict['delta_V'] = self.delta_V.data
        params_dict['tau_g'] = self.tau_g.data

        return params_dict

    def name(self):
        return GLIF.__name__

    def reset(self):
        for p in self.parameters():
            p.grad = None
        self.reset_hidden_state()

        self.v = self.E_L.clone().detach() * torch.ones((self.N,))
        self.spiked = torch.zeros_like(self.v)  # spike prop. for next time-step
        self.g = torch.zeros_like(self.v)  # syn. conductance

    def reset_hidden_state(self):
        self.v = self.v.clone().detach()
        self.spiked = self.spiked.clone().detach()
        self.theta_s = self.theta_s.clone().detach()
        self.theta_v = self.theta_v.clone().detach()
        self.I_additive = self.I_additive.clone().detach()

    def register_backward_clamp_hooks(self):
        self.w.register_hook(lambda grad: static_clamp_for_matrix(grad, 0., 1., self.w))


    def forward(self, I_ext):
        # assuming input weights to be Eye(N,N)
        W_syn = self.w * self.neuron_types

        I_syn = ((self.I_additive + self.g) / 2).matmul(self.self_recurrence_mask * W_syn)
        dv = (self.G * (self.E_L - self.v) + (I_syn + I_ext) * self.norm_R_const) / self.tau_m

        v_next = torch.add(self.v, dv)

        # differentiable
        self.spiked = torch.sigmoid(torch.sub(v_next, (self.theta_s + self.theta_v)))
        # Non-differentiable
        spiked = (v_next >= (self.theta_s + self.theta_v)).float()
        not_spiked = (spiked - 1.) / -1.

        v_reset = self.E_L + self.f_v * (self.v - self.E_L) - self.delta_V
        self.v = spiked * v_reset + not_spiked * v_next  # spike reset

        self.theta_s = (1. - self.b_s) * self.theta_s + spiked * self.delta_theta_s  # always decay
        d_theta_v = self.a_v * (self.v - self.E_L) - self.b_v * (self.theta_v - self.theta_inf)
        self.theta_v = self.theta_v + not_spiked * d_theta_v

        self.I_additive = self.I_additive - self.f_I * self.I_additive + spiked * self.f_I

        dg = -torch.div(self.g, self.tau_g)  # -g/tau_g
        self.g = torch.add(spiked * torch.ones_like(self.g), not_spiked * torch.add(self.g, dg))

        return self.spiked
        # return self.v, self.spiked