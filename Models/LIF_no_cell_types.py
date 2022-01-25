import torch
import torch.nn as nn
from torch import FloatTensor as FT
from torch import tensor as T

from Models.TORCH_CUSTOM import static_clamp_for, static_clamp_for_matrix


class LIF_no_cell_types(nn.Module):
    free_parameters = ['w', 'E_L', 'tau_m', 'tau_s']
    parameter_init_intervals = {'E_L': [-58., -52.], 'tau_m': [2.5, 3.], 'tau_s': [3., 3.5] }

    def __init__(self, parameters, N=4, w_mean=0.05, w_var=0.15):
        super(LIF_no_cell_types, self).__init__()
        # self.device = device

        if parameters:
            for key in parameters.keys():
                if key == 'tau_m':
                    tau_m = FT(torch.ones((N,)) * parameters[key])
                elif key == 'E_L':
                    E_L = FT(torch.ones((N,)) * parameters[key])
                elif key == 'tau_s':
                    tau_s = FT(torch.ones((N,)) * parameters[key])
                elif key == 'w_mean':
                    w_mean = float(parameters[key])
                elif key == 'w_var':
                    w_var = float(parameters[key])

        __constants__ = ['N', 'norm_R_const', 'self_recurrence_mask', 'V_thresh']
        self.N = N
        self.V_thresh = 30.
        self.norm_R_const = 1.1 * (self.V_thresh - E_L)

        self.v = torch.zeros((self.N,))
        self.s = torch.zeros_like(self.v)  # syn. conductance

        self.self_recurrence_mask = torch.ones((self.N, self.N)) - torch.eye(self.N, self.N)
        if parameters.__contains__('preset_weights'):
            # print('Setting w to preset weights.')
            rand_ws = parameters['preset_weights']
            assert rand_ws.shape[0] == N and rand_ws.shape[1] == N, "shape of weights matrix should be NxN"
        else:
            rand_ws = (w_mean - w_var) + 2 * w_var * torch.rand((self.N, self.N))
        self.self_recurrence_mask = torch.ones((self.N, self.N)) - torch.eye(self.N, self.N)
        self.w = nn.Parameter(FT(rand_ws), requires_grad=True)

        self.E_L = nn.Parameter(FT(E_L).clamp(-80., -35.), requires_grad=True)
        self.tau_m = nn.Parameter(FT(tau_m).clamp(1.5, 8.), requires_grad=True)
        self.tau_s = nn.Parameter(FT(tau_s).clamp(1., 12,), requires_grad=True)

        self.register_backward_clamp_hooks()

    def register_backward_clamp_hooks(self):
        self.E_L.register_hook(lambda grad: static_clamp_for(grad, -80., -35., self.E_L))
        self.tau_m.register_hook(lambda grad: static_clamp_for(grad, 1.5, 8., self.tau_m))
        self.tau_s.register_hook(lambda grad: static_clamp_for(grad, 1., 12., self.tau_s))

        self.w.register_hook(lambda grad: static_clamp_for_matrix(grad, -1., 1., self.w))

    def get_parameters(self):
        params_dict = {}

        params_dict['w'] = self.w.data
        params_dict['E_L'] = self.E_L.data
        params_dict['tau_m'] = self.tau_m.data
        params_dict['tau_s'] = self.tau_s.data

        return params_dict

    def reset(self):
        for p in self.parameters():
            p.grad = None
        self.reset_hidden_state()

        self.v = self.E_L.clone().detach() * torch.ones((self.N,))
        self.s = torch.zeros_like(self.v)  # syn. conductance

    def reset_hidden_state(self):
        self.v = self.v.clone().detach()
        self.s = self.s.clone().detach()

    def name(self):
        return self.__class__.__name__

    def forward(self, I_ext):
        W_syn = self.self_recurrence_mask * self.w
        I_syn = W_syn.matmul(self.s)
        I_tot = I_syn + I_ext + 0.15

        dv = torch.div(torch.add(torch.sub(self.E_L, self.v), self.norm_R_const*I_tot), self.tau_m)
        v_next = torch.add(self.v, dv)

        spiked = torch.where(v_next >= self.V_thresh, T(1.), T(0.))
        not_spiked = torch.div(torch.sub(spiked, 1.), -1.)
        # differentiable soft threshold
        soft_spiked = torch.sigmoid(torch.sub(v_next, self.V_thresh))

        self.v = torch.add(torch.mul(spiked, self.E_L), torch.mul(not_spiked, v_next))

        ds = torch.sub(self.s, torch.div(self.s, self.tau_s))
        self.s = torch.add(spiked, torch.mul(not_spiked, ds))
        return self.v, soft_spiked
