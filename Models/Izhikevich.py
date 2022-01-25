import torch
import torch.nn as nn
from torch import FloatTensor as FT
from torch import tensor as T

from Models.TORCH_CUSTOM import static_clamp_for, static_clamp_for_matrix


class Izhikevich(nn.Module):
    free_parameters = ['w', 'a', 'b', 'c', 'd', 'tau_s']
    parameter_init_intervals = {'a': [0.02, 0.05], 'b': [0.25, 0.25], 'c': [-62., -58.], 'd': [4., 8.], 'tau_s': [2., 3.5]}

    def __init__(self, parameters, N=4, w_mean=0.6, w_var=0.15, #):
                 neuron_types=T([1., 1., -1., -1.])):
        super(Izhikevich, self).__init__()
        # self.device = device

        if parameters:
            for key in parameters.keys():
                if key == 'tau_s':
                    tau_s = FT(torch.ones((N,)) * parameters[key])

                elif key == 'N':
                    N = int(parameters[key])
                elif key == 'w_mean':
                    w_mean = float(parameters[key])
                elif key == 'w_var':
                    w_var = float(parameters[key])

                elif key == 'a':
                    a = FT(torch.ones((N,)) * parameters[key])
                elif key == 'b':
                    b = FT(torch.ones((N,)) * parameters[key])
                elif key == 'c':
                    c = FT(torch.ones((N,)) * parameters[key])
                elif key == 'd':
                    d = FT(torch.ones((N,)) * parameters[key])

        __constants__ = ['spike_threshold', 'N']
        self.spike_threshold = T(30.)
        self.N = N

        self.v = c * torch.ones((self.N,))
        self.u = d * torch.ones((self.N,))
        # self.spiked = torch.zeros((self.N,))
        self.s = torch.zeros((self.N,))

        self.self_recurrence_mask = torch.ones((self.N, self.N)) - torch.eye(self.N, self.N)
        if parameters.__contains__('preset_weights'):
            # print('DEBUG: Setting w to preset weights: {}'.format(parameters['preset_weights']))
            # print('Setting w to preset weights.')
            rand_ws = parameters['preset_weights']
            assert rand_ws.shape[0] == N and rand_ws.shape[1] == N, "shape of weights matrix should be NxN"
        else:
            rand_ws = (w_mean - w_var) + 2 * w_var * torch.abs(torch.rand((self.N, self.N)))
        # self.neuron_types = neuron_types
        nt = T(neuron_types).float()
        self.neuron_types = nt * torch.ones((self.N, self.N))

        self.w = nn.Parameter(FT(rand_ws), requires_grad=True)  # initialise with positive weights only

        self.a = nn.Parameter(FT(a), requires_grad=True)
        self.b = nn.Parameter(FT(b), requires_grad=True)
        self.c = nn.Parameter(FT(c), requires_grad=True)
        self.d = nn.Parameter(FT(d), requires_grad=True)
        self.tau_s = nn.Parameter(FT(tau_s), requires_grad=True)

        # self.parameter_names = ['w', 'a', 'b', 'c', 'd', '\\tau_g']
        # self.to(self.device)
        self.register_backward_clamp_hooks()

    def register_backward_clamp_hooks(self):
        self.a.register_hook(lambda grad: static_clamp_for(grad, 0.01, 0.2, self.a))
        self.b.register_hook(lambda grad: static_clamp_for(grad, 0.15, 0.35, self.b))
        self.c.register_hook(lambda grad: static_clamp_for(grad, -80., -40., self.c))
        self.d.register_hook(lambda grad: static_clamp_for(grad, 1., 8., self.d))
        self.tau_s.register_hook(lambda grad: static_clamp_for(grad, 1.15, 3.5, self.tau_s))

        self.w.register_hook(lambda grad: static_clamp_for_matrix(grad, 0., 2., self.w))

    def reset(self):
        for p in self.parameters():
            p.grad = None
        self.reset_hidden_state()

        self.v = self.c.clone().detach() * torch.ones((self.N,))
        self.v = self.d.clone().detach() * torch.ones((self.N,))
        self.spiked = torch.zeros_like(self.v)  # spike prop. for next time-step
        self.s = torch.zeros_like(self.v)  # syn. conductance

    def reset_hidden_state(self):
        # self.spiked = self.spiked.clone().detach()
        self.v = self.v.clone().detach()
        self.u = self.u.clone().detach()
        self.s = self.s.clone().detach()

    def get_parameters(self):
        params = {}
        params['a'] = self.a.data
        params['b'] = self.b.data
        params['c'] = self.c.data
        params['d'] = self.d.data
        params['tau_s'] = self.tau_s.data
        return params

    def forward(self, x_in):
        # I = torch.add(self.w.matmul(self.s), x_in)
        W_syn = self.self_recurrence_mask * self.w * self.neuron_types
        I_syn = W_syn.matmul(self.s)
        I_ext = x_in
        I = I_syn + I_ext

        dv = (T(0.04) * torch.pow(self.v, 2) + T(5.) * self.v + T(140.) - self.u + I + 0.1)
        v_next = self.v + dv

        du = self.a * (self.b * self.v - self.u)
        ds = -self.s / self.tau_s

        # gating = v_next.clamp(0., 1.)
        # ds = (gating * dv.clamp(0., 1.) - self.s) / self.tau_s
        # self.s = self.s + ds

        # dg = - torch.div(self.s, self.tau_g)
        # du = torch.mul(torch.abs(self.a), torch.sub(torch.mul(torch.abs(self.b), self.v), self.u))

        # spiked = (v_next >= self.spike_threshold).float()
        # not_spiked = (spiked - 1.) / -1.  # not op.
        spiked = torch.where(v_next >= self.spike_threshold, T(1.), T(0.))
        not_spiked = torch.div(torch.sub(spiked, 1.), -1.)
        soft_spiked = torch.sigmoid(torch.sub(v_next, self.spike_threshold))

        self.v = not_spiked * v_next + spiked * self.c
        self.u = not_spiked * (self.u + du) + spiked * self.d
        self.s = not_spiked * (self.s + ds) + spiked

        # return self.v, self.s
        return self.v, soft_spiked
        # return self.spiked
