import torch
import torch.nn as nn
from torch import FloatTensor as FT


class nonDifferentiableMicroGIF(nn.Module):
    free_parameters = ['w', 'E_L', 'tau_m', 'tau_s', 'tau_theta', 'J_theta']
    parameter_init_intervals = { 'E_L': [0., 5.], 'tau_m': [9., 12.], 'tau_s': [3., 6.], 'tau_theta': [950., 1050.],
                                 'J_theta': [0.9, 1.1] }
    param_lin_constraints = [[0., 1.], [-10., 30.], [2., 20.], [1.5, 20.], [800., 1500.], [0.5, 1.5]]

    def __init__(self, parameters, N=4, neuron_types=[1, -1]):
        super(nonDifferentiableMicroGIF, self).__init__()

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
                elif key == 'R_m':
                    R_m = FT(torch.ones((N,)) * parameters[key])
                elif key == 'c':
                    c = FT(torch.ones((N,)) * parameters[key])
                elif key == 'pop_sizes':
                    pop_sizes = FT(torch.ones((N,)) * parameters[key])

        __constants__ = ['N', 'norm_R_const', 'self_recurrence_mask', 'Theta_max']
        self.N = N

        if parameters.__contains__('preset_weights'):
            # print('DEBUG: Setting w to preset weights: {}'.format(parameters['preset_weights']))
            # print('Setting w to preset weights.')
            rand_ws = torch.abs(parameters['preset_weights'])
            assert rand_ws.shape[0] == N and rand_ws.shape[1] == N, "shape of weights matrix should be NxN"
        else:
            rand_ws = (0.5 - 0.25) + 2 * 0.25 * torch.rand((self.N, self.N))
        nt = torch.tensor(neuron_types).float()
        self.neuron_types = torch.transpose((nt * torch.ones((self.N, self.N))), 0, 1)
        self.w = nn.Parameter(FT(rand_ws), requires_grad=True)  # initialise with positive weights only
        self.self_recurrence_mask = torch.ones((self.N, self.N)) - torch.eye(self.N, self.N)
        # self.self_recurrence_mask = torch.ones((self.N, self.N))

        self.v = E_L * torch.ones((self.N,))
        self.spiked = torch.zeros((self.N,))
        # self.g = torch.zeros((self.N,))
        self.time_since_spike = torch.zeros((N,))

        self.E_L = nn.Parameter(FT(E_L), requires_grad=True)  # Rest potential
        self.tau_m = nn.Parameter(FT(tau_m), requires_grad=True)
        self.tau_s = nn.Parameter(FT(tau_s), requires_grad=True)
        self.tau_theta = nn.Parameter(FT(tau_theta), requires_grad=True)  # Adaptation time constant
        self.J_theta = nn.Parameter(FT(J_theta), requires_grad=True)  # Adaptation strength
        self.Delta_delay = 1.  # Transmission delay
        self.Delta_u = 5.  # Sensitivity
        self.theta_inf = 15.
        self.reset_potential = 0.
        self.theta_v = FT(self.theta_inf * torch.ones((N,)))
        self.R_m = FT(R_m)
        self.t_refractory = 2.
        self.c = c
        self.pop_sizes = FT(pop_sizes)

    def reset(self):
        for p in self.parameters():
            p.grad = None
        self.reset_hidden_state()

    def reset_hidden_state(self):
        self.v = self.v.clone().detach()
        self.spiked = self.spiked.clone().detach()

    def get_parameters(self):
        params_list = []

        params_list.append(self.w.data)
        params_list.append(self.E_L.data)
        params_list.append(self.tau_m.data)
        params_list.append(self.tau_s.data)

        return params_list

    def name(self):
        return self.__class__.__name__

    def forward(self, I_ext):
        #   adaptation_kernel Can be rewritten to:
        dtheta_v = (self.theta_inf - self.theta_v + self.J_theta * self.spiked) / self.tau_theta
        self.theta_v = self.theta_v + dtheta_v

        def Heaviside(x):
            return (x>0).float()

        epsilon = Heaviside(self.time_since_spike-self.Delta_delay)\
                  *torch.exp(-(self.time_since_spike-self.Delta_delay)/self.tau_s)/self.tau_s

        W_syn = 10. * self.self_recurrence_mask * self.w * self.neuron_types
        # Scale synaptic currents with pop sizes.
        I_syn = (self.tau_m * (epsilon * self.pop_sizes * self.spiked).matmul(W_syn)) / self.R_m
        dv = (self.E_L - self.v + self.R_m * (I_syn + I_ext)) / self.tau_m
        v_next = self.v + dv

        spikes_lambda = self.c * torch.exp((v_next - self.theta_v) / self.Delta_u)
        m = torch.distributions.bernoulli.Bernoulli(spikes_lambda)
        spiked = m.sample()
        self.spiked = spiked
        not_spiked = (spiked - 1.) / -1.

        self.time_since_spike = not_spiked * (self.time_since_spike + 1)
        self.v = not_spiked * v_next + spiked * self.reset_potential

        # return spiked
        # return m.log_prob(spiked)
        return self.v, spiked
