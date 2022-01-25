import torch
import numpy as np


def sut_OU_process_input(t=10000):
    mu = torch.tensor([1., 1., 0, 1.])
    tau = torch.tensor([2000., 2000., 1., 2000.])
    q = torch.tensor([0.5, 0.5, 0, 0.5])
    dW = torch.randn((t, 4))
    # dW = torch.zeros((t, 4))

    I_0 = torch.tensor([0.5, 0.5, 0, 0.5])
    I_interval = I_0.clone().detach()
    I = I_0.clone().detach()
    for t_i in range(t-1):
        dI = (I - mu)/tau + torch.sqrt(2/tau) * q * dW[t_i, :]
        I = I + dI
        I_interval = torch.vstack((I_interval, dI))

    assert I_interval.shape[0] == t and I_interval.shape[1] == 4, "I_interval should be {}x{}. Was: {}".format(t, 4, I_interval.shape)
    # return torch.zeros((t, 4))
    return I_interval


tar_seed = 23
torch.manual_seed(tar_seed)
np.random.seed(tar_seed)

sut = sut_OU_process_input(t=200)
print('sut.sum(): {}'.format(sut.sum()))
assert sut.sum() < 1e06, "sut is over styr: {}".format(sut.sum())
