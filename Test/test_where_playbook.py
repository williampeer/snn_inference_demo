import torch

grad = torch.rand((12,))
l = -80.; m = -35.
p = -35. - 35. * torch.rand((12,))

sut = torch.where((grad < -1e-08) + (1e-08 < grad), p, grad.clone())
print(sut)
