import torch
import torch.nn as nn

from Models.Izhikevich import Izhikevich
from plot import plot_all_param_pairs_with_variance

model = Izhikevich(device='cpu', parameters={}, N=7, a=0.105)
recovered_parameters = {}
target_parameters = {}
for param_i, param in enumerate(list(model.parameters())):
    print('parameter #{}: {}'.format(param_i, param))
    recovered_parameters[param_i] = [param.clone().detach().numpy()]
    target_parameters[param_i] = [param.clone().detach().numpy()]
model.a = nn.Parameter(0.101 * torch.rand(model.a.shape), requires_grad=True)
model.b = nn.Parameter(0.24 * torch.rand(model.b.shape), requires_grad=True)
model.c = nn.Parameter(-66.3 * torch.rand(model.c.shape), requires_grad=True)
model.d = nn.Parameter(7.2 * torch.rand(model.d.shape), requires_grad=True)
for param_i, param in enumerate(list(model.parameters())):
    print('parameter #{}: {}'.format(param_i, param))
    recovered_parameters[param_i].append(param.clone().detach().numpy())
model.a = nn.Parameter(0.096 * torch.rand(model.a.shape), requires_grad=True)
model.b = nn.Parameter(0.27 * torch.rand(model.b.shape), requires_grad=True)
model.c = nn.Parameter(-57.3 * torch.rand(model.c.shape), requires_grad=True)
model.d = nn.Parameter(6.2 * torch.rand(model.d.shape), requires_grad=True)
for param_i, param in enumerate(list(model.parameters())):
    print('parameter #{}: {}'.format(param_i, param))
    recovered_parameters[param_i].append(param.clone().detach().numpy())
model.a = nn.Parameter(0.11 * torch.rand(model.a.shape), requires_grad=True)
model.b = nn.Parameter(0.223 * torch.rand(model.b.shape), requires_grad=True)
model.c = nn.Parameter(-61.3 * torch.rand(model.c.shape), requires_grad=True)
model.d = nn.Parameter(6.4 * torch.rand(model.d.shape), requires_grad=True)
for param_i, param in enumerate(list(model.parameters())):
    print('parameter #{}: {}'.format(param_i, param))
    recovered_parameters[param_i].append(param.clone().detach().numpy())

# plot_all_param_pairs_with_variance(recovered_parameters,
#                                    uuid='test_uuid',
#                                    exp_type='default',
#                                    target_params=target_parameters,
#                                    custom_title="Average inferred parameters across experiments [{}, {}]".format('test', 'None'),
#                                    logger=False)

plot_all_param_pairs_with_variance(recovered_parameters, target_params=target_parameters, param_names=model.parameter_names,
                                   exp_type='default', uuid='test_uuid', fname='test_param_pairs_new',
                                   custom_title='Test plot_all_param_pairs_with_variance_new', logger=False)
