import Log
from Models.GLIF import GLIF
from plot import plot_all_param_pairs_with_variance

param_sim = {}
model = GLIF(device='cpu', parameters={})
param_sim[0] = [model.w.clone().detach()]

model = GLIF(device='cpu', parameters={})
param_sim[0].append(model.w.clone().detach())

model = GLIF(device='cpu', parameters={})
param_sim[0].append(model.w.clone().detach())

model = GLIF(device='cpu', parameters={})
param_sim[0].append(model.w.clone().detach())

weights_dict = {}
for exp_i in range(len(param_sim[0])):
    for w_i in range(len(param_sim[0][exp_i])):
    # for w_i in range(2):
        if exp_i == 0:
            weights_dict[w_i] = [param_sim[0][exp_i][w_i].clone().detach().numpy()]
        else:
            weights_dict[w_i].append(param_sim[0][exp_i][w_i].clone().detach().numpy())

w_names = []
for i in range(1,11):
    w_names.append('w_\{{}\}'.format(i))
plot_all_param_pairs_with_variance(param_means=weights_dict, target_params=False, param_names=w_names, exp_type='default',
                                   uuid='test_plot_weights', fname='test_weights_plot',
                                   custom_title='Test plot weights as KDEs', logger=Log.Logger('test'))
