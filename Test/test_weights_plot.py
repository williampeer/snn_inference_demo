import torch

from TargetModels.TargetModels import *
from plot import heatmap_spike_train_correlations

m1 = glif1()
weights = m1.w.clone().detach().numpy()

heatmap_spike_train_correlations(weights, ['neuron $i$', 'neuron $j$'], exp_type='weights', uuid='test_plot',
                                 fname='test_plot_weights_as_heatmap', bin_size=None,
                                 custom_title='Test plot neuron weights as heatmap', custom_label='$w_{i,j}$')
