import pymuvr
import numpy as np

from data_util import convert_to_sparse_vectors, get_spike_times_list
from experiments import sine_modulated_white_noise_input
from ext_spike_metrics import get_pymuvr_dist

t = 4000
model_spike_train = sine_modulated_white_noise_input(rate=10., t=t, N=4)
target_spike_train = sine_modulated_white_noise_input(rate=10., t=t, N=4)
spike_indices, spike_times = convert_to_sparse_vectors(model_spike_train)
target_indices, target_times = convert_to_sparse_vectors(target_spike_train)

t_min = np.min([spike_times[-1], target_times[-1]])
_, spikes = get_spike_times_list(0, t_min, spike_times, spike_indices, num_nodes=np.unique(spike_indices))
_, targets = get_spike_times_list(0, t_min, target_times, target_indices, num_nodes=np.unique(target_indices))

for cell_i in range(0, len(spikes)):
    spikes[cell_i] = spikes[cell_i].tolist()
    targets[cell_i] = targets[cell_i].tolist()

cos = 0.2; tau = 20.
sut = pymuvr.distance_matrix(trains1=[spikes], trains2=[targets], cos=cos, tau=tau)
# print(sut)

wrapper_dist = get_pymuvr_dist(model_spike_train, target_spike_train, cos=cos, tau=tau)
assert sut[0][0] == wrapper_dist, "test code should arrive at the same spike dist. value. test dist: {}, wrapper dist: {}".format(sut[0][0], wrapper_dist)
