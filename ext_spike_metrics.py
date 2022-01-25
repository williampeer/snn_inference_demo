import pymuvr
import pyspike as spk
import numpy as np
from pyspike import SpikeTrain

from data_util import convert_to_sparse_vectors, get_spike_times_list


def get_pymuvr_dist_sparse(spikes, targets, cos=0.2, tau=20.):
    single_observation_dist = pymuvr.distance_matrix(trains1=[spikes], trains2=[targets], cos=cos, tau=tau)
    return single_observation_dist[0][0]


# greedy matching for now
def get_label_free_isi_dist_sparse(model_spikes, target_spikes, num_nodes, time):
    # TODO: Fix when nodes have no spikes (empty list)
    assert len(model_spikes) == num_nodes, "expecting column / list per node"
    edges = time

    spike_trains_model = []
    spike_trains_target = []
    for i in range(len(model_spikes)):
        spike_trains_model.append(SpikeTrain(model_spikes[i], edges))
        spike_trains_target.append(SpikeTrain(target_spikes[i], edges))

    indices_left = np.arange(0, num_nodes)
    min_distances = np.zeros((num_nodes,))
    for s_i in range(1, len(indices_left)):
        d_i_J = []
        for s_j in range(len(indices_left)):
            dist_i_j = spk.isi_profile(spike_trains_model[s_i], spike_trains_target[s_j])
            d_i_J.append(dist_i_j.avrg())
        d_i_J = np.array(d_i_J)
        min_i_J = d_i_J[0]; min_index = 0
        for ind in range(d_i_J.shape[0]):
            if d_i_J[ind] < min_i_J:
                min_i_J = d_i_J[ind]
                min_index = ind
        min_distances[s_i] = min_i_J
        indices_left = indices_left[indices_left != min_index]

    return np.mean(min_distances)


def get_label_free_spk_dist_sparse(metric, model_spikes, target_spikes, num_nodes, time):
    assert len(model_spikes) == num_nodes, "expecting column / list per node"
    edges = time

    spike_trains_model = []
    spike_trains_target = []
    for i in range(len(model_spikes)):
        spike_trains_model.append(SpikeTrain(model_spikes[i], edges))
        spike_trains_target.append(SpikeTrain(target_spikes[i], edges))

    indices_left = np.arange(0, num_nodes)
    min_distances = np.zeros((num_nodes,))
    for s_i in range(1, len(indices_left)):
        d_i_J = []
        for s_j in range(len(indices_left)):
            if metric == 'isi':
                dist_i_j = spk.isi_profile(spike_trains_model[s_i], spike_trains_target[s_j])
            elif metric == 'spike':
                dist_i_j = spk.spike_profile(spike_trains_model[s_i], spike_trains_target[s_j])
            elif metric == 'spike_sync':
                dist_i_j = spk.spike_sync_profile(spike_trains_model[s_i], spike_trains_target[s_j])
            else:
                raise NotImplementedError()
            d_i_J.append(dist_i_j.avrg())
        d_i_J = np.array(d_i_J)
        min_i_J = d_i_J[0]; min_index = 0
        for ind in range(d_i_J.shape[0]):
            if d_i_J[ind] < min_i_J:
                min_i_J = d_i_J[ind]
                min_index = ind
        min_distances[s_i] = min_i_J
        indices_left = indices_left[indices_left != min_index]

    return np.mean(min_distances)


def get_label_free_spike_dist():
    pass


def get_label_free_spike_sync_dist():
    pass


# ---------------------------------------
def get_pymuvr_dist(spike_train, targets, num_nodes, cos=0.2, tau=20.):
    spike_indices, spike_times = convert_to_sparse_vectors(spike_train)
    target_indices, target_times = convert_to_sparse_vectors(targets)

    t_min = np.min([spike_times[-1], target_times[-1]])
    _, spikes = get_spike_times_list(0, t_min, spike_times, spike_indices, num_nodes=num_nodes)
    _, targets = get_spike_times_list(0, t_min, target_times, target_indices, num_nodes=num_nodes)

    for cell_i in range(0, len(spikes)):
        spikes[cell_i] = spikes[cell_i].tolist()
        targets[cell_i] = targets[cell_i].tolist()

    single_observation_dist = pymuvr.distance_matrix(trains1=[spikes], trains2=[targets], cos=cos, tau=tau)
    return single_observation_dist[0][0]


# greedy matching for now
def get_label_free_isi_dist(model_spikes, target_spikes, num_nodes):
    assert model_spikes.shape[0] > model_spikes.shape[1], "each time step as a row expected, meaning column by node"
    edges = int(model_spikes.shape[0])

    spike_indices, spike_times = convert_to_sparse_vectors(model_spikes)
    target_indices, target_times = convert_to_sparse_vectors(target_spikes)

    t_min = np.min([spike_times[-1], target_times[-1]])
    _, spikes = get_spike_times_list(0, t_min, spike_times, spike_indices, num_nodes=num_nodes)
    _, targets = get_spike_times_list(0, t_min, target_times, target_indices, num_nodes=num_nodes)

    spike_trains_model = []
    spike_trains_target = []
    for i in range(len(spikes)):
        spike_trains_model.append(SpikeTrain(spikes[i], edges))
        spike_trains_target.append(SpikeTrain(targets[i], edges))

    num_nodes = model_spikes.shape[1]
    indices_left = np.arange(0, num_nodes)
    min_distances = np.zeros((model_spikes.shape[1],))
    for s_i in range(1, len(indices_left)):
        d_i_J = []
        for s_j in range(len(indices_left)):
            dist_i_j = spk.isi_profile(spike_trains_model[s_i], spike_trains_target[s_j])
            d_i_J.append(dist_i_j.avrg())
        d_i_J = np.array(d_i_J)
        min_i_J = d_i_J[0]; min_index = 0
        for ind in range(d_i_J.shape[0]):
            if d_i_J[ind] < min_i_J:
                min_i_J = d_i_J[ind]
                min_index = ind
        min_distances[s_i] = min_i_J
        indices_left = indices_left[indices_left != min_index]

    return np.mean(min_distances)
