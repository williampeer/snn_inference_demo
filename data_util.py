import scipy.io as sio
import numpy as np
import torch

import IO

prefix = '/home/user/'  # Ubuntu
# prefix = '/Users/user/'  # OS X
# prefix = '/home/user/'  # server
target_data_path = 'data/target_data/'

matlab_export = 'matlab_export/'


def load_sparse_data(full_path):
    exp_data = sio.loadmat(full_path)['DATA']

    spike_indices = exp_data['clu'][0][0]  # index of the spiking neurons
    spike_times = exp_data['res'][0][0]  # spike times

    node_indices = np.unique(spike_indices)

    return node_indices, spike_times, spike_indices


def convert_to_sparse_vectors(spiketrain, t_offset=0.):
    assert spiketrain.shape[0] > spiketrain.shape[1], "assuming bins x nodes (rows as timesteps). spiketrain.shape: {}".format(spiketrain.shape)

    spike_indices = np.array([], dtype='int8')
    spike_times = np.array([], dtype='float32')
    for ms_i in range(spiketrain.shape[0]):
        for node_i in range(spiketrain.shape[1]):
            if spiketrain[ms_i][node_i] != 0:
                assert spiketrain[ms_i][node_i] <= 2, \
                    "element out of range. row: {}, col: {}, value:{}".format(ms_i, node_i, spiketrain[ms_i][node_i])
                spike_times = np.append(spike_times, np.float32(float(ms_i) +t_offset))
                spike_indices = np.append(spike_indices, np.int8(node_i))

    return spike_indices, spike_times


def save_spiketrain_in_sparse_matlab_format(fname, spike_indices, spike_times):
    exp_data = {}
    exp_data['clu'] = np.reshape(spike_indices, (-1, 1))
    exp_data['res'] = np.reshape(spike_times, (-1, 1))
    mat_data = {'DATA': exp_data}

    IO.makedir_if_not_exists(prefix + target_data_path + matlab_export)
    sio.savemat(file_name=prefix + target_data_path + matlab_export + fname, mdict=mat_data)


def get_spike_train_matrix(index_last_step, advance_by_t_steps, spike_times, spike_indices, node_numbers):
    spikes = torch.zeros((advance_by_t_steps, node_numbers.shape[0]))

    prev_spike_time = spike_times[index_last_step]

    next_step = index_last_step+1
    next_spike_time = spike_times[next_step]
    while next_spike_time < prev_spike_time + advance_by_t_steps:
        spike_arr_index = np.where(node_numbers == spike_indices[next_step])[0]
        time_index = int(np.floor(next_spike_time[0] - prev_spike_time[0]))
        spikes[time_index][spike_arr_index] = 1.0
        next_step = next_step+1
        next_spike_time = spike_times[next_step]

    return next_step, spikes


def transform_states(states, bin_size, target_bin_size):
    res_states = torch.zeros((states.shape[0] * int(bin_size/target_bin_size), states.shape[1]))
    for states_i in range(states.shape[0]):
        for expand_i in range(int(bin_size/target_bin_size)):
            res_states[states_i+expand_i] = torch.from_numpy(states[states_i])

    return res_states


def transform_to_population_spiking(spikes, kernel_indices):
    convolved_spikes = torch.zeros((spikes.shape[0], len(kernel_indices)))
    for t_i in range(spikes.shape[0]):
        for pop_i in range(len(kernel_indices)):
            for idx in range(len(kernel_indices[pop_i])):
                convolved_spikes[t_i, pop_i] += spikes[t_i, kernel_indices[pop_i][idx]]

    return torch.min(convolved_spikes, torch.tensor(1.0))


def load_sparse_data_matlab_format(fname):
    exp_data = sio.loadmat(prefix + 'repos/snn_inference/data/' + fname)['DATA']

    # Custom Matlab-compatible format
    spike_indices = exp_data['clu'][0][0]  # index of the spiking neurons
    spike_times = exp_data['res'][0][0]  # spike times
    qual = exp_data['qual'][0][0]  # neuronal decoding quality
    states = exp_data['score'][0][0]  # state

    satisfactory_quality_node_indices = np.unique(spike_indices)

    return satisfactory_quality_node_indices, spike_times, spike_indices, qual, states


def convert_sparse_spike_train_to_matrix(spike_times, node_indices, unique_node_indices):
    res = {}
    for j in range(len(unique_node_indices)):
        res[int(unique_node_indices[j])] = np.array([])
    for i in range(len(spike_times)):
        res[int(node_indices[i])] = np.concatenate((res[int(node_indices[i])], np.array(spike_times[i])))
    return res


def get_spike_times_list(index_last_step, advance_by_t_steps, spike_times, spike_indices, num_nodes):
    res = []
    for _ in range(num_nodes):
        res.append([])

    if index_last_step == 0:
        prev_spike_time = 0
    else:
        prev_spike_time = spike_times[index_last_step]

    next_step = index_last_step+1
    while next_step < len(spike_times) and spike_times[next_step] < prev_spike_time + advance_by_t_steps:
        cur_node_index = int(spike_indices[next_step])
        res[cur_node_index] = np.concatenate((res[cur_node_index], [spike_times[next_step]]))
        next_step = next_step+1

    return next_step, res


def scale_spike_times(spike_times_list, div=1000.):
    for n_i in range(len(spike_times_list)):
        spike_times_list[n_i] = spike_times_list[n_i] / div
    return  spike_times_list
