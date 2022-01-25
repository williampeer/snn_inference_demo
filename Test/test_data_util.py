from data_util import load_data, get_spike_train_matrix, sample_state_input, transform_states

satisfactory_quality_node_indices, spike_times, spike_indices, states = load_data(4)

t_steps = 120000
_, sut_spikes = get_spike_train_matrix(index_last_step=0, advance_by_t_steps=t_steps,
                                       spike_times=spike_times, spike_indices=spike_indices,
                                       node_numbers=satisfactory_quality_node_indices)

assert sut_spikes.sum() > 0
print('#spikes: {}'.format(sut_spikes.sum()))

bin_size = 4000; n_rows = int(t_steps/bin_size)
state_labels_1d = transform_states(states[:n_rows], bin_size=bin_size, target_bin_size=1)
assert state_labels_1d.shape[0] == n_rows * bin_size, \
    "state_labels_1d.shape: {}, n_rows * bin_size: {}".format(state_labels_1d.shape, n_rows * bin_size)

sut_sample_input = sample_state_input(state_labels_1d=state_labels_1d, n_dim=sut_spikes.shape[1])
assert sut_sample_input.shape[0] == state_labels_1d.shape[0] and sut_sample_input.shape[1] == sut_spikes.shape[1], \
    "sut_sample_input.shape: {}, sut_spikes.shape: {}".format(sut_sample_input.shape, sut_spikes.shape)

sut_sum = sut_sample_input.sum()


def test_rate_in_data_set():
    rates = []
    for node_i in range(sut_spikes.shape[1]):
        rate_i = sut_spikes[:, node_i].sum() / float(t_steps)
        rates.append(rate_i)

    print('rates: {}'.format(rates))


test_rate_in_data_set()
