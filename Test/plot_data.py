import plot
from data_util import load_data, get_spike_train_matrix


def fetch_and_plot_data(exp_num, t_steps, index_previous_step):
    satisfactory_quality_node_indices, spike_times, spike_indices, states = load_data(exp_num)

    next_step, sut_spikes = get_spike_train_matrix(index_last_step=index_previous_step, advance_by_t_steps=t_steps,
                                                   spike_times=spike_times, spike_indices=spike_indices,
                                                   node_numbers=satisfactory_quality_node_indices)

    assert sut_spikes.sum() > 0
    print('#spikes: {}'.format(sut_spikes.sum()))
    avg_freq = sut_spikes.sum() / float(sut_spikes.shape[0] * sut_spikes.shape[1])
    print('average freq.: {}'.format(avg_freq))

    plot.plot_spike_train(sut_spikes, 'Test data spike train', 'test_plot_data',
                          fname='test_plot_data_exp_{}_index_prev_{}'.format(exp_num, index_previous_step))
    return next_step


next_step = fetch_and_plot_data(6, 500, 0)
next_step = fetch_and_plot_data(6, 500, next_step)
next_step = fetch_and_plot_data(6, 500, next_step)
next_step = fetch_and_plot_data(6, 500, next_step)
