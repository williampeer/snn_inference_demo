import numpy as np
import torch
from matplotlib import pyplot as plt

import IO
import experiments
import plot
import spike_metrics


def plot_convolution_with_spikes(convolved_spike_train, spike_train, uuid, exp_type='default', title='Neuron activity',
                ylabel='Membrane potential', fname='plot_neuron_test'):
    data = {'convolved_spike_train': convolved_spike_train, 'spike_train': spike_train, 'title': title, 'uuid': uuid,
            'exp_type': exp_type, 'ylabel': ylabel, 'fname': fname}
    IO.save_plot_data(data=data, uuid='test_uuid', plot_fn='plot_neuron')
    legend = []
    # for i in range(len(convolved_spike_train)):
    #     legend.append('N.{}'.format(i+1))
    fig = plt.figure()
    plt.plot(np.arange(convolved_spike_train.shape[0]), convolved_spike_train)

    time_indices = torch.reshape(torch.arange(spike_train.shape[0]), (spike_train.shape[0], 1))
    # ensure binary values:
    spike_train = torch.round(spike_train)
    neuron_spike_times = spike_train * time_indices.float()
    plt.plot(torch.reshape(neuron_spike_times.nonzero(), (1, -1)).numpy(), 1, '.k', markersize=6.0)

    # plt.legend(legend, loc='upper left', ncol=4)
    # plt.title(title)
    plt.xlabel('Time (ms)')
    plt.ylabel(ylabel)
    # plt.show()
    full_path = './figures/' + exp_type + '/' + uuid + '/'
    IO.makedir_if_not_exists(full_path)
    plt.savefig(fname=full_path + fname)
    plt.close()
    return fig


tau_vr = 4.
neur_spikes = torch.round(torch.abs(torch.rand((120, 1)) /1.6))
neur_vr_conv = spike_metrics.torch_van_rossum_convolution(neur_spikes, tau_vr)

plot_convolution_with_spikes(neur_vr_conv, neur_spikes, 'export', 'vr_conv',
                             title='Van Rossum convolution of spike train',
                             ylabel='Convolved spike train', fname='neur_vr_conv_sample.eps')
