import sys

import numpy as np
import torch

import IO
import experiments
import model_util
import spike_metrics
from Models.GLIF import GLIF
from Models.GLIF_no_cell_types import GLIF_no_cell_types
from Models.LIF import LIF
from Models.LIF_no_cell_types import LIF_no_cell_types
from Models.microGIF import microGIF
from plot import plot_spike_trains_side_by_side, plot_neuron

load_fname = 'snn_model_target_GD_test'
model_class_lookup = { 'LIF': LIF, 'GLIF': GLIF, 'microGIF': microGIF,
                       'LIF_no_cell_types': LIF_no_cell_types, 'GLIF_no_cell_types': GLIF_no_cell_types }

GT_path = '/home/william/repos/snn_inference/Test/saved/'
GT_model_by_type = {'LIF': '12-09_11-49-59-999',
                    'GLIF': '12-09_11-12-47-541',
                    'mesoGIF': '12-09_14-56-20-319',
                    'microGIF': '12-09_14-56-17-312'}
# archive_name = 'data/'
# plot_data_path = experiments_path + 'plot_data/'
model_types = list(GT_model_by_type.keys())

for model_type_str in model_types:
    GT_euid = GT_model_by_type[model_type_str]
    tar_fname = 'snn_model_target_GD_test'
    model_name = model_type_str
    if model_type_str == 'mesoGIF':
        model_name = 'microGIF'
    load_data_target = torch.load(GT_path + model_name + '/' + GT_euid + '/' + tar_fname + IO.fname_ext)
    snn = load_data_target['model']
    model_class = snn.__class__
    euid = GT_model_by_type[model_type_str]
    cur_fname = 'target_GT_model_{}_N_{}'.format(model_type_str, snn.N)
    N = 4

    t = 120
    white_noise = torch.rand((t, snn.N))
    input_type = 'white_noise'
    inputs = white_noise

    print('- SNN test for class {} -'.format(snn.__class__.__name__))
    print('#inputs: {}'.format(inputs.sum()))

    if model_class is microGIF:
        s_lambdas, spikes, vs = model_util.feed_inputs_sequentially_return_args(snn, inputs)
    else:
        vs, spikes = model_util.feed_inputs_sequentially_return_tuple(snn, inputs)

    plot_neuron(vs.detach().numpy(), title='{} neuron plot ({:.2f} spikes)'.format(snn.__class__.__name__, spikes.sum()),
                uuid='export_{}'.format(snn.__class__.__name__),
                fname='export_sample_{}_{}.eps'.format(model_type_str, input_type))


sys.exit(0)
