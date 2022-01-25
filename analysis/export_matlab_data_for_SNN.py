import sys

import numpy as np
import torch

import IO
from Models.LIF import LIF
from data_util import prefix, target_data_path
from spike_train_matlab_export import simulate_and_save_model_spike_train


def main(_):
    man_seed = 3
    torch.manual_seed(man_seed)
    np.random.seed(man_seed)

    num_neurons = 4
    duration = 2 * 60 * 1000

    tar_timestamp_LIF = '12-09_11-49-59-999'  # LIF
    fname = 'snn_model_target_GD_test'
    model_class = LIF
    load_data = torch.load('../Test/' + IO.PATH + model_class.__name__ + '/' + tar_timestamp_LIF + '/' + fname + IO.fname_ext)
    snn = load_data['model']
    saved_target_losses = load_data['loss']

    # cur_fname = 'nuovo_spikes_{}_N_{}_seed_{}_duration_{}'.format(snn.name(), num_neurons, f_i, duration)
    cur_fname = 'GT_{}_N_{}_seed_{}_duration_{}'.format(snn.name(), num_neurons, man_seed, duration)
    save_file_name = prefix + target_data_path + cur_fname + '.mat'
    # if not os.path.exists(save_file_name):
    simulate_and_save_model_spike_train(model=snn, t=duration, exp_num='GT', model_name=model_class.__name__, fname=cur_fname)
    # else:
    #     print('file exists. skipping..')


if __name__ == "__main__":
    main(sys.argv[1:])
    sys.exit()
