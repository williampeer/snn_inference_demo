import os
import sys

import numpy as np
import torch

from Models.GLIF import GLIF
from Models.LIF import LIF
from Models.LIF_ASC import LIF_ASC
from Models.LIF_R import LIF_R
from Models.LIF_R_ASC import LIF_R_ASC
from Models.no_grad.LIF_R_no_grad import LIF_R_no_grad
from data_util import prefix, target_data_path
from experiments import draw_from_uniform
from spike_train_matlab_export import simulate_and_save_model_spike_train


def main(argv):
    print('Argument List:', str(argv))

    # for model_class in [LIF_R, LIF_R_ASC, GLIF]:
    for model_class in [GLIF]:
        N = 10
        N_exp = 4

        # for exp_i in range(4):
        for exp_i in range(N_exp):
            start_seed = 64
            non_overlapping_offset = start_seed + N_exp + 1
            torch.manual_seed(non_overlapping_offset + exp_i)
            # torch.manual_seed(non_overlapping_offset)
            np.random.seed(non_overlapping_offset + exp_i)
            # np.random.seed(non_overlapping_offset)

            init_params_model = draw_from_uniform(model_class.parameter_init_intervals, N)
            programmatic_neuron_types = torch.ones((N,))
            for n_i in range(int(2 * N / 3), N):
                programmatic_neuron_types[n_i] = -1
            neuron_types = programmatic_neuron_types
            snn = model_class(N=N, parameters=init_params_model, neuron_types=neuron_types)
            model_name = snn.name()

            # cur_fname = 'initial_model_spikes_{}_exp_num_{}_seed_{}_60s'.format(model_name, exp_i, non_overlapping_offset+exp_i)
            cur_fname = 'initial_model_spikes_{}_N_{}_seed_{}_60s'.format(model_name, N, non_overlapping_offset+exp_i)
            save_file_name = prefix + target_data_path + cur_fname + '.mat'
            if not os.path.exists(save_file_name):
                simulate_and_save_model_spike_train(model=snn, poisson_rate=10., t=60*1000, exp_num=exp_i,
                                                    model_name=model_name, fname=cur_fname)
            else:
                print('file exists. skipping..')


if __name__ == "__main__":
    main(sys.argv[1:])
