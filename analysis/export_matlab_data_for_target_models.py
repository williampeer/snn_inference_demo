import os
import sys

import numpy as np
import torch

from data_util import prefix, target_data_path
from spike_train_matlab_export import simulate_and_save_model_spike_train


def main(argv):
    num_neurons = 4
    duration = 2 * 60 * 1000

    print('Argument List:', str(argv))

    opts = [opt for opt in argv if opt.startswith("-")]
    args = [arg for arg in argv if not arg.startswith("-")]
    for i, opt in enumerate(opts):
        if opt == '-h':
            print('main.py -N <num-neurons> -d <duration> -GF <glif-only-flag>')
            sys.exit()
        elif opt in ("-d", "--duration"):
            duration = int(args[i])
        elif opt in ("-N", "--num-neurons"):
            num_neurons = int(args[i])

    for m_fn in []:

        for f_i in range(3, 7):
        # for f_i in range(42, 43):
            torch.manual_seed(f_i)
            np.random.seed(f_i)

            # init_params_model = draw_from_uniform(model_class.parameter_init_intervals, num_neurons)
            snn = m_fn(random_seed=f_i, N=num_neurons)

            cur_fname = 'target_model_spikes_{}_N_{}_seed_{}_duration_{}'.format(snn.name(), num_neurons, f_i, duration)
            # cur_fname = 'target_model_sbi_spikes_{}_N_{}_seed_{}_duration_{}'.format(snn.name(), num_neurons, random_seed, duration)
            save_file_name = prefix + target_data_path + cur_fname + '.mat'
            # if not os.path.exists(save_file_name):
            simulate_and_save_model_spike_train(model=snn, poisson_rate=10., t=duration, exp_num=f_i,
                                                    model_name=snn.name(), fname=cur_fname)
            # else:
            #     print('file exists. skipping..')


if __name__ == "__main__":
    main(sys.argv[1:])
