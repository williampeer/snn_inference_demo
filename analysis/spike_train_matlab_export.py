import numpy as np
import torch

import experiments
from IO import save_model_params
from Models.microGIF import microGIF
from data_util import save_spiketrain_in_sparse_matlab_format, convert_to_sparse_vectors
from model_util import feed_inputs_sequentially_return_tuple, feed_inputs_sequentially_return_args
from plot import plot_spike_train


def load_and_export_sim_data(model_path, fname=False, t = 60 * 1000):
    # print('Argument List:', str(argv))

    # loss_fn = model_path.split('loss_fn_')[1].split('_budget')[0]
    # cur_model_descr = model_path.split('fitted_params_')[1].split('_optim')[0]
    cur_model_name = model_path.split('_exp_num')[0].split('/')[-1]
    exp_num = model_path.split('exp_num_')[1].split('_data_set')[0]
    # optim = model_path.split('optim_')[1].split('_loss_fn')[0]
    # id = optim + '_' + model_path.split('.pt')[0].split('-')[-1]
    # lr = ''

    exp_res = torch.load(model_path)
    model = exp_res['model']
    poisson_rate = exp_res['rate']
    # loss = data['loss']

    print('Loaded model.')

    simulate_and_save_model_spike_train(model, t, exp_num, cur_model_name, fname=fname)


def simulate_and_save_model_spike_train(model, t, exp_num, model_name, fname=False):
    interval_size = 6000
    interval_range = int(t / interval_size)
    assert interval_range > 0, "t must be greater than the interval size, {}. t={}".format(interval_size, t)

    spike_indices = np.array([], dtype='int8')
    spike_times = np.array([], dtype='float32')
    input_indices = np.array([], dtype='int8')
    input_times = np.array([], dtype='float32')
    print('Simulating data..')
    spikes = None
    for t_i in range(interval_range):
        model.reset_hidden_state()
        # spiketrain = generate_synthetic_data(model, poisson_rate, t=interval_size)
        t = interval_size
        white_noise = torch.rand((t, model.N))
        # inputs = experiments.sine_modulated_input(white_noise)
        inputs = white_noise
        if model.__class__ is microGIF:
            _, spikes, _ = feed_inputs_sequentially_return_args(model=model, inputs=inputs.clone().detach())
        else:
            _, spikes = feed_inputs_sequentially_return_tuple(model=model, inputs=inputs.clone().detach())
        # for gen spiketrain this may be thresholded to binary values:
        spikes = torch.round(spikes).clone().detach()

        cur_input_indices, cur_input_times = convert_to_sparse_vectors(inputs, t_offset=t_i * interval_size)
        input_indices = np.append(input_indices, cur_input_indices)
        input_times = np.append(input_times, cur_input_times)
        cur_spike_indices, cur_spike_times = convert_to_sparse_vectors(spikes, t_offset=t_i * interval_size)
        spike_indices = np.append(spike_indices, cur_spike_indices)
        spike_times = np.append(spike_times, cur_spike_times)
        print('{} seconds ({:.2f} min) simulated.'.format(interval_size * (t_i + 1) / 1000.,
                                                          interval_size * (t_i + 1) / (60. * 1000)))

    # fname = model_path.split('/')[-1]
    # model_name = fname.split('.pt')[0]
    # save_fname_input = 'poisson_inputs_{}_t_{:.0f}s_rate_{}'.format(model_name, t/1000., poisson_rate).replace('.', '_') + '.mat'
    # save_spiketrain_in_sparse_matlab_format(fname=save_fname_input, spike_indices=input_indices, spike_times=input_times)
    # save_model_params(model, fname=save_fname_input.replace('.mat', '_params'))
    save_fname_output = 'fitted_spikes_{}_{}_{}_t_{}'.format(model_name, id, exp_num, t).replace('.', '_')
    if not fname:
        fname = save_fname_output
    if spikes is not None:
        plot_spike_train(spikes, 'Plot imported SNN', 'plot_imported_model', fname=fname)

    # save_fname_output = 'fitted_spike_train_{}_{}_{}{}_exp_num_{}'.format(cur_model_descr, optim, loss_fn, lr, exp_num).replace('.', '_') + '.mat'
    # save_fname_output = 'fitted_spike_train_{}_id_{}_exp_no_{}'.format(cur_model_name, id, exp_num).replace('.', '_') + '.mat'

    # fname = model.__class__.__name__ + '/' + fname
    save_spiketrain_in_sparse_matlab_format(fname=fname + '.mat', spike_indices=spike_indices, spike_times=spike_times)
    save_model_params(model, fname=fname.replace('.mat', '_params'))
