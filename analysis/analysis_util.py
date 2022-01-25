import os
import sys
import torch
import numpy as np

import IO
import data_util
import experiments
import model_util
from Models.microGIF import microGIF


def get_lfn_loss_from_plot_data_in_folder(exp_folder):
    folder_files = os.listdir(exp_folder)
    loss_file = list(filter(lambda x: x.__contains__('plot_loss'), folder_files))[0]
    plot_data = torch.load(exp_folder + loss_file)['plot_data']
    custom_title = plot_data['custom_title']
    lfn = custom_title.split(',')[0].strip('Loss ')
    loss = plot_data['loss'][-1]
    return lfn, loss


def get_target_model(model_type_str):
    GT_path = '/home/william/repos/snn_inference/Test/saved/'
    GT_model_by_type = {'LIF': '12-09_11-49-59-999',
                        'GLIF': '12-09_11-12-47-541',
                        'mesoGIF': '12-09_14-56-20-319',
                        'microGIF': '12-09_14-56-17-312'}

    GT_euid = GT_model_by_type[model_type_str]
    tar_fname = 'snn_model_target_GD_test'
    model_name = model_type_str
    if model_type_str == 'mesoGIF':
        model_name = 'microGIF'
    load_data_target = torch.load(GT_path + model_name + '/' + GT_euid + '/' + tar_fname + IO.fname_ext)
    target_model = load_data_target['model']
    return target_model


def get_target_rate_for_sleep_exp(exp_str):
    sleep_data_path = data_util.prefix + data_util.sleep_data_path
    sleep_data_files = ['exp108.mat', 'exp109.mat', 'exp124.mat', 'exp126.mat', 'exp138.mat', 'exp146.mat', 'exp147.mat']
    data_file = exp_str + '.mat'
    assert data_file in sleep_data_files, "exp_str: {} not found in sleep data files: {}".format(exp_str, sleep_data_files)

    node_indices, spike_times, spike_indices = data_util.load_sparse_data(sleep_data_path + data_file)
    _, target_spikes = data_util.get_spike_train_matrix(0, 12000, spike_times, spike_indices, node_indices)
    # cur_mean_rate_np, cur_std_rate_np = get_mean_rate_for_spikes(target_spikes)
    return get_mean_rate_for_spikes(target_spikes)
    # return cur_mean_rate_np, stds


def get_mean_rate_for_spikes(spikes):
    normalised_spike_rate = spikes.sum(dim=0) * 1000. / (spikes.shape[0])
    return np.mean(normalised_spike_rate.numpy()), np.std(normalised_spike_rate.numpy())


def get_mean_rate_for_model(model):
    white_noise = torch.rand((4000, model.N))
    # inputs = experiments.sine_modulated_input(white_noise)
    inputs = white_noise
    if model.__class__ is microGIF:
        _, spikes, _ = model_util.feed_inputs_sequentially_return_args(model=model, inputs=inputs.clone().detach())
    else:
        _, spikes = model_util.feed_inputs_sequentially_return_tuple(model=model, inputs=inputs.clone().detach())
    # for gen spiketrain this may be thresholded to binary values:
    spikes = torch.round(spikes).clone().detach()
    normalised_spike_rate = spikes.sum(dim=0) * 1000. / (spikes.shape[0])
    return np.mean(normalised_spike_rate.numpy())


def get_param_dist(model, target_model):
    model_params = model.get_parameters()
    target_params = target_model.get_parameters()
    assert len(model_params) == len(target_params), "parameter dicts should be of equal length.."

    total_mean_param_dist = 0.
    for p_v, p_k in enumerate(model_params):
        # if p_k != 'w':
        m_p = model_params[p_k].numpy()
        t_p = target_params[p_k].numpy()
        p_rmse = np.sqrt(np.mean(np.power(m_p - t_p, 2)))
        total_mean_param_dist += p_rmse

    return (total_mean_param_dist/len(model_params))


def get_init_model(model_class, seed, N, neuron_types = False):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    params_model = experiments.draw_from_uniform(model_class.parameter_init_intervals, N)
    if model_class is not microGIF:
        model = model_class(params_model, N=N, neuron_types=neuron_types)
    else:
        model = model_class(params_model, N=N)

    return model


def get_init_param_dist(target_model):
    model_class = target_model.__class__
    start_seed = 23
    p_dists = []
    for i in range(20):
        neuron_types = False
        if hasattr(target_model, 'neuron_types'):
            neuron_types = target_model.neuron_types
        model = get_init_model(model_class, seed=start_seed+i, N=target_model.N, neuron_types=neuron_types)

        cur_dist = get_param_dist(model, target_model)
        p_dists.append(cur_dist)

    return np.mean(p_dists), np.std(p_dists)
