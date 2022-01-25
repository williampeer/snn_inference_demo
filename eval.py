from enum import Enum

import torch.nn.functional
from torch.nn.functional import kl_div

import model_util
import spike_metrics
from experiments import release_computational_graph, sine_modulated_white_noise
from plot import *


def evaluate_loss(model, inputs, target_spiketrain, neurons_coeff, label='', exp_type=None, train_i=None, exp_num=None, constants=None, converged=False):
    if inputs is not None:
        assert (inputs.shape[0] == target_spiketrain.shape[0]), \
            "inputs and targets should have same shape. inputs shape: {}, targets shape: {}".format(inputs.shape, target_spiketrain.shape)
    else:
        inputs = sine_modulated_white_noise(t=target_spiketrain.shape[0], N=model.N, neurons_coeff=neurons_coeff)

    model_spike_train = model_util.feed_inputs_sequentially_return_spike_train(model, inputs)

    print('-- sanity-checks --')
    print('model:')
    sanity_checks(model_spike_train)
    print('target:')
    sanity_checks(target_spiketrain)
    print('-- sanity-checks-done --')

    loss = calculate_loss(model_spike_train, target_spiketrain, constants=constants)
    print('loss:', loss)

    if exp_type is None:
        exp_type_str = 'default'
    else:
        exp_type_str = exp_type.name

    if train_i % constants.evaluate_step == 0 or converged or train_i == constants.train_iters -1:
        plot_spike_trains_side_by_side(model_spike_train, target_spiketrain, uuid=model.__class__.__name__+'/'+constants.UUID,
                                       exp_type=exp_type_str, title='Spike trains ({}, loss: {:.3f})'.format(label, loss),
                                       fname='spiketrains_set_{}_exp_{}_train_iter_{}'.format(model.__class__.__name__, exp_num, train_i))
    np_loss = loss.clone().detach().numpy()
    release_computational_graph(model, inputs)
    loss = None
    return np_loss


class LossFn(Enum):
    FIRING_RATE_DIST = 'frd'
    VAN_ROSSUM_DIST = 'vrd'
    FANO_FACTOR_DIST = 'FF'
    CV_DIST = 'CV'
    MSE = 'mse'
    KL_DIV = 'kl_div'
    PEARSON_CORRELATION_COEFFICIENT = 'PCC'
    RATE_FANO_HYBRID = 'rfh'
    RATE_PCC_HYBRID = 'rph'
    NLL = 'nll'


def calculate_loss(output, target, constants):
    lfn = LossFn[constants.loss_fn]
    if lfn == LossFn.KL_DIV:
        loss = - kl_div(output, target)
    elif lfn == LossFn.MSE:
        loss = spike_metrics.mse(output, target)
    elif lfn == LossFn.VAN_ROSSUM_DIST:
        loss = spike_metrics.van_rossum_dist(output, target, constants.tau_van_rossum)
    elif lfn == LossFn.FIRING_RATE_DIST:
        # surrogate_gradient_output = torch.sigmoid(8*output-6*torch.ones_like(output))
        # loss = spike_metrics.firing_rate_distance(surrogate_gradient_output, target)
        loss = spike_metrics.firing_rate_distance(output, target)
    elif lfn == LossFn.FANO_FACTOR_DIST:
        loss = spike_metrics.fano_factor_dist(output, target)
    elif lfn == LossFn.CV_DIST:
        loss = spike_metrics.CV_dist(output, target)
    elif lfn == LossFn.PEARSON_CORRELATION_COEFFICIENT:
        loss = spike_metrics.correlation_metric_distance(output, target, constants.bin_size)
    elif lfn == LossFn.RATE_FANO_HYBRID:
        loss = spike_metrics.firing_rate_distance(output, target) + spike_metrics.fano_factor_dist(output, target)
    elif lfn == LossFn.RATE_PCC_HYBRID:
        loss = spike_metrics.firing_rate_distance(output, target) + \
               spike_metrics.correlation_metric_distance(output, target, constants.bin_size)
    elif lfn == LossFn.NLL:
        loss = 1.
    else:
        raise NotImplementedError("Loss function not supported.")

    if constants.silent_penalty_factor is not None:
        silent_penalty = spike_metrics.silent_penalty_term(output, target)
        return loss + constants.silent_penalty_factor * silent_penalty
    else:
        return loss
    # return loss + silent_penalty + activity_term

# --------------------------------------------------------


def sanity_checks(spiketrain):
    neuron_spikes_continuous = spiketrain.sum(0)
    neuron_spikes = torch.round(spiketrain).sum(0)
    silent_neurons = (neuron_spikes == 0).sum()

    print('# silent neurons: ', silent_neurons)
    print('spikes per neuron:', neuron_spikes)
    print('spike signals sum per neuron:', neuron_spikes_continuous)
