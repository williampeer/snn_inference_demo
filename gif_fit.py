import numpy as np
import torch

import PDF_metrics
import experiments
import model_util
from Constants import ExperimentType
from experiments import release_computational_graph, micro_gif_input


def fit_batches(model, gen_inputs, target_spiketrain, optimiser, constants, train_i=None, logger=None):
    avg_abs_grads = []
    for _ in range(len(list(model.parameters()))):
        avg_abs_grads.append([])

    optimiser.zero_grad()
    converged_batches = []

    if constants.EXP_TYPE is ExperimentType.SanityCheck and gen_inputs is not None:
        current_inputs = gen_inputs.clone().detach().requires_grad_(True)
        current_inputs.retain_grad()
    else:
        N = model.N
        t = constants.rows_per_train_iter
        if constants.burn_in:
            burn_in_len = int(target_spiketrain.shape[0] / 10)
            print('simulating burn_in for {} ms..'.format(burn_in_len))
            # burn_in_inputs = micro_gif_input(t=burn_in_len, N=model.N, neurons_coeff=neurons_coeff)
            burn_in_inputs = experiments.get_interesting_inputs(t=burn_in_len, N=model.N)
            _, _ = model_util.feed_inputs_sequentially_return_tuple(model, burn_in_inputs)
        # current_inputs = micro_gif_input(t=constants.rows_per_train_iter, N=model.N, neurons_coeff=neurons_coeff)
        current_inputs = experiments.get_interesting_inputs(t, N)
        current_inputs = torch.tensor(current_inputs.clone().detach(), requires_grad=True)

    spike_probs, expressed_model_spikes = model_util.feed_inputs_sequentially_return_tuple(model, current_inputs)

    loss = PDF_metrics.calculate_loss(spike_probs, target_spiketrain.detach(), constants.loss_fn, constants.bin_size)
    loss.backward(retain_graph=True)

    param_grads_converged = []
    for p_i, param in enumerate(list(model.parameters())):
        logger.log('grad for param #{}: {}'.format(p_i, param.grad))
        if constants.norm_grad_flag is True:
            max_grad = torch.max(param.grad)
            if max_grad > 0:
                param.grad = param.grad/torch.max(param.grad)  # normalise

        avg_abs_grads[p_i].append(np.mean(np.abs(param.grad.clone().detach().numpy())))

        cur_p_mean_grad = np.mean(np.abs(param.grad.clone().detach().numpy()))
        if p_i > 0:
            cur_p_max = model.__class__.parameter_init_intervals[model.__class__.free_parameters[p_i]][1]
        else:  # 'w'
            cur_p_max = 1.

        cur_converged = cur_p_mean_grad < 1e-03 * cur_p_max
        param_grads_converged.append(cur_converged)

    converged = np.array(param_grads_converged).sum() == len(param_grads_converged)
    converged_batches.append(converged)

    optimiser.step()
    release_computational_graph(model, current_inputs)
    avg_unseen_loss = loss.clone().detach()
    spikes = None; loss = None; current_inputs = None

    logger.log('train_i #: {},\navg_abs_grads: {}'.format(train_i, avg_abs_grads))
    gen_inputs = None

    converged = np.array(converged_batches).sum() == len(converged_batches)
    print('converged last batch: {}\nconverged_batches: {}\nconverged: {}'.format(param_grads_converged, converged_batches, converged))

    return avg_unseen_loss, np.mean(np.asarray(avg_abs_grads, dtype=np.float)), converged
