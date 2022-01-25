import numpy as np
import torch

import model_util
from Constants import ExperimentType
from eval import calculate_loss
from experiments import release_computational_graph, sine_modulated_white_noise


def fit_batches(model, gen_inputs, target_spiketrain, optimiser, constants, neurons_coeff, train_i=None, logger=None):
    if gen_inputs is not None:
        assert gen_inputs.shape[0] == target_spiketrain.shape[0], \
            "inputs shape: {}, target spiketrain shape: {}".format(gen_inputs.shape, target_spiketrain.shape)
        gen_inputs = gen_inputs.clone().detach()

    batch_size = constants.batch_size
    batch_N = int(target_spiketrain.shape[0]/batch_size)
    assert batch_N > 0, "batch_N was not above zero. batch_N: {}".format(batch_N)
    print('num. of batches of size {}: {}'.format(batch_size, batch_N))
    batch_losses = []; avg_abs_grads = []
    for _ in range(len(list(model.parameters()))):
        avg_abs_grads.append([])

    optimiser.zero_grad()
    converged_batches = []
    for batch_i in range(batch_N):
        print('batch #{}'.format(batch_i))

        if constants.EXP_TYPE is ExperimentType.SanityCheck and gen_inputs is not None:
            current_inputs = gen_inputs[batch_size * batch_i:batch_size * (batch_i + 1)].clone().detach().requires_grad_(True)
            current_inputs.retain_grad()
        else:
            N = model.N
            if constants.burn_in:
                burn_in_len = int(target_spiketrain.shape[0] / 10)
                print('simulating burn_in for {} ms..'.format(burn_in_len))
                burn_in_inputs = sine_modulated_white_noise(t=burn_in_len, N=N, neurons_coeff=neurons_coeff)
                _ = model_util.feed_inputs_sequentially_return_spike_train(model, burn_in_inputs)
            current_inputs = sine_modulated_white_noise(t=batch_size, N=N, neurons_coeff=neurons_coeff)
            current_inputs.retain_grad()

        spikes = model_util.feed_inputs_sequentially_return_spike_train(model, current_inputs)

        # returns tensor, maintains gradient
        loss = calculate_loss(spikes, target_spiketrain[batch_size * batch_i:batch_size * (batch_i + 1)].detach(), constants=constants)

        # optimiser.zero_grad()
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

            cur_converged = cur_p_mean_grad < 1e-02 * cur_p_max
            param_grads_converged.append(cur_converged)

        converged = np.array(param_grads_converged).sum() == len(param_grads_converged)
        converged_batches.append(converged)

        print('batch loss: {}'.format(loss))
        batch_losses.append(float(loss.clone().detach().data))

    optimiser.step()
    release_computational_graph(model, current_inputs)
    spikes = None; loss = None; current_inputs = None

    avg_batch_loss = np.mean(np.asarray(batch_losses, dtype=np.float))

    logger.log('batch losses: {}'.format(batch_losses))
    logger.log('avg_batch_loss: {}'.format(avg_batch_loss), {'train_i': train_i})
    logger.log('train_i #: {},\navg_abs_grads: {}'.format(train_i, avg_abs_grads))
    gen_inputs = None

    converged = np.array(converged_batches).sum() == len(converged_batches)
    print('converged last batch: {}\nconverged_batches: {}\nconverged: {}'.format(param_grads_converged, converged_batches, converged))

    return avg_batch_loss, np.mean(np.asarray(avg_abs_grads, dtype=np.float)), batch_losses[-1], converged
