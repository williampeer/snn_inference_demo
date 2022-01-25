import torch.distributions.poisson
import torch.tensor as T

import Log
import PDF_metrics
import gif_fit
import model_util
from Constants import ExperimentType
from data_util import load_sparse_data
from eval import sanity_checks
from experiments import draw_from_uniform, release_computational_graph, \
    generate_synthetic_data_tuple, micro_gif_input
from plot import *

torch.autograd.set_detect_anomaly(True)
# ---------------------------------------


def stats_training_iterations(model_parameters, model, train_losses, test_losses, constants, logger, exp_type_str, target_parameters, exp_num, train_i):
    if constants.plot_flag:
        plot_loss(loss=test_losses, uuid=model.__class__.__name__+'/'+constants.UUID, exp_type=exp_type_str,
                  custom_title='Loss ({}, {}, {}, lr={})'.format(model.__class__.__name__, constants.optimiser.__name__,
                                                                 constants.loss_fn, constants.learn_rate),
                  fname='test_loss_exp_{}_loss_fn_{}_tau_vr_{}'.format(exp_num, constants.loss_fn, str(constants.tau_van_rossum).replace('.', '_')))
        plot_loss(loss=train_losses, uuid=model.__class__.__name__+'/'+constants.UUID, exp_type=exp_type_str,
                  custom_title='Loss ({}, {}, {}, lr={})'.format(model.__class__.__name__, constants.optimiser.__name__,
                                                                 constants.loss_fn, constants.learn_rate),
                  fname='train_loss_exp_{}_loss_fn_{}_tau_vr_{}'.format(exp_num, constants.loss_fn,
                                                                       str(constants.tau_van_rossum).replace('.', '_')))

        parameter_names = model.free_parameters
        plot_parameter_inference_trajectories_2d(model_parameters,
                                                 uuid=model.__class__.__name__+'/'+constants.UUID,
                                                 exp_type=exp_type_str,
                                                 target_params=target_parameters,
                                                 param_names=parameter_names,
                                                 custom_title='Inferred parameters across training iterations',
                                                 fname='inferred_param_trajectories_{}_exp_num_{}_train_iters_{}'
                                                 .format(model.__class__.__name__, exp_num, train_i),
                                                 logger=logger)

        # ------------- trajectories weights ------------------
        if model.state_dict().__contains__('w'):
            tar_weights_params = {}
            if target_parameters is not None:
                tar_weights_params['w'] = np.mean(target_parameters['w'].numpy(), axis=1)

            weights = model_parameters['w']
            assert len(weights[0].shape) == 2, "weights should be 2D"
            weights_params = { 'w' : [] }
            w_names = ['w']
            # weights_params[0] = [np.mean(weights[0], axis=1)]
            for n_i in range(len(weights)):
                # cur_w_name = 'w_{}'.format(n_i)
                # w_names.append(cur_w_name)
                weights_params['w'].append(np.mean(weights[n_i], axis=1))

            plot_parameter_inference_trajectories_2d(weights_params, target_params=tar_weights_params,
                                                     uuid=model.__class__.__name__+'/'+constants.UUID,
                                                     exp_type=exp_type_str,
                                                     param_names=w_names,
                                                     custom_title='Avg inferred weights across training iterations',
                                                     fname='avg_inferred_weights_param_trajectories_{}_exp_num_{}_train_iters_{}'
                                                     .format(model.__class__.__name__, exp_num, train_i),
                                                     logger=logger)
        # ------------------------------------------------------

    logger.log('train_losses: #{}'.format(train_losses))
    mean_test_loss = torch.mean(torch.tensor(test_losses)).clone().detach().numpy()
    logger.log('test_losses: #{}'.format(test_losses), ['mean test loss: {}'.format(mean_test_loss)])

    cur_fname = '{}_exp_num_{}_data_set_{}_mean_loss_{:.3f}_uuid_{}'.format(model.__class__.__name__, exp_num, constants.data_set, mean_test_loss, constants.UUID)
    IO.save(model, loss={'train_losses': train_losses, 'test_losses': test_losses}, uuid=constants.UUID, fname=cur_fname)

    del model, mean_test_loss


def convergence_check(validation_losses):
    if len(validation_losses) <= 1:
        return False

    val_diff = validation_losses[-1] - validation_losses[-2]
    return val_diff >= 0.


def overall_gradients_mean(gradients, train_i, loss_fn):
    mean_logger = Log.Logger('gradients_mean_log')
    full_logger = Log.Logger('gradients_full_log')

    avg_grads = []
    for i, grads in enumerate(gradients):
        avg_grads.append(torch.mean(grads))
    overall_mean = torch.mean(torch.tensor(avg_grads))
    mean_logger.log('avg_grads: {}, train_i: {}, loss_fn: {}'.format(avg_grads, train_i, loss_fn))
    mean_logger.log('overall_mean: {}'.format(overall_mean))

    full_logger.log('train_i: {}, loss_fn: {}, gradients'.format(train_i, loss_fn), gradients)
    return float(overall_mean.clone().detach())


def evaluate_loss_tuple(model, inputs, target_spiketrain, label, exp_type, train_i, exp_num, constants, converged, neurons_coeff):
    if inputs is not None:
        assert (inputs.shape[0] == target_spiketrain.shape[0]), \
            "inputs and targets should have same shape. inputs shape: {}, targets shape: {}".format(inputs.shape,
                                                                                                    target_spiketrain.shape)
    else:
        N = model.N
        inputs = micro_gif_input(t=target_spiketrain.shape[0], N=N,
                                            neurons_coeff=neurons_coeff)

    sproba, model_spike_train = model_util.feed_inputs_sequentially_return_tuple(model, inputs)

    print('-- sanity-checks --')
    print('model:')
    sanity_checks(model_spike_train)
    print('target:')
    sanity_checks(target_spiketrain)
    print('-- sanity-checks-done --')

    loss = PDF_metrics.calculate_loss(spike_probabilities=sproba, target_spikes=target_spiketrain, constants=constants.loss_fn, bin_size=constants.bin_size)
    loss.backward(retain_graph=True)

    print('loss:', loss)

    if exp_type is None:
        exp_type_str = 'default'
    else:
        exp_type_str = exp_type.name

    if train_i % constants.evaluate_step == 0 or converged or train_i == constants.train_iters - 1:
        plot_spike_trains_side_by_side(model_spike_train, target_spiketrain, uuid=model.__class__.__name__+'/'+constants.UUID,
                                       exp_type=exp_type_str,
                                       title='Spike trains ({}, loss: {:.3f})'.format(label, loss),
                                       fname='spiketrains_set_{}_exp_{}_train_iter_{}'.format(
                                           model.__class__.__name__, exp_num, train_i))
    np_loss = loss.clone().detach().numpy()
    release_computational_graph(model, inputs)
    loss = None
    return np_loss


def fit_model(logger, constants, model_class, params_model, exp_num, neurons_coeff, target_model=None, target_parameters=None, num_neurons=12):
    params_model['N'] = num_neurons
    neuron_types = np.ones((num_neurons,))
    for i in range(int(num_neurons/2)):
        neuron_types[-(1+i)] = -1

    if constants.EXP_TYPE == ExperimentType.Synthetic:
        # if model_class.__name__.__contains__('microGIF'):
        params_model['R_m'] = target_model.R_m.clone().detach()
        if model_class.__name__.__contains__('weights_only'):
            params_model = target_model.get_parameters()
    elif constants.EXP_TYPE == ExperimentType.SanityCheck:
        params_model = target_model.get_parameters()
        params_model['preset_weights'] = params_model['w']

    model = model_class(N=num_neurons, parameters=params_model, neuron_types=neuron_types)
    logger.log('initial model parameters: {}'.format(params_model), [model_class.__name__])
    parameters = {}
    for p_i, key in enumerate(model.state_dict()):
        parameters[key] = [model.state_dict()[key].numpy()]

    optim_params = list(model.parameters())
    optim = constants.optimiser(optim_params, lr=constants.learn_rate)

    test_losses = np.array([]); train_losses = np.array([]); train_i = 0; converged = False; next_step = 0

    inputs = None
    train_targets, gen_inputs = generate_synthetic_data_tuple(target_model, t=constants.rows_per_train_iter,
                                                              burn_in=constants.burn_in)
    if constants.EXP_TYPE == ExperimentType.SanityCheck:
        inputs = gen_inputs

    loss_prior_to_training = evaluate_loss_tuple(model, inputs=inputs,
                                           target_spiketrain=train_targets, label='train i: {}'.format(train_i),
                                           exp_type=constants.EXP_TYPE, train_i=train_i, exp_num=exp_num,
                                           constants=constants, converged=converged, neurons_coeff=neurons_coeff)
    test_losses = np.concatenate((test_losses, np.asarray([loss_prior_to_training])))

    # while not converged and (train_i < constants.train_iters):
    while train_i < constants.train_iters:
        train_i += 1
        logger.log('training iteration #{}'.format(train_i), [constants.EXP_TYPE])

        # ---- Train ----
        train_input = None
        train_targets, gen_train_input = generate_synthetic_data_tuple(gen_model=target_model, t=constants.rows_per_train_iter,
                                                                       burn_in=constants.burn_in)
        if constants.EXP_TYPE == ExperimentType.SanityCheck:
            train_input = gen_train_input

        avg_unseen_loss, abs_grads_mean, converged = gif_fit.fit_batches(model, gen_inputs=train_input, target_spiketrain=train_targets,
                                                                         optimiser=optim, constants=constants, train_i=train_i, logger=logger)

        cur_params = model.state_dict()
        logger.log('current parameters {}'.format(cur_params))

        for p_i, key in enumerate(cur_params):
            parameters[key].append(cur_params[key].clone().detach().numpy())

        release_computational_graph(target_model, constants.initial_poisson_rate)

        logger.log(parameters=[avg_unseen_loss, abs_grads_mean])
        test_losses = np.concatenate((test_losses, np.asarray([avg_unseen_loss])))

        train_loss = evaluate_loss_tuple(model, inputs=train_input,
                                   target_spiketrain=train_targets, label='train i: {}'.format(train_i),
                                   exp_type=constants.EXP_TYPE, train_i=train_i, exp_num=exp_num,
                                   constants=constants, converged=converged, neurons_coeff=neurons_coeff)
        logger.log(parameters=['train loss', train_loss])
        train_losses = np.concatenate((train_losses, np.asarray([train_loss])))

        release_computational_graph(target_model, constants.initial_poisson_rate)
        release_computational_graph(model, train_input)
        train_targets = None; train_loss = None

    stats_training_iterations(model_parameters=parameters, model=model,
                              train_losses=train_losses, test_losses=test_losses,
                              constants=constants, logger=logger, exp_type_str=constants.EXP_TYPE.name,
                              target_parameters=target_parameters, exp_num=exp_num, train_i=train_i)
    final_model_parameters = {}
    for p_i, key in enumerate(model.state_dict()):
        final_model_parameters[key] = [model.state_dict()[key].numpy()]
    model = None
    return final_model_parameters, test_losses, train_losses, train_i, None


def run_exp_loop(logger, constants, model_class, target_model, pop_sizes, error_logger=Log.Logger('DEFAULT_ERR_LOG')):
    if hasattr(target_model, 'get_parameters'):
        target_parameters = target_model.get_parameters()
    elif target_model is not None:
        target_parameters = target_model.state_dict()
    else:
        target_parameters = False

    recovered_param_per_exp = {}
    for exp_i in range(constants.start_seed, constants.start_seed+constants.N_exp):
        non_overlapping_offset = constants.start_seed + constants.N_exp + 1
        torch.manual_seed(non_overlapping_offset + exp_i)
        np.random.seed(non_overlapping_offset + exp_i)

        if target_model is not None:
            target_model.load_state_dict(target_model.state_dict())
            num_neurons = int(target_model.v.shape[0])
        else:
            node_indices, spike_times, spike_indices = load_sparse_data(full_path=constants.data_path)
            num_neurons = len(node_indices)

        if hasattr(model_class, 'parameter_init_intervals'):
            init_params_model = draw_from_uniform(model_class.parameter_init_intervals, num_neurons)
        else:
            init_params_model = {}
        if len(pop_sizes) == 4:
            neurons_coeff = torch.cat([T(pop_sizes[0] * [0.]), T(pop_sizes[1] * [0.]), T(pop_sizes[2] * [0.25]), T(pop_sizes[3] * [0.1])])
        elif len(pop_sizes) == 2:
            neurons_coeff = torch.cat([T(2 * [0.25]), T(2 * [0.1])])
        recovered_parameters, train_losses, test_losses, train_i, poisson_rates = \
            fit_model(logger, constants, model_class, init_params_model, exp_num=exp_i, target_model=target_model,
                      target_parameters=target_parameters, num_neurons=num_neurons, neurons_coeff=neurons_coeff)

        if train_i >= constants.train_iters:
            print('DID NOT CONVERGE FOR SEED, CONTINUING ON TO NEXT SEED. exp_i: {}, train_i: {}, train_losses: {}, test_losses: {}'
                  .format(exp_i, train_i, train_losses, test_losses))

        for p_i, key in enumerate(recovered_parameters):
            if exp_i == constants.start_seed:
                recovered_param_per_exp[key] = [recovered_parameters[key]]
            else:
                recovered_param_per_exp[key].append(recovered_parameters[key])
    parameter_names = model_class.free_parameters
    if constants.plot_flag:
        plot_all_param_pairs_with_variance(recovered_param_per_exp,
                                           uuid=model_class.__name__+'/'+constants.UUID,
                                           exp_type=constants.EXP_TYPE.name,
                                           target_params=target_parameters,
                                           param_names=parameter_names,
                                           custom_title="Average inferred parameters across experiments [{}, {}]".format(
                                               model_class.__name__, constants.optimiser),
                                           logger=logger, fname='all_inferred_params_{}'.format(model_class.__name__))


def start_exp(constants, model_class, target_model, pop_sizes):
    log_fname = model_class.__name__ + '_{}_{}_{}_lr_{}_batchsize_{}_trainiters_{}_rowspertrainiter_{}_uuid_{}'. \
        format(constants.optimiser.__name__, constants.loss_fn, constants.EXP_TYPE.name,
               '{:1.3f}'.format(constants.learn_rate).replace('.', '_'),
               constants.batch_size, constants.train_iters, constants.rows_per_train_iter, constants.UUID)
    logger = Log.Logger(log_fname)
    err_logger = Log.Logger('ERROR_LOG_{}'.format(log_fname))
    logger.log('Starting exp. with listed hyperparameters.', [constants.__str__()])

    run_exp_loop(logger, constants, model_class, target_model, pop_sizes, err_logger)
