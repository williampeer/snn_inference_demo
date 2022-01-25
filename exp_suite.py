import torch.tensor as T
import Log
from Constants import ExperimentType
from data_util import load_sparse_data, get_spike_train_matrix
from eval import evaluate_loss
from experiments import generate_synthetic_data, draw_from_uniform, release_computational_graph
from fit import fit_batches
from plot import *

torch.autograd.set_detect_anomaly(True)
# ---------------------------------------


def stats_training_iterations(model_parameters, model, train_losses, test_losses, constants, logger, exp_type_str, target_parameters, exp_num, train_i):
    if constants.plot_flag:
        parameter_names = model.parameter_names
        # parameter_names.append('p_rate')
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
            weights_params = {'w': []}
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

        plot_losses(training_loss=train_losses, test_loss=test_losses, uuid=model.__class__.__name__+'/'+constants.UUID,
                    exp_type=exp_type_str,
                    custom_title='Loss ({}, {}, {}, lr={})'.format(model.__class__.__name__, constants.optimiser.__name__,
                                                                       constants.loss_fn, constants.learn_rate),
                    fname='training_and_test_loss_exp_{}_loss_fn_{}_tau_vr_{}'.format(exp_num, constants.loss_fn, str(constants.tau_van_rossum).replace('.', '_')))

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


def fit_model(logger, constants, model_class, params_model, exp_num, neurons_coeff, target_model=None, target_parameters=None, num_neurons=12,
              error_logger=None):
    params_model['N'] = num_neurons
    pop_types = [1, -1, 1, -1]
    neuron_types = np.ones((num_neurons,))
    N_pops = 4
    pop_size = int(num_neurons/N_pops)
    for n_i in range(N_pops):
        for n_j in range(pop_size):
            ind = n_i * pop_size + n_j
            neuron_types[ind] = pop_types[n_i]

    if model_class.__name__.__contains__('fixed_weights'):
        params_model['preset_weights'] = target_model.w.clone().detach()
    elif model_class.__name__.__contains__('weights_only'):
        tar_p_names = target_model.__class__.parameter_names
        tar_params = target_model.get_parameters()
        for p_i in range(1, len(tar_p_names)):
            params_model[tar_p_names[p_i]] = tar_params[p_i].clone().detach()
    elif model_class.__name__.__contains__('lower_dim'):
        model_p_names = model_class.parameter_names
        tar_p_names = target_model.__class__.parameter_names
        tar_params = target_model.get_parameters()
        for p_i in range(1, len(tar_p_names)):
            cur_tar_p_name = tar_p_names[p_i]
            if not model_p_names.__contains__(cur_tar_p_name):
                params_model[cur_tar_p_name] = tar_params[cur_tar_p_name].clone().detach()
    elif model_class.__name__.__contains__('microGIF'):
        params_model['R_m'] = target_model.R_m.clone().detach()
        params_model['c'] = target_model.c.clone().detach()
        params_model['pop_sizes'] = target_model.pop_sizes.clone().detach()

    model = model_class(N=num_neurons, parameters=params_model, neuron_types=neuron_types)
    logger.log('initial model parameters: {}'.format(params_model), [model_class.__name__])
    parameters = {}
    for p_i, key in enumerate(model.state_dict()):
        parameters[key] = [model.state_dict()[key].numpy()]

    optim_params = list(model.parameters())
    optim = constants.optimiser(optim_params, lr=constants.learn_rate)

    test_losses = np.array([]); train_losses = np.array([]); prev_spike_index = 0; train_i = 0; converged = False
    max_grads_mean = np.float(0.)
    next_step = 0

    inputs = None
    if constants.EXP_TYPE is ExperimentType.DataDriven:
        node_indices, spike_times, spike_indices = load_sparse_data(full_path=constants.data_path)
        next_step, train_targets = get_spike_train_matrix(index_last_step=next_step, advance_by_t_steps=constants.rows_per_train_iter,
                                                          spike_times=spike_times, spike_indices=spike_indices, node_numbers=node_indices)
    else:
        N = model.N
        train_targets, gen_inputs = generate_synthetic_data(target_model, t=constants.rows_per_train_iter,
                                                            neurons_coeff=neurons_coeff, burn_in=constants.burn_in)
        if constants.EXP_TYPE == ExperimentType.SanityCheck:
            inputs = gen_inputs

    loss_prior_to_training = evaluate_loss(model, inputs=inputs,
                                           target_spiketrain=train_targets, label='train i: {}'.format(train_i),
                                           exp_type=constants.EXP_TYPE, train_i=train_i, exp_num=exp_num,
                                           constants=constants, converged=converged, neurons_coeff=neurons_coeff)
    test_losses = np.concatenate((test_losses, np.asarray([loss_prior_to_training])))

    while not converged and (train_i < constants.train_iters):
        train_i += 1
        logger.log('training iteration #{}'.format(train_i), [constants.EXP_TYPE])

        # Train:
        train_input = None
        if constants.EXP_TYPE is ExperimentType.DataDriven:
            node_indices, spike_times, spike_indices = load_sparse_data(full_path=constants.data_path)
            next_step, train_targets = get_spike_train_matrix(index_last_step=next_step, advance_by_t_steps=constants.rows_per_train_iter,
                                                              spike_times=spike_times, spike_indices=spike_indices, node_numbers=node_indices)
            train_input = None
        else:
            N = model.N
            train_targets, gen_train_input = generate_synthetic_data(gen_model=target_model, t=constants.rows_per_train_iter,
                                                                     neurons_coeff=neurons_coeff, burn_in=constants.burn_in)
            if constants.EXP_TYPE == ExperimentType.SanityCheck:
                train_input = gen_train_input

        # try:
            avg_unseen_loss, abs_grads_mean, last_loss, converged = \
                fit_batches(model, gen_inputs=train_input, target_spiketrain=train_targets, optimiser=optim,
                            constants=constants, neurons_coeff=neurons_coeff, train_i=train_i, logger=logger)

            cur_params = model.state_dict()
            logger.log('current parameters {}'.format(cur_params))

            for p_i, key in enumerate(cur_params):
                parameters[key].append(cur_params[key].clone().detach().numpy())

            release_computational_graph(target_model, constants.initial_poisson_rate)

            logger.log(parameters=[avg_unseen_loss, abs_grads_mean])
            test_losses = np.concatenate((test_losses, np.asarray([avg_unseen_loss])))

        train_loss = evaluate_loss(model, inputs=train_input,
                                   target_spiketrain=train_targets, label='train i: {}'.format(train_i),
                                   exp_type=constants.EXP_TYPE, train_i=train_i, exp_num=exp_num,
                                   constants=constants, converged=converged,
                                   neurons_coeff=neurons_coeff)
        # validation_loss = last_loss
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
    return final_model_parameters, test_losses, train_losses, train_i


def run_exp_loop(logger, constants, model_class, target_model=None, error_logger=Log.Logger('DEFAULT_ERR_LOG')):
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
            # num_neurons = 12
            node_indices, spike_times, spike_indices = load_sparse_data(full_path=constants.data_path)
            num_neurons = len(node_indices)

        init_params_model = draw_from_uniform(model_class.parameter_init_intervals, num_neurons)
        N = num_neurons
        neurons_coeff = torch.cat([T(int(N / 4) * [0.1]), T(int(N / 4) * [0.1]), T(int(N / 4) * [0.5]), T(int(N / 4) * [0.2])])

        # try:
        recovered_parameters, train_losses, test_losses, train_i = \
            fit_model(logger, constants, model_class, init_params_model, exp_num=exp_i, target_model=target_model,
                      target_parameters=target_parameters, num_neurons=num_neurons,
                      error_logger=error_logger, neurons_coeff=neurons_coeff)

        # logger.log('poisson rates for exp {}'.format(exp_i), poisson_rates)

        if train_i >= constants.train_iters:
            print('DID NOT CONVERGE FOR SEED, CONTINUING ON TO NEXT SEED. exp_i: {}, train_i: {}, train_losses: {}, test_losses: {}'
                  .format(exp_i, train_i, train_losses, test_losses))

        for p_i, key in enumerate(recovered_parameters):
            if exp_i == constants.start_seed:
                recovered_param_per_exp[key] = [recovered_parameters[key]]
            else:
                recovered_param_per_exp[key].append(recovered_parameters[key])
    parameter_names = model_class.parameter_names
    # parameter_names.append('p_rate')
    if constants.plot_flag:
        plot_all_param_pairs_with_variance(recovered_param_per_exp,
                                           uuid=model_class.__name__+'/'+constants.UUID,
                                           exp_type=constants.EXP_TYPE.name,
                                           target_params=target_parameters,
                                           param_names=parameter_names,
                                           custom_title="Average inferred parameters across experiments [{}, {}]".format(
                                               model_class.__name__, constants.optimiser),
                                           logger=logger, fname='all_inferred_params_{}'.format(model_class.__name__))


def start_exp(constants, model_class, target_model=None):
    log_fname = model_class.__name__ + '_{}_{}_{}_lr_{}_batchsize_{}_trainiters_{}_rowspertrainiter_{}_uuid_{}'. \
        format(constants.optimiser.__name__, constants.loss_fn, constants.EXP_TYPE.name,
               '{:1.3f}'.format(constants.learn_rate).replace('.', '_'),
               constants.batch_size, constants.train_iters, constants.rows_per_train_iter, constants.UUID)
    logger = Log.Logger(log_fname)
    err_logger = Log.Logger('ERROR_LOG_{}'.format(log_fname))
    logger.log('Starting exp. with listed hyperparameters.', [constants.__str__()])

    run_exp_loop(logger, constants, model_class, target_model, err_logger)
