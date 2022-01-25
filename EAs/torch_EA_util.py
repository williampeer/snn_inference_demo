import sys

import nevergrad as ng
import numpy as np
import torch

import IO
import ext_spike_metrics
from Log import Logger
from Models.GLIF import GLIF
from Models.LIF import LIF
from TargetModels import TargetEnsembleModels
from experiments import zip_dicts, draw_from_uniform, generate_synthetic_data
from plot import plot_all_param_pairs_with_variance, plot_spike_trains_side_by_side


logger = Logger(log_fname='torch_sparse_train_single_objective_optim')


def get_instrum_for(model_type, target_rate, N, loss_fn, loss_metric, target_model, time_interval):
    w_mean = 0.3; w_var = 0.5; rand_ws = (w_mean - w_var) + 2 * w_var * np.random.random((N, N))

    if model_type == 'GLIF':
        init_params = draw_from_uniform(GLIF.parameter_init_intervals, N)
        return ng.p.Instrumentation(rate=ng.p.Scalar(init=target_rate).set_bounds(1., 40.),
                                    w=ng.p.Array(init=rand_ws).set_bounds(-1., 1.),
                                    E_L=ng.p.Array(init=init_params['E_L']).set_bounds(-80., -35.),
                                    C_m=ng.p.Array(init=init_params['C_m']).set_bounds(1.15, 2.),
                                    G=ng.p.Array(init=init_params['G']).set_bounds(0.1, 0.9),
                                    R_I=ng.p.Array(init=init_params['R_I']).set_bounds(90., 150.),
                                    f_v=ng.p.Array(init=init_params['f_v']).set_bounds(0.01, 0.99),
                                    f_I=ng.p.Array(init=init_params['f_I']).set_bounds(0.01, 0.99),

                                    delta_theta_s=ng.p.Array(init=init_params['delta_theta_s']).set_bounds(6., 30.),
                                    b_s=ng.p.Array(init=init_params['b_s']).set_bounds(0.01, 0.9),
                                    a_v=ng.p.Array(init=init_params['a_v']).set_bounds(0.01, 0.9),
                                    b_v=ng.p.Array(init=init_params['b_v']).set_bounds(0.01, 0.9),
                                    theta_inf=ng.p.Array(init=init_params['theta_inf']).set_bounds(-25., 0.),
                                    delta_V=ng.p.Array(init=init_params['delta_V']).set_bounds(0.01, 35.),
                                    I_A=ng.p.Array(init=init_params['I_A']).set_bounds(0.5, 4.),

                                    loss_fn=loss_fn, loss_metric=loss_metric, target_model=target_model,
                                    target_rate=target_rate, time_interval=time_interval)
    elif model_type == 'LIF':
        init_params = draw_from_uniform(LIF.parameter_init_intervals, N)
        return ng.p.Instrumentation(rate=ng.p.Scalar(init=target_rate).set_bounds(1., 40.),
                                    w=ng.p.Array(init=rand_ws).set_bounds(-1., 1.),
                                    E_L=ng.p.Array(init=init_params['E_L']).set_bounds(-80., -35.),
                                    C_m=ng.p.Array(init=init_params['C_m']).set_bounds(1.15, 3.),
                                    tau_g=ng.p.Array(init=init_params['C_m']).set_bounds(1.15, 4.),
                                    R_I=ng.p.Array(init=init_params['R_I']).set_bounds(90., 150.),

                                    loss_fn=loss_fn, loss_metric=loss_metric, target_model=target_model,
                                    target_rate=target_rate, time_interval=time_interval)
    else:
        raise NotImplementedError("model_type not implemented.")


def get_loss_sparse_spike_trains(model_spike_train, targets, loss_fn, N, time, tau_vr=4.0, metric='isi'):
    if loss_fn.__contains__('get_pymuvr_dist'):
        loss = ext_spike_metrics.get_pymuvr_dist(model_spike_train, targets, num_nodes=N)
    elif loss_fn.__contains__('get_label_free_isi_dist_sparse'):
        loss = ext_spike_metrics.get_label_free_isi_dist_sparse(model_spike_train, targets, num_nodes=N, time=time)
    elif loss_fn.__contains__('spk_dist_sparse'):
        loss = ext_spike_metrics.get_label_free_spk_dist_sparse(metric, model_spike_train, targets, num_nodes=N, time=time)

    return loss


def pytorch_run_sim_sparse_LIF(rate, w, C_m, tau_g, R_I, E_L, loss_fn, target_model, target_rate, loss_metric, time_interval=4000):
    model = LIF({ 'C_m': np.array(C_m, dtype='float32'),
                  'tau_g': np.array(tau_g, dtype='float32'),
                  'R_I': np.array(R_I, dtype='float32'),
                  'E_L': np.array(E_L, dtype='float32'),
                  'preset_weights': np.array(w, dtype='float32')})

    # model_spikes = generate_sparse_data(model, rate, time_interval)
    # target_spikes = generate_sparse_data(target_model, target_rate, time_interval)
    model_spike_train = generate_synthetic_data(model, rate, time_interval).numpy()
    target_spike_train = generate_synthetic_data(target_model, target_rate, time_interval).numpy()

    # simple conversion to not change structure back and forth
    model_spikes = []
    target_spikes = []
    for i in range(model_spike_train.shape[1]):
        model_spikes.append(np.arange(0, model_spike_train.shape[0])[model_spike_train[:, i] == 1])
        target_spikes.append(np.arange(0, target_spike_train.shape[0])[target_spike_train[:, i] == 1])

    loss = get_loss_sparse_spike_trains(model_spikes, target_spikes, loss_fn=loss_fn, N=w.shape[0], time=time_interval,
                                        metric=loss_metric)
    logger.log('loss_fn: {}, loss: {:3.3f}'.format(loss_fn, loss))
    return loss


def pytorch_run_sim_sparse_GLIF(rate, w, C_m, G, R_I, f_v, f_I, E_L, b_s, b_v, a_v, delta_theta_s, delta_V, theta_inf,
                                I_A, loss_fn, target_model, target_rate, loss_metric, time_interval=4000):

    model = GLIF({'f_I': np.array(f_I, dtype='float32'), 'C_m': np.array(C_m, dtype='float32'),
                  'G': np.array(G, dtype='float32'),
                  'R_I': np.array(R_I, dtype='float32'), 'f_v': np.array(f_v, dtype='float32'),
                  'E_L': np.array(E_L, dtype='float32'),
                  'b_s': np.array(b_s, dtype='float32'), 'b_v': np.array(b_v, dtype='float32'),
                  'a_v': np.array(a_v, dtype='float32'),
                  'delta_theta_s': np.array(delta_theta_s, dtype='float32'),
                  'delta_V': np.array(delta_V, dtype='float32'),
                  'theta_inf': np.array(theta_inf, dtype='float32'), 'I_A': np.array(I_A, dtype='float32'),
                  'preset_weights': np.array(w, dtype='float32')})

    # model_spikes = generate_sparse_data(model, rate, time_interval)
    # target_spikes = generate_sparse_data(target_model, target_rate, time_interval)
    model_spike_train = generate_synthetic_data(model, rate, time_interval).numpy()
    target_spike_train = generate_synthetic_data(target_model, target_rate, time_interval).numpy()

    # simple conversion to not change structure back and forth
    model_spikes = []
    target_spikes = []
    for i in range(model_spike_train.shape[1]):
        model_spikes.append(np.arange(0, model_spike_train.shape[0])[model_spike_train[:, i] == 1])
        target_spikes.append(np.arange(0, target_spike_train.shape[0])[target_spike_train[:, i] == 1])

    loss = get_loss_sparse_spike_trains(model_spikes, target_spikes, loss_fn=loss_fn, N=w.shape[0], time=time_interval,
                                        metric=loss_metric)
    logger.log('loss_fn: {}, loss: {:3.3f}'.format(loss_fn, loss))
    return loss


def main(argv):
    print('Argument List:', str(argv))

    num_exps = 5; budget = 5000
    # num_exps = 4; budget = 400
    optim_name = 'CMA'
    # optim_name = 'NGO'
    loss_fn = 'spk_dist_sparse'; loss_metric = 'spike'
    target_rate = 10.; time_interval = 2000
    random_seed = 2
    model_type = 'LIF'

    logger = Logger(log_fname='nevergrad_optimization_{}_{}_budget_{}_{}'.format(model_type, optim_name, budget, loss_fn))

    opts = [opt for opt in argv if opt.startswith("-")]
    args = [arg for arg in argv if not arg.startswith("-")]
    for i, opt in enumerate(opts):
        if opt == '-h':
            print('run_single_objective_network_optim.py -b <budget> -ne <num-experiments> -o <optim>')
            sys.exit()
        elif opt in ("-b", "--budget"):
            budget = int(args[i])
        elif opt in ("-ne", "--num-experiments"):
            num_exps = int(args[i])
        elif opt in ("-o", "--optim"):
            optim_name = args[i]
        elif opt in ("-lfn", "--loss-function"):
            loss_fn = args[i]
        elif opt in ("-lm", "--loss-metric"):
            loss_metric = args[i]
        elif opt in ("-rs", "--random-seed"):
            random_seed = int(args[i])
        elif opt in ("-mt", "--model-type"):
            model_type = args[i]

    if optim_name == 'DE':
        optim = ng.optimizers.DE
    elif optim_name == 'CMA':
        optim = ng.optimizers.CMA
    elif optim_name == 'PSO':
        optim = ng.optimizers.PSO
    elif optim_name == 'NGO':
        optim = ng.optimizers.NGO
    else:
        raise NotImplementedError()

    if model_type == 'LIF':
        model_class = LIF
    elif model_type == 'GLIF':
        model_class = GLIF
    else:
        raise NotImplementedError()

    # for random_seed in range(1,6):
    target_model_name = 'lif_ensembles_model_dales_compliant_rand_seed_{}'.format(random_seed)
    target_model = TargetEnsembleModels.lif_ensembles_model_dales_compliant(random_seed=random_seed)

    logger.log('Target model: {}'.format(target_model_name))
    target_parameters = {}
    index_ctr = 0
    for param_i, key in enumerate(target_model.state_dict()):
        if key not in ['loss_fn', 'rate', 'w']:
            target_parameters[index_ctr] = [target_model.state_dict()[key].clone().detach().numpy()]
            index_ctr += 1

    # --------------------
    params_by_optim = {}
    UUID = IO.dt_descriptor()
    current_plottable_params_for_optim = {}
    other_params_for_optim = {}
    min_loss_per_exp = []
    for exp_i in range(num_exps):
        N = 12
        instrum = get_instrum_for(model_type, target_rate, N, loss_fn, loss_metric, target_model, time_interval)

        optimizer = optim(parametrization=instrum, budget=budget)

        logger.log('setup experiment with the optimizer {}'.format(optimizer.__str__()))

        if model_type == 'LIF':
            recommendation = optimizer.minimize(pytorch_run_sim_sparse_LIF)
        elif model_type == 'GLIF':
            recommendation = optimizer.minimize(pytorch_run_sim_sparse_GLIF)
        else:
            raise NotImplementedError()

        recommended_params = recommendation.value[1]

        logger.log('recommendation.value: {}'.format(recommended_params))

        cur_plot_params = {}  # TODO: fix spaghetti
        index_ctr = 0
        for p_i, key in enumerate(recommended_params):
            if key in ['target_model', 'target_rate', 'time_interval']:
                pass
            elif key not in ['loss_fn', 'rate', 'w']:
                if exp_i == 0:
                    current_plottable_params_for_optim[index_ctr] = [np.copy(recommended_params[key])]
                else:
                    current_plottable_params_for_optim[index_ctr].append(np.copy(recommended_params[key]))
                cur_plot_params[key] = np.copy(recommended_params[key])
                index_ctr += 1
            else:
                if exp_i == 0:
                    other_params_for_optim[key] = [np.copy(recommended_params[key])]
                else:
                    other_params_for_optim[key].append(np.copy(recommended_params[key]))

        # model_spike_train = get_spike_train_for(recommended_params['rate'], zip_dicts(cur_plot_params.copy(), {'preset_weights': other_params_for_optim['w'][exp_i]}))
        # targets = generate_synthetic_data(target_model, target_rate, time_interval)

        m_params = zip_dicts(cur_plot_params.copy(), {'preset_weights': other_params_for_optim['w'][exp_i]})
        for key in m_params.keys():
            m_params[key] = np.array(m_params[key], dtype='float32')
        cur_model = model_class(m_params)
        model_spike_train = generate_synthetic_data(cur_model, recommended_params['rate'], time_interval).numpy()
        target_spike_train = generate_synthetic_data(target_model, target_rate, time_interval).numpy()

        # simple conversion to not change structure back and forth
        model_spikes = []
        target_spikes = []
        for i in range(model_spike_train.shape[1]):
            model_spikes.append(np.arange(0, model_spike_train.shape[0])[model_spike_train[:, i] == 1])
            target_spikes.append(np.arange(0, target_spike_train.shape[0])[target_spike_train[:, i] == 1])

        # min_loss_per_exp.append(recommendation.loss)  # currently doesn't work..
        # cur_min_loss = calculate_loss(model_spike_train, targets, loss_fn, tau_vr=4.0).clone().detach().numpy()
        cur_min_loss = get_loss_sparse_spike_trains(model_spike_train, target_spikes, loss_fn, N, time=time_interval,
                                                    tau_vr=4.0, metric=loss_metric)
        min_loss_per_exp.append(cur_min_loss)

        plot_spike_trains_side_by_side(model_spike_train, target_spike_train, exp_type='single_objective_optim', uuid=UUID,
                                       title='Spike trains model and target ({}, loss: {:.2f})'.format(optim_name, cur_min_loss),  #recommendation.loss),
                                       fname='spike_trains_optim_{}_exp_num_{}'.format(optim_name, exp_i))

        torch.save(recommended_params.copy(),
                   './saved/single_objective_optim/fitted_params_{}_optim_{}_loss_fn_{}_budget_{}_exp_{}.pt'.format(
                       target_model_name, optim_name, loss_fn, budget, exp_i))
        cur_min_loss = None; model_spike_train = None; targets = None

    params_by_optim[optim_name] = zip_dicts(current_plottable_params_for_optim, other_params_for_optim)
    torch.save(params_by_optim, './saved/single_objective_optim/params_tm_{}_by_optim_{}_loss_fn_{}_budget_{}.pt'.format(target_model_name, optim_name, loss_fn, budget))
    torch.save(min_loss_per_exp, './saved/single_objective_optim/min_losses_tm_{}_optim_{}_loss_fn_{}_budget_{}.pt'.format(target_model_name, optim_name, loss_fn, budget))

    plot_all_param_pairs_with_variance(current_plottable_params_for_optim,
                                       exp_type='single_objective_optim',
                                       uuid=UUID,
                                       target_params=target_parameters,
                                       param_names=list(recommended_params.keys())[2:],
                                       custom_title="KDE projection of 2D model parameter".format(optim_name),
                                       logger=logger, fname='single_objective_KDE_optim_{}_target_model_{}'.format(optim_name, target_model_name))


if __name__ == "__main__":
    main(sys.argv[1:])
    sys.exit(0)
