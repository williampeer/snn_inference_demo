import sys

import nevergrad as ng
import numpy as np
import torch

import IO
from Log import Logger
from Models.Unbounded.GLIF_unbounded import GLIF_unbounded
from Models.Unbounded.LIF_unbounded import LIF_unbounded
from TargetModels import TargetEnsembleModels
from eval import calculate_loss
from experiments import zip_dicts, draw_from_uniform, generate_synthetic_data, release_computational_graph
from plot import plot_all_param_pairs_with_variance, plot_spike_trains_side_by_side


logger = Logger(log_fname='torch_EA_single_objective_GLIF_v3')


def get_instrum_for(model_type, target_rate, N, target_model, time_interval, loss_fn='frd', tau_vr=100.0):
    inhib_mask = np.ones((12, 12))
    inhib_mask[8:,:] = -1
    rand_ws = np.random.random((N, N)) * inhib_mask

    if model_type == 'GLIF':
        init_params = draw_from_uniform(GLIF_unbounded.parameter_init_intervals, N)
        return ng.p.Instrumentation(rate=ng.p.Scalar(init=float(target_rate)).set_bounds(1., 40.),
                                    w=ng.p.Array(init=rand_ws).set_bounds(-1., 1.),
                                    E_L=ng.p.Array(init=init_params['E_L']).set_bounds(-75., -40.),
                                    tau_m=ng.p.Array(init=init_params['tau_m']).set_bounds(1.1, 3.),
                                    G=ng.p.Array(init=init_params['G']).set_bounds(0.1, 0.95),
                                    R_I=ng.p.Array(init=init_params['R_I']).set_bounds(40., 60.),
                                    f_v=ng.p.Array(init=init_params['f_v']).set_bounds(0.01, 0.99),
                                    f_I=ng.p.Array(init=init_params['f_I']).set_bounds(0.01, 0.99),

                                    delta_theta_s=ng.p.Array(init=init_params['delta_theta_s']).set_bounds(6., 30.),
                                    b_s=ng.p.Array(init=init_params['b_s']).set_bounds(0.01, 0.9),
                                    a_v=ng.p.Array(init=init_params['a_v']).set_bounds(0.01, 0.9),
                                    b_v=ng.p.Array(init=init_params['b_v']).set_bounds(0.01, 0.9),
                                    theta_inf=ng.p.Array(init=init_params['theta_inf']).set_bounds(-20., -10.),
                                    delta_V=ng.p.Array(init=init_params['delta_V']).set_bounds(1., 35.),
                                    I_A=ng.p.Array(init=init_params['I_A']).set_bounds(0.5, 3.),

                                    target_model=target_model,
                                    target_rate=target_rate, time_interval=time_interval,
                                    loss_fn=loss_fn, tau_vr=tau_vr)
    elif model_type == 'LIF':
        init_params = draw_from_uniform(LIF_unbounded.parameter_init_intervals, N)
        return ng.p.Instrumentation(rate=ng.p.Scalar(init=float(target_rate)).set_bounds(1., 40.),
                                    w=ng.p.Array(init=rand_ws).set_bounds(-1., 1.),
                                    E_L=ng.p.Array(init=init_params['E_L']).set_bounds(-80., -35.),
                                    tau_m=ng.p.Array(init=init_params['tau_m']).set_bounds(1.1, 3.),
                                    tau_g=ng.p.Array(init=init_params['tau_g']).set_bounds(1.5, 3.5),
                                    R_I=ng.p.Array(init=init_params['R_I']).set_bounds(100., 155.),

                                    target_model=target_model,
                                    target_rate=target_rate, time_interval=time_interval,
                                    loss_fn=loss_fn, tau_vr=tau_vr)
    else:
        raise NotImplementedError("model_type not implemented.")


def get_loss(model_spike_train, targets, N, loss_fn='frd', tau_vr=100.0):
    loss = calculate_loss(output=model_spike_train, target=targets, loss_fn=loss_fn, N=N, tau_vr=tau_vr).clone().detach().requires_grad_(False).data
    logger.log('loss_fn: {}, loss: {}'.format(loss_fn, loss))
    return np.float(loss)


def pytorch_run_LIF(rate, w, tau_m, tau_g, R_I, E_L, target_model, target_rate, time_interval=4000, loss_fn='frd', tau_vr=100.0):
    model = LIF_unbounded({ 'tau_m': np.array(tau_m, dtype='float32'),
                            'tau_g': np.array(tau_g, dtype='float32'),
                            'R_I': np.array(R_I, dtype='float32'),
                            'E_L': np.array(E_L, dtype='float32'),
                            'preset_weights': np.array(w, dtype='float32')})

    model_spike_train, _ = generate_synthetic_data(model, rate, time_interval)
    target_spike_train, _ = generate_synthetic_data(target_model, target_rate, time_interval)

    release_computational_graph(model, rate, model_spike_train)
    release_computational_graph(target_model, target_rate, target_spike_train)

    return get_loss(model_spike_train, target_spike_train, N=model_spike_train.shape[0], loss_fn=loss_fn)


def pytorch_run_GLIF(rate, w, tau_m, G, R_I, f_v, f_I, E_L, b_s, b_v, a_v, delta_theta_s, delta_V, theta_inf,
                     I_A, target_model, target_rate, time_interval=4000, loss_fn='frd', tau_vr=100.0):

    model = GLIF_unbounded({'f_I': np.array(f_I, dtype='float32'), 'tau_m': np.array(tau_m, dtype='float32'),
                            'G': np.array(G, dtype='float32'),
                            'R_I': np.array(R_I, dtype='float32'), 'f_v': np.array(f_v, dtype='float32'),
                            'E_L': np.array(E_L, dtype='float32'),
                            'b_s': np.array(b_s, dtype='float32'), 'b_v': np.array(b_v, dtype='float32'),
                            'a_v': np.array(a_v, dtype='float32'),
                            'delta_theta_s': np.array(delta_theta_s, dtype='float32'),
                            'delta_V': np.array(delta_V, dtype='float32'),
                            'theta_inf': np.array(theta_inf, dtype='float32'), 'I_A': np.array(I_A, dtype='float32'),
                            'preset_weights': np.array(w, dtype='float32')})

    model_spike_train, _ = generate_synthetic_data(model, rate, time_interval)
    target_spike_train, _ = generate_synthetic_data(target_model, target_rate, time_interval)

    release_computational_graph(model, rate, model_spike_train)
    release_computational_graph(target_model, target_rate, target_spike_train)

    return get_loss(model_spike_train, target_spike_train, N=model_spike_train.shape[0], loss_fn=loss_fn)


def main(argv):
    print('Argument List:', str(argv))

    num_exps = 5; budget = 8000
    # num_exps = 3; budget = 5
    optim_name = 'CMA'
    # optim_name = 'NGO'
    # optim_name = 'DE'
    # optim_name = 'PSO'
    target_rate = 10.; time_interval = 1600
    model_type = 'LIF'
    loss_fn = 'frd'
    tau_vr = 100.0

    logger = Logger(log_fname='single_objective_optimization_{}_v3_short_optim_{}_budget_{}'.format(model_type, optim_name, budget))

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
        elif opt in ("-mt", "--model-type"):
            model_type = args[i]
        elif opt in ("-lfn", "--loss-fn"):
            loss_fn = args[i]
        elif opt in ("-tvr", "--tau-vr"):
            tau_vr = float(args[i])

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
        model_class = LIF_unbounded
    elif model_type == 'GLIF':
        model_class = GLIF_unbounded
    else:
        raise NotImplementedError()

    # --------------------
    params_by_optim = {}
    UUID = IO.dt_descriptor()
    current_plottable_params_for_optim = {}
    other_params_for_optim = {}
    min_loss_per_exp = []
    for target_seed in range(4):
        if model_type == 'LIF':
            target_model_name = 'lif_ensembles_model_dales_compliant_rand_seed_{}'.format(target_seed)
            target_model = TargetEnsembleModels.lif_ensembles_model_dales_compliant(random_seed=target_seed)
        elif model_type == 'GLIF':
            target_model_name = 'glif_ensembles_model_dales_compliant_rand_seed_{}'.format(target_seed)
            target_model = TargetEnsembleModels.glif_ensembles_model_dales_compliant(random_seed=target_seed)
        else:
            raise NotImplementedError("Target model not found.")
        logger.log('Target model: {}'.format(target_model_name))
        target_parameters = {}
        index_ctr = 0
        for param_i, key in enumerate(target_model.state_dict()):
            if key not in ['rate', 'w', 'loss_fn', 'tau_vr']:
                target_parameters[index_ctr] = [target_model.state_dict()[key].clone().detach().numpy()]
                index_ctr += 1

        for exp_i in range(num_exps):
            N = 12
            instrum = get_instrum_for(model_type, target_rate, N, target_model, time_interval, loss_fn, tau_vr)

            optimizer = optim(parametrization=instrum, budget=budget)

            logger.log('setup experiment with the optimizer {}'.format(optimizer.__str__()))

            if model_type == 'LIF':
                recommendation = optimizer.minimize(pytorch_run_LIF)
            elif model_type == 'GLIF':
                recommendation = optimizer.minimize(pytorch_run_GLIF)
            else:
                raise NotImplementedError()

            recommended_params = recommendation.value[1]

            logger.log('recommendation.value: {}'.format(recommended_params))

            cur_plot_params = {}  # TODO: simplify following implementation
            index_ctr = 0
            for p_i, key in enumerate(recommended_params):
                if key in ['target_model', 'target_rate', 'time_interval', 'loss_fn', 'tau_vr']:
                    pass
                elif key not in ['rate', 'w']:
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

            m_params = zip_dicts(cur_plot_params.copy(), {'preset_weights': other_params_for_optim['w'][exp_i]})
            for key in m_params.keys():
                m_params[key] = np.array(m_params[key], dtype='float32')
            cur_model = model_class(m_params)
            model_spike_train, _ = generate_synthetic_data(cur_model, recommended_params['rate'], time_interval)
            target_spike_train, _ = generate_synthetic_data(target_model, target_rate, time_interval)

            # min_loss_per_exp.append(recommendation.loss)  # currently doesn't work..
            cur_min_loss = get_loss(model_spike_train, target_spike_train, N)
            min_loss_per_exp.append(cur_min_loss)

            release_computational_graph(cur_model, rate_parameter=recommended_params['rate'])
            release_computational_graph(target_model, rate_parameter=target_rate)

            plot_spike_trains_side_by_side(model_spike_train, target_spike_train, exp_type='single_objective_optim', uuid=UUID,
                                           title='Spike trains (Gv2) model and target ({}, loss: {:.2f})'.format(optim_name, cur_min_loss),  #recommendation.loss),
                                           fname='spike_trains_{}_optim_{}_exp_num_{}_GLIF_v2'.format(model_type, optim_name, exp_i))

            torch.save(recommended_params.copy(),
                       './saved/single_objective_optim/fitted_params_{}_optim_{}_budget_{}_exp_{}.pt'.format(
                           target_model_name, optim_name, budget, exp_i))
            del instrum, optimizer, cur_plot_params, m_params, cur_model, model_spike_train, target_spike_train

        params_by_optim[optim_name] = zip_dicts(current_plottable_params_for_optim, other_params_for_optim)
        torch.save(params_by_optim, './saved/single_objective_optim/params_tm_{}_by_optim_{}__budget_{}.pt'.format(target_model_name, optim_name, budget))
        torch.save(min_loss_per_exp, './saved/single_objective_optim/min_losses_tm_{}_optim_{}_budget_{}.pt'.format(target_model_name, optim_name, budget))

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
