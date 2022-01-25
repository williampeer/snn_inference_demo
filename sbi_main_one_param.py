import sys

import torch
from sbi import analysis as analysis
from sbi import utils as utils
from sbi.inference.base import infer

import IO
from Models.no_grad.GLIF_no_grad import GLIF_no_grad
from Models.no_grad.LIF_R_ASC_no_grad import LIF_R_ASC_no_grad
from Models.no_grad.LIF_R_no_grad import LIF_R_no_grad
from Models.no_grad.LIF_no_grad import LIF_no_grad
from TargetModels.TargetModels import lif_continuous_ensembles_model_dales_compliant, \
    glif_continuous_ensembles_model_dales_compliant, lif_r_asc_continuous_ensembles_model_dales_compliant, \
    lif_r_continuous_ensembles_model_dales_compliant
from experiments import sine_modulated_white_noise_input
from model_util import feed_inputs_sequentially_return_spike_train

torch.autograd.set_detect_anomaly(True)

# data_path = data_util.prefix + data_util.path + 'target_model_spikes_GLIF_seed_4_N_3_duration_300000.mat'
# node_indices, spike_times, spike_indices = data_util.load_sparse_data(full_path=data_path)
# next_step, targets = data_util.get_spike_train_matrix(index_last_step=0, advance_by_t_steps=t_interval,
#                                                       spike_times=spike_times, spike_indices=spike_indices, node_numbers=node_indices)


def main(argv):
    NUM_WORKERS = 6

    # t_interval = 2000
    t_interval = 6000
    N = 3
    param_number = -1
    # methods = ['SNPE', 'SNLE', 'SNRE']
    # methods = ['SNPE']
    # method = None
    method = 'SNRE'
    # model_type = None
    model_type = 'GLIF'
    # budget = 40
    budget = 10000

    class_lookup = { 'LIF': LIF_no_grad, 'LIF_R': LIF_R_no_grad, 'LIF_R_ASC': LIF_R_ASC_no_grad, 'GLIF': GLIF_no_grad }

    print('Argument List:', str(argv))

    opts = [opt for opt in argv if opt.startswith("-")]
    args = [arg for arg in argv if not arg.startswith("-")]
    for i, opt in enumerate(opts):
        if opt == '-h':
            print('main.py -m <method> -N <num-neurons> -t <t-interval> -pn <param-number> -b <budget> -nw <num-workers>')
            sys.exit()
        elif opt in ("-m", "--method"):
            method = str(args[i])
        elif opt in ("-mt", "--model-type"):
            model_type = str(args[i])
        elif opt in ("-N", "--num-neurons"):
            N = int(args[i])
        elif opt in ("-t", "--t-interval"):
            t_interval = int(args[i])
        elif opt in ("-pn", "--param-number"):
            param_number = int(args[i])
        elif opt in ("-b", "--budget"):
            budget = int(args[i])
        elif opt in ("-nw", "--num-workers"):
            NUM_WORKERS = int(args[i])

    # assert param_number >= 0, "please specify a parameter to fit. (-pn || --param-number)"
    assert model_type is not None, "please specify a model type (-mt || --model-type)"
    model_class = class_lookup[model_type]
    assert param_number < len(model_class.parameter_names), \
        "param_number: {} cannot be greater than number of parameters: {} in model_class: {}" \
            .format(param_number, len(model_class.parameter_names), model_class)

    if method is not None:
        if param_number == -1:
            results = []
            for p_i in range(len(model_class.param_lin_constraints)):
                results.append(sbi(method, t_interval, N, model_class, p_i, budget, NUM_WORKERS))
            return results
        else:
            return [sbi(method, t_interval, N, model_class, param_number, budget, NUM_WORKERS)]


def sbi(method, t_interval, N, model_class, param_number, budget, NUM_WORKERS=1):
    tar_model_fn_lookup = { 'LIF_no_grad': lif_continuous_ensembles_model_dales_compliant,
                            'LIF_R_no_grad': lif_r_continuous_ensembles_model_dales_compliant,
                            'LIF_R_ASC_no_grad': lif_r_asc_continuous_ensembles_model_dales_compliant,
                            'GLIF_no_grad': glif_continuous_ensembles_model_dales_compliant }
    tar_in_rate = 10.
    tar_model_fn = tar_model_fn_lookup[model_class.__name__]
    tar_model = tar_model_fn(random_seed=42, N=N)

    def simulator(parameter_set):
        programmatic_params_dict = {}
        for i in range(len(model_class.parameter_names)):
            programmatic_params_dict[model_class.parameter_names[i]] = tar_model.get_parameters()[i]

        if param_number == 0:
            parsed_preset_weights = parameter_set
            assert len(parsed_preset_weights) == (N ** 2 - N), "len(parsed_preset_weights): {}, should be N**2-N".format(
                len(parsed_preset_weights))
            preset_weights = torch.zeros((N, N))
            ctr = 0
            for n_i in range(N):
                for n_j in range(N):
                    if (n_i != n_j):
                        preset_weights[n_i, n_j] = parsed_preset_weights[ctr]
                        ctr += 1
            programmatic_params_dict[model_class.parameter_names[param_number]] = preset_weights
        else:
            programmatic_params_dict[model_class.parameter_names[param_number]] = parameter_set

        programmatic_neuron_types = torch.ones((N,))
        for n_i in range(int(2 * N / 3), N):
            programmatic_neuron_types[n_i] = -1

        model = model_class(parameters=programmatic_params_dict, N=N, neuron_types=programmatic_neuron_types)
        inputs = sine_modulated_white_noise_input(rate=tar_in_rate, t=t_interval, N=N)
        outputs = feed_inputs_sequentially_return_spike_train(model=model, inputs=inputs)

        model.reset()
        mean_output_rates = outputs.sum(dim=0) * 1000. / outputs.shape[0]  # Hz
        return mean_output_rates

    if param_number == 0:
        num_dim = 1 + N**2
    else:
        num_dim = 1 + N

    # tar_in_rate = 10.
    # tar_model = lif_continuous_ensembles_model_dales_compliant(random_seed=42, N=N)
    inputs = sine_modulated_white_noise_input(rate=tar_in_rate, t=t_interval, N=N)
    rates = None
    for i in range(10):
        cur_targets = feed_inputs_sequentially_return_spike_train(model=tar_model, inputs=inputs).clone().detach()
        cur_rate = cur_targets.sum(dim=0) * 1000. / cur_targets.shape[0]  # Hz
        if rates is None:
            rates = cur_rate
        else:
            rates = torch.vstack((rates, cur_rate))
    targets = torch.mean(rates, dim=0)

    if param_number == 0:
        parsed_weights = torch.zeros((N ** 2 - N,))
        ctr = 0
        for n_i in range(N):
            for n_j in range(N):
                if (n_i != n_j):
                    parsed_weights[ctr] = tar_model.w[n_i, n_j]
                    ctr += 1
        tar_parameters = torch.hstack([parsed_weights])

        weights_low = torch.zeros((N**2-N,))
        weights_high = torch.ones((N**2-N,))
        limits_low = weights_low
        limits_high = weights_high
    else:
        limits_low = torch.ones((N,)) * model_class.param_lin_constraints[param_number][0]
        limits_high = torch.ones((N,)) * model_class.param_lin_constraints[param_number][1]
        tar_parameters = tar_model.get_parameters()[param_number]

    prior = utils.BoxUniform(low=limits_low, high=limits_high)

    res = {}

    posterior = infer(simulator, prior, method=method, num_simulations=budget, num_workers=NUM_WORKERS)
    res[method] = posterior
    # posterior = infer(LIF_simulator, prior, method=method, num_simulations=10)
    dt_descriptor = IO.dt_descriptor()
    res[method] = posterior
    res['model_class'] = model_class
    res['N'] = N
    res['dt_descriptor'] = dt_descriptor
    res['param_num'] = param_number
    # num_dim = N**2-N+N*(len(model_class.parameter_names)-1)
    num_dim = limits_high.shape[0]

    try:
        IO.save_data(res, 'sbi_res', description='Res from SBI using {}, dt descr: {}'.format(method, dt_descriptor),
                     fname='res_{}_mt_{}_param_num_{}_dt_{}'.format(method, model_class.__name__, param_number, dt_descriptor))

        posterior_stats(posterior, method=method, observation=torch.reshape(targets, (1, -1)), points=tar_parameters,
                        limits=torch.stack((limits_low, limits_high), dim=1), figsize=(num_dim, num_dim), budget=budget,
                        m_name=tar_model.name(), dt_descriptor=dt_descriptor, param_num=param_number)
    except Exception as e:
        print("except: {}".format(e))

    return res


def posterior_stats(posterior, method, observation, points, limits, figsize, budget, m_name, dt_descriptor, param_num):
    print('====== def posterior_stats(posterior, method={}): ====='.format(method))
    print(posterior)

    # observation = torch.reshape(targets, (1, -1))
    data_arr = {}
    samples = posterior.sample((budget,), x=observation)
    data_arr['samples'] = samples
    data_arr['observation'] = observation
    data_arr['tar_parameters'] = points
    data_arr['m_name'] = m_name

    # samples = posterior.sample((10,), x=observation)
    # log_probability = posterior.log_prob(samples, x=observation)
    IO.save_data(data_arr, 'sbi_samples',
                 description='Res from SBI using {}, dt descr: {}'.format(method, dt_descriptor),
                 fname='samples_method_{}_m_name_{}_param_num_{}_dt_{}'.format(method, m_name, param_num, dt_descriptor))

    # checking docs for convergence criterion
    # plot 100d

    try:
        if samples[0].shape[0] <= 10:
            fig, ax = analysis.pairplot(samples, points=points, limits=limits, figsize=figsize)
            if method is None:
                method = IO.dt_descriptor()
            fig.savefig('./figures/analysis_pairplot_{}_one_param_{}_{}.png'.format(method, m_name, IO.dt_descriptor()))
    except Exception as e:
        print("except: {}".format(e))


if __name__ == "__main__":
    main(sys.argv[1:])
    sys.exit(0)
