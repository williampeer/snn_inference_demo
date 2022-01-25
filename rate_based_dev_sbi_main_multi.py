import sys

import numpy as np
import torch
from sbi import analysis as analysis
from sbi import utils as utils
from sbi.inference.base import infer

import IO
import model_util
from Models.microGIF import microGIF
from analysis.sbi_export_plots import export_plots

torch.autograd.set_detect_anomaly(True)


def transform_model_to_sbi_params(model):
    m_params = torch.zeros((model.N**2-model.N,))
    ctr = 0
    for i in range(model.w.shape[0]):
        for j in range(model.w.shape[1]):
            if i!=j:
                m_params[ctr] = model.w[i,j].clone().detach()
                ctr += 1

    model_params_list = model.get_parameters()
    for p_name in model.__class__.free_parameters:
        if p_name != 'w':
            m_params = torch.hstack((m_params, model_params_list[p_name]))
        # model_params_list[(N ** 2 - N) + N * (i - 1):(N ** 2 - N) + N * i] = [model_class.free_parameters[i]]

    return m_params


def main(argv):
    # NUM_WORKERS = 12
    NUM_WORKERS = 2

    t_interval = 4000
    N = 4
    # methods = ['SNPE', 'SNLE', 'SNRE']
    # method = None
    method = 'SNPE'
    # model_type = None
    # model_type = 'LIF'
    # model_type = 'GLIF'
    model_type = 'mesoGIF'
    # budget = 10000
    budget = 100
    rand_seed = 23

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
        elif opt in ("-b", "--budget"):
            budget = int(args[i])
        elif opt in ("-nw", "--num-workers"):
            NUM_WORKERS = int(args[i])
        elif opt in ("-rs", "--rand-seed"):
            rand_seed = int(args[i])

    assert model_type is not None, "please specify a model type (-mt || --model-type)"
    # model_class = class_lookup[model_type]

    if method is not None:
        sbi(method, t_interval, N, model_type, budget, rand_seed, NUM_WORKERS)


def sbi(method, t_interval, N, model_type_str, budget, rand_seed, NUM_WORKERS=3):
    torch.manual_seed(rand_seed)
    np.random.seed(rand_seed)

    # tar_model_fn_lookup = { 'LIF': LIF, 'GLIF': GLIF, 'microGIF': microGIF }
    GT_path = './Test/saved/'
    GT_model_by_type = {'LIF': '12-09_11-49-59-999',
                        'GLIF': '12-09_11-12-47-541',
                        'mesoGIF': '12-09_14-56-20-319' }  #,
                        # 'microGIF': '12-09_14-56-17-312'}
    GT_euid = GT_model_by_type[model_type_str]
    tar_fname = 'snn_model_target_GD_test'
    model_name = model_type_str
    if model_type_str == 'mesoGIF':
        model_name = 'microGIF'
    load_data_target = torch.load(GT_path + model_name + '/' + GT_euid + '/' + tar_fname + IO.fname_ext)
    tar_model = load_data_target['model']
    model_class = tar_model.__class__
    # tar_params = tar_model.get_parameters()

    def simulator(parameter_set):
        programmatic_params_dict = {}
        parsed_preset_weights = parameter_set[:(N**2-N)]
        assert len(parsed_preset_weights) == (N ** 2 - N), "len(parsed_preset_weights): {}, should be N**2-N".format(
            len(parsed_preset_weights))
        preset_weights = torch.zeros((N, N))
        ctr = 0
        for n_i in range(N):
            for n_j in range(N):
                if (n_i != n_j):
                    preset_weights[n_i, n_j] = parsed_preset_weights[ctr]
                    ctr += 1
        programmatic_params_dict[model_class.free_parameters[0]] = preset_weights

        for i in range(1, len(model_class.free_parameters)):
            programmatic_params_dict[model_class.free_parameters[i]] = parameter_set[(N**2-N)+N*(i-1):(N**2-N)+N*i]  # assuming only N-dimensional params otherwise

        # inputs = sine_modulated_white_noise_input(rate=tar_in_rate, t=t_interval, N=N)
        white_noise = torch.rand((t_interval, N))
        current_inputs = white_noise
        if model_class is microGIF:
            model = model_class(parameters=programmatic_params_dict, N=N)
            _, spikes, _ = model_util.feed_inputs_sequentially_return_args(model, current_inputs)
        else:
            programmatic_neuron_types = N * [1]
            n_inhib = int(N / 4)
            programmatic_neuron_types[-n_inhib:] = n_inhib * [-1]
            model = model_class(parameters=programmatic_params_dict, N=N, neuron_types=programmatic_neuron_types)
            _, spikes = model_util.feed_inputs_sequentially_return_tuple(model, current_inputs)

        model.reset()
        mean_output_rates = spikes.clone().detach().sum(dim=0) * 1000. / spikes.shape[0]  # Hz
        return mean_output_rates

    limits_low = torch.zeros((N**2-N,))
    limits_high = torch.ones((N**2-N,))

    for i in range(1, len(model_class.free_parameters)):
        limits_low = torch.hstack((limits_low, torch.ones((N,)) * model_class.param_lin_constraints[i][0]))
        limits_high = torch.hstack((limits_high, torch.ones((N,)) * model_class.param_lin_constraints[i][1]))

    prior = utils.BoxUniform(low=limits_low, high=limits_high)

    tar_sbi_params = transform_model_to_sbi_params(tar_model)
    tar_model_simulations = simulator(tar_sbi_params)

    posterior = infer(simulator, prior, method=method, num_simulations=budget, num_workers=NUM_WORKERS)
    dt_descriptor = IO.dt_descriptor()
    res = {}
    res[method] = posterior
    res['model_class'] = model_class
    res['N'] = N
    res['dt_descriptor'] = dt_descriptor
    res['tar_seed'] = rand_seed
    # num_dim = N**2-N+N*(len(model_class.free_parameters)-1)
    num_dim = limits_high.shape[0]

    try:
        IO.save_data(res, 'sbi_res', description='Res from SBI using {}, dt descr: {}'.format(method, dt_descriptor),
                     fname='res_{}_dt_{}_tar_seed_{}'.format(method, dt_descriptor, rand_seed))

        posterior_stats(posterior, method=method,
                        # observation=torch.reshape(avg_tar_model_simulations, (-1, 1)), points=tar_sbi_params,
                        observation=tar_model_simulations, points=tar_sbi_params,
                        limits=torch.stack((limits_low, limits_high), dim=1), figsize=(num_dim, num_dim), budget=budget,
                        m_name=tar_model.name(), dt_descriptor=dt_descriptor, tar_seed=rand_seed)
    except Exception as e:
        print("except: {}".format(e))

    return res


def posterior_stats(posterior, method, observation, points, limits, figsize, budget, m_name, dt_descriptor, tar_seed):
    print('====== def posterior_stats(posterior, method=None): =====')
    print(posterior)

    # observation = torch.reshape(targets, (1, -1))
    data_arr = {}
    samples = posterior.sample((budget,), x=observation, sample_with_mcmc=True)
    data_arr['samples'] = samples
    data_arr['observation'] = observation
    data_arr['tar_parameters'] = points
    data_arr['m_name'] = m_name

    # samples = posterior.sample((10,), x=observation)
    # log_probability = posterior.log_prob(samples, x=observation)
    # print('log_probability: {}'.format(log_probability))
    IO.save_data(data_arr, 'sbi_samples', description='Res from SBI using {}, dt descr: {}'.format(method, dt_descriptor),
                 fname='samples_method_{}_m_name_{}_dt_{}_tar_seed_{}'.format(method, m_name, dt_descriptor, tar_seed))

    # checking docs for convergence criterion
    # plot 100d
    try:
        # def export_plots(samples, points, lim_low, lim_high, N, method, m_name, description, model_class):
        export_plots(samples, points, limits[0], limits[1], figsize[0], method, m_name, 'sbi_export_{}'.format(dt_descriptor), m_name)
    except Exception as e:
        print('exception in new plot code: {}'.format(e))

    try:
        if samples[0].shape[0] <= 10:
            fig, ax = analysis.pairplot(samples, points=points, limits=limits, figsize=figsize)
            if method is None:
                method = dt_descriptor
            fig.savefig('./figures/analysis_pairplot_{}_one_param_{}_{}.png'.format(method, m_name, dt_descriptor))
    except Exception as e:
        print("except: {}".format(e))


if __name__ == "__main__":
    main(sys.argv[1:])
    sys.exit(0)
