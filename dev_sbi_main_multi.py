import sys

import torch
from sbi import analysis as analysis
from sbi import utils as utils
from sbi.inference.base import infer

import IO
from PDF_metrics import get_binned_spike_counts
from experiments import sine_modulated_white_noise
from model_util import feed_inputs_sequentially_return_spike_train

torch.autograd.set_detect_anomaly(True)


def transform_model_to_sbi_params(model, model_class):
    m_params = torch.zeros((model.N**2-model.N,))
    ctr = 0
    for i in range(model.w.shape[0]):
        for j in range(model.w.shape[1]):
            if i!=j:
                m_params[ctr] = model.w[i, j].clone().detach()
                ctr += 1

    model_params = model.get_parameters()
    for p_i, p_k in enumerate(model_params):
        if p_k is not 'w' and p_k in model_class.parameter_names:
            m_params = torch.hstack((m_params, model_params[p_k]))
        # model_params_list[(N ** 2 - N) + N * (i - 1):(N ** 2 - N) + N * i] = [model_class.parameter_names[i]]

    return m_params


def main(argv):
    NUM_WORKERS = 4
    # NUM_WORKERS = 1

    # t_interval = 12000
    t_interval = 4000
    # N = 4
    method = 'SNPE'
    # model_type = None
    # model_type = 'LIF'
    model_type = 'microGIF'
    # model_type = 'GLIF'
    # budget = 10000
    budget = 20
    tar_seed = 42

    # class_lookup = { 'LIF': LIF, 'GLIF': GLIF, 'microGIF': microGIF }

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
        elif opt in ("-ts", "--tar-seed"):
            tar_seed = int(args[i])

    # assert param_number >= 0, "please specify a parameter to fit. (-pn || --param-number)"
    assert model_type is not None, "please specify a model type (-mt || --model-type)"
    # model_class = class_lookup[model_type]

    if method is not None:
        sbi(method, t_interval, N, model_type, budget, tar_seed, NUM_WORKERS)


def sbi(method, t_interval, N, model_type_str, budget, tar_seed, NUM_WORKERS=5):
    # tar_model = tar_model_fn(random_seed=tar_seed, pop_size=pop_size, N_pops=N_pops)
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
    tar_model = load_data_target['model']
    model_class = tar_model.__class__

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
        programmatic_params_dict[model_class.parameter_names[0]] = preset_weights

        tar_params = tar_model.get_parameters()
        for p_i, p_k in enumerate(tar_model.get_parameters()):
            if not model_class.parameter_names.__contains__(p_k):
                programmatic_params_dict[p_k] = tar_params[p_k].clone().detach()

        for i in range(1, len(model_class.parameter_names)):
            programmatic_params_dict[model_class.parameter_names[i]] = parameter_set[(N**2-N)+N*(i-1):(N**2-N)+N*i]  # assuming only N-dimensional params otherwise

        programmatic_neuron_types = torch.ones((N,))
        for n_i in range(int(N / 2), N):
            programmatic_neuron_types[n_i] = -1

        model = model_class(parameters=programmatic_params_dict, N=N, neuron_types=programmatic_neuron_types)
        inputs = sine_modulated_white_noise(t=t_interval, N=N)
        outputs = feed_inputs_sequentially_return_spike_train(model=model, inputs=inputs)

        return torch.reshape(get_binned_spike_counts(outputs.clone().detach()), (-1,))

    limits_low = torch.zeros((N**2-N,))
    limits_high = torch.ones((N**2-N,))

    for i in range(1, len(model_class.parameter_names)):
        limits_low = torch.hstack((limits_low, torch.ones((N,)) * model_class.param_lin_constraints[i][0]))
        limits_high = torch.hstack((limits_high, torch.ones((N,)) * model_class.param_lin_constraints[i][1]))

    prior = utils.BoxUniform(low=limits_low, high=limits_high)

    tar_sbi_params = transform_model_to_sbi_params(tar_model, model_class)

    posterior = infer(simulator, prior, method=method, num_simulations=budget, num_workers=NUM_WORKERS)
    # posterior = infer(LIF_simulator, prior, method=method, num_simulations=10)
    dt_descriptor = IO.dt_descriptor()
    res = {}
    res[method] = posterior
    res['model_class'] = model_class
    res['N'] = N
    res['dt_descriptor'] = dt_descriptor
    res['tar_seed'] = tar_seed
    # num_dim = N**2-N+N*(len(model_class.parameter_names)-1)
    num_dim = limits_high.shape[0]

    # try:
    IO.save_data(res, 'sbi_res', description='Res from SBI using {}, dt descr: {}'.format(method, dt_descriptor),
                 fname='res_{}_dt_{}_tar_seed_{}'.format(method, dt_descriptor, tar_seed))

    targets = simulator(tar_sbi_params)
    posterior_stats(posterior, method=method,
                    # observation=torch.reshape(avg_tar_model_simulations, (-1, 1)), points=tar_sbi_params,
                    observation=targets, points=tar_sbi_params, model_dim=N, plot_dim=num_dim,
                    limits=torch.stack((limits_low, limits_high), dim=1), figsize=(num_dim, num_dim), budget=budget,
                    m_name=tar_model.name(), dt_descriptor=dt_descriptor, tar_seed=tar_seed, model_class=model_class)
    # except Exception as e:
    #     print("except: {}".format(e))

    return res


def posterior_stats(posterior, method, observation, points, model_dim, plot_dim, limits, figsize, budget,
                    m_name, dt_descriptor, tar_seed, model_class):
    print('====== def posterior_stats(posterior, method=None): =====')
    print(posterior)

    # observation = torch.reshape(targets, (1, -1))
    data_arr = {}
    samples = posterior.sample((budget,), x=observation)
    data_arr['samples'] = samples
    data_arr['observation'] = observation
    data_arr['tar_parameters'] = points
    data_arr['m_name'] = m_name

    IO.save_data(data_arr, 'sbi_samples', description='Res from SBI using {}, dt descr: {}'.format(method, dt_descriptor),
                 fname='samples_method_{}_m_name_{}_dt_{}_tar_seed_{}'.format(method, m_name, dt_descriptor, tar_seed))

    plot_dim = len(points)
    export_plots(samples, points, limits, model_dim, plot_dim, method, m_name, 'sbi_export_{}'.format(dt_descriptor), model_class)
    sys.exit(0)


def export_plots(samples, points, limits, model_dim, plot_dim, method, m_name, description, model_class):
    N = model_dim
    assert limits.shape[1] == 2, "limits.shape[0] should be 2. limits.shape: {}".format(limits.shape)
    lim_low = limits[:,0]
    lim_high = limits[:,1]
    weights_offset = N ** 2 - N

    # WEIGHTS
    cur_mean_limits = torch.stack((torch.zeros((N,)), torch.ones((N,))))
    cur_pt = points[:weights_offset]
    cur_samples = samples[:, :weights_offset]

    weights_mean = torch.tensor([])
    tar_ws_mean = torch.tensor([])
    for n_i in range(N):
        # for w_i in range(N-1):
        weights_mean = torch.hstack([weights_mean, torch.reshape(torch.mean(cur_samples[:, n_i*(N-1):(n_i+1)*(N-1)], axis=-1), (-1, 1))])
        tar_ws_mean = torch.hstack([tar_ws_mean, torch.reshape(torch.mean(cur_pt[n_i*(N-1):(n_i+1)*(N-1)], axis=-1), (-1, 1))])

    fig_subset_mean, ax_mean = analysis.pairplot(weights_mean, points=tar_ws_mean, limits=cur_mean_limits.T, figsize=(N, N))
    path = './figures/sbi/{}/{}/'.format(m_name, description)
    IO.makedir_if_not_exists(path)
    fname = 'export_sut_subset_analysis_pairplot_{}_{}_weights_{}.png'.format(method, m_name, description)
    fig_subset_mean.savefig(path + fname)

    # Marginals only for p_i, p_i
    for p_i in range(1, len(model_class.parameter_names)):
        cur_mean_limits = torch.stack((lim_low[weights_offset+(p_i-1)*N:weights_offset+p_i*N], lim_high[weights_offset+(p_i-1)*N:weights_offset+p_i*N]))
        cur_pt = points[weights_offset+(p_i-1)*N:weights_offset+p_i*N]
        cur_samples = samples[:, weights_offset+(p_i-1)*N:weights_offset+p_i*N]
        fig_subset_mean, ax_mean = analysis.pairplot(cur_samples, points=cur_pt, limits=cur_mean_limits.T, figsize=(N, N))
        path = './figures/sbi/{}/{}/'.format(m_name, description)
        IO.makedir_if_not_exists(path)
        fname = 'export_sut_subset_analysis_pairplot_{}_{}_one_param_{}_{}.png'.format(method, m_name, p_i, description)
        fig_subset_mean.savefig(path + fname)


if __name__ == "__main__":
    main(sys.argv[1:])
    sys.exit(0)
