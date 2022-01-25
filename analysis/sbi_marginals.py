import sys

import numpy as np

import plot
from IO import *
from Models.microGIF import microGIF
from analysis import analysis_util
from analysis.spike_train_matlab_export import simulate_and_save_model_spike_train


def convert_posterior_to_model_params_dict(model_class, posterior_params, target_class, target_points, N):
    if posterior_params.shape[0] == 1:
        posterior_params = posterior_params[0]
    model_params = {}
    p_i = 0
    for p_name in model_class.free_parameters:
        if p_i == 0:
            assert p_name == 'w'
            model_params[p_name] = posterior_params[:(N**2-N)]
        else:
            model_params[p_name] = posterior_params[(N**2-N)+N*(p_i-1):(N**2-N)+N*p_i]
        p_i += 1

    t_p_i = 1
    for t_p_name in target_class.free_parameters:
        if t_p_name not in model_params:
            model_params[t_p_name] = target_points[(N**2-N)+N*(t_p_i-1):(N**2-N)+N*t_p_i]
        t_p_i += 1

    return model_params


def main():
    # experiments_path = '/home/william/repos/archives_snn_inference/GENERIC/archive/saved/data/'
    experiments_path = '/home/william/repos/archives_snn_inference/archive_1612/archive/saved/data/'

    files_sbi_res = os.listdir(experiments_path + 'sbi_res/')

    rates_per_model_type = { 'LIF': [], 'GLIF': [], 'microGIF': [] }
    p_dists_per_model_type = { 'LIF': [], 'GLIF': [], 'microGIF': [] }
    init_dists_per_model_type = { 'LIF': [], 'GLIF': [], 'microGIF': [] }
    target_rate_per_m_t = {}
    for sbi_res_file in files_sbi_res:
        print(sbi_res_file)

        sbi_res_path = experiments_path + 'sbi_res/' + sbi_res_file
        print('Loading: {}'.format(sbi_res_path))
        res_load = torch.load(sbi_res_path)
        print('sbi_res load successful.')

        sbi_res = res_load['data']
        assert sbi_res.keys().__contains__('SNPE'), "method SNPE expected"
        method = 'SNPE'
        posterior = sbi_res[method]
        sut_description = sbi_res['dt_descriptor']
        model_class = sbi_res['model_class']
        m_name = model_class.__name__
        N = sbi_res['N']
        dt_descriptor = sbi_res['dt_descriptor']
        tar_seed = False
        if sbi_res_file.__contains__('tar_seed'):
            tar_seed = int(sbi_res_file.split('tar_seed_')[-1].split('.pt')[0])

        if tar_seed:
            corresponding_samples_fname = 'samples_method_{}_m_name_{}_dt_{}_tar_seed_{}.pt'.format(method, m_name, dt_descriptor, tar_seed)
        else:
            corresponding_samples_fname = 'samples_method_{}_m_name_{}_dt_{}.pt'.format(method, m_name, dt_descriptor)

        data_arr = torch.load(experiments_path + 'sbi_samples/' + corresponding_samples_fname)['data']
        print('sbi_samples load successful.')

        save_fname = 'spikes_sbi_{}_tar_seed_{}'.format(corresponding_samples_fname.strip('.pt')+'', tar_seed)

        torch.manual_seed(tar_seed)
        np.random.seed(tar_seed)

        # samples = data_arr['samples']
        observation = data_arr['observation']
        # points = data_arr['tar_parameters']
        m_name = data_arr['m_name']

        # log_probability = posterior.log_prob(samples, x=observation)
        # print('log_probability: {}'.format(log_probability))

        N_samples = 10
        print('Drawing the {} most likely samples..'.format(N_samples))
        posterior_params = posterior.sample((N_samples,), x=observation)
        print('\nposterior_params: {}'.format(posterior_params))
        target_model = analysis_util.get_target_model(m_name)

        # TODO: Plot marginals.


if __name__ == "__main__":
    main()
    sys.exit()
