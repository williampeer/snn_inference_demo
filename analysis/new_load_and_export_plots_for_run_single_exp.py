import os
import sys

import torch

import Log
import plot
from Models.LowerDim.GLIF_soft_lower_dim import GLIF_soft_lower_dim
from Models.Sigmoidal.GLIF_soft import GLIF_soft


def transform_wrong_w_means(weights_params):
    param_means = { 'w': [] }
    # model_dim = len(weights_params['w_0'])
    for w_i, w_k in enumerate(weights_params):
        param_means['w'].append(weights_params[w_k])
    return param_means


def transform_unnamed_params(tar_param_list, target_class):
    p_names = target_class.parameter_names
    target_params = {}
    for i in range(len(tar_param_list)):
        target_params[p_names[i]] = tar_param_list[i]
    return target_params


def main(argv):
    print('Argument List:', str(argv))

    # opts = [opt for opt in argv if opt.startswith("-")]
    # args = [arg for arg in argv if not arg.startswith("-")]

    # experiments_path = '/home/william/repos/archives_snn_inference/archive_2809/archive/saved/plot_data/09-28_05-24-59-978/'
    experiments_path = '/home/william/repos/archives_snn_inference/archive_1010/saved/plot_data/10-08_14-37-41-286/'

    # for i, opt in enumerate(opts):
    #     if opt == '-h':
    #         print('load_and_export_plot_data.py -p <path>')
    #         sys.exit()
    #     elif opt in ("-p", "--path"):
    #         load_path = args[i]
    #
    # if load_path is None:
    #     print('No path to load model from specified.')
    #     sys.exit(1)

    files = os.listdir(experiments_path)
    id = experiments_path.split('-')[-1]
    plot_all_param_pairs_with_variance_files = []; plot_parameter_inference_trajectories_2d_files = []
    plot_spike_trains_files = []
    for f in files:
        for f in files:
            if f.__contains__('plot_all_param_pairs_with_variance'):
                plot_all_param_pairs_with_variance_files.append(f)
            elif f.__contains__('plot_parameter_inference_trajectories_2d'):
                plot_parameter_inference_trajectories_2d_files.append(f)
            elif f.__contains__('plot_spiketrains_side_by_side'):
                plot_spike_trains_files.append(f)
            elif f.__contains__('plot_losses'):
                f_data = torch.load(experiments_path + f)
                custom_title = f_data['plot_data']['custom_title']
                optimiser = custom_title.split(', ')[1].strip(' ')
                model_type = custom_title.split(',')[0].split('(')[-1]
                lr = custom_title.split(', ')[-1].strip(' =lr').strip(')')
                lfn = f_data['plot_data']['fname'].split('loss_fn_')[1].split('_tau')[0]
                # break

    processed_files = []
    for f in plot_parameter_inference_trajectories_2d_files:
        if f not in processed_files:
            print('processing new file: {}...'.format(f))
            data = torch.load(experiments_path + f)
            print('Loaded saved plot data.')

            plot_data = data['plot_data']
            if 'w_0' in plot_data['param_means']:
                param_means = transform_wrong_w_means(plot_data['param_means'])
            else:
                param_means = plot_data['param_means']
            if str(type(plot_data['target_params'])).__contains__('dict') and 'w' not in plot_data['target_params']:
                target_params = transform_unnamed_params(plot_data['target_params'], GLIF_soft)
            else:
                target_params = plot_data['target_params']
            # plot_fn = data['plot_fn']
            save_fname = 'export_param_inference_X_{}'.format(plot_data['fname'])
            print('Saving to fname: {}'.format(save_fname))

            plot.plot_parameter_inference_trajectories_2d(param_means, target_params,
                                                          GLIF_soft_lower_dim.parameter_names, plot_data['exp_type'], 'export',
                                                          save_fname, plot_data['custom_title'], Log.Logger('export'))
            processed_files.append(f)
        else:
            print('already processed: {}'.format(f))


if __name__ == "__main__":
    main(sys.argv[1:])
