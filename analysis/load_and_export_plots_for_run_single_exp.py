import os
import sys

import numpy as np
import torch

import Log
import plot
from Constants import ExperimentType
from Models.LowerDim.GLIF_soft_lower_dim import GLIF_soft_lower_dim
from Models.Sigmoidal.GLIF_soft import GLIF_soft


def main(argv):
    print('Argument List:', str(argv))

    # opts = [opt for opt in argv if opt.startswith("-")]
    # args = [arg for arg in argv if not arg.startswith("-")]

    # experiments_path = '/Users/william/repos/archives_snn_inference/archive 14/saved/plot_data/'
    # load_path = '/Users/william/repos/archives_snn_inference/archive 13/saved/plot_data/01-22_11-04-09-239/plot_parameter_inference_trajectories_2d01-23_07-27-33-240.pt'
    # experiments_path = '/Users/william/repos/archives_snn_inference/archive 13/saved/plot_data/03-16_10-33-15-060/'
    # experiments_path = '/home/william/repos/snn_inference/saved/plot_data/05-20_11-21-27-889/'
    # experiments_path = '/home/william/repos/archives_snn_inference/archive_3008_all_seed_64_and_sbi_3_and_4/archive/saved/plot_data/08-28_01-13-47-801/'
    # experiments_path = '/home/william/repos/archives_snn_inference/archive_0909/archive/saved/plot_data/09-09_00-33-03-791/'
    # experiments_path = '/home/william/repos/archives_snn_inference/archive_0909/archive/saved/plot_data/09-08_17-26-32-607/'
    # experiments_path = '/home/william/repos/archives_snn_inference/archive_1609/archive/saved/plot_data/09-15_21-53-55-818/'
    experiments_path = '/home/william/repos/archives_snn_inference/archive_2809/archive/saved/plot_data/09-28_05-24-59-978/'

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

        # if model_type != 'LIF':  # file mask
        #     # print("Incomplete exp. len should be 5 exp * 11 plots. was: {}".format(len(plot_spiketrains_files)))
        #     # print(len(plot_spiketrains_files))
        #     pass
        # else:
        # if len(plot_spike_trains_files) == 55:
        #     for f_i in range(5):
        #         f = plot_spike_trains_files[f_i * 11]
        #         data = torch.load(experiments_path + f)
        #         print('Loaded saved plot data.')
        #
        #         plot_data = data['plot_data']
        #         plot_fn = data['plot_fn']
        #
        #         fname = f.split('/')[-1]
        #         fname = fname.split('.pt')[0].replace('.', '_')
        #         save_fname = 'export_{}.eps'.format(fname)
        #         plot.plot_spike_trains_side_by_side(plot_data['model_spikes'], plot_data['target_spikes'], 'export',
        #                                             plot_data['exp_type'], 'Spike trains (Poisson input)',
        #                                             save_fname, export=True)
        #
        #         # dev
        #         s1 = plot_data['model_spikes'].detach().numpy()
        #         s2 = plot_data['target_spikes'].detach().numpy()
        #         bin_size = 500
        #         corrs = stats.spike_train_corr_new(s1=s1, s2=s2, bin_size=bin_size)
        #         plot.heatmap_spike_train_correlations(corrs[12:, :12], axes=['Fitted model', 'Target model'],
        #                                               exp_type=plot_data['exp_type'], uuid='export',
        #                                               fname='heatmap_bin_{}_{}'.format(bin_size, save_fname),
        #                                               bin_size=bin_size)
        #         # std1, r1 = stats.binned_avg_firing_rate_per_neuron(s1, bin_size=bin_size)
        #         # std2, r2 = stats.binned_avg_firing_rate_per_neuron(s2, bin_size=bin_size)
        #         # plot.bar_plot_neuron_rates(r1, r2, std1, std2, bin_size=bin_size, exp_type=plot_data['exp_type'],
        #         #                            uuid='export',
        #         #                            fname='rate_plot_bin_{}_{}'.format(bin_size, save_fname))

    # for f in plot_all_param_pairs_with_variance_files:
    #     data = torch.load(experiments_path + f)
    #     print('Loaded saved plot data.')
    #
    #     plot_data = data['plot_data']
    #     plot_fn = data['plot_fn']
    #
    #     fname = f.split('/')[-1]
    #     fname = fname.split('.pt')[0].replace('.', '_')
    #     # save_fname = 'export_{}.eps'.format(fname)
    #     save_fname = 'export_{}.eps'.format(plot_data['fname'])
    #     print('Saving to fname: {}'.format(save_fname))
    #     print('target params', plot_data['target_params'])
    #     fixed_exp_params = {}
    #     for i in range(1,len(plot_data['param_means'])):
    #         cur_p = np.array(plot_data['param_means'][i])
    #         s = cur_p.shape
    #         assert len(s) == 3, "for reshaping length should be 3"
    #         fixed_exp_params[i-1] = np.reshape(np.array(cur_p), (s[0], s[2]))
    #
    #     tar_params = []
    #     for tar_param in plot_data['target_params']:
    #         # plot_data['target_params'][key] = [plot_data['target_params'][key]]
    #         tar_params.append([tar_param])
    #
    #     plot.plot_all_param_pairs_with_variance(param_means=fixed_exp_params, target_params=tar_params, #plot_data['target_params'],
    #                                             param_names=GLIF_soft_lower_dim.parameter_names,
    #                                             # exp_type=plot_data['exp_type'],
    #                                             exp_type=plot_data['exp_type'],
    #                                             uuid=plot_data['uuid'],
    #                                             fname='export_{}.eps'.format(save_fname),
    #                                             custom_title='',
    #                                             logger=Log.Logger('test'),
    #                                             export_flag=True)

    for f in plot_parameter_inference_trajectories_2d_files:
        data = torch.load(experiments_path + f)
        print('Loaded saved plot data.')

        plot_data = data['plot_data']
        # plot_fn = data['plot_fn']
        save_fname = 'export_param_inference_X_{}'.format(plot_data['fname'])
        print('Saving to fname: {}'.format(save_fname))

        plot.plot_parameter_inference_trajectories_2d(plot_data['param_means'], plot_data['target_params'],
                                                      GLIF_soft_lower_dim.parameter_names, plot_data['exp_type'], 'export',
                                                      save_fname, plot_data['custom_title'], Log.Logger('export'))


if __name__ == "__main__":
    main(sys.argv[1:])
