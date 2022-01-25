import os
import sys

import numpy as np
import torch

import Log
import plot
import stats
from Constants import ExperimentType
from Models.GLIF import GLIF
from Models.LIF import LIF
from Models.microGIF import microGIF


def export_loss(fname, path, model_type_str, euid):
    load_data = torch.load(path + '/' + fname)
    plot_data = load_data['plot_data']
    custom_title = plot_data['custom_title']
    lfn = custom_title.split(',')[0].strip('Loss ')
    lr = custom_title.split(',')[1].split('=')[-1]
    optimiser = custom_title.split(',')[2]
    # bin_size = custom_title.split(',')[3].split('=')[-1]
    # data = {'loss': loss, 'exp_type': exp_type, 'custom_title': custom_title, 'fname': fname}
    plot.plot_loss(plot_data['loss'], 'export_{}/{}'.format(model_type_str, euid), plot_data['exp_type'],
                   fname='export_{}_plot_loss_euid_{}'.format(model_type_str, euid) + '.eps',
                   ylabel='${}$ loss'.format(lfn.replace('_nll', '$ $nll')))


def export_spike_trains(fname, path, model_type_str, euid):
    load_data = torch.load(path + '/' + fname)
    plot_data = load_data['plot_data']
    # data = {'model_spikes': model_spikes, 'target_spikes': target_spikes, 'exp_type': exp_type, 'title': title, 'fname': fname}
    plot.plot_spike_trains_side_by_side(plot_data['model_spikes'], plot_data['target_spikes'],
                                        'export_{}/{}'.format(model_type_str, euid),
                                        plot_data['exp_type'], fname='export_spike_trains_euid_{}'.format(euid) + '.eps')


def export_param_traject_plot(fname, path, model_class, euid):
    data = torch.load(path + '/' + fname)
    print('Loaded saved plot data.')

    plot_data = data['plot_data']
    # data = {'param_means': param_means, 'target_params': target_params, 'exp_type': exp_type, 'uuid': uuid, 'custom_title': custom_title, 'fname': fname}

    plot.plot_parameter_inference_trajectories_2d(plot_data['param_means'], plot_data['target_params'],
                                                  model_class.free_parameters, plot_data['exp_type'],
                                                  'export_{}/{}'.format(model_class.__name__, euid),
                                                  'export_param_inference_X_{}.eps'.format(model_class.__name__, euid),
                                                  plot_data['custom_title'])


def main():
    # experiments_path = '/media/william/p6/archive_14122021/archive/saved/plot_data/sleep_data_no_types/'
    # experiments_path = '/home/william/repos/snn_inference/Test/saved/plot_data/GT/'  # EXPORTED
    experiments_path = '/home/william/repos/snn_inference/Test/saved/plot_data/'  # EXPORTED
    # model_types = ['LIF', 'GLIF', 'microGIF']
    # model_types = ['LIF', 'GLIF']
    model_types = ['microGIF']
    model_class_lookup = { 'LIF': LIF, 'GLIF': GLIF, 'microGIF': microGIF }
    for model_type_str in model_types:
        # print(folder_path)
        full_path = experiments_path + model_type_str + '/'
        exp_folders = os.listdir(full_path)

        for exp_folder in exp_folders:
            exp_path = full_path + exp_folder

            programmatic_loss_file = './figures/GD_test/export_{}/'.format(model_type_str) + 'export_{}_plot_loss_test{}{}.eps'.format(model_type_str, exp_folder, exp_folder)
            if not os.path.exists(programmatic_loss_file):
                files = os.listdir(exp_path)
                inference_traject_files = []; spike_trains_files = []
                for f in files:
                    if f.__contains__('plot_parameter_inference_trajectories_2d'):
                        inference_traject_files.append(f)
                    elif f.__contains__('plot_spiketrains_side_by_side'):
                        spike_trains_files.append(f)
                    elif f.__contains__('plot_loss'):
                        export_loss(f, exp_path, model_type_str, exp_folder)

                inference_traject_files.sort()
                spike_trains_files.sort()
                model_class = model_class_lookup[model_type_str]
                if len(inference_traject_files)>0:
                    export_param_traject_plot(inference_traject_files[-1], exp_path, model_class, exp_folder)
                if len(spike_trains_files)>0:
                    export_spike_trains(spike_trains_files[-1], exp_path, model_type_str, exp_folder)
            else:
                print('exp: {} already processed / has plot loss file. continuing on to next exp folder..'.format(exp_folder))


if __name__ == "__main__":
    main()
    sys.exit()
