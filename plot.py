import os

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, FuncFormatter
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
import matplotlib.cm as cm
from scipy.stats import gaussian_kde

import IO
import data_util

plt.rcParams.update({'font.size': 12})


def plot_spike_train(spike_train, title, uuid, exp_type='default', fname='spiketrain_test'):

    data = {'spike_history': spike_train, 'title': title, 'fname': fname}
    IO.save_plot_data(data=data, uuid=uuid, plot_fn='plot_spiketrain')

    plt.figure()
    # assuming neuron indices to be columns and reshaping to rows for plotting
    time_indices = torch.reshape(torch.arange(spike_train.shape[0]), (spike_train.shape[0], 1))
    # ensure binary values:
    spike_train = torch.round(spike_train)
    neuron_spike_times = spike_train * time_indices.float()

    for neuron_i in range(spike_train.shape[1]):
        if neuron_spike_times[:, neuron_i].nonzero().sum() > 0:
            plt.plot(torch.reshape(neuron_spike_times[:, neuron_i].nonzero(), (1, -1)).numpy(),
                     neuron_i+1, '.k', markersize=4.0)

    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron')
    plt.yticks(range(neuron_i+2))
    plt.title(title)

    full_path = './figures/' + exp_type + '/' + uuid + '/'
    # IO.makedir_if_not_exists('/figures/' + exp_type + '/')
    IO.makedir_if_not_exists(full_path)
    plt.savefig(fname=full_path + fname)
    # plt.show()
    plt.close()


def plot_spike_trains_side_by_side(model_spikes, target_spikes, uuid, exp_type='default', title=False, fname=False, legend=None, export=False):
    assert model_spikes.shape[0] > model_spikes.shape[1], \
        "assert one node per column, one bin per row. spikes shape: {}".format(model_spikes.shape)
    assert model_spikes.shape[0] == target_spikes.shape[0], \
        "assert same number of bins / time interval. m_spikes.shape: {}, target shape: {}".format(model_spikes.shape, target_spikes.shape)

    if not fname:
        fname = 'spiketrains_' + IO.dt_descriptor()

    if not export:
        data = {'model_spikes': model_spikes, 'target_spikes': target_spikes, 'exp_type': exp_type, 'title': title, 'fname': fname}
        IO.save_plot_data(data=data, uuid=uuid, plot_fn='plot_spiketrains_side_by_side')

    fig = plt.figure()
    time_indices = torch.reshape(torch.arange(model_spikes.shape[0]), (model_spikes.shape[0], 1)).float()

    # ensure binary values:
    model_spike_history = torch.round(model_spikes)
    target_spike_history = torch.round(target_spikes)

    model_spike_times = model_spike_history * time_indices
    target_spike_times = target_spike_history * time_indices

    plt.plot(0, -1, '.b')
    plt.plot(0, -1, '.g')
    if legend is not None:
        plt.legend(legend)
    else:
        plt.legend(['Model', 'Target'])

    for neuron_i in range(model_spike_history.shape[1]):
        if model_spike_times[:, neuron_i].nonzero().sum() > 0:
            plt.plot(torch.reshape(model_spike_times[:, neuron_i].nonzero(), (1, -1)).numpy(),
                     neuron_i + 1.1, '.b', markersize=3.0, label='Model')
    for neuron_i in range(target_spike_times.shape[1]):
        if target_spike_times[:, neuron_i].nonzero().sum() > 0:
            plt.plot(torch.reshape(target_spike_times[:, neuron_i].nonzero(), (1, -1)).numpy(),
                     neuron_i + 0.9, '.g', markersize=3.0-0.04*int(neuron_i/10), label='Target')

    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron')
    if neuron_i > 20:
        plt.yticks(range(int((neuron_i+1)/10), neuron_i + 1, int((neuron_i+1)/10)))
    else:
        plt.yticks(range(1, neuron_i + 2))
    plt.ylim(0, neuron_i+2)
    # if not title:
    #     title = 'Spiketrains side by side'
    # plt.title(title)

    full_path = './figures/' + exp_type + '/' +  uuid + '/'
    IO.makedir_if_not_exists('./figures/' + exp_type + '/')
    IO.makedir_if_not_exists(full_path)
    fig.savefig(fname=full_path + fname)
    # plt.show()
    plt.close()
    return fig


def plot_spike_train_projection(spikes, uuid='test', exp_type='default', title=False, fname=False, legend=None, export=False):
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax = Axes3D(fig)

    time_indices = torch.reshape(torch.arange(spikes.shape[0]), (spikes.shape[0], 1)).float()
    # ensure binary values:
    spike_history = torch.round(spikes)
    model_spike_times = spike_history * time_indices

    # ensure binary values:
    for neuron_i in range(spike_history.shape[1]):
        if model_spike_times[:, neuron_i].nonzero().sum() > 0:
            spike_times_reshaped = torch.reshape(model_spike_times[:, neuron_i].nonzero(), (1, -1))
            ax.scatter3D(spike_times_reshaped.numpy(),
                         (torch.ones_like(spike_times_reshaped) * neuron_i + 1.1).numpy(),
                         zs=0, label='Model')

    plt.xlabel('Time ($ms$)')
    plt.ylabel('Neuron')
    ax.set_zlabel('Parameters $P \in \Re^\mathbf{D}$')
    ax.set_zticks(range(-1, 2))
    if neuron_i > 20:
        ax.set_yticks(range(int((neuron_i + 1) / 10), neuron_i + 1, int((neuron_i + 1) / 10)))
    else:
        ax.set_yticks(range(1, neuron_i + 2))
    ax.set_xticks(range(0, 5000, 1000))
    ax.set_ylim(0, neuron_i + 2)

    if not fname:
        fname = 'spike_train_projection' + IO.dt_descriptor()

    full_path = './figures/' + exp_type + '/' + uuid + '/'
    IO.makedir_if_not_exists('./figures/' + exp_type + '/')
    IO.makedir_if_not_exists(full_path)
    fig.savefig(fname=full_path + fname)
    # plt.show()
    plt.close()


def plot_all_spiketrains(spikes_arr, uuid, exp_type='default', title=False, fname=False, legend=None):
    assert spikes_arr[0].shape[0] > spikes_arr[0].shape[1], \
        "assert one node per column, one bin per row. spikes shape: {}".format(spikes_arr[0].shape)

    if not fname:
        fname = 'spiketrains_' + IO.dt_descriptor()

    data = {'spikes_arr': spikes_arr, 'exp_type': exp_type, 'title': title, 'fname': fname}
    IO.save_plot_data(data=data, uuid=uuid, plot_fn='plot_spiketrains_side_by_side')

    fig = plt.figure()
    time_indices = torch.reshape(torch.arange(spikes_arr[0].shape[0]), (spikes_arr[0].shape[0], 1)).float()

    colours = ['.b', '.g', '.c', '.m', '.r']
    for i in range(len(spikes_arr)):
        plt.plot(0, -1, colours[i%len(colours)])

    for s_i in range(len(spikes_arr)):
        # ensure binary values:
        model_spike_history = torch.round(spikes_arr[s_i])
        model_spike_times = model_spike_history * time_indices

        for neuron_i in range(model_spike_history.shape[1]):
            if model_spike_times[:, neuron_i].nonzero().sum() > 0:
                plt.plot(torch.reshape(model_spike_times[:, neuron_i].nonzero(), (1, -1)).numpy(),
                         neuron_i + (1.0+0.5*0.15*len(spikes_arr)-0.15*s_i), colours[s_i%len(colours)], markersize=4.0)

    if legend is not None:
        plt.legend(legend, shadow=False, framealpha=0.5)

    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron')
    plt.yticks(range(1, neuron_i + 2))
    plt.ylim(0, neuron_i+2)
    if not title:
        title = 'Spiketrains side by side'
    plt.title(title)

    full_path = './figures/' + exp_type + '/' +  uuid + '/'
    IO.makedir_if_not_exists('./figures/' + exp_type + '/')
    IO.makedir_if_not_exists(full_path)
    fig.savefig(fname=full_path + fname)
    # plt.show()
    plt.close()


def plot_neuron(membrane_potentials_through_time, uuid, exp_type='default', title='Neuron activity',
                ylabel='Membrane potential', fname='plot_neuron_test'):
    data = {'membrane_potentials_through_time': membrane_potentials_through_time, 'title': title, 'uuid': uuid,
            'exp_type': exp_type, 'ylabel': ylabel, 'fname': fname}
    IO.save_plot_data(data=data, uuid='test_uuid', plot_fn='plot_neuron')
    legend = []
    for i in range(len(membrane_potentials_through_time)):
        legend.append('N.{}'.format(i+1))
    fig = plt.figure()
    plt.plot(np.arange(membrane_potentials_through_time.shape[0]), membrane_potentials_through_time)
    plt.legend(legend, loc='upper left', ncol=4)
    # plt.title(title)
    plt.xlabel('Time (ms)')
    plt.ylabel(ylabel)
    # plt.show()
    full_path = './figures/' + exp_type + '/' + uuid + '/'
    IO.makedir_if_not_exists(full_path)
    plt.savefig(fname=full_path + fname)
    plt.close()
    return fig


def plot_loss(loss, uuid, exp_type='default', custom_title=False, fname=False, ylabel=False):
    if not fname:
        fname = 'loss'+IO.dt_descriptor()
    else:
        fname = fname
    data = {'loss': loss, 'exp_type': exp_type, 'custom_title': custom_title, 'fname': fname}
    IO.save_plot_data(data=data, uuid=uuid, plot_fn='plot_loss')

    plt.plot(loss)
    # plt.legend(['Training loss', 'Test loss'])
    plt.xlabel('Training epoch')
    if ylabel:
        plt.ylabel(ylabel)
    else:
        plt.ylabel('Loss')
    # plt.xticks(range(len(loss_arr+1)))
    # if custom_title:
    #     plt.title(custom_title)
    # else:
    #     plt.title('Loss')

    full_path = './figures/' + exp_type + '/' + uuid + '/'
    IO.makedir_if_not_exists('./figures/' + exp_type + '/')
    IO.makedir_if_not_exists(full_path)
    plt.savefig(fname=full_path + fname)
    # plt.show()
    plt.close()

def plot_losses(training_loss, test_loss, uuid, exp_type='default', custom_title=False, fname=False):
    if not fname:
        fname = 'training_and_test_loss'+IO.dt_descriptor()
    data = {'training_loss': training_loss, 'test_loss': test_loss, 'exp_type': exp_type, 'custom_title': custom_title, 'fname': fname}
    IO.save_plot_data(data=data, uuid=uuid, plot_fn='plot_losses')

    plt.figure()
    if len(test_loss) > len(training_loss):
        plt.plot(range(1, len(training_loss)+1), training_loss)
    else:
        plt.plot(training_loss)

    plt.plot(test_loss)
    plt.legend(['Training loss', 'Test loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.xticks(range(len(loss_arr+1)))
    if custom_title:
        plt.title(custom_title)
    else:
        plt.title('Training and test set loss')

    full_path = './figures/' + exp_type + '/' + uuid + '/'
    IO.makedir_if_not_exists('./figures/' + exp_type + '/')
    IO.makedir_if_not_exists(full_path)
    plt.savefig(fname=full_path + fname)
    # plt.show()
    plt.close()


def plot_avg_losses(avg_train_loss, train_loss_std, avg_test_loss, test_loss_std, uuid, exp_type='default', custom_title=False, fname=False):
    # if not fname:
    #     fname = 'training_and_test_loss'+IO.dt_descriptor()
    # data = {'training_loss': training_loss, 'test_loss': test_loss, 'exp_type': exp_type, 'custom_title': custom_title, 'fname': fname}
    # IO.save_plot_data(data=data, uuid=uuid, plot_fn='plot_losses')

    plt.figure()
    # xs_n = len(avg_train_loss)
    xs_n = 20
    plt.errorbar(np.linspace(1, xs_n, len(avg_train_loss)), y=avg_train_loss, yerr=train_loss_std)
    plt.errorbar(np.linspace(1, xs_n, len(avg_test_loss)), y=avg_test_loss, yerr=test_loss_std)
    plt.xticks(np.arange(11) * 2)

    plt.legend(['Training loss', 'Test loss'])

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.xticks(range(len(loss_arr+1)))
    # if custom_title:
    #     plt.title(custom_title)
    # else:
    #     plt.title('Average training and test loss')

    full_path = './figures/' + exp_type + '/' + uuid + '/'
    IO.makedir_if_not_exists('./figures/' + exp_type + '/')
    IO.makedir_if_not_exists(full_path)
    plt.savefig(fname=full_path + fname)
    # plt.show()
    plt.close()


def plot_avg_losses_composite(loss_res, keys, archive_path='default_export'):
    plt.figure()
    # xs_n = len(avg_train_loss)
    legend = []
    # legend = ['frd', 'vrd', 'a(frd+vrd)']
    # fmts = ['--', '--*', '-']
    # cols = ['c', 'm', 'g']
    ctr = 0
    for key in keys:
        cur_avg_train_loss = np.mean(loss_res[key]['train_loss'], axis=0)
        train_std = np.std(loss_res[key]['train_loss'], axis=0)
        cur_avg_test_loss = np.mean(loss_res[key]['test_loss'], axis=0)
        test_std = np.std(loss_res[key]['test_loss'], axis=0)
        xs_n = 20
        norm_kern = np.max(cur_avg_test_loss)
        cur_linspace = np.linspace(1, xs_n, len(cur_avg_train_loss))
        # plt.errorbar(cur_linspace, y=cur_avg_train_loss/norm_kern, yerr=train_std/norm_kern, fmt=fmts[ctr % len(fmts)])
        plt.errorbar(cur_linspace, y=cur_avg_train_loss/norm_kern, yerr=train_std/norm_kern)
        # plt.errorbar(np.linspace(1, xs_n, len(cur_avg_test_loss)), y=cur_avg_test_loss/norm_kern, yerr=test_std/norm_kern)
        plt.xticks(np.arange(11) * 2)
        ctr +=1

        # legend.append('Training {}'.format(key))
        # legend.append('Test {}'.format(key))
        legend.append('{}'.format(key))

    ctr = 0
    for key in keys:
        cur_avg_test_loss = np.mean(loss_res[key]['test_loss'], axis=0)
        test_std = np.std(loss_res[key]['test_loss'], axis=0)
        xs_n = 20
        norm_kern = np.max(cur_avg_test_loss)
        # plt.errorbar(np.linspace(1, xs_n, len(cur_avg_test_loss)), y=cur_avg_test_loss/norm_kern, yerr=test_std/norm_kern, fmt=fmts[ctr % len(fmts)])
        plt.errorbar(np.linspace(1, xs_n, len(cur_avg_test_loss)), y=cur_avg_test_loss/norm_kern, yerr=test_std/norm_kern)
        plt.xticks(np.arange(11) * 2)
        ctr += 1

    plt.legend(legend, loc='upper left', fontsize='xx-small', fancybox=True, bbox_to_anchor=(0.8, 1.)).get_frame().set_alpha(0.6)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.grid(True)

    full_path = './figures/' + archive_path + '/'
    IO.makedir_if_not_exists(full_path)
    plt.savefig(fname=full_path + 'export_avg_loss_composite')
    # plt.show()
    plt.close()


def plot_losses_nodes(batch_loss_per_node, uuid, exp_type='default', custom_title=False, fname=False):
    if not fname:
        fname = 'batch_loss_per_node'+IO.dt_descriptor()
    data = {'batch_loss_per_node': batch_loss_per_node, 'exp_type': exp_type, 'custom_title': custom_title, 'fname': fname}
    IO.save_plot_data(data=data, uuid=uuid, plot_fn='plot_losses_nodes')

    plt.figure()
    for b_loss in batch_loss_per_node:
        plt.plot(b_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.xticks(range(len(loss_arr+1)))
    # if custom_title:
    #     plt.title(custom_title)
    # else:
    #     plt.title('Batch loss')

    full_path = './figures/' + exp_type + '/' + uuid + '/'
    IO.makedir_if_not_exists('./figures/' + exp_type + '/')
    IO.makedir_if_not_exists(full_path)
    plt.savefig(fname=full_path + fname)
    # plt.show()
    plt.close()


def calculate_kde(p1, p2, logger):
    data = np.vstack([p1, p2])

    std_0 = np.std(p1)
    std_1 = np.std(p2)
    if std_0 == 0 or std_1 == 0:
        raise ArithmeticError("Std was zero - plot point along axis.")

    kde = gaussian_kde(data)

    # evaluate on a regular grid
    std_coeff = 4.0
    x_min = np.mean(p1) - std_coeff * std_0; x_max = np.mean(p1) + std_coeff * std_0
    y_min = np.mean(p2) - std_coeff * std_1; y_max = np.mean(p2) + std_coeff * std_1

    xgrid = np.linspace(x_min, x_max, 40)
    ygrid = np.linspace(y_min, y_max, 40)
    Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)

    Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))

    return Z, Xgrid, x_min, x_max, y_min, y_max


def decompose_param_plot(param_2D, target_params, name, path, custom_title=False):
    params_by_exp = np.array(param_2D).T
    num_of_parameters = params_by_exp.shape[0]

    fig, axs = plt.subplots(nrows=num_of_parameters-1, ncols=num_of_parameters-1)
    [axi.set_axis_off() for axi in axs.ravel()]

    for i in range(num_of_parameters):
        for j in range(i + 1, num_of_parameters):
            cur_ax = axs[i,j-1]
            cur_ax.set_xlabel(name)
            cur_ax.set_ylabel(name)
            try:
                Z, Xgrid, x_min, x_max, y_min, y_max = calculate_kde(params_by_exp[i], params_by_exp[j], False)

                cur_ax.imshow(Z.reshape(Xgrid.shape),
                                  origin='lower', aspect='auto',
                                  extent=[x_min, x_max, y_min, y_max],
                                  cmap='Blues')
                if target_params is not None and hasattr(target_params, 'shape') and len(target_params[0]) > np.max([i, j]):
                    cur_ax.plot(target_params[0][i], target_params[0][j], 'or', markersize=2.8)
            except ArithmeticError as ae:
                print('arithmetic error:\n{}'.format(ae))
                cur_ax.plot(params_by_exp[i], params_by_exp[j], 'xb', markersize=3.5)
                if target_params is not None and hasattr(target_params, 'shape') and len(target_params) > np.max([i, j]):
                    cur_ax.plot(target_params[0][i], target_params[0][j], 'or', markersize=2.8)
            except Exception as e:
                print('exception:\n{}'.format(e))
                print('WARN: Failed to calculate KDE for param.s: {}, {}'.format(params_by_exp[i], params_by_exp[j]))

    if not path:
        path = './figures/{}/{}/param_subplot_inferred_params_{}'.format('default', 'test_uuid', IO.dt_descriptor())
    fig.savefig(path)
    plt.close()


def plot_all_param_pairs_with_variance(param_means, target_params, param_names, exp_type, uuid, fname, custom_title, logger, export_flag=False):
    if export_flag:
        full_path = data_util.prefix + 'data/export/' + exp_type + '/'
        IO.makedir_if_not_exists(data_util.prefix + 'data/export/')
        IO.makedir_if_not_exists(full_path)
        full_path = data_util.prefix + 'data/export/' + exp_type + '/' + uuid + '/'
    else:
        full_path = './figures/' + exp_type + '/' + uuid + '/'
        IO.makedir_if_not_exists('./figures/' + exp_type + '/')
        IO.makedir_if_not_exists(full_path)
        full_path = './figures/' + exp_type + '/'
    IO.makedir_if_not_exists(full_path)

    data = {'param_means': param_means, 'param_names': param_names, 'target_params': target_params, 'exp_type': exp_type,
            'uuid': uuid, 'custom_title': custom_title, 'fname': fname}
    IO.save_plot_data(data=data, uuid=uuid, plot_fn='plot_all_param_pairs_with_variance')

    if not fname:
        fname = 'new_inferred_params_{}'.format(IO.dt_descriptor())
    path = full_path + fname

    # number_of_parameters = min(len(param_names), len(param_means))
    for p_i, p_k in enumerate(param_means):  # assuming a dict., for all parameter combinations

            cur_p = np.array(param_means[p_k])
            name = '{}'.format(p_k)
            if len(cur_p.shape) == 2:
                cur_tar = False
                if target_params and p_k in target_params:
                    cur_tar = target_params[p_k]
                if path.__contains__('.eps'):
                    p_split = path.split('.eps')
                    decompose_param_plot(cur_p, cur_tar, name=name, path=p_split[0]+'_param_{}'.format(name)+'.eps', custom_title=custom_title)
                else:
                    decompose_param_plot(cur_p, cur_tar, name=name, path=path+'_param_{}'.format(name), custom_title=custom_title)


def decompose_param_pair_trajectory_plot(param_2D, current_targets, name, path, params=['p', 'p']):
    if os.path.exists(path):
        return

    params_by_exp = np.array(param_2D).T
    num_of_parameters = params_by_exp.shape[0]

    plt.rcParams.update({'font.size': 8})
    plt.locator_params(axis='x', nbins=2)

    fig = plt.figure()
    big_ax = fig.add_subplot(111)
    big_ax.set_axis_on()
    big_ax.grid(False)
    name = name.replace('tau', '\\tau').replace('spike_threshold', '\\theta')
    big_ax.set_title('Parameter trajectory for ${} \\times {}$'.format(name, name))
    big_ax.set_xlabel('${}$'.format(name), labelpad=20)
    big_ax.set_ylabel('${}$'.format(name), labelpad=34)
    big_ax.set_xticks([])
    big_ax.set_yticks([])
    print('num_of_parameters: {}'.format(num_of_parameters))
    axs = fig.subplots(nrows=num_of_parameters - 1, ncols=num_of_parameters - 1, sharex=True, sharey=True)
    dot_msize = 5.0
    if num_of_parameters == 2:
        # if current_targets is not False:
            # x_min = float('{}'.format(np.min(np.concatenate([params_by_exp[0], [current_targets[0]]]))))
            # x_max = float('\n{}'.format(np.max(np.concatenate([params_by_exp[0], [current_targets[0]]]))))
            # plt.xticks([x_min, x_max])

        p_len = len(params_by_exp[0])
        colors = cm.YlGn(np.linspace(0, 1, p_len))
        for p_i in range(p_len):
            plt.scatter(params_by_exp[0][p_i], params_by_exp[1][p_i], color=colors[p_i], marker='o', s=dot_msize)

        if current_targets is not False:
            plt.scatter(current_targets[0], current_targets[1], color='black', marker='x',
                           s=2. * dot_msize)  # test 2*dot_msize
    else:
        [axi.set_axis_off() for axi in axs.ravel()]

        for i in range(num_of_parameters):
            for j in range(i+1, num_of_parameters):
                cur_ax = axs[j - 1, i]
                cur_ax.set_axis_on()
                cur_ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                # if current_targets is not False:
                #     x_min = float('{}'.format(np.min(np.concatenate([params_by_exp[i], [current_targets[i]]]))))
                #     x_max = float('\n{}'.format(np.max(np.concatenate([params_by_exp[i], [current_targets[i]]]))))
                #     cur_ax.set_xticks([x_min, x_max])

                # try:
                p_len = len(params_by_exp[i])
                colors = cm.YlGn(np.linspace(0, 1, p_len))
                for p_i in range(p_len):
                    cur_ax.scatter(params_by_exp[i][p_i], params_by_exp[j][p_i], color=colors[p_i], marker='o', s=dot_msize)

                if current_targets is not False:
                    cur_ax.scatter(current_targets[i], current_targets[j], color='black', marker='x', s=2.*dot_msize)  # test 2*dot_msize

    if not path:
        path = './figures/{}/{}/param_subplot_inferred_params_{}.png'.format('default', 'test_uuid', IO.dt_descriptor())
    else:
        path = '{}_{}_{}_{}'.format(path[:-4], params[0], params[1], path[-4:])
    # plt.show()
    fig.savefig(path)
    plt.close()


def plot_parameter_inference_trajectories_2d(param_means, target_params, param_names, exp_type, uuid, fname, custom_title):
    full_path = './figures/' + exp_type + '/' + uuid + '/'
    IO.makedir_if_not_exists('./figures/' + exp_type + '/')
    IO.makedir_if_not_exists(full_path)

    if not fname:
        fname = 'new_inferred_params_{}.png'.format(IO.dt_descriptor())
    path = full_path + fname

    if not os.path.exists(path):
        data = {'param_means': param_means, 'target_params': target_params, 'exp_type': exp_type, 'uuid': uuid, 'custom_title': custom_title, 'fname': fname}
        IO.save_plot_data(data=data, uuid=uuid, plot_fn='plot_parameter_inference_trajectories_2d')

        for p_i, p_k in enumerate(param_means):  # assuming a dict., for all parameter combinations
            current_targets = False
            if target_params is not False:
                if p_k in target_params:
                    current_targets = target_params[p_k]

            cur_p = np.array(param_means[p_k])
            name = '{}'.format(p_k)

            # silently fail for 3D params (weights)
            if len(cur_p.shape) == 2:
                # if not os.path.exists(path):
                    # decompose_param_pair_trajectory_plot(cur_p[:,:,:4], current_targets[:,:,:4], name=name, path=param_path)
                if current_targets is not False:
                    max_index = min(5, len(current_targets))
                    decompose_param_pair_trajectory_plot(cur_p[:, :max_index], current_targets[:max_index], name=name, path=path, params=[p_i, p_k])
                else:
                    max_index = 5
                    decompose_param_pair_trajectory_plot(cur_p[:, :max_index], False, name=name, path=path, params=[p_i, p_k])


def bar_plot_neuron_rates(r1, r2, r1_std, r2_std, exp_type, uuid, fname, xticks=False,
                          custom_legend=['Fitted model', 'Target model'], ylabel='$Hz$',
                          custom_colors=['Blue', 'Orange']):
    full_path = './figures/' + exp_type + '/' + uuid + '/'
    IO.makedir_if_not_exists('./figures/' + exp_type + '/')
    IO.makedir_if_not_exists(full_path)

    data = {'r1': r1, 'r2': r2, 'exp_type': exp_type, 'uuid': uuid, 'fname': fname}
    IO.save_plot_data(data=data, uuid=uuid, plot_fn='bar_plot_neuron_rates')

    if len(np.asarray(r1)) > 1:
        rlen = len(np.asarray(r1))
        xs = np.linspace(1, rlen, rlen)
    elif len(np.asarray(r2)) > 1:
        rlen = len(np.asarray(r2))
        xs = np.linspace(1, rlen, rlen)
    else:
        xs = np.linspace(1, 2, 1)
    plt.bar(xs-0.2, r1, yerr=r1_std, width=0.4, color=custom_colors[0])
    plt.bar(xs+0.2, r2, yerr=r2_std, width=0.4, color=custom_colors[1])
    plt.legend(custom_legend)
    rmax1 = np.max(np.asarray(r1))
    rmax2 = np.max(np.asarray(r2))
    r_max = np.max([rmax1, rmax2])
    r1sm = np.max(np.array(r1_std))
    r2sm = np.max(np.array(r2_std))
    rstd_max = np.max([r1sm, r2sm])
    summed_max = r_max + rstd_max
    plt.ylim(0, summed_max + rstd_max*0.05)
    if not xticks:
        plt.xticks(xs)
    else:
        plt.xticks(xs, xticks)
    plt.ylabel(ylabel)
    plt.savefig(fname=full_path + fname)
    plt.close()


def bar_plot(y, y_std, labels, exp_type, uuid, fname, ylabel=False, xlabel=False, baseline=False, custom_colors=False, custom_legend=False):
    full_path = './figures/' + exp_type + '/' + uuid + '/'
    IO.makedir_if_not_exists('./figures/' + exp_type + '/')
    IO.makedir_if_not_exists(full_path)

    data = {'y': y, 'exp_type': exp_type, 'uuid': uuid, 'fname': fname}
    IO.save_plot_data(data=data, uuid=uuid, plot_fn='bar_plot')

    if str(type(y)).__contains__('array') or str(type(y)).__contains__('list'):
        print('len(y): {}'.format(len(y)))
        xs = np.linspace(1, len(y), len(y))
    else:
        xs = np.array([1])
        y = np.reshape(np.array([y]), (1,))
        y_std = np.reshape(np.array([y_std]), (1,))

    if hasattr(y_std, 'shape') and len(y_std.shape) > 0 or hasattr(y_std, 'len') and len(y_std) > 0 or hasattr(y_std, 'append'):
        if custom_colors:
            plt.bar(xs, y, yerr=y_std, width=0.35, color=custom_colors)
        else:
            plt.bar(xs, y, yerr=y_std, width=0.35)
    else:
        if custom_colors:
            plt.bar(xs, y, width=0.35, color=custom_colors)
        else:
            plt.bar(xs, y, width=0.35)

    if baseline:
        plt.plot(xs, np.ones_like(y) * baseline, 'g--')

    r_max = np.max(y)
    rstd_max = np.max(y_std)
    summed_max = r_max + rstd_max
    if not np.isnan(summed_max) and not np.isinf(summed_max):
        plt.ylim(0, summed_max + rstd_max*0.05)
    # plt.ylim(0, 15)
    if labels:
        plt.xticks(xs, labels)
    else:
        plt.xticks(xs)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    else:
        plt.ylabel('Distance')

    if custom_legend:
        plt.legend(custom_legend)
    # if title:
    #     plt.title(title)
    # else:
    #     plt.title('Variance and CV for each setup')
    # plt.show()
    plt.savefig(fname=full_path + fname)
    plt.close()


def bar_plot_pair_custom_labels(y1, y2, y1_std, y2_std, labels, exp_type, uuid, fname, title, ylabel=False, xlabel=False, legend=False, baseline=False, colours=False):
    full_path = './figures/' + exp_type + '/' + uuid + '/'
    IO.makedir_if_not_exists('./figures/' + exp_type + '/')
    IO.makedir_if_not_exists(full_path)

    data = {'y1': y1, 'y2': y2, 'exp_type': exp_type, 'uuid': uuid, 'fname': fname, 'title': title}
    IO.save_plot_data(data=data, uuid=uuid, plot_fn='bar_plot_pair_custom_labels')

    if hasattr(y1, 'len'):
        xs = np.linspace(1, len(y1), len(y1))
    else:
        xs = np.array([1.])

    if hasattr(y1_std, 'shape') and len(y1_std.shape) > 0 or hasattr(y1_std, 'append'):
        if colours:
            plt.bar(xs-0.15, y1, yerr=y1_std, width=0.3, color=colours[0])
        else:
            plt.bar(xs-0.15, y1, yerr=y1_std, width=0.3)
    else:
        if colours:
            plt.bar(xs-0.15, y1, width=0.3, color=colours[1])
        else:
            plt.bar(xs-0.15, y1, width=0.3)
    if hasattr(y2_std, 'shape') and len(y2_std.shape) > 0 or hasattr(y2_std, 'append'):
        if colours:
            plt.bar(xs+0.15, y2, yerr=y2_std, width=0.3, color=colours[1])
        else:
            plt.bar(xs+0.15, y2, yerr=y2_std, width=0.3)
    else:
        if colours:
            plt.bar(xs+0.15, y2, width=0.3, color=colours[1])
        else:
            plt.bar(xs+0.15, y2, width=0.3)

    if not legend:
        plt.legend(['Fitted model', 'Target model'])
    else:
        plt.legend(legend)

    if baseline:
        plt.plot(xs, np.ones_like(y1) * baseline, 'g--')

    r_max = np.max([np.array(y1), np.array(y2)])
    rstd_max = np.max([np.array(y1_std), np.array(y2_std)])
    summed_max = r_max + rstd_max
    if not np.isnan(summed_max) and not np.isinf(summed_max):
        plt.ylim(0, summed_max + rstd_max*0.05)
    # plt.ylim(0, 15)
    if labels:
        plt.xticks(xs, labels)
    else:
        plt.xticks(xs)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    else:
        plt.ylabel('Distance')
    if title:
        plt.title(title)
    # plt.show()
    plt.savefig(fname=full_path + fname)
    plt.close()


def bar_plot_pair_custom_labels_two_grps(y1, y2, y1_std, y2_std, labels, exp_type, uuid, fname, title, xlabel=False,
                                         ylabel=False, legend=False, baseline=False):
    full_path = './figures/' + exp_type + '/' + uuid + '/'
    IO.makedir_if_not_exists('./figures/' + exp_type + '/')
    IO.makedir_if_not_exists(full_path)

    data = {'y1': y1, 'y2': y2, 'exp_type': exp_type, 'uuid': uuid, 'fname': fname, 'title': title}
    IO.save_plot_data(data=data, uuid=uuid, plot_fn='bar_plot_pair_custom_labels')


    num_els = len(y1)
    half = int(num_els/2)
    rest = num_els-half
    width = 0.4
    skip = width*2
    xs = np.linspace(1, half, half)
    xs2 = np.linspace(1+half+skip, half+skip+rest, rest)
    plt.figure()
    plt.bar(xs-width/2, y1[:half], yerr=y1_std[:half], width=width)
    plt.bar(xs+width/2, y2[:half], yerr=y2_std[:half], width=width)
    plt.bar(xs2-width/2, y1[half:], yerr=y1_std[half:], width=width)
    plt.bar(xs2+width/2, y2[half:], yerr=y2_std[half:], width=width)

    r_max = np.max([np.array(y1), np.array(y2)])
    rstd_max = np.max([np.array(y1_std), np.array(y2_std)])
    summed_max = r_max + rstd_max
    plt.ylim(0, summed_max + rstd_max*0.05)
    # plt.ylim(0, 15)
    if labels:
        # plt.xticks(xs, labels)
        plt.xticks(np.concatenate((xs, xs2)), labels)
    else:
        plt.xticks(xs)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    # if title:
    #     plt.title(title)
    # else:
    #     plt.title('Variance and CV for each setup')
    # plt.show()
    plt.savefig(fname=full_path + fname)
    plt.close()



def bar_plot_crosscorrdiag(y1, y1_std, labels, exp_type, uuid, fname, title, xlabel=False, baseline=False):
    full_path = './figures/' + exp_type + '/' + uuid + '/'
    IO.makedir_if_not_exists('./figures/' + exp_type + '/')
    IO.makedir_if_not_exists(full_path)

    data = {'y1': y1, 'exp_type': exp_type, 'uuid': uuid, 'fname': fname, 'title': title}
    IO.save_plot_data(data=data, uuid=uuid, plot_fn='bar_plot_crosscorrdiag')

    xs = np.linspace(1, len(y1), len(y1))
    width = 1.4

    if hasattr(y1_std, 'shape') or hasattr(y1_std, 'append'):
        if baseline:
            above_threshold = np.maximum(y1 - np.ones_like(y1) * baseline, 0)
            below_threshold = np.minimum(y1, baseline)
            plt.bar(xs, below_threshold, yerr=y1_std, width=0.5*width)
            plt.bar(xs, above_threshold, yerr=y1_std, width=0.5*width, bottom=below_threshold)
        else:
            plt.bar(xs, y1, yerr=y1_std, width=width)
    else:
        plt.bar(xs, y1, width=width)

    plt.legend(['Initial', 'Fitted'])

    if baseline:
        plt.plot([xs[0]-width/2, xs[-1]+width/2], [baseline, baseline], 'k--')

    r_max = np.max(np.array(y1))
    rstd_max = np.max(np.array(y1_std))
    summed_max = r_max + rstd_max
    plt.ylim(0, summed_max + rstd_max*0.05)
    # plt.ylim(0, 15)
    if labels:
        plt.xticks(xs, labels)
    else:
        plt.xticks(xs)
    if xlabel:
        plt.xlabel(xlabel)
    # if title:
    #     plt.title(title)
    # else:
    #     plt.title('Variance and CV for each setup')
    # plt.show()
    plt.savefig(fname=full_path + fname)
    plt.close()


def bar_plot_two_grps(y1, y1_std, y2, y2_std, labels, exp_type, uuid, fname, title, xlabel=False, ylabel=False, baseline=False):
    full_path = './figures/' + exp_type + '/' + uuid + '/'
    IO.makedir_if_not_exists('./figures/' + exp_type + '/')
    IO.makedir_if_not_exists(full_path)

    data = {'y1': y1, 'exp_type': exp_type, 'uuid': uuid, 'fname': fname, 'title': title}
    IO.save_plot_data(data=data, uuid=uuid, plot_fn='bar_plot_crosscorrdiag')

    xs = np.linspace(1, len(y1)+1, len(y1))
    width = 1.2
    xs2 = np.linspace(1+len(y1)+2*width, 1+len(y1)+len(y2)+2*width, len(y2))

    if hasattr(y1_std, 'shape') or hasattr(y1_std, 'append'):
        plt.bar(xs, y1, yerr=y1_std, width=width)
        plt.bar(xs2, y2, yerr=y2_std, width=width)
    else:
        plt.bar(xs, y1, width=width)
        plt.bar(xs2, y2, width=width)

    # plt.legend(['Adam', 'SGD'], loc='upper right')

    if baseline:
        plt.plot([xs[0]-width/2, xs2[-1]+width/2], [baseline, baseline], 'k--')

    r_max = np.max(np.array(y1))
    rstd_max = np.max(np.array(y1_std))
    summed_max = r_max + rstd_max
    if not np.isnan(summed_max) and not np.isinf(summed_max):
        plt.ylim(0, summed_max + rstd_max*0.05)
    # plt.ylim(0, 15)
    if labels:
        plt.xticks(np.concatenate((xs, xs2)), labels)
        # plt.xticks(xs2, labels)
    else:
        plt.xticks(xs)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    else:
        plt.ylabel('Relative distance')
    # if title:
    #     plt.title(title)
    # else:
    #     plt.title('Variance and CV for each setup')
    # plt.show()
    plt.savefig(fname=full_path + fname)
    plt.close()


def bar_plot_all_neuron_rates(rates, stds, bin_size, exp_type, uuid, fname, legends):
    full_path = './figures/' + exp_type + '/' + uuid + '/'
    IO.makedir_if_not_exists('./figures/' + exp_type + '/')
    IO.makedir_if_not_exists(full_path)

    data = {'rates': rates, 'stds': stds, 'exp_type': exp_type, 'uuid': uuid, 'fname': fname}
    IO.save_plot_data(data=data, uuid=uuid, plot_fn='bar_plot_neuron_rates')

    xs = np.linspace(1, rates[0].shape[0], rates[0].shape[0])
    width = 0.8/len(rates)
    max_rates = []; max_stds = []
    for i in range(len(rates)):
        print('plotting i: {}'.format(i))
        r = rates[i]; std = stds[i]
        plt.bar(xs-width+i*width, r.numpy(), yerr=std.numpy(), width=width)
        # max_rates.append([np.max(r)])
        # max_stds.append([np.max(std)])

    plt.legend(legends)
    # r_max = np.max(max_rates); rstd_max = np.max(max_stds)
    # summed_max = r_max + rstd_max

    # plt.ylim(0, summed_max + rstd_max*0.05)
    plt.xticks(xs)

    plt.xlabel('Neuron')
    plt.ylabel('$Hz$')
    # plt.title('Mean firing rate per neuron (bin size: {} ms)'.format(bin_size))

    # plt.show()
    plt.savefig(fname=full_path + fname)
    plt.close()


def heatmap_spike_train_correlations(corrs, axes, exp_type, uuid, fname, bin_size, custom_title=False, custom_label=False):
    full_path = './figures/' + exp_type + '/' + uuid + '/'
    IO.makedir_if_not_exists('./figures/' + exp_type + '/')
    IO.makedir_if_not_exists(full_path)

    data = {'corrs': corrs, 'exp_type': exp_type, 'uuid': uuid, 'fname': fname}
    IO.save_plot_data(data=data, uuid=uuid, plot_fn='heat_plot_spike_train_correlations')

    for row_i in range(corrs.shape[0]):
        for col_i in range(corrs.shape[1]):
            if np.isnan(corrs[row_i][col_i]):
                corrs[row_i][col_i] = 0.

    a = plt.imshow(corrs, cmap="PuOr", vmin=-1, vmax=1)
    cbar = plt.colorbar(a)
    if custom_label is not False:
        cbar.set_label(custom_label)
    else:
        cbar.set_label("correlation coeff.")
    # if custom_title is not False:
    #     plt.title(custom_title)
    # else:
    #     plt.title('Pairwise spike correlations (interval: {} ms)'.format(bin_size))
    plt.xticks(np.arange(0, len(corrs)))
    plt.yticks(np.arange(0, len(corrs)))
    plt.ylabel(axes[0])
    plt.xlabel(axes[1])
    # plt.show()
    plt.savefig(fname=full_path + fname)
    plt.close()


def plot_heatmap(heat_mat, axes, exp_type, uuid, fname, target_coords=False, xticks=False, yticks=False, v_min=0, v_max=1, cbar_label='loss'):
    full_path = './figures/' + exp_type + '/' + uuid + '/'
    IO.makedir_if_not_exists('./figures/' + exp_type + '/')
    IO.makedir_if_not_exists(full_path)

    data = {'heat_mat': heat_mat, 'exp_type': exp_type, 'uuid': uuid, 'fname': fname}
    IO.save_plot_data(data=data, uuid=uuid, plot_fn='plot_heatmap')

    for row_i in range(heat_mat.shape[0]):
        for col_i in range(heat_mat.shape[1]):
            if np.isnan(heat_mat[row_i][col_i]):
                heat_mat[row_i][col_i] = 0.

    fig = plt.figure()
    im = plt.imshow(heat_mat, cmap="PuOr", vmin=v_min, vmax=v_max)
    cbar = plt.colorbar(im)
    cbar.set_label(cbar_label)
    ticks_fmt = lambda x: float('{:.2f}'.format(x))
    if xticks:
        N_dim = len(xticks)
        tar_xticks = [xticks[0], xticks[int(N_dim/2)], xticks[-1]]
        tar_xticks = list(map(ticks_fmt, tar_xticks))
        plt.xticks([0, int(N_dim/2), N_dim-1], tar_xticks)
    else:
        plt.xticks(np.arange(0, len(heat_mat), 5))
    if yticks:
        N_dim = len(yticks)
        tar_yticks = [yticks[0], yticks[int(N_dim / 2)], yticks[-1]]
        tar_yticks = list(map(ticks_fmt, tar_yticks))
        plt.yticks([0, int(N_dim / 2), N_dim - 1], [yticks[0], yticks[int(N_dim / 2)], yticks[-1]])
    else:
        plt.yticks(np.arange(0, len(heat_mat)))
    plt.xlabel(axes[0])
    plt.ylabel(axes[1])
    if target_coords:
        plt.scatter(target_coords[0], target_coords[1], color='magenta', marker='x', s=30.0)
    # plt.show()
    plt.savefig(fname=full_path + fname)
    plt.close()
    return fig


def plot_parameter_landscape(p1s, p2s, p1_name, p2_name, summary_statistic, statistic_name, exp_type, uuid, fname):
    full_path = './figures/' + exp_type + '/' + uuid + '/'
    IO.makedir_if_not_exists('./figures/' + exp_type + '/')
    IO.makedir_if_not_exists(full_path)

    data = {'p1s': p1s, 'p2s': p2s, 'summary_statistic': summary_statistic,
            'p1_name': p1_name, 'p2_name': p2_name, 'statistic_name': statistic_name,
            'exp_type': exp_type, 'uuid': uuid, 'fname': fname}
    IO.save_plot_data(data=data, uuid=uuid, plot_fn=fname)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_trisurf(p1s, p2s, summary_statistic, cmap=plt.cm.jet, linewidth=0.01)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('${}$'.format(p1_name))
    ax.set_ylabel('${}$'.format(p2_name))
    ax.set_zlabel('${}$'.format(statistic_name))
    # ax.view_init(30, 45)
    # plt.show()
    plt.savefig(fname=full_path + fname)
    plt.close()
