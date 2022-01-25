import sys

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import IO
import Log
import PDF_metrics
import experiments
import model_util
import plot
from Models.microGIF import microGIF
from Models.microGIF_fixed import microGIF_fixed
from experiments import release_computational_graph

start_seed = 7
num_seeds = 1
# prev_timestamp = '11-12_13-27-54-034'
# prev_timestamp = '11-16_10-50-30-238'
# prev_timestamp = '11-16_11-03-55-161'
# prev_timestamp = '11-16_11-12-37-827'
prev_timestamp = '11-16_11-21-13-903'
for random_seed in range(start_seed, start_seed+num_seeds):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    # pop_sizes, snn = TargetModelMicroGIF.micro_gif_populations_model_full_size(random_seed=random_seed)
    # pop_sizes, snn_target = get_low_dim_micro_GIF_transposed(random_seed=random_seed)
    fname = 'snn_model_target_GD_test'
    load_data = torch.load(IO.PATH + microGIF.__name__ + '/' + prev_timestamp + '/' + fname + IO.fname_ext)
    snn_target = load_data['model']
    saved_target_losses = load_data['loss']

    N = snn_target.N
    t = 1200
    learn_rate = 0.02
    num_train_iter = 300
    plot_every = int(num_train_iter/20)
    bin_size = 100
    # optim_class = torch.optim.SGD(optfig_params, lr=learn_rate)
    optim_class = torch.optim.Adam
    # lfn = PDF_metrics.PDF_LFN.BERNOULLI
    lfn = PDF_metrics.PDF_LFN.POISSON
    config_str = '$\\alpha={}$, lfn: {}, bin_size: {}, optim: {}'.format(learn_rate, lfn.name, bin_size, optim_class.__name__)

    timestamp = IO.dt_descriptor()
    logger = Log.Logger('microGIF_GD_{}.txt'.format(timestamp))
    writer = SummaryWriter('runs/' + timestamp)

    A_coeffs = [torch.randn((4,))]
    phase_shifts = [torch.rand((4,))]
    input_types = [1, 1, 1, 1]


    current_inputs = experiments.generate_composite_input_of_white_noise_modulated_sine_waves(t, A_coeffs, phase_shifts, input_types)
    _, target_spikes, target_vs = model_util.feed_inputs_sequentially_return_args(snn_target, current_inputs)
    target_spikes = target_spikes.clone().detach()
    target_parameters = snn_target.state_dict()

    target_w_signs = torch.sign(snn_target.w.clone().detach().data)

    params_model = experiments.draw_from_uniform(microGIF.parameter_init_intervals, N)
    rand_ws = 5. + 2. * torch.randn((N, N))
    rand_ws = torch.abs(rand_ws) * target_w_signs
    params_model['preset_weights'] = rand_ws
    # snn = microGIF(N=N, parameters=params_model)

    params_model['tau_m'] = snn_target.tau_m.clone().detach().numpy()
    params_model['tau_s'] = snn_target.tau_s.clone().detach().numpy()
    params_model['c'] = snn_target.c.clone().detach().data
    snn = microGIF_fixed(N=N, parameters=params_model)

    # params_model = snn_target.get_parameters()
    # params_model['preset_weights'] = target_w_signs * torch.abs(rand_ws)
    # snn = microGIF_weights_only(N=N, parameters=params_model, neuron_types=torch.tensor([1., 1., -1., -1.]))

    # pop_sizes_snn, snn = get_low_dim_micro_GIF_transposed(random_seed=random_seed)
    fig_W_init = plot.plot_heatmap(snn.w.detach().numpy() / 10., ['W_syn_col', 'W_row'], uuid=snn.__class__.__name__ + '/{}'.format(timestamp),
                                   exp_type='GD_test', fname='plot_heatmap_W_initial.png')

    optim_params = list(snn.parameters())
    optimiser = optim_class(optim_params, lr=learn_rate)

    fig_inputs = plot.plot_neuron(current_inputs.detach().data, uuid=snn.__class__.__name__ + '/{}'.format(timestamp),
                                  exp_type='GD_test', fname='train_inputs.png')
    fig_tar_vs = plot.plot_neuron(target_vs.detach().data, uuid=snn.__class__.__name__ + '/{}'.format(timestamp),
                                  exp_type='GD_test',
                                  fname='membrane_pots_target.png')
    tar_W_heatmap_fig = plot.plot_heatmap(snn_target.w.detach().numpy() / 10., ['W_syn_col', 'W_row'],
                                          uuid=snn.__class__.__name__ + '/{}'.format(timestamp),
                                          exp_type='GD_test', fname='plot_heatmap_W_target.png')

    writer.add_figure('Training input', fig_inputs)
    writer.add_figure('Target W heatmap', tar_W_heatmap_fig)
    writer.add_figure('Target vs', fig_tar_vs)
    # writer.add_figure('Initial model W heatmap', fig_W_init)

    losses = []; prev_write_index = -1
    weights = []
    model_parameter_trajectories = {}
    cur_params = snn.state_dict()
    for p_i, key in enumerate(cur_params):
        model_parameter_trajectories[key] = [cur_params[key].clone().detach().numpy()]
    for i in range(num_train_iter+1):
        optimiser.zero_grad()

        current_inputs = experiments.generate_composite_input_of_white_noise_modulated_sine_waves(t, A_coeffs, phase_shifts, input_types)
        spike_probs, spikes, vs = model_util.feed_inputs_sequentially_return_args(snn, current_inputs)

        if lfn == PDF_metrics.PDF_LFN.POISSON:
            loss = PDF_metrics.poisson_nll(spike_probabilities=spike_probs, target_spikes=target_spikes, bin_size=bin_size)
        elif lfn == PDF_metrics.PDF_LFN.BERNOULLI:
            loss = PDF_metrics.bernoulli_nll(spike_probabilities=spike_probs, target_spikes=target_spikes)
        else:
            raise NotImplementedError()

        loss.backward(retain_graph=True)

        optimiser.step()

        loss_data = loss.clone().detach().data
        losses.append(loss_data)
        print('loss: {}'.format(loss_data))
        writer.add_scalar('training_loss', scalar_value=loss_data, global_step=i)

        if i == 0 or i % plot_every == 0 or i == num_train_iter:
            fig_spikes = plot.plot_spike_trains_side_by_side(spikes, target_spikes, uuid=snn.__class__.__name__ + '/{}'.format(timestamp),
                                                exp_type='GD_test', title='Test {} spike trains'.format(snn.__class__.__name__),
                                                legend=['Initial', 'Target'], fname='spike_trains_train_iter_{}.png'.format(i))
            fig_vs = plot.plot_neuron(vs.detach().data, uuid=snn.__class__.__name__ + '/{}'.format(timestamp), exp_type='GD_test', fname='membrane_pots_train_i_{}.png'.format(i))
            fig_heatmap = plot.plot_heatmap(snn.w.detach().numpy() / 10., ['W_syn_col', 'W_row'], uuid=snn.__class__.__name__ + '/{}'.format(timestamp),
                                   exp_type='GD_test', fname='plot_heatmap_W_train_i_{}.png'.format(i))

            # writer.add_scalars('training_loss', { 'losses': torch.tensor(losses[prev_write_index:]) }, i)
            # for loss_i in range(len(losses) - prev_write_index):
            #     writer.add_scalar('training_loss', scalar_value=losses[prev_write_index+loss_i], global_step=prev_write_index+loss_i)
            # prev_write_index = i

            weights.append(snn.w.clone().detach().flatten().numpy())
            # weights.append(np.mean(snn.w.clone().detach().numpy(), axis=1))

            for p_i, param in enumerate(list(snn.parameters())):
                print('grad for param #{}: {}'.format(p_i, param.grad))

            # ...log a Matplotlib Figure showing the model's predictions on a random mini-batch
            writer.add_figure('Model spikes vs. target spikes', fig_spikes, global_step=i)
            writer.add_figure('Model membrane potentials', fig_vs, global_step=i)
            writer.add_figure('Weights heatmap', fig_heatmap, global_step=i)

        cur_params = snn.state_dict()
        for p_i, key in enumerate(cur_params):
            model_parameter_trajectories[key].append(cur_params[key].clone().detach().numpy())

        release_computational_graph(snn, current_inputs)
        loss = None; current_inputs = None

    plot.plot_loss(losses, uuid=snn.__class__.__name__+'/{}'.format(timestamp), exp_type='GD_test',
                   custom_title='Loss {}, $\\alpha$={}, {}, bin_size={}'.format(lfn.name, learn_rate, optimiser.__class__.__name__, bin_size),
                   fname='plot_loss_test'+IO.dt_descriptor())

    plot.plot_spike_trains_side_by_side(spikes, target_spikes, uuid=snn.__class__.__name__+'/{}'.format(timestamp),
                                        exp_type='GD_test', title='Test {} spike trains'.format(snn.__class__.__name__),
                                        legend=['Fitted', 'Target'])

    _ = plot.plot_heatmap(snn.w.detach().numpy() / 10., ['W_syn_col', 'W_row'], uuid=snn.__class__.__name__+'/{}'.format(timestamp),
                          exp_type='GD_test', fname='plot_heatmap_W_after_training.png')

    hard_thresh_spikes_sum = torch.round(spikes).sum()
    print('spikes sum: {}'.format(hard_thresh_spikes_sum))
    soft_thresh_spikes_sum = (spikes > 0.333).sum()
    zero_thresh_spikes_sum = (spikes > 0).sum()
    print('thresholded spikes sum: {}'.format(torch.round(spikes).sum()))
    print('=========avg. hard rate: {}'.format(1000*hard_thresh_spikes_sum / (spikes.shape[1] * spikes.shape[0])))
    print('=========avg. soft rate: {}'.format(1000*soft_thresh_spikes_sum / (spikes.shape[1] * spikes.shape[0])))
    print('=========avg. zero thresh rate: {}'.format(1000*zero_thresh_spikes_sum / (spikes.shape[1] * spikes.shape[0])))

    # 🍝 weights across iterations plot.
    plot.plot_parameter_inference_trajectories_2d({'w': weights}, target_params={'w': snn_target.w.detach().flatten().numpy() },
                                                  uuid=snn.__class__.__name__ + '/' + timestamp,
                                                  exp_type='GD_test',
                                                  param_names=['w'],
                                                  custom_title='Test weights plot',
                                                  fname='test_weights_inference_trajectories')

    parameter_names = snn.free_parameters
    plot.plot_parameter_inference_trajectories_2d(model_parameter_trajectories,
                                                  uuid=snn.__class__.__name__ + '/' + timestamp,
                                                  exp_type='GD_test',
                                                  target_params=target_parameters,
                                                  param_names=parameter_names,
                                                  custom_title='Inferred parameters across training iterations',
                                                  fname='inferred_param_trajectories_{}_exp_num_{}_train_iters_{}'
                                                  .format(snn.__class__.__name__, None, i))

    IO.save(snn, loss={'losses': losses}, uuid=snn.__class__.__name__ + '/' + timestamp, fname='snn_model_target_GD_test')

    logger.log('snn.parameters(): {}'.format(snn.parameters()), list(snn.parameters()))
    logger.log('model_parameter_trajectories: ', model_parameter_trajectories)

sys.exit(0)
