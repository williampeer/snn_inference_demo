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
import spike_metrics
from Models.GLIF import GLIF
from Models.LIF import LIF
from TargetModels import TargetModelsBestEffort
from experiments import release_computational_graph

# prev_timestamp = '11-12_13-27-54-034'
# prev_timestamp = '11-16_10-50-30-238'
# prev_timestamp = '11-16_11-03-55-161'
# prev_timestamp = '11-16_11-12-37-827'
# prev_timestamp = '11-16_11-21-13-903'
start_seed = 23
num_seeds = 10

tar_timestamp_GLIF = '12-09_11-12-47-541'  # GLIF
tar_timestamp_LIF = '12-09_11-49-59-999'  # LIF
# target_model_timestamps = [tar_timestamp_GLIF, tar_timestamp_LIF]
target_model_timestamps = [tar_timestamp_LIF]
# target_model_classes = [GLIF, LIF]
target_model_classes = [LIF]

for tmt_i, tmt in enumerate(target_model_timestamps):
    # for lfn in ['frd', 'vrd']:
    for lfn in ['vrd']:
        for random_seed in range(start_seed, start_seed+num_seeds):
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
            # pop_sizes, snn = TargetModelMicroGIF.micro_gif_populations_model_full_size(random_seed=random_seed)
            # pop_sizes, snn_target = get_low_dim_micro_GIF_transposed(random_seed=random_seed)
            fname = 'snn_model_target_GD_test'
            model_class = target_model_classes[tmt_i]
            load_data = torch.load(IO.PATH + model_class.__name__ + '/' + tmt + '/' + fname + IO.fname_ext)
            snn_target = load_data['model']
            saved_target_losses = load_data['loss']

            N = snn_target.N
            t = 1200
            learn_rate = 0.001
            num_train_iter = 100
            plot_every = round(num_train_iter/20)
            bin_size = 100
            tau_vr = 4.
            # optim_class = torch.optim.SGD
            optim_class = torch.optim.Adam
            config_str = '$\\alpha={}$, lfn: {}, bin_size: {}, optim: {}'.format(learn_rate, lfn, bin_size, optim_class.__name__)

            timestamp = IO.dt_descriptor()
            logger = Log.Logger('{}_GD_{}.txt'.format(model_class.__name__, timestamp))
            writer = SummaryWriter('runs/' + timestamp)

            A_coeffs = [torch.randn((4,))]
            phase_shifts = [torch.rand((4,))]
            input_types = [1, 1, 1, 1]

            white_noise = torch.rand((t, N))
            assert white_noise.shape[0] > white_noise.shape[1]
            current_inputs = experiments.sine_modulated_input(white_noise)
            target_vs, target_spikes = model_util.feed_inputs_sequentially_return_tuple(snn_target, current_inputs)
            target_spikes = target_spikes.clone().detach()
            target_parameters = snn_target.state_dict()

            params_model = experiments.draw_from_uniform(model_class.parameter_init_intervals, N)
            snn = model_class(parameters=params_model, N=N, neuron_types=torch.tensor([1., 1., -1., -1.]))

            fig_W_init = plot.plot_heatmap((snn.w.clone().detach()), ['W_syn_col', 'W_row'],
                                           uuid=snn.__class__.__name__ + '/{}'.format(timestamp),
                                           exp_type='GD_test', fname='plot_heatmap_W_initial.png')

            optim_params = list(snn.parameters())
            optimiser = optim_class(optim_params, lr=learn_rate)

            fig_inputs = plot.plot_neuron(current_inputs.detach().data, uuid=snn.__class__.__name__ + '/{}'.format(timestamp),
                                          exp_type='GD_test', fname='train_inputs.png')
            fig_tar_vs = plot.plot_neuron(target_vs.detach().data, uuid=snn.__class__.__name__ + '/{}'.format(timestamp),
                                          exp_type='GD_test',
                                          fname='membrane_pots_target.png')
            tar_W_heatmap_fig = plot.plot_heatmap(snn_target.w.detach().numpy(), ['W_syn_col', 'W_row'],
                                                  uuid=snn.__class__.__name__ + '/{}'.format(timestamp),
                                                  exp_type='GD_test', fname='plot_heatmap_W_target.png')

            writer.add_figure('Training input', fig_inputs)
            writer.add_figure('Target W heatmap', tar_W_heatmap_fig)
            writer.add_figure('Target vs', fig_tar_vs)

            losses = []; prev_write_index = -1
            weights = []
            model_parameter_trajectories = {}
            cur_params = snn.state_dict()
            for p_i, key in enumerate(cur_params):
                model_parameter_trajectories[key] = [cur_params[key].clone().detach().numpy()]
            for i in range(num_train_iter+1):
                optimiser.zero_grad()

                white_noise = torch.rand((t, N))
                current_inputs = experiments.sine_modulated_input(white_noise)
                # spike_probs, spikes, vs = model_util.feed_inputs_sequentially_return_args(snn, current_inputs)
                spike_probs = None
                vs, spikes = model_util.feed_inputs_sequentially_return_tuple(snn, current_inputs)

                if lfn == PDF_metrics.PDF_LFN.POISSON:
                    loss = PDF_metrics.poisson_nll(spike_probabilities=spike_probs, target_spikes=target_spikes, bin_size=bin_size)
                elif lfn == PDF_metrics.PDF_LFN.BERNOULLI:
                    loss = PDF_metrics.bernoulli_nll(spike_probabilities=spike_probs, target_spikes=target_spikes)
                elif lfn == 'frd':
                    loss = spike_metrics.firing_rate_distance(model_spikes=spikes, target_spikes=target_spikes)
                elif lfn == 'vrd':
                    loss = spike_metrics.van_rossum_dist(spikes=spikes, target_spikes=target_spikes, tau=tau_vr)
                elif lfn == 'frdvrd':
                    frd_loss = spike_metrics.firing_rate_distance(model_spikes=spikes, target_spikes=target_spikes)
                    vrd_loss = spike_metrics.van_rossum_dist(spikes=spikes, target_spikes=target_spikes, tau=tau_vr)
                    loss = frd_loss + vrd_loss
                elif lfn == 'custom':
                    loss = spike_metrics.correlation_metric_distance(spikes, target_spikes, bin_size=100)
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
                    fig_heatmap = plot.plot_heatmap(snn.w.clone().detach(), ['W_syn_col', 'W_row'], uuid=snn.__class__.__name__ + '/{}'.format(timestamp),
                                           exp_type='GD_test', fname='plot_heatmap_W_train_i_{}.png'.format(i))

                    weights.append((snn.w.clone().detach()).flatten().numpy())
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
                           custom_title='Loss {}, $\\alpha$={}, {}, bin_size={}'.format(lfn, learn_rate, optimiser.__class__.__name__, bin_size),
                           fname='plot_loss_test'+IO.dt_descriptor())

            plot.plot_spike_trains_side_by_side(spikes, target_spikes, uuid=snn.__class__.__name__+'/{}'.format(timestamp),
                                                exp_type='GD_test', title='Test {} spike trains'.format(snn.__class__.__name__),
                                                legend=['Fitted', 'Target'])

            _ = plot.plot_heatmap((snn.w.clone().detach()), ['W_syn_col', 'W_row'], uuid=snn.__class__.__name__+'/{}'.format(timestamp),
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
            # plot.plot_parameter_inference_trajectories_2d({'w': weights}, target_params={'w': snn_target.w.detach().flatten().numpy() },
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
