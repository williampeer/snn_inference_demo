import torch

import model_util
import spike_metrics
from TargetModels import TargetModels
from experiments import sine_modulated_white_noise_input, release_computational_graph


def test_model_grad_is_clamped(model):
    t_interval = 2000
    inputs = sine_modulated_white_noise_input(10., t=t_interval, N=model.N)
    inputs.retain_grad()
    print('#inputs: {}'.format(inputs.sum()))

    spikes = model_util.feed_inputs_sequentially_return_spike_train(model, inputs)
    print('#spikes: {}'.format(spikes.sum()))

    learn_rate = 0.1
    loss_0 = spike_metrics.firing_rate_distance(spikes, sine_modulated_white_noise_input(10., t=t_interval, N=model.N))

    optim_params = list(model.parameters())
    optim = torch.optim.SGD(optim_params, lr=learn_rate)

    # Test normal fitting works:
    tau_m_0 = model.tau_m[0].clone().detach()
    assert loss_0 > 0., "loss should be non-zero"
    loss_0.backward(retain_graph=True)
    assert model.tau_m.grad[0] > 1e-09 or model.tau_m.grad[0] < -1e-09, "gradient should be lower or higher than +-1e-09. model.R_I.grad: {}".format(model.tau_m.grad)
    optim.step()

    release_computational_graph(model, None, inputs)
    loss_0 = loss_0.clone().detach()
    loss_0.grad = None
    optim.zero_grad()

    inputs =  sine_modulated_white_noise_input(10., t_interval, model.N)
    inputs.retain_grad()
    spikes = model_util.feed_inputs_sequentially_return_spike_train(model, inputs)
    loss_1 = spike_metrics.firing_rate_distance(spikes, sine_modulated_white_noise_input(100., t_interval, model.N))
    assert loss_1 > loss_0, "much bigger rate difference should give higher loss. loss_0: {}, loss_1: {}".format(loss_0, loss_1)

    loss_1.backward(retain_graph=True)
    # assert model.tau_m[0] + learn_rate * model.tau_m.grad[0] > tau_m_max, "gradient should be too large and needs clipping"
    assert model.tau_m.grad[0] > 1e-03 or model.tau_m.grad[0] < -1e-03, "gradient should be lower or higher than +-1e-03. \ngrad: {}".format(model.tau_m.grad)
    optim.step()

    tau_m_2 = model.tau_m[0].clone().detach()
    assert tau_m_0 != tau_m_2, "tau_m[0]_0 should have changed at tau_m[0]_2. tau_m[0]_0: {}, tau_m[0]_2: {}".format(tau_m_0, tau_m_2)

    for _ in range(5):
        optim.zero_grad()

        inputs = sine_modulated_white_noise_input(10., t_interval, model.N)
        inputs.retain_grad()
        spikes = model_util.feed_inputs_sequentially_return_spike_train(model, inputs)
        loss_big = spike_metrics.firing_rate_distance(spikes, sine_modulated_white_noise_input(100., t_interval, model.N))
        assert loss_big > 10., "loss: {}".format(loss_big)

        loss_big.backward(retain_graph=True)
        optim.step()

        release_computational_graph(model, None, inputs)
        loss_big.grad = None

    for i in range(len(model.R_I)):
        assert model.tau_m[i] >= 1.1, "model.tau_m[{}] ({}) should be >= 1.1".format(i, model.tau_m[i])
        assert model.tau_m[i] <= 3.0, "model.tau_m[{}] ({}) should be <= 3.0".format(i, model.tau_m[i])

    # release_computational_graph(model, None, inputs)


random_seed = 42
ext_name = 'ensembles_1_dales'

m_LIF = TargetModels.lif_continuous_ensembles_model_dales_compliant(random_seed=random_seed, N = 12)
m_LIF_R = TargetModels.lif_r_continuous_ensembles_model_dales_compliant(random_seed=random_seed, N = 12)
m_LIF_ASC = TargetModels.lif_asc_continuous_ensembles_model_dales_compliant(random_seed=random_seed, N = 12)
m_LIF_R_ASC = TargetModels.lif_r_asc_continuous_ensembles_model_dales_compliant(random_seed=random_seed, N = 12)
m_GLIF = TargetModels.glif_continuous_ensembles_model_dales_compliant(random_seed=random_seed, N = 12)

test_model_grad_is_clamped(m_LIF)
test_model_grad_is_clamped(m_LIF_R)
test_model_grad_is_clamped(m_LIF_ASC)
test_model_grad_is_clamped(m_LIF_R_ASC)
test_model_grad_is_clamped(m_GLIF)
