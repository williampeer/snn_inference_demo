import numpy as np
import torch
import torch.tensor as T

from model_util import generate_model_data, feed_inputs_sequentially_return_tuple


# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


def draw_from_uniform(parameter_intervals, N):
    new_dict = {}
    for i, k in enumerate(parameter_intervals):
        low = parameter_intervals[k][0]; high = parameter_intervals[k][1]
        new_dict[k] = torch.FloatTensor(np.random.uniform(low, high, N))
    return new_dict


def randomise_parameters(initial_parameters, coeff=torch.tensor(0.5), N_dim=None):
    res = initial_parameters.copy()
    for key in initial_parameters.keys():
        # rand_sign = round(torch.randn())*2 -1
        if N_dim is not None:
            rand_sign = 2 * torch.randint(0, 1, (1,))[0] - 1
            res[key] = res[key] + rand_sign * coeff * torch.randn((int(N_dim),)) * res[key]
        else:
            rand_sign = 2 * torch.randint(0, 1, (1,))[0] - 1
            res[key] = res[key] + rand_sign * coeff * torch.randn((1,))[0] * res[key]

    return res


def zip_dicts(a, b):
    res = a.copy()
    for key in b.keys():
        if key in a.keys():
            res[key].append(b[key])
        else:
            res[key] = b[key]
    return res


def zip_tensor_dicts(a, b):
    res = a.copy()
    for key in b.keys():
        if key in a.keys():
            res[key] = torch.cat((res[key], b[key]))
        else:
            res[key] = b[key]
    return res


# Assumes rate in Hz

def micro_gif_input(t, N, neurons_coeff):
    # return sine_modulated_white_noise(t, N, neurons_coeff)
    # return strong_sine_modulated_white_noise(t, N, neurons_coeff)
    return sine_input(t, N, neurons_coeff)


def sine_modulated_white_noise(t, N, neurons_coeff):
    # B sin(ωt) · (1 + qξ(t))
    omega = (3.141592/1200.)
    ret = torch.tensor(neurons_coeff * torch.ones((1, N)) * torch.sin(omega * torch.reshape(torch.arange(0, t), (t, 1))) * (torch.ones((t, N)) + 4. * torch.randn((t, N))), requires_grad=True)
    assert ret.shape[0] == t, "ret.shape[0] should be t, {}, {}".format(ret.shape[0], t)
    assert ret.shape[1] == N, "ret.shape[1] should be N, {}, {}".format(ret.shape[1], N)
    return ret


def strong_sine_modulated_white_noise(t, N, neurons_coeff):
    # B sin(ωt) · (1 + qξ(t))
    omega = (3.141592 / 1200.)
    ret = torch.tensor(neurons_coeff * torch.ones((1, N)) * torch.sin(omega * torch.reshape(torch.arange(0, t), (t, 1))) * (torch.ones((t, N)) + 0.5 * torch.randn((t, N))), requires_grad=True)
    assert ret.shape[0] == t, "ret.shape[0] should be t, {}, {}".format(ret.shape[0], t)
    assert ret.shape[1] == N, "ret.shape[1] should be N, {}, {}".format(ret.shape[1], N)
    return ret


def sine_modulated_input(input):
    t = input.shape[0]
    arange = torch.reshape(torch.arange(0, t), (t, 1))
    period_ms = t/8
    period_rads = (np.pi / period_ms)
    sine_inputs = torch.sin(input * arange * period_rads)
    return sine_inputs


def sine_input(t, N, neurons_coeff, period=1200.):
    # B sin(ωt) · (1 + qξ(t))
    omega = (3.141592/period)
    ret = torch.tensor(neurons_coeff * torch.ones((1, N)) * torch.sin(omega * torch.reshape(torch.arange(0, t), (t, 1))), requires_grad=True)
    assert ret.shape[0] == t, "ret.shape[0] should be t, {}, {}".format(ret.shape[0], t)
    assert ret.shape[1] == N, "ret.shape[1] should be N, {}, {}".format(ret.shape[1], N)
    return ret

# =============================

def generate_sum_of_sinusoids(t=120, period_ms=40, A_coeff = torch.rand((4,)), phase_shifts=torch.rand((4,))):
    period_rads = (np.pi / period_ms)
    return (A_coeff * torch.sin(phase_shifts + period_rads * torch.reshape(torch.arange(0, t), (t, 1)))).sum(dim=1)

def white_noise_sum_of_sinusoids(t=120, A_coeff = torch.rand((4,)), phase_shifts=torch.rand((4,))):
    period_ms = t / 3
    period_ms = torch.tensor([period_ms, period_ms / 2, period_ms / 3, period_ms / 4])

    period_rads = (np.pi / period_ms)
    white_noise = 0.1*torch.rand((t, 1))
    arange = torch.reshape(torch.arange(0, t), (t, 1))
    # return (A_coeff * torch.sin(phase_shifts + period_rads * (white_noise+arange))).sum(dim=1)
    return 0.2 * (A_coeff * torch.sin(phase_shifts + period_rads * (white_noise+arange))).sum(dim=1)

# low-pass filter
def auto_encode_input(inputs, tau_filter=20.):
    outputs = inputs[0,:]/tau_filter
    outputs = torch.vstack([outputs, outputs])
    for t_i in range(inputs.shape[0]-1):
        dv_out = (-outputs[-1, :] + inputs[t_i, :]) / tau_filter
        out_next = outputs[-1, :] + dv_out
        outputs = torch.vstack([outputs, out_next])
    return outputs[1:,:]

A_coeff_1 = torch.randn((4,))
A_coeff_2 = torch.randn((4,))
phase_shifts_1 = torch.rand((4,))
phase_shifts_2 = phase_shifts_1 + torch.rand((4,))

def get_interesting_inputs(t, N):
    inputs_1 = white_noise_sum_of_sinusoids(t=t, A_coeff=A_coeff_1, phase_shifts=phase_shifts_1)
    inputs_2 = white_noise_sum_of_sinusoids(t=t, A_coeff=A_coeff_2, phase_shifts=phase_shifts_2)

    current_inputs = torch.vstack([inputs_1, inputs_2])
    for _ in range(N - 2):
        current_inputs = torch.vstack([current_inputs, torch.rand((1, t)).clamp(0., 1.)])
    return current_inputs.T

# =============================

def release_computational_graph(model, inputs=None):
    if model is not None:
        model.reset()
    if inputs is not None and hasattr(inputs, 'grad'):
        inputs.grad = None


def generate_synthetic_data(gen_model, t, neurons_coeff, burn_in=False):
    gen_model.reset()
    if burn_in:
        gen_input = sine_modulated_white_noise(t=int(t/10), N=gen_model.N, neurons_coeff=neurons_coeff)
        _ = generate_model_data(model=gen_model, inputs=gen_input)
    gen_input = sine_modulated_white_noise(t=t, N=gen_model.N, neurons_coeff=neurons_coeff)
    gen_spiketrain = generate_model_data(model=gen_model, inputs=gen_input)
    gen_spiketrain = torch.round(gen_spiketrain)
    gen_spiketrain.grad = None

    return gen_spiketrain.clone().detach(), gen_input.clone().detach()


def generate_synthetic_data_tuple(gen_model, t, burn_in=False):
    gen_model.reset()
    if burn_in:
        gen_input = get_interesting_inputs(t=int(t/10), N=gen_model.N)
        _, _ = feed_inputs_sequentially_return_tuple(model=gen_model, inputs=gen_input)
    gen_input = get_interesting_inputs(t=t, N=gen_model.N)
    _, gen_spiketrain = feed_inputs_sequentially_return_tuple(model=gen_model, inputs=gen_input)
    # for gen spiketrain this may be thresholded to binary values:
    gen_spiketrain = torch.round(gen_spiketrain)
    gen_spiketrain.grad = None

    return gen_spiketrain.clone().detach(), gen_input.clone().detach()


def train_test_split(data, train_test_split_factor=0.85):
    assert data.shape[0] > data.shape[1], "assert row as timestep matrix. data.size: {}".format(data.size)
    splice_index = int(train_test_split_factor * data.size(0))
    train_data = data[:splice_index]
    test_data = data[splice_index:]
    return train_data, test_data


def train_val_test_split(data, train_factor=0.6, val_factor=0.2, test_factor=0.2):
    f_sum = (train_factor + val_factor + test_factor)
    assert f_sum == 1, "the factors must sum to one. sum:{}".format(f_sum)

    splice_train_index = int(train_factor*data.size(0))
    splice_val_index = int(val_factor*data.size(0))

    train_data = data[:splice_train_index]
    val_data = data[splice_train_index:splice_val_index]
    test_data = data[(splice_train_index+splice_val_index):]

    return train_data, val_data, test_data
