import torch


def feed_inputs_sequentially_return_tuple(model, inputs):
    print('Feeding {} inputs sequentially through SNN in time'.format(inputs.size(0)))
    membrane_potentials, model_spiketrain = model(inputs[0])
    for x_in in inputs[1:]:
        v, spikes = model(x_in)
        membrane_potentials = torch.vstack([membrane_potentials, v])
        model_spiketrain = torch.vstack([model_spiketrain, spikes])

    return membrane_potentials, model_spiketrain


def feed_inputs_sequentially_return_args(model, inputs):
    print('Feeding {} inputs sequentially through SNN in time'.format(inputs.size(0)))
    s_lambdas, model_spiketrain, vs = model(inputs[0])
    for x_in in inputs[1:]:
        s_lambda, spikes, v = model(x_in)
        s_lambdas = torch.vstack([s_lambdas, s_lambda])
        model_spiketrain = torch.vstack([model_spiketrain, spikes])
        vs = torch.vstack([vs, v])

    return s_lambdas, model_spiketrain, vs


def feed_inputs_sequentially_return_spike_train(model, inputs):
    print('Feeding {} inputs sequentially through SNN in time'.format(inputs.shape[0]))
    model_spiketrain = model(inputs[0])
    for x_in in inputs[1:]:
        spikes = model(x_in)
        model_spiketrain = torch.vstack([model_spiketrain, spikes])

    return model_spiketrain


def feed_inputs_sequentially_return_membrane_potentials(model, inputs):
    print('Feeding {} inputs sequentially through SNN in time'.format(inputs.shape[0]))
    membrane_potentials, _ = model(inputs[0])
    for x_in in inputs[1:]:
        v, _ = model(x_in)
        membrane_potentials = torch.vstack([membrane_potentials, v])

    return membrane_potentials


def generate_model_data(model, inputs):
    print('Simulating model with provided input')

    model_spiketrain = feed_inputs_sequentially_return_spike_train(model, inputs)

    print('Simulated model data for t =', inputs.shape[0], 'returning spiketrain')

    return model_spiketrain
