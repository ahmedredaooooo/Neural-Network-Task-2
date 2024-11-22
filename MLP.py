import copy
import numpy as np
FEATURES = 5
CLASSES = 3


def mlp(data,hidden_layers, eta, epochs, bias_flag, sigmoid0_tangent1):

    layers = []
    weights = []  # --> should contatin the bias also if bias_flag true
    for epoch in range(epochs):
        for row in data:
            activation_outputs = forward(row, weights,sigmoid0_tangent1)
            weights=backward(row, activation_outputs, weights, sigmoid0_tangent1)
            weights_update(row, activation_outputs, weights, eta)
    return weights

def forward(input_layer, weights, bias,bias_flag, sigmoid0_tangent1):
    return

def backward(input_layer, expected, activation_outputs, weights, sigmoid0_tangent1):
    input_layer = [1, 1, 0, 1]
    #t = [0] * CLASSES
    #t[expected] = 1
    activation_outputs = np.array([[1, 0.731, 0.5], [0.4711]])
    weights = np.array([[[0.5,-0.5,0,1], [-0.5,0,0,0.5]], [[0, -0.5,0.5]]])
    def sigmoid_dash(layer_idx, neuron_idx):
        return activation_outputs[layer_idx][neuron_idx] * (1 - activation_outputs[layer_idx][neuron_idx])
    def tangent_dash(layer_idx, neuron_idx):
        return (1 - activation_outputs[layer_idx][neuron_idx]) * (1 + activation_outputs[layer_idx][neuron_idx])

    f_dash = tangent_dash if sigmoid0_tangent1 else sigmoid_dash
    signal_errors = copy.deepcopy(activation_outputs)

    for it in range(len(signal_errors)):
        for jt in range(len(signal_errors[it])):
            signal_errors[it][jt] = 0

    layers = len(activation_outputs)
    for j in range(len(activation_outputs[-1])):
        signal_errors[-1][j] = f_dash(-1, j) * (1 - activation_outputs[-1][j])
    for i in range(layers - 2, -1, -1):
        for j in range(len(activation_outputs[i])):
            ii = i + 1
            sigma = 0
            for jj in range(len(activation_outputs[ii])):
                sigma += signal_errors[ii][jj] * weights[ii][jj][j]
            signal_errors[i][j] = f_dash(i, j) * sigma
    print(signal_errors)
    return



def weights_update(row, activation_outputs, weights, eta):
    return

def sigmoid():
    return

def tangent():
    return
