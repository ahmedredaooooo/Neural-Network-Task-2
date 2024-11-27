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
    #expected should be (0,1,2) the encoded value for the class ex: 1
    #activation should be ...
    #weights should be ...
    #sigmoid0_tangent1 should be (0, 1) ex: 0

    input_layer = np.array([1, 0, 1])
    expected = 1
    activation_outputs = np.array([[0.6682, 0.7311, 1], [0.6478, 0.7099, 0.7765, 1]])
    weights = np.array([[[0.1, 0.3, 0.5, 0.1], [0.2, 0.4, 0.6, 0.2]], [[0.7, 0.1, 0.1], [0.8, 0.3, 0.2], [0.9, 0.5, 0.3]]])
    sigmoid0_tangent1 = 0
    t = [0] * CLASSES
    t[expected] = 1
    #activation_outputs = [[1, 0.731, 0.5], [0.4711]]
    #weights = np.array([[[0.5,-0.5,0,1], [-0.5,0,0,0.5]], [[0, -0.5,0.5]]])


    def sigmoid_dash(layer_idx, neuron_idx):
        return activation_outputs[layer_idx][neuron_idx] * (1 - activation_outputs[layer_idx][neuron_idx])
    def tangent_dash(layer_idx, neuron_idx):
        return (1 - activation_outputs[layer_idx][neuron_idx]) * (1 + activation_outputs[layer_idx][neuron_idx])

    f_dash = tangent_dash if sigmoid0_tangent1 else sigmoid_dash
    signal_errors = copy.deepcopy(activation_outputs)

    for it in range(len(signal_errors)):
        for jt in range(len(signal_errors[it])):
            signal_errors[it][jt] = 0.0
    print(signal_errors)
    layers = len(activation_outputs)
    for j in range(len(activation_outputs[-1]) - 1):
        signal_errors[-1][j] = f_dash(-1, j) * (t[j] - activation_outputs[-1][j])
    for i in range(layers - 2, -1, -1):
        for j in range(len(activation_outputs[i]) - 1):
            ii = i + 1
            sigma = 0
            for jj in range(len(activation_outputs[ii]) - 1):
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
