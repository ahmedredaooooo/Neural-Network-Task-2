
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

def backward(input_layer, activation_output, weights, bias, bias_flag, eta, sigmoid0_tangent1):
    return

def weights_update(row, activation_outputs, weights, eta):
    return

def sigmoid():
    return

def tangent():
    return