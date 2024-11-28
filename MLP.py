import copy
import numpy as np
FEATURES = 5
CLASSES = 3
import math
import numpy as np

def mlp(x_train,y_train,  x_test,y_test,hidden_layers, eta, epochs, bias_flag, sigmoid0_tangent1):

    activation_outputs = []
    number_of_rows = len(y_train)
    weights = weight_initialize(hidden_layers)   # --> should contatin the bias also if bias_flag true
    for epoch in range(epochs):
        for i in range(number_of_rows):
            print("Start")
            activation_outputs = forward(x_train[i], weights,bias_flag,sigmoid0_tangent1)
            print("activation_output")
            print(activation_outputs)
            print("Done mine")
            signal_error=backward(x_train[i], y_train[i], activation_outputs, weights, sigmoid0_tangent1)
            #weights_update(row, activation_outputs, weights, eta)
    return weights,activation_outputs

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def tangent(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))


def weight_initialize(hidden_layers):
    #NOTE!!!! FISRT index represents the number of {{ input layer features}}
    #hidden_layers = np.array([4,2,3]) # first index isn't one of the hidden layers
    hidden_layers = np.append(np.array([FEATURES]),hidden_layers)

    weights = []
    for i in range(len(hidden_layers)-1):
        x = np.random.rand(hidden_layers[i+1], hidden_layers[i])

        new_column = np.ones((x.shape[0], 1))
        # Add the new column to the array
        result = np.hstack((x, new_column))
        weights.append(result)

    x = np.random.rand(3, hidden_layers[len(hidden_layers)-1])
    #x = np.random.rand(3, 2)

    new_column = np.ones((3, 1))
    result = np.hstack((x, new_column))
    weights.append(result)
    print(weights)
    
    return weights

def backward(input_layer, expected, activation_outputs, weights, sigmoid0_tangent1):
    #expected should be (0,1,2) the encoded value for the class ex: 1
    #activation should be ...
    #weights should be ...
    #sigmoid0_tangent1 should be (0, 1) ex: 0

    input_layer = np.array([1, 0, 1])
    expected = 1
    activation_outputs = [[0.6682, 0.7311, 1], [0.6478, 0.7099, 0.7765, 1]]
    weights = [[[0.1, 0.3, 0.5, 0.1], [0.2, 0.4, 0.6, 0.2]], [[0.7, 0.1, 0.1], [0.8, 0.3, 0.2], [0.9, 0.5, 0.3]]]
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

def forward(input_layer, weights,bias_flag, sigmoid0_tangent1):
    number_of_layers = len(weights)
    output = []
    temp = []
    if sigmoid0_tangent1 == False:
        for layer_index in range(number_of_layers):
            if layer_index == 0:
                for perceptron in weights[layer_index]:
                    input_layers = np.append(np.array(input_layer),1)
                    temp.append(sigmoid(np.dot(input_layers,perceptron)))
                temp.append(bias_flag)
                output.append(temp)
            else:
                i = 1
                temp = []
                for perceptron in weights[layer_index]:
                    x = np.dot(output[layer_index-1],perceptron)
                    y =sigmoid(x)
                    temp.append(y)
                    i = i + 1
                temp.append(bias_flag)
                output.append(temp)
        return output
    else:
        for layer_index in range(number_of_layers):
            if layer_index == 0:
                for perceptron in weights[layer_index]:
                    input_layers = np.append(np.array(input_layer),1)
                    temp.append(tangent(np.dot(input_layers,perceptron)))
                temp.append(bias_flag)
                output.append(temp)
            else:
                i = 1
                temp = []
                for perceptron in weights[layer_index]:
                    x = np.dot(output[layer_index-1],perceptron)
                    y =tangent(x)
                    temp.append(y)
                    i = i + 1
                temp.append(bias_flag)                
                output.append(temp)
        return output
    
