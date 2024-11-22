
FEATURES = 5
CLASSES = 3 
import math
import numpy as np

def mlp(data,hidden_layers, eta, epochs, bias_flag, sigmoid0_tangent1):

    activation_outputs = []
    weights = weight_initialize(hidden_layers)   # --> should contatin the bias also if bias_flag true
    for epoch in range(epochs):
        for row in data:
            activation_outputs = forward(row, weights,bias_flag,sigmoid0_tangent1)
            #weights=backward(row, activation_outputs, weights, sigmoid0_tangent1)
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
        result = np.hstack((x, new_column))
        weights.append(result)

    x = np.random.rand(3, hidden_layers[len(hidden_layers)-1])
    result = np.hstack((x, new_column))
    weights.append(result)
    #print(weights)
    return weights

def backward(input_layer, activation_output, weights, bias, bias_flag, eta, sigmoid0_tangent1):
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
    
layers = np.array([4,2,3])
data = np.array([[1,2,3,4,5]])
w,a =mlp(data,layers,1,1,1,False)
print(a)