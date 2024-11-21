
FEATURES = 5
CLASSES = 3 


def mlp(data,hidden_layers, eta, epochs, bias_flag, sigmoid0_tangent1):

    layers = []

    for epoch in range(epochs):
        for row in data:
            activation_output = forward(row, wights, bias,bias_flag, sigmoid0_tangent1)
            wights,bias=backward(row, activation_output, wights, bias, bias_flag, eta, sigmoid0_tangent1)
    return wights,bias

def forward(row, wights, bias,bias_flag, sigmoid0_tangent1):
    return

def backward(row, activation_output, wights, bias, bias_flag, eta, sigmoid0_tangent1):
    
    return