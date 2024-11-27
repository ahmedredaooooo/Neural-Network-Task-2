from preprocessing import *
from MLP import *

def main(hidden_layers, eta, epochs, bias_flag, sigmoid0_tangent1):
    print(hidden_layers," ",eta," ",epochs," ",bias_flag," ",sigmoid0_tangent1)
    x_train, x_test, y_train, y_test = preprocess()
    weights,activation_outputs = mlp(x_train,y_train,  x_test,y_test, hidden_layers, eta, epochs, bias_flag, sigmoid0_tangent1)
    print(weights,activation_outputs)
    return