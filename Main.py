from preprocessing import *
from MLP import *

def main(hidden_layers, eta, epochs, bias_flag, sigmoid0_tangent1):
    modified_data = preprocess()
    mlp(modified_data,hidden_layers, eta, epochs, bias_flag, sigmoid0_tangent1)
    return