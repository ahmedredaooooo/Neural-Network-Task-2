import numpy as np
import math
from MLP import forward
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def tangent(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

def y_actual(inputs, weights,bias_flag, sigmoid0_tangent1):
    neourons = forward(inputs, weights,bias_flag, sigmoid0_tangent1)
    output_layer = neourons[-1]
    output_values = output_layer[:3]
    actual_class = output_values.index(max(output_values))
    return actual_class
