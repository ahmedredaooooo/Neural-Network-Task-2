import numpy as np
from MLP import forward


def y_actual(inputs, weights,bias_flag, sigmoid0_tangent1):
    neourons = forward(inputs, weights,bias_flag, sigmoid0_tangent1)
    output_layer = neourons[-1]
    output_values = output_layer[:3]
    print(output_values)
    actual_class = output_values.index(max(output_values))
    return actual_class

weights = [
    [[0.1, 0.2, 0.3, 0.1], [0.2, 0.3, 0.4, 0.2]],  # Layer 1
    [[0.8, 0.7, 0.6], [0.9, 0.8, 0.7], [0.4, 0.3, 0.1]]  # Output layer
]

inputs = [0.5, 0.8, 0.2]
output = y_actual(inputs, weights, 1, 0)
print(output)