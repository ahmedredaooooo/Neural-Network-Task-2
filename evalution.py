from MLP import forward


def y_actual(x_test, weights,bias_flag, sigmoid0_tangent1):
    actual_class =[]
    for test_element in x_test:
        neurons = forward(test_element, weights,bias_flag, sigmoid0_tangent1)
        output_layer = neurons[-1]
        output_values = output_layer[:3]
        print(output_values)
        actual_class.append(output_values.index(max(output_values)))
    return actual_class


