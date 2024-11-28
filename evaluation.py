import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from MLP import forward


# e = np.array([1,1,0,2,1,0])
# p = np.array([1,1,2,1,1,0])
#
#
# def test_NN(w, b, x, y, title):
#     x = np.array(x)
#     y = np.array(y)
#     output = (x @ w + b).reshape(-1)
#     output = np.where(output >= 0, 1, -1)
#     confusion_matrix(y, output, title)
#
# def h(w, x, b):
#     x = x.reshape(-1, 1)
#     return (w.T @ x + b).item()
#
# def predict(w, b, x_test, y_test, add_bias):
#     b = np.array(b)
#     x_test = np.array(x_test)
#     y_test = np.array(y_test)
#     # print(f"\ny_test shape: {y_test.shape}")
#     # print(f"w shape: {w.shape}")
#     # print(f"b shape: {b.shape}")
#     # print(f"\nx_test shape: {x_test.shape}")
#     y_pred = (x_test @ w + b).reshape(-1)
#     predicted_classes = np.where(y_pred < 0, -1, 1)
#     accuracy = np.mean(predicted_classes == y_test)
#     actual_val = predicted_classes.tolist()
#
#     print(f"Accuracy: {accuracy}")
#     return accuracy, actual_val

def evaluate(x_test, weights,bias_flag, sigmoid0_tangent1):
    actual_class =[]
    for test_element in x_test:
        neurons = forward(test_element, weights,bias_flag, sigmoid0_tangent1)
        output_layer = neurons[-1]
        output_values = output_layer[:3]
        #print(output_values)
        actual_class.append(output_values.index(max(output_values)))
    return actual_class


def confusion_matrix(expected, predicted, title):
    conf_matrix = np.zeros((3, 3), dtype=int)
    true_predictions = 0

    for i in range(len(expected)):
        conf_matrix[expected[i]][predicted[i]] += 1
        if expected[i] == predicted[i]:
            true_predictions += 1

    total_predictions = len(expected)
    accuracy = float(true_predictions) / float(total_predictions)

    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Class A', 'Class B', 'Class C'], yticklabels=['Class A', 'Class B', 'Class C'])

    plt.gca().xaxis.set_label_position('top')
    plt.gca().xaxis.tick_top()

    plt.xlabel('Prediction', labelpad=15)
    plt.ylabel('Expected', labelpad=15)
    plt.title(title, pad=20, fontsize=16)
    plt.figtext(0.5, 0.02, f"Accuracy: {accuracy:.2f}", ha="center", fontsize=14, color="black")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

