
import numpy as np
import tkinter as tk
from tkinter import ttk

from MLP import forward


e = np.array([1,1,0,2,1,0])
p = np.array([1,1,2,1,1,0])


def test_NN(w, b, x, y, title):
    x = np.array(x)
    y = np.array(y)
    output = (x @ w + b).reshape(-1)
    output = np.where(output >= 0, 1, -1)
    confusion_matrix(y, output, title)

def h(w, x, b):
    x = x.reshape(-1, 1)
    return (w.T @ x + b).item()

def predict(w, b, x_test, y_test, add_bias):
    b = np.array(b)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    # print(f"\ny_test shape: {y_test.shape}")
    # print(f"w shape: {w.shape}")
    # print(f"b shape: {b.shape}")
    # print(f"\nx_test shape: {x_test.shape}")
    y_pred = (x_test @ w + b).reshape(-1)
    predicted_classes = np.where(y_pred < 0, -1, 1)
    accuracy = np.mean(predicted_classes == y_test)
    actual_val = predicted_classes.tolist()

    print(f"Accuracy: {accuracy}")
    return accuracy, actual_val

def y_actual(x_test, weights,bias_flag, sigmoid0_tangent1):
    actual_class =[]
    for test_element in x_test:
        neurons = forward(test_element, weights,bias_flag, sigmoid0_tangent1)
        output_layer = neurons[-1]
        output_values = output_layer[:3]
        print(output_values)
        actual_class.append(output_values.index(max(output_values)))
    return actual_class

def confusion_matrix(expected, predicted, title):
    conf_matrix = np.zeros((3, 3), dtype=int)
    true_predictions = 0

    for i in range(expected.shape[0]):
        conf_matrix[expected[i]][predicted[i]] += 1
        if expected[i] == predicted[i]:
            true_predictions += 1

    root = tk.Toplevel()
    root.title(title)

    columns = ('', 'Predicted Class 0', 'Predicted Class 1', 'Predicted Class 2')
    tree = ttk.Treeview(root, columns=columns, show='headings')

    tree.heading('', text='Expected\\Predicted')
    tree.heading('Predicted Class 0', text='Class 0')
    tree.heading('Predicted Class 1', text='Class 1')
    tree.heading('Predicted Class 2', text='Class 2')

    tree.insert('', tk.END, values=('Expected Class 0', conf_matrix[0][0], conf_matrix[0][1], conf_matrix[0][2]))
    tree.insert('', tk.END, values=('Expected Class 1', conf_matrix[1][0], conf_matrix[1][1], conf_matrix[1][2]))
    tree.insert('', tk.END, values=('Expected Class 2', conf_matrix[2][0], conf_matrix[2][1], conf_matrix[2][2]))

    tree.pack(expand=True, fill='both')

    total_predictions = expected.shape[0]
    accuracy = float(true_predictions) / float(total_predictions)
    accuracy_label = tk.Label(root, text=f"Accuracy: {accuracy:.2f}")
    accuracy_label.pack(pady=10)
    root.mainloop()


confusion_matrix(e,p,"ahmed redaoooooooooooooo")

