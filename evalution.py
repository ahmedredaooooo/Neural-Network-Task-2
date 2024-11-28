import tkinter as tk
from tkinter import ttk
import numpy as np


def confusion_matrix(expected, result, title):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    true_predictions = 0

    for i in range(expected.shape[0]):
        if result[i] == expected[i]:
            if result[i] == 1:
                true_positive += 1
            else:
                true_negative += 1
            true_predictions += 1
        else:
            if result[i] == 1:
                false_positive += 1
            else:
                false_negative += 1

    root = tk.Toplevel()
    root.title(title)

    columns = ('', 'Predicted Positive', 'Predicted Negative')
    tree = ttk.Treeview(root, columns=columns, show='headings')

    tree.heading('', text='')
    tree.heading('Predicted Positive', text='Predicted Positive')
    tree.heading('Predicted Negative', text='Predicted Negative')

    tree.insert('', tk.END, values=('Actual Positive', true_positive, false_negative))
    tree.insert('', tk.END, values=('Actual Negative', false_positive, true_negative))

    tree.pack(expand=True, fill='both')

    Accuracy = float(true_predictions) / float(result.shape[0])
    accuracy_label = tk.Label(root, text=f"Accuracy: {Accuracy}")
    accuracy_label.pack(pady=10)

    return root


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