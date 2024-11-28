
from preprocessing import *
from Main import *
import tkinter as tk
from tkinter import ttk

FEATURES = 5
CLASSES = 3


def on_submit():
    hidden_layers = []
    #eta, epochs, bias_flag, sigmoid0_tangent1 = 0

    neurons_list = list(map(int, neurons.get().split(',')))
    bias_flag = 0
    if bias_var.get() == True:
        bias_flag = 1
    else:
        bias_flag = 0
    main(neurons_list, float(eta.get()), int(epochs.get()), bias_flag,activation_function.get()=="Hyperbolic Tangent" )




root = tk.Tk()
root.title("Task 2")

tk.Label(root, text="Enter number of hidden layers:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
hidden_layers = tk.Entry(root)
hidden_layers.grid(row=0, column=1, padx=5, pady=5)

tk.Label(root, text="Enter neurons in each hidden layer (comma-separated):").grid(row=1, column=0, sticky="w",
                                                                                  padx=5, pady=5)
neurons = tk.Entry(root)
neurons.grid(row=1, column=1, padx=5, pady=5)

tk.Label(root, text="Enter learning rate:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
eta = tk.Entry(root)
eta.grid(row=2, column=1, padx=5, pady=5)

tk.Label(root, text="Enter number of epochs:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
epochs = tk.Entry(root)
epochs.grid(row=3, column=1, padx=5, pady=5)

bias_var = tk.BooleanVar()
bias_checkbox = tk.Checkbutton(root, text="Add bias", variable=bias_var)
bias_checkbox.grid(row=4, columnspan=2, pady=5)


tk.Label(root, text="Choose activation function:").grid(row=5, column=0, sticky="w", padx=5, pady=5)
activation_function = tk.StringVar(value="Sigmoid")

radio_frame = tk.Frame(root)
radio_frame.grid(row=5, column=1, sticky="w")
tk.Radiobutton(radio_frame, text="Sigmoid", variable=activation_function, value="Sigmoid").pack(side="left", padx=5)
tk.Radiobutton(radio_frame, text="Hyperbolic Tangent", variable=activation_function, value="Hyperbolic Tangent").pack(side="left", padx=5)

submit_button = tk.Button(root, text="Submit", command=on_submit)
submit_button.grid(row=6, columnspan=2, pady=10)

root.mainloop()
