
from preprocessing import *

FEATURES = 5
CLASSES = 3 

def run_gui():
    return

run_gui()

def on_submit():
    hidden_layers = [] 
    eta, epochs, bias_flag, sigmoid0_tangent1 = 0

    modified_data = preprocess()
    mlp(modified_data,hidden_layers, eta, epochs, bias_flag, sigmoid0_tangent1)