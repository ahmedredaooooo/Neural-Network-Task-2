
from preprocessing import *
from Main import *

FEATURES = 5
CLASSES = 3 

def run_gui():
    return

run_gui()

def on_submit():
    hidden_layers = [] 
    eta, epochs, bias_flag, sigmoid0_tangent1 = 0
    
    main(hidden_layers, eta, epochs, bias_flag, sigmoid0_tangent1)