from preprocessing import *
from MLP import *
from evaluation import *
def main(hidden_layers, eta, epochs, bias_flag, sigmoid0_tangent1):
    #print(hidden_layers," ",eta," ",epochs," ",bias_flag," ",sigmoid0_tangent1)
    x_train, x_test, y_train, y_test = preprocess()
    weights = mlp(x_train,y_train, hidden_layers, eta, epochs, bias_flag, sigmoid0_tangent1)
    #print(weights)
    y_pred = evaluate(x_train, weights, bias_flag, sigmoid0_tangent1)
    confusion_matrix(y_train, y_pred, "TRAIN")

    y_pred = evaluate(x_test, weights, bias_flag, sigmoid0_tangent1)
    confusion_matrix(y_test, y_pred, "TEST")

    return