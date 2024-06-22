from math import log

def mean_squared_error(y_true: list, y_predict: list) ->float:
    mse = sum((y_true[i] - y_predict[i])**2 for i in range(len(y_true))) / len(y_true)
    return mse

def binary_cross_entrophy(y_true: list, y_predict: list) ->float:
    bce = (-1/len(y_true)) * sum(y_true[i] * log(y_predict[i]) + (1 - y_true[i]) * log(1 - y_predict[i]) 
                                 for i in range(len(y_true)))
    return bce 
