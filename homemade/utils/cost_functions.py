from math import log
from .classess import Array
from .analysis import calculate_mean

def mean_squared_error(y_true: Array, y_predict: Array)->Array:
    squared_error = (y_true - y_predict) ** 2
    mse = calculate_mean(squared_error)
    return mse

def binary_cross_entrophy(y_true: list, y_predict: list) ->float:
    bce = (-1/len(y_true)) * sum(y_true[i] * log(y_predict[i]) + (1 - y_true[i]) * log(1 - y_predict[i]) 
                                 for i in range(len(y_true)))
    return bce 
