import math
from sklearn.metrics import mean_squared_error, log_loss
from homemade.utils.classess import Array
from homemade.utils.cost_functions import calculate_mean_squared_error, calculate_binary_cross_entropy

def test_mean_squared_error():
    y_true = [3, -0.5, 2, 7, 4.2]
    y_predict = [2.5, 0.0, 2, 8, 4.1]
    mse = calculate_mean_squared_error(y_true=Array(y_true), y_predict=Array(y_predict))
    skl_mse = mean_squared_error(y_true,y_predict)
    assert math.isclose(mse,skl_mse)

def test_binary_cross_entrophy():
    y_true = [1, 0, 1, 0]
    y_predict = [0.9, 0.1, 0.8, 0.4] 
    bce = calculate_binary_cross_entropy(y_true=Array(y_true), y_predict=Array(y_predict))
    skl_bce = log_loss(y_true, y_predict)
    assert math.isclose(bce,skl_bce)   


