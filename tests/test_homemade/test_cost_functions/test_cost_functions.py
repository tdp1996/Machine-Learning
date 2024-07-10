import math
import sklearn.metrics 
from homemade.cost_functions.cost_functions import mean_squared_error, binary_cross_entrophy

def test_mean_squared_error():
    y_true = [3, -0.5, 2, 7, 4.2]
    y_predict = [2.5, 0.0, 2, 8, 4.1]
    mse = mean_squared_error(y_true=y_true, y_predict=y_predict)
    skl_mse = sklearn.metrics.mean_squared_error(y_true,y_predict)
    assert math.isclose(mse,skl_mse)

def test_binary_cross_entrophy():
    y_true = [1, 0, 1, 0]
    y_predict = [0.9, 0.1, 0.8, 0.4] 
    bce = binary_cross_entrophy(y_true=y_true, y_predict=y_predict)
    skl_bce = sklearn.metrics.log_loss(y_true, y_predict)
    assert math.isclose(bce,skl_bce)   


