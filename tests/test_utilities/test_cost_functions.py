import math
import sklearn.metrics 
from utilities.cost_functions import mean_squared_error

def test_mean_squared_error():
    y_true = [3, -0.5, 2, 7, 4.2]
    y_predict = [2.5, 0.0, 2, 8, 4.1]
    mse = mean_squared_error(y_true=y_true, y_predict=y_predict)
    skl_mse = sklearn.metrics.mean_squared_error(y_true,y_predict)
    assert math.isclose(mse,skl_mse)


