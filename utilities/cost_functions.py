
def mean_squared_error(y_true: list, y_predict: list) ->float:
    mse = sum((y_true[i] - y_predict[i])**2 for i in range(len(y_true))) / len(y_true)
    return mse
