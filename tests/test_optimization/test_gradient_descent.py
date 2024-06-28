from optimization.gradient_descent import gradient_descent

def test_gradient_descent():
    X_train = [[1,2,4,5],[3,4,6,7]]
    y_train = [6, 9]
    y_predict = [10, 6]
    weights = [0.5, 0.5, 0.5, 0.5]
    bias = 0.5
    learning_rate = 0.001
    output = gradient_descent(X_train=X_train, y_train=y_train, y_predict=y_predict, weights=weights, bias=bias, learning_rate=learning_rate)

    assert isinstance(output,tuple)
    assert isinstance(output[0],list) and len(output[0]) == len(weights)
    assert isinstance(output[1],(int,float))

    