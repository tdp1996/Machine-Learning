from linear_regression.main import train_LinearRegression, LinearRegression_model

def test_LinearRegression():
    X_train = [[1,2,3,4],[5,4,7,8]]
    y_train = [11 ,13]
    learning_rate = 0.001
    output = train_LinearRegression(X_train=X_train, y_train=y_train, learning_rate=learning_rate)

    assert isinstance(output,tuple)
    assert len(output[0]) == 4
    assert isinstance(output[1], float)

def test_LinearRegression_model():
    X = [1,2,3,4]
    weights = [0.5]
    bias = 0.1
    output = LinearRegression_model(X=X, weights=weights, bias=bias)
    assert isinstance(output,list)
