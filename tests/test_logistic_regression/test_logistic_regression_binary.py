from logistic_regression.logistic_regression_binary import train_LogisticRegression_binary, LogisticRegression_binary_model

def test_train_LogisticRegression():
    X_train = [[1,2,4,5], [7,8,4,2], [2,3,4,8]]
    y_train = [0,1,0]
    learning_rate = 0.001
    iterations = 100

    output = train_LogisticRegression_binary(X_train=X_train, y_train=y_train, learning_rate=learning_rate, iterations=iterations)
    assert isinstance(output,tuple)
    assert len(output[0]) == 4
    assert isinstance(output[1], float)

def test_LogisticRegression_model():
    X = [1,2,3,4]
    weights = [0.5]
    bias = 0.1
    output = LogisticRegression_binary_model(X=X, weights=weights, bias=bias)

    assert isinstance(output,list)
    assert all(output[i] in [0, 1] for i in range(len(output)))
