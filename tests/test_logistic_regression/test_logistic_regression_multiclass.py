from logistic_regression.logistic_regression_multiclass import train_LogisticRegression_multiclass, LogisticRegression_multiclass_model

def test_train_LogisticRegression_multiclass():
    X_train = [
        [0.5, 1.5],
        [2.0, 1.0],
        [1.0, 2.5],
        [1.5, 1.0],
        [2.5, 2.0],
        [3.0, 3.5]
    ]
    y_train = [0, 1, 2, 1, 0, 2]
    learning_rate = 0.01
    iterations = 10
    output = train_LogisticRegression_multiclass(X_train=X_train,y_train=y_train,learning_rate=learning_rate,iterations=iterations)
    assert all(isinstance(output[i],tuple) for i in range(len(output)))


def test_LogisticRegression_multiclass_model():
    X = [
        [0.5, 1.5],
        [2.0, 1.0],
        [1.0, 2.5],
        [1.5, 1.0],
        [2.5, 2.0],
        [3.0, 3.5]
    ]
    model_params = [([0.5,0.5],0.5), ([0.1,0.1],0.1), ([0.3,0.3],0.3)]
    classess = [0,1,2]
    output = LogisticRegression_multiclass_model(X=X,model_params=model_params,classess=classess)

    assert isinstance(output,list)
    assert all(output[i] in classess for i in range(len(output)))