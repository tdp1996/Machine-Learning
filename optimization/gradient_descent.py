
def gradient_descent(X_train: list[list], y_train:list, y_predict: list, weights: list, bias: float, learning_rate: float) ->tuple[list,float]:
    """
    Calculates the gradients for weights and bias.

    Args:
        X_train (list[list]): Training data. Each inner list represents the features of one training example.
        y_train (List[float]): Target values corresponding to the training data.
        y_predict (List[float]): Predicted values corresponding to the training data.

    Returns:
        Tuple[List[float], float]: A tuple containing the gradients for weights (list of floats) and bias (float).
    """

    numb_features = len(X_train[0])
    numb_samples = len(y_train)
    weight_derivative = [0] * numb_features
    bias_derivative = 0
    for i in range(numb_samples):
        error = y_train[i] - y_predict[i]
        for j in range(numb_features):
            weight_derivative[j] += (-2 / numb_samples) * error * X_train[i][j]
        bias_derivative += (-2 / numb_samples) * error

    # Update weights and bias
    weights = [weights[j] - (learning_rate * weight_derivative[j]) for j in range(numb_features)]
    bias -= learning_rate * bias_derivative

    return weights, bias


