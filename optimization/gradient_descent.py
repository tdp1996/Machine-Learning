from typing import Union

def gradient_descent(X_train: list[list[Union[float,int]]], 
                     y_train:list[Union[float,int]], 
                     y_predict: list[Union[float,int]], 
                     weights: list[Union[float,int]], 
                     bias: Union[float,int], 
                     learning_rate: float) ->tuple[list[float],float]:
    """
    Calculates the gradients for weights and bias.

    Args:
        X_train (list[list[Union[float,int]]]): Training data. Each inner list represents the features of one training example.
        y_train (list[Union[float,int]]): Target values corresponding to the training data.
        y_predict (list[Union[float,int]]): Predicted values corresponding to the training data.
        weights (list[Union[float,int]]): initial values of weights
        bias (Union[float,int]): initial value of bias
        learning_rate (float): The learning rate for gradient descent optimization.

    Returns:
        tuple[list[float], float]: A tuple containing the gradients for weights (list of floats) and bias (float).
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


