import random
from typing import Union
from utilities.cost_functions import mean_squared_error

STOPPING_THRESHOLD = 1e-6

def gradient_descent(X_train: Union[list,list[list]], 
                     y_train: list, 
                     learning_rate: float, 
                     stopping_threshold: float = STOPPING_THRESHOLD) ->tuple[list,float]:
    """
    Performs gradient descent optimization to find the optimal weights and bias for linear regression.

    Args:
    X_train (list or list of lists): Training data. If a list, it represents single variable training data. 
                                     If a list of lists, it represents multiple variable training data.
    y_train (list): Target values corresponding to the training data.
    learning_rate (float): The learning rate for gradient descent optimization.
    stopping_threshold (float): The threshold for stopping the optimization when the change in cost function is below this value. Default is 1e-6.

    Returns:
    tuple: A tuple containing the final weights (list) and bias (float).
    """
    # check whether this is multiple regression or not, initialize weight
    if all(isinstance(X_i,list) for X_i in X_train):
        numb_features = len(X_train[0])
        current_weight = [random.uniform(-1,1)]*numb_features
    else:
        numb_features = 1
        X_train = [[X_i] for X_i in X_train]
        current_weight = [random.uniform(-1,1)]
    
    current_bias = random.uniform(-1,1)
    numb_samples = len(X_train)
    previous_cost = float('inf')
    iteration = 0
    
    while True:
        # Predicting y values
        y_predict = []
        for X_i in X_train:
            y_predict_i = sum(X_i[j]*current_weight[j] for j in range(numb_features)) + current_bias
            y_predict.append(y_predict_i)

        # Calculate current cost using mean squared error
        current_cost = mean_squared_error(y_train, y_predict)
        if abs(previous_cost - current_cost) <= stopping_threshold:
            break
        previous_cost = current_cost

        #calculate gradient
        weight_derivative = [0] * numb_features
        bias_derivative = 0
        for i in range(numb_samples):
            error = y_train[i] - y_predict[i]
            for j in range(numb_features):
                weight_derivative[j] += (-2 / numb_samples) * error * X_train[i][j]
            bias_derivative += (-2 / numb_samples) * error

        # Update weights and bias
        current_weight = [current_weight[j] - (learning_rate * weight_derivative[j]) for j in range(numb_features)]
        current_bias -= learning_rate * bias_derivative
        iteration += 1

    print(f"Optimization finished after {iteration} iterations.")
    print(f"Final parameters: Cost: {current_cost}, Weight: {current_weight}, Bias: {current_bias}")
    
    return current_weight, current_bias





