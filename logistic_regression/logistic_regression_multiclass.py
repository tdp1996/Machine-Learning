import random
from typing import Union
from optimization.gradient_descent import gradient_descent
from utilities.activations import sigmoid
from utilities.cost_functions import binary_cross_entrophy

STOPPING_THRESHOLD = 1e-6

def get_unique_classes(y_train: list[int]) -> list[int]:
    """
    Extract unique classes from the training labels.

    Args:
        y_train (list[int]): list of training labels.

    Returns:
        List[int]: list of unique classes.
    """
    classess = []
    for y_i in y_train:
        if y_i not in classess:
            classess.append(y_i)
    return classess

def multiclass_logistic_regression(X_train: Union[list[float], list[list[float]]], y_train: list[int], learning_rate: float, stopping_threshold: float = STOPPING_THRESHOLD) -> list[tuple[list[float], float]]:
    """
    Train logistic regression models for multiclass classification using One-vs-Rest strategy.

    Args:
        X_train (Union[list[float], list[list[float]]]: Training data.
        y_train (list[int]): Training labels.
        learning_rate (float): Learning rate for gradient descent.
        stopping_threshold (float): Threshold for stopping criterion based on cost change.

    Returns:
        list[tuple[list[float], float]]: List of tuples containing weights and bias for each class model.
    """
    # Determine if this is multiple regression and initialize weights and bias
    if all(isinstance(X_i,list) for X_i in X_train):
        numb_features = len(X_train[0])
        weights = [random.uniform(-1,1)]*numb_features
    else:
        numb_features = 1
        X_train = [[X_i] for X_i in X_train]
        weights = [random.uniform(-1,1)]
    bias = random.uniform(-1,1)
    previous_cost = float('inf')
    iteration = 0
    model_params = []

    classess = get_unique_classes(y_train)
    for c in classess:
        y_binary_i = [1 if y_i == c else 0 for y_i in y_train]
        while True:
            # Predicting y values
            y_predict = []
            for X_i in X_train:
                y_predict_i = sum(X_i[j]*weights[j] for j in range(numb_features)) + bias
                y_predict.append(sigmoid(y_predict_i))
        
            # Calculate current cost using mean squared error
            current_cost = binary_cross_entrophy(y_binary_i, y_predict)
            if abs(previous_cost - current_cost) <= stopping_threshold:
                break
            previous_cost = current_cost

            weight_derivative, bias_derivative = gradient_descent(X_train, y_binary_i, y_predict)
            
            # Update weights and bias
            weights = [weights[j] - (learning_rate * weight_derivative[j]) for j in range(numb_features)]
            bias -= learning_rate * bias_derivative
            iteration += 1
        print(f"Optimization finished after {iteration} iterations.")
        print(f"Final parameters: Cost: {current_cost}, Weight: {weights}, Bias: {bias}")

        model_params.append((weights, bias))

    return model_params


def predict(X_test:  Union[list, list[list]], model_params: list[tuple[list, float]], classes: list[int]) -> list[int]:
    """
    Predict class labels for the given test data.

    Args:
        X_test ( Union[list, list[list]]): Test data.
        model_params (list[tuple[list[float], float]]): List of tuples containing weights and bias for each class model.
        classes (list[int]): List of unique classes.

    Returns:
        List[int]: Predicted class labels for the test data.
    """
    if not all(isinstance(X_i, list) for X_i in X_test):
        X_test = [[X_i] for X_i in X_test]
    predictions = []

    for X_i in X_test:
        class_scores = []
        for weights, bias in model_params:
            linear_combination = sum(X_i[j] * weights[j] for j in range(len(X_i))) + bias
            class_scores.append(sigmoid(linear_combination))
        predicted_class = classes[class_scores.index(max(class_scores))]
        predictions.append(predicted_class)
    
    return predictions


        

    
