import pandas as pd
import random
from typing import Union
from optimization.gradient_descent import gradient_descent
from utilities.activations import sigmoid
from utilities.cost_functions import binary_cross_entrophy
from sklearn.model_selection import train_test_split

STOPPING_THRESHOLD = 1e-6

def LogisticRegression_multiclass_model(X:  Union[list, list[list]], model_params: list[tuple[list, float]], classess: list[int]) -> list[int]:
    """
    Predict class labels for the given test data.

    Args:
        X ( Union[list, list[list]]): input data.
        model_params (list[tuple[list[float], float]]): List of tuples containing weights and bias for each class model.
        classes (list[int]): List of unique classes.

    Returns:
        list[int]: Predicted class labels for the input data.
    """
    if not all(isinstance(X_i, list) for X_i in X):
        X = [[X_i] for X_i in X]

    predictions = []
    for X_i in X:
        class_scores = []
        for weights, bias in model_params:
            linear_combination = sum(X_i[j] * weights[j] for j in range(len(X_i))) + bias
            class_scores.append(sigmoid(linear_combination))
        predicted_class = classess[class_scores.index(max(class_scores))]
        predictions.append(predicted_class)
    
    return predictions

def train_LogisticRegression_multiclass(X_train: Union[list[Union[float,int]],list[list[Union[float,int]]]], 
                                y_train: list[int], 
                                learning_rate: float,
                                iterations: int,  
                                stopping_threshold: float = STOPPING_THRESHOLD) -> list[tuple[list[float], float]]:
    """
    Train logistic regression models for multiclass classification using One-vs-Rest strategy.

    Args:
        - X_train (Union[list[Union[float,int]],list[list[Union[float,int]]]]): Training data.
        - y_train (list[int]): Training labels.
        - learning_rate (float): Learning rate for gradient descent.
        - iterations (int): Number of iterations for gradient descent.
        - stopping_threshold (float): Threshold for stopping criterion based on cost change.

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
    
    model_params = []

    classess = list(set(y_train))
    for c in classess:
        y_binary_i = [1 if y_i == c else 0 for y_i in y_train]
        iteration = 0
        while iteration < iterations:
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

            if iteration % 1000 == 0:
                print(f"Class {c} - Model parameters after {iteration} iterations: Cost: {current_cost:.4f}, Weights: {weights}, Bias: {bias:.4f}")

            weights, bias = gradient_descent(X_train, y_binary_i, y_predict, weights, bias, learning_rate)            
            iteration += 1

        model_params.append((weights, bias))

    return model_params


if __name__ == "__main__":

    data = pd.read_csv('logistic_regression/data_test/multiclass_classification_data.csv')

    # Drop the missing values
    data = data.dropna()

    # Split data into features (X) và target (y)
    X = data.drop('y', axis=1)  # X is all columns except 'Target' column
    y = data['y']  # y is 'Target' column

    # Divide into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Turn training set into list[features]
    X_train = X_train.values.tolist()
    y_train = y_train.values.tolist()
    X_test = X_test.values.tolist()
    y_test = y_test.values.tolist()
    classess = list(set(y_test))

    #train logistic regression model
    model_params =  train_LogisticRegression_multiclass(X_train=X_train, y_train=y_train,learning_rate= 0.01,iterations= 10000)
    # predict
    y_predict = LogisticRegression_multiclass_model(X=X_test, model_params=model_params, classess=classess)
    
    
       

    
