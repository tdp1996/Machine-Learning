import random
import pandas as pd
from typing import Union
from optimization.gradient_descent import gradient_descent
from utilities.activations import sigmoid
from utilities.cost_functions import binary_cross_entrophy
from sklearn.model_selection import train_test_split

STOPPING_THRESHOLD = 1e-6

def LogisticRegression(X_train: Union[list,list[list]], y_train: list, learning_rate: float, stopping_threshold: float=STOPPING_THRESHOLD):
    """
    Perform logistic regression using gradient descent optimization to find optimal weights and bias.

    Args:
    - X_train (Union[list, list[list]]): Training data features. If single-variable, it should be a list of values.
      If multi-variable, it should be a list of lists where each sublist represents a feature vector.
    - y_train (list): Training data labels.
    - learning_rate (float): Learning rate for gradient descent, determines step size for each iteration.
    - stopping_threshold (float, optional): Threshold for stopping criteria based on change in cost function. Default is 1e-6.

    Returns:
    - tuple: Final optimized weights and bias.

    This function optimizes the logistic regression model using gradient descent until the change in cost function
    between iterations is less than or equal to the stopping threshold. It prints the number of iterations taken
    and the final optimized parameters.
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

    while True:
        # Predicting y values
        y_predict = []
        for X_i in X_train:
            y_predict_i = sum(X_i[j]*weights[j] for j in range(numb_features)) + bias
            y_predict.append(sigmoid(y_predict_i))

        # Calculate current cost using mean squared error
        current_cost = binary_cross_entrophy(y_train, y_predict)
        if abs(previous_cost - current_cost) <= stopping_threshold:
            break
        previous_cost = current_cost
        weights, bias = gradient_descent(X_train, y_train, y_predict, weights, bias, learning_rate)
        
        iteration += 1
    print(f"Optimization finished after {iteration} iterations.")
    print(f"Final parameters: Cost: {current_cost}, Weight: {weights}, Bias: {bias}")
    
    return weights, bias

def predict(X_test: Union[list[list],float,int], weights: list, bias: float) -> int:
    """
    Predicts the class label based on input features using logistic regression model parameters.

    Args:
    - X_test (Union[list[list],float,int]): 
      Input features to predict. If single-variable, it can be a single value. 
      If multi-variable, it can be a list of values corresponding to each feature, 
      or a list of lists for multiple samples.
    - weights (list): Optimized weights from logistic regression model.
    - bias (float): Optimized bias term from logistic regression model.

    Returns:
    - list[int]: List of predicted class labels (0 or 1).

    This function computes the predicted class labels using the logistic regression model parameters (weights and bias).
    It applies the sigmoid function to compute the probability and then classifies based on a threshold of 0.5.
    """
    predictions = []
    if isinstance(X_test,(float,int)):
        X_test = [[X_test]]
    elif isinstance(X_test[0],(float,int)) and len(weights)==1:
        X_test = [[X_i] for X_i in X_test]
    elif isinstance(X_test[0],(float,int)):
        X_test = [X_test]

    for X_i in X_test:
        z = sum(X_i[j] * weights[j] for j in range(len(X_i))) + bias
        predict_i = sigmoid(z)
        predicted_class_i = 1 if predict_i >= 0.5 else 0
        predictions.append(predicted_class_i)
    return predictions


if __name__ == "__main__":

    data = pd.read_csv('logistic_regression/binary_classification_data.csv')

    # Drop the missing values
    data = data.dropna()

    # Split data into features (X) và target (y)
    X = data.drop('Label', axis=1)  # X is all columns except 'Target' column
    y = data['Label']  # y is 'Target' column

    # Divide into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Turn training set into list[features]
    X_train = X_train.values.tolist()
    y_train = y_train.values.tolist()
    X_test = X_test.values.tolist()
    y_test = y_test.values.tolist()
    weights, bias = LogisticRegression(X_train=X_train, y_train=y_train,learning_rate= 0.001)
    print(predict(X_test, weights, bias))
    