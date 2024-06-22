import random
from typing import Union
import matplotlib.pyplot as plt
import pandas as pd
# from sklearn.model_selection import train_test_split
from optimization.gradient_descent import gradient_descent
from utilities.cost_functions import mean_squared_error

STOPPING_THRESHOLD = 1e-6

def LinearRegression(X_train: Union[list,list[list]], y_train: list, learning_rate: float, stopping_threshold: float=STOPPING_THRESHOLD) ->tuple[list[float], float]:
    """
    Trains a linear regression model using gradient descent optimization.

    Args:
        X_train (Union[list,list[list]]): Training data. If a list, it represents single variable training data. 
                                                         If a list of lists, it represents multiple variable training data.
        y_train (List): Target values corresponding to the training data.
        learning_rate (float): The learning rate for gradient descent optimization.

    Returns:
        Tuple[List[float], float]: A tuple containing the final weights (list of floats) and bias (float).
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
            y_predict.append(y_predict_i)

        # Calculate current cost using mean squared error
        current_cost = mean_squared_error(y_train, y_predict)
        if abs(previous_cost - current_cost) <= stopping_threshold:
            break
        previous_cost = current_cost

        weight_derivative, bias_derivative = gradient_descent(X_train, y_train, y_predict)
        
        # Update weights and bias
        weights = [weights[j] - (learning_rate * weight_derivative[j]) for j in range(numb_features)]
        bias -= learning_rate * bias_derivative
        iteration += 1
    print(f"Optimization finished after {iteration} iterations.")
    print(f"Final parameters: Cost: {current_cost}, Weight: {weights}, Bias: {bias}")
    
    return weights, bias
    
    
if __name__ == "__main__":

    #example 1
    data = pd.read_csv('data/data_for_lr.csv')

    # Drop the missing values
    data = data.dropna()

    # training dataset and labels
    X_train= list(data.x[0:550])
    y_train = list(data.y[0:550])

    # valid dataset and labels
    X_test = list(data.x[550:700])
    y_test = list(data.y[550:700])

    weight, bias = LinearRegression(X_train=X_train, y_train=y_train, learning_rate= 0.00001)
    y_predict = [(X_test[i]*weight[0] + bias) for i in range(len(X_test))]


    # visualize results
    plt.figure(figsize=(14, 10))
    plt.scatter(X_test, y_test, color='blue', label='Actual data')
    plt.plot(X_test, y_predict, color='red', linewidth=2, label='Regression line')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Regression')
    plt.legend()
    plt.grid(True)
    plt.show()

    # # example 2
    # data = pd.read_csv('data/linear_regression_data.csv')

    # # Drop the missing values
    # data = data.dropna()

    # # Split data into features (X) và target (y)
    # X = data.drop('Target', axis=1)  # X is all columns except 'Target' column
    # y = data['Target']  # y is 'Target' column

    # # Divide into training and test sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # # Turn training set into list[features]
    # X_train = X_train.values.tolist()
    # y_train = y_train.values.tolist()
    # X_test = X_test.values.tolist()
    # y_test = y_test.values.tolist()

    # weight, bias = LinearRegression(X_train=X_train, y_train=y_train, learning_rate= 0.00001)

    # y_predict = [sum(X_i[j] * weight[j] for j in range(len(X_test[0]))) + bias for X_i in X_test]