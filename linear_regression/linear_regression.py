import random
from typing import Union
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
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

        weights, bias = gradient_descent(X_train, y_train, y_predict, weights, bias, learning_rate)   
        iteration += 1
    print(f"Optimization finished after {iteration} iterations.")
    print(f"Final parameters: Cost: {current_cost}, Weight: {weights}, Bias: {bias}")
    
    return weights, bias


def predict(X_test: Union[int,float,list,list[list]], weights: list, bias: float) ->list:
    """
    Predict the output based input features using linear regression model parameters

    Args:
        X_test (Union[int,float,list,list[list]]): Input features.
        weights (list): Weights of the linear regression model.
        bias (float): Bias of the linear regression model.

    Returns:
        list: Predicted output values.
    """
    # Ensure X_test is a list of lists
    if isinstance(X_test,(float,int)):
        X_test = [[X_test]]
    elif isinstance(X_test[0],(float,int)):
        X_test = [[X_i] for X_i in X_test]

    predict = [sum(X_i[j]*weights[j] for j in range(len(X_i))) + bias for X_i in X_test]
           
    return predict

    
if __name__ == "__main__":

    #example 1
    data = pd.read_csv('linear_regression/data_test/simple_linear_regression_data.csv')

    # Drop the missing values
    data = data.dropna()

    # training dataset and labels
    X_train= list(data.X[0:80])
    y_train = list(data.y[0:80])

    # valid dataset and labels
    X_test = list(data.X[80:100])
    y_test = list(data.y[80:100])

    weight, bias = LinearRegression(X_train=X_train, y_train=y_train, learning_rate= 0.001)
    y_predict = predict(X_test=X_test, weights=weight,bias=bias)


    # # visualize results
    # plt.figure(figsize=(14, 10))
    # plt.scatter(X_test, y_test, color='blue', label='Actual data')
    # plt.plot(X_test, y_predict, color='red', linewidth=2, label='Regression line')
    # plt.xlabel('X')
    # plt.ylabel('y')
    # plt.title('Linear Regression')
    # plt.legend()
    # plt.grid(True)
    # plt.show()


    # # example 2
    # data = pd.read_csv('linear_regression/data_test/multiple_linear_regression_data.csv')

    # # Drop the missing values
    # data = data.dropna()

    # # Split data into features (X) và target (y)
    # X = data.drop('y', axis=1)  # X is all columns except 'Target' column
    # y = data['y']  # y is 'Target' column

    # # Divide into training and test sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # # Turn training set into list[features]
    # X_train = X_train.values.tolist()
    # y_train = y_train.values.tolist()
    # X_test = X_test.values.tolist()
    # y_test = y_test.values.tolist()

    # weight, bias = LinearRegression(X_train=X_train, y_train=y_train, learning_rate= 0.001)

    # print(predict(X_test=X_test, weights=weight,bias=bias))
