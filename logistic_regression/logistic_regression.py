import random
from typing import Union
from optimization.gradient_descent import gradient_descent
from utilities.activations import sigmoid
from utilities.cost_functions import binary_cross_entrophy

STOPPING_THRESHOLD = 1e-6

def LogisticRegression(X_train: Union[list,list[list]], y_train: list, learning_rate: float, stopping_threshold: float=STOPPING_THRESHOLD):
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

        weight_derivative, bias_derivative = gradient_descent(X_train, y_train, y_predict)
        
        # Update weights and bias
        weights = [weights[j] - (learning_rate * weight_derivative[j]) for j in range(numb_features)]
        bias -= learning_rate * bias_derivative
        iteration += 1
    print(f"Optimization finished after {iteration} iterations.")
    print(f"Final parameters: Cost: {current_cost}, Weight: {weights}, Bias: {bias}")
    
    return weights, bias

def predict(x: Union[list,Union[float,int]], weight: list, bias: float) -> int:
    if not isinstance(x,list):
        x = [x]
    z = sum(x[i] * weight[i] for i in range(len(x))) + bias
    predict = sigmoid(z)
    predicted_class = 1 if predict >= 0.5 else 0
    return [predicted_class]


if __name__ == "__main__":
    X_train = [3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]
    y_train = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    weights, bias = LogisticRegression(X_train=X_train, y_train=y_train, learning_rate= 0.01)

    print(predict(3.78,weights,bias))