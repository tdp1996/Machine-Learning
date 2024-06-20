import random
from utilities.cost_functions import mean_squared_error

STOPPING_THRESHOLD = 1e-6
def gradient_descent(X_train: list, y_train: list, learning_rate: float, stopping_threshold: float = STOPPING_THRESHOLD) -> tuple:
    
    # Initializing weight,bias
    current_weight = random.uniform(-1,1)
    current_bias = random.uniform(-1,1)
    n = len(X_train)
    previous_cost = float('inf')
    iteration = 0

    while True:
        y_predict = [X_train[i]*current_weight + current_bias for i in range(n)]
        current_cost = mean_squared_error(y_train, y_predict)
        if abs(previous_cost - current_cost) <= stopping_threshold:
            break
        # # Print progress every 100 iterations
        # if iteration % 100 == 0:
        #     print(f"Iteration {iteration}: Cost: {current_cost}, Weight: {current_weight}, Bias: {current_bias}")

        previous_cost = current_cost

        # Calculate gradients
        weight_derivative = (-2/n) * sum((y_train[i] - y_predict[i]) * X_train[i] for i in range(n))
        bias_derivative = (-2/n) * sum(y_train[i] - y_predict[i] for i in range(n))
        
        # Update weights and bias
        current_weight -= learning_rate * weight_derivative
        current_bias -= learning_rate * bias_derivative
        
        iteration += 1
    
    print(f"Optimization finished after {iteration} iterations.")
    print(f"Final parameters: Cost: {current_cost}, Weight: {current_weight}, Bias: {current_bias}")
    
    return current_weight, current_bias

def gradient_descent_mutilple_variables(X_train: list[list], y_train: list, learning_rate: float, stopping_threshold: float = STOPPING_THRESHOLD) ->tuple:

    #Initializing weight,bias
    numb_features = len(X_train[0])
    current_weight = [random.uniform(-1,1)]*numb_features
    current_bias = random.uniform(-1,1)
    numb_samples = len(X_train)
    previous_cost = float('inf')
    iteration = 0

    while True:
        y_predict = []
        for X_i in X_train:
            y_predict_i = sum(X_i[j]*current_weight[j] for j in range(numb_features)) + current_bias
            y_predict.append(y_predict_i)
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


        current_weight = [current_weight[j] - (learning_rate * weight_derivative[j]) for j in range(numb_features)]
        current_bias -= learning_rate * bias_derivative
        iteration += 1

    print(f"Optimization finished after {iteration} iterations.")
    print(f"Final parameters: Cost: {current_cost}, Weight: {current_weight}, Bias: {current_bias}")
    
    return current_weight, current_bias





