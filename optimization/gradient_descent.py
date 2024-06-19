import random
from utilities.cost_functions import mean_squared_error

STOPPING_THRESHOLD = 1e-6
def gradient_descent(X_train: list, y_train: list, learning_rate: float, stopping_threshold=STOPPING_THRESHOLD) -> tuple:
    
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
        # Print progress every 100 iterations
        if iteration % 100 == 0:
            print(f"Iteration {iteration}: Cost: {current_cost}, Weight: {current_weight}, Bias: {current_bias}")

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
