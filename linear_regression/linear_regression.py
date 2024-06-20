from typing import Union
import matplotlib.pyplot as plt
import pandas as pd
from optimization.gradient_descent import gradient_descent
# from sklearn.model_selection import train_test_split


def LinearRegression(X_train: Union[list,list[list]], y_train: list, learning_rate: float) ->tuple[float]:
    
    weight, bias = gradient_descent(X_train, y_train, learning_rate)

    return weight, bias
    
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

#example 2
# data = pd.read_csv('data/linear_regression_data.csv')

# # Drop the missing values
# data = data.dropna()

# # Chia dữ liệu thành features (X) và target (y)
# X = data.drop('Target', axis=1)  # X là tất cả các cột ngoại trừ cột 'Target'
# y = data['Target']  # y là cột 'Target'

# # Chia thành tập train và test
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Đưa tập train thành list[features]
# X_train = X_train.values.tolist()
# y_train = y_train.values.tolist()
# X_test = X_test.values.tolist()
# y_test = y_test.values.tolist()

# weight, bias = gradient_descent(X_train=X_train, y_train=y_train, learning_rate= 0.00001)

# y_predict = [sum(X_i[j] * weight[j] for j in range(len(X_test[0]))) + bias for X_i in X_test]
