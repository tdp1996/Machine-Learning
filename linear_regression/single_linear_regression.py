import matplotlib.pyplot as plt
import pandas as pd
from optimization.gradient_descent import gradient_descent

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


weight, bias = gradient_descent(X_train=X_train, y_train=y_train, learning_rate= 0.00001)
y_predict = [(X_test[i]*weight + bias) for i in range(len(X_test))]


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


